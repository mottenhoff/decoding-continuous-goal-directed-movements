'''
https://github.com/ShanechiLab/PyPSID

[ ] Check if PSID version = 1.10+. Then extracting the mean and add it after learning is not necessary anymore
[ ] Check n1 <= nz * i  AND   nx <= ny *i. Important to report i in paper
[ ] Extract only relevant states by setting nx = n1
[ ] Use PSID.evaluation.evalPrediction
[ ] Plot EigenValues of A Matrix of learned models and 'True' models to check the accurate learning (probably not possible with our data)

PSID tutorial: https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/example/PSID_tutorial.ipynb
'''

import logging
import cProfile
import pstats
import io
from pathlib import Path
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np

import learner
from libs import checks
from libs import data_cleaning
from libs import prepare
from libs import timeshift
from libs import utils
from libs.load import go as load_leap
from libs.plotting import plot_trajectory
from libs.data_cleaning import cleanup, remove_non_eeg
from figures import all_figures
from figures import checks as fig_checks

from figures.figure_cc_per_band_per_kinematic import plot_band_correlations

c = utils.load_yaml('./config.yml')
logger = logging.getLogger(__name__)

def setup():
    main_path = Path(c.learn.save_path)
    today = dt.today().strftime('%Y%m%d_%H%M')
    path = main_path/today
    path.mkdir(parents=True, exist_ok=True)
    
    log_filename = f'output.log'
    logging.basicConfig(format="[%(filename)10s:%(lineno)3s - %(funcName)20s()] %(message)s",
                        level=logging.INFO if not c.debug.log else logging.DEBUG,
                        handlers=[
                            logging.FileHandler(path/f'{log_filename}'),
                            logging.StreamHandler()])

    return path

def setup_debug(eeg, xyz):
    # Make XYZ a (random) linear combination of your data to be sure that your 
    # data is decodable.
    #   Useful for checking things
    # Also add some time lag (3 samples in this case) to simulate some dynamics
    timeshift = 3 # sample
    dims = 2
    n = 10000
    n = eeg['data'].shape[0]
    some_random_matrix = np.array([[ 0.46257236, -0.35283946,  0.06892439],
                                   [-0.44375285, -0.40580064,  1.15588792]])

    logger.debug(f'Creating timeshifted linear transformation of {2} features and {timeshift} samples timeshift')
    logger.debug(f'Used linear transformation: {some_random_matrix}')
    
    xyz[timeshift: ,:] = eeg['data'][:-timeshift, :dims] @ some_random_matrix
    xyz[:timeshift, :] = 0

    logger.debug(f'Shortening data to the first {n} samples')
    eeg['data'], xyz = eeg['data'][:n, :], xyz[:n, :]

    if c.debug.reduce_channels:
        eeg['data'], eeg['channel_names'] = eeg['data'][:, :c.debug.reduce_channels], eeg['channel_names'][:c.debug.reduce_channels]
        # eeg['data'], eeg['channel_names'] = eeg['data'][:, :5], eeg['channel_names'][:5]

    return eeg, xyz

def go(save_path):
    fig_checks.reset()
 
    # data_path = Path('./data/kh036/')
    # data_path = Path('./data/kh040/')
    # data_path = Path('./data/kh041/')
    data_path = Path('./data/kh042/')

    filenames = [p for p in data_path.glob('*.xdf')]
    
    if not c.combine:
        filenames = [filenames[0]]

    datasets = []
    chs_to_remove = np.array([], dtype=np.int16)
    for filename in filenames:
        logger.info(f'Loaded {filename}')

        eeg, xyz, trials = load_leap(filename) #data_path/filename)

        eeg['data'], eeg['channel_names'] = remove_non_eeg(eeg['data'], eeg['channel_names'])

        # fig_checks.plot_xyz(xyz)
        # fig_checks.plot_eeg(eeg['data'], eeg['channel_names'], 'raw', loc_map=eeg['channel_mapping'])

        # TODO: Move to debug file
        if c.debug.go and c.debug.dummy_data:
            print('DEBUG is active!')
            eeg, xyz = setup_debug(eeg, xyz)
        
        # Note that when combining the second loop additional channel might be removed
        chs_to_remove = np.append(chs_to_remove, cleanup(eeg['data'], eeg['channel_names'], 
                                                          eeg['ts'], eeg['fs'],
                                                          pid=filename.parts[-2],
                                                          sid=filename.stem[-1]))

        # fig_checks.plot_eeg(np.delete(eeg['data'], chs_to_remove, axis=1),
        #                     np.delete(eeg['changit nel_names'], chs_to_remove),
        #                     f'after quality check | Removed: {[eeg["channel_names"][ch] for ch in chs_to_remove]}',
        #                     loc_map=eeg['channel_mapping'])

        if c.debug.go and c.debug.short:
            eeg['data'] = eeg['data'][:20000, :]
            eeg['ts'] = eeg['ts'][:20000]
            xyz = xyz[:20000, :]

        if c.checks.trials_vs_cont:
            checks.data_size_trial_vs_continuous(trials[:, 0], xyz)

        if c.target_vector:
            if not c.pos:
                logger.error(f'Target vector can only be calculated if position is used as Z.')
            
            # TODO: CHANGE THIS, THIS IS ERROR PRONE
            #       Supply Trial information (including a target vector?) 
            eeg['data'] = np.hstack((eeg['data'], trials[:, 1:] - xyz))

        eeg, xyz = timeshift.shift(eeg, xyz, t=c.timeshift)

        # with open(f'{data_path.name}_data.npy', 'wb') as f:
        #     np.save(f, np.hstack((eeg['data'], xyz)))
        
        datasets += prepare.go(eeg, xyz)

        print('')

    chs_to_remove = np.unique(chs_to_remove)
    for ds in datasets:
        ds.eeg = np.delete(ds.eeg, chs_to_remove, axis=1)
        ds.channels = np.delete(ds.channels, chs_to_remove)
    
    logger.info(f'Removed {chs_to_remove.size} channels')

    learner.fit(datasets, save_path)

    # Figures over sessions

    if c.figures.make_all:
        all_figures.make_overview(save_path)

def main():
    save_path = setup()

    with cProfile.Profile() as pr:
        go(save_path)
    
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s)
    stats.dump_stats(f'{save_path}/profile.prof')

    with open(f'{save_path}/profile.txt', 'w') as f:
        ps = pstats.Stats(f'{save_path}/profile.prof', stream=f)
        ps.sort_stats(pstats.SortKey.TIME)
        ps.print_stats()


if __name__=='__main__':
    # path = Path('/home/coder/project/results/20221207_1730')  #Kh40
    # path = Path('/home/coder/project/results/20221207_1709')  #Kh41
    # path = Path('/home/coder/project/results/20221207_0215')  #Kh42
    # all_figures.make_overview(path)
# 
    # fig_checks.check_used_data('')

    # main()


    

# Strong channel removal
# Select only beta
# Task correlations
# inlude hands
# Input dimensions to 30

