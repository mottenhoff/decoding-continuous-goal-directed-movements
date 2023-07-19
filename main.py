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
import re
import sys
from multiprocessing import Process, Pool

from pathlib import Path
from datetime import datetime as dt
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml

import learner
from libs import checks
from libs import data_cleaning
from libs import prepare
from libs import timeshift
from libs import utils
from libs import explore
from libs.rereference import common_electrode_reference, common_average_reference
from libs.load import go as load_leap
from libs.plotting import plot_trajectory
from libs.data_cleaning import cleanup, remove_non_eeg
from figures import all_figures
from figures import checks as fig_checks

from figures.figure_cc_per_band_per_kinematic import plot_band_correlations

c = utils.load_yaml('./config.yml')
logger = logging.getLogger(__name__)

def setup(main_path, id_):

    path = main_path/id_
    path.mkdir(parents=True, exist_ok=True)

    i = 0
    while True:
        new_path = path/f'{i}'
        
        try:
            new_path.mkdir(parents=False, exist_ok=False)
            path = new_path
            break
        except Exception:
            i += 1
            continue

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    
    log_filename = f'output.log'
    logging.basicConfig(format="[%(filename)10s:%(lineno)3s - %(funcName)20s()] %(message)s",
                        level=logging.INFO if not c.debug.log else logging.DEBUG,
                        handlers=[
                            logging.FileHandler(path/f'{log_filename}'),
                            # logging.StreamHandler()
                            console_handler])

    return path

def setup_debug(eeg, xyz):
    # Make XYZ a (random) linear combination of your data to be sure that your 
    # data is decodable.
    #   Useful for checking things
    # Also add some time lag (3 samples in this case) to simulate some dynamics
    logger.info('Setting up dummy data for debugging')

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

def include(id_):
    info = utils.load_yaml(f'./data/kh{id_:03d}/info.yaml', return_dict=True)
    
    if info['problems_left']:
        return False
    
    return True

def go(save_path, filenames, ppt_id):
    fig_checks.reset()
        
    # all_trials = []
    datasets = []
    chs_to_remove = np.array([], dtype=np.int16)
    n_targets = 0

    # print(f'{filenames} ran succesfully!')
    # return
    for filename in filenames:
        logger.info(f'Loaded {filename}')

        eeg, xyz, trials, events = load_leap(filename) #data_path/filename)
        eeg['id'] = ppt_id

        n_targets += events.target_reached.shape[0]

        eeg['data'], eeg['channel_names'] = remove_non_eeg(eeg['data'], eeg['channel_names'])

        # fig_checks.plot_xyz(xyz)
        # fig_checks.plot_eeg(eeg['data'], eeg['channel_names'], 'raw', loc_map=eeg['channel_mapping'])

        # TODO: Move to debug file
        if c.debug.go and c.debug.dummy_data:
            eeg, xyz = setup_debug(eeg, xyz)

        flagged_channels = cleanup(eeg['data'], eeg['channel_names'], eeg['ts'], eeg['fs'],
                                   pid=filename.parts[-2], sid=filename.stem[-1])

        # assert flagged_channels.size == 0, 'Channels are flagged!'
            
        chs_to_remove = np.append(chs_to_remove, flagged_channels)

        # fig_checks.plot_eeg(np.delete(eeg['data'], chs_to_remove, axis=1),
        #                     np.delete(eeg['changit nel_names'], chs_to_remove),
        #                     f'after quality check | Removed: {[eeg["channel_names"][ch] for ch in chs_to_remove]}',
        #                     loc_map=eeg['channel_mapping'])

        # Common average reference
        # common_average = eeg['data'].mean(axis=1, keepdims=True)
        eeg['data'] = common_average_reference(eeg['data'])
        # eeg['data'] = common_electrode_reference(eeg['data'], eeg['channel_names'])

        if c.debug.go and c.debug.short:
            n_samples = 20000
            logger.info(f'Shortening data for debugging to {n_samples} samples')
            eeg['data'] = eeg['data'][:20000, :]
            eeg['ts'] = eeg['ts'][:20000]
            xyz = xyz[:20000, :]

        if c.checks.trials_vs_cont:
            checks.data_size_trial_vs_continuous(trials[:, 0], xyz)

        if c.target_vector:
            logger.info('Calculating target vector')
            if not c.pos:
                logger.error(f'Target vector can only be calculated if position is used as Z.')
            
            # TODO: CHANGE THIS, THIS IS ERROR PRONE
            #       Supply Trial information (including a target vector?) 
            eeg['data'] = np.hstack((eeg['data'], trials[:, 1:] - xyz))

        eeg, xyz = timeshift.shift(eeg, xyz, t=c.timeshift)  # TODO: comment out if not using

        # with open(f'{data_path.name}_data.npy', 'wb') as f:
        #     np.save(f, np.hstack((eeg['data'], xyz)))

        # Save info
        
        datasets += prepare.go(eeg, xyz)

        print('')

    # TODO: To function 
    chs_to_remove = np.unique(chs_to_remove)
    for ds in datasets:
        ds.eeg = np.delete(ds.eeg, chs_to_remove, axis=1)
        ds.channels = np.delete(ds.channels, chs_to_remove)
    
    logger.info(f'Removed {chs_to_remove.size} channels')
    
    with open(save_path/'info.yml', 'w+') as f:
        info = {'ppt_id': f'kh{ppt_id:03d}',
                'datasize': sum([d.xyz.shape[0] for d in datasets]),
                'n_targets': n_targets}
        yaml.dump(info, f)

    explore.main(datasets, save_path)

    learner.fit(datasets, save_path)

def main(filelist: list, main_path: Path):
    ppt_id = filelist[0].parts[-2]
    id_ = int(re.findall(r'\d+', ppt_id)[0])

    save_path = setup(main_path, ppt_id)

    with cProfile.Profile() as pr:
        go(save_path, filelist, id_)
    
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s)
    stats.dump_stats(f'{save_path}/profile.prof')

    with open(f'{save_path}/profile.txt', 'w') as f:
        ps = pstats.Stats(f'{save_path}/profile.prof', stream=f)
        ps.sort_stats(pstats.SortKey.TIME)
        ps.print_stats()


if __name__=='__main__':

    main_path = Path(c.learn.save_path)
    today = dt.today().strftime('%Y%m%d_%H%M')
    main_path = main_path/today
    main_path.mkdir(parents=True, exist_ok=True)

    filenames = [p for p in Path('./data/').rglob('*.xdf')]
    
    # Filter ppt_ids:
    if ids := []:
        filenames = [file for file in filenames if int(file.parts[-2][-2:]) in ids]

    if c.combine:
        # TODO: Switch for one or both hands.
        files_per_ppt = defaultdict(list)

        for file in filenames:
            files_per_ppt[file.parts[-2]].append(file)

        jobs = list(files_per_ppt.values())
    else:
        jobs = [[file] for file in filenames]

    if c.parallel:
        
        pool = Pool(processes=5)
        for job in jobs:
            pool.apply_async(main, args=(job, main_path))
        pool.close()
        pool.join()

    else:

        for job in jobs:
            main(job, main_path)  # job = [Filename, ...]

    print('done')
    

# Strong channel removal
# Select only beta
# Task correlations
# inlude hands
# Input dimensions to 30