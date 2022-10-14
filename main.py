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
from pathlib import Path
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np

import learner
from libs import checks
from libs import prepare
from libs import data_cleaning
from libs import utils
from libs.load import go as load_leap
from libs.plotting import plot_trajectory
from figures import all_figures

c = utils.load_yaml('./config.yml')

def setup():
    main_path = Path(c.learn.save_path)
    today = dt.today().strftime('%Y%m%d_%H%M')
    path = main_path/today
    path.mkdir(parents=True, exist_ok=True)
    
    log_filename = f'output.log'
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.INFO if not c.debug_log else logging.DEBUG,
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

    logging.debug(f'Creating timeshifted linear transformation of {2} features and {timeshift} samples timeshift')
    logging.debug(f'Used linear transformation: {some_random_matrix}')
    
    xyz[timeshift: ,:] = eeg['data'][:-timeshift, :dims] @ some_random_matrix
    xyz[:timeshift, :] = 0

    logging.debug(f'Shortening data to the first {n} samples')
    eeg['data'], xyz = eeg['data'][:n, :], xyz[:n, :]

    return eeg, xyz

def go():
    save_path = setup()
    data_path = Path('./data/kh036/')
    data_path = Path('./data/kh040/')

    filenames = [p for p in data_path.glob('*.xdf')]
    
    if not c.combine:
    # if True:
        filenames = [filenames[0]]
    # n_sets = 3 if c.combine else 1 # TODO: get dynamical

    datasets = []
    for filename in filenames:
        logging.info(f'Loaded {filename}')

        eeg, xyz, trials = load_leap(filename) #data_path/filename)

        if c.debug:
            eeg, xyz = setup_debug(eeg, xyz)

        if c.checks.trials_vs_cont:
            checks.data_size_trial_vs_continuous(trials, xyz)

        datasets += prepare.go(eeg, xyz)

    learner.fit(datasets, save_path)

    # if c.figures.make_all:
    #     all_figures.make(save_path)

if __name__=='__main__':
    go()


