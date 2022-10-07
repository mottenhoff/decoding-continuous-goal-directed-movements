import logging

import numpy as np
import matplotlib.pyplot as plt

import PSID
from PSID.evaluation import evalPrediction as eval_prediction

from libs.feature_selection.kbest import select_k_best
from libs import utils

c = utils.load_yaml('./config.yml')

def data_size_trial_vs_continuous(trials, xyz):


    # Fill middle nans
    idc = np.where(~np.isnan(trials))[0]
    for idx_f, idx_l in zip(idc[:-1], idc[1:]):
        trials[idx_f:idx_l] = trials[idx_f]

    # Remove leading and trailing nans
    mask = ~np.isnan(trials).ravel()
    trials = trials[mask]
    xyz = xyz[mask, :]

    # N samples in xyz
    idc = np.where(~np.isnan(xyz[:, 0]))[0]
    missing = np.where(np.diff(idc)>100)[0]
    trials_with_gaps = np.unique(trials[idc[missing]])
    u, c = np.unique(trials, return_counts=True)
    
    data_with_removed_trials = np.where(~np.isin(trials, trials_with_gaps))[0].shape[0]
    data_lost = xyz.shape[0] - data_with_removed_trials
            
    logging.info(f'''Checks | data_size_trial_vs_continuous\n
    Data_size_trial_vs_continuous:
        Total data_size:            {trials.shape[0]} | [{trials.shape[0]/1024:.2f}s]
        Samples per trial:          {dict(zip(u, c))}
        Trials with missing values: {trials_with_gaps}
        Data lost:                  {xyz.shape[0]} - {data_with_removed_trials} = {data_lost}
        Data lost [seconds, %]:     {data_lost/1024:.2f} s | {data_lost/xyz.shape[0]*100:.1f}%
    ''')
    logging.info(f'''Size of gaps: {np.diff(idc)[missing]} samples | {np.diff(idc)[missing]/1024} s''')

def concatenate_vs_separate_datasets(datasets):

    # select_k_best on concatenated y_train
    # features = [211, 154, 293, 273, 136,  98, 237, 195,  45, 124,  41, 108, 126,
    #                 149,  63, 125, 329,  71, 137,   9, 210,  85, 242,  48,  12,  97,
    #                 46, 128,  30,  10, 248,   7, 274, 184, 303, 185, 282,  66, 278,
    #                 150,  35, 292, 224, 239, 187,  28, 116, 222,  29, 244]
    # features = np.random.randint(0, datasets[0].eeg.shape[1], 50)

    features = np.arange(datasets[0].eeg.shape[1])

    # Check if horizon i adheres to constraints (See examples/tutorial)
    nx, n1, i = 10, 10, 10

    y_test = datasets[-1].eeg  # Seperate set without gaps
    z_test = datasets[-1].xyz

    y_test = y_test[:, features]
    
    logging.info('Score comparison | Concatenation vs separate dataset as input of PSID')
    logging.info(f'\tSettings: nx={nx} | n1={n1}')

    for i in [5, 10, 25]:
        total_samples_affected = len(datasets[:-2])*2*(i-1)
        logging.info(f'\tTotal samples affected with i={i}: {total_samples_affected}/{sum([s.eeg.shape[0] for s in datasets])} | {total_samples_affected / sum([s.eeg.shape[0] for s in datasets])*100:.1f}% ')
        for j, opt in enumerate(['con', 'sep']):
            
            y_train = [s.eeg[:, features] for s in datasets[:-1]]
            z_train = [s.xyz for s in datasets[:-1]]

            if opt == 'con':
                y_train = [np.vstack(y_train)]
                z_train = [np.vstack(z_train)]

            id_sys = PSID.PSID(y_train, z_train, nx, n1, i)
            
            z_hat, y_hat, _ = id_sys.predict(y_test)

            z_score = eval_prediction(z_test, z_hat, 'R2')
            y_score = eval_prediction(y_test, y_hat, 'R2')
            
            if j == 0:
                z_score_first = z_score.mean()
                y_score_first = y_score.mean()

            logging.info(f'\tR2 [i={i}] | {"Concatenated" if opt=="con" else "Separate":>24} | y = {y_score.mean():.3f} | z = {z_score.mean():.3f}')
        logging.info(f'\tR2 [i={i}] | {"Concatenated - Separate":>24} |dy = {y_score_first - y_score.mean():.3f} |dz = {z_score_first - z_score.mean():.3f}')
        logging.info('')