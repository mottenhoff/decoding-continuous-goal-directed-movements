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

import numpy as np
import yaml

import learner
from libs import checks
from libs import data_cleaning
from libs import prepare
from libs import utils
from libs import explore
from libs.timeshift import timeshift
from libs.target_vector import target_vector
from libs.rereference import common_electrode_reference
from libs.load import load_dataset
from libs.plotting import plot_trajectory
from libs.data_cleaning import flag_irrelevant_channels

from figures.plot_dataset import plot_dataset, plot_subsets
from figures import all_figures
from figures import checks as fig_checks

from figures.figure_cc_per_band_per_kinematic import plot_band_correlations

DATASETS = 0
FLAGGED = 1


c = utils.load_yaml('./config.yml')
logger = logging.getLogger(__name__)

def save_dataset_info(targets_reached, n_samples, n_gaps, ppt_id, save_path):

    with open(save_path/'info.yml', 'w+') as f:
        info = {'ppt_id': ppt_id,
                'datasize': n_samples,
                'n_targets': targets_reached,
                'n_gaps': n_gaps}
        yaml.dump(info, f)

def run(save_path, filenames, ppt_id):
    fig_checks.reset()
        
    datasets = []
    n_targets, n_samples = 0, 0
    for filename in filenames:
        logger.info(f'Loaded {filename}')

        # Load and a bit of cleanup
        ds = load_dataset(filename, ppt_id) #data_path/filename)

        ds.eeg, _ = flag_irrelevant_channels(ds.eeg)

        ds.eeg.timeseries = common_electrode_reference(ds.eeg.timeseries, ds.eeg.channels)

        plot_dataset(ds)    

        # Some optional extra features
        if c.target_vector:
            vector = target_vector(ds.eeg.timeseries, ds.trials, ds.xyz)

        if c.timeshift:
            ds.eeg.timeseries, ds.xyz = timeshift(ds.eeg.timeseries, ds.xyz, t=c.timeshift)

        n_targets  += ds.events.target_reached.shape[0]
        n_samples += ds.xyz[~np.isnan(ds.xyz[:, 0]), 0].size
        
        datasets  += prepare.go(ds, save_path)

    n_gaps = len(datasets) - len(filenames)
    # plot_subsets(datasets, save_path)

    explore.main(datasets, save_path)

    save_dataset_info(n_targets, n_samples, n_gaps, ds.ppt_id, save_path)

    learner.fit(datasets, save_path)
    

# i=0; plt.figure(); plt.plot(datasets[0].xyz[:, i]); plt.plot(datasets[0].xyz[:, i+3]); plt.plot(datasets[0].xyz[:, i+6]); plt.savefig('tmp.png')
# plt.figure(); plt.plot(datasets[0].xyz[:, 9]); plt.plot(datasets[0].xyz[:, 10]); plt.plot(datasets[0].xyz[:, 11]); plt.savefig('tmp.png')

# Strong channel removal
# Select only beta
# Task correlations
# inlude hands
# Input dimensions to 30