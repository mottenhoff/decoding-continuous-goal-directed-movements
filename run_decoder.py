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
from libs import dataset_info
from libs.timeshift import timeshift
from libs.rereference import common_electrode_reference, laplacian_reference
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

def save_dataset_info(targets_reached, n_samples, n_gaps, time_between_targets, total_time,
                      ppt_id, recorded_channels, save_path):
    
    with open(save_path/'info.yml', 'w+') as f:
        info = {'ppt_id': ppt_id,
                'datasize': n_samples,
                'n_targets': targets_reached,
                'n_gaps': n_gaps,
                'total_time': total_time}
        yaml.dump(info, f)

    # with open(save_path/'time_between_targets.npy', 'wb') as f:
    np.save(save_path/'recorded_channel_names.npy', recorded_channels)
    np.save(save_path/'time_between_targets.npy', np.concatenate(time_between_targets))

def run(save_path, filenames, ppt_id):
    fig_checks.reset()
        
    datasets = []
    n_targets, n_samples, total_time = 0, 0, 0
    behavior_per_trial, time_between_targets = [], []

    locations = []
    for i, filename in enumerate(filenames):
        logger.info(f'Loaded {filename}')

        # Load and a bit of cleanup
        try:
            ds = load_dataset(filename, ppt_id) #data_path/filename)
        except Exception as err:
            logger.error(f'cannot load {filename}\n{err}')
            continue

        # print(filename)
        # continue
        ds.eeg, _ = flag_irrelevant_channels(ds.eeg)


        if c.rereferencing.common_electrode_reference:
            ds.eeg.timeseries = common_electrode_reference(ds.eeg.timeseries, ds.eeg.channels)
        
        elif c.rereferencing.laplacian:
            ds.eeg.timeseries = laplacian_reference(ds.eeg.timeseries, ds.eeg.channels)

        if c.timeshift:
            ds.eeg.timeseries, ds.xyz = timeshift(ds.eeg.timeseries, ds.xyz, t=c.timeshift)

        n_targets  += dataset_info.get_number_of_targets(ds)
        n_samples += dataset_info.get_number_of_samples(ds)
        time_between_targets += [dataset_info.get_time_between_targets(ds)]

        if type(ds.eeg.total_time) != float:
            logger.warning(f'Total time not float or int! {filename}')

        total_time += ds.eeg.total_time  # TODO: check value for kH040, might not be a complete experiment, but exception not caught?

        datasets  += prepare.go(ds, save_path, i)
    # u, counts = np.unique([ch.strip('0123456789') for ch in ds.eeg.channels], return_counts=True)
    # np.save(save_path.parent.parent/ f'kh{ds.ppt_id:03d}_locations.npy', list(ds.eeg.channel_map.values()))
    # np.save(save_path.parent.parent/f'kh{ds.ppt_id:03d}_info.npy', [ds.ppt_id,
    #                                         ds.eeg.channels.size,
    #                                         len(u),
    #                                         min(counts),
    #                                         max(counts)])
    # return
    # print(n_targets)
    n_gaps = len(datasets) - len(filenames)
    # plot_subsets(datasets, save_path)

    # explore.main(datasets, save_path)
    # return

    # print(total_time)
    # save_dataset_info(n_targets, n_samples, n_gaps, time_between_targets, total_time,
    #                   ds.ppt_id, datasets[0].channels, save_path)
    
    learner.fit(datasets, save_path)
    

# i=0; plt.figure(); plt.plot(datasets[0].xyz[:, i]); plt.plot(datasets[0].xyz[:, i+3]); plt.plot(datasets[0].xyz[:, i+6]); plt.savefig('tmp.png')
# plt.figure(); plt.plot(datasets[0].xyz[:, 9]); plt.plot(datasets[0].xyz[:, 10]); plt.plot(datasets[0].xyz[:, 11]); plt.savefig('tmp.png')

# Strong channel removal
# Select only beta
# Task correlations
# inlude hands
# Input dimensions to 30