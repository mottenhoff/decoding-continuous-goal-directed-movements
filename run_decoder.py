import logging

import numpy as np
import yaml

import learner
from libs import prepare, utils, explore, dataset_info
from libs.rereference import common_electrode_reference, laplacian_reference
from libs.load import load_dataset
from libs.data_cleaning import flag_irrelevant_channels

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

    np.save(save_path/'recorded_channel_names.npy', recorded_channels)
    np.save(save_path/'time_between_targets.npy', np.concatenate(time_between_targets))

def run(save_path, filenames, ppt_id):
    datasets = []
    n_targets, n_samples, total_time = 0, 0, 0
    _, time_between_targets = [], []

    for i, filename in enumerate(filenames):
        logger.info(f'Loaded {filename}')

        # Load and a bit of cleanup
        try:
            ds = load_dataset(filename, ppt_id)
        except Exception as err:
            logger.error(f'cannot load {filename}\n{err}')
            continue

        ds.eeg, _ = flag_irrelevant_channels(ds.eeg)

        if c.rereferencing.common_electrode_reference:
            ds.eeg.timeseries = common_electrode_reference(ds.eeg.timeseries, ds.eeg.channels)
        elif c.rereferencing.laplacian:
            ds.eeg.timeseries = laplacian_reference(ds.eeg.timeseries, ds.eeg.channels)

        if type(ds.eeg.total_time) != float:
            logger.warning(f'Total time not float or int! {filename}')

        total_time += ds.eeg.total_time
        n_targets  += dataset_info.get_number_of_targets(ds)
        n_samples += dataset_info.get_number_of_samples(ds)
        time_between_targets += [dataset_info.get_time_between_targets(ds)]

        datasets  += prepare.go(ds, save_path, i)

    explore.main(datasets, save_path)

    save_dataset_info(n_targets, n_samples, -1, time_between_targets, total_time,
                      ds.ppt_id, datasets[0].channels, save_path)
    
    learner.fit(datasets, save_path)