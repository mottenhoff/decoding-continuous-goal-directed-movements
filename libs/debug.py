import logging
import numpy as np

from libs import utils

c = utils.load_yaml('./config.yml')

logger = logging.getLogger(__name__)

def dummy_xyz(eeg, xyz):
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

def shorten_dataset(eeg, xyz):
    n_samples = 20000
    logger.warning(f'DEBUG=ON | Shortening data for debugging to {n_samples} samples')
    eeg['data'] = eeg['data'][:20000, :]
    eeg['ts'] = eeg['ts'][:20000]
    xyz = xyz[:20000, :]
