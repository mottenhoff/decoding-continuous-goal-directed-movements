import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from libs import utils


START = 0
END = 1

c = utils.load_yaml('./config.yml')

@dataclass
class Subset:
    eeg: np.array
    xyz: np.array
    ts:  np.array
    fs:  np.array
    channels: np.array

def get_windows(eeg, xyz, ts, wl, ws, fs):
    eeg = utils.window(eeg, ts, wl, ws, fs).mean(axis=1)
    xyz = utils.window(xyz, ts, wl, ws, fs).mean(axis=1)

    return eeg, xyz

def fill_missing_values(xyz):
    '''
    Linearly interpolates the xyz data, unless
    there is a "large" gap.

    "Large" is defined arbitrarily. The larger 
    the gap, the less the assumption of linear
    progression holds in the data.

    Thus this fuction will do:
    
    Check for gaps
    if gap:
        split data
    
    for set in datasets:
        interpolate(set)
        strip leading and trailing nans 

    return list of subsets + the corresponding idc
    '''
    n = c.missing_values.xyz_samples_for_gaps

    idc = np.where(~np.isnan(xyz[:, 0]))[0]
    diff = np.diff(idc)
    gaps = np.where(diff > n*diff.mean())[0]  # about 100 samples...
                                              # idx = start of gap
    logging.info(f'Gaps defined as diff > {n}*diff.mean()')

    # Split into seperate datasets from gap to gap.
    # if there are not gaps it will select the whole dataset
    gaps = np.hstack([0, gaps, idc.size-1])

    subsets = []
    for start_idc, end_idc in zip(gaps[:-1], gaps[1:]):
        start, end = idc[start_idc], idc[end_idc]
        subset = xyz[start:end, :]
        subset = pd.DataFrame(subset).interpolate().to_numpy()

        is_leading_or_trailing_nans = np.all(np.isnan(subset), axis=1)
        subset = subset[~is_leading_or_trailing_nans, :]

        subsets += [[subset, [start, end]]]

    logging.info(f'Interpolated xyz and removed trailing and leading nans')

    # TODO: check if this is missing a sample at the end
    return subsets

def list_to_dataclass(eeg, xyzs):
    
    subsets = []
    for xyz, idc in xyzs:
        start, end = idc[START], idc[END]
        subsets += [Subset(
            eeg = eeg['data'][start:end, :],
            xyz = xyz,
            ts = eeg['ts'][start:end],
            fs = eeg['fs'],
            channels = eeg['channel_names'])]

    return subsets

def go(eeg, xyz):



    if not c.debug:
        # TODO: Save order of powerbands somewhere (incl channels?)
        eeg['data'] = utils.instantaneous_powerbands(eeg['data'], eeg['fs'], c.bands)
    
    xyz_subsets = fill_missing_values(xyz)  # Return subsets
    subsets = list_to_dataclass(eeg, xyz_subsets)

    # Check if there is enough data to create min_windows
    min_windows = 2  # TODO: move to config
    n_samples = min_windows*c.window.length/1000*eeg['fs']
    subsets = [s for s in subsets if s.eeg.shape[0] > n_samples]

    for subset in subsets:

        if c.debug:
            # Select only the first 30 channels
            n = 30
            logging.debug(f'Reducing amount of features to {n}')
            subset.eeg = subset.eeg[:, :n]
            subset.channels = subset.channels[:n]

        # are these modified in the list?
        subset.eeg, subset.xyz = get_windows(
                                    subset.eeg, 
                                    subset.xyz,
                                    subset.ts,
                                    c.window.length,
                                    c.window.shift,
                                    subset.fs)

    # if conf['velocity']:
    #     # TODO: Includes high spikes....
    #     # move to before splitting into subsets

    #     labels = np.diff(labels, axis=0)
    #     eeg = eeg[1:, :]

    return subsets