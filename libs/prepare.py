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

def get_windows(subset, wl, ws):
    subset.eeg = utils.window(subset.eeg, subset.ts,
                              wl, ws, subset.fs)
    subset.xyz = utils.window(subset.xyz, subset.ts,
                              wl, ws, subset.fs)

    subset.eeg = np.mean(subset.eeg, axis=1)
    subset.xyz = np.nanmean(subset.xyz, axis=1)

    if subset.xyz.ndim == 1:
        subset.xyz = subset.xyz[:, np.newaxis]

    is_all_nan = np.all(np.isnan(subset.xyz), axis=1)

    subset.eeg = subset.eeg[~is_all_nan, :]
    subset.xyz = subset.xyz[~is_all_nan, :]
                    
    return subset

def fill_missing_values(data):
    return pd.DataFrame(data).interpolate().to_numpy()

def get_subset_idc(xyz):

    n = c.missing_values.xyz_samples_for_gaps

    idc = np.where(~np.isnan(xyz[:, 0]))[0] 
    diff = np.diff(idc)
    gaps = np.where(diff > n*diff.mean())[0] # about 100 samples...

    gaps = np.hstack((-1, gaps, idc.size-1))
    
    subset_idc = [(idc[start_i], idc[end_i]) for start_i, end_i in zip(gaps[:-1]+1, gaps[1:])]

    # Sanity check
    if True:
        # Green = Start of gap
        # Red = End of gap
        plt.figure()
        plt.plot(idc, xyz[idc, -1])
        ylim_max = plt.ylim()[1]
        for si, ei in subset_idc:
            plt.vlines(si, ymin=0, ymax=ylim_max, colors='g', linewidth=1, linestyles='--')
            plt.vlines(ei, ymin=0, ymax=ylim_max, colors='r', linewidth=1, linestyles='--')
            plt.savefig(f'./figures/checks/get_subset_idc_{si}_{ei}.svg')

    return subset_idc

def go(eeg, xyz):

    if not c.debug:
        eeg['data'] = utils.instantaneous_powerbands(eeg['data'], eeg['fs'], c.bands)

    subset_idc = get_subset_idc(xyz)

    subsets = []
    for s, e in subset_idc:
        
        subset = Subset(eeg = eeg['data'][s:e, :],
                        xyz = xyz[s:e, :],
                        ts  = eeg['ts'][s:e],
                        fs  = eeg['fs'],
                        channels = eeg['channel_names'])

        # Fill missing values
        subset.xyz = fill_missing_values(subset.xyz)

        # Window
        subset = get_windows(subset,
                             c.window.length,
                             c.window.shift)

        subsets.append(subset)

    return subsets


    # # Check if there is enough data to create min_windows
    # min_windows = c.missing_values.min_windows_to_incl_set  # TODO rename
    # n_samples = min_windows*c.window.length/1000*eeg['fs']
    # subsets = [s for s in subsets if s.eeg.shape[0] > n_samples]

    # if c.debug_reduce_channels:
    #     # Select only the first 30 channels
    #     # TODO: also remove channel_names
    #     n = 30
    #     logging.debug(f'Reducing amount of features to {n}')
    #     subset.eeg = subset.eeg[:, :n]
    #     subset.channels = subset.channels[:n]