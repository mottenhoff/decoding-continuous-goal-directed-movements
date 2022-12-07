import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from libs import utils
from figures import checks as fig_checks

from figures.figure_cc_per_band_per_kinematic import plot_band_correlations

START = 0
END = 1

logger = logging.getLogger(__name__)
c = utils.load_yaml('./config.yml')

@dataclass
class Subset:
    eeg: np.array
    xyz: np.array
    ts:  np.array
    fs:  np.array
    channels: np.array
    mapping: dict

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
    # TODO: check if needed to handle  leading and trailing nans
    return pd.DataFrame(data).interpolate().to_numpy()

def get_subset_idc(xyz, fs):

    n = c.missing_values.xyz_samples_for_gaps

    # Get absolute amount of samples in subset for at least 1 windows
    min_samples = c.window.length * 0.001 * 1024
    # Check if the user set amount of windows is met
    min_samples = np.ceil(c.missing_values.min_windows_to_incl_set * min_samples)

    idc = np.where(~np.isnan(xyz[:, 0]))[0] 
    diff = np.diff(idc)
    gaps = np.where(diff > n*diff.mean())[0] # about 100 samples...

    gaps = np.hstack((-1, gaps, idc.size-1))
    
    subset_idc = [(idc[start_i], idc[end_i]) for start_i, end_i in zip(gaps[:-1]+1, gaps[1:]) \
                                             if (idc[end_i] - idc[start_i]) > min_samples]

    fig_checks.plot_gap_cuts(xyz, idc, subset_idc)

    return subset_idc

def go(eeg, xyz):

    if not c.debug.go:
        # eeg['data'] = utils.instantaneous_powerbands(eeg['data'], eeg['fs'], c.bands.__dict__)
        # logger.info(f'Filtered: {c.bands.__dict__}')
        pass

    if c.target_vector:
        # Get target_vector
        # Stack to eeg
        pass

    subset_idc = get_subset_idc(xyz, eeg['fs'])

    largest_subset = np.argmax([s[1]-s[0] for s in subset_idc])

    subsets = []
    for i, (s, e) in enumerate(subset_idc):
        
        subset = Subset(eeg = eeg['data'][s:e, :],
                        xyz = xyz[s:e, :],
                        ts  = eeg['ts'][s:e],
                        fs  = eeg['fs'],
                        channels = eeg['channel_names'],
                        mapping = eeg['channel_mapping'])

        # subset.eeg = fill_missing_values(subset.eeg)
        subset.xyz = fill_missing_values(subset.xyz)

        if i == largest_subset:
            # Do exploratory analysis here
            pass
            # plot_band_correlations(subset.eeg, subset.xyz, '', get_bands=False)
            # plot_band_correlations(subset.eeg, subset.xyz, '', get_bands=True)

        # Window
        subset = get_windows(subset,
                             c.window.length,
                             c.window.shift)

        if i == largest_subset:
            # plot_band_correlations(subset.eeg, subset.xyz, '_windowed')
            pass

        subsets.append(subset)

    return subsets