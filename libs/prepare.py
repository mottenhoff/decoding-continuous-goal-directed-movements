import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from libs import utils
from figures import checks as fig_checks


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

    fig_checks.plot_gap_cuts(xyz, idc, subset_idc)

    return subset_idc

def go(eeg, xyz):
    
    if not c.debug.go:
        tv = eeg['data'][:, -3:] if c.target_vector else np.empty((eeg.shape[0], 0))
        eeg['data'] = eeg['data'][:, :-3] if c.target_vector else eeg['data']
        eeg['data'] = utils.instantaneous_powerbands(eeg['data'], eeg['fs'], c.bands)
        eeg['data'] = np.hstack((eeg['data'], tv))

    eeg['data'] = fill_missing_values(eeg['data'])  # This is only for the target vector, eeg itself shouldnt contain any missing values
    xyz = fill_missing_values(xyz)
            
    subset_idc = get_subset_idc(xyz)

    subsets = []
    for s, e in subset_idc:
        
        subset = Subset(eeg = eeg['data'][s:e, :],
                        xyz = xyz[s:e, :],
                        ts  = eeg['ts'][s:e],
                        fs  = eeg['fs'],
                        channels = eeg['channel_names'])

        # Window
        subset = get_windows(subset,
                             c.window.length,
                             c.window.shift)

        subsets.append(subset)

    return subsets