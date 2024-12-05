import logging
from dataclasses import dataclass
import pickle

from bisect import bisect_right
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import libs.utils
from libs.utils import filter_eeg, hilbert
from libs import kinematics
from figures import checks as fig_checks

# from figures.figure_cc_per_band_per_kinematic import plot_band_correlations

from libs import kinematics

START = 0
END = 1

logger = logging.getLogger(__name__)
c = libs.utils.load_yaml('./config.yml')

@dataclass
class Subset:
    eeg: np.array
    xyz: np.array
    target_vector: np.array
    trials: np.array
    ts: np.array
    fs: np.array
    channels: np.array
    mapping: dict

def get_fft(signal, fs):
    powerspectrum = np.abs(np.fft.rfft(signal)**2)
    freqs = np.fft.rfftfreq(signal.size, 1/fs)
    return freqs, powerspectrum

def get_behavior_per_trial(subset):

    # Get all unique trial numbers
    trial_nums = np.unique(subset.trials[np.where(~np.isnan(subset.trials[:, 0])), 0])

    # Find the start of each trial by occurence of the first appearance of the trial num
    idc_trial_starts = [np.where(subset.trials[:, 0]==num)[0][0] for num in trial_nums]

    # Slice the data based on the start index and the start of the second index.
    return [subset.xyz[s:e, :] for s, e in zip(idc_trial_starts[:-1], idc_trial_starts[1:])]

def get_windows(ts, signal, fs, wl, ws):
    size_start = signal.shape

    signal = libs.utils.window(signal, ts, wl, ws, fs)

    logger.info(f"Creating windows of length = {wl}ms and stepsize = {ws}ms from {size_start} sample to {signal.shape} windows [win, samp, chs]")
    
    signal = np.nanmean(signal, axis=1)

    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    # is_all_nan = np.all(np.isnan(subset.xyz), axis=1)  # Why is this here?

    # subset.eeg = subset.eeg[~is_all_nan, :]
    # subset.xyz = subset.xyz[~is_all_nan, :]

    # logger.info(f'Removed {is_all_nan.sum()} out of {is_all_nan.size} windows with only nan')

    return signal

def fill_missing_values(data):
    # TODO: check if needed to handle  leading and trailing nans

    data = pd.DataFrame(data)
    isnan = data.isna().sum(axis=0)
    logger.info(f'Interpolating missing values: On average {isnan.mean()} samples ({isnan.mean()/data.shape[0]*100:.1f}%) ')

    return pd.DataFrame(data).interpolate().to_numpy()

# # NumPy-only function
# def fill_missing_values_np(data):
#     data = np.array(data)
#     nan_indices = np.isnan(data).sum(axis=0)
#     nan_mean = np.mean(nan_indices)
#     print(f'Interpolating missing values: On average {nan_mean} samples ({nan_mean/data.shape[0]*100:.1f}%)')

#     data = np.nan_to_num(data, nan=np.nan, copy=False)
#     for i in range(data.shape[1]):
#         column = data[:, i]
#         indices = np.arange(len(column))
#         valid_indices = indices[~np.isnan(column)]
#         invalid_indices = indices[np.isnan(column)]
#         data[:, i][invalid_indices] = np.interp(invalid_indices, valid_indices, column[valid_indices])

#     return data
def locate_pos(ts, target_ts):
    pos = bisect_right(ts, target_ts)
    if pos == 0:
        return 0
    if pos == len(ts):
        return len(ts)-1
    if abs(ts[pos]-target_ts) < abs(ts[pos-1]-target_ts):
        return pos
    else:
        return pos-1

def align_matrices_with_diff_fs(large, ts_large, small, ts_small):
    '''
    ts = timestamps
    idc = align indices where values of small should be inserted in large
    '''
    small = small[:, np.newaxis] if small.ndim == 1 else small

    small_ext = np.full((large.shape[0], small.shape[1]), np.nan)

    idc = np.array([locate_pos(ts_large, nt) for nt in ts_small])  # inserted at
    small_ext[idc, :] = small

    return small_ext, idc

def get_subset_idc(xyz, fs, dataset_num, path=None):
    '''
    Identifies the start and end index of subsets
    in the data, by two metrics:

    1) A gap is identified if n samples are missing. 
        n is user defined in config.missing_values.xyz_samples_for_gaps.
    2) A subset is rejected if it contains less than 
        n windows, 
        n windows is user defined in c.missing_values.min_windows_to_incl_set. 

    Returns:
        list of tuples containing start and end index of a subset
        [(idx_start, idx_end), ...]
    '''
    n = c.missing_values.xyz_samples_for_gaps  # Samples in Xyz for missing

    # Get absolute amount of samples in subset for at least 1 windows
    min_samples = c.window.length * 0.001 * 1024

    # Check if the user set amount of windows is met
    min_samples = np.ceil(c.missing_values.min_windows_to_incl_set * min_samples)

    # Idenify gaps
    idc = np.where(~np.isnan(xyz[:, 0]))[0] 
    diff = np.diff(idc)
    gaps = np.where(diff > n*diff.mean())[0] # about 100 samples... (5 samples in xyz time)
    gaps = np.hstack((-1, gaps, idc.size-1))
    
    # Find start and end indices of subsets, and reject
    # subsets with too little data.
    subset_idc = [(idc[start_i], idc[end_i]) for start_i, end_i in zip(gaps[:-1]+1, gaps[1:]) \
                                             if (idc[end_i] - idc[start_i]) > min_samples]

    fig_checks.plot_gap_cuts(xyz, idc, subset_idc, dataset_num, path)

    return subset_idc

def frequency_decomposition(eeg: np.array, fs: float):
    frequency_delta = [0, 4]
    frequency_ab    = [8, 30]
    frequency_bbhg  = [55, 200]

    filtered = []
    
    if c.filters.delta:
        filtered += [filter_eeg(eeg, fs, frequency_delta[0], frequency_delta[1])]

    if c.filters.alphabeta:
        filtered += [hilbert(filter_eeg(eeg, fs, frequency_ab[0], frequency_ab[1]))]
    
    if c.filters.bbhg:
        filtered += [hilbert(filter_eeg(eeg, fs, frequency_bbhg[0], frequency_bbhg[1]))]

    if not filtered:
        filtered = [eeg]

    return np.hstack(filtered)


def go(ds, save_path, ds_idx):
    '''
    o Extract all features:
        1. Delta activity:  < 5 Hz
            Explicitly extract phase? https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
        2. AlphaBeta Power: 8 - 30 Hz
        3. Broadband Gamma power: 55 - 200 Hz
    
    o Split into subsets to address gaps in handtracking data
        1. Get identify gaps and get start and end
        2. split eeg and xyz
        3. For each subset:
            a. Fill missing values in xyz --> Or downsample, check if methods behave correctly at gaps
            b. Downsample to (say) 20 Hz,  equalize framerates.
    '''
    
    # import matplotlib.pyplot as plt

    ds.eeg.timeseries = frequency_decomposition(ds.eeg.timeseries, ds.eeg.fs)
    # ds.eeg.channels = np.concatenate([list(map(lambda x: x + f'-{band}', ds.eeg.channels)) for band in ['delta', 'alphabeta', 'bbhg']])  # Uncomment if problems later

    # Align xyz to higher samples eeg
    xyz, inserted_idc_xyz = align_matrices_with_diff_fs(
                                ds.eeg.timeseries,
                                ds.eeg.timestamps,
                                ds.xyz,
                                ds.xyz_timestamps)
    
    trials, inserted_idc_trials = align_matrices_with_diff_fs(
                                    ds.eeg.timeseries,
                                    ds.eeg.timestamps,
                                    ds.trials, 
                                    ds.xyz_timestamps)

    target_vector, inserted_idc_trials = align_matrices_with_diff_fs(
                                    ds.eeg.timeseries,
                                    ds.eeg.timestamps,
                                    ds.target_vector, 
                                    ds.xyz_timestamps)

    ds.xyz = xyz
    ds.trials = trials
    ds.target_vector = target_vector
    
    # indices_with_value = np.where(~np.isnan(ds.xyz[:, 0]))[0]
    # ds

    assert inserted_idc_xyz[0] == inserted_idc_trials[0], logger.error('xyz and trials misaligned!')
    assert inserted_idc_xyz.size == inserted_idc_trials.size, logger.error('xyz and trials differ in size!')
    

    subset_idc = get_subset_idc(ds.xyz, ds.eeg.fs, ds_idx, path=save_path)

    subsets = []
    behavior_per_trial = []
    for i, (s, e) in enumerate(subset_idc):

        logger.info(f'Cutting subset {i} from: {s} to {e}')
        
        subset = Subset(eeg =           ds.eeg.timeseries[s:e, :],
                        xyz =           ds.xyz[s:e, :],
                        target_vector = ds.target_vector[s:e, :],
                        trials =        ds.trials[s:e, :],
                        ts  =           ds.eeg.timestamps[s:e],
                        fs  =           ds.eeg.fs,
                        channels =      ds.eeg.channels,
                        mapping =       ds.eeg.channel_map)

        # fig, ax = plt.subplots()
        # idc = np.where(~np.isnan(ds.trials[:, 0]))[0]
        # ax.scatter(idc, ds.trials[idc, 1])
        # ax.plot(idc, ds.trials[idc, 1])
        # ax.axvline(s, color='green')
        # ax.axvline(e, color='red')

        # Sometimes there is no trial information, making the 
        print()
        # Breaks when there is a gap, but too short to be filled as subset
        # assert (np.where(~np.isnan(subset.xyz[:, 0]))[0] == np.where(~np.isnan(subset.trials[:, 0]))[0]).all(), logger.error('xyz and trials idc are not identical!')

        # if (~np.isnan(subset.xyz)).sum() == 0:
        #     logger.error('No values in subset!')
        #     raise Exception('No values in subset!')

        if c.target_vector:  # If triggered, then xyz and trials are misalinged

            # if (~np.isnan(subset.target_vector)).sum() == 0:
            #     print(s, e)

            #     # plt.show()
            #     # raise Exception('TV has no values in subset!')
            #     logger.warning('TV has no values in subset!')
            #     continue
            
            subset.xyz = subset.target_vector

        subset.xyz = kinematics.get_all(subset, has_target_vector=c.target_vector)
        
        before = subset.xyz.copy()
        subset.xyz = fill_missing_values(subset.xyz)
        

        behavior_per_trial += get_behavior_per_trial(subset)

        # if subset.xyz[:, 10].max() > 2000:
            
        #     print('high spike in speed detected', s, e)
        #     fig, ax = plt.subplots(nrows=3)
        #     ax[0].plot(subset.xyz[s:e, 10])
        #     ax[1].plot(subset.trials[s:e, 1])
        #     ax[2].plot(subset.xyz[s:e, 0])
        #     plt.show()

        subset.eeg = get_windows(subset.ts, subset.eeg, subset.fs, c.window.length, c.window.shift)
        subset.xyz = get_windows(subset.ts, subset.xyz, subset.fs, c.window.length, c.window.shift)

        if np.isnan(subset.xyz).sum() > 0:
            logger.error('Missing values found after filling missing values van windowing!')
            raise Exception('Missing values found after filling missing values!')

        subsets.append(subset)

    with open(save_path/f'behavior_per_trial_{ds_idx}.pkl', 'wb') as f:
        pickle.dump(behavior_per_trial, f)

    # Quickplot: plt.close('all');plt.figure();[plt.plot(subset.xyz[:100, 10]) for subset in subsets];plt.savefig('tmp.png')

    return subsets

