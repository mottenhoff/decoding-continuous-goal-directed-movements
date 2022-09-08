# Builtin
import pathlib
import sys
import bisect

# 3th party
import numpy as np
import scipy.signal
import pandas as pd
from scipy import fftpack
from scipy.stats import mode
from mne.filter import filter_data, notch_filter

#local
from libs.ftd.load import load_seeg, load_locs
from libs.ftd.quality import QualityChecker

def fill_missing(arr):
    return pd.DataFrame(arr).interpolate(method='linear', axis=0).to_numpy()
    # return pd.DataFrame(arr).fillna(method='ffill', axis=0).to_numpy()

def window(data: np.ndarray, ts: np.array, ws: int, fs: int, sr: int) -> np.ndarray: 
    # ts = timestamps [s]
    # ws = window size [ms]
    # fs = frameshift [ms]
    # sr = samplerate [Hz]
    fs *= 0.001
    ws *= 0.001

    data = np.expand_dims(data, axis=1) if data.ndim <= 1 else data
    ts -= ts[0] # set start of timeframe to 0 #TODO: align time series

    window_starts = np.arange(0, ts[-1]-ws, fs)  # start of window in milliseconds

    idc = np.searchsorted(ts, window_starts, side='right')

    samples_per_window = int(np.round(sr*ws))

    windows = np.dstack([data[idx:idx+samples_per_window, :] for idx in idc]).T
    windows = windows.transpose((0, 2, 1)) if windows.ndim == 3 else windows

    return windows.squeeze()

def check_quality(seeg):
    flagged_channels = np.array([])

    qc = QualityChecker()

    if any(qc.consistent_timestamps(seeg['eeg_ts'], seeg['fs'])):
        return 'invalid', seeg
  
    irrelevant_channels = np.concatenate([
        qc.get_marker_channels(seeg['eeg'], 
                               channel_names=seeg['channel_names']),
        qc.get_ekg_channel(seeg['eeg'], 
                           channel_names=seeg['channel_names']),
        qc.get_disconnected_channels(seeg['eeg'], 
                                     channel_names=seeg['channel_names'])
        ]).astype(int)

    noisy_channels = np.concatenate([
        qc.flat_signal(seeg['eeg']),
        qc.excessive_line_noise(seeg['eeg'],
                                seeg['fs'],
                                freq_line=50,
                                plot=0),
        qc.abnormal_amplitude(seeg['eeg'], plot=0)
        ]).astype(int)
    
    flagged_channels = np.union1d(irrelevant_channels, noisy_channels)

    return flagged_channels.astype(int)

def preprocess(eeg, fs):
    hilbert3 = lambda x: scipy.signal.hilbert(x, fftpack.next_fast_len(len(x)), 
                                              axis=0)[:len(x)]

    # Expects a [samples x channels] matrix
    eeg = scipy.signal.detrend(eeg, axis=0)
    eeg -= eeg.mean(axis=0)
    eeg = notch_filter(eeg.T, fs, np.arange(50, 201, 50)).T

    # delta = filter_data(eeg.T, sfreq=fs, l_freq=2, h_freq=4).T
    theta = filter_data(eeg.T, sfreq=fs, l_freq=4, h_freq=8).T
    alpha = filter_data(eeg.T, sfreq=fs, l_freq=8, h_freq=12).T
    beta = filter_data(eeg.T, sfreq=fs, l_freq=12, h_freq=30).T
    gamma = filter_data(eeg.T, sfreq=fs, l_freq=30, h_freq=55).T
    high_gamma = filter_data(eeg.T, sfreq=fs, l_freq=55, h_freq=90).T
    bb_gamma = filter_data(eeg.T, sfreq=fs, l_freq=90, h_freq=170).T

    bands = np.concatenate([
                            # delta,
                            theta,
                            alpha,
                            beta,
                            gamma,
                            high_gamma,
                            bb_gamma
                            ],
                           axis=1) 
    bands = abs(hilbert3(bands))

    return bands

def go(path):
    # 33
    # loc_path = r'./data/kh018/electrode_locations.csv'
    # seeg_path = r'L:\FHML_MHeNs\sEEG\kh017\20201030/followthedotwithhand_1.xdf'
    # # loc_path = r'./data/kh018/electrode_locations.csv'

    seeg = load_seeg(path)
    # seeg['locations'] = load_locs(loc_path)

    # Fill missing values in coordinates
    seeg['trial_labels'][:, 1:] = fill_missing(seeg['trial_labels'][:, 1:])


    # Remove data
    idx = np.where(seeg['trial_labels'][:, 0]==0)[0][0]
    seeg['eeg'] = seeg['eeg'][idx:, :]
    seeg['eeg_ts'] = seeg['eeg_ts'][idx:]
    seeg['trial_labels'] = seeg['trial_labels'][idx:, :]

    # Change to levels
    u = np.unique(np.round(np.diff(seeg['trial_labels'][:, 1:], axis=0).sum(axis=1), 1))
    translation = dict(zip(u, np.arange(u.size)))
    levels = np.vectorize(translation.get)(np.round(np.diff(seeg['trial_labels'][:, 1:], axis=0).sum(axis=1), 1))
    levels = np.concatenate([np.array([0]), levels])
    seeg['trial_labels'] = np.hstack((seeg['trial_labels'], np.expand_dims(levels, axis=1)))

    # Remove channels we cant use
    flagged_channels = check_quality(seeg)
    seeg['eeg'] = np.delete(seeg['eeg'], flagged_channels, axis=1)
    seeg['channel_names'] = np.delete(seeg['channel_names'], flagged_channels)    
    
    # Extract bandpower in each channel
    seeg['eeg'] = preprocess(seeg['eeg'], seeg['fs'])

    window_me = True
    if window_me:
        ws, fs = 300, 50
        seeg['eeg'] = window(seeg['eeg'], seeg['eeg_ts'], ws, fs, seeg['fs']).mean(axis=1)
        seeg['trial_labels'] = window(seeg['trial_labels'][:, 1:-1], seeg['eeg_ts'], ws, fs, seeg['fs']).mean(axis=1)
        # trial_nums = mode(window(seeg['trial_labels'][:, 0], seeg['eeg_ts'], ws, fs, seeg['fs']), axis=1)[0]
        # levels = mode(window(seeg['trial_labels'][:, -1], seeg['eeg_ts'], ws, fs, seeg['fs']), axis=1)[0]
    
    # Select specific data
    # levels = np.abs(levels - np.median(levels))
    # to_remove = np.where(levels==0)[0]
    
    # seeg['eeg'] = np.delete(seeg['eeg'], to_remove, axis=0)
    # # trial_nums = np.delete(trial_nums, to_remove, axis=0)
    # levels = np.delete(levels, to_remove, axis=0)
       
    # seeg['trial_labels'] = np.hstack([trial_nums, levels])

    return seeg

if __name__=='__main__':
    seeg = go(17)
    print('')