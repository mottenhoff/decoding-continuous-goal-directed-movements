# Builtin
from collections import Counter

# 3th party
import numpy as np
import scipy.signal
from scipy import fftpack
from mne.filter import filter_data, notch_filter

# Local

def mode(arr: np.array, axis=1):
    '''
     Takes mode over a specified axis from
     n-dimensional numpy array 

    Parameters
    ----------
    arr: np.array [n dims]
    axis: int
          Axis to take the mode

    Returns
    ----------
    arr: np.array [n-1 dims]
         arr where 1 dimension is condensed 
         to the mode of that axis
    '''

    mode_1d_arr = lambda x: Counter(x.tolist()).most_common()[0][0]
    return np.apply_along_axis(mode_1d_arr, axis, arr)

def window(arr: np.ndarray, ts: np.array, wl: int, ws: int, fs: int) -> np.ndarray: 
    '''
    Windowing function

    Parameters
    ----------
    arr: np.array[samples x channels]
         data to window
    ts:  np.array[timestamps x 1]
         timestamps corresponding to arr [seconds]
    wl:  int
         length of window [milliseconds]
    ws:  int
         window shift [milliseconds]
    fs:  int
         sample frequency of amplifier [Hz]
    
    Returns
    ----------
    arr: 3d np.array[windows x samples_per_window x channels]
         windowed data
    '''

    # Because ts is in seconds. This saves multiplications
    #     Also, multiplication is many times quicker than
    #     division
    wl *= 0.001
    ws *= 0.001

    arr = np.expand_dims(arr, axis=1) if arr.ndim <= 1 else arr

    ts -= ts[0] # set start of timeframe to 0

    window_starts = np.arange(0, ts[-1]-wl, ws)  # start of window in seconds

    idc = np.searchsorted(ts, window_starts, side='right')

    samples_per_window = int(np.round(fs*wl))

    windows = np.dstack([arr[idx:idx+samples_per_window, :] for idx in idc]).T
    windows = windows.transpose((0, 2, 1)) if windows.ndim == 3 else windows

    return windows.squeeze()

def instantaneous_powerbands(eeg, fs):
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
                            # theta,
                            alpha,
                            beta,
                            # gamma,
                            high_gamma,
                            # bb_gamma
                            ],
                           axis=1) 
    bands = abs(hilbert3(bands))

    return bands