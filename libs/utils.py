# Builtin
import logging
from collections import Counter
from types import SimpleNamespace

# 3th party
import numpy as np
import scipy.signal
import yaml
import mne
from scipy import fftpack
from mne.filter import filter_data, notch_filter

# Local
mne.set_log_level('WARNING')

def nested_namespace_to_dict(ns):
    if type(ns) == SimpleNamespace:
        d = ns.__dict__
    else:
        return ns

    for k, v in d.items():
        if type(ns) == SimpleNamespace:
            d[k] = nested_namespace_to_dict(v)
        else:
            d[k] = v

    return d

def nested_dict_to_namespace(d):
    new_dict = {}
    for k, v in d.items():
    
        if type(v) == dict:
            new_dict[k] = nested_dict_to_namespace(v)
        else:
            new_dict[k] = v
    
    return SimpleNamespace(**new_dict)

def load_yaml(path):
    """Loads the data from a .yaml file

    Gets the data from a .yaml file, the user should specify the full path to the file.

    Arguments:
        path_to_file -- full path to the .yaml file to load

    Returns:
        data contained on the .yaml file
    """

    try:
        with open(path) as file_obj:
            config = yaml.load(file_obj, Loader=yaml.FullLoader)
        return nested_dict_to_namespace(config)
    except Exception:
        raise Exception('Failed to load config file from: {}.'.format(path))

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

def instantaneous_powerbands(eeg, fs, bands):

    logging.info(f'Filtering data | fs={fs}, bands:', bands)

    if eeg.dtype is not (required_type := 'float64'):
        eeg = eeg.astype(required_type)

    hilbert3 = lambda x: scipy.signal.hilbert(x, fftpack.next_fast_len(len(x)), 
                                              axis=0)[:len(x)]

    # Expects a [samples x channels] matrix
    eeg = scipy.signal.detrend(eeg, axis=0)
    eeg -= eeg.mean(axis=0)
    eeg = notch_filter(eeg.T, fs, np.arange(50, 201, 50)).T

    filtered = np.concatenate([filter_data(eeg.T, sfreq=fs,
                                           l_freq=f[0], h_freq=f[1]).T \
                               for band, f in bands.items()], axis=1)

    return abs(hilbert3(filtered))

if __name__=='__main__':

    conf = load_yaml('config.yml')

    d = nested_namespace_to_dict(conf)
    
    print('')

