'''
https://github.com/ShanechiLab/PyPSID

'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PSID

import libs.ftd.prepare
import utils
from libs.leap.load import go as load_leap
from libs.leap.plotting import plot_trajectory

PLOT = True
DEBUG = False

def z_score(x: np.array, 
            u: np.array, s:np.array) -> np.array:
    #  x: [n x m]  u: [1 x m]  s: [1 x m]
    return (x-u) / s

def normalize(a, b) -> np.array:
    u, s = a.mean(axis=0), b.std(axis=0)
    return z_score(a, u, s), z_score(b, u, s)

def remove_missing_values(data, ts):
    # Interpolates and strips dangling nans
    # TODO: Handles nans at start and end differently. Should have markers
    # in new exp.

    labels = pd.DataFrame(data[:, -3:]).interpolate().to_numpy()
    mask = ~np.isnan(labels)[:, 0]
    labels = labels[mask]
    eeg = data[mask, :-3]
    ts = ts[mask]

    return eeg, labels, ts

def get_windows(eeg, labels, ts, wl, ws, fs):
    eeg = utils.window(eeg, ts, wl, ws, fs).mean(axis=1)
    labels = utils.mode(utils.window(labels, ts, wl, ws, fs))

    return eeg, labels

def prepare(data, ts):
    wl, ws, fs = 500, 100, 1024

    eeg, labels, ts = remove_missing_values(data, ts)

    if DEBUG:
        eeg = eeg[:, np.random.randint(0, eeg.shape[1], 30)]

    eeg = utils.instantaneous_powerbands(eeg, fs)
    eeg, labels = get_windows(eeg, labels, ts, wl, ws, fs)

    return eeg, labels, ts

def fit(y, z):
    '''
    y = Neural data
    z = behavioral data

    Fs = 1024
    Downsample: 20 Hz  --> wl = 300ms, fs = 50ms
    Bands: Theta, alpha, low beta, high beta, low gamma, high gamma, broadband gamma
    Z-score    
    
    '''

    # y = data['eeg']           # Neural data
    # z = data['trial_labels']  # Behavioral data

    # Split
    size = 0.8
    cutoff = int(size * y.shape[0])
    y_train, y_test = y[:cutoff, :], y[cutoff:, :]
    z_train, z_test = z[:cutoff, :], z[cutoff:, :]
    
    # Z-score both neural activity and behavior
    y_train, y_test = normalize(y_train, y_test)
    z_train, z_test = normalize(z_train, z_test)

    nx = 6                       # Total Number of latent states
    n1 = 3                       # Latent states dedicated to behaviorally relevant dynamics
    i = 5                      # Subspace horizon (low dim y = high i)

    id_sys = PSID.PSID(y_train, z_train, nx, n1, i)
    print('Fitted PSID')
    zh, yh, xh = id_sys.predict(y_test)

    if PLOT:
        ax = plot_trajectory(z_test, zh)
        plt.show()

    print('done')

if __name__=='__main__':
     
    data_path = Path('./data/kh036/')
    filename = f'bubbles_{1}.xdf'
    data, data_ts, idc, trials, events = load_leap(data_path/filename)

    eeg, labels, ts = prepare(data, data_ts)

    fit(eeg, labels)

    # else:
    #     path = Path(f'./data/kh035/followthedotwithhand_1.xdf')
    #     data = libs.prepare.go(path)
    #     go(data['eeg'], data['trial_labels'])
