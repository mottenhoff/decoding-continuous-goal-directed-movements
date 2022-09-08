'''
https://github.com/ShanechiLab/PyPSID

'''
import itertools
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PSID
from sklearn.decomposition import PCA

import libs.ftd.prepare
import utils
from libs.leap.load import go as load_leap
from libs.leap.plotting import plot_trajectory

PLOT  = 1
DEBUG = 0

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
    nx = [20, 30, 40]          # Total Number of latent states
    n1 = [3, 5, 10]            # Latent states dedicated to behaviorally relevant dynamics
    i  = 5                      # Subspace horizon (low dim y = high i)

    # y = data['eeg']           # Neural data
    # z = data['trial_labels']  # Behavioral data
    n_folds = 5
    n_inner_folds = 4
    # opts = list(itertools.product(nx, n1, [i]))

    folds = np.array_split(np.arange(y.shape[0]), n_folds)
    for idx, fold in enumerate(folds):

        y_train, y_test = y[fold, :], np.delete(y, fold, axis=0)
        z_train, z_test = z[fold, :], np.delete(z, fold, axis=0)

        # Z-score both neural activity and behavior
        y_train, y_test = normalize(y_train, y_test)
        z_train, z_test = normalize(z_train, z_test)

        # Dim red.
        pca = PCA(n_components=20).fit(y_train)
        y_train = pca.transform(y_train)
        y_test = pca.transform(y_test)

        id_sys = PSID.PSID(y_train, z_train, 30, 10, i)
        print(f'{idx}: Fitted PSID')
        zh, yh, xh = id_sys.predict(y_test)

        if PLOT:
            ax = plot_trajectory(z_test, zh, label=idx)

    plt.show()
    print('done')

if __name__=='__main__':
     
    data_path = Path('./data/kh036/')
    filename = f'bubbles_{1}.xdf'
    data, data_ts, idc, trials, events = load_leap(data_path/filename)

    eeg, labels, ts = prepare(data, data_ts)

    if DEBUG:
        eeg = eeg[:200, :]
        labels = labels[:200, :]
    fit(eeg, labels)

    # else:
    #     path = Path(f'./data/kh035/followthedotwithhand_1.xdf')
    #     data = libs.prepare.go(path)
    #     go(data['eeg'], data['trial_labels'])
