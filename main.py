'''
https://github.com/ShanechiLab/PyPSID

'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PSID

import libs.ftd.prepare
from libs.leap.load import go as load_leap
from libs.windowing import window, mode

def go(y, z):
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
    
    y = (y-y.mean(axis=0))/y.std(axis=0)  # Z-score

    # For dev, select 50 random channels
    y = y[:, np.random.randint(0, y.shape[1], 50)]

    size = 0.8
    cutoff = int(size * y.shape[0])
    y_train, y_test = y[:cutoff, :], y[cutoff:, :]
    z_train, z_test = z[:cutoff, :], z[cutoff:, :]

    nx = 6                       # Total Number of latent states
    n1 = 3                       # Latent states dedicated to behaviorally relevant dynamics
    i = 5                      # Subspace horizon (low dim y = high i)

    id_sys = PSID.PSID(y_train, z_train, nx, n1, i)
    print('Fitted PSID')
    zh, yh, xh = id_sys.predict(y_test)

    plt.figure()
    plt.plot(z_test[:, 0], z_test[:, 1], label='True')
    plt.plot(zh[:, 0], zh[:, 1], label='Predicted')
    plt.show()

    print('done')

if __name__=='__main__':
    bubbles = True
    if bubbles:            
        data_path = Path('./data/kh036/')
        filename = f'bubbles_{1}.xdf'
        data, data_ts, idc, trials, events = load_leap(data_path/filename)

        # preprocess here
        # TEMP:
        import pandas as pd
        labels = pd.DataFrame(data[:, -3:]).interpolate().to_numpy()
        mask = ~np.isnan(labels)[:, 0]
        labels = labels[mask]
        eeg = data[mask, :-3]
        data_ts = data_ts[mask]

        wl = 500
        ws = 100
        fs = 1024

        eeg = window(eeg, data_ts, wl, ws, fs).mean(axis=1)
        labels = mode(window(labels, data_ts, wl, ws, fs))
        print('preprocessing done')
        go(eeg, labels)

    else:
        path = Path(f'./data/kh035/followthedotwithhand_1.xdf')
        data = libs.prepare.go(path)
        go(data['eeg'], data['trial_labels'])
