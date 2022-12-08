import matplotlib.pyplot as plt
import numpy as np

CC = 0
R2 = 1
MSE = 2
RMSE = 3

def load(path):
    m =  np.load(path/'metrics.npy')  # CC, R2, MSE, RMSE
    z =  np.load(path/'z.npy')
    # y =  np.load(path/'neural_activity.npy')
    y =  np.load(path/'y.npy')
    zh = np.load(path/'trajectories.npy')
    yh = np.load(path/'neural_reconstructions.npy')    
    xh = np.load(path/'latent_states.npy')

    return m, z, y, zh, yh, xh

def make(path):
    
    m, z, y, zh, yh, xh = load(path)

    z, zh = z.ravel(), zh.ravel()
    m = m.squeeze()

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(z, label='z-true')
    ax.plot(zh, label='z-pred')
    ax.set_title(f'Z reconstruction | cc={m[:, CC].mean():.3f}\u00b1{m[:, CC].std():.3f}')
    ax.set_xlabel(f'Time [windows]')
    ax.set_ylabel(f'Speed')
    fig.legend()
    fig.savefig(path/'z-reconstruction.svg')
    fig.savefig(path/'z-reconstruction.png')    

    zpsd = z - z.mean()
    psd = np.abs(np.fft.rfft(z))**2
    freqs = np.fft.rfftfreq(z.size, 0.05)
    idx = np.argsort(freqs)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(freqs[idx], psd[idx])
    plt.savefig(path/'z-psd.png')