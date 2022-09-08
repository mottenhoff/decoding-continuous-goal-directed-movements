import matplotlib.pyplot as plt
import numpy as np


def plot_effective_framerate(ts):
        # Spikes are suspected to be hand out of bounds. 
    # No label included in initial pilots
    plt.plot(ts, np.concatenate([[0], np.diff(ts)]))
    plt.title(f'Effective framerate: {1/np.diff(ts).mean():.2f} Hz')
    plt.xlabel('time diff [s]')
    plt.ylabel('time [s]')

def plot_trajectory(y, y_hat):
    # y = 3d coordinates [n x 3]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(y[:, 0], y[:, 1], y[:, 2])
    ax.plot(y_hat[:, 0], y_hat[:, 1], y_hat[:, 2])

    ax.set_xlabel('x [left right]')
    ax.set_ylabel('y [up down]')
    ax.set_zlabel('z [depth]')
    
    return ax

def check_distance(xyz, targets):

    xyz = xyz[targets[:, 0].astype(int), :]

    d = np.sqrt(((xyz-targets[:, 1:])**2).sum(axis=1))

    plt.figure()
    plt.hist(d, bins=20)
    plt.title('mm to target')
    plt.xlabel('mm')
    plt.ylabel('count')
