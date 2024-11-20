'''
Inspiration:
    https://matplotlib.org/stable/gallery/animation/multiple_axes.html#sphx-glr-gallery-animation-multiple-axes-py
'''

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

METRICS = 0
Z = 1
Y = 2
ZH = 3
YH = 4
XH = 5

CC = 0

LABEL_FONTSIZE = 'xx-large'

X_AXIS_MIN = 0
X_AXIS_MAX = 1000


def load_hand_trajectory(path):
    raise NotImplementedError

def load_brain_activations(path):
    raise NotImplementedError

def load_psid_data(path):
    m =  np.load(path/'metrics.npy')  # CC, R2, MSE, RMSE
    z =  np.load(path/'z.npy')
    y =  np.load(path/'y.npy')
    zh = np.load(path/'trajectories.npy')
    yh = np.load(path/'neural_reconstructions.npy')    
    xh = np.load(path/'latent_states.npy')

    return m, z, y, zh, yh, xh

def brain_activations(ax, data, frame):
    return ax

def hand_trajectory(ax, data, frame):

    z = data

    # ax.set_title('Trajectory [Highest total correlation]')
    
    ax.plot(z[:, 0], z[:, 1], z[:, 2], color='b', label='True')

    # ax.set_xlabel('X [left - right]')
    # ax.set_ylabel('Y [up - down]')
    # ax.set_zlabel('Z [front - back]')

    ax.set_xlim(z.min(axis=0)[0], z.max(axis=0)[0])
    ax.set_ylim(z.min(axis=0)[1], z.max(axis=0)[1])
    ax.set_zlim(z.min(axis=0)[2], z.max(axis=0)[2])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_axis_off()
    ax.grid(False)
    # 3d plot of hand trajectory
    # hand as 'cursor'

    return ax

def speed_reconstruction(ax, data, frame):
    xlim = 0, data[Z].shape[0]
    ylim = min(data[Z].min(), data[ZH].min()), max(data[Z].max(), data[ZH].max())

    z, zh, m = data[Z][:frame], data[ZH].squeeze()[:frame], data[METRICS]


    n_samples = z.shape[0]

    for line in ax.lines:
        ax.lines.remove(line)

    ax.plot(z, color='k', label='z-true')
    ax.plot(zh, color='orange', label='z-pred')
    
    if frame == 0:
        # ax.set_title(f'Z reconstruction | CC={m[:, CC].mean():.2f} \u00b1 {m[:, CC].std():.2f}', fontsize=LABEL_FONTSIZE)
        # ax.set_xlabel(f'Time [windows]')
        ax.set_xticks(np.arange(0, n_samples, 1000))
        ax.set_ylabel(f'Speed', fontsize=LABEL_FONTSIZE)
       
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
       
        # ax.set_xlim(0, n_samples)
        

        # ax.set_xlim(1000, 2000)
        # ax.get_xaxis().set_visible(False)
        ax.set_xticks(np.arange(0, xlim[1], 500))
        ax.set_xticklabels([])
        # ax.tick_params('x', labelcolor='w')

        ax.set_xlabel(f'\u2192 Time', fontsize=LABEL_FONTSIZE, loc='right', labelpad=-20)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # ax.legend()

    return ax

def latent_states(ax, data, frame):
    # TODO: GET MIN MAX VALUES HERE
    xlim = 0, data[XH].squeeze().shape[0]
    ylim = data[XH].min() - 3, data[XH].max()

    xh = data[XH].squeeze()[:frame, :]

    scale = 2
    n_states = xh.shape[1]
    states = np.arange(n_states)

    for line in ax.lines: 
        ax.lines.remove(line)

    for i, state in enumerate(xh.T):
        ax.plot(state-i*scale, label=f'x{i}', color='k')
    
    if frame == 0:
        # ax.set_xlabel('Time [windows]', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Latent states', fontsize=LABEL_FONTSIZE)
        ax.set_yticks(-np.arange(n_states)*scale)
        ax.set_yticklabels([f"x{s}" for s in states], fontsize=LABEL_FONTSIZE)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # ax.set_xlim(1000, 2000)
        ax.set_xticks([])
        # ax.set_xlabel(f'\u2192 Time', fontsize=LABEL_FONTSIZE, loc='right', labelpad=0)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    return ax

def make(path):
    # TODO: Set axes to max value already

    savepath = Path('./animations/')
    savepath.mkdir(exist_ok=True)

    # brain_activation_data = load_brain_activation_data(path)
    hand_trajectory_data = np.load('hand_trajectory_kh040.npy')
    psid_data = load_psid_data(path)

    fig = plt.figure(figsize=(16, 9))
    grid = GridSpec(2, 3)

    ax_hand_trajectory = fig.add_subplot(grid[0, 0], projection='3d')
    ax_latent_states =   fig.add_subplot(grid[0, 1:])
    ax_speed =           fig.add_subplot(grid[1, 1:])

    start = 0
    end = psid_data[Z].shape[0]
    stepsize = 100

    for frame in np.arange(start, end, stepsize):
        # brain_activations(fig.add_subplot(grid[0, 1]), brain_activation_data, frame=frame)
        hand_trajectory(ax_hand_trajectory, hand_trajectory_data, frame=frame)
        latent_states(ax_latent_states, psid_data, frame=frame)
        speed_reconstruction(ax_speed, psid_data, frame=frame)
        # plt.show()
        fig.savefig(savepath/f'frame_{frame:05d}.png')

        break