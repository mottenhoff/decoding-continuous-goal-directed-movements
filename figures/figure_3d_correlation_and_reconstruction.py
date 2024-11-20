from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec


try:
    import libs.utils
    conf = libs.utils.load_yaml('./config.yml')
except Exception:
    import utils
    conf = utils.load_yaml('./config.yml')

DIMS = np.array(['X', 'Y', 'Z'])
COLORS = dict(zip(DIMS, ['r', 'g', 'b']))

TOP_HEIGHT = 1
MIDDLE_HEIGHT = 6
BOTTOM_HEIGHT = 3

def load(path):
    m =  np.load(path/'metrics.npy')
    z =  np.load(path/'z.npy')
    # y =  np.load(path/'neural_activity.npy')
    y =  np.load(path/'y.npy')
    zh = np.load(path/'trajectories.npy')
    yh = np.load(path/'neural_reconstructions.npy')    
    xh = np.load(path/'latent_states.npy')

    return m, z, y, zh, yh, xh

def overview_top(fig, grid, m):
    mean = m.mean(axis=1)
    std = m.std(axis=1)

    for i, score in enumerate(['CC', 'R2', 'MSE', 'RMSE']):
        ax = fig.add_subplot(grid[:TOP_HEIGHT, i])

        xticks = np.arange(mean.shape[2])

        ax.errorbar(x=xticks, y=mean[0, i, :], yerr=std[0, i, :],
                    fmt = 'o', markersize=2, capsize=2, color='r' )  # TODO: Color xyz differently
        ax.axhline(0, linestyle='--', c='k', linewidth=1)

        # TODO:
        # ax.axhline(chance_level)

        if score in ['CC', 'R2']:
            y_min, y_max = -1.1, 1.1

        else:
            y_min, y_max = 0, None
        
        ax.set_xlim(-1, 3)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(xticks)
        ax.set_xticklabels(DIMS[xticks])
        ax.set_title(f'{score} | {mean[0, i, :].mean():.1f}\u00B1{mean[0, i, :].std():.1f}')

        y_min = -1.1 if score in ['CC', 'R2'] else 0
        y_max =  1.1 if score in ['CC', 'R2'] else None
        ax.set_ylim(y_min, y_max)

    return fig, grid

def overview_middle(fig, grid, m, z, zh):

    # ax3d = fig.add_subplot(grid[1:7, :], projection='3d')
    ax = fig.add_subplot(grid[TOP_HEIGHT:TOP_HEIGHT+MIDDLE_HEIGHT, :], projection='3d')

    ax.set_title('Trajectory [Highest total correlation]')
    
    z_hat = zh.squeeze()  # Might change when multiple options used
    ax.plot(z_hat[:, 0], z_hat[:, 1], z_hat[:, 2], label='Predicted')
    ax.plot(z[:, 0], z[:, 1], z[:, 2], label='True')

    ax.set_xlabel('X [left - right]')
    ax.set_ylabel('Y [up - down]')
    ax.set_zlabel('Z [front - back]')
    ax.legend()

    return fig, grid

def overview_bottom(fig, grid, z, zh):
    

    z_hat = zh.squeeze()
    if z_hat.ndim == 1:
        z_hat = z_hat[:, np.newaxis]

    for i, dim in enumerate(DIMS[:z.shape[1]]):
        ax = fig.add_subplot(grid[TOP_HEIGHT+MIDDLE_HEIGHT+i, :])
        ax.plot(z_hat[:, i])
        ax.plot(z[:, i])
        ax.set_title(dim)
        ax.set_xlabel('time [windows]')
        ax.set_ylabel('mm')

    return fig, grid

def overview(m, z, y, zh, yh, xh, path):

    fig = plt.figure(figsize=(10, 15))
    fig.suptitle('Results')
    grid = GridSpec(TOP_HEIGHT+MIDDLE_HEIGHT+BOTTOM_HEIGHT, 4)

    fig, grid = overview_top(fig, grid, m)
    if z.shape[1] > 1:
        fig, grid = overview_middle(fig, grid, m, z, zh)
    fig, grid = overview_bottom(fig, grid, z, zh)

    plt.tight_layout()

    fig.savefig(path/'fig_1_correlation_and_reconstructed_signal.png')
    
    return fig, grid

def make(path):
    c, z, y, zh, yh, xh = load(path)
    overview(c, z, y, zh, yh, xh, path=path)

if __name__=='__main__':
    date = None  # If none, get most recent result
    date = '20221209_0257/3_3_5'

    if not date:
        path = sorted([p for p in Path(conf.save_path).iterdir() \
                       if p.is_dir()])[-1]   
    else:
        path = Path(f"{conf.save_path}/{date}")
    
    make(path)

    plt.show()
