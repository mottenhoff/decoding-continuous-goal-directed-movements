import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np

from libs.load_scores import get_scores
from libs.utils import load_yaml

POS_X, POS_Y, POS_Z = 0, 1, 2
VEL_X, VEL_Y, VEL_Z = 3, 4, 5
ACC_X, ACC_Y, ACC_Z = 6, 7, 8
DIST, SPEED, FORCE  = 9, 10, 11
Z_NAMES = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'dist', 'speed', 'force']

NAME = 0
VALUE = 1

MEAN = 0
STD = 1

CC = 0
R2 = 1
MSE = 2
RMSE = 3

LABEL_FONTSIZE = 'xx-large'
FIG = 0
LABEL_CBAR = 1

logger = logging.getLogger(__name__)

def load(path):
    # TODO: Again, copy pasted from figures_1d_score_overview.py
    m =  np.load(path/'metrics.npy')  # CC, R2, MSE, RMSE
    z =  np.load(path/'z.npy')
    y =  np.load(path/'y.npy')
    zh = np.load(path/'trajectories.npy')
    yh = np.load(path/'neural_reconstructions.npy')    
    xh = np.load(path/'latent_states.npy')

    return m, z, y, zh, yh, xh

def z_reconstruction(ax, z, zh, m, zoomed=False):
    
    n_samples = z.shape[0]

    ax.plot(z, color='k', label='z-true')
    ax.plot(zh, color='orange', label='z-pred')

    ax.set_title(f'Z reconstruction | CC={m[:, CC].mean():.2f} \u00b1 {m[:, CC].std():.2f}', fontsize=LABEL_FONTSIZE)

    ax.set_xticks(np.arange(0, n_samples, 1000))
    ax.set_ylabel(f'Speed', fontsize=LABEL_FONTSIZE)


    if zoomed:
        ax.set_xlim(1000, 5000)
        ax.set_xticks([])
        ax.get_xaxis().set_visible(False)
        ax.set_ylim(-2, 30)
        ax.tick_params('x', labelcolor='w')

    else:
        ax.set_xlabel(f'Time [windows]')
        ax.set_xlim(0, n_samples)
    

    # ax.set_xticklabels(np.arange(1000, 2000, 200))
    ax.set_xlabel(f'\u2192 Time', fontsize=LABEL_FONTSIZE, loc='right', labelpad=-20)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.legend()

    return ax

def performance_landscape(ax, x, y, z):

    mesh_x, mesh_y = np.meshgrid(x[VALUE], y[VALUE])
    surface = ax.plot_surface(mesh_x, mesh_y, z[VALUE], cmap=cm.plasma)

    ax.set_xlabel(x[NAME])
    ax.set_ylabel(y[NAME])
    ax.set_title(z[NAME], fontsize=LABEL_FONTSIZE)
    # ax.set_zlabel(z[NAME])
    ax.set_xticks(x[VALUE])
    ax.set_yticks(y[VALUE])
    ax.set_xlim(0, max(x[VALUE]))
    ax.set_ylim(0, max(y[VALUE]))

    return ax

def latent_states(ax, xh, zoomed=False):
    if zoomed:
        total_states = xh.shape[1]
        # xh = xh[:, :5]
        xh = np.hstack([xh[:, :4], xh[:, -1:]])

    scale = 8
    n_states = xh.shape[1]
    states = np.arange(n_states)

    for i, state in enumerate(xh.T):

        if zoomed and i == 3:
            ax.plot([], label=f'x{i}', color='k')
            continue

        ax.plot(state-i*scale, label=f'x{i}', color='k')

    ax.set_yticks(-np.arange(n_states)*scale)
    ax.set_ylabel('Latent states', fontsize=LABEL_FONTSIZE)

    if zoomed:
        ax.set_xlim(1000, 2000)
        ax.set_xticks([])
        ax.get_xaxis().set_visible(False)
        ax.tick_params('x', labelcolor='w')
        # ax.set_ylim(-8, 0)
        ax.set_yticklabels(['x1', 'x2', 'x3', '...', f'x{total_states}'], fontsize='xx-large')
    else:
        ax.set_xlabel('Time [windows]', fontsize=LABEL_FONTSIZE)

        ax.set_yticklabels([f"x{s}" for s in states], fontsize=LABEL_FONTSIZE)
        ax.set_xlim(0, xh.shape[0])

    ax.set_ylim(-n_states*scale-scale, scale)
    




    # ax.set_xlim(1000, 2000)
    # ax.set_xticks([])

    ax.set_xlabel(f'\u2192 Time', fontsize=LABEL_FONTSIZE, loc='right', labelpad=0)


    # ax.set_xticks(np.arange(0, xh.shape[0], 1000))

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return ax

def state_dynamics(ax, xa, xb, s, cbar=None):

    x1, x2 = xa[VALUE], xb[VALUE]
    s = s.squeeze()

    if any(s <= 0):
        v = 0.1
        logger.warning(f'Value <= 0 encountered! These value are changed to {v} for log scaling')
        s[s<=0] = v
    
    s = np.log(s)

    points = np.array([x1, x2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(s.min(), s.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(s)
    lc.set_linewidth(1)
    line = ax.add_collection(lc)

    if cbar != None:
        colorbar = cbar[FIG].colorbar(line, ax=ax)
        colorbar.set_label(cbar[LABEL_CBAR], rotation=270)

    ax.set_xlim(x1.min()-np.abs(x1.min()*0.05), x1.max()+x1.max()*0.05)
    ax.set_ylim(x2.min()-np.abs(x2.min()*0.05), x2.max()+x2.max()*0.05)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_xlabel(xa[NAME], fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(xb[NAME], fontsize=LABEL_FONTSIZE)

    return ax

def make(path):
    zoomed = False

    save_path = Path(r'./figure_output/reconstructions')  
    save_path.mkdir(exist_ok=True, parents=True)

    ppt_id = path.parts[-3]


    c = load_yaml(path/'config.yml')

    i  = ('Horizons',      c.learn.psid.i)
    n1 = ('n states [n1]', c.learn.psid.n1)
    nx = ('n states [nx]', c.learn.psid.nx)
    nx = n1 if not nx else nx

    metrics, zs, y, zhs, yh, xh = load(path)  # TODO: Auto select folder with highest performance 

    for dim in [DIST, SPEED, FORCE]:
        m = metrics[:, :, :, dim]
        z, zh = zs[:, dim], zhs.squeeze()[:, dim]

        fig = plt.figure(figsize=(20, 12))
        grid = GridSpec(3, 5)

        z_reconstruction(fig.add_subplot(grid[:2, :3]), z.squeeze(), zh.squeeze(), m.squeeze(), zoomed=zoomed)

        # performance_landscape(fig.add_subplot(grid[0, 3], projection='3d'), n1, i, ('CC',   scores[0, :, :, 0, 0]))
        # performance_landscape(fig.add_subplot(grid[0, 4], projection='3d'), n1, i, ('R2',   scores[0, :, :, 1, 0]))
        # performance_landscape(fig.add_subplot(grid[1, 3], projection='3d'), n1, i, ('MSE',  scores[0, :, :, 2, 0]))
        # performance_landscape(fig.add_subplot(grid[1, 4], projection='3d'), n1, i, ('RMSE', scores[0, :, :, 3, 0]))

        latent_states(fig.add_subplot(grid[2, :3]), xh.squeeze(), zoomed=zoomed)

        # state_dynamics(fig.add_subplot(grid[2, 3]), ('x1', xh.squeeze()[:, 0]), ('x2', xh.squeeze()[:, 1]), z)
        # state_dynamics(fig.add_subplot(grid[2, 4]), ('x2', xh.squeeze()[:, 1]), ('x3', xh.squeeze()[:, 2]), z, cbar=(fig, 'log(speed)'))

        # Selected locations + explained variance or sth? -> 3D plot with highlighted electrodes
        # Some explaining overview of latent states?

        # fig.tight_layout()
        # fig.savefig(path/'speed_reconstruction.svg')
        # fig.savefig(path/'speed_reconstruction.png')
        fig.savefig(save_path/f'{ppt_id}_{Z_NAMES[dim]}_reconstruction.png')

    # fig.savefig('speed_reconstruction.svg')
    # fig.savefig('speed_reconstruction.png')