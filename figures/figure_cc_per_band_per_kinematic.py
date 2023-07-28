import logging
from itertools import product
from pathlib import Path


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
# from pandas import DataFrame

from libs import utils
# from libs.kinematics import velocity, vector_length

logger = logging.getLogger(__name__)
c = utils.load_yaml('./config.yml')

def plot_band_correlations(eeg: np.ndarray,
                           xyz: np.ndarray,
                           name: str,
                           get_bands=True):
    ''' Plots correlation of each powerband with all kinematics
        Should not contain missing valuess
        xyz should hold position, velocity and speed in order
        
        eeg: np.array[samples x channels]
        xyz: np.array[samples x 7]
        name: kh_identifier
    '''

    # pos = DataFrame(pos).interpolate().to_numpy()
    # is_not_nan = np.where(~np.any(np.isnan(pos), axis=1))[0]
    # pos = pos[is_not_nan, :]
    # eeg = eeg[is_not_nan, :]
    bands = {'delta': [0, 4],
             'theta': [4, 8],
             'alpha': [8, 12],
             'beta':  [12, 30],
             'gamma': [30, 55],
             'gamma+': [55, 90],
             'gamma++': [90, 200]} \
             if get_bands else \
             {'raw': []}

    ax_names = list(product(['pos', 'vel', 'speed'], ['X', 'Y', 'Z']))

    n_channels = eeg.shape[1]

    fig = plt.figure(figsize=(12, 16))
    grid = GridSpec(4, 2, figure=fig)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[2, 0])

    ax3 = fig.add_subplot(grid[0, 1])
    ax4 = fig.add_subplot(grid[1, 1])
    ax5 = fig.add_subplot(grid[2, 1])
    
    ax6 = fig.add_subplot(grid[3, :])

    for bname, bfreqs in bands.items():
        if get_bands:
            b = utils.instantaneous_powerbands(eeg, 1024, {bname: bfreqs})
        else:
            b = eeg

        corr = np.corrcoef(np.hstack((b, xyz)).T)

        for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5]):
            ax.plot(corr[:n_channels, n_channels+i], label=bname)
            ax.set_title('_'.join(ax_names[i]))



        # for j, axs in enumerate([[ax0, ax1, ax2], [ax3, ax4, ax5]]):
            
        #     for i in range(3):
        #         axs[i].plot(corr[:eeg.shape[1], eeg.shape[1]+3*j+i], label=bname)
        #         axs[i].set_title(f'{kin_name[j]}_{ax_name[i-1]}')
        
        ax6.plot(corr[:-7, -1], label=bname)

    ax6.set_xlabel('Channels')
    ax6.set_ylabel('CC')
    ax6.set_title('Speed')


    ax6.legend(bbox_to_anchor=(1, 1))
    fig.suptitle('CC per band')
    # plt.tight_layout()
    fig.savefig(f"figures/checks/cc_per_band_{name}_{'raw' if not get_bands else ''}.svg")
    return
    