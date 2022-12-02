from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

from libs import utils

c = utils.load_yaml('./config.yml')

    # from sklearn.decomposition import FastICA

    # ica = FastICA(n_components=10)
    # cs = ica.fit_transform(eeg)

    # fig, ax = plt.subplots(nrows=cs.shape[1], ncols=1, figsize=(12, 16), dpi=200)
    # for i, row in enumerate(cs.T):
    #     ax[i].plot(row)
    
    # fig.savefig('ica.svg')

def plot_eeg(eeg, channel_names, name):
    # including markers


    n_channels = eeg.shape[1]
    n_rows = 20

    idc_per_col = np.array_split(np.arange(n_channels), n_channels/n_rows)
    max_rows = max([len(idc) for idc in idc_per_col])

    fig, ax = plt.subplots(nrows=max_rows, ncols=len(idc_per_col), 
                           dpi=200, figsize=(24, 16))
    
    for col_i, ax_col in enumerate(idc_per_col):
        for row_i, ch_num in enumerate(ax_col):
            ax[row_i, col_i].plot(eeg[:, ch_num], linewidth=1, color='k')
            ax[row_i, col_i].spines[['top', 'right', 'left', 'bottom']].set_visible(False)
            ax[row_i, col_i].tick_params(axis='x', which='both', 
                                         bottom=False, labelbottom=False)
            ax[row_i, col_i].tick_params(axis='y', which='both',
                                         left=False, labelleft=False)
            ax[row_i, col_i].set_ylabel(channel_names[ch_num])
            ax[row_i, col_i].set_ylim(-1000, 1000)

    fig.suptitle(f'Eeg | {name}')
    fig.tight_layout()
    fig.savefig(f'./figures/checks/eeg_{name}_car.svg')

def plot_xyz(xyz):
    idc = np.where(~np.isnan(xyz[:, 0]))[0]
    plt.figure(dpi=300, figsize=(12, 8))
    plt.scatter(idc, xyz[idc, 0], s=3, c='orange')
    plt.plot(idc, xyz[idc, 0], label='pos_x')
    plt.scatter(idc, xyz[idc, -1], s=3, c='green')
    plt.plot(idc, xyz[idc, -1], c='brown', label='speed')
    plt.legend()
    plt.savefig('./figures/checks/xyz_first_and_last_column.svg')

def plot_target_vector(xyz, trials):
    # 2D view

    xyz = xyz[~np.all(np.isnan(xyz), axis=1)]
    trials = trials[~np.all(np.isnan(trials[:, 1:]), axis=1), :]

    target_vec = trials - xyz

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.plot(xyz[:, 0], xyz[:, 1], c='grey', label='trajectory')
    ax.scatter(trials[:, 0], trials[:, 1], c='red', s=10, label='goals')

    for idx in np.arange(0, target_vec.shape[0], 100):
        ax.arrow(xyz[idx, 0], xyz[idx, 1],
                 target_vec[idx, 0], 
                 target_vec[idx, 1],
                 color='orange', 
                 label='target_vector' if idx==0 else None)
    ax.set_title('Target_vector')
    ax.set_xlabel('x-pos')
    ax.set_ylabel('y_pos')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.legend()

    fig.savefig('./figures/checks/target_vector.png')

def plot_gap_cuts(xyz, idc, subset_idc):
    # Green = Start of gap
    # Red = End of gap
    plt.figure()
    plt.plot(idc, xyz[idc, -1])
    ylim_max = plt.ylim()[1]
    for si, ei in subset_idc:
        plt.vlines(si, ymin=0, ymax=ylim_max, colors='g', linewidth=1, linestyles='--')
        plt.vlines(ei, ymin=0, ymax=ylim_max, colors='r', linewidth=1, linestyles='--')
    plt.savefig(f'./figures/checks/gap_cut_idc_{si}_{ei}.svg')

def plot_events(xyz, events, trials, markers):
    cmap = get_cmap('tab20')

    ts = xyz['ts'] - xyz['ts'][0]
    xyz = xyz['data'][:, 7:10]

    fig, ax = plt.subplots(nrows=xyz.shape[1], ncols=1,
                           figsize=(16, 12), dpi=200)
    for i in range(xyz.shape[1]):
        ax[i].plot(ts, xyz[:, i], color='k')
        for k, field in enumerate(fields(events)):

            if k not in [2]:
            # if k!=0:
                continue

            for j, row in enumerate(getattr(events, field.name)):
                t = ts[row[0].astype(int)]

                ax[i].axvline(t, linewidth='1', linestyle='--', 
                                 label=field.name if j==0 else None,
                                 color=cmap(k))

                if row.ndim > 1:
                    ax[i].scatter(t, row[i], s=1)
    
    ax[0].axhline(0, color='grey', linestyle='dotted', label='middle')
    ax[1].axhline(250, color='grey', linestyle='dotted')
    ax[2].axhline(0, color='grey', linestyle='dotted')
    ax[0].legend(bbox_to_anchor=(1, 1))
    
    fig.savefig('./figures/checks/events.png')

    return