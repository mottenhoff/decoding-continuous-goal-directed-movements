import csv  # TEMP
import re

import logging
from dataclasses import fields
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from mne.stats import fdr_correction
from scipy.stats import pearsonr, spearmanr

from libs import utils
from libs import kinematics as kin

logger = logging.getLogger(__name__)
c = utils.load_yaml('./config.yml')

    # from sklearn.decomposition import FastICA

    # ica = FastICA(n_components=10)
    # cs = ica.fit_transform(eeg)

    # fig, ax = plt.subplots(nrows=cs.shape[1], ncols=1, figsize=(12, 16), dpi=200)
    # for i, row in enumerate(cs.T):
    #     ax[i].plot(row)
    
    # fig.savefig('ica.svg')

def load(path):
    m =  np.load(path/'metrics.npy')  # CC, R2, MSE, RMSE
    z =  np.load(path/'z.npy')
    y =  np.load(path/'y.npy')
    zh = np.load(path/'trajectories.npy')
    yh = np.load(path/'neural_reconstructions.npy')    
    xh = np.load(path/'latent_states.npy')

    return m, z, y, zh, yh, xh

def reset():
    path = Path(r'./figures/checks/')

    for f in path.glob('**/*'):
        if f.is_file():
            f.unlink(missing_ok=True)



def plot_eeg(eeg, channel_names, name, loc_map={}):
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
            channel_name = channel_names[ch_num]
            ax[row_i, col_i].set_title(loc_map.get(channel_name, ''), fontsize='small')
            ax[row_i, col_i].set_ylabel(channel_name)
            ax[row_i, col_i].set_ylim(-1000, 1000)

    fig.suptitle(f'Eeg | {name}')
    fig.tight_layout()
    fig.savefig(f'./figures/checks/eeg_{name}.svg')

def plot_xyz(xyz):
    idc = np.where(~np.isnan(xyz[:, 0]))[0]

    pos = xyz[:, [0, 1, 2]]
    vel = xyz[:, [3, 4, 5]]
    spe = xyz[:, 6]

    fig, ax = plt.subplots(4, 1, dpi=300, figsize=(12, 8))
    for j, k in enumerate([pos, vel]):
        ax[0].scatter(idc, k[:, 0], s=2, c='blue' if j==0 else 'orange')
        ax[0].plot(idc, k[:, 0], linestyle='--' if j==0 else '-',
                                   label='pos' if j==0 else 'vel', 
                                   c='blue' if j==0 else 'orange')
        ax[1].scatter(idc, k[:, 1], s=2, c='blue' if j==0 else 'orange')
        ax[1].plot(idc, k[:, 1], linestyle='--' if j==0 else '-',
                                   label='pos' if j==0 else 'vel', 
                                   c='blue' if j==0 else 'orange')
        ax[2].scatter(idc, k[:, 2], s=2, c='blue' if j==0 else 'orange')
        ax[2].plot(idc, k[:, 2], linestyle='--' if j==0 else '-',
                                   label='pos' if j==0 else 'vel', 
                                   c='blue' if j==0 else 'orange')
    ax[3].plot(idc, spe, label='speed')

    fig.legend()
    fig.savefig('./figures/checks/xyz_first_and_last_column.svg')

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

def plot_gap_cuts(xyz, idc, subset_idc, save_path=None):
    # Green = Start of gap
    # Red = End of gap
    plt.figure(figsize=(16,12))
    plt.plot(idc, xyz[idc, 0])
    ylim_max = plt.ylim()[1]
    for si, ei in subset_idc:
        plt.vlines(si, ymin=0, ymax=ylim_max, colors='g', linewidth=1, linestyles='--')
        plt.vlines(ei, ymin=0, ymax=ylim_max, colors='r', linewidth=1, linestyles='--')

    if save_path:
        path = save_path/'gap_cuts'
    else:
        path = f'./figures/checks/gap_cuts'

    plt.savefig(str(path) + '.svg')
    plt.savefig(str(path) + '.png')

def plot_events(dataset):

    xyz, events, trials = dataset.xyz, dataset.events, dataset.trials[:, 0]

    cmap = get_cmap('tab20')

    ts = xyz['ts'] - xyz['ts'][0]
    xyz = xyz['data'][:, 7:10]

    fig, ax = plt.subplots(nrows=xyz.shape[1], ncols=1,
                           figsize=(16, 12), dpi=200)
    for i in range(xyz.shape[1]):
        ax[i].plot(ts, xyz[:, i], color='k')
        for k, field in enumerate(fields(events)):

            if field.name in ['off_target']:
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

def plot_datasets(datasets_o, target_kinematics_o):
    datasets = datasets_o.copy()
    target_kinematics = target_kinematics_o.copy()
    if not datasets:
        logger.warning('Empty lists of datasets, returning...')
        return

    channels = datasets[0].channels
    mapping = datasets[0].mapping

    logger.info(f'Correlations per subset')
    for ds in datasets:
        eeg, xyz = ds.eeg, ds.xyz[:, -1][:, np.newaxis]
        data = np.hstack((eeg, xyz))
        cc = np.corrcoef(data.T)
        cc_xyz = cc[:-1, -1]
        cc_xyz = np.sort(cc_xyz)
        logger.info(f'{eeg.shape[0]:>5} {cc_xyz[:3]} {cc_xyz[-3:]}')
    logger.info('')

    y = np.vstack([s.eeg for s in datasets])
    z = np.vstack([s.xyz[:, target_kinematics] for s in datasets])

    cc = np.vstack([(pearsonr(ch, z).statistic[0], pearsonr(ch, z).pvalue) for ch in y.T])
    logger.info(f'{y.shape[0]:>5} {np.sort(cc[:, 0])[:3]} {np.sort(cc[:, 0])[-3:]}')

    cc[:, 1] = fdr_correction(cc[:, 1])[1]
    
    alpha = 0.05
    is_sig = np.where(cc[:, 1] < alpha)[0]
    sig_chs = channels[is_sig]
    sig_vals = cc[:, 1][is_sig]
    sig_cc = cc[:, 0][is_sig]

    sorted_idc = np.argsort(sig_vals)
    sig_chs, sig_vals, sig_cc = sig_chs[sorted_idc], sig_vals[sorted_idc], sig_cc[sorted_idc]
    logger.info('')
    logger.info(f'Significant channels [alpha={alpha}, FDR]')
    logger.info(f'---------------------------------')
    for ch, corr, pval in zip(sig_chs, sig_cc, sig_vals):
        logger.info(f'{ch}_{mapping.get(ch, ""):<35}: cc={corr:<6.3f} [[pval= {pval:<7.4f}]')

    fig, ax = plt.subplots(5, 1, figsize=(12, 8))
    for i, (ch, corr, pval) in enumerate(zip(sig_chs, sig_cc, sig_vals)):
        ax[i].plot(y[:, channels==ch], label='y')
        ax[i].plot(z[:, -1], label='z')
        ax[i].set_title(f"{ch} | cc = {corr:.3f} [p={pval:.3f}]")

        if i==4:
            ax[i].set_xlabel(f'Windows [{c.window.length} + {c.window.shift} ms step]')
            ax[i].set_ylabel(f'Value')
            ax[i].legend()
            break

    fig.tight_layout()
    fig.savefig('figures/checks/dataset_correlations.svg')

def load_locations(path):
    if not path.exists():
        return {}

    with open(path) as f:
        reader = csv.DictReader(f)
        data = {row['electrode_name_1']: row['location'] for row in reader}

    return data


def check_used_data(path):
    ppt = 'kh042'
    mapping = load_locations(Path(f"/home/coder/project/data/{ppt}/electrode_locations.csv"))
    ch_names = list(mapping.keys())
    
    lst = "[ 42  47  45  96  14  81  95  59  60  48   5  44  27  26  40  25 100  11 43 101 103 102  49 104 110 105 109 106 108 107]"
    # lst = "[49 43 40 44 48 45 46 47 25 42 95 81 0 80 41 20 1 13 24 79 99 77 19 51 21 36 78 50 94 15]"
    
    selected_chs = [int(ch) for ch in re.findall('[0-9]+', lst)]
    # selected_chs = ast.literal_eval(lst.replace(' ', ','))

    path = Path("/home/coder/project/results/20221209_0217/3_3_10")
    m, z, y, zh, yh, xh = load(path)
    yh = yh.squeeze()
    
    fig, ax = plt.subplots(nrows=len(selected_chs), ncols=1, figsize=(12, 20))
    for i, ch in enumerate(selected_chs):
        if i == 0:
            ax[i].plot(y[:, ch], label='True')
            ax[i].plot(yh[:, i], label='Reconstructed')
        else:
            ax[i].plot(y[:, ch])
            ax[i].plot(yh[:, i])
            
        ax[i].set_title(f'{ch_names[ch]}_{mapping[ch_names[ch]]}')

    fig.legend()
    fig.suptitle('Reconstructed Neural data of top N selected channels')
    # fig.tight_layout()
    fig.savefig(f'{path}/Reconstructed_neural_data.png')

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(10, 15))
    for i, ch in enumerate(selected_chs[:5]):
        if i==0:
            ax[i].plot(y[:, ch], label='Y-true')
            ax[i].plot(z, label='Z-true')
        else: 
            ax[i].plot(y[:, ch])
            ax[i].plot(z)
        
        ax[i].set_title(f'{ch_names[ch]}_{mapping[ch_names[ch]]}')
    fig.legend()
    fig.suptitle('Neural data and Behavior used in learning. (top 5 selected)')
    fig.savefig(f'{path}/yt_zt.png')
    print('')


