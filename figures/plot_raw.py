import matplotlib.pyplot as plt
import numpy as np

from libs.load.load_yaml import load_yaml

c = load_yaml('./config.yml')

def raw_eeg(session, savepath):
    # Actually a session
    # including markers --> color by direction?

    n_channels = session.eeg.shape[1]
    n_rows = 20

    idc_per_col = np.array_split(np.arange(n_channels), n_channels/n_rows)
    max_rows = max([len(idc) for idc in idc_per_col])

    fig, ax = plt.subplots(nrows=max_rows, ncols=len(idc_per_col), 
                           dpi=200, figsize=(24, 16))
    
    for col_i, ax_col in enumerate(idc_per_col):
        for row_i, ch_num in enumerate(ax_col):
            ax[row_i, col_i].plot(session.eeg[:, ch_num], linewidth=1, color='k')
            ax[row_i, col_i].spines[['top', 'right', 'left', 'bottom']].set_visible(False)
            ax[row_i, col_i].tick_params(axis='x', which='both', 
                                         bottom=False, labelbottom=False)
            ax[row_i, col_i].tick_params(axis='y', which='both',
                                         left=False, labelleft=False)
            ax[row_i, col_i].set_ylabel(session.channels[ch_num])
            ax[row_i, col_i].set_ylim(-1000, 1000)

    fig.suptitle('Raw signal')
    fig.tight_layout()
    fig.savefig('./figure_output/raw_eeg.svg')


def plot_task_correlation(session, savepath):
    eeg = session.eeg
    labels = session.trial_names

    classes = np.unique(labels)
    n_classes = len(classes)
    labels = np.vstack([labels==class_ for class_ in classes]).T
    cm = np.corrcoef(np.hstack((eeg, labels)), rowvar=False)
    
    task_correlation = cm[:-n_classes, -n_classes:]

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           dpi=200)
    im = ax.imshow(task_correlation.T, vmin=-1, vmax=1)
    fig.savefig('raw_task_correlation.png')

def make_all(session, savepath):
    raw_eeg(session, savepath)
    # plot_average_trial_per_class(session, savepath)
    # plot_task_correlation(session, savepath)