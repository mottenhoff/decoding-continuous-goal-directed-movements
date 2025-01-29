from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

FREQS, PPTS, STATES, FOLDS, METRICS, KINEMATICS = np.arange(6)

# Set some information
CONFIG_I =        (5, 10, 25, 50, 100)
CONFIG_N1 =       (3, 5, 10, 20, 30)
KINEMATIC_ORDER = (0, 1, 2, 9, 3, 4, 5, 10, 6, 7, 8, 11)    # x, y, z, sum. For 4x12 plot
CC, R2, MSE, RMSE = np.arange(4)

SCORE_NAMES =     ['freqs', 'ppts', 'states', 'folds', 'metrics', 'kinematics']
CONDITIONS =      ['delta', 'alphabeta', 'bbhg']
CONDITION_NAMES = ['Delta activity', 'Beta', 'Broadband high-gamma']
STATE_OPTIONS =   list(product(CONFIG_N1, CONFIG_I))

KINEMATIC_NAMES = [
          r'$r_x$',      r'$r_y$',      r'$r_z$',
          r'$v_x$',      r'$v_y$',      r'$v_z$',
          r'$\alpha_x$', r'$\alpha_y$', r'$\alpha_z$',
          r'$\vec r$',   r'$\vec v$',   r'$\vec \alpha$'
]

CMAP = cm.batlow

def hline_per_bar(ax, x_ticks, chance_levels, label=True):
    # NOTE: If plotted iteratively (i.e. this functions is called multiple times), 
    # make sure the xlims of the plot stay the same.

    # Chance levels: array of length bars in ax.

    abs_to_rel = lambda x, lim: (x - lim[0]) / (lim[1] - lim[0]) 
    
    xlim = ax.get_xlim()

    for ci, (tick, bar_patch) in enumerate(zip(x_ticks, ax.patches)):

        half_width = abs_to_rel(xlim[0]+bar_patch.get_width()/2, xlim)
        anchor = abs_to_rel(tick, xlim)
        xmin, xmax = anchor - half_width, anchor + half_width
        
        ax.axhline(chance_levels[ci], xmin=xmin, xmax=xmax, 
                        color='black', 
                        # linestyle='--', 
                        alpha=0.7,
                        **{'label': 'chance level'} if label and ci==0 else {})
    return ax

def plot_overview(results, condition):

    # Set some values
    col_titles = ['X', 'Y', 'Z', r'$\sum$']
    row_titles = ['Position', 'Velocity', 'Acceleration']

    metric = CC  # options: [CC, R2, MSE, RMSE]

    scores = {ppt: values['scores'] for ppt, values in results.items()}
    chance_levels = {ppt: values['chance_levels_prediction'] for ppt, values in results.items()}

    # Guarantee order
    ppts = sorted(chance_levels.keys())
    scores = np.stack([scores[ppt] for ppt in ppts]).squeeze()
    chance_levels = np.stack([chance_levels[ppt] for ppt in ppts])

    xticks = np.arange(len(ppts))
    colors = [CMAP(int(i)) for i in np.linspace(0, 255, len(ppts))]

    fig_shape = (3, 4)
    fig, axs = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1], figsize=(16, 9))

    for idx, ax in enumerate(axs.flatten()):
        
        mean = scores[:, :, metric, KINEMATIC_ORDER[idx]].mean(axis=1)
        std  = scores[:, :, metric, KINEMATIC_ORDER[idx]].std(axis=1)

        ax.bar(xticks, mean, yerr=np.vstack([np.zeros(std.size), std]), color=colors)  # np.stack([(0, f) for f in std[freqs_i, :]]).T for only top errorbar
        
        ax = hline_per_bar(ax, xticks, chance_levels[:, KINEMATIC_ORDER[idx]])

        ax.spines[['top', 'left', 'right']].set_visible(False)
        
        ax.set_xticks(np.arange(len(ppts)))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)

        ax.set_axisbelow(True)
        ax.yaxis.grid()

    
    for i in range(3):
        axs[i, 0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[i, 0].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[i, 0].set_ylabel(f'{row_titles[i]}\nCC', fontsize='x-large')
        axs[i, 0].spines['left'].set_visible(True)

    for i in range(4):
        axs[0, i].set_title(col_titles[i], fontsize='xx-large')

        axs[-1, i].set_xticks(np.arange(scores.shape[0]))
        axs[-1, i].set_xticklabels([ppt.split('_')[0].capitalize() for ppt in ppts], fontsize='small', rotation=45, ha='right')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)

    fig.savefig(f'figure_output/decoder_scores_{condition}.png')
    fig.savefig(f'figure_output/decoder_scores_{condition}.svg')

    plt.close('all')
