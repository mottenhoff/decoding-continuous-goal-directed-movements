from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm


FREQS, PPTS, STATES, FOLDS, METRICS, KINEMATICS = np.arange(6)

# Set some information
CONFIG_I =        (5, 10, 25, 50, 100)  
CONFIG_N1 =       (3, 5, 10, 20, 30)
KINEMATIC_ORDER = (0, 1, 2, 9, 3, 4, 5, 10, 6, 7, 8, 11)    # x, y, z, sum. For 4x12 plot

SCORE_NAMES =     ['freqs', 'ppts', 'states', 'folds', 'metrics', 'kinematics']
CONDITIONS =      ['delta', 'alphabeta', 'bbhg']
CONDITION_NAMES = ['Delta activity', 'Beta', 'Broadband high-gamma']
STATE_OPTIONS =   list(product(CONFIG_N1, CONFIG_I))

KINEMATIC_NAMES = [
          r'$r_x$',      r'$r_y$',      r'$r_z$',
          r'$v_x$',      r'$v_y$',      r'$v_z$',
          r'$\alpha_x$', r'$\alpha_y$', r'$\alpha_z$',
          r'$\vec r$',   r'$\vec v$',   r'$\vec \alpha$']
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
                        alpha=0.7,
                        **{'label': 'chance level'} if label and ci==0 else {})
    return ax

def stack_scores(data, key=None):
    stack_ppt_scores = lambda ppts: np.stack([values['scores' if not key else key] for values in ppts.values()])
    return np.stack([stack_ppt_scores(ppts) for _, ppts in data.items()], axis=0)

def plot_freqs_vs_states(scores):

    xticklabels = [f'{n1}_{i}' for n1, i in STATE_OPTIONS]

    nrows, ncols = 3, 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 9))

    for i_ax, ax in enumerate(axs.flatten()):

        for i_cond, condition in enumerate(CONDITIONS):
            mean = scores.mean(axis=(PPTS, FOLDS))[i_cond, :, 0, KINEMATIC_ORDER[i_ax]]

            ax.plot(mean, label=condition if i_ax==0 else '')
                            
            ax.set_ylim(0, 1)
            ax.spines[['top', 'right']].set_visible(False)

            if i_ax >= 8:
                ax.set_xticks(np.arange(len(STATE_OPTIONS)))
                ax.set_xticklabels(xticklabels, rotation=90)    

    fig.legend(frameon=False)

    fig.tight_layout()
    fig.savefig('./figure_output/overview_freqs_vs_states.png')
    fig.savefig('./figure_output/overview_freqs_vs_states.svg')

    return fig, ax

def plot_mean_performance(scores, chance_levels, name):

    jitter = lambda n: (np.arange(n) - n/2) + 0.5
    if scores.ndim == 4:
        scores = scores[np.newaxis, :, :, :, :]

    mean = scores.mean(axis=(1, 2))[:, 0, :]
    std =  scores.std( axis=(1, 2))[:, 0, :]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    bar_width = 0.20
    jitter_idc = jitter(mean.shape[0]) * bar_width
    colors = [CMAP(0), CMAP(0.33), CMAP(0.66)]
    ax.set_xlim(-1, 13)
    for freqs_i, freqs in enumerate(mean):
    
        x_ticks = np.arange(mean.shape[1]) + jitter_idc[freqs_i]
        
        ax.bar(x_ticks, freqs, yerr=np.stack([(0, f) for f in std[freqs_i, :]]).T,
               width=bar_width, label=CONDITION_NAMES[freqs_i],
               color=colors[freqs_i])
        
        ax = hline_per_bar(ax, x_ticks, chance_levels[freqs_i].max(axis=0), 
                           label=True if freqs_i==0 else False)
    
    ax.set_axisbelow(True)
    ax.yaxis.grid()

    ax.set_xticks(np.arange(mean.shape[1]))
    ax.set_xticklabels(KINEMATIC_NAMES, fontsize='xx-large')

    ax.set_ylim(0, 1)
    ax.set_ylabel('CC', fontsize='xx-large')
    ax.spines[['top', 'right']].set_visible(False)

    fig.legend(frameon=False) 
    fig.savefig(f'./figure_output/mean_performance_per_band_per_kinematic{"_"+name}.png')
    fig.savefig(f'./figure_output/mean_performance_per_band_per_kinematic{"_"+name}_.svg')

def plot(results, name):
    # results = [(best_paths, scores), ...]  delta, ab, bbhg
    # Gather the data
    scores =        stack_scores(results).squeeze()
    chance_levels = stack_scores(results, 'chance_levels_prediction')

    plot_mean_performance(scores, chance_levels, name)