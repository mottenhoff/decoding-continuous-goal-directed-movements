from pathlib import Path
from itertools import product

import yaml
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm


FREQS, PPTS, STATES, FOLDS, METRICS, KINEMATICS = np.arange(6)

    # Set some information
CONFIG_I =        (5, 10, 25, 50, 100)  # Should be dynamically from config, but didnt change them anyway
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


cmap = cm.batlow

permutations = 10

def calculate_chance_level(z, zh, alpha=0.05, block_size=.1, n_repetitions=10000):
    # block size = % of data

    n_samples = z.shape[0]
    boundary = int(n_samples * block_size)

    permuted_ccs = np.empty((0, z.shape[1]))
    for _ in np.arange(n_repetitions):

        split_idx = np.random.choice(np.arange(boundary, n_samples - boundary))
        z_permuted = np.concatenate([zh[split_idx:, :], zh[:split_idx, :]])

        cc = [np.abs(np.corrcoef(zi, zhi)[0, 1]) for zi, zhi in zip(z.T, z_permuted.T)]

        permuted_ccs = np.vstack([permuted_ccs, cc])

    true_ccs = [np.corrcoef(zi, zhi)[0, 1] for zi, zhi in zip(z.T, zh.T)]  # abs?

    chance_idx = int(n_repetitions * (1-alpha))
    chance_level = np.sort(permuted_ccs, axis=0)[chance_idx, :]

    return chance_level, permuted_ccs

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
            # std =  scores.std( axis=(PPTS, FOLDS))[i_cond, :, 0, KINEMATIC_ORDER[i_ax]]

            ax.plot(mean, label=condition if i_ax==0 else '')
            # ax.fill_between(np.arange(len(state_options)), mean-std, mean+std,
            #                 alpha=0.5)
                            
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

def plot_scores_per_state(scores):

    scores_per_state = (scores.mean(axis=(0, 1, 3, 5))[:, 0], scores.std(axis=(0, 1, 3, 5))[:, 0])

    x_ticks = np.arange(len(STATE_OPTIONS))
    x_ticklabels = [f'{n1}_{i}' for n1, i in STATE_OPTIONS]

    fig, ax = plt.subplots()
    ax.plot(x_ticks, scores_per_state[0], color='black')
    ax.fill_between(x_ticklabels, 
                    scores_per_state[0] - scores_per_state[1],
                    scores_per_state[0] + scores_per_state[1],
                    alpha=0.5, color='black')

    ax.axvline(scores_per_state[0].argmax(), 
               color='orange', linestyle='dashed', linewidth=5, label='Max performance')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, rotation=90)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('N states & Horizons [<state>_<horizon>]')
    ax.set_ylabel('CC [mean]')

    ax.set_title('Mean performance per states over conditions, ppts, fold and kinematics')
    # ax.set_ylim(0, 1)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig('./figure_output/max_performance_per_state.png')
    fig.savefig('./figure_output/max_performance_per_state.svg')

    return scores_per_state[0].argmax(), scores_per_state[0].max()

def plot_mean_performance(scores, chance_levels, name):

    jitter = lambda n: (np.arange(n) - n/2) + 0.5

    if scores.ndim == 4:
        scores = scores[np.newaxis, :, :, :, :]

    mean = scores.mean(axis=(1, 2))[:, 0, :]
    std =  scores.std( axis=(1, 2))[:, 0, :]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    bar_width = 0.20
    jitter_idc = jitter(mean.shape[0]) * bar_width
    colors = [cmap(0), cmap(0.33), cmap(0.66)]
    # colors = [cmap(int(i)) for i in np.linspace(0, 255, mean.shape[0])]
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

    fig.legend(frameon=False) #, fontsize='x-large')
    # plt.show(block=True)
    fig.savefig(f'./figure_output/mean_performance_per_band_per_kinematic{"_"+name}.png')
    fig.savefig(f'./figure_output/mean_performance_per_band_per_kinematic{"_"+name}_.svg')

    return

def plot(results, name):
    # results = [(best_paths, scores), ...]  delta, ab, bbhg
    # Gather the data
    scores =        stack_scores(results).squeeze()
    chance_levels = stack_scores(results, 'chance_levels_prediction')

    ### Plot it
    # _, _ = plot_freqs_vs_states(scores)

    # best_state_i, best_state = plot_scores_per_state(scores)


    plot_mean_performance(scores, chance_levels, name)

    return