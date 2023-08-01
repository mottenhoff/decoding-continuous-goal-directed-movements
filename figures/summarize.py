from pathlib import Path
from collections import OrderedDict

import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

try:
    import libs.maps as maps
except ModuleNotFoundError:
    import maps

Y_DIMS = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'dist', 'speed', 'force']
N_FOLDS = 5
N_SCORES = 3  # CC, n_samples, n_targets

REPETITIONS, FOLDS, METTRICS, Z_DIMS = 0, 1, 2, 3
CC, R2, MSE, RMSE = 0, 1, 2, 3
MEAN, STD = 1, 2

PATH = -1

def load(path):
    # TODO: Again, copy pasted from figures_1d_score_overview.py
    m =  np.load(path/'metrics.npy')  # CC, R2, MSE, RMSE
    z =  np.load(path/'z.npy')
    y =  np.load(path/'y.npy')
    zh = np.load(path/'trajectories.npy')
    yh = np.load(path/'neural_reconstructions.npy')    
    xh = np.load(path/'latent_states.npy')

    return m, z, y, zh, yh, xh

def get_results(path_main, skip=False):

    all_runs = [session for ppt in path_main.iterdir() if ppt.is_dir() for session in ppt.iterdir()]
    
    results = {}
    for run in all_runs:
        
        ppt_id = run.parts[-2]

        if Path(run/'profile.prof') not in run.iterdir():
            continue

        # with open(run/'info.yml') as f:
        #     run_info = yaml.load(f, Loader=yaml.FullLoader)
        run_info = {'datasize': -1,
                    'n_targets': 50}
        
        with open(f'./data/{ppt_id}/info.yaml') as f:
            recording_info = yaml.load(f, Loader=yaml.FullLoader)

        if recording_info['problems_left'] and skip:
            print(f'Skipping {ppt_id}')
            continue

        scores = np.empty((0, 5, 4, 12))
        paths = np.array([])
        for result in run.iterdir():
            
            if not result.is_dir():
                continue

            metrics = np.load(result/'metrics.npy') # Folds, Scoretype, Ydims

            scores = np.vstack([scores, metrics])
            paths = np.append(paths, result)

        results.update({'_'.join(run.parts[-2:]): {'scores': scores,
                                                   'paths': paths,
                                                   'datasize': run_info['datasize'],
                                                   'n_targets': run_info['n_targets']}})

    return results

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


def annote_significance_above_max_y_value(ax, y_values: list, p_values: list, rotate=True, fontsize='small'):

    x_ticks = ax.get_xticks()

    is_sig =  lambda p_value: '***' if p_value < 0.001 else \
                              '**'  if p_value < 0.01  else \
                              '*'   if p_value < 0.05  else \
                              ''  # n.s.

    for x_tick, y_value, p_value in zip(x_ticks, y_values, p_values):
        ax.annotate(f'{is_sig(p_value)}', xy=(x_tick, y_value+0.2),
                    ha='center', va='center', fontsize=fontsize, 
                    rotation='vertical' if rotate else 'horizontal')

    return ax

def plot_states_and_horizons(paths, scores):

    scores = []
    states = []
    for path in paths:
        scores += [np.load(path/'metrics.npy')[0, :, 0, -1].mean()]
        states += [(int(path.name.split('_')[1]), 
                    int(path.name.split('_')[2]))]
    
    scores = np.hstack([np.vstack(states), np.array(scores)[:, np.newaxis]])

    ppts = sorted(list(maps.ppt_id().keys()))
    colors = [maps.cmap()[maps.ppt_id()[ppt]] for ppt in ppts]

    fig, ax = plt.subplots(ncols=2)
    ax[0].scatter(scores[:, 0], scores[:, 2])
    ax[1].scatter(scores[:, 1], scores[:, 2])
    fig.savefig('./figure_output/states_vs_performance.png')

    fig, ax = plt.subplots()
    xticks = np.arange(len(paths))
    ax.bar(xticks, scores[:, 0], color=colors)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{i+1}' for i, _ in enumerate(ppts)], fontsize='x-large')
    ax.set_ylabel('Behaviorally relevant states', fontsize='x-large')
    # ax.set_title('States required for optimal performance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('./figure_output/states_per_ppt.png')


    return

def plot_overview(results, name):
    n_metrics = 12

    # cosine_similarity = lambda signal1, signal2: np.dot(signal1, signal2) / (np.linalg.norm(signal1) * np.linalg.norm(signal2))
    n_permutations = 100

    n_ys = list(results.values())[0]['scores'].shape[-1]
    ppts = np.array(sorted(list(results.keys())))
    ppt_ids = [ppt[:5] for ppt in ppts]

    colors = [maps.cmap()[maps.ppt_id()[ppt]] for ppt in ppt_ids]

    # Loop to guarantee correct order
    scores = np.empty((0, 5, n_metrics))
    datasizes = []
    n_targets = []
    paths = []
    chance_levels = np.empty((0, n_metrics))
    all_permuted = np.empty((0, n_metrics))

    for ppt in ppts:
        mean_scores = results[ppt]['scores'][:, :, CC, :].mean(axis=1)
        idx_max = np.where(mean_scores==mean_scores.max())[0]
        path = results[ppt]['paths'][idx_max][0]

        scores = np.vstack([scores, results[ppt]['scores'][idx_max, :, CC, :]])
        datasizes = np.append(datasizes, results[ppt]['datasize'])
        n_targets = np.append(n_targets, results[ppt]['n_targets'])
        paths += [path]

        _, z, _, zh, _, xh = load(path)
        zh = zh.squeeze()

        chance_level, permuted = calculate_chance_level(z, zh.squeeze(), n_repetitions=n_permutations)
        chance_levels = np.vstack([chance_levels, chance_level])
        all_permuted = np.vstack([all_permuted, permuted])
        
    
    # plot_states_and_horizons(paths, scores)


    xticks = np.arange(len(ppts))
    xlabels = [maps.ppt_id().get(ppt, ppt) for ppt in ppts]
    colors = [maps.cmap()[maps.ppt_id()[ppt]] for ppt in ppt_ids]

    fig, ax = plt.subplots(nrows=2, figsize=(5, n_metrics))
    
    ax[0].bar(xticks, datasizes, color=colors)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels([f'{i+1}' for i, _ in enumerate(ppts)], fontsize='medium')
    ax[0].set_ylabel('Samples', fontsize='x-large')
    
    ax[1].bar(xticks, n_targets, color=colors)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels([f'{i+1}' for i, _ in enumerate(ppts)], fontsize='medium')
    ax[1].set_ylabel('Targets reached', fontsize='x-large')
    fig.tight_layout()
    fig.savefig('./figure_output/datasize.png')
    # xticks = np.arange(len(paths))
    # ax.bar(xticks, scores[:, 0], color=colors)
    # ax.set_ylabel('Behaviorally relevant states', fontsize='x-large')
    # # ax.set_title('States required for optimal performance')

    fig, ax = plt.subplots(n_ys, 4, figsize=(9, 16))

    for y in np.arange(n_ys):

        xticks = np.arange(len(ppts))
        xlabels = [maps.ppt_id().get(ppt, ppt) for ppt in ppts]
        colors = [maps.cmap()[maps.ppt_id()[ppt]] for ppt in ppt_ids]

        score = scores[:, :, y]
        print(score.mean(axis=1))
        # print(score.mean(axis=1), score.std(axis=1))

        x_ppts = np.arange(ppts.size)
        ax[y, 0].bar(x_ppts, score.mean(axis=1), yerr=score.std(axis=1), capsize=0, ecolor='grey', color=colors)
        ax[y, 1].bar(x_ppts, datasizes, color=colors)
        ax[y, 2].scatter(datasizes, score.mean(axis=1), s=25, color=colors)
        ax[y, 3].scatter(n_targets, score.mean(axis=1), s=25, color=colors)
        

        ax[y, 0].set_title(f'{Y_DIMS[y]}')
        ax[y, 0].set_xticks(xticks)        
        # annote_significance_above_max_y_value(ax[y, 0], score.mean(axis=1), p_values[:, y])

        if y == 0:
            ax[y, 1].set_title('Datasize')
            ax[y, 2].set_title('datasize vs performance')
            ax[y, 3].set_title('# targets vs performance')

        if y == n_ys-1:
            ax[y, 0].set_xticklabels(xlabels)
            ax[y, 1].set_xticks(xticks)
            ax[y, 1].set_xticklabels(xlabels)
            ax[y, 0].set_xticklabels(ppts, rotation=45, ha='right')

            ax[y, 1].set_xticklabels(ppts, rotation=45, ha='right')
            ax[y, 2].set_xlabel('Datasize [samples]')
            ax[y, 3].set_xlabel('# targets')

        
        ax[y, 0].set_ylabel('CC')
        ax[y, 0].set_ylim(0, 1)

    fig.suptitle(name)
    fig.tight_layout()

    fig.savefig('./figure_output/summarize_overview.png')

    # Trials per score?



    ## Selection

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    names = ['Speed', 'Velocity X', 'Velocity Y', 'Velocity Z']

    alpha = 0.05

    for yi, y in enumerate([6, 3, 4, 5]):
        score = scores[:, :, y]

        ax[yi].bar(x_ppts, score.mean(axis=1), yerr=score.std(axis=1), capsize=0, ecolor='grey', color=colors)
        ax[yi].set_title(names[yi], fontsize='xx-large')
        ax[yi].set_xticks(xticks)
        ax[yi].set_xticklabels([f'{i+1}' for i in xticks], fontsize='large')
        ax[yi].set_yticklabels('')
        
        ax[yi].spines['top'].set_visible(False)
        ax[yi].spines['right'].set_visible(False)

        xlim = ax[yi].get_xlim()
        for ci, (tick, bar_patch) in enumerate(zip(xticks, ax[yi].patches)):
            abs_to_rel = lambda x, lim: (x - lim[0]) / (lim[1] - lim[0]) 
            half_width = abs_to_rel(xlim[0]+bar_patch.get_width()/2, xlim)
            anchor = abs_to_rel(tick, xlim)
            xmin, xmax = anchor - half_width, anchor + half_width
            
            ax[yi].axhline(chance_levels[ci, y], xmin=xmin, xmax=xmax, 
                           color='black', linestyle='--', alpha=0.7, 
                           **{'label': 'chance level'} if yi==3 and ci==0 else {})

        if yi==0:
            ax[yi].set_xlabel('Participant', fontsize='xx-large')
            ax[yi].set_ylabel('CC', fontsize='xx-large')
            ax[yi].set_yticklabels([f'{i:.1f}' for i in np.linspace(0, 1, 6)], fontsize='xx-large')

        ax[yi].set_ylim(0, 1)
        ax[yi].grid(visible=True, axis='y', linewidth=1, linestyle='dotted', color='grey')

    fig.legend(loc='upper right', fontsize='small', frameon=False)
    fig.tight_layout()
    fig.savefig('./figure_output/speed_overview.png')

    return 

def main(path):

    name = path.name.split('_')[0]

    results = get_results(path)

    plot_overview(results, name)


if __name__=='__main__':
    # path = Path('results/')
    # path = Path('./finished_runs/beta_all/')
    path = Path('./finished_runs/delta/')
    main(path)
