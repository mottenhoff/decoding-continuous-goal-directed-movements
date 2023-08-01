from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

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

        with open(run/'info.yml') as f:
            run_info = yaml.load(f, Loader=yaml.FullLoader)
        
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


def plot_overview(path):
    Y_DIMS = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'dist', 'speed', 'force']
    METRIC = 0  # [CC, R2, MSE, RMSE]
    results = get_results(path)

    # a = [(ppt, result['scores']) for ppt, result in results.items()]
    # a[0][1][:, :, 0, :].mean(axis=1).max(axis=0)

    scores = np.empty((0, 13))
    for ppt, result in results.items():
        score = result['scores']
        metric_max = score[:, :, METRIC, :].mean(axis=1).max(axis=0)

        scores = np.vstack([scores, np.hstack([ppt, metric_max])])

    scores = scores[np.argsort(scores[:, 0]), :]  # Sort by ppt_id

    order = np.array([['pos_x', 'pos_y', 'pos_z', 'dist'],
                      ['vel_x', 'vel_y', 'vel_z', 'speed'],
                      ['acc_x', 'acc_y', 'acc_z', 'force']])
    fig_shape = order.shape

    xticks = np.arange(scores.shape[0])
    colors = [cm.batlow(int(i)) for i in np.linspace(0, 255, scores.shape[0])]

    fig, ax = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1], figsize=(16, 9))
    for n, (idx, score_name) in enumerate(zip(np.ndindex(*fig_shape), order.flatten()), start=1):
        ax[idx].bar(xticks, scores[:, Y_DIMS.index(score_name)+1].astype(np.float32), color=colors)
        
        ax[idx].set_title(score_name)
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        ax[idx].set_xticklabels([])
        ax[idx].set_ylim(0, 1)

        if idx[0] == order.shape[0]-1: # Bottom
            ax[idx].set_xticks(np.arange(scores.shape[0]))
            ax[idx].set_xticklabels([s.split('_')[0] for s in scores[:, 0]], fontsize='small', rotation=45, ha='right')

        if idx[1] == 0:  # left side
            ax[idx].set_ylabel('CC', fontsize='x-large')
        
    fig.savefig('figure_output/decoder_scores.png')
        
    return
