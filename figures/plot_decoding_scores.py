from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm



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
    col_titles = ['X', 'Y', 'Z', r'$\sum$']
    row_titles = ['Position', 'Velocity', 'Acceleration']
    MEAN, STD = 0, 1
    cmap = cm.batlow
    METRIC = 0  # [CC, R2, MSE, RMSE]

    results = get_results(path)

    # a = [(ppt, result['scores']) for ppt, result in results.items()]
    # a[0][1][:, :, 0, :].mean(axis=1).max(axis=0)

    scores = []
    best_paths = []
    for ppt, result in results.items():
        score = result['scores']

        if summed_cc := True:
            # Select on highest summed CC over kinematics
            best_params = score[:, :, METRIC, :].mean(axis=1).sum(axis=1).argmax()  
            max_mean = score[best_params, :, METRIC, :].mean(axis=0)
            max_std =  score[best_params, :, METRIC, :].std(axis=0)
        
        else:
            # Select on highest CC per individual kinematics
            #   i.e. varying parameters per kinematic.
            #   TODO: !! Probably Wrong!
            best_params = np.where(score[:, :, METRIC, :].mean(axis=1) == score[:, :, METRIC, :].mean(axis=1).max(axis=0))
            max_mean = score[best_params[0], :, METRIC, best_params[1]].mean(axis=1)
            max_std =  score[best_params[0], :, METRIC, best_params[1]].std(axis=1)

        scores += [(ppt, np.vstack([max_mean, max_std]))]
        best_paths += [result['paths'][best_params]]

    scores = sorted(scores)
    ppts =   [score[0] for score in scores]
    scores = np.dstack([score[1] for score in scores]).transpose(2, 1, 0)

    order = np.array([['pos_x', 'pos_y', 'pos_z', 'dist'],
                      ['vel_x', 'vel_y', 'vel_z', 'speed'],
                      ['acc_x', 'acc_y', 'acc_z', 'force']])
    fig_shape = order.shape

    xticks = np.arange(len(ppts))
    colors = [cmap(int(i)) for i in np.linspace(0, 255, len(ppts))]

    fig, ax = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1], figsize=(16, 9))
    for idx, score_name in zip(np.ndindex(*fig_shape), order.flatten()):
        mean = scores[:, Y_DIMS.index(score_name), MEAN]
        std =  scores[:, Y_DIMS.index(score_name), STD]
        ax[idx].bar(xticks, mean, color=colors, yerr=std)  # np.stack([(0, f) for f in std[freqs_i, :]]).T for only top errorbar
        
        # ax[idx].set_title(score_name)  # Sanity check
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        ax[idx].spines['left'].set_visible(False if idx[1] != 0 else True)
        ax[idx].set_xticks(np.arange(len(ppts)))
        ax[idx].set_xticklabels([])
        ax[idx].set_yticklabels([])
        ax[idx].set_ylim(0, 1)

        ax[idx].set_axisbelow(True)
        ax[idx].yaxis.grid()

        if idx[0] == 0:  # Top row
            ax[idx].set_title(col_titles[idx[1]], fontsize='xx-large')

        if idx[0] == order.shape[0]-1: # Bottom row
            ax[idx].set_xticks(np.arange(scores.shape[0]))
            ax[idx].set_xticklabels([ppt.split('_')[0] for ppt in ppts], fontsize='small', rotation=45, ha='right')

        if idx[1] == 0:  # Left column
            ax[idx].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax[idx].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax[idx].set_ylabel(f'{row_titles[idx[0]]}\nCC', fontsize='x-large')
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(f'figure_output/decoder_scores_{path.stem}.png')
        
    return best_paths, scores
 