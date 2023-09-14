from pathlib import Path

import yaml
import numpy as np

n_permutations = 10000

def calculate_chance_level(z, zh, alpha=0.05, block_size=.1, n_repetitions=10):
    # block size = % of data

    n_samples = z.shape[0]
    boundary = int(n_samples * block_size)

    permuted_ccs = np.empty((0, z.shape[1]))
    for _ in np.arange(n_repetitions):

        split_idx = np.random.choice(np.arange(boundary, n_samples - boundary))
        z_permuted = np.concatenate([zh[split_idx:, :], zh[:split_idx, :]])

        cc = [np.abs(np.corrcoef(zi, zhi)[0, 1]) for zi, zhi in zip(z.T, z_permuted.T)]

        permuted_ccs = np.vstack([permuted_ccs, cc])

    # true_ccs = [np.corrcoef(zi, zhi)[0, 1] for zi, zhi in zip(z.T, zh.T)]  # abs?

    chance_idx = int(n_repetitions * (1-alpha))
    chance_level = np.sort(permuted_ccs, axis=0)[chance_idx, :]

    return chance_level, permuted_ccs

def get_best_scores(results):
    metric = lambda v: v['scores'][:, :, 0, :].mean(axis=1).sum(axis=-1).argmax()

    best_score_idc = {ppt: metric(values) for ppt, values in results.items()}
    best_results = {ppt: {'scores': v['scores'][best_score_idc[ppt], :, :, :],
                    'paths':  v['paths'][best_score_idc[ppt]],
                    'chance_levels': v['chance_levels'][best_score_idc[ppt], :]}
                        for ppt, v in results.items()}

    return best_results, best_score_idc

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
        chance_levels = np.empty((0, 12))

        for result in run.iterdir():
            
            if not result.is_dir():
                continue

            metrics = np.load(result/'metrics.npy') # Folds, Scoretype, Ydims

            z, zh = np.load(result/'z.npy'), np.load(result/'trajectories.npy').squeeze()

            chance_level, _ = calculate_chance_level(z, zh, n_repetitions=n_permutations)
            chance_levels = np.vstack([chance_levels, chance_level])
            scores = np.vstack([scores, metrics])
            paths = np.append(paths, result)

        results.update({'_'.join(run.parts[-2:]): {'scores': scores,
                                                   'paths': paths,
                                                   'chance_levels': chance_levels,
                                                   'datasize': run_info['datasize'],
                                                   'n_targets': run_info['n_targets']}})

    return results