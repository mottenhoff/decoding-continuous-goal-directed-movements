from pathlib import Path

import yaml
import numpy as np

def load_chance_levels(path, name, percentile=0.95):

    filepath = list(path.glob(f'chance_levels_{name}_*'))
    if not filepath:
        raise FileNotFoundError(f'Could not find chance levels file for {path}. Did you run the calculation script?') 

    filepath = sorted(list(filepath))[-1]
    # Load the file with the most permutations    
    chance_levels = np.load(filepath)

    if name == 'prediction':
        chance_levels = chance_levels[:, *np.diag_indices(chance_levels.shape[1])]

    n_permutations = int(filepath.stem.split("_")[-1])
    idx_nth_percentile = int(n_permutations * percentile)
    return np.sort(np.abs(chance_levels), axis=0)[idx_nth_percentile, :]

def get_best_scores(results):
    metric = lambda v: v['scores'][:, :, 0, :].mean(axis=1).sum(axis=-1).argmax()

    best_score_idc = {ppt: metric(values) for ppt, values in results.items()}
    best_results = {ppt: {'scores': v['scores'][best_score_idc[ppt], :, :, :],
                    'paths':  v['paths'][best_score_idc[ppt]],
                    'chance_levels': v['chance_levels'][best_score_idc[ppt], :]}
                        for ppt, v in results.items()}

    return best_results, best_score_idc

def get_results(path_main, path_data, skip=False):

    # all_runs = [session for ppt in path_main.iterdir() if ppt.is_dir() for session in ppt.iterdir()]
    all_runs = [ppt for ppt in path_main.iterdir() if ppt.is_dir()]


    results = {}
    for run in all_runs:
        
        ppt_id = run.name

        if Path(run/'profile.prof') not in run.iterdir():
            continue

        with open(run/'info.yml') as f:
            run_info = yaml.load(f, Loader=yaml.FullLoader)
        
        with open(path_data/ppt_id/'info.yaml') as f:
            recording_info = yaml.load(f, Loader=yaml.FullLoader)

        if recording_info['problems_left'] and skip:
            print(f'Skipping {ppt_id}')
            continue

        result = np.load(run/'results.npy')
        params = np.vstack([np.load(run/f'{i}'/'selected_params.npy') for i in range(5)])

        # chance_levels_prediction = load_chance_levels(run, 'prediction')
        chance_levels_task = load_chance_levels(run, 'task_correlation')

        results.update({ppt_id: {'scores': result,
                                 'params': params,
                                 'paths': run,
                            #    'chance_levels_prediction': chance_levels_prediction,
                                 'chance_levels_task_correlation': chance_levels_task,
                                 'datasize': run_info['datasize'],
                                 'n_targets': run_info['n_targets']}})

    return results