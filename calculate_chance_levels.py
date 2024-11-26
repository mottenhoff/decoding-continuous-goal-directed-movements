from pathlib import Path
from itertools import product

from multiprocessing import Pool

import numpy as np

# Local
from monte_carlo_simulations import random_array_swap

KINEMATICS = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'r', 'v', 'a']
N_FOLDS = 5

AVG_BRAIN_PATH = Path(r'C:\Users\p70066129\Maarten\Resources\codebase\brainplots\models\cvs_avg35_inMNI152')
PATH_SERVER = Path(r'L:/FHML_MHeNs\sEEG/')

def permutation_metric(original, permuted):
    n_original = original.shape[1]
    
    correlation_matrix = np.corrcoef(np.hstack([original, permuted]), rowvar=False)

    return correlation_matrix[:n_original:, n_original:]

def calculate_chance_level_task_correlations(path, n_permutations):
    # TODO: main_path from env file
    print(f'Running task correlation: {path}', flush=True)
    
    y, z = np.load(path/'y.npy'), np.load(path/'z.npy')

    chance_levels = random_array_swap(y,
                                      z,
                                      permutation_metric,
                                      alpha=0.05,
                                      min_block_size=.1,
                                      repetitions=n_permutations)
    np.save(path/f'chance_levels_task_correlation_{n_permutations}.npy', chance_levels)

def calculate_chance_level_prediction(path, n_permutations):
    print(f'Running prediction: {path}', flush=True)
    
    zh = np.vstack([np.load(path/f'{i}'/'trajectories.npy') for i in range(N_FOLDS)])
    z = np.load(path/'0'/'z.npy')
    
    chance_levels = random_array_swap(z,
                                      z,
                                      permutation_metric,
                                      alpha=0.05,
                                      min_block_size=.1,
                                      repetitions=n_permutations)

    outpath = path/'0'
    outpath.mkdir(parents=True, exist_ok=True)
    np.save(outpath/f'chance_levels_prediction_{n_permutations}.npy', chance_levels)

def main():
    run_parallel = False

    np.random.seed(2024)
    n_permutations = 100

    main_path = Path(f'finished_runs_2/')

    conditions = main_path.rglob('sub-*')
    
    if run_parallel:
        pool = Pool(processes=8)
        for path in conditions:
            pool.apply_async(calculate_chance_level_task_correlations, args=(path, n_permutations))
            pool.apply_async(calculate_chance_level_prediction, args=(path, n_permutations))
        pool.close()
        pool.join()

    else:
        for path in conditions:
            calculate_chance_level_task_correlations(path, n_permutations)
            calculate_chance_level_prediction(path, n_permutations)

if __name__=='__main__':
    main()