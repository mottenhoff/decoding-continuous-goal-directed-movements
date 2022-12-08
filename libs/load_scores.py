from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from libs import utils

MEAN = 0
STD = 1
SCORE_TYPES = [MEAN, STD]
N_METRICS = 4

# Score dims
REPETITIONS = 0
FOLDS = 1
METRICS = 2
Z_DIMS = 3

def get_psid_params(n_states, relevant, horizons):
    # TODO: This is now copy paste from learn.py! bc: Circular import error
    if not n_states:
        for n1, i in product(relevant, horizons):
            yield n1, n1, i
    else:
        for nx, n1, i in product(n_states, relevant, horizons):
            yield nx, n1, i

def get_scores(path):

    to_num = lambda s: tuple(int(x) for x in s.split('_'))

    folders = [d for d in path.glob('*/**') if d.is_dir()]

    config = utils.load_yaml(folders[0]/'config.yml')

    all_i =  np.array(config.learn.psid.i)
    all_n1 = np.array(config.learn.psid.n1)
    all_nx = np.array(config.learn.psid.nx)

    results = np.full((all_nx.size if all_nx else 1, all_n1.size, all_i.size, N_METRICS, len(SCORE_TYPES)), np.nan)

    for nx, n1, i in get_psid_params(all_nx, all_n1, all_i):

        for folder in folders:

            if (nx, n1, i) != to_num(folder.name):
                continue

            # Get indices
            idx_nx =        np.where(all_nx == nx) if all_nx else 0
            idx_n1, idx_i = np.where(all_n1 == n1), np.where(all_i==i)

            scores = np.load(folder/'metrics.npy')  # [repetitions, folds, metrics, z_dims]  -> reps usually 1, only used with non-deterministic processes

            dims = scores.shape
            scores = scores.reshape(dims[REPETITIONS]*dims[FOLDS], dims[METRICS], dims[Z_DIMS])

            results[idx_nx, idx_n1, idx_i, :, :] = np.hstack((scores.mean(axis=0),
                                                              scores.std(axis=0)))

    return results