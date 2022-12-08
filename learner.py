import datetime as dt
import logging
from copy import deepcopy
from pathlib import Path
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import PSID
import yaml
from numpy.random import MT19937, RandomState, SeedSequence
from PSID.evaluation import evalPrediction as eval_prediction
from sklearn.linear_model import LinearRegression

import libs.utils
from libs import checks
from libs.feature_selection.forward_feature_selection import forward_feature_selection
from libs.feature_selection.kbest import select_k_best
from figures import all_figures
from figures import checks as fig_checks

logger = logging.getLogger(__name__)
c = libs.utils.load_yaml('./config.yml')
rs = RandomState(MT19937(SeedSequence(62277783366)))

def save(path, **kwargs):
    config = deepcopy(c)
    for name, value in kwargs.items():
        
        # Saves only numpy objects
        with open(path/f'{name}.npy', 'wb') as f:
            np.save(f, value)
    
    with open(path/'config.yml', 'w') as f:
        yaml.dump(libs.utils.nested_namespace_to_dict(config), f)

    logger.info(f'Saved to {path}')

def select_features(n_dims, n_folds, y, z, nx, n1, i):

    if c.learn.fs.greedy_forward: return forward_feature_selection(n_dims, n_folds, y, z, nx, n1, i)
    if c.learn.fs.kbest:          return select_k_best(y, z, n_dims)
    # if c.learn.fs.task_corr:      return select_highest_correlation(y, z, n_dims)
    if c.learn.fs.random:         return np.random.randint(0, y.shape[1], n_dims)

    logger.warning("Feature selection is enabled, but no method is selected. Using all features.")
    return np.arange(y.shape[1])

def select_valid_datasets(datasets, i, k):
    '''Checks if dataset is large enough to be included in
       the learning

        hard limitations: 
            H1: A dataset should contain at least 2*i + 1 samples
                This allows PSID to look back and forth exactly 
                1 time
        
        Soft limitation
            S1: Smaller datasets introduce more noise when a horizon
                crossed the gap.
    ''' 

    selected = []
    for d in datasets:

        n_samples = d.eeg.shape[0]
        min_sample_size = 2*i+1

        if n_samples < min_sample_size:
            continue
        
        # Set k to None to skip
        if k and n_samples < k*min_sample_size:
            continue

        selected += [d]
    
    return selected


def sanity_check(datasets):
    nx, n1, i = 10, 10, 10

    # eeg, xyz = datasets[0].eeg, datasets[0].xyz
    
    if len(datasets) > 1:
        y_train = np.vstack([datasets[0].eeg, datasets[1].eeg])
        z_train = np.vstack([datasets[0].xyz, datasets[1].xyz])
    else:
        y_train = datasets[0].eeg
        z_train = datasets[0].xyz
    y_test  = datasets[2].eeg
    z_test  = datasets[2].xyz 

    # samples = 900

    # # n = 2520
    # y_train, z_train = eeg[:samples, :], xyz[:samples, :]  # = 2000 samples
    # y_test, z_test   = eeg[samples:, :], xyz[samples:, :]  # = 520 samples

    id_sys = PSID.PSID(y_train, z_train, nx, n1, i)

    z_hat, y_hat, x_hat = id_sys.predict(y_test)

    logger.info('Sanity check:')
    logger.info(f'\tR2 [z]: {eval_prediction(z_test, z_hat, "R2").mean():.3f}')
    logger.info(f'\tR2 [y]: {eval_prediction(y_test, y_hat, "R2").mean():.3f}')

    return 0

def get_psid_params(n_states, relevant, horizons):
    if not n_states:
        for n1, i in product(relevant, horizons):
            yield n1, n1, i
    else:
        for nx, n1, i in product(n_states, relevant, horizons):
            yield nx, n1, i

def fit_and_score(z, y, nx, n1, i, save_path):

    n_samples =        c.learn.cv.n_repeats
    n_dims =           c.learn.fs.n_dims      if c.learn.fs.dim_reduction else y.shape[1]
    n_folds =          c.learn.cv.outer_folds
    n_inner_folds =    c.learn.cv.inner_folds
    z_dims = z.shape[1]

    results =                np.empty((n_samples, n_folds,    4, z_dims))   #  4 measures per channel
    trajectories =           np.empty((n_samples, y.shape[0], z_dims))      #  Z
    neural_reconstructions = np.empty((n_samples, y.shape[0], n_dims)) #  Y
    latent_states =          np.empty((n_samples, y.shape[0], nx))     #  X

    logger.info(f'''Input for learning: 
                        z={z.shape} | y={y.shape} | n_dims={n_dims}
                        nx={nx} | n1={n1} | i={i}''')
    for j in range(n_samples):
        
        folds = np.array_split(np.arange(y.shape[0]), n_folds)
        for idx, fold in enumerate(folds):

            y_test, y_train = y[fold, :], np.delete(y, fold, axis=0)
            z_test, z_train = z[fold, :], np.delete(z, fold, axis=0)
 
            print('LR: ', end='')
            for idim in np.arange(z_train.shape[1]):
                lr = LinearRegression() 
                lr = lr.fit(y_train, z_train[:, idim])
                r = lr.score(y_test, z_test[:, idim])
                print(f' {r:.2f}', end='')
            print('\n')

            # TODO: THIS doesnt seem to select on training data!!!
            features = select_features(n_dims, n_inner_folds, y_train, z_train, nx, n1, i) \
                       if c.learn.fs.dim_reduction else np.arange(y.shape[1])

            y_test, y_train = y_test[:, features], y_train[:, features]

            id_sys = PSID.PSID(y_train, z_train, nx, n1, i)
            logger.info(f'Fold {j}_{idx}: Fitted PSID [{y_train.shape}]')
            zh, yh, xh = id_sys.predict(y_test)

            metrics = np.vstack([eval_prediction(z_test, zh, measure) for measure in ['CC', 'R2', 'MSE', 'RMSE']])
            logger.info(f'Fold {j}_{idx} | CC: {metrics[0, :].mean():.2f}+{metrics[0, :].std():.2f} | RMSE: {metrics[3, :].mean():.1f}+{metrics[3, :].std():.1f}')
            # logger.info(metrics)
            results[j, idx, :, :] =              metrics
            trajectories[j, fold, :] =           zh
            neural_reconstructions[j, fold, :] = yh
            latent_states[j, fold, :] =          xh

    if c.learn.save:
        save(save_path,
             metrics=results,
             trajectories=trajectories,
             neural_reconstructions=neural_reconstructions,
             latent_states=latent_states,
             z=z,
             y=y)

def fit(datasets, save_path): 
    '''
    x = Latent states
    y = Neural data
    z = behavioral data

    nx:   Total Number of latent states
    n1:   Latent states dedicated to behaviorally relevant dynamics
    i:    Subspace horizon (low dim y = high i)   

    n_samples:     Amount of samples for random feature selection
    n_dims:        Amount of dimensions to include
    n_folds:       Model evaluation
    n_inner_folds: Hyperparameter optimization
    '''

    # Select what do decode
    target_kinematics = np.hstack([[0, 1, 2] if c.pos else [],
                                   [3, 4, 5] if c.vel else [],
                                   [6] if c.speed else []]).astype(np.int16)  # Z


    # sanity_check(datasets)
    n_z = target_kinematics.size
    
    n_states, relevant, horizons = c.learn.psid.nx, c.learn.psid.n1, c.learn.psid.i

    if c.checks.concat_datasets:
        checks.concatenate_vs_separate_datasets(datasets)

    for nx, n1, i in get_psid_params(n_states, relevant, horizons):

        if (nx < n1) or (n1 > i*n_z):
            continue
        
        # if not c.debug.go:
        if True:
            # TODO: If going for separate sets, the code needs updating
            # TODO: I think this is now handled ealier
            datasets = select_valid_datasets(datasets, i, c.learn.data.min_n_windows)
            # if i == min(horizons):
            if True:
                fig_checks.plot_datasets(datasets, target_kinematics)
            
        y = np.vstack([s.eeg for s in datasets])
        z = np.vstack([s.xyz[:, target_kinematics] for s in datasets])

        path = save_path/f'{nx}_{n1}_{i}'
        path.mkdir()
        
        fit_and_score(z, y, nx, n1, i, path)
        all_figures.make(path)  # Figures per session  # Rename to all_figures.make_session

        try:
            if c.figures.make_all:
                pass
        except Exception as e:
            logger.error(e)