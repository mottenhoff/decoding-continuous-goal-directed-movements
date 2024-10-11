import datetime as dt
import logging
import pickle
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
from libs.explore import task_correlations
from libs.feature_selection.forward_feature_selection import forward_feature_selection
from libs.feature_selection.kbest import select_k_best
from libs.feature_selection.top_correlated import select_top_correlated
from figures import all_figures
from figures import checks as fig_checks


CC = 0

logger = logging.getLogger(__name__)
c = libs.utils.load_yaml('./config.yml')
rs = RandomState(MT19937(SeedSequence(62277783366)))  # TODO: Make sure to use this random state correctly.

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

    if c.learn.fs.task_corr:      return select_top_correlated(y, z, n_dims)
    if c.learn.fs.greedy_forward: return forward_feature_selection(n_dims, n_folds, y, z, nx, n1, i)
    if c.learn.fs.kbest:          return select_k_best(y, z, n_dims)
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
    if k <= 0:
        return datasets

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
    
    logger.info(f'Selected {len(selected)} out of {len(datasets)} datasets ')

    return selected

def plot_kinematics(z, subset_starts, save_path, name):

    fig, axs = plt.subplots(nrows=3, ncols=4, sharex=True,
                            figsize=(19, 13))

    order = [0, 1, 2, 9, 3, 4, 5, 10, 6, 7, 8, 11]
    for i_kin, ax in zip(order, axs.flatten()):
        ax.plot(z[:, i_kin])

        for vline in subset_starts:
            ax.axvline(vline, color='r', linestyle='--')
        ax.set_title(f'{name} {i_kin}')
    
    fig.tight_layout()
    fig.savefig(save_path/f'{name}.png')
    fig.savefig(save_path/f'{name}.svg')

def get_psid_params(n_states, relevant, horizons):
    if not n_states:
        for n1, i in product(relevant, horizons):
            yield n1, n1, i
    else:
        for nx, n1, i in product(n_states, relevant, horizons):
            yield nx, n1, i

def fit(datasets, save_path):

    # Select datasets
    target_kinematics = np.hstack([[0, 1, 2] if c.pos   else [],
                                   [3, 4, 5] if c.vel   else [],
                                   [6, 7, 8] if c.acc   else [],
                                   [9]       if c.dist  else [],
                                   [10]      if c.speed else [],
                                   [11]      if c.force else []]).astype(np.int16)

    datasets = select_valid_datasets(datasets, max(c.learn.psid.i), c.learn.data.min_n_windows)

    task_correlations(datasets, save_path)

    y = np.vstack([s.eeg for s in datasets])
    z = np.vstack([s.xyz[:, target_kinematics] for s in datasets])

    n_z = target_kinematics.size

    # Define CV params
    n_outer_folds = c.learn.cv.outer_folds
    n_inner_folds = c.learn.cv.inner_folds
    n_samples =     c.learn.cv.n_repeats

    outer_folds = np.array_split(np.arange(y.shape[0]), n_outer_folds)

    # Define the grid
    n_states, relevant, horizons = c.learn.psid.nx, c.learn.psid.n1, [5]
    grid_params = list(get_psid_params(n_states, relevant, horizons))
    n_grid_params = len(grid_params)

    # Define machine learning params
    n_dims = c.learn.fs.n_dims if c.learn.fs.dim_reduction else y.shape[1]  # Feature selection dims
    n_metrics = 4
    
    # CV outer
    results =        np.empty((n_samples, n_outer_folds, n_metrics, n_z))
    cv_best_params = np.empty((n_samples, n_outer_folds, 3))     #  nx, n1, i

    for i_outer, outer_fold in enumerate(outer_folds):
        
        y_test, y_train = y[outer_fold, :], np.delete(y, outer_fold, axis=0)
        z_test, z_train = z[outer_fold, :], np.delete(z, outer_fold, axis=0)

        features = select_features(n_dims, None, y_train, z_train, None, None, None)
        y_test, y_train = y_test[:, features], y_train[:, features]

        # CV the grid
        inner_scores = np.empty((n_inner_folds, n_grid_params, n_metrics, n_z))
        inner_folds = np.array_split(np.arange(y_train.shape[0]), n_inner_folds)
        
        for i_grid, (nx, n1, i) in enumerate(grid_params):

            if (nx < n1) or (n1 > i*n_z):
                continue
        
            for i_inner, inner_fold in enumerate(inner_folds):

                y_train_test, y_train_train = y_train[inner_fold, :], np.delete(y_train, inner_fold, axis=0)
                z_train_test, z_train_train = z_train[inner_fold, :], np.delete(z_train, inner_fold, axis=0)
                
                # Fit and score PSID
                try:
                    id_sys = PSID.PSID(y_train_train, z_train_train, nx, n1, i) #, zscore_Y=True, zscore_Z=True)
                except np.linalg.LinAlgError as e:
                    logger.error('SVD did not converge.')
                except UnboundLocalError as e:
                    logger.error('Error undefined, but probably: SVD did not converge.')
                    raise Exception


                zh, yh, xh = id_sys.predict(y_train_test)
                metrics = np.vstack([eval_prediction(z_train_test, zh, measure) for measure in ['CC', 'R2', 'MSE', 'RMSE']])   # returns metrics x kinematics (=n_z)

                logger.info(f'Fold: {i_outer}-{i_inner} | cc={metrics[0, -2]:.2f} | nx={nx} n1={n1} i={i}')

                inner_scores[i_inner, i_grid, :, :] = metrics

        logger.info(f'Fold: {i_outer} | cc={metrics[0, -2]:.2f} | nx={nx} n1={n1} i={i}')

        # Calculate best params
        best_scores = inner_scores[:, :, 0, :].sum(axis=-1)
        i_best_params = np.argmax(best_scores.mean(axis=0))  # Selects best params on highest summed correlation
        best_params = grid_params[i_best_params]
        logger.info(f'Fold {i_outer} | summed CC: {best_scores.mean(axis=0)[i_best_params]:.2f} + {best_scores.std(axis=0)[i_best_params]:.2f} | Best params: n1={best_params[1]}, i={best_params[2]}')
        
        # Re-train PSID
        id_sys = PSID.PSID(y_train, z_train, *best_params, zscore_Y=True, zscore_Z=True)
        zh, yh, xh = id_sys.predict(y_test)

        metrics = np.vstack([eval_prediction(z_test, zh, measure) for measure in ['CC', 'R2', 'MSE', 'RMSE']])

        path = save_path/f'{i_outer}'
        path.mkdir(exist_ok=True)
        
        np.save(path/'z.npy', z_test)
        np.save(path/'y.npy', y)
        np.save(path/'trajectories.npy', zh)
        np.save(path/'latent_states.npy', yh)
        np.save(path/'selected_params.npy', best_params)
        np.save(path/'selected_channels.npy', features)

        results[0, i_outer, :, :] = metrics
        cv_best_params[0, i_outer, :] = best_params

    # Save overal information (results from best params)'
    
    # id_sys = PSID.PSID(y, z, *best_params, zscore_Y=True, zscore_Z=True)  # TODO: This selects the params of the last fold
    id_sys = PSID.PSID(y, z, 30, 30, 5, zscore_Y=True, zscore_Z=True)  # TODO: This selects the params of the last fold
    pickle.dump(id_sys, open(save_path/'trained_model.pkl', 'wb'))

    np.save(save_path/'y.npy', y)
    np.save(save_path/'z.npy', z)

    np.save(save_path/'results.npy', results)
    np.save(save_path/'cv_best_params.npy', cv_best_params)

    save(save_path)  # Can take kwargs, without it only saves config

