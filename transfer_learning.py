'''
y = neural data
x = latent states
z = kinematics

z = x . Cz

kinematics:
0, 1, 2 = pos x y z
3, 4, 5 = vel x y z
6, 7, 8 = acc x y z
9       = dist
10      = speed
11      = acc


'''
import pickle
from pathlib import Path
from copy import deepcopy
from itertools import combinations, product
from multiprocessing import Pool

# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import PSID
from PSID.evaluation import evalPrediction as eval_prediction

N_KINEMATICS = 12
N_FOLDS = 5
MEASURE = 'CC'

A_TO_B = 2
B_TO_A = 3
SPEED = 10

def cv_split(y, z, n_folds):
    # Test size = 1/n_folds

    assert y.shape[0] == z.shape[0], 'samples is y and z are not equal'

    idc = np.arange(y.shape[0])

    folds = np.array_split(idc, n_folds)
    for fold in folds:

        y_test, y_train = y[fold, :], np.delete(y, fold, axis=0)
        z_test, z_train = z[fold, :], np.delete(z, fold, axis=0)

        yield y_train, y_test, z_train, z_test

def train_baseline(y, z):

    baseline = np.empty((N_FOLDS, N_KINEMATICS))

    for i, (y_train, y_test, z_train, z_test) in enumerate(cv_split(y, z, N_FOLDS)):

        model = PSID.PSID(y_train, z_train, 30, 30, 10)
        z_hat, y_hat, x_hat = model.predict(y_test)
        baseline[i, :] = eval_prediction(z_test, z_hat, 'CC')

    return baseline

def get_pair_names(pairs):
    return [(pair_a.name, pair_b.name) for pair_a, pair_b in pairs]

def highest_pairs(results, pairs):
    # Do higher source and/or target scores lead to higher
    KINEMATIC_SPEED = 10
    A_TO_B = 2
    B_TO_A = 3

    pair_names = get_pair_names(pairs)
    
    results_mean = np.abs(results.mean(axis=2))

    highest_pair_score = results_mean[:, :2, KINEMATIC_SPEED].sum(axis=1)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    ax[0, 0].scatter(highest_pair_score, results_mean[:, A_TO_B, KINEMATIC_SPEED])
    ax[0, 0].set_xlabel('summed_individual_score')
    ax[0, 0].set_ylabel('A to B performance')
    ax[0, 1].scatter(results_mean[:, 0, KINEMATIC_SPEED], results_mean[:, 1, KINEMATIC_SPEED], s=results_mean[:, A_TO_B, KINEMATIC_SPEED]*10000)
    ax[0, 1].set_xlabel('decoding performance A')
    ax[0, 1].set_ylabel('decoding performance B')
    ax[0, 1].set_title('size by A to B score')

    ax[1, 0].scatter(highest_pair_score, results_mean[:, B_TO_A, KINEMATIC_SPEED])
    ax[1, 0].set_xlabel('summed_individual_score')
    ax[1, 0].set_ylabel('B to A performance')
    ax[1, 1].scatter(results_mean[:, 0, KINEMATIC_SPEED], results_mean[:, 1, KINEMATIC_SPEED], s=results_mean[:, B_TO_A, KINEMATIC_SPEED]*10000)
    ax[1, 1].set_xlabel('decoding performance A')
    ax[1, 1].set_ylabel('decoding performance B')
    ax[1, 1].set_title('size by A to B score')

    fig.savefig(f'relationship_baseline_and_transferability_{pairs[0][0].parent.name}.png')
    return


def plot_matrix(results, pairs, savepath):

    results = results.mean(axis=2)

    pair_names = np.array(get_pair_names(pairs))
    ppts = np.unique(pair_names)
    mapping = dict(zip(ppts, np.arange(ppts.size)))

    matrix = np.zeros((ppts.size, ppts.size))

    for idx, (a, b) in enumerate(pair_names):
        matrix[mapping[a], mapping[b]] = results[idx, A_TO_B, SPEED]
        matrix[mapping[b], mapping[a]] = results[idx, B_TO_A, SPEED]

    # # TODO: add diagonal of within score

    fig, ax = plt.subplots(figsize=(12, 12))

    # matrix = matrix.T

    print(results[:3, 2, 10], results[:3, 3, 10])


    # ax.imshow(matrix, cmap='plasma', vmin=0, vmax=1)
    color_boundary = np.abs(matrix).max()
    ax.imshow(matrix, cmap='coolwarm', vmin=-color_boundary, vmax=color_boundary)

    for s, t in product(np.arange(matrix.shape[0]), np.arange(matrix.shape[1])):
        ax.annotate(f'{matrix[s, t]:0.2f}', (s, t), ha='center', va='center', color='w')

    ax.set_xlabel('Source', fontsize='xx-large')
    ax.set_ylabel('Target', fontsize='xx-large')
    ax.set_xticks(np.arange(len(ppts)))
    ax.set_yticks(np.arange(len(ppts)))
    ax.set_yticklabels(ppts)
    ax.set_xticklabels(ppts)
    ax.invert_yaxis()

    fig.suptitle(f'{pairs[0][0].parent.name}', fontsize='xx-large')
    fig.savefig(f'figure_output/transfer_learning/transfer_learning_matrix_{pairs[0][0].parent.name}.png')

    return ax
    

def read_results(path, pairs):
    files = [file for file in path.glob('**/*') if file.is_file()]
    pairs = np.array([(a.parts[2], b.parts[2]) for a, b in pairs])
    results = np.empty((len(pairs), 4, N_FOLDS, N_KINEMATICS))

    for file in files:
        pair_a, pair_b = file.stem.split('_')
        idx = np.where(np.logical_and(pairs[:, 0] == pair_a, pairs[:, 1] == pair_b))
        results[idx] = np.load(file)

    return results


# def transfer_learn(y_a, y_b, z_a, z_b):
def transfer_learn(path_a: Path, path_b: Path, savepath: Path):

    print(f'Fitting: {path_a.name} and {path_b.name}', flush=True)

    y_a = np.load(path_a/'0'/'y.npy')
    z_a = np.load(path_a/'0'/'z.npy')
    y_b = np.load(path_b/'0'/'y.npy')
    z_b = np.load(path_b/'0'/'z.npy')

    # train_model a for fold i
    # train_model b for fold i
    # Swap Cz
    # Reevaluate

    baseline_a = np.empty((N_FOLDS, N_KINEMATICS))
    baseline_b = np.empty((N_FOLDS, N_KINEMATICS))
    transfer_ab = np.empty((N_FOLDS, N_KINEMATICS))
    transfer_ba = np.empty((N_FOLDS, N_KINEMATICS))

    cv_a = cv_split(y_a, z_a, N_FOLDS)
    cv_b = cv_split(y_b, z_b, N_FOLDS)
    for i in np.arange(N_FOLDS):

        # Train model a
        y_a_train, y_a_test, z_a_train, z_a_test = next(cv_a)
        model_a = PSID.PSID(y_a_train, z_a_train, 30, 30, 10)
        z_hat_a, _, _ = model_a.predict(y_a_test)
        cc_a = eval_prediction(z_a_test, z_hat_a, MEASURE)

        # Train model b
        y_b_train, y_b_test, z_b_train, z_b_test = next(cv_b)
        model_b = PSID.PSID(y_b_train, z_b_train, 30, 30, 10)
        z_hat_b, _, _ = model_b.predict(y_b_test)
        cc_b = eval_prediction(z_b_test, z_hat_b, MEASURE)

        # Swap 
        cz_a = deepcopy(model_a.Cz)
        cz_b = deepcopy(model_b.Cz)

        model_a.Cz = cz_b
        model_b.Cz = cz_a

        # Re-evaluate
        tl_z_hat_a, _, _ = model_a.predict(y_a_test)
        tl_z_hat_b, _, _ = model_b.predict(y_b_test)
        cc_ab = eval_prediction(z_a_test, tl_z_hat_a, MEASURE)
        cc_ba = eval_prediction(z_b_test, tl_z_hat_b, MEASURE)

        baseline_a[i, :] = cc_a
        baseline_b[i, :] = cc_b
        transfer_ab[i, :] = cc_ab
        transfer_ba[i, :] = cc_ba

    result = np.stack([baseline_a, baseline_b, transfer_ab, transfer_ba])
    np.save(savepath/f'{path_a.name}_{path_b.name}.npy', result)

def run_async(pairs, savepath):
    pool = Pool(processes=8)
    for path_a, path_b in pairs:
        pool.apply_async(transfer_learn, args=(path_a, path_b, savepath))
    pool.close()
    pool.join()

if __name__=='__main__':

    run_parallel = True

# Negative correlations might be because of the wide variation of correlations in delta, as some are strongly negatively correlated, and others strongly positively.
#      if so, then this should not be the case for bbhg and or alphabeta. The latter may not have sufficient performane in the first place.
# --> This is not true, as a similar distribution is found for bbhg, albeit lower.

    # Input data
    main_path = Path(r'finished_runs\delta')
    # main_path = Path(r'finished_runs\bbhg')
    # main_path = Path(r'finished_runs\alphabeta')

    savepath = Path(f'results_transfer_learning/{main_path.name}/pairs')
    savepath.mkdir(exist_ok=True, parents=True)

    pairs = [(path_a, path_b) for path_a, path_b in combinations(main_path.glob('kh*'), 2)]

    if False:
        # Rerun transfer learning
        if run_parallel:
            run_async(pairs, savepath)
        else:
            for path_a, path_b in pairs:
                transfer_learn(path_a, path_b, savepath)

    results = read_results(savepath, pairs)

    plot_matrix(results, pairs, savepath)

    highest_pairs(results, pairs)

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))

    for i in range(4):
        scores = results.mean(axis=2)[:, i, :]
        idc = np.tile(np.arange(12), scores.shape[0]).reshape((scores.shape[0], -1))
        axs[i].violinplot(scores, np.arange(N_KINEMATICS), showextrema=False)
        axs[i].scatter(idc, scores, color='black', marker='o', s=2)

    axs[0].set_title('Baseline A')
    axs[1].set_title('Baseline B')
    axs[2].set_title('Swapped\ninserted Cz_b to model A\nRepredicted y_a')
    axs[3].set_title('Swapped\ninserted Cz_a to model B')
    axs[0].set_ylabel('Reconstruction CC')
    axs[0].set_xlabel('kinematics')
    fig.suptitle(f'{main_path.name}')

    for ax in axs:
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(N_KINEMATICS))

    outpath = Path(f'figure_output/transfer_learning')
    outpath.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath/f'transfer_learning_{main_path.name}.png')

    plt.show()