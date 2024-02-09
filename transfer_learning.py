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
from itertools import combinations

# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import PSID
from PSID.evaluation import evalPrediction as eval_prediction

N_KINEMATICS = 12
N_FOLDS = 5
MEASURE = 'CC'

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

def transfer_learn(y_a, y_b, z_a, z_b):

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

    return baseline_a, baseline_b, transfer_ab, transfer_ba


path_a = Path(r'results\20240207_1556\20221103\0')
path_b = Path(r'results\20240207_2001\Bubbles_15_53_18\0')

main_path = Path(r'results\20240208_2240')

results = np.empty((0, 4, N_FOLDS, N_KINEMATICS))
for path_a, path_b in combinations(main_path.glob('kh*'), 2):
    try:
        y_a = np.load(path_a/'0'/'y.npy')
        z_a = np.load(path_a/'0'/'z.npy')
        y_b = np.load(path_b/'0'/'y.npy')
        z_b = np.load(path_b/'0'/'z.npy')
    except FileNotFoundError as e:
        print(e)
        print('skipping...')

    result_pair = np.stack(transfer_learn(y_a, y_b, z_a, z_b))
    results = np.vstack([results, np.expand_dims(result_pair, 0)])
    print('.', end='')

np.save('results.npy', results)

fig, axs = plt.subplots(nrows=1, ncols=4)

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

for ax in axs:
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.arange(N_KINEMATICS))

fig.show()








# y_a_train, y_a_test, z_a_train, z_a_test = train_test_split(y_a, z_a, .8)

# model_a = PSID.PSID(y_a_train, z_a_train, 30, 30, 10)
# cz_a = model_a.Cz.copy()
# z_hat_a, _, _ = model_a.predict(y_a_test)
# baseline_a = eval_prediction(z_a_test, z_hat_a, 'CC')

# # model_a = pickle.load(open(path_a/'trained_model.pkl', 'rb'))
# # cz_a = model_a.Cz.copy()
# # zh_a, yh_a, xh_a = model_a.predict(y_a)  # TODO: This data somehow produces a non-steady state kalman filter. Perhaps retry with different data. Might be that it only affects y to x
# # baseline_a = eval_prediction(z_a, zh_a, 'CC')  ## TODO Evaluate in split or CV
# y_b_train, y_b_test, z_b_train, z_b_test = train_test_split(y_b, z_b, .8)

# model_b = PSID.PSID(y_b_train, z_b_train, 30, 30, 10)
# cz_b = model_b.Cz.copy()
# z_hat_b, y_hat_b, _ = model_b.predict(y_b_test)
# baseline_b = eval_prediction(z_b_test, z_hat_b, 'CC')

# # model_b = pickle.load(open(path_b/'trained_model.pkl', 'rb'))
# # cz_b = model_b.Cz.copy()
# # zh_b, yh_b, xh_b = model_b.predict(y_b)
# # baseline_b = eval_prediction(z_b, zh_b, 'CC')  ## TODO Evaluate in split or CV


# # Transfer learning
# model_ab = deepcopy(model_a)
# model_ab.Cz = cz_b

# model_ba = deepcopy(model_b)
# model_ba.Cz = cz_a

# transfer_ab_zh, transfer_ab_yh, transfer_ab_xh = model_ab.predict(y_a)
# transfer_ba_zh, transfer_ba_yh, transfer_ba_xh = model_ba.predict(y_b)

# cc_ab_a = eval_prediction(transfer_ab_zh, z_a, 'CC')
# # cc_ab_b = eval_prediction(transfer_ab_zh, z_b, 'CC')  # Causes error on window size, which should not be there?

# # cc_ba_a = eval_prediction(transfer_ba_zh, z_a, 'CC')
# cc_ba_b = eval_prediction(transfer_ba_zh, z_b, 'CC')

# # TODO: Write down what swapping actually means in words! And is this useful for practise?
# #       You still need to learn the mapping from y to z, which is the most expensive part anyway.

# fig, axs = plt.subplots(nrows=1, ncols=4)
# axs[0].bar(np.arange(N_KINEMATICS), baseline_a)
# axs[1].bar(np.arange(N_KINEMATICS), baseline_b)
# axs[2].bar(np.arange(N_KINEMATICS), cc_ab_a)
# axs[3].bar(np.arange(N_KINEMATICS), cc_ba_b)

# axs[0].set_title('Baseline A')
# axs[1].set_title('Baseline B')
# axs[2].set_title('Swapped\ninserted Cz_b to model A\nRepredicted y_a')
# axs[3].set_title('Swapped\ninserted Cz_a to model B')
# axs[0].set_ylabel('Reconstruction CC')
# axs[0].set_xticks(np.arange(N_KINEMATICS))
# axs[1].set_xticks(np.arange(N_KINEMATICS))
# axs[2].set_xticks(np.arange(N_KINEMATICS))
# axs[3].set_xticks(np.arange(N_KINEMATICS))
# axs[0].set_xlabel('kinematics')

# fig.show()




print()