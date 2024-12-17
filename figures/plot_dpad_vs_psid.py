from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
from libs import maps

CONDITION, FOLDS, METRICS, KINEMATICS = 0, 1, 2, 3
CC = 0
RX, RY, RZ = 1, 2, 3
VX, VY, VZ = 4, 5, 6
AX, AY, AZ = 7, 8, 9
DISTANCE, SPEED, ACCELERATION = 10, 11, 12

N_KINEMATICS = 12

def plot(results_dpad, results_psid, ax=None):

    ppts = sorted(list(results_dpad.keys()))
    cmap = cm.batlow
    colors = [cmap(int(i)) for i in np.linspace(0, 255, len(ppts))]

    dpad_scores = {ppt: values['scores'] for ppt, values in results_dpad.items()}
    psid_scores = {ppt: values['scores'] for ppt, values in results_psid.items()}

    # ppt_in_both = set(dpad_scores.keys()) & set(psid_scores.keys())
    
    mean_scores = {}
    for ppt in ppts:
        mean_scores[ppt] = [psid_scores[ppt].mean(axis=(0, 1))[CC],
                            dpad_scores[ppt].mean(axis=(0, 1))[CC]]
    mean_scores = {}
    for ppt in ppts:
        mean_scores[ppt] = [np.median(psid_scores[ppt], axis=(0, 1))[CC],
                            np.median(dpad_scores[ppt], axis=(0, 1))[CC]]

    if not ax:
        _, ax = plt.subplots(nrows=1, ncols=1)
    
    for ppt, score in mean_scores.items():
        ax.scatter(score[0][SPEED], score[1][SPEED], label=ppt, color=colors[ppts.index(ppt)])

    ax.plot([0, 1], [0, 1], linestyle='dashed', linewidth=1, color='black')

    ax.set_xlabel('PSID', fontsize='x-large')
    ax.set_ylabel('DPAD', fontsize='x-large')
    ax.set_title('mean decoding performance')
    
    return ax