import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

import cmcrameri.cm as cmc

cmap = cmc.batlow

POS, SPEED, ACC = [9, 10, 11]

def load(path):
    # TODO: Again, copy pasted from figures_1d_score_overview.py
    m =  np.load(path/'metrics.npy')  # CC, R2, MSE, RMSE
    z =  np.load(path/'z.npy')
    # y =  np.load(path/'y.npy')
    # zh = np.load(path/'trajectories.npy')
    # yh = np.load(path/'neural_reconstructions.npy')    
    # xh = np.load(path/'latent_states.npy')

    return m, z, # y, zh, yh, xh

def resample(arr, n_samples):
    
    if arr.shape[0]< n_samples:
        raise Exception('Cant resample, n_samples < samples in array')

    new_idx = np.linspace(0, arr.shape[0]-1, n_samples).astype(int)

    return arr[new_idx, :]

def plot_average_time_to_target(paths):

    ppt_times = {path.parents[1].stem: np.load(path.parents[0]/'time_between_targets.npy') 
                for path in paths}

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    colors = [cmap(int(i)) for i in np.linspace(0, 255, len(ppt_times))]

    # Ax 2: Histograms

    for i, (ppt, times) in enumerate(ppt_times.items(), start=0):
        axs[0].hist(times, label=ppt, bins=100, color=colors[i])
    
    axs[0].semilogx()
    axs[0].set_xlabel('Time [log(s)])')    
    axs[0].set_ylabel('Count')
    axs[0].spines[['top', 'right']].set_visible(False)

    # Ax 1: Violins

    times = np.concatenate([time for time in ppt_times.values()])
    
    for i, opt in enumerate([{'showmedians':  True}, {'showmeans': True}]):
    
        vp = axs[1].violinplot(times, positions=[i+1], showextrema=False, **opt)

        for body in vp['bodies']:
            body.set_color(cmap(i*85))
            # body.set_facecolor(cmap(i*85))
            # body.set_edgecolor(cmap(i*85))
            body.set_alpha(0.65)
        
        # vp['cbars'].set_color(cmap(i*85))
        vp['cmedians' if i==0 else 'cmeans'].set_color(cmap(i*85))

    axs[1].set_xticks([1, 2])
    axs[1].set_xticklabels([f'Median\n{np.median(times):.2f}+-{np.std(times):.2f}s',
                            f'Mean\n{np.mean(times):.2f}+-{np.std(times):.2f}s'])
    axs[1].set_ylabel('Time [s]')
    
    axs[1].spines[['top', 'right']].set_visible(False)


    # Median times
    median_times = [np.median(time) for time in ppt_times.values()]

    axs[2].bar(0, np.mean(median_times), yerr=np.std(median_times), zorder=0, color=cmap(85), alpha=0.75)
    axs[2].scatter(np.zeros(len(median_times)), median_times, s=25, zorder=1, color='black')

    axs[2].set_xlim(-1, 1)
    axs[2].spines[['top', 'left', 'right']].set_visible(False)
    axs[2].set_xticks([])
    axs[2].set_ylabel('Median time to target')
    
    fig.suptitle('Time to target')
    fig.tight_layout()
    fig.savefig('./figure_output/time_to_target.png')
    

    return

def plot_average_trajectory(paths):
    # pos, speed, force = [9, 10, 11]
    suspicious_trials = {}    
    for path in paths:
        ppt_id = path.parents[1].stem
        # m, z = load(paths[0]) 
        exclude_by_hand = {'kh049': [0]}
        exclude_by_hand = []

        with open(f'finished_runs/window/bbhg/{ppt_id}/0/behavior_per_trial.pkl', 'rb') as f:
            trials = pickle.load(f)

        for to_exclude in exclude_by_hand:
            trials.pop(to_exclude)
        

        suspicious_trials[ppt_id] = [i for i, trial in enumerate(trials) if trial[0, SPEED] > 100]

        trials = [trial for trial in trials if trial[0, SPEED] < 100]

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))

        for trial in trials:

            trial -= trial[0, :]

            for iax, ax in enumerate(axs, start=POS):
                x = np.linspace(0, 100, trial.shape[0])  # normalized to percentage of trial
                ax.plot(x, trial[:, iax], linewidth=1)
            
        axs[0].set_title('Position')
        axs[1].set_title('Speed')
        axs[2].set_title('Acceleration')

        fig.savefig(f'trials_{ppt_id}_1.png')

        continue

        # Resample trials
        n_samples = 1000
        x = np.linspace(0, 100, n_samples)

        trials = np.dstack([resample(trial, n_samples) for trial in trials])

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))

        for iax, ax in enumerate(axs, start=POS):
            trial_kinematic = np.abs(trials[:, iax, :])

            if iax==2:
                trial_kinematic = np.log(trial_kinematic)

            ax.plot(x, trial_kinematic, linewidth=1, color='black')
            ax.plot(x, trial_kinematic.mean(axis=-1), linewidth=3, color='orange')
        
        fig.savefig(f'trials_{ppt_id}_2.png')
        

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))

        for iax, ax in enumerate(axs, start=POS):
            trial_kinematic = np.abs(trials[:, iax, :])

            # if iax==2:
            #     trial_kinematic = np.log(trial_kinematic)
            trial_mean = trial_kinematic.mean(axis=-1)
            trial_std = trial_kinematic.std(axis=-1)

            ax.plot(np.arange(trial_mean.size), trial_mean, linewidth=2, color='black')
            ax.fill_between(np.arange(trial_mean.size), trial_mean-trial_std, trial_mean+trial_std, color='grey')

            # ax.plot(x, trial_kinematic.mean(axis=-1), linewidth=3, color='orange')
        
        fig.savefig(f'trials_{ppt_id}_3.png')


    print(suspicious_trials)
    # fig, ax = plt.subplots(figsize=(8, 8))


    

    return

def plot_target_distributions(paths):

    return

def all(paths):
    # plot_average_time_to_target(paths)
    plot_average_trajectory(paths)
    return