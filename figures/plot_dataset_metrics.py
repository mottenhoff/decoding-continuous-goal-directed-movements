import pickle
from pathlib import Path
from random import random
from collections import defaultdict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
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

def plot_histograms(ax, ppt_times, median_time, colors):

    # Ax 0: Histograms
    # Reversed the enumerate to make the colors look nicer :)
    for i, (ppt, times) in list(enumerate(ppt_times.items()))[::-1]:
        ax.hist(times, label=ppt, bins=100, color=colors[i])
    
    ax.axvline(median_time, color='black', linestyle='dashed')

    ax.annotate(f'{median_time:.1f}s', xy=(median_time+46, -15), ha='center',
                    xycoords='axes points', fontsize='large')
    ax.annotate('Median time', xy=(median_time, 1.01), ha='center',
                    xycoords=('data', 'axes fraction'), fontsize='small') 

    ax.semilogx()
    ax.set_xlabel('Time [log(s)])')    
    ax.set_ylabel('Count')
    ax.spines[['top', 'right']].set_visible(False)

    return ax

def plot_violins(ax, ppt_times, cmap):

    # Ax 1: Violins
    times = np.concatenate([time for time in ppt_times.values()])

    for i, opt in enumerate([{'showmedians':  True}, {'showmeans': True}]):
    
        vp = ax.violinplot(times, positions=[i+1], showextrema=False, **opt)

        for body in vp['bodies']:
            body.set_color(cmap(i*85))
            # body.set_facecolor(cmap(i*85))
            # body.set_edgecolor(cmap(i*85))
            body.set_alpha(0.65)
        
        # vp['cbars'].set_color(cmap(i*85))
        vp['cmedians' if i==0 else 'cmeans'].set_color(cmap(i*85))

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Median\n{np.median(times):.2f}+-{np.std(times):.2f}s',
                            f'Mean\n{np.mean(times):.2f}+-{np.std(times):.2f}s'])
    ax.set_ylabel('Time [s]')
    
    ax.spines[['top', 'right']].set_visible(False)
    return ax

def plot_medium_times(ax, ppt_times):
    # Median times
    median_times = [np.median(time) for time in ppt_times.values()]

    ax.bar(0, np.mean(median_times), yerr=np.std(median_times), zorder=0, color=cmap(85), alpha=0.75)
    ax.scatter(np.zeros(len(median_times)), median_times, s=25, zorder=1, color='black')

    ax.set_xlim(-1, 1)
    ax.spines[['top', 'left', 'right']].set_visible(False)
    ax.set_xticks([])
    ax.set_ylabel('Median time to target')

def plot_average_time_to_target(path):

    ppt_times = {path.parents[1].stem: np.load(path) 
                for path in path.rglob('time_between_targets.npy')}

    median_time = np.median(np.hstack(list(ppt_times.values())))

    colors = [cmap(int(i)) for i in np.linspace(0, 255, len(ppt_times))]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    axs[0] = plot_histograms(axs[0], ppt_times, median_time, colors)
    axs[1] = plot_violins(axs[1], ppt_times, cmap)
    axs[2] = plot_medium_times(axs[2], ppt_times)

    fig.suptitle('Time to target')
    fig.tight_layout()
    fig.savefig('./figure_output/time_to_target.png')
    fig.savefig('./figure_output/time_to_target.svg')

def plot_average_trajectory_hist(main_path):
    # pos, speed, force = [9, 10, 11]

    paths = list(main_path.rglob('behavior_per_trial*'))

    ppt_ids = sorted(set([path.parts[-3] for path in paths]))

    trials = []

    for ppt in ppt_ids:

        ppt_trials = []
        for path in paths:
            
            if ppt not in str(path):
                continue
            
            file_trials = []
            with open(path, 'rb') as f:
                subset_trials = pickle.load(f)
                file_trials += subset_trials
                
            ppt_trials += [abs(t[-1, POS] - t[0, POS]) for t in file_trials]
        
        if ppt_trials:
            trials += [ppt_trials]
    
    colors = [cmap(int(i)) for i in np.linspace(0, 255, len(ppt_ids))]

    # Pemute order for color readabilit
    # new_order = np.random.permutation(len(ppt_ids))
    # trials = [trials[i] for i in new_order]
    # colors = [colors[i] for i in new_order]
    # ppt_ids = [ppt_ids[i] for i in new_order]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(trials, bins=10, stacked=True, color=colors, label=ppt_ids)
    
    ax.set_xlabel('Distance [mm]', fontsize='xx-large')
    ax.set_ylabel('Count [trials]', fontsize='xx-large')
    ax.set_title('Distance from start to end of trial', fontsize='xx-large')
    ax.spines[['top', 'right']].set_visible(False)
    # fig.legend()
    
    fig.tight_layout()
    fig.savefig(r'figure_output/distance_per_trials.png')
    fig.savefig(r'figure_output/distance_per_trials.svg')


# def plot_average_trajectory(main_path):
#     # pos, speed, force = [9, 10, 11]
#     paths = list(main_path.rglob('behavior_per_trial*'))

#     ppt_ids = sorted(set([path.parts[-3] for path in paths]))

#     trials = []

#     for ppt in ppt_ids:

#         ppt_trials = []
#         for path in paths:
            
#             if ppt not in str(path):
#                 continue
            
#             file_trials = []
#             with open(path, 'rb') as f:
#                 subset_trials = pickle.load(f)
#                 file_trials += subset_trials
                
#             ppt_trials += [t[:, SPEED] for t in file_trials]
#             # ppt_trials += [abs(t[-1, POS] - t[0, POS]) for t in file_trials]
        
#         if ppt_trials:
#             trials += [ppt_trials]
       
#         fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

#         for trial in trials:

#             trial -= trial[0, :]
#             trial = np.abs(trial)

#             for iax, ax in enumerate(axs, start=POS):
#                 x = np.linspace(0, 100, trial.shape[0])  # normalized to percentage of trial
#                 ax.plot(x, trial[:, iax], linewidth=1, color=cmap(random()))
            
#         axs[0].set_xlabel('Progression in trial [%]')
#         axs[0].set_ylabel('Position [mm]')

#         axs[1].set_xlabel('Progression in trial [%]')
#         axs[1].set_ylabel('Speed [mm/s]')

#         axs[2].set_xlabel('Progression in trial [%]')
#         axs[2].set_ylabel(r'Acceleration $mm/s^2$')

#         for ax in axs:
#             ax.spines[['top', 'left', 'right']].set_visible(False)
 
#         fig.tight_layout()
#         outpath = Path(r'figure_output/trial_trajectories/')
#         outpath.mkdir(exist_ok=True, parents=True)
#         fig.savefig(outpath/f'trials_{ppt_id}_1.png')
#         fig.savefig(outpath/f'trials_{ppt_id}_1.svg')

#     return
 
def plot_speed_curve(main_path):
    # pos, speed, force = [9, 10, 11]

    paths = list(main_path.rglob('behavior_per_trial*'))


    data_per_ppt = defaultdict(list)
    for path in paths:
        ppt = str(path.parts[-3])
        
        with open(path, 'rb') as f:
            data_per_ppt[ppt].append(pickle.load(f))

    # Flatten lists and find max value
    data_per_ppt = {ppt: list(chain.from_iterable(trials)) for ppt, trials in data_per_ppt.items()}

    # upsample and average per participant

    mean_trial_per_ppt = {}
    for ppt, trials in data_per_ppt.items():
        max_len = max([trial.shape[0] for trial in trials])
        upsampled_trials = []
        for trial in trials:
            trial_len, _ = trial.shape
            upsampled_trial = np.interp(np.arange(max_len),
                                        np.arange(trial_len),
                                        trial[:, SPEED])
            upsampled_trials += [upsampled_trial]

        mean_trial = np.vstack(upsampled_trials).mean(axis=0)
        mean_trial_per_ppt[ppt] = mean_trial
        

    colors = [cmap(int(i)) for i in np.linspace(0, 255, len(mean_trial_per_ppt.keys()))]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    for i, ppt in enumerate(sorted(mean_trial_per_ppt)):
        trial = mean_trial_per_ppt[ppt]
        trial = (trial-trial.mean()) / trial.std()
        
        trial = np.array([(trial[i:i+300]).mean() for i in np.arange(0, trial.size-299, 50)])  # window: 300 len, 50 shift, same as during decoding

        x = np.linspace(0, 100, trial.shape[0])
        ax.plot(x, trial, color=colors[i], label=ppt)

    ax.set_xlabel('Progression through trial [%]', fontsize='x-large')
    ax.set_ylabel('Speed [normalized]', fontsize='x-large')
    ax.set_title('Mean movement speed over trials')
    ax.set_xlim(0, 100)
    ax.spines[['top', 'right']].set_visible(False)
    # fig.legend()
    fig.tight_layout()
    
    fig.savefig(r'figure_output/mean_movement_speed_per_trial.png')
    fig.savefig(r'figure_output/mean_movement_speed_per_trial.svg')


    
    