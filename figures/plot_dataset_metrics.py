import sys
sys.path.append(r'~/main/resources/code/')

import pickle
from pathlib import Path
from random import random

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

def plot_average_time_to_target(path, savepath):

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
    fig.savefig(savepath/'time_to_target.png')
    fig.savefig(savepath/'time_to_target.svg')


def plot_average_trajectory(main_path, savepath):
    # pos, speed, force = [9, 10, 11]

    paths = list(main_path.rglob('behavior_per_trial*'))

    ppt_ids = set([path.parts[-3] for path in paths])

    for ppt_id in ppt_ids:
        
        ppt_paths = [path for path in paths if ppt_id in path.parts]
        
        trials = []
        for path in ppt_paths:
            
            with open(path, 'rb') as f:

                subset_trials = pickle.load(f)
                trials += subset_trials
        
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        for trial in trials:

            trial -= trial[0, :]
            trial = np.abs(trial)

            for iax, ax in enumerate(axs, start=POS):
                x = np.linspace(0, 100, trial.shape[0])  # normalized to percentage of trial
                ax.plot(x, trial[:, iax], linewidth=1, color=cmap(random()))
            
        axs[0].set_xlabel('Progression in trial [%]')
        axs[0].set_ylabel('Position [mm]')

        axs[1].set_xlabel('Progression in trial [%]')
        axs[1].set_ylabel('Speed [mm/s]')

        axs[2].set_xlabel('Progression in trial [%]')
        axs[2].set_ylabel(r'Acceleration $mm/s^2$')

        for ax in axs:
            ax.spines[['top', 'left', 'right']].set_visible(False)
 
        fig.tight_layout()
        outpath = Path(savepath/'trial_trajectories/')
        outpath.mkdir(exist_ok=True, parents=True)
        fig.savefig(outpath/f'trials_{ppt_id}_1.png')
        fig.savefig(outpath/f'trials_{ppt_id}_1.svg')

    return
 
def plot_speed_curve(main_path, savepath):
    # pos, speed, force = [9, 10, 11]

    paths = list(main_path.rglob('behavior_per_trial*'))

    ppt_ids = set([path.parts[-3] for path in paths])

    trials = []
    for ppt_id in ppt_ids:
        
        ppt_paths = [path for path in paths if ppt_id in path.parts]
        
        for path in ppt_paths:
            
            with open(path, 'rb') as f:

                subset_trials = pickle.load(f)
                trials += subset_trials

        # distances = np.stack([np.abs(t[-1] - t[0]) for t in trials])

        # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        # trials = np.stack([scipy.signal.resample(trial, 100, axis=0) for trial in trials])
        # mean = trials[:, :, SPEED].mean(axis=0)
        # std = trials[:, :, SPEED].std(axis=0)

        # x = np.linspace(0, 100, 100)
        # for iax, ax in enumerate(axs, start=POS):
        #     ax.plot(x, trials[:, :, iax].T, linewidth=1)
            


        # for trial in trials:

        #     trial -= trial[0, :]
        #     trial = np.abs(trial)

        #     for iax, ax in enumerate(axs, start=POS):
        #         x = np.linspace(0, 100, trial.shape[0])  # normalized to percentage of trial
        #         ax.plot(x, trial[:, iax], linewidth=1, color=cmap(random()))

        # axs[1].plot(mean)
        # axs[1].fill_between(np.linspace(0, 100, 100), mean-std, mean+std, alpha=.4)
            
        # axs[0].set_xlabel('Progression in trial [%]')
        # axs[0].set_ylabel('Position [mm]')

        # axs[1].set_xlabel('Progression in trial [%]')
        # axs[1].set_ylabel('Speed [mm/s]')

        # axs[2].set_xlabel('Progression in trial [%]')
        # axs[2].set_ylabel(r'Acceleration $mm/s^2$')

        # for ax in axs:
        #     ax.spines[['top', 'left', 'right']].set_visible(False)
 
        # fig.tight_layout()
        # outpath = Path(r'figure_output/trial_trajectories/')
        # outpath.mkdir(exist_ok=True, parents=True)
        # plt.show(block=True)
        # print()
        # fig.savefig(outpath/f'trials_{ppt_id}_1.png')
        # fig.savefig(outpath/f'trials_{ppt_id}_1.svg')

    