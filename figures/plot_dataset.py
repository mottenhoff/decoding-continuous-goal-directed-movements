import matplotlib.pyplot as plt
import numpy as np

import cmcrameri.cm as cmc

def plot_random_selection_per_trial(ds):

    colors = ['blue', 'orange', 'green']
    colors = [cmc.batlow(64), cmc.batlow(128), cmc.batlow(192)]
    ylabels = ['X', 'Y', 'Z']

    random_channels = np.random.randint(0, ds.eeg.channels.size, 5)
    random_channels = [33, 51, 86, 15, 53]
    print(random_channels)

    eeg = ds.eeg.timeseries[:, random_channels]
    eeg_ts = ds.eeg.timestamps
    eeg_ts = ds.eeg.timestamps - eeg_ts[0]
    
    xyz = ds.xyz[:, :3]
    has_xyz_value = ~np.isnan(ds.xyz[:, 0])
    
    # trials = ds.trials[~np.isnan(ds.trials[:, 0]), :]
    trial_idc = [list(ds.trials[:, 0]).index(float(i)) for i in range(50)]


    fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(16, 9))

    for i, ax in enumerate(axs[:5]):
        
        ax.plot(eeg_ts, eeg[:, i], linewidth=0.1, color='k')
        
        ax.axhline(0, linewidth=0.1, linestyle='dotted')

        ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel(f'EEG {i+1}', rotation=0, fontsize='x-large')
        ax.set_xticks([])

        for j, idx in enumerate(trial_idc, start=1):
            ax.axvline(eeg_ts[idx], color=cmc.lajolla(128), linestyle='dashed', linewidth=0.5)

            if i == 0:
                ax.annotate(f"{'Trial ' if j == 1 else ''}{j}",
                            (eeg_ts[idx], 1.1), xycoords=('data', 'axes fraction'),
                            ha='center')  # fontsize='x-small'

    for i, ax in enumerate(axs[5:]):
        
        xyz_ts = ds.xyz_timestamps
        xyz_ts = xyz_ts - xyz_ts[0]
        ax.plot(xyz_ts, xyz[has_xyz_value, i], color=colors[i])
        # ax.axhline(0, linewidth=0.1, linestyle='dotted')

        ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel(ylabels[i], rotation=0, fontsize='xx-large')
        
        if i == 2:
            ax.set_xlabel('time [s]', fontsize='xx-large')
            
        else:    
            ax.set_xticks([])

        for idx in trial_idc:
            ax.axvline(eeg_ts[idx], color=cmc.lajolla(128), linestyle='dashed', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(f'figure_output/{ds.ppt_id}_bubble_example.png')

def plot_subsets(subsets, save_path):
    np.random.seed(1337)

    colors = [cmc.batlow(64), cmc.batlow(128), cmc.batlow(192), cmc.batlow(255)]
    ylabels = ['X', 'Y', 'Z', 'SUM']

    n_eeg = 3
    n_timestamps = 0.1  # % of total timestamps

    # fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(16, 9))
    fig, axs = plt.subplots(nrows=n_eeg+3+1, ncols=1, figsize=(16, 9))

    for j, ss in enumerate(subsets):
        random_channels = np.random.randint(0, ss.eeg.shape[1], n_eeg)
        print(random_channels)
        eeg = ss.eeg[:, random_channels]
        ts  = ss.ts
        ts =  np.linspace(0, ts[-1]-ts[0], eeg.shape[0])
        xyz = ss.xyz

        n_ts = int(n_timestamps * ts.size)
        for i, ax in enumerate(axs[:n_eeg]):
            
            ax.plot(ts[:n_ts], eeg[:n_ts, i], linewidth=0.1, color='k')
            
            ax.axhline(0, linewidth=0.1, linestyle='dotted')

            ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
            ax.set_yticks([])
            ax.set_ylabel(f'EEG {i+1}', rotation=0, fontsize='x-large')
            ax.set_xticks([])

        for i, ax in enumerate(axs[n_eeg:]):
            
            for metric in [8]: #0, 4, 8]:

                ax.plot(ts[:n_ts], xyz[:n_ts, i+metric], color=colors[i], linewidth=1)

                ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
                ax.set_yticks([])
                ax.set_ylabel(ylabels[i], rotation=0, fontsize='xx-large')
                

            if i == 3:
                ax.set_xlabel('time [s]', fontsize='xx-large')
                
            else:    
                ax.set_xticks([])


            fig.savefig(f'tmp{i}{metric}.png')

    fig.tight_layout()
    # fig.savefig(f'figure_output/{ds.ppt_id}_bubble_example.png')

    # for i in range(xyz.shape[1]):
    #     plt.figure()
    #     plt.plot(xyz[:100, i])
    #     plt.savefig(f'tmp{i}.png')



def plot_dataset(ds):
    # plot_random_selection_per_trial(ds)
    # plot_average_time_to_target(ds)
    return






        # if c.checks.trials_vs_cont:
        #     checks.data_size_trial_vs_continuous(trials[:, 0], xyz)


