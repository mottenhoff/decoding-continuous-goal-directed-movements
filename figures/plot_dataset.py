import matplotlib.pyplot as plt
import numpy as np

import cmcrameri.cm as cmc

np.random.seed(seed=6227836)

def plot_random_selection_per_trial(ds, num, savepath):

    colors = ['blue', 'orange', 'green']
    colors = [cmc.batlow(64), cmc.batlow(128), cmc.batlow(192)]
    ylabels = ['X', 'Y', 'Z']

    # random_channels = np.random.randint(0, ds.channels.size, 5)
    random_channels = [33, 51, 86, 15, 53]  # TODO: get from seed.
    # print(random_channels)

    xyz = ds.xyz[:, :3]
    has_xyz_value = ~np.isnan(ds.trials[:, 0])
    
    eeg = ds.eeg[:, random_channels]
    eeg_ts = ds.ts[has_xyz_value]
    eeg_ts = ds.ts - eeg_ts[0]
    
    fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(16, 9))

    for i, ax in enumerate(axs[:5]):
        
        ax.plot(eeg[:, i], linewidth=0.1, color='k')
        
        ax.axhline(0, linewidth=0.1, linestyle='dotted')

        ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel(f'EEG {random_channels[i]}', rotation=0, fontsize='x-large')
        ax.set_xticks([])


    for i, ax in enumerate(axs[5:]):
        
        # xyz_ts = ds.xyz_timestamps
        # xyz_ts = xyz_ts - xyz_ts[0]
        ax.plot(xyz[:, i], color=colors[i])
        # ax.axhline(0, linewidth=0.1, linestyle='dotted')

        ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel(ylabels[i], rotation=0, fontsize='xx-large')
        
        if i == 2:
            ax.set_xlabel('time [s]', fontsize='xx-large')
            
        else:    
            ax.set_xticks([])

    fig.tight_layout()
    fig.savefig(savepath/f'bubble_example_{num}.png')

    return

def plot_subsets(subsets, savepath):
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


            fig.savefig(savepath/f'tmp{i}{metric}.png')

    fig.tight_layout()
    # fig.savefig(f'figure_output/{ds.ppt_id}_bubble_example.png')