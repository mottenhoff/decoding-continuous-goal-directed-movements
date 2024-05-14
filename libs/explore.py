import numpy as np
import matplotlib.pyplot as plt

from cmcrameri import cm

cmap = cm.batlow

def task_correlations(datasets, savepath):
    
    y = np.vstack([s.eeg for s in datasets])
    z = np.vstack([s.xyz for s in datasets])

    cm = np.corrcoef(np.hstack([y, z]), rowvar=False)
    task_correlations = cm[-z.shape[1]:, :y.shape[1]]

    task_correlations = np.vstack([datasets[0].channels, task_correlations]) # first row is the corresponding channel_names

    with open(savepath/'task_correlations.npy', 'wb') as f:
        np.save(f, task_correlations)

def plot_trajectory(datasets, savepath, all_sets=True):

    # TODO: z and target seems to be in differnt coordinate spaces.

    if all_sets:
        targets = np.vstack([np.unique(subset.trials[~np.isnan(subset.trials[:, 0])], axis=0)
                            for subset in datasets])
        z = np.vstack([subset.xyz[:, :3] for subset in datasets])
    else:
        targets = np.unique(datasets[0].trials[~np.isnan(datasets[0].trials[:, 0]), :], axis=0)
        z = datasets[0].xyz[:, :3]
        

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    # Switched Y and Z around for the correct perspective. Labels as well
    ax.plot(z[:, 0], z[:, 2], z[:, 1], color=cmap(0))  
    ax.scatter(targets[:, 1], targets[:, 3], targets[:, 2], color=cmap(0.62), s=200)

    ax.set_xlabel('X [left - right]', fontsize='xx-large')
    ax.set_ylabel('Z [front - back]', fontsize='xx-large')
    ax.set_zlabel('Y [down - up]',    fontsize='xx-large')

    # fig.savefig('tmp.png')
    fig.savefig(savepath/'trajectories_with_targets.png')
    fig.savefig(savepath/'trajectories_with_targets.svg')

def main(datasets, savepath):
    task_correlations(datasets, savepath)
    # plot_trajectory(datasets, savepath)
    plt.close('all')
    


    return