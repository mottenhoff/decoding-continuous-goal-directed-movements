from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
import matplotlib as mpl

from cmcrameri import cm

DIST, SPEED, ACC = 9, 10, 11

CC = 0

def load(path):
    # TODO: Again, copy pasted from figures_1d_score_overview.py
    m =  np.load(path/'metrics.npy')  # CC, R2, MSE, RMSE
    z =  np.load(path/'z.npy')
    y =  np.load(path/'y.npy')
    zh = np.load(path/'trajectories.npy')
    yh = np.load(path/'neural_reconstructions.npy')    
    xh = np.load(path/'latent_states.npy')

    return m, z, y, zh, yh, xh

def plot_selected_states(ax, paths):
    # ax = [ax1, ax2]

    scores  =  np.array([np.load(path/'metrics.npy')[0, :, CC, :].mean(axis=0)  # Selects mean correlation
                         for path in paths])
    n_states = np.array([list(map(int, path.stem.split('_')[1:]))               # Selects optimal behaviorally relevant states and size of horizon
                         for path in paths])
    ppts = np.array([path.parts[-3] for path in paths])



    # cmap = cm.batlow
    # # cmap = mpl.cm.get_cmap('inferno')

    # sc = ax.scatter(n_states[:, 0], n_states[:, 1], s=scores[:, SPEED]*1000, c=scores[:, SPEED], cmap=cmap)
    # ax.set_xlabel('Behaviorally relevant states')
    # ax.set_ylabel('Horizon')

    # plt.colorbar(sc)

    # plt.savefig('tmp.png')


    # possible_states = np.concatenate([[3], np.arange(5, 31, 5)])
    # offset = 0.8

    # for state in possible_states:
    #     idc = np.where(n_states[:, 0] == state)[0]
        
    #     if idc.size == 0:
    #         continue

    #     local_idc = np.arange(idc.size)
    #     local_idc = (local_idc - (local_idc[-1] - local_idc[0]) / 2) * offset
        
    #     values = scores[idc, SPEED]

    #     x_pos = local_idc + state
    #     ax.bar(x_pos, values, width=offset)

    # ax.spines[['top', 'right']].set_visible(False)
    # ax.set_xticks(possible_states)
    # ax.set_xlabel('N behaviorally relevant states')
    # ax.set_ylabel('Reconstruction correlation')
    # ax.set_title('Selected states for Speed using Delta activity')

    # plt.savefig('tmp.png')


    ax.violinplot(np.vstack([scores[:, SPEED], n_states[:, 0]]).T)
    plt.savefig('tmp2.png')


    possible_states = np.concatenate([[3], np.arange(5, 31, 5)])
    offset = 0.8



    for state in possible_states:
        idc = np.where(n_states[:, 0] == state)[0]
        
        if idc.size == 0:
            continue

        # x_pos = np.ones(idc.size) * state

        values = scores[idc, SPEED]

        # local_idc = np.arange(idc.size)
        # local_idc = (local_idc - (local_idc[-1] - local_idc[0]) / 2) * offset
        


        # x_pos = local_idc + state
        ax.violin(values, state) #width=offset)

    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks(possible_states)
    ax.set_xlabel('N behaviorally relevant states')
    ax.set_ylabel('Reconstruction correlation')
    ax.set_title('Selected states for Speed using Delta activity')

    plt.savefig('tmp2.png')





    scores = []
    states = []
    for path in paths:
        scores += [np.load(path/'metrics.npy')[0, :, 0, -1].mean()]
        states += [(int(path.name.split('_')[1]), 
                    int(path.name.split('_')[2]))]
    
    scores = np.hstack([np.vstack(states), np.array(scores)[:, np.newaxis]])

    ppts = sorted(list(maps.ppt_id().keys()))
    colors = [maps.cmap()[maps.ppt_id()[ppt]] for ppt in ppts]

    ax[0].scatter(scores[:, 0], scores[:, 2])
    ax[1].scatter(scores[:, 1], scores[:, 2])
    fig.savefig('./figure_output/states_vs_performance.png')

    fig, ax = plt.subplots()
    xticks = np.arange(len(paths))
    ax.bar(xticks, scores[:, 0], color=colors)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{i+1}' for i, _ in enumerate(ppts)], fontsize='x-large')
    ax.set_ylabel('Behaviorally relevant states', fontsize='x-large')
    # ax.set_title('States required for optimal performance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('./figure_output/states_per_ppt.png')




def main(paths):
    fig, ax = plt.subplots()
    plot_selected_states(ax, paths)





