from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import pearsonr

CC = 0

POS, SPEED, ACC = 9, 10, 11

def load(path):

    scores =  np.load(path/'metrics.npy')  # CC, R2, MSE, RMSE
    # z =  np.load(path/'z.npy')
    # y =  np.load(path/'y.npy')
    # zh = np.load(path/'trajectories.npy')
    # yh = np.load(path/'neural_reconstructions.npy')    
    # xh = np.load(path/'latent_states.npy')

    scores = scores.squeeze()[:, CC, :]
    
    info_path = Path('/'.join(path.parts[:-1]))/'info.yml'
    
    with open(info_path) as f:
        run_info = yaml.load(f, Loader=yaml.FullLoader)

    n_gaps = run_info['n_gaps']

    return scores, n_gaps


def plot_relationship(paths, ax=None):

    # if not ax:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))

    scores = [load(path) for path in paths]
    
    for i, metric in enumerate([POS, SPEED, ACC]):
        mean = np.array([(score[0].mean(axis=0)[metric], score[1]) for score in scores])
        cc = pearsonr(mean[:, 1], mean[:, 0])
        a, b = np.polyfit(mean[:, 1], mean[:, 0], 1)
        coords = [[mean[:, 1].min(), mean[:, 1].max()],
                  [b + a * mean[:, 1].min(), b + a * mean[:, 1].max()]]
        ax[i].plot(coords[0], coords[1], color='black', linewidth=1)

        ax[i].annotate(f'r={cc.statistic:.2f}, p={cc.pvalue:.3f}', 
                    (coords[0][1]-0.2, coords[1][1]),
                    (coords[0][1]-0.2, coords[1][1]+0.3), ha='right')

        ax[i].scatter(mean[:, 1], mean[:, 0])
        ax[i].set_xlabel('Gaps in dataset')
        ax[i].set_ylabel('Decoding Correlation')
        # ax[i].set_aspect('equal')
        ax[i].set_ylim(0, 1)
        ax[i].spines[['top', 'right']].set_visible(False)

    
    ax[0].set_title('Position')
    ax[1].set_title('Speed')
    ax[2].set_title('Acceleration')

    fig.tight_layout()
    fig.savefig('figure_output/gaps_vs_performance.png')

    return 