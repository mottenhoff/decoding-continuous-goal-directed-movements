from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from libs.load import load_dataset
from locations.transform_location_strings import beautify_str

SPEED = 10
N_FOLDS = 5

def load_csv(ppt_id):
    try:
        return pl.read_csv(f'./data/{ppt_id}/electrode_locations.csv')
    except FileNotFoundError:
        return None

def interchannel_correlation(path):
    # TODO Channel names do not seem to be aligned.
    # TODO: Y is missing

    outpath = Path(f'./figure_output/interchannel_correlations/{path.name}')
    outpath.mkdir(exist_ok=True, parents=True)

    kinematic = SPEED

    for ppt_path in path.glob('kh*'):

        ppt = ppt_path.name
 
        if ppt not in ['kh056', 'kh067']:
            continue

        y = np.load(ppt_path/'0/y.npy')
        task_correlations = np.load(ppt_path/'0'/'task_correlations.npy')

        c = np.corrcoef(y.T)
        idc = np.arange(c.shape[0])
        c[idc, idc] = task_correlations[kinematic, idc]

        recorded_channels = np.load(ppt_path/'0'/'recorded_channel_names.npy')
        locations = load_csv(ppt_path.name)
        selected_channels = np.array(
            [np.load(f'finished_runs\delta_cer\{ppt}\\0\{i}\selected_channels.npy')
            for i in range(N_FOLDS)]
        )
        selected_channels, counts = np.unique(selected_channels.ravel(), return_counts=True)
        selected_channels = selected_channels[np.argsort(counts)[-30:]] # selects the 30 most selected channels over folds.

        if type(locations) == pl.dataframe.frame.DataFrame:
            anatomical_label_map = dict(zip(locations['electrode_name_1'], locations['location']))

            channel_names = recorded_channels
            channel_labels = [anatomical_label_map[ch] for ch in recorded_channels]
            channel_labels = [beautify_str(ch) for ch in channel_labels]
        else:
            channel_names = recorded_channels
            channel_labels = channel_names
        

        fig, ax = plt.subplots(figsize=(16, 16))

        im = ax.imshow(c, cmap='coolwarm', vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(channel_names)))
        ax.set_yticks(np.arange(len(channel_names)))
        ax.set_xticklabels(channel_labels, fontsize='x-small', rotation=90)
        ax.set_yticklabels(channel_labels, fontsize='x-small')

        if selected_channels.size > 0:
            for idx in selected_channels:
                ax.get_xticklabels()[idx].set_color('red')
                ax.get_yticklabels()[idx].set_color('red')

        ax.set_title(f'{ppt}\ndiagonal (top-left to bottom-right) =task correlation')
        
        fig.colorbar(im)
        ax.set_aspect('auto')

        fig.tight_layout()
        fig.savefig(outpath/f'icc_{ppt}.png')
        # plt.show()
        plt.close(fig=fig)

    return


def plot_all(main_path):
    
    interchannel_correlation(main_path)

if __name__=='__main__':
    paths = [
        Path(r'finished_runs\delta_cer'),
        Path(r'finished_runs\alphabeta_cer'),
        Path(r'finished_runs\bbhg_cer'),
        Path(r'finished_runs\delta_cer_tv'),
        Path(r'finished_runs\alphabeta_cer_tv'),
        Path(r'finished_runs\bbhg_cer_tv'),
        Path(r'finished_runs\delta_lap'),
        Path(r'finished_runs\alphabeta_lap'),
        Path(r'finished_runs\bbhg_lap'),
    ]

    for path in paths:
        plot_all(path)