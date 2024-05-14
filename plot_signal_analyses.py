from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from libs.load import load_dataset

SPEED = 10

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
        # Load
        y = np.load(ppt_path/'0/y.npy')
        task_correlations = np.load(ppt_path/'0'/'task_correlations.npy')
        locations = load_csv(ppt_path.name)
        recorded_channels = np.load(ppt_path/'0'/'recorded_channel_names.npy')

        try:
            selected_channels = np.load(ppt_path/'0'/'selected_channels.npy')
            u, c = np.unique(selected_channels.ravel(), return_counts=True)
            selected_channels = u[np.argsort(c)[-30:]]
        except Exception:
            selected_channels = np.empty(0)


        # Check some stuff
        if 'EE1' in recorded_channels: print(f'Error channels! {ppt}')

        # Calculate
        c = np.corrcoef(y.T)

        print(c)
        idc = np.arange(c.shape[0])

        c[idc, idc] = task_correlations[kinematic, idc]

        

        if not type(locations) == pl.dataframe.frame.DataFrame:
            channel_names = np.arange(c.shape[0])
        else:
            channel_names = recorded_channels
            # channel_names = locations['electrode_name_1'].to_list()
            # anatomical_locations = locations['location'].to_list()

        fig, ax = plt.subplots(figsize=(16, 16))

        ax.imshow(c, cmap='coolwarm', vmin=-1, vmax=1)

        ax.set_xticks(np.arange(len(channel_names)))
        ax.set_yticks(np.arange(len(channel_names)))
        ax.set_xticklabels(channel_names, fontsize='xx-small', rotation=90)
        ax.set_yticklabels(channel_names, fontsize='xx-small')

        if selected_channels.size > 0:
            for idx in selected_channels:
                ax.get_xticklabels()[idx].set_color('red')
                ax.get_yticklabels()[idx].set_color('red')

        ax.set_title(f'{ppt}\ndiagonal (top-left to bottom-right) =task correlation')

        fig.savefig(outpath/f'icc_{ppt}.png')
        plt.close(fig=fig)

    return


def plot_all(main_path):
    
    interchannel_correlation(main_path)

if __name__=='__main__':
    main_path = Path(r'finished_runs\delta_lap')
    # main_path = Path(r'finished_runs\alphabeta')
    # main_path = Path(r'finished_runs\bbhg')
    plot_all(main_path)