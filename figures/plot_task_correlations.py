import sys
from pathlib import Path
from multiprocessing import Pool

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

from libs.transform_location_strings import beautify_str

# Local
PATH_DATA = Path('../data')
KINEMATICS = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'r', 'v', 'a']
SPEED = 10

np.random.seed(2024)

def load(main_path: Path):

    contacts = {}

    for path in main_path.glob('sub-*'):

        ppt_id = path.name

        try:
            contact_set = load_contacts(ppt_id)
        except OSError as error:
            print(error)
            continue

        task_correlations = np.load(path/'0'/'task_correlations.npy')
        task_correlations = {'channels': task_correlations[0,  :],
                             'corrs':    task_correlations[1:, :].astype(np.float64)}

        contact_set.task_correlations = task_correlations
        contacts[ppt_id] = contact_set

    return contacts

def load_chance_level(root_path, percentile=.95):
    # Loads chance levels per participant and returns the 95% percentile of all these percentiles
    # per kinematic

    chance_level_files = root_path.rglob('chance_levels_task_correlation_1000.npy')
    permutations = np.vstack([np.load(filepath).reshape(-1, len(KINEMATICS)) for filepath in chance_level_files])
    
    idx_percentile = int(permutations.shape[0] * percentile)
        
    chance_levels = np.sort(np.abs(permutations), axis=0)[idx_percentile, :]
    chance_levels[:] = chance_levels.max()  # Corrects for multiple testing

    return chance_levels

def load_contacts(ppt_id):

    path = PATH_DATA/ppt_id/'electrode_locations.csv'

    if not path.exists():
        return pl.DataFrame()

    contacts = pl.read_csv(path)

    contacts = contacts.select(['electrode_name_1',
                                'location'])\
                       .rename({'electrode_name_1': 'electrode',
                                'location': 'anatomical_location'})
    
    contacts = contacts.with_columns(ppt = pl.lit(ppt_id))

    return contacts

def load_correlations(ppt_path, kinematic):

    correlations = np.load(ppt_path/'0'/'task_correlations.npy')
    
    return pl.DataFrame([correlations[0,  :],
                           correlations[kinematic+1, :].astype(np.float64)],
                           schema={
                               'electrode': str,
                               'correlations': float
                           },
                           strict=False)

def task_correlations_per_channel(df, chance_level, outpath,
                                  only_significant_channels=False):


    cmap = cm.batlowW
    ppt_ids = sorted(df.get_column('ppt').unique().to_list())
    colors = {ppt: cmap(i) for ppt, i in zip(ppt_ids, np.linspace(0, 0.9, len(ppt_ids)))}

    if only_significant_channels:
        df = df.filter(pl.col('correlations') > chance_level)

    # Sort by average correlation
    unique_locations = df.group_by('anatomical_location') \
                         .mean() \
                         .sort(by='correlations')
    unique_locations = unique_locations.get_column('anatomical_location').to_list()

    # Plot it
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 20))

    for ppt in ppt_ids:
        ppt_df = df.filter(pl.col('ppt')==ppt)

        if ppt_df.is_empty():
            continue

        y_idc = [unique_locations.index(loc) for loc in ppt_df.get_column('anatomical_location')]

        ax.scatter(ppt_df.get_column('correlations'), 
                   y_idc, 
                   s=100,
                   color=colors[ppt],
                   label=ppt)
        
    ax.axvline(chance_level, color='black')

    ax.set_xlim(0, .55)
    ax.tick_params(axis='x', labelsize='x-large')
    ax.set_yticks(np.arange(len(unique_locations)))
    ax.set_yticklabels([beautify_str(loc) for loc in unique_locations], 
                       fontsize='xx-large')

    ax.axvline(0, color='grey', alpha=.5)

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(visible=True, axis='y', alpha=0.5)

    fig.legend(bbox_to_anchor=[1, 0.1],
               fontsize='xx-large', 
               loc='lower right')
    fig.tight_layout()

    outpath = outpath/'only_significant_channels' if only_significant_channels else outpath
    outpath.mkdir(exist_ok=True, parents=True)

    fig.savefig(outpath/'task_correlations_per_location.png')
    fig.savefig(outpath/'task_correlations_per_location.svg')

def plot_condition(path):

    chance_levels = load_chance_level(path, 0.95)

    condition_name = path.name
    for kinematic_idx, kinematic_name in enumerate(KINEMATICS):
        print(f'Running: {path} {kinematic_name}', flush=True)
        
        if not (condition_name == 'bbhg_lap' and kinematic_name =='v'):
            continue

        df = pl.DataFrame(schema={'ppt': str,
                                'electrode': str,
                                'anatomical_location': str,
                                'correlations': float})

        outpath = Path(f'figure_output/channel_correlations/{condition_name}/{kinematic_name}')
        outpath.mkdir(exist_ok=True, parents=True)

        for path_ppt in path.glob('sub-*'):

            # Load contacts
            contacts = load_contacts(path_ppt.name)

            if contacts.is_empty():
                continue

            correlations = load_correlations(path_ppt, kinematic_idx)
            contacts = contacts.join(correlations, on='electrode')

            df = pl.concat([df.select(sorted(df.columns)) for df in [df, contacts]])    

        task_correlations_per_channel(df, chance_levels[kinematic_idx], outpath)
        task_correlations_per_channel(df, chance_levels[kinematic_idx], outpath,
                                      only_significant_channels=True)
        plt.close('all')

def main():

    run_parallel = 0

    main_path = Path(f'finished_runs/')
    
    conditions = main_path.glob('*')

    if run_parallel:
        pool = Pool(processes=8)
        for path in conditions:
            pool.apply_async(plot_condition, args=(path))
        pool.close()
        pool.join()

    else:
        for path in conditions:
            plot_condition(path)

if __name__=='__main__':
    main()
