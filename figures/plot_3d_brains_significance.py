
from pathlib import Path

import numpy as np
import polars as pl
from mayavi import mlab

from libs import maps
from brainplots import brainplots

AVG_BRAIN_PATH = Path(r'C:\Users\micro\main\resources\code\brainplots\models\cvs_avg35_inMNI152')
PATH_SERVER = Path(r'L:/FHML_MHeNs\sEEG/')
KINEMATICS = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'r', 'v', 'a']

ELECTRODE_NAMES = 0

def load_brain():

    file_left_hemisphere =  AVG_BRAIN_PATH/'cvs_avg35_inMNI152_lh_pial.mat'
    file_right_hemisphere = AVG_BRAIN_PATH/'cvs_avg35_inMNI152_rh_pial.mat'

    return brainplots.Brain(id_='avg', 
                            file_left=  file_left_hemisphere,
                            file_right= file_right_hemisphere)

def load_chance_level(root_path, percentile=.95):
   
    # May currently even be 99th percentile
    # 1) Get the 95th percentile per participant
    # 2) Take the max correlation over all channels
    chance_levels = {}
    for path in root_path.glob('kh*'):
        chance_level_per_ppt = np.load(path/'0'/'chance_levels_task_correlation_1000.npy')
        chance_level_per_ppt = np.sort(chance_level_per_ppt, axis=0)[int(chance_level_per_ppt.shape[0]*percentile), :]
        chance_level_per_ppt = np.max(chance_level_per_ppt, axis=0)
        chance_levels[path.name] = chance_level_per_ppt

    return chance_levels

def load_contacts(path, ppt_id):

    ppt_id = maps.ppt_id()[ppt_id]
    electrode_path = list(path.rglob(f'{ppt_id}/elec_all*'))[0]

    contacts = brainplots.Contacts(electrode_path)
    contacts.interpolate_electrodes()
    return contacts
    
def load_correlations(ppt_path, kinematic):

    correlations = np.load(ppt_path/'0'/'task_correlations.npy')
    
    return pl.DataFrame([correlations[ELECTRODE_NAMES, :],
                        correlations[kinematic+1, :].astype(np.float64)],
                        schema={
                            'electrode': str,
                            'correlations': float},
                        strict=False)

def save_numbers(contacts):
    unique_idx = lambda c: np.unique(np.where(c.colors == np.array([201, 79, 75, 255]))[0])
    map_to_aloc = lambda c, contact_names: [c.name_map[cname] for cname in c.names[contact_names]]

    all_contacts = np.concatenate([c.names for c in contacts])
    all_anatomical_locations = np.concatenate([list(c.name_map.values()) for c in contacts])

    unique, counts = np.unique(all_anatomical_locations, return_counts=True)
    unique, counts = unique[np.argsort(counts)], np.sort(counts)

    significant_areas = [map_to_aloc(c, unique_idx(c)) for c in contacts]
    unique_sig, counts_sig = np.unique(np.concatenate(significant_areas), return_counts=True)

    is_left = np.array([True if c[0] in ['L'] else False for c in all_contacts])

    with open('contact_information.txt', 'w+') as f:
        f.write(f'n_contacts: {all_contacts.size}\n')
        f.write(f'n_locations: {unique.size}\n')     
        f.write(f'n_left_hemisphere: {sum(is_left)}\n')
        f.write(f'n_right_hemisphere: {sum(~is_left)}\n')
        f.write(f'n_unique_sig_locs: {unique_sig.size}\n')

        f.write('\nMost sampled areas\n')
        for u, c in zip(unique, counts):
            f.write(f'{u:35} {c}\n')
        
    return

def color_by_significance(contact_set, correlations):

    color_sig =  np.array([201, 79, 75, 255])
    color_nsig = np.array([220, 220, 220, 100])

    colors = np.tile(color_nsig, (contact_set.names.size, 1))

    for contact, is_sig in correlations[['electrode', 'is_sig']].iter_rows():

        if not is_sig:
            continue

        colors[np.where(contact_set.names == contact), :] = color_sig

    contact_set.add_color(colors)

    return contact_set

def main():

    main_path = Path(f'finished_runs/')
    main_data = Path(f'../../../resources/data/bubbles-psid-2024')
    main_outpath = Path(f'figure_output/')

    ppts = [ppt_path.name for ppt_path in Path('data').glob('kh*')]

    brain = load_brain()
    contacts = {ppt: load_contacts(main_data, ppt) for ppt in ppts}

    conditions = main_path.glob('*')
    for condition in conditions:

        if condition.name != 'bbhg_lap':
            continue

        chance_levels = load_chance_level(condition, percentile=.95)

        for kinematic_idx, kinematic in enumerate(KINEMATICS):
            print(f'running {condition} {kinematic}', flush=True)
            contact_sets = []

            if kinematic != 'v':
                continue
            ratio = []

            for ppt in condition.glob('kh*'):
                
                if not contacts[ppt.name]:
                    continue

                correlations = load_correlations(ppt, kinematic_idx)
                chance_level = chance_levels[ppt.name]
                correlations = correlations.with_columns(
                        (correlations['correlations'].abs() > chance_level[kinematic_idx])\
                                           .alias('is_sig'))

                contact_set = color_by_significance(contacts[ppt.name],
                                                    correlations)
                
                print(f'{ppt.name} | n_is_sig: {correlations["is_sig"].sum()} | n_contacts: {correlations.shape[0]} | ratio: {correlations["is_sig"].sum()/correlations.shape[0]:.2f}')
                ratio.append(correlations["is_sig"].sum()/correlations.shape[0])
                contact_sets.append(contact_set)

            print(f'mean ratio of significant channels: {np.mean(ratio)}')
            save_numbers(contact_sets)


            scene = brainplots.plot(brain, contact_sets, show=True)

            outpath = main_outpath/'brains3D'/condition.name/kinematic/'significant_channels'
            outpath.mkdir(parents=True, exist_ok=True)
            brainplots.take_screenshots(scene,
                                        outpath,
                                        azimuths =   [0, 180],
                                        elevations = [0, 90],
                                        distances =  [400])
            mlab.close(scene)

if __name__=='__main__':
    main()