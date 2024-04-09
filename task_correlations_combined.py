from pathlib import Path
import sys

import numpy as np
from mayavi import mlab
import matplotlib as mpl
import matplotlib.pyplot as plt

from cmcrameri import cm

# Local
from brainplots import brainplots

KINEMATICS = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'r', 'v', 'a']
SPEED = 10

AVG_BRAIN_PATH = Path(r'C:\Users\p70066129\Maarten\Resources\codebase\brainplots\models\cvs_avg35_inMNI152')
PATH_SERVER = Path(r'L:/FHML_MHeNs\sEEG/')

def load(main_path: Path):

    contacts = {}

    for path in main_path.glob('kh*'):

        ppt_id = path.name

        try:
            contact_set = load_contacts(ppt_id)
        except OSError as error:
            print(error)
            continue

        task_correlations = np.load(path/'0'/'task_correlations.npy')
        task_correlations = {'channels': task_correlations[0,  :],
                             'corrs':    task_correlations[1:, :].astype(np.float64)}

        weights = load_weights(contact_set, task_correlations)
        contact_set.add_weights(weights)

        contact_set.task_correlations = task_correlations
        contacts[ppt_id] = contact_set

    return contacts

def load_brain():

    file_left_hemisphere =  AVG_BRAIN_PATH/'cvs_avg35_inMNI152_lh_pial.mat'
    file_right_hemisphere = AVG_BRAIN_PATH/'cvs_avg35_inMNI152_rh_pial.mat'

    return brainplots.Brain(id_='avg', 
                            file_left=  file_left_hemisphere,
                            file_right= file_right_hemisphere)

def load_contacts(ppt_id):
    path_to_img_pipe = f'{ppt_id}/imaging/img_pipe/elecs'
    filename = 'elecs_all_warped.mat'

    if not (PATH_SERVER/path_to_img_pipe/filename).exists():
        filename = 'elecs_all_nearest_warped.mat'

    contacts = brainplots.Contacts(PATH_SERVER/path_to_img_pipe/filename)

    contacts.interpolate_electrodes()

    return contacts

def load_weights(contacts, task_correlations, kinematic_idx=SPEED):
    find_contact_idx = lambda c: np.where(task_correlations['channels'] == c)[0][0]

    existing_contacts = set(contacts.names) & set(task_correlations['channels'])

    weights = {contact: task_correlations['corrs'][kinematic_idx, find_contact_idx(contact)]
                            for contact in existing_contacts}

    return weights

def remove_brain_areas(contacts, areas):

    to_remove = [k for k, v in contacts.name_map.items() if v in areas]
    idc = np.hstack([np.where(contacts.names==tr)[0] for tr in to_remove])
    contacts.xyz = np.delete(contacts.xyz, idc, axis=0)
    
    if isinstance(contacts.weights, np.ndarray) and contacts.weights.size > 0:
        contacts.weights = np.delete(contacts.weights, idc)

    return contacts

def plot_histogram(contacts, outpath):

    corrs = np.concatenate([contact_set.task_correlations['corrs'][SPEED, :] 
                            for contact_set in contacts.values()])
    fig, ax = plt.subplots()

    ax.hist(corrs, bins=100, color=cm.batlow(0))

    ax.spines[['top', 'right', 'left']].set_visible(False)

    ax.set_xlim(-np.abs(corrs).max(), np.abs(corrs).max())
    ax.set_xlabel('Correlation', fontsize='xx-large')
    ax.set_ylabel('Count', fontsize='xx-large')
    ax.tick_params(axis='x', labelsize='x-large')
    ax.tick_params(axis='y', labelsize='x-large')

    # ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(outpath/'task_corr_distribution.svg')
    fig.savefig(outpath/'task_corr_distribution.png')

def plot_brain(contacts, outpath):
    
    savepath = outpath/'brains'
    brain = load_brain()

    scene = brainplots.plot(brain, list(contacts.values()), show=False)
    brainplots.take_screenshots(scene, 
                                outpath=savepath,
                                azimuths=[0, 90, 180, 270],
                                elevations=[0, 90],
                                distances=[400])
    mlab.show()
    mlab.close(all=True)

def plot_colorbar():
    # # Plot the colorbar
    # scale_lims_per_ppt = np.array([(values['corrs'][i_kin, :].min(),
    #                                 values['corrs'][i_kin, :].max()) 
    #                         for values in task_correlations.values()])
    
    # scale_lims = (scale_lims_per_ppt.min(), scale_lims_per_ppt.max())
    # cap = max(abs(scale_lims[0]), abs(scale_lims[1]))
    # scale_lims = [-cap, cap]
    # # scale_lims = (-0.75, 0.75)
    # ticks = [scale_lims[0], 0, scale_lims[1]]

    # fig = plt.figure()
    # cb = fig.colorbar(mpl.cm.ScalarMappable(
    #                                 norm=mpl.colors.TwoSlopeNorm(vmin=scale_lims[0], vcenter=0, vmax=scale_lims[1]),
    #                                 # norm=mpl.colors.TwoSlopeNorm(vmin=-.75, vcenter=0, vmax=0.75),
    #                                 cmap=mpl.cm.coolwarm),
    #                 orientation='horizontal', 
    #                 ticks=ticks,
    #                 format="%.2f")
    # cb.ax.tick_params(labelsize='xx-large')
    # # plt.show()
    # fig.savefig(f'./task_correlations/{condition}_{kinematic}/task_correlations_colorbar.svg')

    raise NotImplementedError


def list_correlated_locations(contacts, outpath):

    outpath = outpath/'correlation_locations/'
    outpath.mkdir(exist_ok=True, parents=True)

    for ppt_id, contact in contacts.items():

        sorted_corrs_index = np.argsort(contact.task_correlations['corrs'][10, :])  # lo -> hi
        
        sorted_channels = contact.task_correlations['channels'][sorted_corrs_index]
        sorted_corrs = contact.task_correlations['corrs'][10, sorted_corrs_index]
        
        with open(outpath/f'{ppt_id}_locations_correlations.txt', 'w') as f:
            for ch, corr in zip(sorted_channels, sorted_corrs):
                print(f'{ch}, {contact.name_map.get(ch, None)}, {corr:.3f}', file=f)

    # One single list
    all_correlations = []
    for file in outpath.glob('kh*'):

        ppt_id = file.name.split('_')[0]

        with open(file, 'r') as f:
            correlations = [[f'{ppt_id}'] + list(line.strip().split(', ')) for line in f.readlines()]
        
        all_correlations += [correlations]

    all_correlations = np.vstack(all_correlations)

    all_correlations = all_correlations[np.argsort(all_correlations[:, -1].astype(np.float64)), :]
        
    with open(outpath/'all_task_correlations.txt', 'w') as f:
        for row in all_correlations:
            f.write(', '.join([r for r in row]) + '\n')


def main():
    cmap = cm.hawaii
    cmap = cm.batlowW
    
    kinematics = [10]  # Speed

    conditions = [
        'delta', 
        'alphabeta', 
        'bbhg'
        ]

    condition = conditions[1]

    main_path = Path(f'finished_runs/')
    outpath = Path(f'figure_output/channel_correlations/{condition}')
    outpath.mkdir(exist_ok=True, parents=True)

    contacts = load(main_path/condition)

    ppt_ids = list(contacts.keys())
    colors = [cmap(i) for i in np.linspace(0, 1, len(ppt_ids))]


    # plot_histogram(contacts, outpath)
    # plot_brain(contacts, outpath)
    # plot_colorbar()
    list_correlated_locations(contacts, outpath)

    # TODO: List locations highest (absolute) correlations. Make 
    #       sure to include electrode and ppt_id, especially delta
    #       may be on the same electrode, which may have some 
    #       implications




   


            

    return
    

if __name__=='__main__':
    main()
