
from pathlib import Path

import numpy as np
import polars as pl
from mayavi import mlab
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri.cm as cm

from brainplots import brainplots

cmap = cm.batlow

AVG_BRAIN_PATH = Path(r'C:\Users\p70066129\Maarten\Resources\codebase\brainplots\models\cvs_avg35_inMNI152')
PATH_SERVER = Path(r'L:/FHML_MHeNs\sEEG/')
KINEMATICS = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'r', 'v', 'a']

ELECTRODE_NAMES = 0

def load_brain():

    file_left_hemisphere =  AVG_BRAIN_PATH/'cvs_avg35_inMNI152_lh_pial.mat'
    file_right_hemisphere = AVG_BRAIN_PATH/'cvs_avg35_inMNI152_rh_pial.mat'

    return brainplots.Brain(id_='avg', 
                            file_left=  file_left_hemisphere,
                            file_right= file_right_hemisphere)

def load_chance_level(root_path, percentile=0.95):
    # Loads chance levels per participant and returns the 95% percentile of all these percentiles
    # per kinematic
    chance_levels = np.vstack([np.load(path/'0'/'chance_levels_task_correlation_10000.npy')\
                                 .reshape(-1, len(KINEMATICS))
                               for path in root_path.glob('kh*')])
    
    chance_levels = np.sort(chance_levels, axis=0)[int(chance_levels.shape[0]*percentile), :]

    print(chance_levels)
    return chance_levels

def load_contacts(ppt_id):
    path_to_img_pipe = f'{ppt_id}/imaging/img_pipe/elecs'
    
    
    for filename in ['elecs_all_warped.mat',
                     'elecs_all_nearest_warped.mat']:
        path = PATH_SERVER/path_to_img_pipe/filename

        if path.exists():
            contacts = brainplots.Contacts(path)
            contacts.interpolate_electrodes()
            return contacts

    else:
        return
    
def load_correlations(ppt_path, kinematic):

    correlations = np.load(ppt_path/'0'/'task_correlations.npy')
    
    return pl.DataFrame([correlations[ELECTRODE_NAMES, :],
                        correlations[kinematic+1, :].astype(np.float64)],
                        schema={
                            'electrode': str,
                            'correlations': float},
                        strict=False)

def add_weights(contact_set, correlations):

    # TODO: are contacts that do not exist in the weightmap automatically excluded?
    # # remove electrodes that do not exist in recording
    # used_contacts = set(contacts[ppt].names) & set(correlations['electrode'].to_list())

    # Add weights
    weight_map = dict(zip(correlations['electrode'].to_list(), correlations['correlations'].to_list()))

    # weight_map = {contact: 1 if is_sig else 0.1 for contact, is_sig in weight_map.items()}

    contact_set.add_weights(weight_map)
    contact_set.set_colormap('coolwarm')

    return contact_set

def plot_histogram(correlations, outpath):

    corrs = correlations['correlations']

    fig, ax = plt.subplots()

    ax.hist(corrs, bins=100, color=cm.batlow(0))

    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_xlim(-np.abs(corrs).max(), np.abs(corrs).max())
    ax.set_xlabel('Correlation', fontsize='xx-large')
    ax.set_ylabel('Count', fontsize='xx-large')
    ax.tick_params(axis='x', labelsize='x-large')
    ax.tick_params(axis='y', labelsize='x-large')

    # ax.set_yscale('log')
    # plt.show()
    fig.tight_layout()
    fig.savefig(outpath/'task_corr_distribution.svg')
    fig.savefig(outpath/'task_corr_distribution.png')


def plot_colorbar(vmin, vmax, outpath):

    cap = np.abs((vmin, vmax)).max()
    scale_lims = [-cap, cap]

    ticks = [-cap, 0, cap]

    fig = plt.figure()
    scalar_map = mpl.cm.ScalarMappable(
                        norm=mpl.colors.TwoSlopeNorm(vmin=scale_lims[0], 
                                                     vcenter=0, 
                                                     vmax=scale_lims[1]),
                        # norm=mpl.colors.TwoSlopeNorm(vmin=-.75, vcenter=0, vmax=0.75),
                        cmap=mpl.cm.coolwarm)
    
    ax = fig.add_axes([0.25, 0.5, 0.5, 0.03])

    colorbar = fig.colorbar(scalar_map,
                            cax=ax,
                            orientation='horizontal', 
                            ticks=ticks,
                            format="%.2f")
    
    colorbar.ax.tick_params(labelsize='xx-large')
    fig.savefig(outpath/'colorbar.svg')

def main():
    
    main_path = Path(f'finished_runs/')
    main_outpath = Path(f'figure_output/')

    brain = load_brain()

    conditions = main_path.glob('*')
    condition = next(conditions)

    ppts = sorted([ppt.name for ppt in condition.glob('kh*')])
    contacts = {ppt: load_contacts(ppt) for ppt in ppts}
    colors = [cmap(int(i)) for i in np.linspace(0, 255, len(ppts))]

    for i, ppt in enumerate(ppts):

        color = np.array(colors[i][:-1])*255 #[np.newaxis, :-1]*255
        contacts[ppt].add_color(color.astype(int))

    # Plot it
    outpath = main_outpath/'brains3D'/'colored_per_participant'
    outpath.mkdir(parents=True, exist_ok=True)

    scene = brainplots.plot(brain, list(contacts.values()), show=True)
    
    brainplots.take_screenshots(scene,
                                outpath,
                                azimuths =   [0, 180],
                                elevations = [0, 90],
                                distances =  [400])
    mlab.close(scene)
    plt.close('all')

    #             if not contacts[ppt.name]:
    #                 continue

    #             correlations = load_correlations(ppt, kinematic_idx)
    #             all_correlations.append(correlations)
                
    #             contact_set = add_weights(contacts[ppt.name],
    #                                       correlations)

    #             contact_sets.append(contact_set)
            
    #         all_correlations = pl.concat(all_correlations)


            
    #         plot_histogram(all_correlations, outpath)
    #         print(all_correlations['correlations'].min(),
    #               all_correlations['correlations'].max())
    #         plot_colorbar(all_correlations['correlations'].min(),
    #                       all_correlations['correlations'].max(),
    #                       outpath)

    #         scene = brainplots.plot(brain, contact_sets, show=False)
    #         brainplots.take_screenshots(scene,
    #                                     outpath,
    #                                     azimuths =   [0, 180],
    #                                     elevations = [0, 90],
    #                                     distances =  [400])
    #         mlab.close(scene)
    #         plt.close('all')
    # return


if __name__=='__main__':
    main()




# import numpy as np
# from mayavi.mlab import quiver3d

# def test_quiver3d():
#     x, y, z = np.mgrid[-2:3, -2:3, -2:3]
#     r = np.sqrt(x ** 2 + y ** 2 + z ** 4)
#     u = y * np.sin(r) / (r + 0.001)
#     v = -x * np.sin(r) / (r + 0.001)
#     w = np.zeros_like(z)
#     obj = quiver3d(x, y, z, u, v, w, line_width=3, scale_factor=1)
#     return obj

# test_quiver3d()