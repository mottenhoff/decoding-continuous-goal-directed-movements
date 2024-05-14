from pathlib import Path
import sys

import numpy as np
from mayavi import mlab

sys.path.append(r"C:\Users\p70066129\Projects\Libs")
from brainplots import brainplots
import libs.maps as maps

def load(path, id_):

    return np.load(path/f'task_correlations_kh{id_:03d}.npy')

def remove_brain_areas(contacts, areas):

    to_remove = [k for k, v in contacts.name_map.items() if v in areas]
    idc = np.hstack([np.where(contacts.names==tr)[0] for tr in to_remove])
    contacts.xyz = np.delete(contacts.xyz, idc, axis=0)
    
    if isinstance(contacts.weights, np.ndarray) and contacts.weights.size > 0:
        contacts.weights = np.delete(contacts.weights, idc)

    return contacts

def main():
    # 42: p4
    # 49: p10
    # 51: p12


    # cmap_name = 'gist_ncar'
    # cmap_name = 'CMRmap'
    # cmap_name = 'rainbow'
    cmap_name = 'tab20'
    colors = list(maps.cmap(cmap_name).values())
    ppt_ids = list(maps.ppt_id().keys())

    ids = [36, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52]
    # Top 5 in order (h-l):  42, 51, 49, 46, 50
    ppt_id = 49  # 42 51 49 

    weight_names = ['xpos', 'ypos', 'zpos', 'xvel', 'yvel', 'zvel', 'spe']
    weights_path = Path('./data/task_correlations/raw_windowed/')

    name = 'Speed'
    name_idx = 6
    # weight_sets = {id_: load('', id_) for id_ in ppt_ids}
        
    # X = LR
    # Y = UD
    # Z = FB
    path_main = Path(r'L:/FHML_MHeNs\sEEG/')
    path_to_img_pipe = path_main/f'kh{ppt_id:03d}/imaging/img_pipe'
    weights_path = Path('./data/task_correlations/raw_windowed/')

    
    file_left_hemisphere  = path_to_img_pipe/'Meshes'/f'patient_{ppt_id}_lh_pial.mat'
    file_right_hemisphere = path_to_img_pipe/'Meshes'/f'patient_{ppt_id}_rh_pial.mat'

    brain = brainplots.Brain(id_=str(ppt_id), 
                            file_left=  file_left_hemisphere,
                            file_right= file_right_hemisphere)
    
    brain.full.opacity = 0.2
    
    # Paths
    path_to_img_pipe = path_main/f'kh{ppt_id:03d}/imaging/img_pipe'
    path_contacts = path_to_img_pipe/'elecs'


    # Load contacts
    try:
        contacts = brainplots.Contacts(path_contacts/'elecs_all.mat')
    except OSError as e:
        print(f'cant load electodes for {ppt_id:03d}')
        raise e
    
    # contacts.interpolate_electrodes()
    # contacts.add_color(np.array(colors[i])[:-1]*255)

    # weights = load(weights_path, ppt_id)[j, :-len(weight_names)]
    weights = np.load(weights_path/f'task_correlations_kh{ppt_id:03d}.npy')[name_idx, :-len(weight_names)]


    assert ~np.any(weights==1), 'Max correlation found!'

    weight_map = dict(zip(contacts.names, weights))
    contacts.add_weights(weight_map)


    weight_info = [(k, v, contacts.name_map[k]) for k, v, in weight_map.items()]
    weight_info = sorted(weight_info, key=lambda x: x[1], reverse=True)
    
    # save 
    with open(f'figures/kh{ppt_id:03d}_contact_task_correlation.txt', 'w') as f:
        for row in weight_info:
            f.write(f'{row[0]} {row[1]} {row[2]}\n')
        # f.writelines(str([f'{row[0]} {row[1]}, {row[2]}\n' for row in weight_info]))




    # areas = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'WM-hypointensities']
    # contacts = remove_brain_areas(contacts, areas)

    # all_contacts += [contacts]


    # # print(name)
    # all_weights = np.concatenate([c.weights for c in all_contacts])
    # print('Highest n corr:', np.sort(np.abs(all_weights))[-20:])

    # print(np.sort(np.abs(weights.astype(np.float32)[:contacts.names.size]))[::-1])
    scene = brainplots.plot(brain, contacts, show=True)
    # brainplots.take_screenshots(scene, outpath=f'./tmp')
    # brainplots.mlab.colorbar()

    # mlab.close(scene=scene)
    print()
    # break


    return
    

if __name__=='__main__':
    main()
