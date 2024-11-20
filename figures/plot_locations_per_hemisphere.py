from pathlib import Path

from cmcrameri import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def get_renaming_mapping():
    replacements = [
        # Remove unnecessary stuff
        ('_', ' '),
        ('-', ' '),
        ('ctx', ' '),
        ('rh', ' '),
        ('lh', ' '),
        ('left ', ' '),
        ('right ', ' '),
        ('cerebral', ' '),
        
        # Specific names
        ('ins lg and s cent ins', 'insular long and central sulcus'), # Might just change to insula
	    ('plan polar', 'planum polare'),
	    ('plan tempo', 'planum temporale'),
	    ('intrapariet and p trans', 'intraparietal and parietal transverse'),
	    ('temp sup g t transv', 'temporal superior anterior transverse'),
        
        # Abbreviations
        (' g ', ' gyrus '),
        (' s ', ' sulcus '),
        (' inf ', ' inferior '),
        (' lat ', ' lateral '),
        (' wm ', 'white matter '), 
        (' ant ', ' anterior '),
        (' cingul ', ' cingulate '), 
        (' front ', ' frontal '),
        (' oc ', ' occipital '),
        (' temp ', ' temporal '),
        (' pariet ', ' parietal '),
        (' fusifor ', ' fusiform'),
        (' med ', ' medial '),
        (' fis ', ' fissure '),
        (' sup ', ' superior '),
        (' post ', ' posterior '),
        (' lat ', ' lateral '),
        (' transv ', ' transverse '),
        (' vent ', ' ventricle '),
        (' supramar ', ' supramarginal '),
        (' horizont ', ' horizontal '),
        (' triangul ', ' triangular '),
        (' parahip ', ' parahippocampal '),
        (' collat ', ' collateral '),
        (' frontomargin ', ' frontomarginal '),
        (' intrapariet ', ' intraparietal '),
            
	    # Remove double spaces
        ('  ', ' '),
    ]

    return replacements

def beautify_str(loc, replace_dict = {}):
    loc = loc.center(len(loc)+2) #add spaces
    loc = loc.lower()

    replacement_mapping = get_renaming_mapping()

    for to_replace, replacement in replacement_mapping:
        loc = loc.replace(to_replace, replacement)
    return loc.strip().capitalize()

def get_filenames(path_main, extension, keywords=[], exclude=['_archive']):
    ''' Recursively retrieves all files with 'extension', 
    and subsequently filters by given keywords. 

    keywords: list[str,]
        Selects file when substring exists in filename

    exlude: list[str,]
        Removes file if substring in filename or string equals
        any complete parent foldername.
    '''

    if not path_main.exists():
        print("Cannot access path <{}>."\
                .format(path_main))
        raise NameError

    keywords = extension if len(keywords)==0 else keywords
    extension = f'*.{extension}' if extension[0] != '.' else f'*{extension}'
    files = [path for path in path_main.rglob(extension) \
             if any(kw in path.name for kw in keywords)]

    if any(exclude):
        files = [path for path in files for excl in exclude \
                   if excl not in path.name
                   and excl not in path.parts]
    return files

def load(path, ppts):
    # electrode_filenames = get_filenames(path, 'csv', keywords=['electrode_locations'])

    electrode_filenames = []
    for ppt in ppts:
        current_path = Path(path/ppt/'electrode_locations.csv')

        if current_path.exists():
            electrode_filenames.append(current_path)


    data = {}
    for file in electrode_filenames:
        # if ppt != None and ppt in str(file):    
        df = pd.read_csv(file)
        data[file.parts[3]] = dict(zip(df['electrode_name_1'],
                                       df['location']))
        data[file.parts[3]]['order'] = df['electrode_name_1'].to_list()
    return data
               
def cleanup_locs(locs, to_remove, inverse=False):
    if inverse:
        return {ppt_id: {name: beautify_str(label) for name, label in contacts.items() 
                                 if label in to_remove
                                 and name != 'order'}
                    for ppt_id, contacts in locs.items()}

    return {ppt_id: {name: beautify_str(label) for name, label in contacts.items() 
                                 if label not in to_remove
                                 and name != 'order'}
                    for ppt_id, contacts in locs.items()}

def count(locs, sort_by):
    all_ = np.hstack([list(label.values()) for label in locs.values()])
    u, c = np.unique(all_, return_counts=True)

    if sort_by == 'count':
        return u[np.argsort(c)], sorted(c)
    else:
        return u, c

def split_by_hemisphere(locs):
    nested_dict = lambda d, v: \
                    {outer_k: {inner_k: inner_v for inner_k, inner_v in outer_v.items() 
                                                if inner_k[0]==v}
                               for outer_k, outer_v in d.items()}

    return nested_dict(locs, 'L'), nested_dict(locs, 'R')

def plot_locs_per_hs(locs_all,
                     sort_by='count',
                     show_wm=True,
                     show_unknown=True,
                     inset=True,
                     save_to='./'):
    



    # replace_dict = {
    #             'ctx': '',
    #             '_lh': ' L',
    #             '_rh': ' R',
    #             'Left-Cerebral-White-Matter': 'White matter (Left)',
    #             'Right-Cerebral-White-Matter': 'White matter (Right)'}

    defaultColors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']
    # rgbc = [tuple(int(c.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) for c in defaultColors]
    
    
    # Construct colormap per ppt
    cmap = cm.batlow
    # cmap = cm.lipari

    color_indices = np.linspace(0, 1, len(locs_all))
    # np.random.shuffle(color_indices)
    ppt_colormap = {ppt: cmap(idx) for ppt, idx in zip(locs_all.keys(), color_indices)}


    # ppts = list(locs_all.keys())

    to_remove = []
    if not show_wm:
        to_remove += ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter',
                            'Wm hypointensities']
    if not show_unknown:
        to_remove += ['Unknown']

    locs = cleanup_locs(locs_all, to_remove)
    locs_ins = cleanup_locs(locs_all, to_remove, inverse=True)

    unique, counts = count(locs, sort_by=sort_by)
    unique_ins, count_ins = count(locs_ins, sort_by=sort_by)

    # Split per hemisphere
    lh, rh = split_by_hemisphere(locs)
    lh_ins, rh_ins = split_by_hemisphere(locs_ins)

    # position = np.flip(np.arange(len(unique)))
    position = np.arange(len(unique))
    position_ins = np.arange(len(unique_ins))

    mapping = dict(zip(unique, position))  # TODO: One per patient
    mapping_ins = dict(zip(unique_ins, position_ins))

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(9, 16))

    for i, hem in enumerate([lh, rh]):
        start_idc = np.zeros(len(unique))
        for j, (ppt, ppt_locs) in enumerate(hem.items()):
            ulocs, counts = np.unique(list(ppt_locs.values()), return_counts=True)
            
            idc = [mapping[ul] for ul in ulocs]   

            cax = axs[i].barh(
                y=idc,
                width=counts,
                height=1,
                left=start_idc[idc],
                color=ppt_colormap[ppt],
                label=f'sub-{j+1:02d}')

            start_idc[idc] += counts
    
    if inset:
        xlim_max = -1
        for i, hem in enumerate([lh_ins, rh_ins]):
            ax_ins = inset_axes(axs[i], width="40%", height="10%", loc='center left' if i==0 else 'center right')
            start_idc_ins = np.zeros(len(unique_ins))
            for j, (ppt, l_ins) in enumerate(hem.items()):
                ulocs_ins, counts_ins = np.unique(list(l_ins.values()), return_counts=True)
                
                idc = [mapping_ins[ul] for ul in ulocs_ins]                   

                cax = ax_ins.barh(
                    y=idc,
                    width=counts_ins,
                    height=0.33,
                    left=start_idc_ins[idc],
                    color=ppt_colormap[ppt],
                    label=f'sub-{j+1:02d}')

                start_idc_ins[idc] += counts_ins
            
            if i==0:
                ax_ins.invert_xaxis()
                ax_ins.yaxis.tick_right()
            ax_ins.grid(axis='x', alpha=0.3)
            ax_ins.set_yticks(list(mapping_ins.values()))
            ax_ins.set_yticklabels(unique_ins)
            ax_ins.set_ylim(-0.5, len(mapping_ins)-0.5)
            ax_ins.set_xlabel('Number of contacts')
            # ax_ins.set_xlim((301, 0) if i==0 else (0, 301))
            # ax_ins.spines['left' if i==0 else 'right'].set_visible(False)
            # ax_ins.spines['top'].set_visible(False)

    axs[0].invert_xaxis()
    axs[0].spines['left'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].yaxis.tick_right()
    axs[0].grid(axis='x', alpha=.3)
    axs[1].grid(axis='x', alpha=.3)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_yticks(list(mapping.values()))
    axs[1].set_yticklabels([])
    axs[1].set_ylim(-.5, len(mapping)+.5)
    xlim_max = max(axs[0].get_xlim()[0], axs[1].get_xlim()[1])
    axs[0].set_xlim(xlim_max, 0)
    axs[1].set_xlim(0, xlim_max)
    axs[0].locator_params(axis='x', integer=True)
    axs[1].locator_params(axis='x', integer=True)
    axs[0].set_xlabel('Number of contacts', fontsize='xx-large')
    axs[1].set_xlabel('Number of contacts', fontsize='xx-large')

    # Plot yaxis in middle
    middle = 0.5 # TODO: FIND MIDDLE OF SUBPLOTS INSTEAD OF FIG
    for ylabel, yloc in mapping.items():
        axs[1].annotate(ylabel, (middle, yloc),
                        xycoords=('figure fraction', 'data'),
                        ha='center', va='center',
                        fontsize=11)
    axs[0].set_title('Left hemisphere', fontsize='xx-large')
    axs[1].set_title('Right hemisphere', fontsize='xx-large')

    title = 'Electrode contact locations'

    if len(locs.keys()) > 1:
        # axs[0].legend()
        handles_left, labels_left = axs[0].get_legend_handles_labels()
        handles_right, labels_right = axs[1].get_legend_handles_labels()
        right_legend = tuple(zip(handles_right, labels_right))
        # Remove empty handles, because plt.legend can't handle them when empty
        right_legend = [(handles, labels) for handles, labels in right_legend if len(handles)>0]
        labels_right = [rl[1] for rl in right_legend]
        # Find the handles in the left plot to replace the empty ones
        for i, label in enumerate(labels_left):
            if label not in labels_right and len(handles_left[i])>0:
                right_legend += [(handles_left[i], label)]
        # right_legend = [(h, l[0:2]+'0'+l[2:] if len(l)==3 else l) for h, l in right_legend]
        right_legend.sort(key=lambda x: x[1])
        handles, labels = list(zip(*right_legend))
        axs[1].legend(handles=handles, labels=labels, loc='lower right')
    else:
        title = '{}\n{}'.format(title, list(locs.keys())[0])

    fig.suptitle(title, fontsize='x-large', y=0.99)
    # plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(wspace=.5)
    plt.tight_layout()

    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to/f"all_electrodes_per_hemisphere_{'inset' if inset else ''}.png", dpi=300)
    plt.savefig(save_to/f"all_electrodes_per_hemisphere_{'inset' if inset else ''}.svg", dpi=300)
    return axs

if __name__ == "__main__":

    # Choose PPTS here
    ppts = [f'kh0{i:02d}' for i in [36, 40, 41, 42, 43, 
                                    45, 46, 47, 48, 49,
                                    50, 51, 52, 53, 54,  
                                    55, 56, 67]]

    path_main = Path(r'L:\FHML_MHeNs\sEEG')

    locs = load(path_main, ppts)

    # # Choose your numbers
    # ppts_to_include = [f'kh0{i:02d}' for i in ppts]
    # locs = {k:v for k, v in locs.items() if k in ppts_to_include}

    axs = plot_locs_per_hs(locs, show_wm=False, show_unknown=False, inset=True)
    plt.show()

    print('')