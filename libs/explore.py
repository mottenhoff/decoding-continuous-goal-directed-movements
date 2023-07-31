import numpy as np
import matplotlib.pyplot as plt

def task_correlations(datasets, savepath):
    
    y = np.vstack([s.eeg for s in datasets])
    z = np.vstack([s.xyz for s in datasets])


    cm = np.corrcoef(np.hstack([y, z]), rowvar=False)
    task_correlations = cm[-z.shape[1]:, :]

    with open(savepath/'task_correlations.npy', 'wb') as f:
        np.save(f, task_correlations)


def main(data, savepath):
    task_correlations(data, savepath)
    
    plt.close('all')
    

    return