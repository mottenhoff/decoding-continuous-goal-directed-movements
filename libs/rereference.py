import re
import numpy as np

def common_electrode_reference(eeg, channels):

    electrodes = set([ch.strip('0123456789') for ch in channels])
    rereferenced = np.zeros(eeg.shape)

    for electrode in electrodes:

        # Get all contacts on electrode
        contacts_idc = np.where(np.char.find(channels, electrode) + 1)[0]

        cer = eeg[:, contacts_idc].mean(axis=1, keepdims=True)

        rereferenced[:, contacts_idc] = eeg[:, contacts_idc] - cer

    return rereferenced

def common_average_reference(eeg):

    return eeg - eeg.mean(axis=1, keepdims=True)

def laplacian_reference(eeg, channels, step_size=1):
    '''
    Apply a laplacian re-reference to the data
    
    Parameters
    ----------
    eeg: array [samples x channels]
        EEG time series
    channels: list of channel names
    step_size: neighbor channel to use, 2 would skip the first neighbor
               and use the second one.

    Returns
    ----------
    eeg: array (samples, channels)
        Laplacian re-referenced EEG time series   
    '''
    channels = np.array(channels)

    shafts = set([ch.strip('0123456789') for ch in channels])
    new = np.empty(eeg.shape)

    for shaft in shafts:
        # Get all contacts on shaft
        mask = np.where(np.char.find(channels, shaft) + 1)[0]  # +1 because returns 0 if true and -1 is false...
        contacts = channels[mask]
        nums = [int(re.sub('\D', '', c)) for c in contacts]  # Extract the numbers

        prev_adj = np.array([])

        for e_curr in nums:
            # If adjacent contact is missing, np.where wont return a value
            i_adj = np.where((channels == f'{shaft}{e_curr - step_size}') | \
                             (channels == f'{shaft}{e_curr + step_size}'))[0]

            i_curr = np.where(channels==f'{shaft}{e_curr}')[0]
            
            # If any neighbors AND two neighbors are not uniquely neigbors
            # from each other, the apply the laplacian, otherwise do nothing.
            # In the case of a unique neighbors applying a laplacian results
            # in substracting a from b and b from a, inducing a corrrelation
            # between a and b of -1. High collinearity causes problems for 
            # the machine learning models.
            if i_adj.size > 0 and \
               not (prev_adj.size==1 and i_adj.size==1):  
                new[:, i_curr] = eeg[:, i_curr] - np.expand_dims(eeg[:, i_adj].mean(axis=1), axis=1)
            else:
                # In rare case of no adjacent contacts
                new[:, i_curr] = eeg[:, i_curr]

            prev_adj = i_adj

    return new

if __name__=='__main__':
    channels = [f'RH{i}' for i in range(1, 6)]
    channels_missing = ['RH1', 'RH2', 'RH4', 'RH5', 'RH6']
    eeg = np.zeros((1, 5))

    for idx, i in enumerate([1, 2, 3, 5, 7]):
        eeg[:, idx] = i

    print(eeg)
    print(laplacian_reference(eeg, channels_missing))
    
    # Should be: [-1, 0, -.5, 0, +2]
    # Should be: [-1, 1, -2, 0, 2] for with a missing channel