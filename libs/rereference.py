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


def laplacian_reference_ols(eeg, channels):
    '''
    Apply a laplacian re-reference to the data
    
    Parameters
    ----------
    eeg: array [samples x channels]
        EEG time series
    channels: list of channel names

    Returns
    ----------
    eeg: array (samples, channels)
        Laplacian re-referenced EEG time series   
    '''

    shafts = set([ch.strip('0123456789') for ch in channels])

    referenced = []
    for shaft in shafts:
        channel_idc = np.where(np.char.find(channels, shaft) + 1)[0]  # +1 because returns 0 if true and -1 is false...

        # If at corner, act as bipolar. Replace else statement with np.zeros(eeg[:, middle].shape) to switch to half laplacian
        referenced += [eeg[:, middle] - ((eeg[:, left]  if left >=  channel_idc[0]  else eeg[:, right]) +
                                        (eeg[:, right] if right <= channel_idc[-1] else eeg[:, left])) / 2
                                         for left, middle, right in zip(channel_idc - 1, 
                                                                        channel_idc,
                                                                        channel_idc + 1)]

    return np.vstack(referenced).T

def laplacian_reference(eeg, channels, step_size=1):

    channels = np.array(channels)

    shafts = set([ch.strip('0123456789') for ch in channels])
    new = np.empty(eeg.shape)

    for shaft in shafts:
        # Get all contacts on shaft
        mask = np.where(np.char.find(channels, shaft) + 1)[0]  # +1 because returns 0 if true and -1 is false...
        contacts = channels[mask]
        nums = [int(re.sub('\D', '', c)) for c in contacts]  # Extract the numbers

        for e_curr in nums:
            # If adjacent contact is missing, np.where wont return a value
            i_adj = np.where((channels == f'{shaft}{e_curr - step_size}') | \
                             (channels == f'{shaft}{e_curr + step_size}'))[0]

            i_curr = np.where(channels==f'{shaft}{e_curr}')[0]
            
            if i_adj.size > 0:
                new[:, i_curr] = eeg[:, i_curr] - np.expand_dims(eeg[:, i_adj].mean(axis=1), axis=1)
            else:
                # In rare case of no adjacent contacts
                new[:, i_curr] = eeg[:, i_curr]

    return new

if __name__=='__main__':
    channels = [f'RH{i}' for i in range(1, 6)]
    channels_missing = ['RH1', 'RH2', 'RH4', 'RH5', 'RH6']
    eeg = np.zeros((1, 5))

    for idx, i in enumerate([1, 2, 3, 5, 7]):
        eeg[:, idx] = i

    print(eeg)
    print(laplacian_reference_old(eeg, channels))
    print(laplacian_reference(eeg, channels_missing))
    
    # Should be: [-1, 0, -.5, 0, +2]
    # Should be: [-1, 1, -2, 0, 2] for with a missing channel