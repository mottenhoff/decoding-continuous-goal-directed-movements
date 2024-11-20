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

            # print(f'unique neighbors: {prev_adj.size==1 and i_adj.size==1}')
            prev_adj = i_adj

    # import matplotlib.pyplot as plt
    # mask = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])

    # fig, ax = plt.subplots(nrows=mask.size, ncols=2, sharex=True, figsize=(20, 12))
    # for i, ch in enumerate(mask):
    #     ax[i, 0].plot(eeg[:, ch])
    #     ax[i, 1].plot(new[:, ch])

    # ax[0, 0].set_title('before')
    # ax[0, 1].set_title('after')
    # plt.show()



    # fig, ax = plt.subplots(figsize=(12, 12)); 
    # im = ax.imshow(np.corrcoef(eeg, rowvar=False), cmap='viridis'); 
    # ax.set_title('original')
    # fig.colorbar(im);

    # fig.savefig('lap_corrmat_eeg_before.png')


    # fig, ax = plt.subplots(figsize=(12, 12)); 
    # im = ax.imshow(np.corrcoef(new, rowvar=False), cmap='viridis'); 
    # ax.set_title('new')
    # fig.colorbar(im);
    # fig.savefig('lap_corrmat_eeg_after.png')
    # plt.show()
    # plt.close('all')

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