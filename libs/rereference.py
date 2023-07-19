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