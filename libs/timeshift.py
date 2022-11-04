import numpy as np

def shift(eeg, xyz, t=0):
    ''' t: shift in time in ms.
           positive values shifts xyz into the future
        Not tested for negative values for t
    '''
    if t==0:
       return eeg, xyz

    # shift_idx = np.where((ts - ts[0]) >= t*0.001)[0]  # Takes longer, but dynamic
    shift_idx = round(t*0.001*1024)  # Assumes static fs

    eeg['data'] = eeg['data'][:-shift_idx, :]
    eeg['ts'] = eeg['ts'][:-shift_idx]
    xyz = xyz[shift_idx:, :]

    # TODO: if eeg is larger, then xyz doesnt need to be shortened. 
    #       but then the timeshift should be performed before cutting the experiment
    
    return eeg, xyz