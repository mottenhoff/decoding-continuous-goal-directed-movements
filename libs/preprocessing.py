# Builtin
import logging
import re

# 3th party
import numpy as np

def flag_disconnected_channels(channels):
    pattern = '(?<![A-Za-z])[Ee][l\d]' # (E of e)l followed by a number
    return [i for i, channel in enumerate(channels)\
              if re.search(pattern, channel)]

def flag_marker_channels(channels):
    return [i for i, channel in enumerate(channels)\
              if 'MKR' in channel]

def flag_ekg_channels(channels):
    return [i for i, channel in enumerate(channels)\
            if 'EKG' in channel]

def flag_flat_signal(data):
    return np.where(np.all(data==data[0], axis=0))[0]

def flag_pattern(channels, patterns: list=[]):
    # NOTE: could be turned into regex for more versatility
    return np.hstack([i for i, channel in enumerate(channels) \
                        for p in patterns \
                        if p in channel])
    
def remove_irrelevant_channels(eeg, channels):
    '''
    - Marker channels
    - EKG channels
    - Disconnected channels
    - Flat signal
    
    NOTE: Also remove flagged electrodes from 
          ppt.exp.channels
    '''
    # HACK
    n_chs = len(channels)
    flagged = np.hstack([flag_disconnected_channels(channels),
                         flag_marker_channels(channels),
                         flag_ekg_channels(channels),
                         flag_flat_signal(eeg[:, :n_chs]),
                         flag_pattern(channels, ['+'])])
    flagged = np.unique(flagged).astype(int)
    
    eeg = np.delete(eeg, flagged, axis=1)
    channels = np.delete(channels, flagged)
    
    logging.info(f'Removed {len(flagged)} irrelevant channels.')
    
    return eeg, channels
