# Builtin
import yaml
import logging

# 3th party
import numpy as np
import matplotlib.pyplot as plt

from libs.check_quality import QualityChecker

logger = logging.getLogger(__name__)

def flag_non_eeg(qc, eeg, channel_names, plot=True):
    return np.concatenate([
            qc.get_marker_channels(eeg, channel_names, plot=plot),
            qc.get_ekg_channel(eeg, channel_names, plot=plot),
            qc.get_disconnected_channels(eeg, channel_names, plot=plot)]).astype(int)

def get_hand_selected(channels, ppt_id, session_id):
    
    with open('./data/hand_selected_channels_to_remove.yml') as f:
        s = yaml.load(f, Loader=yaml.FullLoader)
    
    chs = s[ppt_id][int(session_id)]

    flagged = [channels.index(ch) for ch in chs if ch in channels]

    logger.info(f"Channels flagged by hand: {chs}")

    return flagged

def flag_irrelevant_channels(eeg):
    
    qc = QualityChecker()
    
    # Check if data is valid
    qc.consistent_timestamps(eeg.timestamps, eeg.fs)
    
    # Remove non-eeg channels
    flagged = flag_non_eeg(qc, eeg.timeseries, eeg.channels)
    flagged_names = [eeg.channels[flag] for flag in flagged]

    eeg.timeseries = np.delete(eeg.timeseries, flagged, axis=1)
    eeg.channels   = np.delete(eeg.channels, flagged)
    
    # Remove empty eeg channels
    flagged_eeg = qc.flat_signal(eeg.timeseries, channel_names=eeg.channels)

    logger.info(f'Removed non-eeg: {flagged_names}, Flagged empty eeg: {[f"{flag}: {eeg.channels[flag]}" for flag in flagged_eeg]}')

    plt.close('all')

    return eeg, flagged_eeg