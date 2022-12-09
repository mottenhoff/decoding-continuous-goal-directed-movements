# Builtin
import yaml
import logging

# 3th party
import numpy as np

from libs.check_quality import QualityChecker

logger = logging.getLogger(__name__)

def local_outlier_factor(eeg, k=None):

    k = 20 if not k else k

    lof = LocalOutlierFactor(n_neighbors=k)
    flags = lof.fit_predict(eeg)
    print(zip(flags, lof.negative_outlier_factor_))
    return flags


def flag_irrelevant_channels(qc, eeg, channel_names, plot=True):
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



def cleanup(eeg, channels, ts, fs, pid, sid):
       
    qc = QualityChecker()

    invalid_timestamps = qc.consistent_timestamps(ts, fs, plot=True)

    flags_flat = qc.flat_signal(eeg, channel_names=channels, plot=True)
    flags_line = qc.excessive_line_noise(eeg, fs, plot=True, channel_names=channels)
    # flags_aa = qc.abnormal_amplitude(seeg, plot=True, channel_names=channels)
    # flags_lof = local_outlier_factor(seeg)
    # flags_hand = get_hand_selected(channels, pid, sid)

    return np.hstack((flags_flat, flags_line)).astype(np.int16)
    # return np.hstack((irrelevant_channels, flags_flat, flags_line, flags_aa, flags_hand)).astype(np.int16)

def remove_non_eeg(eeg, channels):
    qc = QualityChecker()

    flagged = flag_irrelevant_channels(qc, eeg, channels, plot=True)
    eeg = np.delete(eeg, flagged, axis=1)
    channels = np.delete(channels, flagged)    

    logger.info(f'Removing non-eeg channels: {flagged}')
    return eeg, channels