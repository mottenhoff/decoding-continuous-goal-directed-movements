# Builtin
import re

# 3th party
import numpy as np

from libs.check_quality import QualityChecker


def local_outlier_factor(eeg, k=None):

    k = 20 if not k else k

    lof = LocalOutlierFactor(n_neighbors=k)
    flags = lof.fit_predict(eeg)
    print(zip(flags, lof.negative_outlier_factor_))
    return flags


def flag_irrelevant_channels(qc, eeg, channel_names):
    return np.concatenate([
            qc.get_marker_channels(eeg, channel_names),
            qc.get_ekg_channel(eeg, channel_names),
            qc.get_disconnected_channels(eeg, channel_names)]).astype(int)


def cleanup(eeg, channels, ts, fs):
       
    qc = QualityChecker()

    invalid_timestamps = qc.consistent_timestamps(ts, fs, plot=True)
    irrelevant_channels = flag_irrelevant_channels(qc, eeg, channels)
    seeg = np.delete(eeg, irrelevant_channels, axis=1)

    flags_flat = qc.flat_signal(seeg)
    flags_line = qc.excessive_line_noise(seeg, fs, plot=True)
    flags_aa = qc.abnormal_amplitude(seeg, plot=True)
    # flags_lof = local_outlier_factor(seeg)
        
    return np.hstack((irrelevant_channels, flags_flat, flags_line, flags_aa))