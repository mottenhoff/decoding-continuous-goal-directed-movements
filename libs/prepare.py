import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample, detrend

import libs.utils
from libs.utils import filter_eeg, hilbert
from libs import kinematics
from figures import checks as fig_checks

from figures.figure_cc_per_band_per_kinematic import plot_band_correlations


START = 0
END = 1

logger = logging.getLogger(__name__)
c = libs.utils.load_yaml('./config.yml')

@dataclass
class Subset:
    eeg: np.array
    xyz: np.array
    trials: np.array
    ts: np.array
    fs: np.array
    channels: np.array
    mapping: dict

def get_fft(signal, fs):
    powerspectrum = np.abs(np.fft.rfft(signal)**2)
    freqs = np.fft.rfftfreq(signal.size, 1/fs)
    return freqs, powerspectrum

def get_windows(ts, signal, fs, wl, ws):
    size_start = signal.shape

    signal = libs.utils.window(signal, ts, wl, ws, fs)

    logger.info(f"Creating windows of length = {wl}ms and stepsize = {ws}ms from {size_start} sample to {signal.shape} windows [win, samp, chs]")
    
    signal = np.nanmean(signal, axis=1)

    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    # is_all_nan = np.all(np.isnan(subset.xyz), axis=1)  # Why is this here?

    # subset.eeg = subset.eeg[~is_all_nan, :]
    # subset.xyz = subset.xyz[~is_all_nan, :]

    # logger.info(f'Removed {is_all_nan.sum()} out of {is_all_nan.size} windows with only nan')

    return signal

def fill_missing_values(data):
    # TODO: check if needed to handle  leading and trailing nans

    data = pd.DataFrame(data)
    isnan = data.isna().sum(axis=0)
    logger.info(f'Interpolating missing values: On average {isnan.mean()} samples ({isnan.mean()/data.shape[0]*100:.1f}%) ')

    return pd.DataFrame(data).interpolate().to_numpy()

def get_subset_idc(xyz, fs, name=None):
    n = c.missing_values.xyz_samples_for_gaps

    # Get absolute amount of samples in subset for at least 1 windows
    min_samples = c.window.length * 0.001 * 1024
    # Check if the user set amount of windows is met
    min_samples = np.ceil(c.missing_values.min_windows_to_incl_set * min_samples)

    idc = np.where(~np.isnan(xyz[:, 0]))[0] 
    diff = np.diff(idc)
    gaps = np.where(diff > n*diff.mean())[0] # about 100 samples...

    gaps = np.hstack((-1, gaps, idc.size-1))
    
    subset_idc = [(idc[start_i], idc[end_i]) for start_i, end_i in zip(gaps[:-1]+1, gaps[1:]) \
                                             if (idc[end_i] - idc[start_i]) > min_samples]

    fig_checks.plot_gap_cuts(xyz, idc, subset_idc, name)

    return subset_idc

def frequency_decomposition(eeg: np.array, fs: float):
    frequency_delta = [0, 4]
    frequency_ab    = [8, 30]
    frequency_bbhg  = [55, 200]

    delta_activity =   filter_eeg(eeg, fs, frequency_delta[0], frequency_delta[1])
    # alpha_beta_power = hilbert(filter_eeg(eeg, fs, frequency_ab[0],   frequency_ab[1]))
    # bbhg_power =       hilbert(filter_eeg(eeg, fs, frequency_bbhg[0], frequency_bbhg[1]))

    return np.hstack([
                     delta_activity, 
                    #  alpha_beta_power, 
                    #  bbhg_power
                     ])

def go(ds, save_path):
    '''
    o Extract all features:
        1. Delta activity:  < 5 Hz
            Explicitly extract phase? https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
        2. AlphaBeta Power: 8 - 30 Hz
        3. Broadband Gamma power: 55 - 200 Hz
    
    o Split into subsets to address gaps in handtracking data
        1. Get identify gaps and get start and end
        2. split eeg and xyz
        3. For each subset:
            a. Fill missing values in xyz --> Or downsample, check if methods behave correctly at gaps
            b. Downsample to (say) 20 Hz,  equalize framerates.
    '''
    
    ds.eeg.timeseries = frequency_decomposition(ds.eeg.timeseries, ds.eeg.fs)
    # ds.eeg.channels = np.concatenate([list(map(lambda x: x + f'-{band}', ds.eeg.channels)) for band in ['delta', 'alphabeta', 'bbhg']])  # Uncomment if problems later

    subset_idc = get_subset_idc(ds.xyz, ds.eeg.fs, name=save_path)

    subsets = []
    for i, (s, e) in enumerate(subset_idc):

        logger.info(f'Cutting subset {i} from: {s} to {e}')
        
        subset = Subset(eeg =      ds.eeg.timeseries[s:e, :],
                        xyz =      ds.xyz[s:e, :],
                        trials =   ds.trials[s:e, :],
                        ts  =      ds.eeg.timestamps[s:e],
                        fs  =      ds.eeg.fs,
                        channels = ds.eeg.channels,
                        mapping =  ds.eeg.channel_map)


        subset.xyz = fill_missing_values(subset.xyz)

        # TODO: Include kinematics here

        
        # Downsample to 20 Hz (same as frameshift of 50ms)
        #   if signal is periodic (= eeg) then use fft downsample
        #   for xyz, interpolate linearly (reasonable assumption, since no large gaps), 
        #            and then downsample by selecting every nth sample

        # EEG
        subset.eeg = libs.utils.downsample(subset.eeg, subset.fs, c.downsample_rate/2)

        # Behaviour
        target_samples = subset.eeg.shape[0]
        samples = np.linspace(0, subset.xyz.shape[0]-1, target_samples).round().astype(int)
        subset.xyz = subset.xyz[samples, :]


        subsets.append(subset)

    return subsets
