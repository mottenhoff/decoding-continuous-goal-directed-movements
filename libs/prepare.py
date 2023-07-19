import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample, detrend

from libs import utils
from figures import checks as fig_checks

from figures.figure_cc_per_band_per_kinematic import plot_band_correlations

START = 0
END = 1

logger = logging.getLogger(__name__)
c = utils.load_yaml('./config.yml')

@dataclass
class Subset:
    eeg: np.array
    xyz: np.array
    ts:  np.array
    fs:  np.array
    channels: np.array
    mapping: dict

def get_fft(signal, fs):
    powerspectrum = np.abs(np.fft.rfft(signal)**2)
    freqs = np.fft.rfftfreq(signal.size, 1/fs)
    return freqs, powerspectrum

def get_windows(subset, wl, ws):
    size_start = subset.eeg.shape

    subset.eeg = utils.window(subset.eeg, subset.ts,
                              wl, ws, subset.fs)
    subset.xyz = utils.window(subset.xyz, subset.ts,
                              wl, ws, subset.fs)

    logger.info(f"Creating windows of length = {wl}ms and stepsize = {ws}ms from {size_start} sample to {subset.eeg.shape} windows [win, samp, chs]")

    subset.eeg = np.mean(subset.eeg, axis=1)
    subset.xyz = np.nanmean(subset.xyz, axis=1)
    
    print(subset.xyz.shape)

    if subset.xyz.ndim == 1:
        subset.xyz = subset.xyz[:, np.newaxis]

    is_all_nan = np.all(np.isnan(subset.xyz), axis=1)  # Why is this here?

    subset.eeg = subset.eeg[~is_all_nan, :]
    subset.xyz = subset.xyz[~is_all_nan, :]

    logger.info(f'Removed {is_all_nan.sum()} out of {is_all_nan.size} windows with only nan')

    return subset

def fill_missing_values(data):
    # TODO: check if needed to handle  leading and trailing nans

    data = pd.DataFrame(data)
    isnan = data.isna().sum(axis=0)
    logger.info(f'Interpolating missing values: On average {isnan.mean()} samples ({isnan.mean()/data.shape[0]*100:.1f}%) ')

    return pd.DataFrame(data).interpolate().to_numpy()

def get_subset_idc(xyz, fs):

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

    fig_checks.plot_gap_cuts(xyz, idc, subset_idc)

    return subset_idc

def go(eeg, xyz):

    # filter resample
    # eeg['data'] = utils.instantaneous_powerbands(eeg['data'], eeg['fs'], c.bands.__dict__)
    

    # eeg['data'] -= eeg['data'].mean(axis=0)
    # eeg['data'] = filter_data(eeg['data'].T, sfreq=eeg['fs'], l_freq=c.bands.__dict__)

    # # window

    if not c.debug.go:
        # eeg['data'] = utils.instantaneous_powerbands(eeg['data'], eeg['fs'], c.bands.__dict__)
        pass

    if c.target_vector:
        # Get target_vector, stack to eeg
        pass

    subset_idc = get_subset_idc(xyz, eeg['fs'])

    # largest_subset = np.argmax([s[1]-s[0] for s in subset_idc])

    subsets = []
    for i, (s, e) in enumerate(subset_idc):
        logger.info(f'Cutting subset {i} from: {s} to {e}')
        subset = Subset(eeg = eeg['data'][s:e, :],
                        xyz = xyz[s:e, :],
                        ts  = eeg['ts'][s:e],
                        fs  = eeg['fs'],
                        channels = eeg['channel_names'],
                        mapping = eeg['channel_mapping'])

        subset.xyz = fill_missing_values(subset.xyz) # Check if this can be commented out because of nanmean
        
        # car = eeg['car'][s:e, :]
        # fig, ax_car = plt.subplots(nrows=7, figsize=(9, 16))
        # for yi, y in enumerate(subset.xyz.T):
        #     cc = np.corrcoef(car, y, rowvar=False)[0, 1]
        #     ax_car[yi].plot(y, color='b')
        #     ax_car[yi].plot(car, color='r')
        #     ax_car[yi].set_title(f'{cc:.3f}')
        # fig.legend(['behavior', 'common average'])
        # fig.tight_layout()
        # fig.savefig(f"./figure_output/control/ppt_{eeg['id']}_subset_{i}_compare_with_car.png")

    
        # subset.eeg -= subset.eeg.mean(axis=0)
        # subset.eeg = detrend(subset.eeg, axis=0)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].plot(*get_fft(subset.eeg[:, 0], subset.fs))
        # ax[0].set_title(f'before (n_samples: {subset.eeg.shape[0]}')
        # ax[0].set_xscale('log')
        # ax[0].set_yscale('log')
        # ax[0].set_xlim(0.01, 512)


        subset = get_windows(subset,
                             c.window.length,
                             c.window.shift)

        ## TODO: make function the target new frequency and make a separate function to calculate that by max target frequency
        # Aimed freq content is Delta (<5hz)
        # target_max_freq = 10
        # subset.eeg = utils.downsample(subset.eeg, subset.fs, target_max_freq)
        # subset.xyz = utils.downsample(subset.xyz, subset.fs, target_max_freq)

        # ax[1].plot(*get_fft(subset.eeg[:, 0], subset.fs))
        # ax[1].set_title(f'after (n_samples: {subset.eeg.shape[0]}')
        # ax[1].set_xscale('log')
        # ax[1].set_yscale('log')
        # ax[1].set_xlim(0.1, 512)
        # fig.savefig(f"./figure_output/control/fft_before_and_after_windowing/ppt_{eeg['id']}_subset_{i}_fft_compare.png")

        # target_number_of_samples = int(subset.eeg.shape[0]/subset.fs*1000 / c.window.shift - c.window.length / c.window.shift)
        # # target_number_of_samples = int(subset.eeg.shape[0] / subset.fs / (c.window.shift * .001))  # 
        # subset.eeg = resample(subset.eeg, target_number_of_samples, axis=0)
        # subset.xyz = resample(subset.xyz, target_number_of_samples, axis=0)

        subsets.append(subset)

    return subsets

        # subset.eeg = fill_missing_values(subset.eeg)
        # subset2 = copy(subset)

        # if i == largest_subset:
            # Do exploratory analysis here
            # pass
            # plot_band_correlations(subset.eeg, subset.xyz, '', get_bands=False)
            # plot_band_correlations(subset.eeg, subset.xyz, '', get_bands=True)

        # Window
        # subset = get_windows(subset,
        #                      c.window.length,
        #                      c.window.shift)

        # if i == largest_subset:
            # plot_band_correlations(subset.eeg, subset.xyz, '_windowed')
            # pass