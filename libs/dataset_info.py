import numpy as np

def get_time_between_targets(ds):

    targets = ds.events.target_reached
    ts = ds.xyz_timestamps[np.where(~np.isnan(ds.xyz_timestamps))[0]]

    ts_new_targets = ts[targets[:, 0].astype(int)]
    time_between_targets = np.diff(ts_new_targets)

    return time_between_targets

def get_number_of_samples(ds):
    return ds.xyz[~np.isnan(ds.xyz[:, 0]), 0].size

def get_number_of_seconds(ds):
    return

def get_number_of_targets(ds):
    return ds.events.target_reached.shape[0]