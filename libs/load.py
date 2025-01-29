import logging
import csv
from bisect import bisect_right
from dataclasses import dataclass, fields

import numpy as np

from libs.read_xdf import read_xdf
from libs import plotting
from libs import utils
from libs import debug
from libs.target_vector import get_target_vector

logger = logging.getLogger(__name__)
c = utils.load_yaml('./config.yml')

LEAP_COMPLETE_HAND_XYZ_IDC = [7, 8, 9,
                              11, 12, 13,
                              15, 16, 17,
                              19, 20, 21,
                              23, 24, 25,
                              27, 28, 29]

@dataclass
class Dataset:
    ppt_id: int
    eeg: dataclass
    xyz: np.array
    xyz_timestamps: np.array
    trials: np.array
    events: dataclass
    target_vector: np.array

@dataclass
class Eeg:
    timeseries: np.array
    timestamps: np.array
    fs: float
    total_time: float
    channels: np.array
    channel_map: dict

@dataclass
class Events:
    target_new: np.array
    target_reached: np.array
    cursor_reset_start: np.array
    cursor_reset_end: np.array
    on_target: np.array
    off_target: np.array

def load_locations(path):
    if not path.exists():
        return {}

    with open(path) as f:
        reader = csv.DictReader(f)
        data = {row['electrode_name_1']: row['location'] for row in reader}

    return data

def locate_pos(ts, target_ts):
    pos = bisect_right(ts, target_ts)
    if pos == 0:
        return 0
    if pos == len(ts):
        return len(ts)-1
    if abs(ts[pos]-target_ts) < abs(ts[pos-1]-target_ts):
        return pos
    else:
        return pos-1

def str_to_list(s): 
    s = s.strip('[]')
    return s.split(',') if ',' in s else s.split()

def generate_fn_t(i, ol, m, h, ob):
    def func(v):
        return i*(v+ol)/m*h+h+ob
    return func

def parse_markers(markers, ts):
    
    # Tranformation vectors
    fn_t = {}
    try:
        for marker in markers[1:4]:
            marker = marker[0].split(';')
            i, ol, m, h, ob = [float(v) for v in marker[1:]]
            fn_t[marker[0]] = generate_fn_t(i, ol, m, h, ob)
    except Exception:
        # For pilot experiments
        fn_t['xt'] = generate_fn_t(1, 0, 300, 800, 0)
        fn_t['yt'] = generate_fn_t(-1, -250, 150, 450, 0)
        fn_t['zt'] = generate_fn_t(1, 0, 200, 50, 10)

    # Events
    events = {'t_new':         np.empty((0, 4)), 
              't_reached':     np.empty((0, 4)),
              'c_reset_start': np.empty((0, 5)),
              'c_reset_end':   np.empty((0, 5)),
              'on_target':     np.empty((0, 1)),
              'off_target':    np.empty((0, 1))}

    # Dont parse marker before experiment_start
    for i, m in enumerate(markers):
        
        if 'experiment_started' in m:
            exp_start_idx = i
        elif 'experiment_finished' in m:
            exp_end_idx = i
    
    for i, (t, m) in enumerate(zip(     ts[exp_start_idx+1:exp_end_idx], 
                                   markers[exp_start_idx+1:exp_end_idx])):
        event = m[0].split(';')
        
        if event[0] == 'skip':
            # Has to be on top, because skip event is len==0
            pass

        elif event[1] == 'new_target':  # [skip]  [skip]
            xyz = np.array(str_to_list(event[-1])).astype(float)
            events['t_new'] = np.vstack([events['t_new'], np.append(t, xyz)])

        elif event[1] == 'target_reached':
            xyz = np.array(str_to_list(event[-1])).astype(float)
            events['t_reached'] = np.vstack([events['t_reached'], np.append(t, xyz)])

        # Can be used to differentiate initial move to target
        #      and 'popping' the bubble.
        elif event[1] == 'on_target':
            events['on_target'] = np.vstack((events['on_target'], t))

        elif event[1] == 'off_target':
            events['off_target'] = np.vstack((events['off_target'], t))

        # Cursor_reset identifies jumps in trajectory. 
        # xyz = bubble_cursor position at event
        elif event[1] == 'cursor_reset_start':
            xyz = np.array(str_to_list(event[-1])).astype(float)
            events['c_reset_start'] = np.vstack([events['c_reset_start'],
                                                    np.hstack([t, float(event[0]), xyz])])

        elif event[1] == 'cursor_reset_end':
            xyz = np.array(str_to_list(event[-1])).astype(float)
            events['c_reset_end'] = np.vstack([events['c_reset_start'],
                                                np.hstack([t, float(event[0]), xyz])])

    events = Events(events['t_new'], events['t_reached'],
                    events['c_reset_start'], events['c_reset_end'],
                    events['on_target'], events['off_target'])

    return events, fn_t, (ts[exp_start_idx], ts[exp_end_idx])

def cut_experiment(stream, cutoff):
    ts_s = locate_pos(stream['ts'], cutoff[0])
    ts_e = locate_pos(stream['ts'], cutoff[1])

    stream['data'] = stream['data'][ts_s:ts_e, :]
    stream['ts'] = stream['ts'][ts_s:ts_e]

    return stream

def get_screen_dimensions(markers):

    for marker in markers['data']:
    
        if 'screen_dimensions' in marker[0]:
    
            return list(map(int, marker[0].split(';')[1:]))

def fix_too_many_trial_starts(start, end):

    # Delete trailing trial_starts
    start = np.delete(start, np.where(start > end[-1]-1))

    while start.size != end.size:
        
        flag = None
        for i in np.arange(end.size):
            
            if start[i+1] < end[i]:
                flag = i+1
                break
        
        if flag:
            start = np.delete(start, flag)

    return start

def get_trials(leap, events):
    # trials from new_target to target_reached
    t_start_idc = np.array([locate_pos(leap['ts'], nt) for nt in events.target_new[:, 0]])
    t_end_idc = np.array([locate_pos(leap['ts'], tr) for tr in events.target_reached[:, 0]])

    # Exp sends one t_start to many, so remove
    t_start_idc = t_start_idc[:-1]
    trial_nums = np.empty(leap['ts'].shape[0])
    trial_nums.fill(np.nan)

    # Sometimes t_start_idc are send multiple times,
    # these should be removed
    if t_start_idc.size != t_end_idc.size:
        t_start_idc = fix_too_many_trial_starts(t_start_idc, t_end_idc)
    
    # Align with leap
    for i, (s, e) in enumerate(zip(t_start_idc, t_end_idc)):
        trial_nums[s:e] = i

    trial_nums[-1] = i  #  Last one is not included in the loop

    # Fix where diff between trial_end and _start is shifted a sample
    d_end_start = (t_start_idc[1:] - t_end_idc[:-1])
    
    #   Let me know if there is a large difference, then it needs more attention
    if any(d_end_start > (warn_diff := 5)):  # Arbitrary number
        logger.warning(f'Difference between end and start is larger than {warn_diff} samples')

    for i in np.where(d_end_start)[0]:
        trial_nums[t_end_idc[i]] = i
    
    targets = get_target_per_sample(trial_nums, events.target_reached)
    trials = np.hstack((np.expand_dims(trial_nums, axis=1), targets))

    if np.isnan(trials[0][0]):
        # Occasionally, the values are filled from the second
        # index. If so fill the first row with the second row.
        trials[0, :] = trials[1, :]

    return trials

def get_target_per_sample(trial_nums, targets):
    ''' points: 3d cursor coordinates
        targets: 3d coordinates of target
        events: 
    '''

    mask = np.nan_to_num(trial_nums, nan=999)
    unique, inv = np.unique(mask, return_inverse=True)
    trial_num_to_target = dict(zip(unique, targets[:, 1:]))
    trial_num_to_target[999] = np.array([np.nan, np.nan, np.nan])
    target_per_samp = np.array([trial_num_to_target[x] for x in unique])[inv]

    return target_per_samp

def ts_to_idx(ts_target, ts):
    return np.array([locate_pos(ts_target, t) for t in ts])

def align(leap, events):

    trials = get_trials(leap, events)

    for field in fields(events):
        data = getattr(events, field.name)
        data[:, 0] = ts_to_idx(leap['ts'], data[:, 0])
        setattr(events, field.name, data)

    return events, trials

def leap_to_bubble_space(xyz, fn_t):
    '''
    new order: palm_x, thumb_x, .. , pinky_x, palm_y, etc
    '''
    
    points = np.reshape(LEAP_COMPLETE_HAND_XYZ_IDC, (-1, 3))

    if c.complete_model:
        ix, iy, iz = points[:, 0], points[:, 1], points[:, 2]
    else:
        ix, iy, iz = points[0, :]

    x = fn_t['xt'](xyz[:, ix])
    y = fn_t['yt'](xyz[:, iy])
    z = fn_t['zt'](xyz[:, iz])
    
    xyz = np.hstack([x, y, z]) if c.complete_model else np.vstack([x, y, z]).T
    
    return xyz

def load_dataset(path, ppt_id):
       
    data, _ = read_xdf(path)
    leap = data['LeapLSL']
    markers = data['Bubble']
    eeg = data['Micromed']

    events, fn_t, exp_time = parse_markers(markers['data'], markers['ts'])
    leap = cut_experiment(leap, exp_time)
    eeg = cut_experiment(eeg, exp_time)

    # Plots
    plotting.plot_effective_framerate(leap['ts'])

    # Trials = [:, [trial_num, target_vec_x, ..._y, ..._z]]
    events, trials = align(leap, events)

    # Also selects the hand model to use
    xyz = leap_to_bubble_space(leap['data'], fn_t)        
    xyz_ts = leap['ts']

    target_vector = get_target_vector(trials, xyz) # if c.target_vector else np.array([])

    eeg['channel_mapping'] = load_locations(path.parent/'electrode_locations.csv')

    eeg = Eeg(eeg['data'], eeg['ts'], eeg['fs'], 
              float(eeg['total_stream_time']), eeg['channel_names'], eeg['channel_mapping'])

    if c.debug.active and c.debug.short:
        debug.shorten_dataset(eeg, xyz)

    dataset = Dataset(ppt_id, eeg, xyz, xyz_ts, trials, events, target_vector)

    return dataset