
from bisect import bisect_right
from pathlib import Path
from dataclasses import dataclass, fields

import matplotlib.pyplot as plt
import numpy as np

from libs.read_xdf import read_xdf
from libs.leap import plotting

PLOT = True

@dataclass
class Events:
    target_new: np.array
    target_reached: np.array
    cursor_reset_start: np.array
    cursor_reset_end: np.array
    on_target: np.array
    off_target: np.array


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
        
        # TODO: check for exp start
        if event[1] == 'new_target':
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
        elif event[1] == 'skip':
            # TODO. (make another test session)
            pass

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

def get_trials(leap, events):
    # trials from new_target to target_reached
    t_start_idc = np.array([locate_pos(leap['ts'], nt) for nt in events.target_new[:, 0]])
    t_end_idc = np.array([locate_pos(leap['ts'], tr) for tr in events.target_reached[:, 0]])

    # Exp sends one t_start to many, so remove
    t_start_idc = t_start_idc[:-1]

    # Align with leap
    trial_nums = np.zeros(leap['ts'].shape[0])
    for i, (s, e) in enumerate(zip(t_start_idc, t_end_idc)):
        trial_nums[s:e] = i

    trial_nums[-1] = i  # Last one is not included in the loop

    return trial_nums

def ts_to_idx(ts_target, ts):
    return np.array([locate_pos(ts_target, t) for t in ts])

def align(leap, events):

    trials = get_trials(leap, events)

    for field in fields(events):
        data = getattr(events, field.name)
        data[:, 0] = ts_to_idx(leap['ts'], data[:, 0])
        setattr(events, field.name, data)

    return events, trials

def align_matrices_with_diff_fs(l, ts_l, s, ts_s):
    '''
    l = larger matrix
    s = smaller matrix

    ts = timestamps
    idc = align indices where values of s should be inserted in l
    '''

    s_ext = np.full((l.shape[0], s.shape[1]), np.nan)

    idc = np.array([locate_pos(ts_l, nt) for nt in ts_s])
    s_ext[idc, :] = s

    return np.hstack((l, s_ext)), idc

def leap_to_bubble_space(xyz, fn_t):

    x = fn_t['xt'](xyz[:, 0])
    y = fn_t['yt'](xyz[:, 1])
    z = fn_t['zt'](xyz[:, 2])
    
    return np.vstack([x, y, z]).T

def go(path):
       
    data, _ = read_xdf(path)
    leap = data['LeapLSL']
    markers = data['Bubble']
    eeg = data['Micromed']

    events, fn_t, exp_time = parse_markers(markers['data'], markers['ts'])
    leap = cut_experiment(leap, exp_time)
    eeg = cut_experiment(eeg, exp_time)

    if PLOT:
        plotting.plot_effective_framerate(leap['ts'])

    events, trials = align(leap, events)

    leap['data'] = leap_to_bubble_space(leap['data'][:, 7:10], fn_t)

    aligned, idc = align_matrices_with_diff_fs(eeg['data'], eeg['ts'], 
                                               leap['data'], leap['ts'])



    return aligned, eeg['ts'], idc, trials, events


if __name__=='__main__':
    data_path = Path('./data/kh036/')
    filename = f'bubbles_{1}.xdf'
    go(data_path/filename)