import bisect
from ast import literal_eval
from datetime import datetime
from os.path import exists, getctime
from pathlib import Path
from collections import OrderedDict

import pyxdf
import numpy as np
from pandas import read_csv


def _get_created_date(file, dt_format='%Y%m%d%H%M%S'):
    # Returns the formatted date of creation of a file
    return datetime.fromtimestamp(getctime(file)).strftime(dt_format)

def _locate_pos(available_tss, target_ts):
    # Locate the the closest index within a list of indices
    pos = bisect.bisect_right(available_tss, target_ts)
    if pos == 0:
        return 0
    if pos == len(available_tss):
        return len(available_tss)-1
    if abs(available_tss[pos]-target_ts) < abs(available_tss[pos-1]-target_ts):
        return pos
    else:
        return pos-1

def switch_anchor(targets):
    x_min = targets['ul'][0]
    x_max = targets['ur'][0]
    y_bot = targets['ul'][1]
    y_top = targets['bl'][1]
    return {'ul': (x_min, y_top), 
             'ur': (x_max, y_top), 
             'bl': (x_min, y_bot),
             'br': (x_max, y_bot)}

def get_targets(targets, speed):
    width = abs(targets['ur'][0] - targets['ul'][0])
    height = abs(targets['br'][1] - targets['ul'][1])

    return {'ul': targets['ul'],
            'ur': (targets['ul'][0] + width*speed, targets['ul'][1]),
            'bl': (targets['ul'][0], targets['ul'][1] - height*speed),
            'br': (targets['ul'][0] + width*speed, targets['ul'][1] - height*speed)}


def _get_trials_info(eeg, eeg_ts, markers, marker_ts):
    # Create a label and trial numbers per timestamp
    # TODO: Change string labels to numerical
    STEP_SIZE = 0.01  # sec
    STEPS_PER_TRIAL = 150  # Amount of time the dot moves per trial
    fs = 1 // (eeg_ts[1]-eeg_ts[0])

    # Find which markers correspond to the start and end of a trial
    trial_start_mask = [marker[0].split(':')[0]=='trial_start' for marker in markers]
    trial_end_mask = [marker[0].split(':')[0]=='trial_end' for marker in markers]

    # Find the indices corresponding to the start and end of the trial
    trial_idc_start = np.array([_locate_pos(eeg_ts, trial) for trial in marker_ts[trial_start_mask]])
    trial_idc_end = np.array([_locate_pos(eeg_ts, trial) for trial in marker_ts[trial_end_mask]])

    goals = [marker[0].split('//')[1:] for marker in markers if marker[0].split(':')[0]=='goal_update']
    targets_main = literal_eval(markers[1][0][27:]) # Anchor is topleft, instead of bottom left
    targets_main = switch_anchor(targets_main)
    ppmm = float(markers[2][0].split(':')[-1])

    # returns 3d array: [trial_num, xcoord, ycoord]. 
    # rest trial = -1
    trial_labels = np.tile((-1, 0, 0), (trial_idc_start[0], 1)) # pre trial info
    for i, (ti_start, ti_end) in enumerate(zip(trial_idc_start, trial_idc_end)):
        trial_len = ti_end - ti_start
        speed = float(goals[i][2].split(':')[1])
        targets = get_targets(targets_main, speed)
        goal_start = np.array(targets[goals[i][0][-3:-1]])
        goal_stop = np.array(targets[goals[i][1][-3:-1]])

        if i > 0:
            rest_trial_len = ti_start - trial_idc_end[i-1]
            rest_trials = np.hstack((np.full((rest_trial_len, 1), -1),
                                     np.tile(goal_start, 
                                            (rest_trial_len, 1))))
            trial_labels = np.vstack((trial_labels,
                                      rest_trials))

        # New
        # Target moves 4.32*speed pixels every 10 ms for 150 steps
        # trial_time = 1.5s. Sleeps in between
        t = np.full((trial_len, 2), fill_value=np.nan)
        samples_per_step = STEP_SIZE * fs
        sample_idc = [round(i*samples_per_step) for i in range(int(trial_len/samples_per_step))]
        step_size = (goal_stop - goal_start)/STEPS_PER_TRIAL
        t[sample_idc] = np.linspace(goal_start+step_size, goal_stop, len(sample_idc))
        
        # Old
        # t = np.linspace(goal_start, goal_stop, ti_end - ti_start)

        t = np.hstack((np.full((t.shape[0], 1), i), t))
        trial_labels = np.vstack((trial_labels, t))

    return trial_labels, goals

def _get_experiment_data(result, markerstream_name):
    # TODO: Offset markers? (see load_grasp_data)
    marker_idx_exp_start = result[markerstream_name]['data'].index(['experiment_start'])
    marker_idx_exp_end = result[markerstream_name]['data'].index(['experiment_end'])

    eeg_idx_exp_start = _locate_pos(result['Micromed']['ts'], 
                                result[markerstream_name]['ts'][marker_idx_exp_start])
    eeg_idx_exp_end = _locate_pos(result['Micromed']['ts'],
                                result[markerstream_name]['ts'][marker_idx_exp_end])

    eeg = result['Micromed']['data'][eeg_idx_exp_start:eeg_idx_exp_end, :]
    eeg_ts = result['Micromed']['ts'][eeg_idx_exp_start:eeg_idx_exp_end]

    marker = result[markerstream_name]['data'][marker_idx_exp_start:marker_idx_exp_end]
    marker_ts = result[markerstream_name]['ts'][marker_idx_exp_start:marker_idx_exp_end]

    return eeg, eeg_ts, marker, marker_ts

def read_xdf(path):
    data, header = pyxdf.load_xdf(path)
    result = {}
    for stream in data:
        stream_name = stream['info']['name'][0]
        result[stream_name] = {}

        # Info
        result[stream_name]['fs'] = float(stream['info']['nominal_srate'][0])
        result[stream_name]['type'] = stream['info']['type'][0].lower()
        result[stream_name]['first_ts'] = float(stream['footer']['info']['first_timestamp'][0])
        result[stream_name]['last_ts'] = float(stream['footer']['info']['last_timestamp'][0])
        result[stream_name]['total_stream_time'] = result[stream_name]['last_ts'] - result[stream_name]['first_ts']
        result[stream_name]['sample_count'] = int(stream['footer']['info']['sample_count'][0])
        result[stream_name]['data_type'] = stream['info']['channel_format'][0]
        result[stream_name]['hostname'] = stream['info']['hostname'][0]

        # Data
        result[stream_name]['data'] = stream['time_series']
        result[stream_name]['ts'] = stream['time_stamps']

        # Manually added stream description
        if stream['info']['desc'][0] is not None:
            for desc in stream['info']['desc']:
                # TODO: differentiate between channel types (e.g. gtec has eeg and accelerometer channels (and more))
                if 'channels' in desc.keys():
                    result[stream_name]['channel_names'] = [ch['label'][0] for ch in desc['channels'][0]['channel']]
                if 'manufacturer' in desc.keys():
                    result[stream_name]['manufacturer'] = desc['manufacturer'][0]

    return result, data

def load_seeg(file):
    ''' Loads xdf file and returns a dict with all necessary information'''
    # import within function to not make whole module dependent on local import

    file = Path(file)
    print('Loading file: {}'.format(file))

    result, raw_data = read_xdf(str(file))

    eeg, eeg_ts, markers, markers_ts = _get_experiment_data(result, 'FollowTheDotMarkerStream')
    trials, goals = _get_trials_info(eeg, eeg_ts, markers, markers_ts)

    multiple_measurements = 'kh' not in file.parts[-2]

    seeg = {}
    seeg['subject'] = file.parts[-2] if not multiple_measurements else file.parts[-3]
    seeg['experiment_type'] = file.parts[-1].split('.xdf')[0]
    seeg['experiment_date'] = file.parts[-2] if multiple_measurements else _get_created_date(file) # Returns created date if no date folder is present
    seeg['channel_names'] = result['Micromed']['channel_names']
    seeg['eeg'] = eeg.astype(np.float64)
    seeg['eeg_ts'] = eeg_ts
    seeg['trial_labels'] = trials
    seeg['trial_goals'] = goals
    seeg['fs'] = result['Micromed']['fs']
    seeg['dtype'] = result['Micromed']['data_type']
    seeg['first_ts'] = result['Micromed']['first_ts']
    seeg['last_ts'] = result['Micromed']['last_ts']
    seeg['total_stream_time'] = result['Micromed']['total_stream_time']
    seeg['samplecount'] = result['Micromed']['sample_count']

    return seeg

def get_filenames(path_main, extension, keywords=[], exclude=['_archive']):
    ''' Recursively retrieves all files with 'extension', 
    and subsequently filters by given keywords. 

    keywords: list[str,]
        Selects file when substring exists in filename

    exlude: list[str,]
        Removes file if substring in filename or string equals
        any complete parent foldername.
    '''

    if not exists(path_main):
        print("Cannot access path <{}>. Make sure you're on the university network"\
                .format(path_main))
        raise NameError

    keywords = extension if len(keywords)==0 else keywords
    extension = f'*.{extension}' if extension[0] != '.' else f'*{extension}'
    files = [path for path in Path(path_main).rglob(extension) \
             if any(kw in path.name for kw in keywords)]

    if any(exclude):
        files = [path for path in files for excl in exclude \
                   if excl not in path.name
                   and excl not in path.parts]
    return files

def get_all_files(path, extension, keywords=[], exclude=[]):
    seeg_filenames = get_filenames(path, extension, 
                                    keywords=keywords,
                                    exclude=exclude)
    contact_filenames  = get_filenames(path, 'csv', keywords=['electrode_locations'])

    results = []
    for seeg_filename in seeg_filenames:
        for contact_filename in contact_filenames:
            if contact_filename.parts[3] == seeg_filename.parts[3]:
                loc = contact_filename
                break
            loc = []
        results += [(seeg_filename, loc)]

    return results

def load_locs(path):
    df = read_csv(path)
    locs = OrderedDict(zip(df['electrode_name_1'],
                           df['location']))
    return locs