import pyxdf

def read_xdf(path):
    data, header = pyxdf.load_xdf(path)
    result = {}
    for stream in data:
        stream_name = stream['info']['name'][0]
        result[stream_name] = {}

        # Info
        result[stream_name]['fs'] = float(stream['info']['nominal_srate'][0])
        result[stream_name]['type'] = stream['info']['type'][0].lower()

        if 'footer' in stream:
            result[stream_name]['first_ts'] = float(stream['footer']['info']['first_timestamp'][0])
            result[stream_name]['last_ts'] = float(stream['footer']['info']['last_timestamp'][0])
            result[stream_name]['total_stream_time'] = result[stream_name]['last_ts'] - result[stream_name]['first_ts']
            result[stream_name]['sample_count'] = int(stream['footer']['info']['sample_count'][0])
        else:
            result[stream_name]['first_ts'] = stream['time_stamps'][0]
            result[stream_name]['last_ts'] = stream['time_stamps'][-1]
            result[stream_name]['total_stream_time'] = result[stream_name]['last_ts'] - result[stream_name]['first_ts']
            result[stream_name]['sample_count'] = stream['time_stamps'].size
        
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


if __name__=='__main__':
    path = r''
    result = read_xdf(path)
