from pathlib import Path
import numpy as np

def parse_log_file(path):
    ppt = path.parts[2]

    all_values = []
    with open(path, 'r') as f:

        values = []

        # Find the start
        while True:

            line = f.readline()

            if 'run_decoder' in line and ppt in line:
                break

            if line == '':
                print(f'No channels found for {ppt}')
                return
        
        # Fine the channels
        while True:

            line = f.readline()
            
            if 'Saved to results' in line:
                break

            if 'top_correlated' not in line:
                continue
            
            # Line 1
            line = line.strip().split()

            try:
                start = line.index('[') + 1
                values = [int(v) for v in line[start:]]
            except ValueError:
                # In case of triple digit channel number
                values = []
                start = line.index('correlation:') + 1
                values = [int(line[start][1:])] + [int(v) for v in line[start + 1:]]

            # Line 2
            line = f.readline()
            line = line[:-2].strip().split()
            values += [int(v) for v in line]

            all_values += [values]
        
    np.save(path.parent/'selected_channels.npy', np.array(all_values))


def main():

    main_path = Path(r'finished_runs\delta')
    main_path = Path(r'finished_runs\alphabeta')
    # main_path = Path(r'finished_runs\bbhg')

    for path in main_path.rglob('*/output.log'):
        parse_log_file(path)

if __name__=='__main__':

    main()