from pathlib import Path
from dataclasses import dataclass
from datetime import timedelta
from datetime import datetime as dt

import yaml

import numpy as np


@dataclass
class Info:
    datasize: int
    n_gaps: int
    n_targets: int
    ppt_id: int
    total_time: float

def load_yaml(path):

    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def print_totals(infos):

    print_metrics = lambda name, lst: print(f'{name:<10} {np.mean(lst)} +- {np.std(lst)} | n={np.sum(lst)}')

    n_samples = [i.datasize for i in infos]
    n_gaps = [i.n_gaps for i in infos]
    n_targets = [i.n_targets for i in infos]
    total_time = [i.total_time for i in infos]

    print_metrics('samples', n_samples)
    print_metrics('gaps', n_gaps)
    print_metrics('targets', n_targets)
    print_metrics('time',total_time)

    total_time = sum(total_time)
    print(f'{total_time//(60*60):02.0f}:{total_time//60%60:02.0f}:{total_time%60:2.0f}')
    # print(total_time/60/60, 's', )

main_path = Path('finished_runs\delta_cer')
infos = [Info(**load_yaml(info)) for info in main_path.rglob('info.yml')]
print_totals(infos)


print()

# data = np.stack([np.load(file) for file in main_path.rglob('*_info.npy')])

# locations = [np.load(file) for file in main_path.rglob('*_locations.npy')]


# np.savetxt('elec_info.csv', data, delimiter=',')f