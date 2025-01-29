from pathlib import Path
import numpy as np
from libs.utils import load_yaml

path = Path('./finished_runs/delta_lap')

files = path.rglob('info.yml')
recording_times = np.array([float(load_yaml(f).total_time) for f in files])

n_ppts = recording_times.size
print(recording_times.mean()/60, recording_times.std()/60)

print()
