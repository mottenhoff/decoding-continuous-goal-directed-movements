from pathlib import Path

import numpy as np

main_path = Path('results/20230918_1437')

data = np.stack([np.load(file) for file in main_path.rglob('*_info.npy')])

locations = [np.load(file) for file in main_path.rglob('*_locations.npy')]


np.savetxt('elec_info.csv', data, delimiter=',')