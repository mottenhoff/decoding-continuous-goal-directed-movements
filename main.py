'''
https://github.com/ShanechiLab/PyPSID

[ ] Check if PSID version = 1.10+. Then extracting the mean and add it after learning is not necessary anymore
[ ] Check n1 <= nz * i  AND   nx <= ny *i. Important to report i in paper
[ ] Extract only relevant states by setting nx = n1
[ ] Use PSID.evaluation.evalPrediction
[ ] Plot EigenValues of A Matrix of learned models and 'True' models to check the accurate learning (probably not possible with our data)

PSID tutorial: https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/example/PSID_tutorial.ipynb
'''

import logging
import cProfile
import pstats
import io
import re
import sys
from multiprocessing import Process, Pool
from pathlib import Path
from datetime import datetime as dt
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml

import run_decoder
from libs import utils

from figures.figure_cc_per_band_per_kinematic import plot_band_correlations

c = utils.load_yaml('./config.yml')
logger = logging.getLogger(__name__)

def init(main_path, id_):

    path = main_path/id_
    path.mkdir(parents=True, exist_ok=True)

    i = 0
    while True:
        new_path = path/f'{i}'
        
        try:
            new_path.mkdir(parents=False, exist_ok=False)
            path = new_path
            break
        except Exception:
            i += 1
            continue

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    
    log_filename = f'output.log'
    logging.basicConfig(format="[%(filename)10s:%(lineno)3s - %(funcName)20s()] %(message)s",
                        level=logging.INFO if not c.debug.log else logging.DEBUG,
                        handlers=[
                            logging.FileHandler(path/f'{log_filename}'),
                            # logging.StreamHandler()
                            console_handler])

    return path

def init_run(filelist: list, main_path: Path):
    ppt_id = filelist[0].parts[-2]
    id_ = int(re.findall(r'\d+', ppt_id)[0])

    save_path = init(main_path, ppt_id)

    with cProfile.Profile() as pr:
        run_decoder.run(save_path, filelist, id_)
    
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s)
    stats.dump_stats(f'{save_path}/profile.prof')

    with open(f'{save_path}/profile.txt', 'w') as f:
        ps = pstats.Stats(f'{save_path}/profile.prof', stream=f)
        ps.sort_stats(pstats.SortKey.TIME)
        ps.print_stats()


if __name__=='__main__':

    main_path = Path(c.learn.save_path)
    today = dt.today().strftime('%Y%m%d_%H%M')
    main_path = main_path/today
    main_path.mkdir(parents=True, exist_ok=True)

    filenames = [p for p in Path('./data/').rglob('*.xdf')]
    
    # Filter ppt_ids:
    if ids := []:
        filenames = [file for file in filenames if int(file.parts[-2][-2:]) in ids]

    if c.combine:
        # TODO: Switch for one or both hands.
        files_per_ppt = defaultdict(list)

        for file in filenames:
            files_per_ppt[file.parts[-2]].append(file)

        jobs = list(files_per_ppt.values())
    else:
        jobs = [[file] for file in filenames]

    if c.parallel:
        
        pool = Pool(processes=5)
        for job in jobs:
            pool.apply_async(init_run, args=(job, main_path))
        pool.close()
        pool.join()

    else:

        for job in jobs:
            init_run(job, main_path)  # job = [Filename, ...]
