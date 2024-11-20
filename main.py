'''
https://github.com/ShanechiLab/PyPSID

[ ] Check if PSID version = 1.10+. Then extracting the mean and add it after learning is not necessary anymore
[ ] Check n1 <= nz * i  AND   nx <= ny *i. Important to report i in paper
[ ] Extract only relevant states by setting nx = n1
[ ] Use PSID.evaluation.evalPrediction
[ ] Plot EigenValues of A Matrix of learned models and 'True' models to check the accurate learning (probably not possible with our data)

PSID tutorial: https://github.com/ShanechiLab/PyPSID/blob/main/source/PSID/example/PSID_tutorial.ipynb
'''
import sys
import logging
import cProfile
import pstats
import io
import re
from os import cpu_count
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime as dt
from collections import defaultdict

import yaml

# Setup config file.
# This is before the last imports because it needs to be 
# updated before the imported modules load the config file.
from libs import utils   # Local

logger = logging.getLogger(__name__)

try: 
    config_path = sys.argv[1]
    config = utils.load_yaml(config_path)

    with open('config.yml', 'w') as f:
        yaml.dump(utils.nested_namespace_to_dict(config), f)
except IndexError:
    # logger = logging.getLogger(__name__)
    logger.warning('No config supplied, using last available config file')
finally:
    c = utils.load_yaml('./config.yml')  
  
import run_decoder


def init_logging(results_path):

    console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(logging.WARNING)
    console_handler.setLevel(logging.INFO)
    # console_handler.setLevel(logging.DEBUG)
    
    log_filename = f'output.log'
    logging.basicConfig(format="[%(filename)10s:%(lineno)3s - %(funcName)20s()] %(message)s",
                        level=logging.INFO if not c.debug.log else logging.DEBUG,
                        handlers=[
                            logging.FileHandler(results_path/f'{log_filename}'),  # save to file
                            logging.StreamHandler(),  # print to terminal
                            # console_handler
                            ])
    return

def init_results_path(main_path, ppt_id):
    path = main_path/ppt_id
    path.mkdir(parents=True, exist_ok=True)

    # Create new folder if already exists.
    # For example when multiple sessions per sub
    dirs = sorted([d for d in path.iterdir() if d.is_dir()])
    if dirs:
        new_path = path/f'{int(dirs[-1].name) + 1}'
    else:
        new_path = path/'0'
    new_path.mkdir(parents=True, exist_ok=False)
    
    return path

def init_run(filelist: list, main_results_path: Path):

    ppt_id = filelist[0].parts[-3]

    save_path = init_results_path(main_results_path, ppt_id)
    init_logging(save_path)

    # The main loop starts here
    with cProfile.Profile() as pr:
        run_decoder.run(save_path, filelist, ppt_id)
    
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s)
    stats.dump_stats(f'{save_path}/profile.prof')

    with open(f'{save_path}/profile.txt', 'w') as f:
        ps = pstats.Stats(f'{save_path}/profile.prof', stream=f)
        ps.sort_stats(pstats.SortKey.TIME)
        ps.print_stats()

def main():

    # Setup some paths
    main_path = Path(c.learn.save_path)
    today = dt.today().strftime('%Y%m%d_%H%M')

    main_path = main_path/today
    main_path.mkdir(parents=True, exist_ok=True)

    # data_path = Path(sys.argv[2])
    data_path = Path(r'../data/')
    filenames = list(data_path.rglob('*.xdf'))  # All xdf files.

    # Combine multiple sessions of one participant
    if c.combine:
        files_per_ppt = defaultdict(list)

        for file in filenames:
            
            ppt_id = file.parts[-3]
            files_per_ppt[ppt_id].append(file)

        jobs = list(files_per_ppt.values())
    else:
        jobs = [[file] for file in filenames] 

    if c.parallel:
        
        pool = Pool(processes=cpu_count())
        for job in jobs:
            pool.apply_async(init_run, args=(job, main_path))
        pool.close()
        pool.join()

    else:
        for job in jobs:
            init_run(job, main_path)  # job = [Filename, ...]
            

if __name__=='__main__':
    main()


    
