import numpy as np


def target_vector(eeg, trials, xyz):
        # TODO: Not sure if correct
        return np.hstack((eeg, trials[:, 1:] - xyz))

        
# if c.target_vector:
#     logger.info('Calculating target vector')
#     if not c.pos:
#         logger.error(f'Target vector can only be calculated if position is used as Z.')
        
#     # TODO: CHANGE THIS, THIS IS ERROR PRONE
#     #       Supply Trial information (including a target vector?) 