import logging

import numpy as np

logger = logging.getLogger(__name__)

def timeshift(eeg, xyz, t=0):
   ''' t: shift in time in ms.
   '''

   # shift_idx = np.where((ts - ts[0]) >= t*0.001)[0]  # Takes longer, but dynamic
   shift_idx = abs(round(t*0.001*1024))  # Assumes static fs

   if t == 0:
      return eeg, xyz
      
   elif t > 0: 
      eeg['data'] = eeg['data'][:-shift_idx, :]
      eeg['ts'] = eeg['ts'][:-shift_idx]
      xyz = xyz[shift_idx:, :]

   else:
      eeg['data'] = eeg['data'][shift_idx:, :]
      eeg['ts'] = eeg['ts'][shift_idx:]
      xyz = xyz[:-shift_idx, :]

   logger.info(f'applied timeshift = {t} ms')

   return eeg, xyz

