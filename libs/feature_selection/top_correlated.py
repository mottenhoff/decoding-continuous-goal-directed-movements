import logging

import numpy as np
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

def select_top_correlated(y, z, n_dims):

    scores = np.empty((z.shape[1], n_dims), dtype=np.int32)

    for dim in np.arange(z.shape[1]):
        cc = np.array([pearsonr(ch, z).statistic[0] for ch in y.T])
        sorted_chs = np.argsort(np.abs(cc))
        scores[dim, :] = sorted_chs[-n_dims:] #, np.newaxis]
        
        logger.info(f'Selected features for dim {dim}:')
        logger.info(' | '.join([f'{ch}: {ch_cc:.2f}' for ch, ch_cc in list(zip(sorted_chs, cc[sorted_chs]))[-n_dims:]]))
        
    features = np.array([], dtype=np.int16)

    for fdim in np.arange(scores.shape[1]):
        for zdim in np.arange(scores.shape[0]):

            f = scores[zdim, fdim]

            if f not in features:
                features = np.append(features, f)
            
            if features.size == n_dims:
                logger.info(f'selected features (ch_nums): {features}')
                return features






