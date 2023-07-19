import logging

import numpy as np
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def select_top_correlated(y, z, n_features):
    # Significantly correlated --> n_dims needs to be constant, so
    # need to select n_highest_correlated
    # correlations = [(i, pearsonr(ch, zdim).statistic) for i, ch in enumerate(y.T) for zdim in z.T]
    # best_n_features = np.argsort(np.abs(correlations))[-n_features:]


    # for pair in product(y.T, z.T):

    corrcoefs = np.abs(np.corrcoef(np.hstack([y, z]), rowvar=False)[-z.shape[-1]:, :y.shape[1]])

    # # # Highest summed correlation
    # # max_summed_cc = np.abs(corrcoefs[:, :y.shape[1]]).sum(axis=0)
    # # return np.argsort(max_summed_cc)[-n_features:]  

    # Sort each dimension seperately
    # sorted_ccs = np.vstack([np.argsort(np.abs(row)) for row in corrcoefs])
    sorted_ccs = np.argsort(corrcoefs, axis=1)

    # Gets highest for each dim, iteratively
    features = set()
    for ccs in sorted_ccs.T[::-1, :]:
        n_to_add = min((n_features - len(features)), ccs.size)

        features = features | set(ccs[:n_to_add])
        
        if len(features) == n_features:
            logger.info(f'selected features (ch_nums): {features}')
            logger.info('Summed correlation: ' + ' | '.join([f'{c:.2f}' for c in corrcoefs[:, list(features)].sum(axis=0)]))
            return np.array(list(features))
