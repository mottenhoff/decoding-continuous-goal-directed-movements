import logging

import numpy as np

logger = logging.getLogger(__name__)

def select_top_correlated(y, z, n_features):
    # Significantly correlated --> n_dims needs to be constant, so
    # need to select n_highest_correlated

    corrcoefs = np.abs(np.corrcoef(np.hstack([y, z]), rowvar=False)[-z.shape[-1]:, :y.shape[1]])
    sorted_ccs = np.argsort(corrcoefs)

    selected_features = []
    for feature in sorted_ccs.ravel()[::-1]:

        if feature not in selected_features:
            selected_features.append(feature)
        
        if len(selected_features) == n_features:
            logger.info(f'Selected {n_features} features: {np.array(selected_features).astype(np.int16)}')
            return np.array(selected_features)