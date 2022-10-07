import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression

def select_k_best(y, z, n_dims):
    '''
    Selects n_dims//zdims. Could be less if overlap in 
    predictive features
    
    returns list of indices
    '''

    scores = np.empty((z.shape[1], y.shape[1]), dtype=np.int32)

    for dim in np.arange(z.shape[1]):
        s = SelectKBest(f_regression, k=n_dims//z.shape[1]) \
                       .fit(y, z[:, dim])
        scores[dim, :] = np.argsort(s.scores_)

    features = np.array([], dtype=np.int32)
    for fdim in np.arange(scores.shape[1]):
        for zdim in np.arange(scores.shape[0]):

            f = scores[zdim, fdim]

            if f not in features:
                features = np.append(features, f)

            if len(features) == n_dims:
                return features