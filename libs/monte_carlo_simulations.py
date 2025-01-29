from typing import Callable

import numpy as np
from numpy.typing import NDArray

def random_array_swap(original: NDArray, 
                      permutable: NDArray, 
                      metric: Callable[[list, list], list],  # TODO ArrayLike?
                      alpha: float=0.05, 
                      min_block_size: float=.1, 
                      repetitions: int=100) -> float:
    '''
    original: [samples x features]
    permutable: [samples x features]  -> Same size as original
    block_size: float  -> Percentage of total signal that will 
                          left at the edges of the timeseries.
    metric: Callable that takes original and permuted as arguments,
            needs to return a single row of size n_features of 
            metrics. 
            TODO: If None return the permuted matrices? May be 
                  may be memory intensive.

    returns repetitions
    '''

    n_samples, _ = original.shape
    min_split_length = int(n_samples * min_block_size)

    sample_range = np.arange(min_split_length, n_samples - min_split_length)

    permutations = np.empty((repetitions, original.shape[1], permutable.shape[1]))
    for i in np.arange(repetitions):
        split_idx = np.random.choice(sample_range)
        
        first_part =  permutable[:split_idx, :]
        second_part = permutable[split_idx:, :]

        permuted = np.concatenate([second_part, first_part])

        permutations[i, :, :] = metric(original, permuted)

    return permutations