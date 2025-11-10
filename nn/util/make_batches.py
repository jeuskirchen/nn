import numpy as np 
from typing import Iterator, Tuple 

def make_batches(x_train: np.ndarray, y_train: np.ndarray, batch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Put dataset into batches of size batch_size 
    - randomly shuffle indices
    - yield consecutive slices of size `batch_size`
    - last batch may be smaller if dataset size is not divisible by batch_size

    x_train: (n_examples, n_inputs)
    y_train: (n_examples, n_outputs)
    batch_size: desired batch size
    """
    n_examples = x_train.shape[0]
    indices = np.random.permutation(n_examples)  # random order

    for start in range(0, n_examples, batch_size):
        end = min(start + batch_size, n_examples)
        idx = indices[start:end]
        yield x_train[idx], y_train[idx]
