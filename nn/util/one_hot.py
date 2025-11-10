import numpy as np 

def one_hot(y_flat: np.ndarray, n_classes: int) -> np.ndarray:
    """
    y_flat: (batch_size,)
    returns: (batch_size, n_classes)
    """
    batch_size = y_flat.shape[0]
    y_onehot = np.zeros((batch_size, n_classes), dtype=int)
    y_onehot[np.arange(batch_size), y_flat] = 1
    return y_onehot
