import numpy as np 

def prepend_ones(x: np.ndarray) -> np.ndarray:
    """
    Prepends ones to the feature dimension so that each [x1,...,xn] becomes [1,x1,...,xn]
    x: (batch_size, n_features)
    returns: (batch_size, 1+n_features)
    """
    ones = np.ones(x.shape[0]).reshape(-1, 1)
    x_prepended = np.concatenate([ones, x], axis=1)
    return x_prepended 
