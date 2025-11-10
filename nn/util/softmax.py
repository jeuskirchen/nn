import numpy as np 

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """ 
    x: (n_batch, n_inputs, n_features) 
    """
    h = x - x.max(axis=1, keepdims=True)
    z = np.exp(h/temperature)
    return z/z.sum(axis=1, keepdims=True) 
