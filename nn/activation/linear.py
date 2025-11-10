import numpy as np 
from nn.activation import Activation

class Linear(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

# Quick to use instance
linear = Linear()
