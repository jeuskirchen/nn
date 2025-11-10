import numpy as np 
from nn.activation import Activation

class Sigmoid(Activation):

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.divide(1, 1+np.exp(-x))
    
    def backward(self, a: np.ndarray) -> np.ndarray:
        """
        a: forward output, sigmoid(x), for some x 
        returns: derivative sigmoid'(x) = a * (1 - a) = sigmoid(x) * (1 - sigmoid(x))
        """
        return a * (1 - a)

# Quick to use instance
sigmoid = Sigmoid()
