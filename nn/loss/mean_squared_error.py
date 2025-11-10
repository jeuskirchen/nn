import numpy as np 
from nn.loss import Loss 

class MeanSquaredError(Loss):

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean squared error loss 
        y_true: (batch_size, n_targets)
        y_pred: (batch_size, n_targets)
        returns: scalar mean squared error
        """
        return 0.5 * ((y_pred - y_true) ** 2).mean().item() 
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Here we divide not just by the number of examples, but also number of output dimensions
        m, k = y_pred.shape
        return (y_pred - y_true) / (m * k)

# Quick to use instance
mse = MeanSquaredError()
