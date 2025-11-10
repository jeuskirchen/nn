import numpy as np  
import nn.activation
import nn.util 

class Layer:

    def __init__(self, n_in: int, n_out: int, 
                 activation_fn: nn.activation.Activation = nn.activation.linear,
                 std: float = 0.01):
        # If no activation function is passed, it uses nn.activation.linear, i.e. no nonlinearity is applied 
        # std: standard deviation for parameter initialization 
        self.params = np.random.normal(0, std, size=(n_in+1, n_out)) 
        self.activation_fn = activation_fn
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch_size, n_in)
        returns: (batch_size, n_out)
        """
        h = nn.util.prepend_ones(x) 
        return self.activation_fn(h @ self.params)
    
    def backward(self, a: np.ndarray, delta: np.ndarray, reg_coeff: float = 0.01) -> np.ndarray:
        """
        a: activation tensor from forward pass through this layer (batch_size, n_units)
        delta: ... (batch_size, ...)
        reg_coeff: regularization coefficient λ 
        """
        batch_size = a.shape[0]
        # Activations, with constant at index 0
        h = nn.util.prepend_ones(a) # (batch_size, n_{l}+1)
        # Weights (i.e. non-bias parameters)
        weights = self.params[1:] # (n_{l}, n_{l+1})
        # Gradient for layer l parameters (incl. bias)
        # grad: dJ/dW 
        grad = (h.T @ delta) / batch_size # (n_{l}+1, n_{l+1})
        # L2 regularization 
        grad[1:, :] += reg_coeff * weights  
        # Prepare delta for layer l-1
        # delta: dJ/dz 
        delta = (delta @ weights.T) * self.activation_fn.backward(a) 
        return grad, delta
