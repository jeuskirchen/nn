import numpy as np 
from typing import Callable
from copy import deepcopy 
import nn.layer 
import nn.util
import nn.loss 

class Model:
    # Same idea as torch's nn.Sequential
    
    def __init__(self, layers: list[nn.layer.Layer]):
        self.layers = layers
        self.history = {
            "running_loss": [],
            "epoch_loss": [],
        } 
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        For convenience 
        Returns last-layer activations from forward pass 
        """
        return self.forward(x)[-1]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Returns activations of all layers from the forward pass 
        """
        a_list = [x]
        a = x
        for layer in self.layers: 
            a = layer(a) 
            a_list.append(a)
        return a_list 

    def backward(self, y_true: np.ndarray, a_list: list[np.ndarray], 
                 loss_fn: nn.loss.Loss, 
                 reg_coeff: float) -> list[np.ndarray]:
        """
        Returns gradient as list of arrays (one array per layer)
        """
        n_layers = len(self.layers) 
        grad_list = [None] * n_layers # list of gradients per layer 
        y_pred = a_list[-1] # last-layer activations
        layer = self.layers[-1]
        delta = loss_fn.backward(y_true, y_pred) * layer.activation_fn.backward(y_pred) # last-layer delta
        for l in reversed(range(n_layers)):
            layer = self.layers[l] 
            grad, delta = layer.backward(a_list[l], delta, reg_coeff)
            grad_list[l] = grad 
        return grad_list 

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, 
            n_epochs: int, batch_size: int, lr: float, loss_fn: Callable, 
            reg_coeff: float = 0.0, verbose: bool = False, 
            gradient_check: bool = False) -> None:
        """
        Iteratively updates model parameters via backpropagation
        """
        # Training loop 
        for epoch in range(1, n_epochs+1):
            if verbose:
                print("Epoch", epoch, end=" ")
            running_loss = 0.0
            batches = nn.util.make_batches(x_train, y_train, batch_size) 
            for x_batch, y_batch in batches:
                # Forward pass
                a_list = self.forward(x_batch) # (n_layers, (batch_size, n_classes))
                # Backward pass 
                grad_list = self.backward(y_batch, a_list, loss_fn, reg_coeff) # (n_layers, (batch_size, n_out)) 
                # Optional: gradient checking (see supplementary material)
                if gradient_check:
                    return nn.util.gradient_check(self, grad_list, loss_fn, x_train, y_train)
                # Parameter update (step)
                for l, layer in enumerate(self.layers): 
                    layer.params = layer.params - lr * grad_list[l] 
                # Keeping track of loss for this step 
                y_pred = a_list[-1]
                loss = loss_fn(y_batch, y_pred)
                self.history["running_loss"].append(loss)
                running_loss += loss * x_batch.shape[0] 
            # Keeping track of loss for this epoch 
            epoch_loss = running_loss / len(x_train)
            self.history["epoch_loss"].append(epoch_loss)
            if verbose:
                print(epoch_loss)
        if verbose:
            print("Finished.")
