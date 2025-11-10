import numpy as np 
from typing import Callable 
from copy import deepcopy 

def gradient_check(model, grad_list: list[np.ndarray], loss_fn: Callable, 
                   x_train: np.ndarray, y_train: np.ndarray) -> list[np.ndarray]:
    eps = 1e-4
    grad_errs = [] # list of gradient error matrices (one matrix per layer, same shape as param matrix)
    for l, layer in enumerate(model.layers):
        layer_grad_errs = []
        for i in range(len(layer.params)):
            for j in range(len(layer.params[i])):
                # J(θ + EPSILON)
                model_plus = deepcopy(model)
                model_plus.layers[l].params[i, j] += eps 
                loss1 = loss_fn(y_train, model_plus(x_train))
                # J(θ - EPSILON)
                model_minus = deepcopy(model)
                model_minus.layers[l].params[i, j] -= eps 
                loss2 = loss_fn(y_train, model_minus(x_train)) 
                # (J(θ + EPSILON) - J(θ - EPSILON)) / (2 * EPSILON)
                grad_numeric = (loss1 - loss2) / (2 * eps)  # numeric approx of gradient 
                grad = grad_list[l][i, j]  # gradient to check
                grad_err = abs(grad - grad_numeric)  # error between analytical and numerical gradients 
                layer_grad_errs.append(grad_err)
        layer_grad_errs = np.array(layer_grad_errs).reshape(layer.params.shape) # put in same shape as param matrix
        grad_errs.append(layer_grad_errs) 
    return grad_errs
