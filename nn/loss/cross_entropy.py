import numpy as np 
from nn.loss import Loss 

class CrossEntropy(Loss):

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Categorical cross-entropy loss function
        Collapses to binary cross-entropy for n_classes=2, but requiring 2 model outputs: (1-p, p)
        y_true: (batch_size, n_classes) of true class probabilities
        y_pred: (batch_size, n_classes) of predicted class probabilities
        returns: scalar mean loss over batch
        """
        # Alternatively, cross_entropy_from_logits, and then use that on top of logits,
        # rather than explicitly using a softmax layer
        eps = 1e-10 # small number for numerical stability, in particular to avoid log(0)
        p = np.clip(y_pred, eps, 1-eps)
        return -(y_true * np.log(p)).sum(axis=1).mean().item() 
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Gradient of categorical cross-entropy w.r.t. predicted probabilities
        y_true: (batch_size, n_classes) one-hot or class-probabilities
        y_pred: (batch_size, n_classes) probabilities (e.g., softmax outputs)
        returns: (batch_size, n_classes) dL/d(y_pred)
        """
        eps = 1e-10 # small number for numerical stability, in particular to avoid log(0)
        p = np.clip(y_pred, eps, 1)
        return -(y_true / p)

# Quick to use instance
cross_entropy = CrossEntropy()
