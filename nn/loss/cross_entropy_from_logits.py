import numpy as np 
from nn.loss import Loss 
import nn.util 

class CrossEntropyFromLogits(Loss):

    def forward(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        """
        Categorical cross-entropy loss function
        y_true: (batch_size, n_classes) of true class targets (one hot)
        y_pred: (batch_size, n_classes) of predicted class logits 
        returns: scalar mean loss over batch
        """
        logits_shifted = logits - logits.max(axis=1, keepdims=True) # for stability
        logsumexp = np.log(np.exp(logits_shifted).sum(axis=1, keepdims=True)) 
        log_softmax = logits_shifted - logsumexp
        return -(y_true * log_softmax).sum(axis=1).mean().item() 
    
    def backward(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        """
        Gradient of categorical cross-entropy w.r.t. predicted logits
        y_true: (batch_size, n_classes) 
        y_pred: (batch_size, n_classes) 
        returns: (batch_size, n_classes) dL/d(logits) 
        """
        return nn.util.softmax(logits) - y_true

# Quick to use instance
cross_entropy_from_logits = CrossEntropyFromLogits()
