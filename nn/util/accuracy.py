import numpy as np 

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Exact-class accuracy for multi-class classification with one-hot targets
    given predicted class probabilities.
    y_true: (batch_size, n_classes), one-hot ground truth
    y_pred: (batch_size, n_classes), predicted class probabilities 
    Returns: scalar mean accuracy over the batch 
    """
    # Predicted class via argmax over probabilities
    pred_cls = np.argmax(y_pred, axis=1)
    # True class index via argmax over one-hot targets
    true_cls = np.argmax(y_true, axis=1)
    # Mean correctness
    return np.mean(pred_cls == true_cls).item()
