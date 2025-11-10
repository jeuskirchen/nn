import numpy as np 
from abc import ABC, abstractmethod 

class Loss(ABC):

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.forward(y_true, y_pred)
    
    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass 
    
    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass 
