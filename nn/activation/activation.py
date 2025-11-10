import numpy as np 
from abc import ABC, abstractmethod 

class Activation(ABC):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass 
    
    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        pass 
