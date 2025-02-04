import numpy as np
from typing import Tuple, Optional

from .base import BaseOptimizer
from .direct import DirectOptimizer

class BisectionOptimizer(BaseOptimizer):
    def __init__(self, beta: float = 0.1, gamma: float = 0.9, max_iter: int = 16):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.direct_optimizer = DirectOptimizer()
        
    def optimize(self, input_matrix: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        min_lambda = 0
        max_lambda = 10 * self.beta
        
        for _ in range(self.max_iter):
            lambda_value = (min_lambda + max_lambda) / 2
            new_value = (
                self.gamma * self.beta * 
                self.direct_optimizer.optimize(input_matrix)[0] - 
                lambda_value
            )
            
            if (max_lambda - min_lambda) < self.tolerance:
                break
            elif new_value < 0:
                max_lambda = lambda_value
            elif new_value > 0:
                min_lambda = lambda_value
            else:
                break
                
        return lambda_value, None
