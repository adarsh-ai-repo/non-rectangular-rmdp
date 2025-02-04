from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple

class BaseOptimizer(ABC):
    """Base class for optimization algorithms"""
    
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance
        
    @abstractmethod
    def optimize(self, input_matrix: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Optimize the given input matrix
        Returns:
            Tuple of (optimal_value, optimal_solution)
        """
        pass
