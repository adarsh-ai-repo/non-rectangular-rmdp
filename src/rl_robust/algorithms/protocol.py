from typing import Protocol, Tuple, Optional
import numpy as np

class OptimizerProtocol(Protocol):
    """Protocol defining the interface for optimization algorithms"""
    
    tolerance: float
    
    def optimize(self, input_matrix: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Optimize the given input matrix
        Returns:
            Tuple of (optimal_value, optimal_solution)
        """
        ...
