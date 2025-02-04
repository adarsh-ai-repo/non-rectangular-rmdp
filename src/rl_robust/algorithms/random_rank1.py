import numpy as np
from typing import Tuple, Optional
from .protocol import OptimizerProtocol

class RandomRank1Optimizer(OptimizerProtocol):
    def __init__(self, num_guesses: int = 10000, tolerance: float = 1e-5):
        self.tolerance = tolerance
        self.num_guesses = num_guesses
        
    def optimize(self, input_matrix: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        S, A = input_matrix.shape
        best_value = float('-inf')
        best_b = None
        best_k = None
        
        for _ in range(self.num_guesses):
            # Generate random b
            b = np.random.rand(S * A)
            b *= self.tolerance / np.linalg.norm(b)
            
            # Generate random k
            k = np.random.randn(S)
            k -= np.mean(k)  # Ensure k sums to 0
            k /= np.linalg.norm(k)
            
            x = np.concatenate([b, k])
            value = self.objective(x, input_matrix)
            
            if value > best_value:
                best_value = value
                best_b = b
                best_k = k
                
        return best_value, best_b
        
    def objective(self, x: np.ndarray, input_matrix: np.ndarray) -> float:
        """Compute objective value for rank-1 optimization"""
        S, A = input_matrix.shape
        b, k = x[:S*A], x[S*A:]
        b_reshaped = b.reshape(S, A)
        
        numerator = np.dot(k, input_matrix @ b)
        denominator = 1 + self.tolerance * np.dot(k, b_reshaped.sum(axis=1))
        return numerator / denominator
