import numpy as np
from typing import Tuple, Optional
from .protocol import OptimizerProtocol

class RandomKernelOptimizer(OptimizerProtocol):
    def __init__(self, num_samples: int = 10000, tolerance: float = 1e-5):
        self.tolerance = tolerance
        self.num_samples = num_samples
        
    def project_to_simplex(self, v: np.ndarray) -> np.ndarray:
        """Project vector onto probability simplex"""
        n = len(v)
        sorted_v = np.sort(v)[::-1]
        cumulative_sum = np.cumsum(sorted_v)
        rho = np.where(sorted_v > (cumulative_sum - 1) / (np.arange(1, n + 1)))[0][-1]
        theta = (cumulative_sum[rho] - 1) / (rho + 1)
        return np.maximum(v - theta, 0)
        
    def sample_random_kernel(self, nominal_kernel: np.ndarray, beta: float) -> np.ndarray:
        """Sample random transition kernel"""
        noise = np.random.normal(0, beta, size=nominal_kernel.shape)
        noise = noise / np.linalg.norm(noise)
        noise = beta * noise * np.clip(1 - np.random.exponential(1/nominal_kernel.shape[0]), 0, 1)
        P = nominal_kernel + noise
        
        for s in range(P.shape[0]):
            for a in range(P.shape[1]):
                P[s, a, :] = self.project_to_simplex(P[s, a, :])
                
        return P
        
    def optimize(self, input_matrix: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        best_value = float('-inf')
        best_kernel = None
        
        for _ in range(self.num_samples):
            kernel = self.sample_random_kernel(input_matrix, self.tolerance)
            value = np.linalg.norm(kernel)
            
            if value > best_value:
                best_value = value
                best_kernel = kernel
                
        return best_value, best_kernel
