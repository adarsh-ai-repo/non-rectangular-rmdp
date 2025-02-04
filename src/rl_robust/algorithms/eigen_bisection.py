import numpy as np
from typing import Tuple, Optional
from .protocol import OptimizerProtocol

class EigenBisectionOptimizer(OptimizerProtocol):
    def __init__(self, beta: float = 0.1, gamma: float = 0.9, max_iter: int = 16, tolerance: float = 1e-5):
        self.tolerance = tolerance
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        
    def get_max_eigenvalue(self, A: np.ndarray) -> float:
        """Compute maximum eigenvalue using eigenvalue heuristics"""
        ATA = A.T @ A
        s, V = np.linalg.eigh(ATA)
        
        idx = np.argsort(s)[::-1]
        s = s[idx]
        V = V[:, idx]
        
        for i in range(V.shape[1]):
            v_plus_norm = np.linalg.norm(np.maximum(V[:, i], 0))
            v_minus_norm = np.linalg.norm(np.minimum(V[:, i], 0))
            if v_plus_norm < v_minus_norm:
                V[:, i] = -V[:, i]
                
        V_plus = np.maximum(V, 0)
        U = V_plus / np.linalg.norm(V_plus, axis=0)
        scores = s * np.einsum('ij,ij->j', V, U)
        
        j = np.argmax(scores)
        u_j = U[:, j]
        
        return np.linalg.norm(A @ u_j)
        
    def optimize(self, input_matrix: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        min_lambda = 0
        max_lambda = 10 * self.beta
        
        for _ in range(self.max_iter):
            lambda_value = (min_lambda + max_lambda) / 2
            new_value = (
                self.gamma * self.beta * 
                self.get_max_eigenvalue(input_matrix) - 
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
