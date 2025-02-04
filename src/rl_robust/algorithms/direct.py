import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional

from .base import BaseOptimizer

class DirectOptimizer(BaseOptimizer):
    def __init__(self, num_trials: int = 8, time_limit: float = 120.0):
        super().__init__()
        self.num_trials = num_trials
        self.time_limit = time_limit
        
    def optimize(self, input_matrix: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        if input_matrix.ndim != 2:
            raise ValueError("Input matrix must be 2D")
            
        n = input_matrix.shape[1]
        cons = {"type": "ineq", "fun": lambda x: 1 - np.linalg.norm(x, 2)}
        bounds = [(0, None) for _ in range(n)]
        
        best_x = None
        best_obj = -np.inf
        
        for _ in range(self.num_trials):
            x0 = np.random.rand(n)
            x0 /= np.linalg.norm(x0, 2)
            
            try:
                res = minimize(
                    lambda x: -np.linalg.norm(input_matrix @ x, 2),
                    x0,
                    method="SLSQP",
                    constraints=cons,
                    bounds=bounds,
                    options={"maxiter": 10, "ftol": 5e-4}
                )
                
                if res.success and round(-res.fun, 3) > round(best_obj, 3):
                    best_x = res.x
                    best_obj = -res.fun
            except Exception:
                continue
                
        if best_x is None:
            raise RuntimeError("Optimization failed for all attempts")
            
        return best_obj, best_x
