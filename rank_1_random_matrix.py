import time
from datetime import datetime

import numpy as np

from datamodels import AlgorithmPerformanceData, PMDerivedValues, PMUserParameters


def optimize_using_random_rank_1_kernel(
    params: PMUserParameters,
    derived_values: PMDerivedValues,
    performance_data: AlgorithmPerformanceData,
    rc_hash: str = "unknown",
    num_guesses: int = 10000,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """
    Optimize using random rank-1 kernel with the new data models.

    Parameters:
    params: PMUserParameters - Contains basic parameters like S, A, gamma, beta
    derived_values: PMDerivedValues - Contains derived values like v_pi, d_pi, D_pi, H
    performance_data: AlgorithmPerformanceData - Dictionary to store performance metrics
    rc_hash: str - Hash of random components for tracking
    num_guesses: int - Number of random guesses to try

    Returns:
    tuple: (best_value, best_b, best_k) where
        best_value: float - The maximum objective value found
        best_b: ndarray - The corresponding b vector
        best_k: ndarray - The corresponding k vector
    """
    # Access parameters (equivalent to pm.S, pm.A, etc.)
    S, A = params.S, params.A
    v_pi, d_pi, D_pi = derived_values.v_pi, derived_values.d_pi, derived_values.D_pi
    gamma, beta = params.gamma, params.beta

    def objective(x):
        b, k = x[: S * A], x[S * A :]
        # Use H from derived_values instead of pm.H
        b_pi = derived_values.H @ b
        v_pi_b = D_pi @ b_pi
        numerator = np.dot(k, v_pi) * np.dot(d_pi, b_pi)
        denominator = 1 + gamma * np.dot(k, v_pi_b)
        return numerator / denominator

    best_value = float("-inf")
    best_b = None
    best_k = None
    start_time = time.time()

    for i in range(num_guesses):
        # Generate random b
        b = np.random.rand(S * A)
        b *= beta / np.linalg.norm(b)

        # Generate random k
        k = np.random.randn(S)
        k -= np.mean(k)  # Ensure k sums to 0
        k /= np.linalg.norm(k)

        x = np.concatenate([b, k])
        value = objective(x)

        # At the end of the iteration
        iteration_time = time.time() - start_time
        performance_data["algorithm_name"].append("random_rank_1_kernel")
        performance_data["iteration_count"].append(i + 1)
        performance_data["time_taken"].append(iteration_time)
        performance_data["Penalty"].append(float(value))
        performance_data["S"].append(params.S)
        performance_data["A"].append(params.A)
        performance_data["beta"].append(params.beta)
        performance_data["hash"].append(rc_hash)
        performance_data["start_time"].append(datetime.fromtimestamp(start_time))

        if value > best_value:
            best_value = value
            best_b = b
            best_k = k

    return best_value, best_b, best_k
