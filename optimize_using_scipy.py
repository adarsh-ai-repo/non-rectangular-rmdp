import time
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from datamodels import AlgorithmPerformanceData, PMDerivedValues, PMRandomComponents, PMUserParameters


def optimize_using_slsqp_method(
    params: PMUserParameters,
    derived_values: PMDerivedValues,
    random_components: PMRandomComponents,
    start_time: float,
    penalties: list[float],
    iteration: int,
    performance_data: AlgorithmPerformanceData,
) -> None:
    """
    Optimize using SLSQP method with the new data models.

    Parameters:
    params: PMUserParameters - Contains basic parameters like S, A, beta, gamma
    derived_values: PMDerivedValues - Contains derived values like v_pi, d_pi, D_pi, H

    Returns:
    tuple: (optimal_value, b_opt, k_opt, performance_data)
    """

    S, A = params.S, params.A
    v_pi, d_pi, D_pi = derived_values.v_pi, derived_values.d_pi, derived_values.D_pi
    gamma, beta = params.gamma, params.beta
    H = derived_values.H

    # Store the latest function value for access in the callback

    def objective(x):
        b, k = x[: S * A], x[S * A :]
        b_pi = H @ b
        v_pi_b = D_pi @ b_pi
        numerator = np.dot(k, v_pi) * np.dot(d_pi, b_pi)
        denominator = 1 + gamma * np.dot(k, v_pi_b)
        obj_value = -numerator / denominator  # Negative because we're maximizing
        penalties.append(obj_value)  # Store the latest value
        return obj_value

    def calculate_penalty() -> float:
        """
        float: Penalty value calculated as gamma * (-latest_obj_value)
        """
        return -1 * gamma * (min(penalties))

    def constraint_b(x):
        b = x[: S * A]
        return beta - np.linalg.norm(b, ord=2)

    def constraint_k(x):
        k = x[S * A :]
        return 1 - np.linalg.norm(k, ord=2)

    def constraint_k_sum(x):
        k = x[S * A :]
        return np.sum(k)  # This should equal 0

    constraints = [
        {"type": "ineq", "fun": constraint_b},
        {"type": "ineq", "fun": constraint_k},
        {"type": "eq", "fun": constraint_k_sum},
    ]

    x0 = np.random.rand(S * A + S)  # Initial guess
    x0[: S * A] *= beta / np.linalg.norm(x0[: S * A])  # Scale b to satisfy its constraint
    x0[S * A :] -= np.mean(x0[S * A :])  # Ensure initial k sums to 0
    x0[S * A :] /= np.linalg.norm(x0[S * A :])  # Scale k to satisfy its constraint

    # Correct bounds: b is non-negative, k is unbounded
    bounds = [(0, None)] * (S * A) + [(None, None)] * S

    _result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100, "ftol": 5e-4, "iprint": 1, "disp": True},
    )

    current_time = time.time()
    time_taken = current_time - start_time

    # Calculate penalty using the latest objective function value
    penalty = calculate_penalty()

    # Update performance data
    performance_data["algorithm_name"].append("slsqp_method")
    performance_data["iteration_count"].append(iteration)  # 1-based indexing
    performance_data["time_taken"].append(time_taken)
    performance_data["j_pi"].append(derived_values.j_pi - penalty)
    performance_data["S"].append(S)
    performance_data["A"].append(A)
    performance_data["beta"].append(beta)
    performance_data["hash"].append(random_components.md5_hash)  # Placeholder hash
    performance_data["start_time"].append(datetime.fromtimestamp(start_time))

    # Increment iteration counter

    # if result.success:
    #     b_opt, k_opt = result.x[: S * A], result.x[S * A :]
    #     optimal_value = -result.fun  # Remember we minimized the negative
    #     return optimal_value, b_opt, k_opt
    # else:
    #     raise ValueError("Optimization failed: " + result.message)
