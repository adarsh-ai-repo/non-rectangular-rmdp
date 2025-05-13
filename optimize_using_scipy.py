import time
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from datamodels import AlgorithmPerformanceData, PMDerivedValues, PMRandomComponents, PMUserParameters


# Algorithm 1 (Ours)
def optimize_using_slsqp_method(
    params: PMUserParameters,
    derived_values: PMDerivedValues,
    random_components: PMRandomComponents,
    start_time: float,
    robust_returns: list[float],
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
        # penalties.append(obj_value)  # Store the latest value
        return obj_value

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

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100, "ftol": params.tolerance, "iprint": 0, "disp": True},
    )

    current_time = time.time()
    time_taken = current_time - start_time

    # Calculate penalty using the latest objective function value
    # penalty = calculate_penalty()

    if result.success:
        b_opt, k_opt = result.x[: S * A], result.x[S * A :]
        b_opt = b_opt / np.linalg.norm(b_opt, 2)
        k_opt = params.beta * k_opt / np.linalg.norm(k_opt, 2)
        _optimal_value = -result.fun  # Remember we minimized the negative
    else:
        return

    P_n = random_components.P.reshape(S * A * S) - np.hstack([b_opt * k_i for k_i in k_opt])
    # P_n = project_simplex(P_n).reshape(S, A, S)
    for s in range(S):
        for a in range(A):
            summ = np.sum(P_n[s, a])
            assert all(np.abs(P_n[s, a] - (P_n[s, a] / summ)) < 0.001)

    v_pi_updated = PMDerivedValues.compute_value_function(
        np.einsum("sa,sab->sb", random_components.pi, P_n),  # P_pi from P_n
        derived_values.R_pi,
        params.gamma,
    )

    # Calculate expected return under current policy and transition kernel
    robust_return = float(derived_values.mu @ v_pi_updated)
    robust_returns.append(robust_return)

    # Update performance data
    performance_data["algorithm_name"].append("slsqp_method_simplex_projection")
    performance_data["iteration_count"].append(iteration)  # 1-based indexing
    performance_data["time_taken"].append(time_taken)
    performance_data["j_pi"].append(min(robust_returns))
    performance_data["S"].append(S)
    performance_data["A"].append(A)
    performance_data["beta"].append(beta)
    performance_data["hash"].append(random_components.md5_hash)  # Placeholder hash
    performance_data["start_time"].append(datetime.fromtimestamp(start_time))
    performance_data["nominal_return"].append(derived_values.j_pi)

    # Increment iteration counter
