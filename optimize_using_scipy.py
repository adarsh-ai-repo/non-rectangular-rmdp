import numpy as np
from scipy.optimize import minimize

from datamodels import PMDerivedValues, PMUserParameters


def optimize_using_slsqp_method(
    params: PMUserParameters, derived_values: PMDerivedValues
):
    """
    Optimize using SLSQP method with the new data models.

    Parameters:
    params: PMUserParameters - Contains basic parameters like S, A, beta, gamma
    derived_values: PMDerivedValues - Contains derived values like v_pi, d_pi, D_pi, H

    Returns:
    tuple: (optimal_value, b_opt, k_opt)
    """
    S, A = params.S, params.A
    v_pi, d_pi, D_pi = derived_values.v_pi, derived_values.d_pi, derived_values.D_pi
    gamma, beta = params.gamma, params.beta
    H = derived_values.H

    def objective(x):
        b, k = x[: S * A], x[S * A :]
        b_pi = H @ b
        v_pi_b = D_pi @ b_pi
        numerator = np.dot(k, v_pi) * np.dot(d_pi, b_pi)
        denominator = 1 + gamma * np.dot(k, v_pi_b)
        return -numerator / denominator  # Negative because we're maximizing

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
    x0[: S * A] *= beta / np.linalg.norm(
        x0[: S * A]
    )  # Scale b to satisfy its constraint
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
        options={"maxiter": 100, "ftol": 5e-4, "iprint": 1, "disp": True},
    )

    if result.success:
        b_opt, k_opt = result.x[: S * A], result.x[S * A :]
        optimal_value = -result.fun  # Remember we minimized the negative
        return optimal_value, b_opt, k_opt
    else:
        raise ValueError("Optimization failed: " + result.message)
