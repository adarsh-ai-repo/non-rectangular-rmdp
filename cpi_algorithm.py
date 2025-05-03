import time
from datetime import datetime

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from rich.progress import track
from scipy.optimize import minimize

from datamodels import (
    AlgorithmPerformanceData,
    PMDerivedValues,
    PMRandomComponents,
    PMUserParameters,
)

S = "S"
SAS = "SAS"
SA = "SA"


def run_cpi_algorithm(
    params: PMUserParameters,
    random_components: PMRandomComponents,
    derived_values: PMDerivedValues,
    performance_data: AlgorithmPerformanceData,
    iteration_number: int,
    best_robust_return: float,
    max_iter: int = 300,
) -> tuple[float, int]:
    """
    Implements the CPI Algorithm 3.2 for Robust Policy Evaluation.

    Finds the worst-case transition kernel P within an L1 uncertainty ball around
    a nominal kernel P_hat, and computes the expected return under this worst-case kernel.

    Args:
        params: PMUserParameters containing S, A, gamma, tolerance, beta.
        random_components: PMRandomComponents containing P (nominal kernel P_hat),
                           pi (policy), R (reward).
        max_iter: Maximum number of iterations for the algorithm.

    Returns:
        tuple: (best_robust_return, iteration_number)
            - best_robust_return: The best (minimum) robust return found
            - iteration_number: The updated iteration number

    Raises:
        RuntimeError: If time or memory limits are exceeded
    """

    P_hat: Float[np.ndarray, "S A S"] = random_components.P

    # f(P) := (1 / (1 - γ)) Σ_(s,a,s') d_pi_hat(s) π(a|s) A^π_P̂(s,a,s') P(s'|s,a)
    # where A^π_P(s,a,s') := γ [ P(s'|s,a) v^π_P(s') - Σ_(s") P(s"|s,a) v^π_P(s') ]

    P_n: Float[np.ndarray, "S A S"] = P_hat.copy()

    # Get hash value of random components
    rc_hash = random_components.md5_hash
    start_time = time.time()
    for n in track(list(range(max_iter))):
        x: Float[np.ndarray, "1 A S"] = P_n - jnp.einsum("ias->as", P_n)[None, :, :]
        # 2. Calculate value function v^pi_{P_n}
        v_pi_Pn: Float[np.ndarray, "S"] = PMDerivedValues.compute_value_function(
            np.einsum("sa,sab->sb", random_components.pi, P_n), derived_values.R_pi, params.gamma
        )
        A_pi_Pn: Float[np.ndarray, "S A S"] = params.gamma * (P_n - x) * v_pi_Pn[:, None, None]

        fpn_list: list[float] = []

        def objective(p: Float[np.ndarray, "SAS"]) -> float:
            p_reshaped = p.reshape(params.S, params.A, params.S)
            fpn = jnp.einsum(
                "s,sa,sai,ias->", derived_values.d_pi, random_components.pi, A_pi_Pn, p_reshaped
            ).item()
            fpn_list.append(fpn)
            return fpn

        # Define bounds for each probability: 0 ≤ P(s'|s,a) ≤ 1
        bounds = [(0, 1)] * (params.S * params.A * params.S)

        # Define the L2 norm constraint: ||P-P_hat||_2 ≤ β
        def constraint_l2_norm(p: Float[np.ndarray, "SAS"]) -> float:
            p_hat_reshaped = P_hat.reshape(params.S * params.A * params.S)
            return params.beta - float(np.linalg.norm(p - p_hat_reshaped, ord=2))

        # Define the probability sum constraint: ∑_{s'} P(s'|s,a) = 1 for all s,a
        def constraint_prob_sum(p: Float[np.ndarray, "SAS"]) -> Float[np.ndarray, "SA"]:
            p_hat_reshaped = p.reshape(params.S, params.A, params.S)
            return (np.sum(p_hat_reshaped, axis=2) - 1.0).reshape(params.A * params.S)

        # Create constraint dictionaries for scipy.optimize.minimize
        constraints = [
            {"type": "ineq", "fun": constraint_l2_norm},  # L2 norm constraint
            {"type": "eq", "fun": constraint_prob_sum},  # Probability sum constraint
        ]
        try:
            result = minimize(
                objective,
                P_n.reshape(params.S * params.A * params.S),  # Flatten P_n for optimization
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 100, "ftol": params.tolerance, "iprint": 1, "disp": False},
            )
        except KeyError as e:
            print(f"SLSQP optimizer encountered an invalid exit mode: {e}")
            # Create a dummy result with the current P_n as the solution
            from types import SimpleNamespace

            result = SimpleNamespace(
                x=P_n.reshape(params.S * params.A * params.S),
                success=False,
                message="SLSQP optimizer failed with invalid exit mode",
            )

        # Reshape the result back to the original shape
        P_star = result.x.reshape(params.S, params.A, params.S)

        # Get the objective function value f(P*) - the last value in fpn_list
        f_P_star = fpn_list[-1] if fpn_list else 0.0

        # Calculate alpha_n according to the formula: α_n = -((1-γ)^3)/(4γ^2) * f(P*)
        alpha_n = -((1 - params.gamma) ** 3) / (4 * params.gamma**2) * f_P_star

        # Store the previous P_n for convergence check
        P_n_prev = P_n.copy()

        # Update P_n according to the formula: P_{n+1} = (1-α_n)P_n + α_n P*
        P_n = (1 - alpha_n) * P_n + alpha_n * P_star

        # Calculate value function for the updated P_n
        v_pi_updated = PMDerivedValues.compute_value_function(
            np.einsum("sa,sab->sb", random_components.pi, P_n),  # P_pi from P_n
            derived_values.R_pi,
            params.gamma,
        )

        # Calculate expected return under current policy and transition kernel
        j_pi_current = float(derived_values.mu @ v_pi_updated)

        print(f"{n=:<6} {j_pi_current=}")

        # Calculate difference for convergence check (L2 norm of the difference)
        diff = float(np.linalg.norm(P_n - P_n_prev))
        best_robust_return = min(best_robust_return, j_pi_current)

        # Record data for this iteration
        iteration_time = time.time() - start_time
        performance_data["algorithm_name"].append("cpi_algorithm")
        performance_data["iteration_count"].append(iteration_number)
        performance_data["time_taken"].append(iteration_time)
        performance_data["j_pi"].append(best_robust_return)
        performance_data["S"].append(params.S)
        performance_data["A"].append(params.A)
        performance_data["beta"].append(params.beta)
        performance_data["hash"].append(rc_hash)
        performance_data["start_time"].append(datetime.fromtimestamp(start_time))
        performance_data["nominal_return"].append(derived_values.j_pi)

        iteration_number += 1

        if diff < params.tolerance and n > 1:
            print(f"CPI Algorithm converged after {n + 1} iterations.")
            break
    else:
        # Loop finished without converging
        print(f"CPI Algorithm did not converge within {max_iter} iterations.")
    return best_robust_return, iteration_number
