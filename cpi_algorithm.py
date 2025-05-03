import time
from datetime import datetime

import numpy as np
from rich.progress import track

from datamodels import (
    AlgorithmPerformanceData,
    PMDerivedValues,
    PMRandomComponents,
    PMUserParameters,
)

# Note: compute_return is not directly used as the value function is calculated within the loop.
# from brute_force import compute_return, project_to_simplex # project_to_simplex not needed for the chosen P* method


def _compute_P_star_sa(C_sa: np.ndarray, P_hat_sa: np.ndarray, beta: float, S: int) -> np.ndarray:
    """
    Solves argmin_{p} C_sa @ p subject to sum(p)=1, p>=0, ||p - P_hat_sa||_1 <= beta
    for a single state-action pair (s, a).

    Uses a greedy approach by shifting probability mass from the highest cost successor state
    to the lowest cost successor state, constrained by the L1 ball.

    Args:
        C_sa: Cost vector for the given (s, a), shape (S,).
        P_hat_sa: Nominal transition probabilities for (s, a), shape (S,).
        beta: Radius of the L1 uncertainty ball.
        S: Number of states.

    Returns:
        The optimized probability vector p*, shape (S,).
    """
    p_star = P_hat_sa.copy()
    if S == 1:  # Trivial case if only one successor state
        return p_star

    s_min = np.argmin(C_sa)
    s_max = np.argmax(C_sa)

    if s_min == s_max:  # All costs are the same, no improvement possible
        return p_star

    # The maximum probability mass we can shift is beta / 2.
    # We shift from s_max to s_min.
    delta = beta / 2.0

    # Ensure the shift doesn't violate probability constraints (p >= 0).
    # We can decrease p_star[s_max] by at most p_star[s_max].
    # We can increase p_star[s_min] by at most 1.0 - p_star[s_min].
    actual_delta = min(delta, p_star[s_max], 1.0 - p_star[s_min])

    # Apply the shift
    p_star[s_min] += actual_delta
    p_star[s_max] -= actual_delta

    # Due to potential floating-point inaccuracies, ensure constraints hold.
    p_star = np.maximum(p_star, 0.0)  # Ensure non-negative
    p_star /= np.sum(p_star)  # Ensure sums to 1

    # Optional: Verify L1 constraint (should hold by construction)
    # l1_dist = np.sum(np.abs(p_star - P_hat_sa))
    # assert l1_dist <= beta + 1e-9, f"L1 distance {l1_dist} exceeds beta {beta}"

    return p_star


def run_cpi_algorithm(
    params: PMUserParameters,
    random_components: PMRandomComponents,
    performance_data: AlgorithmPerformanceData,
    max_iter: int = 300,
) -> None:
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
        A polars DataFrame containing:
        - iteration_count: Number of iterations performed
        - time_taken: Time taken for each iteration in seconds
        - Penalty: The penalty value (J^π-J^π_{U_2})
        - S: Number of states
        - A: Number of actions
        - beta: Uncertainty radius
        - hash: MD5 hash of PMRandomComponents
    """
    S, A = params.S, params.A
    gamma = params.gamma
    tolerance = params.tolerance
    beta = params.beta

    P_hat = random_components.P
    pi = random_components.pi
    R = random_components.R

    # Initial state distribution (uniform)
    mu = np.ones(S) / S

    # Initialize P_n with the nominal kernel
    P_n = P_hat.copy()
    R_pi = np.sum(pi * R, axis=1)  # Policy-averaged reward (constant)

    # Calculate nominal return (J^π)
    P_pi_nominal = np.einsum("sai,sa->si", P_hat, pi, optimize=True)
    v_pi_nominal = PMDerivedValues.compute_value_function(P_pi_nominal, R_pi, gamma)
    J_pi_nominal = float(mu @ v_pi_nominal)

    # Get hash value of random components
    rc_hash = random_components.md5_hash

    for n in track(list(range(max_iter))):
        start_time = time.time()
        # 1. Calculate policy-averaged kernel for P_n
        P_pi_n = np.einsum("sai,sa->si", P_n, pi, optimize=True)

        # 2. Calculate value function v^pi_{P_n}
        v_pi_Pn = PMDerivedValues.compute_value_function(P_pi_n, R_pi, gamma)

        # 3. Calculate occupation measure d^pi_{P_n}
        _, d_pi_Pn = PMDerivedValues.compute_occupation_measures(P_pi_n, gamma)  # We only need d_pi

        # 4. Calculate Advantage A^pi_{P_n}(s, a, s')
        # A(s,a,s') = gamma * (v(s') - sum_{s"} P_n(s"|s,a) v(s"))
        v_sum_sa = np.einsum("sat,t->sa", P_n, v_pi_Pn, optimize=True)  # sum_{s"} P_n(s"|s,a) v(s")
        # Expand v_pi_Pn to (S, A, S) and v_sum_sa to (S, A, S) for broadcasting
        A_pi_Pn = gamma * (v_pi_Pn[None, None, :] - v_sum_sa[:, :, None])

        # 5. Define coefficients C for the optimization objective f(P)
        # C(s,a,s') = (1/(1-gamma)) * d^pi_{P_n}(s) * pi(a|s) * A^pi_{P_n}(s, a, s')
        C = (1.0 / (1.0 - gamma)) * (d_pi_Pn[:, None, None] * pi[:, :, None] * A_pi_Pn)

        # 6. Compute P* = argmin_{P in Uc} f(P) = argmin_{P in Uc} sum_{s,a,s'} C(s,a,s') P(s'|s,a)
        # This decomposes into independent minimizations for each (s, a) pair.
        P_star = np.zeros_like(P_n)
        for s in range(S):
            for a in range(A):
                P_star[s, a, :] = _compute_P_star_sa(C[s, a, :], P_hat[s, a, :], beta, S)

        # 7. Calculate f(P*) = sum_{s,a,s'} C(s,a,s') P*(s'|s,a)
        # Note: C already includes the (1/(1-gamma)) factor.
        f_P_star = np.sum(C * P_star)

        # 8. Calculate step size alpha_n
        # alpha_n = - ( (1-gamma)^3 / (4 * gamma^2) ) * f(P*)
        # Ensure alpha_n is non-negative (f_P_star should be <= 0 theoretically)
        alpha_n_num = -(((1.0 - gamma) ** 3) * f_P_star)
        alpha_n_den = 4.0 * (gamma**2)
        if alpha_n_den == 0:
            # Avoid division by zero if gamma is 0 (though gamma > 0 constraint exists)
            alpha_n = 0.0 if alpha_n_num == 0 else np.inf  # Or handle as error
        else:
            alpha_n = alpha_n_num / alpha_n_den

        # Clamp alpha_n for stability, although theory might guarantee 0 <= alpha_n <= 1
        alpha_n = np.clip(alpha_n, 0.0, 1.0)

        # 9. Update P_{n+1}
        P_next = (1.0 - alpha_n) * P_n + alpha_n * P_star

        # 10. Check for convergence
        diff = np.max(np.abs(P_next - P_n))

        # Calculate current return J^π_{P_n}
        P_pi_n = np.einsum("sai,sa->si", P_n, pi, optimize=True)
        v_pi_n = PMDerivedValues.compute_value_function(P_pi_n, R_pi, gamma)
        J_pi_P_n = float(mu @ v_pi_n)

        # Calculate penalty (J^π - J^π_{P_n})
        penalty = J_pi_nominal - J_pi_P_n

        # Record data for this iteration
        iteration_time = time.time() - start_time
        performance_data["algorithm_name"].append("cpi_algorithm")
        performance_data["iteration_count"].append(n + 1)
        performance_data["time_taken"].append(iteration_time)
        performance_data["Penalty"].append(penalty)
        performance_data["S"].append(S)
        performance_data["A"].append(A)
        performance_data["beta"].append(beta)
        performance_data["hash"].append(rc_hash)
        performance_data["start_time"].append(datetime.fromtimestamp(start_time))

        if diff < tolerance:
            P_n = P_next  # Store the final converged kernel
            print(f"CPI Algorithm converged after {n + 1} iterations.")
            break

        # Update P_n for the next iteration
        P_n = P_next
    else:
        # Loop finished without converging
        print(f"CPI Algorithm did not converge within {max_iter} iterations.")
