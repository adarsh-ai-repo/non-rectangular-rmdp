import time
from datetime import datetime

import numpy as np

from brute_force import compute_return
from datamodels import AlgorithmPerformanceData, PMDerivedValues, PMRandomComponents, PMUserParameters


def spectral_span_norm(x: np.ndarray) -> float:
    """
    Compute the spectral span norm ||x||_sp = (max_i x_i - min_i x_i) / 2

    Args:
        x: Input vector

    Returns:
        float: Spectral span norm value
    """
    return (np.max(x) - np.min(x)) / 2.0


def compute_E_pi_lambda(
    lambda_val: float,
    params: PMUserParameters,
    derived_values: PMDerivedValues,
) -> np.ndarray:
    """
    Compute the matrix E^π_λ such that E^π_λ(:, (s,a)) = γ * π(a|s) * [d^π(s) * v^π - λ * D^π(:, s)]

    This matches the closed form: F(λ) = γβ max_{s,a} π(a|s)||d^π(s)v^π - λD^π(·,s)||_sp

    Args:
        lambda_val: The lambda value
        params: User parameters
        derived_values: Derived values containing D_pi, d_pi, v_pi, H

    Returns:
        np.ndarray: E^π_λ matrix of shape (S, S*A)
    """
    S, A = params.S, params.A
    gamma = params.gamma
    D_pi = derived_values.D_pi  # (S, S)
    d_pi = derived_values.d_pi  # (S,) - stationary distribution
    v_pi = derived_values.v_pi  # (S,) - value function
    H = derived_values.H  # (S, S*A)

    # Initialize E^π_λ matrix
    E_pi_lambda = np.zeros((S, S * A))

    # For each (s,a) pair, compute the corresponding column
    for s in range(S):
        for a in range(A):
            col_index = s * A + a

            # Get π(a|s) from H matrix
            pi_a_s = H[s, col_index]

            # Compute d^π(s) * v^π - λ * D^π(:, s)
            term = d_pi[s] * v_pi - lambda_val * D_pi[:, s]  # (S,)

            # E^π_λ(:, (s,a)) = γ * π(a|s) * term
            E_pi_lambda[:, col_index] = gamma * pi_a_s * term

    return E_pi_lambda


def compute_F_lambda_L1(
    lambda_val: float,
    params: PMUserParameters,
    random_components: PMRandomComponents,
    derived_values: PMDerivedValues,
) -> tuple[float, int, int]:
    """
    Compute F(λ) using the closed form solution for L1 norm from Proposition.

    F(λ) = γβ max_{s∈S,a∈A} π(a|s)||d^π(s)v^π - λD^π(·,s)||_sp

    where:
    - ||x||_sp = (max_i x_i - min_i x_i)/2 is the spectral span norm
    - d^π(s) is the s-th component of the stationary distribution
    - v^π is the value function vector
    - D^π(·,s) is the s-th column of matrix D^π = (I - γP̂^π)^{-1}

    Args:
        lambda_val: The lambda value
        params: User parameters including gamma, beta
        random_components: Random components including policy pi
        derived_values: Derived values containing d_pi, v_pi, D_pi

    Returns:
        tuple: (F_lambda_value, optimal_s, optimal_a)
    """
    S, A = params.S, params.A
    gamma, beta = params.gamma, params.beta

    # Get required vectors and matrices
    d_pi = derived_values.d_pi  # (S,) - stationary distribution
    v_pi = derived_values.v_pi  # (S,) - value function
    D_pi = derived_values.D_pi  # (S, S) - fundamental matrix

    max_value = 0.0
    optimal_s, optimal_a = 0, 0

    for s in range(S):
        for a in range(A):
            # π(a|s) term
            pi_term = random_components.pi[s, a]

            # Compute the vector: d^π(s)v^π - λD^π(·,s)
            # d^π(s) is scalar, v^π is vector (S,), so d^π(s)v^π is vector (S,)
            d_pi_s_times_v_pi = d_pi[s] * v_pi  # (S,)

            # D^π(·,s) is the s-th column of D^π
            D_pi_col_s = D_pi[:, s]  # (S,)

            # The vector inside the spectral span norm
            vector = d_pi_s_times_v_pi - lambda_val * D_pi_col_s  # (S,)

            # Compute spectral span norm: ||vector||_sp
            span_norm = spectral_span_norm(vector)

            # Overall value: π(a|s) * ||d^π(s)v^π - λD^π(·,s)||_sp
            value = pi_term * span_norm

            if value > max_value:
                max_value = value
                optimal_s, optimal_a = s, a

    # Final F(λ) = γβ * max_value
    F_lambda = gamma * beta * max_value

    return F_lambda, optimal_s, optimal_a


def get_optimal_b_k_L1(
    lambda_val: float,
    params: PMUserParameters,
    random_components: PMRandomComponents,
    derived_values: PMDerivedValues,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get optimal b and k vectors for L1 norm case using Lemma app:rs:L1opt.

    From the research paper, the optimizers are:
    - b*(i) = β𝟙(i=i*) where i* = argmax_i ||E^π_λ(·,i)||_sp
    - k*(i_max) = -k*(i_min) = 1/2, where i_max/i_min are argmax/argmin of Eb*

    The relationship is: max_{b ∈ B, k ∈ K} k^T E^π_λ b = max_i ||E^π_λ(·,i)||_sp

    Args:
        lambda_val: The lambda value
        params: User parameters
        random_components: Random components
        derived_values: Derived values

    Returns:
        tuple: (optimal_b, optimal_k) where b ∈ R^{S×A}, k ∈ R^S
    """
    S, A = params.S, params.A
    beta = params.beta

    # Step 1: Find the optimal (s,a) pair using the same logic as closed form
    # This ensures consistency between F(λ) computation and (b,k) calculation
    _, optimal_s, optimal_a = compute_F_lambda_L1(lambda_val, params, random_components, derived_values)

    # Step 2: Convert (s,a) pair to column index in E^π_λ
    # The H matrix maps (s,a) to column index as: column = s*A + a
    optimal_i = optimal_s * A + optimal_a

    # Step 3: Construct optimal b* = β𝟙(i=i*)
    # b* is one-hot vector with β at the optimal index i*
    b = np.zeros(S * A)
    b[optimal_i] = beta

    # Step 4: Compute E^π_λ matrix and get the optimal column
    E_pi_lambda = compute_E_pi_lambda(lambda_val, params, derived_values)  # (S, S*A)

    # Step 5: Compute Eb* = E^π_λ @ b*
    # Since b* has only one non-zero element β at index optimal_i:
    # Eb* = β * E^π_λ[:, optimal_i]
    Eb = beta * E_pi_lambda[:, optimal_i]  # (S,)

    # Step 6: Construct optimal k*
    # k*(i_max) = -k*(i_min) = 1/2
    # where i_max = argmax_i (Eb*)(i), i_min = argmin_i (Eb*)(i)
    k = np.zeros(S)
    i_max = np.argmax(Eb)
    i_min = np.argmin(Eb)

    k[i_max] = 0.5
    k[i_min] = -0.5

    # Verify constraints:
    # - ||b||_1 = β ≤ β ✓
    # - ||k||_1 = |0.5| + |-0.5| = 1.0 ≤ 1 ✓
    # - 1^T k = 0.5 + (-0.5) = 0 ✓

    return b, k


def optimize_using_L1_closed_form_and_bisection(
    params: PMUserParameters,
    random_components: PMRandomComponents,
    derived_values: PMDerivedValues,
    performance_data: AlgorithmPerformanceData,
    rc_hash: str,
) -> float:
    """
    Optimize using L1 closed form solution and bisection method.

    Uses the closed form expression for F(λ) from the L1 norm case:
    F(λ) = γβ max_{s∈S,a∈A} π(a|s)||d^π(s)v^π - λD^π(·,s)||_sp

    Parameters:
    params: PMUserParameters - Contains user-defined parameters like beta, gamma
    random_components: PMRandomComponents - Contains policy, transitions, rewards
    derived_values: PMDerivedValues - Contains derived matrices and values
    performance_data: AlgorithmPerformanceData - Dictionary to store performance metrics
    rc_hash: str - Hash of random components for tracking

    Returns:
    float: Optimized lambda value (robust return = J^π - λ*)
    """
    min_lambda_value = 0.0
    max_lambda_value = 1.0 / (1.0 - params.gamma)
    lambda_value = (min_lambda_value + max_lambda_value) / 2.0
    start_time = time.time()

    max_iterations = 100  # Allow sufficient iterations for convergence
    for i in range(max_iterations):
        lambda_value = (min_lambda_value + max_lambda_value) / 2.0

        # Compute F(λ) using closed form L1 solution
        F_lambda, optimal_s, optimal_a = compute_F_lambda_L1(
            lambda_value, params, random_components, derived_values
        )

        # The bisection condition: F(λ) > λ iff λ > λ*
        new_value = F_lambda - lambda_value

        # Get optimal b and k for this lambda
        b, k = get_optimal_b_k_L1(lambda_value, params, random_components, derived_values)

        # Reshape b to match transition kernel dimensions for perturbation
        b_reshaped = b.reshape(params.S, params.A)

        # Create perturbed transition kernel P* = P - bk^T (rank-1 perturbation)
        # Note: k is S-dimensional, b is SA-dimensional reshaped to S×A
        P_star = random_components.P.copy()
        for s in range(params.S):
            for a in range(params.A):
                for s_next in range(params.S):
                    P_star[s, a, s_next] -= b_reshaped[s, a] * k[s_next]

        # Apply simplex projection to ensure valid probability distributions
        P_star = np.maximum(P_star, 0)  # Non-negativity
        row_sums = np.sum(P_star, axis=2, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        P_star = P_star / row_sums  # Normalize to sum to 1

        # Compute robust return with perturbed kernel
        robust_return = compute_return(P_star, params, random_components, derived_values)
        nominal_return = compute_return(random_components.P, params, random_components, derived_values)

        print(f"Iteration {i + 1}: λ={lambda_value:.6f}, F(λ)={F_lambda:.6f}, F(λ)-λ={new_value:.6f}")
        print(f"  Optimal (s,a)=({optimal_s},{optimal_a})")
        print(f"  Nominal return={nominal_return:.6f}, Robust return={robust_return:.6f}")
        print(f"  Theory: J^π - λ = {derived_values.j_pi - lambda_value:.6f}")

        # Record data for this iteration
        iteration_time = time.time() - start_time
        performance_data["algorithm_name"].append("L1_closed_form_bisection")
        performance_data["iteration_count"].append(i + 1)
        performance_data["time_taken"].append(iteration_time)
        performance_data["j_pi"].append(robust_return)
        performance_data["S"].append(params.S)
        performance_data["A"].append(params.A)
        performance_data["beta"].append(params.beta)
        performance_data["hash"].append(rc_hash)
        performance_data["nominal_return"].append(derived_values.j_pi)
        performance_data["start_time"].append(datetime.fromtimestamp(start_time))

        # Bisection logic: F(λ) > λ iff λ > λ*
        if abs(new_value) < params.tolerance:
            print(f"Converged at iteration {i + 1}")
            break
        elif new_value > 0:  # F(λ) > λ, so λ < λ*, increase λ
            min_lambda_value = lambda_value
        else:  # F(λ) < λ, so λ > λ*, decrease λ
            max_lambda_value = lambda_value

        if (max_lambda_value - min_lambda_value) < params.tolerance:
            print(f"Tolerance reached at iteration {i + 1}")
            break

    print(f"Final lambda_value={lambda_value:.6f}")
    print(f"Final robust return = J^π - λ* = {derived_values.j_pi - lambda_value:.6f}")

    return robust_return  # pyright: ignore[reportPossiblyUnboundVariable]


# Keep the old function name for backward compatibility
def optimize_using_eigen_value_and_bisection(
    params: PMUserParameters,
    random_components: PMRandomComponents,
    derived_values: PMDerivedValues,
    performance_data: AlgorithmPerformanceData,
    rc_hash: str,
) -> float:
    """
    Backward compatibility wrapper for the L1 closed form optimization.
    Now uses L1 closed form solution instead of eigenvalue heuristics.
    """
    return optimize_using_L1_closed_form_and_bisection(
        params, random_components, derived_values, performance_data, rc_hash
    )
