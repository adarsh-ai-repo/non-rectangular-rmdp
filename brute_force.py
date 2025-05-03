import numpy as np

from datamodels import AlgorithmPerformanceData, PMDerivedValues, PMRandomComponents, PMUserParameters


def compute_return(
    P: np.ndarray,
    params: PMUserParameters,
    random_components: PMRandomComponents,
    derived_values: PMDerivedValues,
) -> float:
    P_pi = np.einsum("sai,sa->si", P, random_components.pi)
    R_pi = np.sum(random_components.pi * random_components.R, axis=1)
    v_pi = np.linalg.inv(np.eye(params.S) - params.gamma * P_pi) @ R_pi
    return float(derived_values.mu @ v_pi)


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    n = len(v)
    sorted_v = np.sort(v)[::-1]
    cumulative_sum = np.cumsum(sorted_v)
    rho = np.where(sorted_v > (cumulative_sum - 1) / (np.arange(1, n + 1)))[0][-1]
    theta = (cumulative_sum[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def sample_random_kernel(params: PMUserParameters, random_components: PMRandomComponents) -> np.ndarray:
    noise = np.random.normal(0, params.beta, size=random_components.P.shape)
    noise = noise / np.linalg.norm(noise)
    noise = params.beta * noise * np.clip(1 - np.random.exponential(1 / params.S), 0, 1)
    P = random_components.P.copy()
    P = P + noise
    for s in range(params.S):
        for a in range(params.A):
            P[s, a, :] = project_to_simplex(P[s, a, :])
    return P


def RPE_Brute_Force(
    params: PMUserParameters,
    random_components: PMRandomComponents,
    num_samples: int,
    derived_values: PMDerivedValues,
    performance_data: AlgorithmPerformanceData,
    rc_hash: str,
) -> float:
    """
    Compute the Robust Policy Evaluation using brute force sampling.

    Args:
        params: User-defined parameters
        random_components: Random components of the MDP
        num_samples: Number of samples to generate
        derived_values: Derived values from the MDP
        performance_data: Dictionary to store performance metrics
        rc_hash: Hash of random components for tracking

    Returns:
        Maximum penalty found
    """
    nominal_return = compute_return(random_components.P, params, random_components, derived_values)
    penalty_list = []

    for i in range(num_samples):
        P_sample = sample_random_kernel(params, random_components)
        J_pi = compute_return(P_sample, params, random_components, derived_values)
        penalty = nominal_return - J_pi
        penalty_list.append(penalty)

        # Record performance data
        # At the end of the iteration

    return max(penalty_list)
