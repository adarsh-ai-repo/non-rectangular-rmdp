import numpy as np

from datamodels import PMDerivedValues, PMUserParameters


def get_max_eigenvalue(A: np.ndarray):
    #########  EIGENVALUE HEURISTICS ############
    ATA = A.T @ A
    # Eigenvalue decomposition of A^T A
    s, V = np.linalg.eigh(ATA)  # Eigenvalues and eigenvectors

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(s)[::-1]
    s = s[idx]
    V = V[:, idx]

    for i in range(V.shape[1]):
        v_plus_norm = np.linalg.norm(np.maximum(V[:, i], 0))
        v_minus_norm = np.linalg.norm(np.minimum(V[:, i], 0))
        if v_plus_norm < v_minus_norm:
            V[:, i] = -V[:, i]

    # Compute positive parts of eigenvectors
    V_plus = np.maximum(V, 0)  # Positive part of eigenvectors

    # Normalize positive parts to get u_i
    U = V_plus / np.linalg.norm(V_plus, axis=0)

    # Compute s_i <v_i, u_i> for all i
    scores = s * np.einsum("ij,ij->j", V, U)

    # Find j that maximizes the score
    j = np.argmax(scores)
    u_j = U[:, j]

    # Compute the maximum using the derived u_j
    max_norm_u_j = np.linalg.norm(A @ u_j)
    return max_norm_u_j


def optimize_using_eigen_value_and_bisection(
    params: PMUserParameters, derived_values: PMDerivedValues
) -> float:
    """
    Optimize using eigenvalue and bisection method with the new data models.

    Parameters:
    params: PMUserParameters - Contains user-defined parameters like beta, gamma
    derived_values: PMDerivedValues - Contains derived matrices and values

    Returns:
    float: Optimized lambda value
    """
    min_lambda_value = 0
    max_lambda_value = 10 * params.beta
    lambda_value = (min_lambda_value + max_lambda_value) / 2

    def get_input_matrix(lambda_value: float) -> np.ndarray:
        return derived_values.matrix_part1 - lambda_value * derived_values.matrix_part2

    for i in range(16):
        lambda_value = (min_lambda_value + max_lambda_value) / 2
        new_value = (
            params.gamma
            * params.beta
            * get_max_eigenvalue(get_input_matrix(lambda_value))
            - lambda_value
        )

        if (max_lambda_value - min_lambda_value) < 1e-4:
            break
        elif new_value < 0:
            max_lambda_value = lambda_value
        elif new_value > 0:
            min_lambda_value = lambda_value
        else:
            break

        print(f"current lambda_value={lambda_value}")

    return lambda_value
