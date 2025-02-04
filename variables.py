import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from solve_problem import solve_problem
from rl_robust.cache import memory


def kernel(S, A):
    p = np.random.rand(S, A, S)
    for s in range(S):
        for a in range(A):
            summ = np.sum(p[s, a])
            p[s, a] = p[s, a] / summ
    return p


def compute_value_function(P_pi: np.ndarray, R_pi: np.ndarray, gamma: float):
    """
    Compute the value function v^π given the policy-averaged transition kernel and reward function.

    Parameters:
    P_pi : numpy array, shape (n, n)
        The policy-averaged transition kernel.
    R_pi : numpy array, shape (n,)
        The policy-averaged reward function.
    gamma : float
        The discount factor, 0 <= gamma < 1.

    Returns:
    v_pi : numpy array, shape (n,)
        The value function.
    """
    n_states = P_pi.shape[0]
    I_matrix = np.eye(n_states)
    A = I_matrix - gamma * P_pi
    A_inv = np.linalg.inv(A)
    v_pi = A_inv @ R_pi
    return v_pi


def compute_occupation_measures(P_pi: np.ndarray, gamma: float):
    """
    Compute the occupation matrix D^π and occupation measure d^π.

    Parameters:
    P_pi : numpy array, shape (n, n)
        The policy-averaged transition kernel.
    gamma : float
        The discount factor, 0 <= gamma < 1.
    initial_distribution : assumes uniform distribution.

    Returns:
    D_pi : numpy array, shape (n, n)
        The occupation matrix.
    d_pi : numpy array, shape (n,)
        The occupation measure.
    """
    n_states = P_pi.shape[0]

    initial_distribution = np.ones(n_states) / n_states

    # Compute (I - γP^π)^(-1)
    I_matrix = np.eye(n_states)
    A_inv = np.linalg.inv(I_matrix - gamma * P_pi)

    # Compute d^π
    d_pi = initial_distribution @ A_inv

    # Compute D^π
    D_pi = np.diag(d_pi) @ P_pi

    return D_pi, d_pi


class set_pm:
    def __init__(self, S, A):
        self.S = S
        self.A = A
        self.v0 = np.random.randn(S)

        self.dim_b_vector = self.S * self.A
        # Uncertianty radius
        self.beta = 0.1  # 0.5 / S
        self.gamma = 0.9
        self.tolerance = 1e-5
        self.mu = np.ones(S) / S

        self.pi = np.random.rand(S, A)  # Random Initial policy
        self.R = np.random.randn(S, A)  # Random Reward function

        for s in range(S):
            self.pi[s] = self.pi[s] / np.sum(self.pi[s])

        self.P_pi = np.zeros((S, S))
        self.P = kernel(S, A)
        for s in range(S):
            for t in range(S):
                self.P_pi[s, t] = self.pi[s, :] @ self.P[s, :, t]

        self.R_pi = np.sum(self.R * self.pi, axis=1)

        self.v_pi = compute_value_function(self.P_pi, self.R_pi, self.gamma)

        self.D_pi, self.d_pi = compute_occupation_measures(self.P_pi, self.gamma)

        self.H = np.zeros((self.S, self.S * self.A))
        for i in range(self.S):
            for j in range(self.A):
                self.H[i, i * self.A + j] = self.pi[i, j]  # = \pi(j|i))

        self.PHI = np.eye(self.S, self.S) - 1 / self.S

        self.matrix_part1 = self.PHI @ np.outer(self.v_pi, self.d_pi) @ self.H
        self.matrix_part2 = self.PHI @ self.D_pi @ self.H

    def get_input_matrix(self, lambda_value: float) -> np.ndarray:
        return self.matrix_part1 - lambda_value * self.matrix_part2


# plt.figure(figsize=(10, 6))
# plt.plot(lambda_value_list, values, "b-o")  # Plot actual values

# plt.xscale("log")  # Set x-axis to log scale
# plt.yscale(
#     "symlog", linthresh=1e-3
# )  # Set y-axis to symmetric log scale with linear region near zero

# plt.xlabel("λ")
# plt.ylabel("F(λ)")
# plt.title("F(λ) vs λ (Log Plot)")
# plt.grid(True, which="both", ls="-", alpha=0.5)

# # Add a horizontal line at y=0
# plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)

# plt.tight_layout()

# # Zoom in more around 0 on the y-axis
# yabs_max = max(abs(min(values)), abs(max(values)))
# y_limit = yabs_max * 0.1  # Adjust this factor to zoom in/out
# plt.ylim([-y_limit, y_limit])


# plt.savefig("F(λ)_vs_λ_2.png", dpi=300, bbox_inches="tight")
# plt.close()


def compute_return(P, pm):
    """Compute the return J^pm.pi for a given transition kernel P."""
    P_pi = np.einsum("sai,sa->si", P, pm.pi)  # Policy-weighted transition matrix
    R_pi = np.sum(pm.pi * pm.R, axis=1)  # Policy-weighted reward
    v_pi = np.linalg.inv(np.eye(pm.S) - pm.gamma * P_pi) @ R_pi
    return np.sum(pm.mu @ v_pi)


def project_to_simplex(v):
    """
    Project a vector `v` onto the probability simplex using the algorithm from
    Wang & Carreira-Perpiñán (2013).
    """
    n = len(v)
    sorted_v = np.sort(v)[::-1]
    cumulative_sum = np.cumsum(sorted_v)
    rho = np.where(sorted_v > (cumulative_sum - 1) / (np.arange(1, n + 1)))[0][-1]
    theta = (cumulative_sum[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def sample_random_kernel(pm):
    """Sample a random transition kernel using normal distribution and project to the simplex."""
    noise = np.random.normal(
        0, pm.beta, size=pm.P.shape
    )  # Sample noise from normal distribution
    noise = noise / np.linalg.norm(noise)  # Normalizing the noise  norm radius 1
    noise = (
        pm.beta * noise * np.clip(1 - np.random.exponential(1 / pm.S), 0, 1)
    )  # Trying to imitate, noise being uniform in a ball
    P = pm.P + noise
    # Project each row of P to the simplex
    for s in range(pm.S):
        for a in range(pm.A):
            P[s, a, :] = project_to_simplex(P[s, a, :])
    return P


def check_kernel_from_uncertainty_set(pm: set_pm, p):
    """
    Check if the sampled kernel is within the uncertainty set with detailed error reporting.
    """
    p_min = np.min(p)
    p_max = np.max(p)
    psum = np.sum(p, axis=-1)
    ptp = np.ptp(psum)
    r = np.linalg.norm(pm.P - p)

    if p_min < -pm.tolerance:
        print(
            f"Error: Minimum probability {p_min} is below pm.tolerance {pm.tolerance}."
        )
    if p_max > 1:
        print(f"Error: Maximum probability {p_max} exceeds 1.")
    if np.abs(ptp) > pm.tolerance:
        print(
            f"Error: Row sums deviation {np.abs(ptp)} exceeds pm.tolerance {pm.tolerance}."
        )
    if r > pm.beta:
        print(
            f"Error: Distance from nominal kernel {r} exceeds uncertainty radius {pm.beta}."
        )

    return r / pm.beta  # Return normalized distance as a measure of deviation


def RPE_Brute_Force(pm: set_pm, num_samples: int):
    nominal_return = compute_return(pm.P, pm=pm)
    penalty_list = []
    radius_sampled_kernel = []

    for _ in range(num_samples):
        P_sample = sample_random_kernel(
            pm=pm
        )  # Sample a kernel from the uncertainty set
        radius_sampled_kernel.append(
            check_kernel_from_uncertainty_set(pm, P_sample)
        )  # Check if kernel is valid
        J_pi = compute_return(P_sample, pm=pm)
        penalty = nominal_return - J_pi
        penalty_list.append(penalty)

    return max(penalty_list)


def solve_robust_optimization(pm: set_pm):
    S, A = pm.S, pm.A
    v_pi, d_pi, D_pi = pm.v_pi, pm.d_pi, pm.D_pi
    gamma, beta = pm.gamma, pm.beta

    def objective(x):
        b, k = x[: S * A], x[S * A :]
        b_pi = pm.H @ b
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


def solve_robust_optimization_random(pm: set_pm, num_guesses=10000):
    S, A = pm.S, pm.A
    v_pi, d_pi, D_pi = pm.v_pi, pm.d_pi, pm.D_pi
    gamma, beta = pm.gamma, pm.beta

    def objective(x):
        b, k = x[: S * A], x[S * A :]
        b_pi = pm.H @ b
        v_pi_b = D_pi @ b_pi
        numerator = np.dot(k, v_pi) * np.dot(d_pi, b_pi)
        denominator = 1 + gamma * np.dot(k, v_pi_b)
        return numerator / denominator  # Positive because we're maximizing

    best_value = float("-inf")
    best_b = None
    best_k = None

    for _ in range(num_guesses):
        # Generate random b
        b = np.random.rand(S * A)
        b *= beta / np.linalg.norm(b)

        # Generate random k
        k = np.random.randn(S)
        k -= np.mean(k)  # Ensure k sums to 0
        k /= np.linalg.norm(k)

        x = np.concatenate([b, k])
        value = objective(x)

        if value > best_value:
            best_value = value
            best_b = b
            best_k = k

    return best_value, best_b, best_k


# Usage example:
# optimal_value, b_opt, k_opt = solve_robust_optimization_random(pm)

# pm = set_pm(64, 8)

# all_correction_using_direct_optimization = []
# for _ in range(5):
#     try:
#         optimal_value, b_opt, k_opt = solve_robust_optimization(pm)
#         print(f"Optimal value: {optimal_value}")
#         print(f"Optimal b shape: {b_opt.shape}")
#         print(f"Norm of b: {np.linalg.norm(b_opt)}")
#         print(f"Optimal k shape: {k_opt.shape}")
#         print(f"Norm of k: {np.linalg.norm(k_opt)}")
#         print(f"Sum of k: {np.sum(k_opt)}")  # Should be very close to 0

#         # Calculate the robust return
#         # J_pi = np.dot(pm.d_pi, pm.v_pi)
#         lambda_value_from_direct_optimization = pm.gamma * optimal_value
#         print(f"Correction to Robust return: {lambda_value_from_direct_optimization}")
#         all_correction_using_direct_optimization.append(
#             lambda_value_from_direct_optimization
#         )

#     except ValueError as e:
#         print(f"Optimization error: {e}")

# print("=" * 100)
# print(
#     f"Optimal Correction using direction method: {max(all_correction_using_direct_optimization)} "
# )
# print("=" * 100)

# lambda_value_list = (2 * np.ones(16)) ** (np.arange(16) - 13)


def solve_robust_optimization_using_bisection(pm: set_pm):
    min_lambda_value = 0
    max_lambda_value = 10 * pm.beta

    for i in range(16):
        lambda_value = (min_lambda_value + max_lambda_value) / 2
        new_value = (
            pm.gamma * pm.beta * solve_problem(pm.get_input_matrix(lambda_value), 4)
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

    print("Optimal value using bisection method")
    print(lambda_value)
    return lambda_value


def solve_robust_optimization_using_bisection_eigen(pm: set_pm):
    min_lambda_value = 0
    max_lambda_value = 10 * pm.beta

    for i in range(16):
        lambda_value = (min_lambda_value + max_lambda_value) / 2
        new_value = (
            pm.gamma * pm.beta * get_max_eigenvalue(pm.get_input_matrix(lambda_value))
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

    print("Optimal value using bisection method")
    print(lambda_value)
    return lambda_value


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


@memory.cache
def run_experiments(S, A, num_trials=8, time_limit=120):  # time_limit in seconds
    pm = set_pm(S, A)

    # Direct optimization
    direct_results = []
    direct_start_time = time.time()
    for i in range(num_trials):
        if ((time.time() - direct_start_time) > time_limit) and (i > 4):
            break
        try:
            optimal_value, _, _ = solve_robust_optimization(pm)
            direct_results.append(pm.gamma * optimal_value)
        except ValueError:
            pass
    direct_result = max(direct_results) if direct_results else np.nan
    direct_time = time.time() - direct_start_time

    # Bisection method
    bisection_result = solve_robust_optimization_using_bisection(pm)
    # bisection_result = direct_result - np.random.random() / 10

    # Random search
    random_results = []
    random_start_time = time.time()
    while time.time() - random_start_time < direct_time:
        optimal_value, _, _ = solve_robust_optimization_random(pm)
        random_results.append(pm.gamma * optimal_value)
    random_result = max(random_results)

    # Random search
    random_kernel_results = []
    random_start_time = time.time()
    while time.time() - random_start_time < direct_time:
        optimal_value = RPE_Brute_Force(pm, 10000)
        random_kernel_results.append(pm.gamma * optimal_value)
    random_kernel_result = max(random_kernel_results)

    return direct_result, bisection_result, random_result, random_kernel_result


# Run experiments for different state space sizes
state_sizes = np.linspace(10, 300, 29, dtype=int)
action_size = 8  # Fixed action space size

# time_vs_state_size = []
# for S in state_sizes:
#     pm = set_pm(S, action_size)
#     start_time = time.time()
#     # get_max_eigenvalue(pm.get_input_matrix(0.01))
#     solve_problem(pm.get_input_matrix(0.01), 2)
#     time_taken = time.time() - start_time
#     time_vs_state_size.append({"state_size": S, "time_taken": time_taken})
time_vs_state_size = [
    {"state_size": np.int64(10), "time_taken": 0.012964963912963867},
    {"state_size": np.int64(20), "time_taken": 0.04794597625732422},
    {"state_size": np.int64(30), "time_taken": 0.12467813491821289},
    {"state_size": np.int64(41), "time_taken": 0.4566478729248047},
    {"state_size": np.int64(51), "time_taken": 0.612633228302002},
    {"state_size": np.int64(61), "time_taken": 1.5095431804656982},
    {"state_size": np.int64(72), "time_taken": 2.480285882949829},
    {"state_size": np.int64(82), "time_taken": 2.3833799362182617},
    {"state_size": np.int64(92), "time_taken": 3.5002050399780273},
    {"state_size": np.int64(103), "time_taken": 4.733496189117432},
    {"state_size": np.int64(113), "time_taken": 6.551486015319824},
    {"state_size": np.int64(123), "time_taken": 8.226361751556396},
    {"state_size": np.int64(134), "time_taken": 10.529484033584595},
    {"state_size": np.int64(144), "time_taken": 20.39603614807129},
    {"state_size": np.int64(155), "time_taken": 17.749484062194824},
    {"state_size": np.int64(165), "time_taken": 21.05328679084778},
    {"state_size": np.int64(175), "time_taken": 24.60878610610962},
    {"state_size": np.int64(186), "time_taken": 31.157560110092163},
    {"state_size": np.int64(196), "time_taken": 35.512206077575684},
    {"state_size": np.int64(206), "time_taken": 41.30090808868408},
    {"state_size": np.int64(217), "time_taken": 48.56358981132507},
    {"state_size": np.int64(227), "time_taken": 60.57559776306152},
    {"state_size": np.int64(237), "time_taken": 64.8502049446106},
    {"state_size": np.int64(248), "time_taken": 73.2350161075592},
    {"state_size": np.int64(258), "time_taken": 126.29916095733643},
    {"state_size": np.int64(268), "time_taken": 99.29913306236267},
    {"state_size": np.int64(279), "time_taken": 108.30011200904846},
    {"state_size": np.int64(289), "time_taken": 184.93371319770813},
    {"state_size": np.int64(300), "time_taken": 146.53172898292542},
]

print(time_vs_state_size)

state_sizes = [item["state_size"] for item in time_vs_state_size]
times_taken = [item["time_taken"] for item in time_vs_state_size]


# Extract data from the time_vs_state_size list
state_sizes = np.array([item["state_size"] for item in time_vs_state_size])
times_taken = np.array([item["time_taken"] for item in time_vs_state_size])

# Create the plot
plt.figure(figsize=(12, 7))
plt.scatter(state_sizes, times_taken, color="blue", label="Data points")

# Add polynomial fit
degree = 2  # You can adjust this to change the degree of the polynomial fit
coeffs = np.polyfit(state_sizes, times_taken, degree)
poly = np.poly1d(coeffs)

# Create a smooth curve for the polynomial fit
x_smooth = np.linspace(state_sizes.min(), state_sizes.max(), 200)
y_smooth = poly(x_smooth)

plt.plot(x_smooth, y_smooth, color="red", label=f"Polynomial fit (degree {degree})")

# Customize the plot
plt.title("State Size vs. Time Taken for One Iteration of Algorithm 1")
plt.xlabel("State Size")
plt.ylabel("Time Taken (seconds)")
plt.grid(True, linestyle="--", alpha=0.7)


# Ensure y-axis starts from 0
plt.ylim(bottom=0)

# Add legend
plt.legend()

print(coeffs)
# Generate equation string
# Generate equation string
eq_terms = [f"{coeff:.2e}x^{degree - i}" for i, coeff in enumerate(coeffs[:-1])]
eq_terms.append(f"{coeffs[-1]:.2e}")
eq_string = "y = " + " + ".join(eq_terms).replace(" + -", " - ")

# Add equation text to plot
plt.text(
    0.05,
    0.95,
    eq_string,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
)

# Show the plot
plt.tight_layout()
plt.show()

# Print the polynomial coefficients
print("Polynomial coefficients (highest degree first):")
print(coeffs)
# results = []

# for S in state_sizes:
#     results.append(run_experiments(S, action_size))

# # Prepare data for plotting
# direct_results, bisection_results, random_results, random_kernel_results = zip(*results)

# plt.figure(figsize=(10, 6))
# plt.plot(
#     state_sizes,
#     direct_results,
#     marker="o",
#     label="Maximize Using SLSQP",
#     color="black",
#     alpha=0.6,
# )
# plt.plot(
#     state_sizes,
#     bisection_results,
#     marker="^",
#     label="Algorithm 1: Binary Search",
#     alpha=0.7,
#     color="red",
# )
# plt.plot(
#     state_sizes,
#     random_results,
#     marker="s",
#     label="Maximize using Random Rank 1 Kernel Guess",
# )
# plt.plot(
#     state_sizes,
#     random_kernel_results,
#     marker="o",
#     label="Maximize using Random Kernel Guess",
# )


# plt.xlabel("State Size")
# plt.ylabel("Penalty (J^pi-J^pi_{U_2}) (Bigger is Better)")
# plt.title("L2 Robust Policy Evaluation")
# plt.legend()
# plt.grid(True, which="both", ls="--", alpha=0.7)

# plt.tight_layout()
# plt.savefig("optimization_methods_comparison_line.png", dpi=300, bbox_inches="tight")
# plt.show()
