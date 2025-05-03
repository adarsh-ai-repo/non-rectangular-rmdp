import hashlib
from datetime import datetime
from functools import cached_property
from typing import Tuple, TypedDict

import numpy as np
from jaxtyping import Float, Shaped
from pydantic import BaseModel, Field

S = "S"
A = "A"


class PMUserParameters(BaseModel):
    S: int = Field(..., gt=0, description="Number of states")
    A: int = Field(..., gt=0, description="Number of actions")
    beta: float = Field(..., gt=0, description="Uncertainty radius")
    gamma: float = Field(0.9, gt=0, lt=1, description="Discount factor")
    tolerance: float = Field(1e-4, gt=0, description="Convergence tolerance")


def array_string_repr(array: Shaped[np.ndarray, "..."]) -> str:
    return np.array2string(
        np.array(array), precision=2, separator=",", suppress_small=True, max_line_width=100_000
    )


class PMRandomComponents(BaseModel):
    v0: Float[np.ndarray, "S"]  # initial value function
    pi: Float[np.ndarray, "S A"]  # initial policy
    R: Float[np.ndarray, "S A"]  # reward function
    P: Float[np.ndarray, "S A S"]  # transition kernel

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def kernel(S: int, A: int) -> Float[np.ndarray, "S A S"]:
        p = np.random.rand(S, A, S)
        for s in range(S):
            for a in range(A):
                summ = np.sum(p[s, a])
                p[s, a] = p[s, a] / summ
        return p

    @classmethod
    def generate(cls, params: PMUserParameters):
        S, A = params.S, params.A

        # Generate random components
        v0 = np.random.randn(S)
        pi = np.random.rand(S, A)
        # Normalize policy
        for s in range(S):
            pi[s] = pi[s] / np.sum(pi[s])

        R = np.random.randn(S, A)
        P = cls.kernel(S, A)

        return cls(v0=v0, pi=pi, R=R, P=P)

    @cached_property
    def md5_hash(self) -> str:
        """
        Calculate MD5 hash based on the string representation of arrays with 2 decimal places.

        Returns:
            str: MD5 hash of the components
        """
        # Format each array with 2 decimal places
        v0_str = array_string_repr(self.v0)
        pi_str = array_string_repr(self.pi)
        R_str = array_string_repr(self.R)
        P_str = array_string_repr(self.P)

        # Combine all representations
        combined_str = f"{v0_str}\n{pi_str}\n{R_str}\n{P_str}"

        # Calculate MD5 hash
        md5 = hashlib.md5(combined_str.encode("utf-8"))

        return md5.hexdigest()


class PMDerivedValues(BaseModel):
    P_pi: Float[np.ndarray, "S S"]  # policy-averaged transition kernel
    R_pi: Float[np.ndarray, "S"]  # policy-averaged reward
    v_pi: Float[np.ndarray, "S"]  # value function
    D_pi: Float[np.ndarray, "S S"]  # occupation matrix
    d_pi: Float[np.ndarray, "S"]  # occupation measure
    H: Float[np.ndarray, "S SA"]  # policy matrix
    PHI: Float[np.ndarray, "S S"]  # phi matrix
    matrix_part1: Float[np.ndarray, "S SA"]
    matrix_part2: Float[np.ndarray, "S SA"]
    mu: Float[np.ndarray, "S"]  # initial state distribution
    dim_b_vector: int
    j_pi: float

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def compute_value_function(
        P_pi: Float[np.ndarray, "S S"], R_pi: Float[np.ndarray, "S"], gamma: float
    ) -> Float[np.ndarray, "S"]:
        """
        Compute the value function v^π given the policy-averaged transition kernel and reward function.
        """
        n_states = P_pi.shape[0]
        I_matrix = np.eye(n_states)
        A = I_matrix - gamma * P_pi
        A_inv = np.linalg.inv(A)
        v_pi = A_inv @ R_pi
        return v_pi

    @staticmethod
    def compute_occupation_measures(
        P_pi: Float[np.ndarray, "S S"], gamma: float
    ) -> Tuple[Float[np.ndarray, "S S"], Float[np.ndarray, "S"]]:
        """
        Compute the occupation matrix D^π and occupation measure d^π.
        """
        n_states = P_pi.shape[0]
        initial_distribution = np.ones(n_states) / n_states

        I_matrix = np.eye(n_states)
        A_inv = np.linalg.inv(I_matrix - gamma * P_pi)

        d_pi = initial_distribution @ A_inv
        D_pi = np.diag(d_pi) @ P_pi

        return D_pi, d_pi

    @classmethod
    def calculate(cls, params: PMUserParameters, random_components: PMRandomComponents):
        S, A = params.S, params.A
        gamma = params.gamma
        # Calculate initial state distribution
        mu = np.ones(S) / S

        # Calculate dimension of b vector
        dim_b_vector = S * A

        # Calculate P_pi
        P_pi = np.zeros((S, S))
        for s in range(S):
            for t in range(S):
                P_pi[s, t] = random_components.pi[s, :] @ random_components.P[s, :, t]

        # Calculate R_pi
        R_pi = np.sum(random_components.R * random_components.pi, axis=1)

        # Calculate v_pi using class method
        v_pi = cls.compute_value_function(P_pi, R_pi, gamma)

        # Calculate D_pi and d_pi using class method
        D_pi, d_pi = cls.compute_occupation_measures(P_pi, gamma)

        # Calculate H
        H = np.zeros((S, S * A))
        for i in range(S):
            for j in range(A):
                H[i, i * A + j] = random_components.pi[i, j]

        # Calculate PHI
        PHI = np.eye(S, S) - 1 / S

        # Calculate matrix parts
        matrix_part1 = PHI @ np.outer(v_pi, d_pi) @ H
        matrix_part2 = PHI @ D_pi @ H
        j_pi = float(mu @ v_pi)
        print(f"Nominal {j_pi=}")
        return cls(
            P_pi=P_pi,
            R_pi=R_pi,
            v_pi=v_pi,
            D_pi=D_pi,
            d_pi=d_pi,
            H=H,
            PHI=PHI,
            matrix_part1=matrix_part1,
            matrix_part2=matrix_part2,
            mu=mu,
            dim_b_vector=dim_b_vector,
            j_pi=j_pi,
        )


class AlgorithmPerformanceData(TypedDict):
    """
    TypedDict for data collected during CPI Algorithm execution.

    This structure matches the dictionary used to collect data for the DataFrame
    in the run_cpi_algorithm function.

    Attributes:
        algorithm_name: List of algorithm names (always "cpi_algorithm")
        iteration_count: List of iteration numbers (1-based)
        time_taken: List of execution times per iteration in seconds
        Penalty: List of penalty values (J^π - J^π_{P_n})
        S: List of state space sizes
        A: List of action space sizes
        beta: List of uncertainty radius values
        hash: List of MD5 hashes of PMRandomComponents
    """

    algorithm_name: list[str]
    iteration_count: list[int]
    time_taken: list[float]
    j_pi: list[float]
    S: list[int]
    A: list[int]
    beta: list[float]
    hash: list[str]
    start_time: list[datetime]


def initialize_empty_performance_data() -> AlgorithmPerformanceData:
    return {
        "algorithm_name": [],
        "iteration_count": [],
        "time_taken": [],
        "j_pi": [],
        "S": [],
        "A": [],
        "beta": [],
        "hash": [],
        "start_time": [],
    }
