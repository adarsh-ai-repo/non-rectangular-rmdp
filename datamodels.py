from typing import Tuple

import numpy as np
from pydantic import BaseModel, Field


class PMUserParameters(BaseModel):
    S: int = Field(..., gt=0, description="Number of states")
    A: int = Field(..., gt=0, description="Number of actions")
    beta: float = Field(0.1, gt=0, description="Uncertainty radius")
    gamma: float = Field(0.9, gt=0, lt=1, description="Discount factor")
    tolerance: float = Field(1e-5, gt=0, description="Convergence tolerance")

    def __init__(self, **data):
        super().__init__(**data)
        if self.mu is None:
            self.mu = np.ones(self.S) / self.S
        self.dim_b_vector = self.S * self.A

    class Config:
        arbitrary_types_allowed = True


class PMRandomComponents(BaseModel):
    v0: np.ndarray  # initial value function
    pi: np.ndarray  # initial policy
    R: np.ndarray  # reward function
    P: np.ndarray  # transition kernel

    @staticmethod
    def kernel(S: int, A: int) -> np.ndarray:
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

    class Config:
        arbitrary_types_allowed = True


class PMDerivedValues(BaseModel):
    P_pi: np.ndarray  # policy-averaged transition kernel
    R_pi: np.ndarray  # policy-averaged reward
    v_pi: np.ndarray  # value function
    D_pi: np.ndarray  # occupation matrix
    d_pi: np.ndarray  # occupation measure
    H: np.ndarray  # policy matrix
    PHI: np.ndarray  # phi matrix
    matrix_part1: np.ndarray
    matrix_part2: np.ndarray
    mu: np.ndarray  # initial state distribution
    dim_b_vector: int

    @staticmethod
    def compute_value_function(
        P_pi: np.ndarray, R_pi: np.ndarray, gamma: float
    ) -> np.ndarray:
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
        P_pi: np.ndarray, gamma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        )

    class Config:
        arbitrary_types_allowed = True
