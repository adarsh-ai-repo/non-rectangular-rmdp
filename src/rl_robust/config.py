from typing import List
from pathlib import Path

from pydantic import BaseModel, Field


class AlgorithmConfig(BaseModel):
    enabled: bool = True
    num_trials: int = 8
    time_limit: float = 120.0


class AlgorithmsConfig(BaseModel):
    direct_optimization: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    bisection: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    random_search: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    random_kernel: AlgorithmConfig = Field(default_factory=AlgorithmConfig)


class ProblemConfig(BaseModel):
    state_sizes: List[int] = Field(default_factory=lambda: [10, 20, 30, 41, 51])
    action_size: int = 8
    gamma: float = 0.9
    beta: float = 0.1
    tolerance: float = 1e-5


class PlotConfig(BaseModel):
    figsize: tuple[int, int] = (12, 7)
    title: str = "L2 Robust Policy Evaluation"
    xlabel: str = "State Size"
    ylabel: str = "Penalty (J^pi-J^pi_{U_2})"


class ExperimentConfig(BaseModel):
    cache_dir: Path = Field(default="cache")
    output_dir: Path = Field(default="results")
    plot_settings: PlotConfig = Field(default_factory=PlotConfig)


class Config(BaseModel):
    algorithms: AlgorithmsConfig = Field(default_factory=AlgorithmsConfig)
    problem: ProblemConfig = Field(default_factory=ProblemConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
