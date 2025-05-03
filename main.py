import time

import numpy as np
import pandas as pd
import polars as pl
import typer
from rich import print

from algorithm_1 import optimize_using_eigen_value_and_bisection
from brute_force import RPE_Brute_Force
from cpi_algorithm import run_cpi_algorithm
from datamodels import (
    PMDerivedValues,
    PMRandomComponents,
    PMUserParameters,
    initialize_empty_performance_data,
)
from optimize_using_scipy import optimize_using_slsqp_method
from rank_1_random_matrix import optimize_using_random_rank_1_kernel

app = typer.Typer(help="Experiment runner for penalty values calculation")

MAX_TIME_LIMIT_IN_SEC = 30


def run_experiments(S: int, A: int, beta: float, num_trials: int = 8) -> pl.DataFrame:
    """
    Run experiments with different optimization methods.

    Args:
        S: Number of states
        A: Number of actions
        num_trials: Number of trials for direct optimization

    Returns:
        tuple: Results from different optimization methods
    """
    # Initialize parameters and components
    params = PMUserParameters(S=S, A=A, beta=beta, gamma=0.9, tolerance=1e-5)
    random_components = PMRandomComponents.generate(params)
    performance_data = initialize_empty_performance_data()

    derived_values = PMDerivedValues.calculate(params, random_components)

    # Bisection method
    bisection_result = optimize_using_eigen_value_and_bisection(
        params, derived_values, performance_data, random_components.md5_hash
    )

    run_cpi_algorithm(params, random_components, performance_data, max_iter=4)

    # Direct optimization using SLSQP
    direct_results: list[float] = []
    slsqp_iteration_count = 0
    direct_start_time = time.time()
    for i in range(num_trials):
        if ((time.time() - direct_start_time) > MAX_TIME_LIMIT_IN_SEC) and (i > 2):
            break
        optimize_using_slsqp_method(
            params,
            derived_values,
            random_components,
            direct_start_time,
            direct_results,
            slsqp_iteration_count,
            performance_data,
        )

    # Random rank-1 kernel search
    random_results: list[float] = []
    random_start_time = time.time()
    while (time.time() - random_start_time) < MAX_TIME_LIMIT_IN_SEC:
        optimal_value, _, _ = optimize_using_random_rank_1_kernel(
            params, derived_values, performance_data, random_components.md5_hash, num_guesses=2000
        )
        random_results.append(params.gamma * optimal_value)
    random_result = max(random_results)

    # Brute force random kernel search
    random_kernel_results: list[float] = []
    random_start_time = time.time()
    while (time.time() - random_start_time) < MAX_TIME_LIMIT_IN_SEC:
        optimal_value = RPE_Brute_Force(
            params,
            random_components,
            num_samples=2000,
            derived_values=derived_values,
            performance_data=performance_data,
            rc_hash=random_components.md5_hash,
        )
        random_kernel_results.append(params.gamma * optimal_value)
    random_kernel_result = max(random_kernel_results)

    return pl.DataFrame(performance_data)


@app.command()
def main(
    start: int = typer.Argument(..., help="Starting value"),
    step: int = typer.Argument(..., help="Step size between values"),
    count: int = typer.Argument(..., help="Number of values to generate"),
    beta: float = typer.Argument(..., help="Beta value"),
) -> None:
    """
    Run experiments with different parameters and save results to CSV.
    """
    print("[green]Starting experiment with:[/green]")
    print(f"Start: {start}")
    print(f"Step: {step}")
    print(f"Count: {count}")
    print(f"Beta: {beta}")

    state_sizes = np.arange(start, start + (step * count), step)
    action_size = 8  # Fixed action space size

    results: list[pl.DataFrame] = []

    filename = f"penalty_values_S_{start}_step_{step}_count_{count}_beta_{beta}.parquet"

    with typer.progressbar(state_sizes) as progress:
        for S in progress:
            results.append(run_experiments(int(S), action_size, beta))
            results_df = pl.concat(results)
            results_df.write_parquet(filename)

    print(f"[green]Results saved to:[/green] {filename}")


if __name__ == "__main__":
    app()
