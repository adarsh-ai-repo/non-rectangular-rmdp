import time

import numpy as np
import pandas as pd
import typer
from rich import print

from algorithm_1 import optimize_using_eigen_value_and_bisection
from brute_force import RPE_Brute_Force
from datamodels import PMDerivedValues, PMRandomComponents, PMUserParameters
from optimize_using_scipy import optimize_using_slsqp_method
from rank_1_random_matrix import optimize_using_random_rank_1_kernel

app = typer.Typer(help="Experiment runner for penalty values calculation")


def run_experiments(S: int, A: int, beta: float, num_trials: int = 8) -> tuple:
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
    params = PMUserParameters(S=S, A=A, beta=beta)
    random_components = PMRandomComponents.generate(params)
    derived_values = PMDerivedValues.calculate(params, random_components)

    # Bisection method
    start_time = time.time()
    bisection_result = optimize_using_eigen_value_and_bisection(params, derived_values)
    time_limit = time.time() - start_time

    # Direct optimization using SLSQP
    direct_results = []
    direct_start_time = time.time()
    for i in range(num_trials):
        if ((time.time() - direct_start_time) > time_limit) and (i > 2):
            break
        try:
            optimal_value, _, _ = optimize_using_slsqp_method(params, derived_values)
            direct_results.append(params.gamma * optimal_value)
        except ValueError:
            pass
    direct_result = max(direct_results) if direct_results else np.nan
    direct_time = time.time() - direct_start_time

    # Random rank-1 kernel search
    random_results = []
    random_start_time = time.time()
    while (time.time() - random_start_time) < direct_time:
        optimal_value, _, _ = optimize_using_random_rank_1_kernel(
            params, derived_values, num_guesses=2000
        )
        random_results.append(params.gamma * optimal_value)
    random_result = max(random_results)

    # Brute force random kernel search
    random_kernel_results = []
    random_start_time = time.time()
    while time.time() - random_start_time < direct_time:
        optimal_value = RPE_Brute_Force(
            params, random_components, num_samples=2000, derived_values=derived_values
        )
        random_kernel_results.append(params.gamma * optimal_value)
    random_kernel_result = max(random_kernel_results)

    return S, A, direct_result, bisection_result, random_result, random_kernel_result


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

    results = []

    filename = f"penalty_values_S_{start}_step_{step}_count_{count}_beta_{beta}.csv"

    with typer.progressbar(state_sizes) as progress:
        for S in progress:
            results.append(run_experiments(S, action_size, beta))
            results_df = pd.DataFrame(
                np.array(results),
                columns=[
                    "S",
                    "A",
                    "Maximize Using SLSQP",
                    "Algorithm 1: Binary Search",
                    "Maximize using Random Rank 1 Kernel Guess",
                    "Maximize using Random Kernel Guess",
                ],
            )
            results_df.to_csv(filename)

    print(f"[green]Results saved to:[/green] {filename}")


if __name__ == "__main__":
    app()
