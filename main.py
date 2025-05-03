import os
import time
from datetime import datetime

import numpy as np
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
from db_operations import initialize_database, load_performance_data, save_performance_data
from optimize_using_scipy import optimize_using_slsqp_method
from rank_1_random_matrix import optimize_using_random_rank_1_kernel

app = typer.Typer(pretty_exceptions_enable=False, help="Experiment runner for penalty values calculation")
EXTRA_TIME_LIMIT_IN_SEC = 5


def run_experiments(S: int, A: int, beta: float, db_path: str, num_trials: int = 5) -> None:
    """
    Run experiments with different optimization methods and save to SQLite.

    Args:
        S: Number of states
        A: Number of actions
        beta: Beta value
        db_path: Path to SQLite database
        num_trials: Number of trials for direct optimization
    """
    # Initialize parameters and components
    params = PMUserParameters(S=S, A=A, beta=beta, gamma=0.9, tolerance=1e-5)
    random_components = PMRandomComponents.generate(params.S, params.A, params.beta)
    performance_data = initialize_empty_performance_data()

    derived_values = PMDerivedValues.calculate(params, random_components)

    # Bisection method
    _bisection_result = optimize_using_eigen_value_and_bisection(
        params, derived_values, performance_data, random_components.md5_hash
    )
    # start_time = time.time()
    # iteration_count = 1
    # best_robust_return = 10000000.0

    # best_robust_return, iteration_count = run_cpi_algorithm(
    #     params,
    #     random_components,
    #     derived_values,
    #     performance_data,
    #     iteration_count,
    #     best_robust_return,
    #     max_iter=20,
    # )
    # cpi_algorithm_time_taken = time.time() - start_time

    # cpi_based_time_limit = cpi_algorithm_time_taken / 3 + EXTRA_TIME_LIMIT_IN_SEC
    cpi_based_time_limit = 3600
    # Direct optimization using SLSQP

    start_time = time.time()
    direct_results: list[float] = []
    direct_start_time = time.time()
    for i in range(num_trials):
        if ((time.time() - direct_start_time) > cpi_based_time_limit) and (i > 2):
            break
        optimize_using_slsqp_method(
            params,
            derived_values,
            random_components,
            direct_start_time,
            direct_results,
            i + 1,
            performance_data,
        )
    scipy_algorithm_time_taken = time.time() - start_time
    allowed_time_limit = scipy_algorithm_time_taken + EXTRA_TIME_LIMIT_IN_SEC

    # Random rank-1 kernel search
    random_results: list[float] = []
    random_start_time = time.time()
    iteration_count = 0
    while (time.time() - random_start_time) < allowed_time_limit:
        optimal_value, _, _ = optimize_using_random_rank_1_kernel(
            params,
            derived_values,
            num_guesses=2000,
        )
        random_results.append(params.gamma * optimal_value)
        iteration_time = time.time() - random_start_time
        performance_data["algorithm_name"].append("random_rank_1_kernel")
        performance_data["iteration_count"].append(iteration_count)
        performance_data["time_taken"].append(iteration_time)
        performance_data["j_pi"].append(derived_values.j_pi - max(random_results))
        performance_data["S"].append(params.S)
        performance_data["A"].append(params.A)
        performance_data["beta"].append(params.beta)
        performance_data["hash"].append(random_components.md5_hash)
        performance_data["start_time"].append(datetime.fromtimestamp(start_time))
        performance_data["nominal_return"].append(derived_values.j_pi)

        iteration_count += 2000
    _random_result = max(random_results)

    # Brute force random kernel search
    random_kernel_results: list[float] = []
    random_start_time = time.time()
    iteration_count = 1
    while (time.time() - random_start_time) < allowed_time_limit:
        optimal_value = RPE_Brute_Force(
            params,
            random_components,
            num_samples=200,
            derived_values=derived_values,
            performance_data=performance_data,
            rc_hash=random_components.md5_hash,
        )
        random_kernel_results.append(params.gamma * optimal_value)
        iteration_time = time.time() - random_start_time
        performance_data["algorithm_name"].append("brute_force")
        performance_data["iteration_count"].append(iteration_count)
        performance_data["time_taken"].append(iteration_time)
        performance_data["j_pi"].append(derived_values.j_pi - params.gamma * max(random_kernel_results))
        performance_data["S"].append(params.S)
        performance_data["A"].append(params.A)
        performance_data["beta"].append(params.beta)
        performance_data["hash"].append(random_components.md5_hash)
        performance_data["start_time"].append(datetime.fromtimestamp(start_time))
        performance_data["nominal_return"].append(derived_values.j_pi)

        iteration_count += 200
    _random_kernel_result = max(random_kernel_results)

    # Save performance data to SQLite
    save_performance_data(db_path, performance_data)


@app.command()
def main(
    start: int = typer.Argument(..., help="Starting value"),
    step: int = typer.Argument(..., help="Step size between values"),
    count: int = typer.Argument(..., help="Number of values to generate"),
    beta: float = typer.Argument(..., help="Beta value"),
    db_name: str = typer.Option("results.db", help="SQLite database name"),
) -> None:
    """
    Run experiments with different parameters and save results to SQLite.
    """
    print("[green]Starting experiment with:[/green]")
    print(f"Start: {start}")
    print(f"Step: {step}")
    print(f"Count: {count}")
    print(f"Beta: {beta}")

    state_sizes = np.arange(start, start + (step * count), step)
    action_size = 8  # Fixed action space size

    # Create database directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", db_name)

    # Initialize the database
    initialize_database(db_path)

    print(f"[green]Using SQLite database:[/green] {db_path}")

    with typer.progressbar(state_sizes) as progress:
        for S in progress:
            run_experiments(int(S), action_size, beta, db_path)

    print(f"[green]Results saved to SQLite database:[/green] {db_path}")


if __name__ == "__main__":
    app()
