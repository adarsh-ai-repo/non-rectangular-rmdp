import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import typer
from joblib import Parallel, delayed
from rich import print
from tqdm import tqdm

from algorithm_1 import optimize_using_eigen_value_and_bisection
from brute_force_optimized import RPE_Brute_Force_Numba

# from brute_force import RPE_Brute_Force
from cpi_algorithm import run_cpi_algorithm
from datamodels import (
    PMDerivedValues,
    PMRandomComponents,
    PMUserParameters,
    initialize_empty_performance_data,
)
from db_operations import initialize_database, save_performance_data

# from rank_1_random_matrix import optimize_using_random_rank_1_kernel

app = typer.Typer(pretty_exceptions_enable=False, help="Experiment runner for penalty values calculation")


def run_parallel_brute_force_iterations(
    params: PMUserParameters,
    random_components: PMRandomComponents,
    derived_values: PMDerivedValues,
    allowed_time_limit: float,
    start_time: float,
    num_samples: int = 10_000,
    n_jobs: int = -1,
    batch_size: int = 50,
) -> tuple[list[float], list[dict[str, Any]]]:
    """
    Run multiple brute force iterations in parallel using joblib.
    Each iteration processes exactly 10,000 samples and tracks performance.

    Args:
        params: User-defined parameters
        random_components: Random components of the MDP
        derived_values: Derived values from the MDP
        allowed_time_limit: Maximum time allowed for execution
        start_time: Start timestamp for the experiment
        num_samples: Number of samples per iteration (should be 10,000)
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        batch_size: Number of iterations to run in parallel batches

    Returns:
        tuple: (list of optimal values, list of performance data dictionaries)
    """
    all_optimal_values: list[float] = []
    all_performance_data: list[dict[str, Any]] = []
    iteration_count = 1
    experiment_start_time = time.time()

    while (time.time() - experiment_start_time) < allowed_time_limit:
        # Calculate remaining time
        remaining_time = allowed_time_limit - (time.time() - experiment_start_time)
        if remaining_time <= 0:
            break

        # Run batch of iterations in parallel
        batch_results: list[float] = Parallel(n_jobs=n_jobs)(  # type: ignore
            delayed(RPE_Brute_Force_Numba)(params, random_components, num_samples, derived_values)
            for _ in tqdm(range(batch_size))
        )

        # Process each result from the batch
        for i, optimal_value in enumerate(batch_results):
            all_optimal_values.append(optimal_value)

            # Calculate metrics for this specific iteration
            current_time = time.time()
            gamma_adjusted_value: float = optimal_value  # params.gamma * optimal_value
            current_best = max(
                [val for val in all_optimal_values]
            )  # params.gamma * val for val in all_optimal_values])

            # Create performance data entry for this iteration
            perf_data = {
                "algorithm_name": "brute_force",
                "iteration_count": iteration_count,
                "time_taken": current_time - experiment_start_time,
                "j_pi": derived_values.j_pi - current_best,
                "S": params.S,
                "A": params.A,
                "beta": params.beta,
                "hash": random_components.md5_hash,
                "start_time": datetime.fromtimestamp(start_time),
                "nominal_return": derived_values.j_pi,
                "optimal_value": optimal_value,
                "gamma_adjusted_value": gamma_adjusted_value,
                "batch_index": i,
                "total_iterations_so_far": len(all_optimal_values),
            }

            all_performance_data.append(perf_data)
            iteration_count += num_samples

        # Check if we should continue
        if (time.time() - experiment_start_time) >= allowed_time_limit:
            break

    return all_optimal_values, all_performance_data


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
    random_components = PMRandomComponents.generate(params.S, params.A)
    performance_data = initialize_empty_performance_data()

    derived_values = PMDerivedValues.calculate(params, random_components)

    # Bisection method
    _bisection_result = optimize_using_eigen_value_and_bisection(
        params, random_components, derived_values, performance_data, random_components.md5_hash
    )
    iteration_count = 1
    best_robust_return = 10000000.0

    best_robust_return, iteration_count = run_cpi_algorithm(
        params,
        random_components,
        derived_values,
        performance_data,
        iteration_count,
        best_robust_return,
        max_iter=1000,
    )

    # cpi_based_time_limit = cpi_algorithm_time_taken / 3 + EXTRA_TIME_LIMIT_IN_SEC
    max_time_limit = 10 * 60
    # Direct optimization using SLSQP

    # start_time = time.time()
    # direct_results: list[float] = []
    # direct_start_time = time.time()
    # for i in range(num_trials):
    #     if ((time.time() - direct_start_time) > max_time_limit) and (i > 2):
    #         break
    #     optimize_using_slsqp_method(
    #         params,
    #         derived_values,
    #         random_components,
    #         direct_start_time,
    #         direct_results,
    #         i + 1,
    #         performance_data,
    #     )
    # scipy_algorithm_time_taken = time.time() - start_time
    # allowed_time_limit = scipy_algorithm_time_taken + EXTRA_TIME_LIMIT_IN_SEC

    # # Random rank-1 kernel search
    # start_time = time.time()
    # random_results: list[float] = []
    # random_start_time = time.time()
    # iteration_count = 0
    # while (time.time() - random_start_time) < 7200:
    #     optimal_value, _, _ = optimize_using_random_rank_1_kernel(
    #         params,
    #         derived_values,
    #         num_guesses=2000,
    #     )
    #     random_results.append(params.gamma * optimal_value)
    #     iteration_time = time.time() - random_start_time
    #     performance_data["algorithm_name"].append("random_rank_1_kernel")
    #     performance_data["iteration_count"].append(iteration_count)
    #     performance_data["time_taken"].append(iteration_time)
    #     performance_data["j_pi"].append(derived_values.j_pi - max(random_results))
    #     performance_data["S"].append(params.S)
    #     performance_data["A"].append(params.A)
    #     performance_data["beta"].append(params.beta)
    #     performance_data["hash"].append(random_components.md5_hash)
    #     performance_data["start_time"].append(datetime.fromtimestamp(start_time))
    #     performance_data["nominal_return"].append(derived_values.j_pi)

    #     iteration_count += 2000
    # _random_result = max(random_results)

    # Brute force random kernel search using parallel processing
    random_start_time = time.time()

    # Run parallel brute force iterations
    all_optimal_values, all_performance_data = run_parallel_brute_force_iterations(
        params=params,
        random_components=random_components,
        derived_values=derived_values,
        allowed_time_limit=max_time_limit,
        num_samples=100_000,
        start_time=random_start_time,
        n_jobs=-1,  # Use all available CPU cores
    )

    # Convert to gamma-adjusted values for compatibility
    random_kernel_results = [params.gamma * val for val in all_optimal_values]
    _random_kernel_result = max(random_kernel_results) if random_kernel_results else 0.0

    # Convert performance data from list of dicts to dict of lists format
    for perf_data in all_performance_data:
        performance_data["algorithm_name"].append(perf_data["algorithm_name"])
        performance_data["iteration_count"].append(perf_data["iteration_count"])
        performance_data["time_taken"].append(perf_data["time_taken"])
        performance_data["j_pi"].append(perf_data["j_pi"])
        performance_data["S"].append(perf_data["S"])
        performance_data["A"].append(perf_data["A"])
        performance_data["beta"].append(perf_data["beta"])
        performance_data["hash"].append(perf_data["hash"])
        performance_data["start_time"].append(perf_data["start_time"])
        performance_data["nominal_return"].append(perf_data["nominal_return"])

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
