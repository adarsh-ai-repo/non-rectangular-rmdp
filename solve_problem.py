import csv
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize


def objective(x: np.ndarray, A: np.ndarray) -> np.floating:
    return -np.linalg.norm(A @ x, 2)


def constraint(x: np.ndarray) -> np.floating:
    return 1 - np.linalg.norm(x, 2)


def solve_problem(A: np.ndarray, num_tries: int = 8) -> float:
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    n: int = A.shape[1]
    cons: dict = {"type": "ineq", "fun": constraint}
    bounds = [(0, None) for _ in range(n)]

    best_x = None
    best_obj = -np.inf

    for _ in range(num_tries):
        x0: np.ndarray = np.random.rand(n)
        x0 /= np.linalg.norm(x0, 2)  # Ensure initial guess is in feasible region

        try:
            res = minimize(
                objective,
                x0,
                args=(A,),
                method="SLSQP",
                constraints=cons,
                bounds=bounds,
                options={"maxiter": 10, "ftol": 5e-4, "iprint": 1, "disp": True},
            )

            if res.success and round(-res.fun, 3) > round(best_obj, 3):
                best_x = res.x
                best_obj = -res.fun
        except Exception:
            continue

    if best_x is None:
        raise RuntimeError("Optimization failed for all attempts")

    return best_obj


if __name__ == "__main__":
    dimensions = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    times = []

    for dim in dimensions:
        time_vec = np.zeros(4)
        for i in range(4):
            start_time = time.time()
            A: np.ndarray = np.random.randn(dim, dim)
            solution: float = solve_problem(A)
            end_time = time.time()

            elapsed_time = end_time - start_time
            time_vec[i] = elapsed_time

        times.append(time_vec.mean())

        print(f"Dimension: {dim}")
        print(f"Time taken: {elapsed_time:.4f} seconds")
        print()

    # Calculate the slope of log(time) vs log(dimension)
    log_dimensions = np.log2(dimensions)
    log_times = np.log2(times)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_dimensions, log_times
    )

    print(f"Slope of log(time) vs log(dimension): {slope:.4f}")
    print(f"R-squared: {r_value**2:.4f}")  # type: ignore

    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, times, marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("Dimension")
    plt.ylabel("Time (seconds)")
    plt.title("Time taken vs Dimension")
    plt.grid(True)

    # Save the plot to a file
    plt.savefig("time_vs_dimension.png")
    plt.close()

    # Save the data to a CSV file

    with open("time_vs_dimension.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Dimension", "Time (seconds)"])
        for dim, t in zip(dimensions, times):
            writer.writerow([dim, t])

    print("Plot saved as 'time_vs_dimension.png'")
    print("Data saved as 'time_vs_dimension.csv'")
