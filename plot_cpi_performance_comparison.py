from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from create_plot_from_data import (
    create_figure,
    save_and_close_plot,
    set_plot_style,
    setup_plot_basics,
)


def load_performance_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load performance data from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame containing the performance data
    """
    return pd.read_csv(file_path)


def get_unique_beta_values(df: pd.DataFrame) -> list[float]:
    """
    Get unique beta values from the DataFrame.

    Args:
        df: DataFrame containing performance data

    Returns:
        List of unique beta values
    """
    return sorted(df["beta"].unique())


def plot_robust_return_vs_iterations(df: pd.DataFrame, beta_value: float, output_dir: Path) -> None:
    """
    Create a plot of robust return vs iteration count for a specific beta value.

    Args:
        df: DataFrame containing performance data
        beta_value: Beta value to filter data for
        output_dir: Directory to save the plot
    """
    # Filter data for the current beta value
    beta_df = df[df["beta"] == beta_value]

    # Filter data for each algorithm
    cpi_data = beta_df[beta_df["algorithm_name"] == "cpi_algorithm"]
    algorithm_1 = beta_df[beta_df["algorithm_name"] == "eigen_bisection"]

    algorithm_1 = pd.concat(
        [
            algorithm_1,
            algorithm_1.loc[lambda df: df["iteration_count"] == df["iteration_count"].max()].assign(
                iteration_count=cpi_data["iteration_count"].max()
            ),
        ]
    )

    # Get nominal return value from CPI algorithm data

    # Create figure
    create_figure()

    # Plot data
    plt.plot(cpi_data["iteration_count"], cpi_data["robust_return"], label="CPI Algorithm")
    plt.plot(algorithm_1["iteration_count"], algorithm_1["robust_return"], label="Algorithm 1 (Ours)")

    # Add horizontal line for nominal return
    nominal_return = cpi_data["nominal_return"].iloc[0]
    plt.axhline(y=nominal_return, color="r", linestyle="--", label=f"Nominal Return: {nominal_return:.3f}")
    plt.axhline(y=-0.8364, linestyle=":", color="darkcyan", label=f"Empirical Robust Return: {-0.8364:.3f}")

    # Check if there are any very negative values that need special handling
    min_robust_return = min(cpi_data["robust_return"].min(), algorithm_1["robust_return"].min())

    # If the minimum value is significantly more negative than the nominal return,
    # use a broken axis approach to indicate very negative values
    if min_robust_return < nominal_return - 0.5:
        # Set y-axis limits to show a reasonable range below nominal return
        range_size = abs((abs(nominal_return) - algorithm_1["robust_return"].abs().min()))
        y_min = max(min_robust_return + 1.0, algorithm_1["robust_return"].max() - 3 * range_size)
        plt.ylim(y_min, nominal_return + range_size)

        # Add diagonal lines to indicate broken axis
        d = 0.015  # Size of the diagonal lines in axes coordinates

        # Get the current axes
        ax = plt.gca()

        # Draw the diagonal lines at the bottom of the plot
        # Use explicit parameters instead of kwargs to avoid type errors
        ax.plot((-d, +d), (-d, +d), transform=ax.transAxes, color="k", clip_on=False, linewidth=1)
        ax.plot((1 - d, 1 + d), (-d, +d), transform=ax.transAxes, color="k", clip_on=False, linewidth=1)

        # Add text annotation for very negative values
        plt.annotate(
            f"Values down to {min_robust_return:.2f}",
            xy=(0.2, 0.01),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8),
        )
    else:
        # Use the original approach if there are no extremely negative values
        range_size = abs((abs(nominal_return) - algorithm_1["robust_return"].abs().min()))
        plt.ylim(algorithm_1["robust_return"].max() - 3 * range_size, nominal_return + range_size)

    # Set x-axis to logarithmic scale (base 10)
    plt.xscale("log", base=10)

    # Setup plot basics
    setup_plot_basics(
        xlabel="Iterations (Log Scale)",
        ylabel="Robust Return (Lower is Better)",
        title="Complexity (Iterations)",
    )

    # Save plot
    output_path = output_dir / f"{output_dir.name}_beta_{beta_value}.png"
    save_and_close_plot(output_path)


def plot_robust_return_vs_time(df: pd.DataFrame, beta_value: float, output_dir: Path) -> None:
    """
    Create a plot of robust return vs time taken for a specific beta value.

    Args:
        df: DataFrame containing performance data
        beta_value: Beta value to filter data for
        output_dir: Directory to save the plot
    """
    # Filter data for the current beta value
    beta_df = df[df["beta"] == beta_value]

    # Filter data for each algorithm
    cpi_data = beta_df[beta_df["algorithm_name"] == "cpi_algorithm"]
    algorithm_1 = beta_df[beta_df["algorithm_name"] == "eigen_bisection"]
    cpi_data = pd.concat(
        [
            cpi_data.loc[lambda df: df["iteration_count"] == df["iteration_count"].min()].assign(
                time_taken=algorithm_1["time_taken"].min(), iteration_count=0
            ),
            cpi_data,
        ]
    )
    algorithm_1 = pd.concat(
        [
            algorithm_1,
            algorithm_1.loc[lambda df: df["iteration_count"] == df["iteration_count"].max()].assign(
                time_taken=cpi_data["time_taken"].max()
            ),
        ]
    )
    # Create figure
    create_figure()

    # Plot data
    plt.plot(cpi_data["time_taken"], cpi_data["robust_return"], label="CPI Algorithm")
    plt.plot(algorithm_1["time_taken"], algorithm_1["robust_return"], label="Algorithm 1 (Ours)")

    nominal_return = cpi_data["nominal_return"].iloc[0]
    plt.axhline(
        y=nominal_return,
        color="r",
        linestyle="--",
        label=f"Nominal Return: {nominal_return:.4f}",
    )
    plt.axhline(y=-0.8364, linestyle=":", color="darkcyan", label=f"Empirical Robust Return: {-0.8364:.3f}")

    # Check if there are any very negative values that need special handling
    min_robust_return = min(cpi_data["robust_return"].min(), algorithm_1["robust_return"].min())

    # If the minimum value is significantly more negative than the nominal return,
    # use a broken axis approach to indicate very negative values
    if min_robust_return < nominal_return - 0.5:
        # Set y-axis limits to show a reasonable range below nominal return
        range_size = abs((abs(nominal_return) - algorithm_1["robust_return"].abs().min()))
        y_min = max(min_robust_return + 1.0, algorithm_1["robust_return"].max() - 3 * range_size)
        plt.ylim(y_min, nominal_return + range_size)

        # Add diagonal lines to indicate broken axis
        d = 0.015  # Size of the diagonal lines in axes coordinates

        # Get the current axes
        ax = plt.gca()

        # Draw the diagonal lines at the bottom of the plot
        # Use explicit parameters instead of kwargs to avoid type errors
        ax.plot((-d, +d), (-d, +d), transform=ax.transAxes, color="k", clip_on=False, linewidth=1)
        ax.plot((1 - d, 1 + d), (-d, +d), transform=ax.transAxes, color="k", clip_on=False, linewidth=1)

        # Add text annotation for very negative values
        plt.annotate(
            f"Values down to {min_robust_return:.2f}",
            xy=(0.2, 0.01),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8),
        )
    else:
        # Use the original approach if there are no extremely negative values
        range_size = abs((abs(nominal_return) - algorithm_1["robust_return"].abs().min()))
        plt.ylim(algorithm_1["robust_return"].max() - 3 * range_size, nominal_return + range_size)

    # Set x-axis to logarithmic scale (base 10)
    plt.xscale("log", base=10)

    # Setup plot basics
    setup_plot_basics(
        xlabel="Time (seconds, Log Scale)",
        ylabel="Robust Return (Lower is Better)",
        title="Complexity (Time)",
    )

    # Save plot
    output_path = output_dir / f"{output_dir.name}_beta_{beta_value}.png"
    save_and_close_plot(output_path)


def main() -> None:
    """Main function to drive the data loading, processing, and plot generation."""
    # Set plot style
    set_plot_style()

    # Define paths
    data_path = Path("performance_data_202505111634.csv")
    iteration_plots_dir = Path("plots/cpi_performance_comparison/iteration_plots")
    time_plots_dir = Path("plots/cpi_performance_comparison/time_plots")

    # Load data
    df = load_performance_data(data_path).loc[lambda x: x["S"] == 10]

    # Get unique beta values
    beta_values = get_unique_beta_values(df)

    # Create plots for each beta value
    for beta in beta_values:
        # Create standard plots
        plot_robust_return_vs_iterations(df, beta, iteration_plots_dir)
        plot_robust_return_vs_time(df, beta, time_plots_dir)


if __name__ == "__main__":
    main()
