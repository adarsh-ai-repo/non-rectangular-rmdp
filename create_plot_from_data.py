from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, Field


class BasePlotConfig(BaseModel):
    """Base configuration for all plots."""

    xlabel: str = Field(..., description="X-axis label")
    ylabel: str = Field(..., description="Y-axis label")
    title: str = Field(..., description="Plot title")
    output_path: Path = Field(..., description="Path to save the plot")


class ConvergencePlotConfig(BasePlotConfig):
    """Configuration for convergence comparison plot."""

    input_files: Dict[str, Path] = Field(
        description="""{
            "slsqp": "Path to SLSQP data",
            "eigen": "Path to eigenvalue data",
            "rank1": "Path to rank-1 data",
            "random_kernel": "Path to random kernel data",
        },"""
    )


class PenaltyPlotConfig(BasePlotConfig):
    """Configuration for penalty comparison plots."""

    input_dir: Path = Field(..., description="Directory containing penalty CSV files")
    output_dir: Path = Field(..., description="Directory for saving penalty plots")


class RandomKernelPlotConfig(BasePlotConfig):
    """Configuration for random kernel comparison plot."""

    input_files: Dict[str, Path] = Field(
        description="""{
            "random_kernel": "Path to random kernel data",
            "eigen": "Path to eigenvalue data",
        },"""
    )


def set_plot_style():
    """Configure the global plotting style settings."""
    plt.style.use("bmh")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 17,
            "legend.fontsize": 14,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def create_figure():
    """Create and return a new figure with standard size."""
    return plt.figure(figsize=(12, 8), dpi=300)


def setup_plot_basics(xlabel: str, ylabel: str, title: str):
    """Set up basic plot elements like labels and grid."""
    plt.xlabel(xlabel, labelpad=10)
    plt.ylabel(ylabel, labelpad=10)
    plt.title(title, pad=15)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)


def save_and_close_plot(filepath: Path):
    """Save the plot to file and close the figure."""
    plt.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def plot_convergence_comparison(config: ConvergencePlotConfig):
    """Create and save the convergence comparison plot."""
    data = {key: pd.read_csv(filepath) for key, filepath in config.input_files.items()}

    create_figure()

    # Plot SLSQP
    plt.plot(
        data["slsqp"]["time"].abs().tolist() + [175],
        data["slsqp"]["Maximize Using SLSQP"].tolist()
        + [data["slsqp"]["Maximize Using SLSQP"].max()],
        label="SLSQP",
        linewidth=2.5,
    )

    # Plot Eigenvalue
    plt.plot(
        [0] + data["eigen"]["time"].abs().tolist() + [170],
        [0]
        + data["eigen"]["Algorithm 1"].tolist()
        + [data["eigen"]["Algorithm 1"].max()],
        label="Eigenvalue",
        linewidth=2.5,
    )

    # Plot Random Rank 1
    plt.plot(
        data["rank1"]["time"].abs().tolist() + [175],
        data["rank1"]["Random Rank 1 kernel"].tolist()
        + [data["rank1"]["Random Rank 1 kernel"].max()],
        label="Random Rank 1",
        linewidth=2.5,
    )

    # Plot Random Kernel
    plt.plot(
        data["random_kernel"]["time"].abs(),
        data["random_kernel"]["Random kernel"],
        label="Random Kernel",
        linewidth=2.5,
    )

    setup_plot_basics(config.xlabel, config.ylabel, config.title)
    save_and_close_plot(config.output_path)


def plot_penalty_comparisons(config: PenaltyPlotConfig):
    """Create and save penalty comparison plots for each penalty CSV file."""
    for file in config.input_dir.glob("penalty*.csv"):
        df = pd.read_csv(file)
        create_figure()

        algorithms = {
            "Maximize Using SLSQP": ("SLSQP", "o"),
            "Algorithm 1: Binary Search": ("Binary Search", "s"),
            "Maximize using Random Rank 1 Kernel Guess": ("Random Rank 1 Kernel", "^"),
            "Maximize using Random Kernel Guess": ("Random Kernel", "*"),
        }

        for col, (label, marker) in algorithms.items():
            plt.plot(df["S"], df[col], marker=marker, label=label)

        setup_plot_basics(config.xlabel, config.ylabel, config.title)
        # Change the extension from .csv to .png
        output_filename = file.stem + ".png"
        save_and_close_plot(config.output_dir / output_filename)


def plot_random_kernel_comparison(config: RandomKernelPlotConfig):
    """Create and save random kernel comparison plot."""
    random_kernel = pd.read_csv(config.input_files["random_kernel"])
    eigen = pd.read_csv(config.input_files["eigen"])

    create_figure()

    plt.plot(
        eigen["time"].abs().tolist(),
        eigen["Algorithm 1"].tolist(),
        label="Optimal Value",
        linestyle=":",
        linewidth=2,
    )

    plt.plot(
        random_kernel["time"].abs(),
        random_kernel["Random kernel"],
        label="Random Kernel",
    )

    setup_plot_basics(config.xlabel, config.ylabel, config.title)
    save_and_close_plot(config.output_path)


if __name__ == "__main__":
    # Common ylabel for all plots
    penalty_ylabel = "Penalty ($J^\pi-J^\pi_{U_2}$) (Bigger is Better)"

    # Configuration for convergence plot
    convergence_config = ConvergencePlotConfig(
        xlabel="Time (in seconds)",
        ylabel=penalty_ylabel,
        title="Convergence",
        output_path=Path("plots/method_vs_time_comparison.png"),
        input_files={
            "slsqp": Path("data/time_vs_slsqp.csv"),
            "eigen": Path("data/time_vs_eigin_value.csv"),
            "rank1": Path("data/time_vs_rank_1_random.csv"),
            "random_kernel": Path("data/time_vs_random_kernel.csv"),
        },
    )

    # Configuration for penalty plots
    penalty_config = PenaltyPlotConfig(
        xlabel="S (State Size)",
        ylabel=penalty_ylabel,
        title="Performance",
        output_path=Path("plots"),
        input_dir=Path("data"),
        output_dir=Path("plots"),
    )

    # Configuration for random kernel plot
    random_kernel_config = RandomKernelPlotConfig(
        xlabel="Time (sec)",
        ylabel=penalty_ylabel,
        title="Brute Force Random Kernel Penalty vs Time",
        output_path=Path("plots/Random Kernel Penalty vs Time.png"),
        input_files={
            "random_kernel": Path("data/time_vs_random_kernel_s_32.csv"),
            "eigen": Path("data/time_vs_eigin_value_s_32.csv"),
        },
    )

    # Set style and create plots
    set_plot_style()
    plot_convergence_comparison(convergence_config)
    plot_penalty_comparisons(penalty_config)
    plot_random_kernel_comparison(random_kernel_config)
