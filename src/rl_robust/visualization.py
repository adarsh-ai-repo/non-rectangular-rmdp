from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import polars as pl

from .config import PlotConfig

def create_time_dimension_plot(
    data_df: pl.DataFrame,
    output_dir: Path,
    plot_config: PlotConfig,
    degree: int = 2
) -> None:
    """Create plot showing time vs dimension relationship"""
    
    # Extract data
    state_sizes = data_df.get_column("state_size").to_numpy()
    times_taken = data_df.get_column("time_taken").to_numpy()
    
    # Create plot
    plt.figure(figsize=plot_config.figsize)
    plt.scatter(state_sizes, times_taken, color="blue", label="Data points")
    
    # Add polynomial fit
    coeffs = np.polyfit(state_sizes, times_taken, degree)
    poly = np.poly1d(coeffs)
    
    # Create smooth curve
    x_smooth = np.linspace(state_sizes.min(), state_sizes.max(), 200)
    y_smooth = poly(x_smooth)
    
    plt.plot(x_smooth, y_smooth, color="red", label=f"Polynomial fit (degree {degree})")
    
    # Generate equation string
    eq_terms = [f"{coeff:.2e}x^{degree - i}" for i, coeff in enumerate(coeffs[:-1])]
    eq_terms.append(f"{coeffs[-1]:.2e}")
    eq_string = "y = " + " + ".join(eq_terms).replace(" + -", " - ")
    
    # Add equation text
    plt.text(
        0.05, 0.95,
        eq_string,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )
    
    # Customize plot
    plt.title("State Size vs. Time Taken for One Iteration of Algorithm 1")
    plt.xlabel(plot_config.xlabel)
    plt.ylabel("Time Taken (seconds)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(bottom=0)
    plt.legend()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "time_vs_dimension.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_algorithm_comparison_plot(
    data_df: pl.DataFrame,
    output_dir: Path,
    plot_config: PlotConfig
) -> None:
    """Create plot comparing different algorithms"""
    
    plt.figure(figsize=plot_config.figsize)
    
    algorithms = ["direct_optimization", "bisection", "random_search", "random_kernel"]
    markers = ["o", "^", "s", "o"]
    colors = ["black", "red", "blue", "green"]
    labels = [
        "Maximize Using SLSQP",
        "Algorithm 1: Binary Search",
        "Maximize using Random Rank 1 Kernel Guess",
        "Maximize using Random Kernel Guess"
    ]
    
    state_sizes = data_df.get_column("state_size").unique().sort()
    
    for alg, marker, color, label in zip(algorithms, markers, colors, labels):
        alg_data = data_df.filter(pl.col("algorithm") == alg)
        plt.plot(
            alg_data.get_column("state_size"),
            alg_data.get_column("penalty"),
            marker=marker,
            label=label,
            color=color,
            alpha=0.7
        )
    
    plt.xlabel(plot_config.xlabel)
    plt.ylabel(plot_config.ylabel)
    plt.title(plot_config.title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "algorithm_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
