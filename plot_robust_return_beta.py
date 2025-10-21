from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from create_plot_from_data import (
    create_figure,
    save_and_close_plot,
    set_plot_style,
    setup_plot_basics,
)


def load_robust_return_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load robust return vs beta data from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame containing the robust return data
    """
    return pd.read_csv(file_path)


def plot_robust_return_vs_beta(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a plot of robust return vs beta with zoom inset for close values.

    Args:
        df: DataFrame containing robust return data
        output_path: Path to save the plot
    """
    # Create figure
    create_figure()

    # Get the main axes
    ax = plt.gca()

    # Plot the three lines
    plt.plot(
        df["beta"],
        df["nominal_return"],
        linestyle=":",
        linewidth=2,
        color="#FF0000",
        label="Nominal Return",
    )

    plt.plot(
        df["beta"],
        df["eigen_bisection_robust_return"],
        linewidth=2.5,
        label="Algorithm 1 (Ours)",
    )

    plt.plot(
        df["beta"],
        df["randon_kernel_robust_return"],
        linewidth=2.5,
        label="Random Kernel",
    )

    # Set x-axis to logarithmic scale
    plt.xscale("log", base=10)

    # Setup plot basics
    setup_plot_basics(
        xlabel="Uncertainty Radius $\\beta$ (Log Scale)",
        ylabel="Robust Return",
        title="Robust Return vs Beta",
    )

    # Create inset for zoom (beta range 0.0001 to 0.01)
    # Position the inset in the lower left area
    axins = inset_axes(
        ax, width="40%", height="40%", bbox_to_anchor=(-0.45, -0.45, 1, 1), bbox_transform=ax.transAxes
    )

    # Filter data for the zoom range
    zoom_data = df[(df["beta"] >= 0.0001) & (df["beta"] <= 0.01)]

    # Plot the same lines in the inset
    axins.plot(
        zoom_data["beta"],
        zoom_data["nominal_return"],
        linestyle=":",
        linewidth=2,
        color="#FF0000",
    )

    axins.plot(
        zoom_data["beta"],
        zoom_data["eigen_bisection_robust_return"],
        linewidth=2.5,
    )

    axins.plot(
        zoom_data["beta"],
        zoom_data["randon_kernel_robust_return"],
        linewidth=2.5,
    )

    # Set inset properties
    axins.set_xscale("log", base=10)
    axins.set_xlim(0.0001, 0.01)

    # Set y-limits for the inset to show the close values clearly
    y_min = (
        zoom_data[["nominal_return", "eigen_bisection_robust_return", "randon_kernel_robust_return"]]
        .min()
        .min()
    )
    y_max = (
        zoom_data[["nominal_return", "eigen_bisection_robust_return", "randon_kernel_robust_return"]]
        .max()
        .max()
    )
    y_range = y_max - y_min
    axins.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Add grid to inset for better readability
    axins.grid(True, alpha=0.3)

    # Add labels to inset
    axins.set_xlabel("Uncertainty Radius $\\beta$ (Log Scale)", fontsize=10)
    axins.set_ylabel("Robust Return", fontsize=10)
    axins.tick_params(labelsize=8)

    # Add a border around the inset
    axins.spines["top"].set_visible(True)
    axins.spines["right"].set_visible(True)
    axins.spines["bottom"].set_visible(True)
    axins.spines["left"].set_visible(True)

    # Indicate the zoomed region with lines connecting to the inset
    from matplotlib.patches import ConnectionPatch

    # Create connection lines from the main plot to the inset
    # Bottom left connection
    con1 = ConnectionPatch(
        xyA=(0.0001, y_min),
        coordsA=ax.transData,
        xyB=(0, 0),
        coordsB=axins.transAxes,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    # Bottom right connection
    con2 = ConnectionPatch(
        xyA=(0.01, y_min),
        coordsA=ax.transData,
        xyB=(1, 0),
        coordsB=axins.transAxes,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    ax.add_artist(con1)
    ax.add_artist(con2)

    # Save plot
    save_and_close_plot(output_path)


def main() -> None:
    """Main function to drive the data loading and plot generation."""
    # Set plot style
    set_plot_style()

    # Define paths
    data_path = Path("robust_return_vs_beta.csv")
    output_path = Path("plots/robust_return_vs_beta.png")

    # Load data
    df = load_robust_return_data(data_path)

    # Create plot
    plot_robust_return_vs_beta(df, output_path)

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
