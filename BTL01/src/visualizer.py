"""Visualization helpers for Gaussian data exploration artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import GaussianDataGenerator


class Visualizer:
    """Create publication-ready figures and tables for T0203.

    Args:
        figures_dir: Directory used for generated figures.
        dpi: Output figure resolution.  The project requires at least 300.
        figure_format: Matplotlib format such as ``png``.

    Raises:
        ValueError: If ``dpi`` is below 300.
    """

    def __init__(
        self,
        figures_dir: Path,
        dpi: int = 300,
        figure_format: str = "png",
    ) -> None:
        """Initialize the visualizer and ensure the output directory exists."""
        if dpi < 300:
            raise ValueError("dpi must be at least 300.")
        self.figures_dir = figures_dir
        self.dpi = dpi
        self.figure_format = figure_format
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def plot_data_distribution(
        self,
        splits: Mapping[str, pd.DataFrame],
        output_name: str = "data_distribution_3d.png",
    ) -> Path:
        """Plot a 3-D scatter view of generated train/val/test data.

        Args:
            splits: Mapping of split names to DataFrames with columns
                ``x1``, ``x2``, and ``y``.
            output_name: File name written under ``figures_dir``.

        Returns:
            Path to the generated PNG file.
        """
        output_path = self.figures_dir / output_name
        fig = plt.figure(figsize=(8.0, 6.0))
        axis = fig.add_subplot(111, projection="3d")
        colors = {
            "train_full": "#1f77b4",
            "val": "#ff7f0e",
            "test": "#2ca02c",
        }
        labels = {
            "train_full": "Train full",
            "val": "Validation",
            "test": "Test",
        }
        for split_name, frame in splits.items():
            axis.scatter(
                frame["x1"],
                frame["x2"],
                frame["y"],
                s=9,
                alpha=0.58,
                color=colors.get(split_name),
                label=labels.get(split_name, split_name),
                depthshade=True,
            )
        axis.set_title("Gaussian 2D Generated Data Distribution")
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.set_zlabel("Density y")
        axis.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_true_surface(
        self,
        generator: GaussianDataGenerator,
        grid_size: tuple[int, int] = (101, 101),
        output_name: str = "true_gaussian_surface_3d.png",
    ) -> Path:
        """Plot the exact Gaussian density surface over the domain.

        Args:
            generator: Configured Gaussian generator.
            grid_size: Number of grid points for ``x1`` and ``x2``.
            output_name: File name written under ``figures_dir``.

        Returns:
            Path to the generated PNG file.
        """
        output_path = self.figures_dir / output_name
        x1 = np.linspace(
            generator.x1_range[0],
            generator.x1_range[1],
            grid_size[0],
        )
        x2 = np.linspace(
            generator.x2_range[0],
            generator.x2_range[1],
            grid_size[1],
        )
        mesh_x1, mesh_x2 = np.meshgrid(x1, x2)
        grid = np.column_stack((mesh_x1.ravel(), mesh_x2.ravel()))
        density = generator.compute_density(grid).reshape(mesh_x1.shape)

        fig = plt.figure(figsize=(8.0, 6.0))
        axis = fig.add_subplot(111, projection="3d")
        surface = axis.plot_surface(
            mesh_x1,
            mesh_x2,
            density,
            cmap="viridis",
            linewidth=0.0,
            antialiased=True,
            alpha=0.92,
        )
        axis.set_title("True 2D Gaussian Density Surface")
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.set_zlabel("Density y")
        fig.colorbar(surface, ax=axis, shrink=0.62, label="Density y")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def build_summary_table(
        self,
        splits: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Build summary statistics for coordinates and labels.

        Args:
            splits: Mapping of split names to DataFrames with columns
                ``x1``, ``x2``, and ``y``.

        Returns:
            Table with split, feature, count, mean, std, min, and max columns.
        """
        rows: list[dict[str, float | int | str]] = []
        for split_name, frame in splits.items():
            for column in ["x1", "x2", "y"]:
                series = frame[column]
                rows.append(
                    {
                        "split": split_name,
                        "feature": column,
                        "count": int(series.count()),
                        "mean": float(series.mean()),
                        "std": float(series.std(ddof=1)),
                        "min": float(series.min()),
                        "max": float(series.max()),
                    }
                )
        return pd.DataFrame(rows)
