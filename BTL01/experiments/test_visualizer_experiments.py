"""Tests for Experiment B/C summary visualizations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.visualizer import Visualizer


class ExperimentVisualizationTests(unittest.TestCase):
    """Ensure experiment plots are generated from persisted summary-shaped data."""

    def test_factor_error_and_train_validation_gap_plots(self) -> None:
        """B plots include error bars and the train-validation gap."""
        summary = pd.DataFrame(
            {
                "width": [1, 10],
                "test_mse_mean": [0.1, 0.01],
                "test_mse_std": [0.02, 0.003],
                "train_mse_mean": [0.08, 0.005],
                "val_mse_mean": [0.12, 0.012],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = Visualizer(Path(tmpdir))
            error_path = visualizer.plot_factor_error(
                summary, "width", "test_mse", "b_error.png", "Hidden width"
            )
            gap_path = visualizer.plot_train_validation_gap(
                summary, "width", "b_gap.png", "Hidden width"
            )

            self.assertTrue(error_path.exists())
            self.assertTrue(gap_path.exists())
            self.assertGreater(error_path.stat().st_size, 0)
            self.assertGreater(gap_path.stat().st_size, 0)

    def test_depth_convergence_time_plot_requires_timing_columns(self) -> None:
        """C plot requires explicit convergence and timing metrics."""
        summary = pd.DataFrame(
            {
                "depth": [1, 2],
                "test_mse_mean": [0.1, 0.05],
                "test_mse_std": [0.02, 0.01],
                "training_time_seconds_mean": [1.0, 2.0],
                "training_time_seconds_std": [0.1, 0.2],
                "best_epoch_mean": [10.0, 20.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = Visualizer(Path(tmpdir))
            path = visualizer.plot_depth_convergence_time(
                summary, "depth_convergence.png"
            )
            self.assertTrue(path.exists())

            missing = summary.drop(columns=["training_time_seconds_mean"])
            with self.assertRaises(ValueError):
                visualizer.plot_depth_convergence_time(missing, "bad.png")


if __name__ == "__main__":
    unittest.main()
