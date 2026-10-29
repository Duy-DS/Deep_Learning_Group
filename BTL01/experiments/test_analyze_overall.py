"""Unit tests for overall A/B/C comparative analysis and visualizations (T0701)."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_overall import build_overall_summary, choose_best_runs
from src.data_generator import GaussianDataGenerator
from src.model import FunctionApproximatorMLP
from src.visualizer import Visualizer


class OverallAnalysisTests(unittest.TestCase):
    """Test best model selection logic, overall table construction, and plotting."""

    def test_choose_best_runs_and_summary_generation(self) -> None:
        """Validate selection by validation loss without data leakage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tables_dir = Path(tmpdir)
            # Tạo dữ liệu giả lập cho runs_exp_A, B, C
            runs_a = pd.DataFrame(
                [
                    {
                        "run_id": "exp_A_train_size1000_seed42",
                        "status": "completed",
                        "train_size": 1000,
                        "seed": 42,
                        "hidden_dims": "[10]",
                        "parameter_count": 41,
                        "train_mse": 0.05,
                        "val_mse": 0.04,
                        "test_mse": 0.045,
                        "test_rmse": 0.21,
                        "test_mae": 0.15,
                        "training_time_seconds": 10.0,
                        "best_epoch": 200,
                    },
                    {
                        "run_id": "exp_A_train_size1400_seed42",
                        "status": "completed",
                        "train_size": 1400,
                        "seed": 42,
                        "hidden_dims": "[10]",
                        "parameter_count": 41,
                        "train_mse": 0.03,
                        "val_mse": 0.02,
                        "test_mse": 0.025,
                        "test_rmse": 0.15,
                        "test_mae": 0.12,
                        "training_time_seconds": 12.0,
                        "best_epoch": 250,
                    },
                ]
            )
            runs_b = pd.DataFrame(
                [
                    {
                        "run_id": "exp_B_hidden_width10_seed42",
                        "status": "completed",
                        "width": 10,
                        "seed": 42,
                        "hidden_dims": "[10]",
                        "parameter_count": 41,
                        "train_mse": 0.04,
                        "val_mse": 0.03,
                        "test_mse": 0.035,
                        "test_rmse": 0.18,
                        "test_mae": 0.14,
                        "training_time_seconds": 11.0,
                        "best_epoch": 220,
                    }
                ]
            )
            runs_c = pd.DataFrame(
                [
                    {
                        "run_id": "exp_C_hidden_depth3_seed42",
                        "status": "completed",
                        "depth": 3,
                        "seed": 42,
                        "hidden_dims": "[10, 10, 10]",
                        "parameter_count": 261,
                        "train_mse": 0.005,
                        "val_mse": 0.003,
                        "test_mse": 0.004,
                        "test_rmse": 0.06,
                        "test_mae": 0.04,
                        "training_time_seconds": 15.0,
                        "best_epoch": 300,
                    }
                ]
            )
            summary_a = pd.DataFrame(
                [
                    {
                        "train_size": 1000,
                        "test_mse_mean": 0.045,
                        "test_mse_std": 0.005,
                        "test_rmse_mean": 0.21,
                        "test_rmse_std": 0.02,
                        "test_mae_mean": 0.15,
                        "test_mae_std": 0.01,
                        "val_mse_mean": 0.04,
                        "n_runs": 1,
                    },
                    {
                        "train_size": 1400,
                        "test_mse_mean": 0.025,
                        "test_mse_std": 0.003,
                        "test_rmse_mean": 0.15,
                        "test_rmse_std": 0.01,
                        "test_mae_mean": 0.12,
                        "test_mae_std": 0.01,
                        "val_mse_mean": 0.02,
                        "n_runs": 1,
                    },
                ]
            )
            summary_b = pd.DataFrame(
                [
                    {
                        "width": 10,
                        "test_mse_mean": 0.035,
                        "test_mse_std": 0.004,
                        "test_rmse_mean": 0.18,
                        "test_rmse_std": 0.02,
                        "test_mae_mean": 0.14,
                        "test_mae_std": 0.01,
                        "val_mse_mean": 0.03,
                        "n_runs": 1,
                    }
                ]
            )
            summary_c = pd.DataFrame(
                [
                    {
                        "depth": 3,
                        "test_mse_mean": 0.004,
                        "test_mse_std": 0.001,
                        "test_rmse_mean": 0.06,
                        "test_rmse_std": 0.01,
                        "test_mae_mean": 0.04,
                        "test_mae_std": 0.005,
                        "val_mse_mean": 0.003,
                        "n_runs": 1,
                    }
                ]
            )

            runs_a.to_csv(tables_dir / "runs_exp_A.csv", index=False)
            runs_b.to_csv(tables_dir / "runs_exp_B.csv", index=False)
            runs_c.to_csv(tables_dir / "runs_exp_C.csv", index=False)
            summary_a.to_csv(tables_dir / "summary_exp_A.csv", index=False)
            summary_b.to_csv(tables_dir / "summary_exp_B.csv", index=False)
            summary_c.to_csv(tables_dir / "summary_exp_C.csv", index=False)

            best_runs = choose_best_runs(tables_dir)
            self.assertEqual(len(best_runs), 4)  # baseline + A + B + C

            overall = build_overall_summary(best_runs, tables_dir)
            self.assertEqual(len(overall), 4)
            self.assertIn("test_mse_mean", overall.columns)
            self.assertIn("val_mse_mean", overall.columns)

    def test_visualizer_new_methods(self) -> None:
        """Test Visualizer 3D comparison, parity scatter, residual, and overall comparison plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            figures_dir = Path(tmpdir)
            visualizer = Visualizer(figures_dir)

            generator = GaussianDataGenerator()
            model_config = {
                "hidden_dims": [10],
                "hidden_depth": 1,
                "activation": "tanh",
                "output_activation": "linear",
                "bias": True,
                "weight_init": "xavier_normal",
                "bias_init": "zeros",
                "hidden_init_gain": 5.0 / 3.0,
                "output_init_gain": 1.0,
            }
            data_config = {"input_dim": 2, "output_dim": 1}
            model = FunctionApproximatorMLP(model_config, data_config)

            # 1. Surface comparison 3D
            surf_path = visualizer.plot_surface_comparison(
                generator,
                model,
                device="cpu",
                grid_size=(21, 21),
                output_name="test_surf.png",
            )
            self.assertTrue(surf_path.exists())
            self.assertGreater(surf_path.stat().st_size, 0)

            # 2. Parity scatter
            y_true = np.array([0.1, 0.2, 0.3, 0.4])
            y_pred = np.array([0.11, 0.19, 0.31, 0.39])
            parity_path = visualizer.plot_parity_scatter(
                y_true, y_pred, output_name="test_parity.png"
            )
            self.assertTrue(parity_path.exists())
            self.assertGreater(parity_path.stat().st_size, 0)

            # 3. Residual surface
            res_path = visualizer.plot_residual_surface(
                generator,
                model,
                device="cpu",
                grid_size=(21, 21),
                output_name="test_res.png",
            )
            self.assertTrue(res_path.exists())
            self.assertGreater(res_path.stat().st_size, 0)

            # 4. Overall comparison bar chart
            summary_overall = pd.DataFrame(
                {
                    "config_name": ["Baseline", "Best A", "Best B", "Best C"],
                    "test_mse_mean": [0.0001, 0.00005, 0.00002, 0.00001],
                    "test_rmse_mean": [0.01, 0.007, 0.004, 0.003],
                    "test_mae_mean": [0.008, 0.005, 0.003, 0.002],
                }
            )
            comp_path = visualizer.plot_overall_comparison(
                summary_overall, output_name="test_comp.png"
            )
            self.assertTrue(comp_path.exists())
            self.assertGreater(comp_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
