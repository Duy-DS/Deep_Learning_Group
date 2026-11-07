"""Kiểm thử ràng buộc (contract tests) cho bộ thực nghiệm độ sâu mạng của Thực nghiệm C."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import yaml

from src.data_generator import GaussianDataset
from src.runner import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]


class ExperimentCContractTests(unittest.TestCase):
    """Kiểm tra lưới độ sâu cố định chiều rộng và định dạng artifact lưu trữ của Thực nghiệm C."""

    @classmethod
    def setUpClass(cls) -> None:
        """Tải cấu hình baseline chuẩn mực và đặc tả Thực nghiệm C."""
        with (ROOT / "configs/baseline.yaml").open(encoding="utf-8") as stream:
            cls.baseline = yaml.safe_load(stream)
        with (ROOT / "configs/exp_specs.yaml").open(encoding="utf-8") as stream:
            cls.specs = yaml.safe_load(stream)
        cls.runner = ExperimentRunner(cls.baseline, torch.device("cpu"))

    def test_depth_grid_has_15_trials_and_declared_parameter_counts(self) -> None:
        """Lưới tham số C chứa 5 mốc độ sâu nhân với 3 hạt giống cố định (15 trials)."""
        trials = self.runner.build_depth_trials(self.specs["experiments"]["exp_C"])

        self.assertEqual(len(trials), 15)
        self.assertEqual(
            [trial["hidden_dims"] for trial in trials if trial["seed"] == 42],
            [[10], [10, 10], [10, 10, 10], [10, 10, 10, 10], [10] * 5],
        )
        self.assertEqual(
            [trial["parameter_count"] for trial in trials if trial["seed"] == 42],
            [41, 151, 261, 371, 481],
        )

    def test_depth_suite_writes_rows_and_summary(self) -> None:
        """Bộ thực nghiệm C trả về các dòng kết quả hoàn thành và một dòng tóm tắt cho mỗi độ sâu."""
        experiment = {
            "train_size": 1,
            "depths": [1, 2],
            "width_per_layer": 10,
            "multi_seeds": [42],
        }
        train = GaussianDataset(ROOT / "data/train_full.csv")
        val = GaussianDataset(ROOT / "data/val.csv")
        test = GaussianDataset(ROOT / "data/test.csv")
        fake_result = {
            "best_epoch": 1,
            "epochs_run": 1,
            "optimizer_steps": 1,
            "training_time_seconds": 0.01,
            "train_mse": 0.1,
            "train_rmse": 0.316,
            "train_mae": 0.2,
            "val_mse": 0.2,
            "val_rmse": 0.447,
            "val_mae": 0.3,
            "test_mse": 0.3,
            "test_rmse": 0.548,
            "test_mae": 0.4,
            "history": {"train_loss": [0.2], "val_loss": [0.3]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                self.runner, "run_single_trial", return_value=fake_result
            ):
                runs, summary = self.runner.run_depth_experiment(
                    train,
                    val,
                    test,
                    experiment,
                    self.baseline["model_baseline"],
                    self.baseline["training_baseline"],
                    output_root=Path(tmpdir),
                )

        self.assertIsInstance(runs, pd.DataFrame)
        self.assertEqual(len(runs), 2)
        self.assertEqual(set(summary["depth"]), {1, 2})


if __name__ == "__main__":
    unittest.main()
