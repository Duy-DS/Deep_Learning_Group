"""Kiểm thử ràng buộc (contract tests) cho bộ thực nghiệm chiều rộng tầng ẩn của Thực nghiệm B."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from src.data_generator import GaussianDataset
import yaml

from src.runner import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]


class ExperimentBContractTests(unittest.TestCase):
    """Kiểm tra lưới tham số cố định của Thực nghiệm B trước khi chạy các thử nghiệm tốn kém tài nguyên."""

    @classmethod
    def setUpClass(cls) -> None:
        """Tải cấu hình baseline chuẩn mực và đặc tả Thực nghiệm B."""
        with (ROOT / "configs/baseline.yaml").open(encoding="utf-8") as stream:
            cls.baseline = yaml.safe_load(stream)
        with (ROOT / "configs/exp_specs.yaml").open(encoding="utf-8") as stream:
            cls.specs = yaml.safe_load(stream)
        cls.runner = ExperimentRunner(cls.baseline, torch.device("cpu"))

    def test_width_grid_has_39_trials_and_declared_seeds(self) -> None:
        """Lưới tham số B chứa 13 giá trị chiều rộng nhân với 3 hạt giống cố định (39 trials)."""
        trials = self.runner.build_width_trials(self.specs["experiments"]["exp_B"])

        self.assertEqual(len(trials), 39)
        self.assertEqual(
            sorted({trial["hidden_dims"][0] for trial in trials}),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64],
        )
        self.assertEqual(
            sorted({trial["seed"] for trial in trials}), [42, 100, 2026]
        )

    def test_width_trial_resolves_baseline_train_configuration(self) -> None:
        """Mỗi thử nghiệm B chỉ thay đổi chiều rộng nơ-ron và giữ nguyên kích thước tập train của B/C."""
        trials = self.runner.build_width_trials(self.specs["experiments"]["exp_B"])
        trial = next(item for item in trials if item["width"] == 16 and item["seed"] == 100)

        self.assertEqual(trial["hidden_dims"], [16])
        self.assertEqual(trial["train_size"], 1000)
        self.assertEqual(trial["seed"], 100)

    def test_width_suite_writes_per_run_and_summary_tables(self) -> None:
        """Bộ thực nghiệm lưu trữ chi tiết từng lượt chạy và chỉ tổng hợp kết quả của các lượt chạy thành công."""
        experiment = {
            "train_size": 1,
            "neuron_widths": [1],
            "extended_widths": [2],
            "multi_seeds": [42],
        }
        train = GaussianDataset(ROOT / "data/train_full.csv")
        val = GaussianDataset(ROOT / "data/val.csv")
        test = GaussianDataset(ROOT / "data/test.csv")
        fake_result = {
            "best_epoch": 2,
            "epochs_run": 2,
            "optimizer_steps": 2,
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
            "history": {"train_loss": [0.2, 0.1], "val_loss": [0.3, 0.2]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                self.runner, "run_single_trial", return_value=fake_result
            ):
                runs, summary = self.runner.run_width_experiment(
                    train,
                    val,
                    test,
                    experiment,
                    self.baseline["model_baseline"],
                    self.baseline["training_baseline"],
                    output_root=Path(tmpdir),
                )
            manifest_count = len(
                list(Path(tmpdir).glob("tables/manifest_exp_B_*.json"))
            )

        self.assertIsInstance(runs, pd.DataFrame)
        self.assertEqual(len(runs), 2)
        self.assertEqual(set(runs["status"]), {"completed"})
        self.assertEqual(set(summary["width"]), {1, 2})
        self.assertEqual(manifest_count, 2)

    def test_width_suite_forwards_epoch_override(self) -> None:
        """Chế độ smoke-test cho phép thay đổi số epoch mà không làm biến đổi cấu hình gốc trong baseline."""
        experiment = {
            "train_size": 1,
            "neuron_widths": [1],
            "extended_widths": [],
            "multi_seeds": [42],
        }
        train = GaussianDataset(ROOT / "data/train_full.csv")
        val = GaussianDataset(ROOT / "data/val.csv")
        test = GaussianDataset(ROOT / "data/test.csv")
        observed: list[int] = []

        def fake_trial(*args, **kwargs):
            observed.append(args[4]["epochs"])
            return {
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
            with patch.object(self.runner, "run_single_trial", side_effect=fake_trial):
                self.runner.run_width_experiment(
                    train,
                    val,
                    test,
                    experiment,
                    self.baseline["model_baseline"],
                    self.baseline["training_baseline"],
                    output_root=Path(tmpdir),
                    epochs=3,
                )

        self.assertEqual(observed, [3])
        self.assertEqual(self.baseline["training_baseline"]["epochs"], 1000)


if __name__ == "__main__":
    unittest.main()
