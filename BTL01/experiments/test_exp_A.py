"""Kiểm thử ràng buộc (contract tests) cho bộ thực nghiệm quy mô mẫu huấn luyện của Thực nghiệm A."""

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


class ExperimentAContractTests(unittest.TestCase):
    """Kiểm tra tính cố định của lưới tham số Thực nghiệm A và định dạng artifact lưu trữ."""

    @classmethod
    def setUpClass(cls) -> None:
        """Tải cấu hình dự án một lần phục vụ các bài kiểm thử ràng buộc."""
        with (ROOT / "configs/baseline.yaml").open(encoding="utf-8") as stream:
            cls.baseline = yaml.safe_load(stream)
        with (ROOT / "configs/exp_specs.yaml").open(encoding="utf-8") as stream:
            cls.specs = yaml.safe_load(stream)
        cls.runner = ExperimentRunner(cls.baseline, torch.device("cpu"))

    def test_train_size_grid_has_30_trials(self) -> None:
        """Thực nghiệm A chứa 10 mốc kích thước mẫu nhân với 3 hạt giống cố định (30 trials)."""
        trials = self.runner.build_train_size_trials(
            self.specs["experiments"]["exp_A"]
        )
        self.assertEqual(len(trials), 30)
        self.assertEqual(
            sorted({trial["train_size"] for trial in trials}),
            list(range(200, 2001, 200)),
        )
        self.assertEqual(
            sorted({trial["seed"] for trial in trials}), [42, 100, 2026]
        )
        self.assertTrue(all(trial["hidden_dims"] == [10] for trial in trials))

    def test_train_size_suite_writes_rows_and_summary(self) -> None:
        """Thực nghiệm A ghi nhận đầy đủ các dòng chạy thành công và một dòng tóm tắt cho mỗi kích thước mẫu."""
        experiment = {
            "sample_sizes": [200, 400],
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
                runs, summary = self.runner.run_train_size_experiment(
                    train,
                    val,
                    test,
                    experiment,
                    self.baseline["model_baseline"],
                    self.baseline["training_baseline"],
                    output_root=Path(tmpdir),
                )
            manifest_count = len(
                list(Path(tmpdir).glob("tables/manifest_exp_A_*.json"))
            )

        self.assertIsInstance(runs, pd.DataFrame)
        self.assertEqual(len(runs), 2)
        self.assertEqual(set(runs["status"]), {"completed"})
        self.assertEqual(set(summary["train_size"]), {200, 400})
        self.assertEqual(manifest_count, 2)


if __name__ == "__main__":
    unittest.main()
