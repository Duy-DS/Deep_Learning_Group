"""Kiểm thử trình thực thi thực nghiệm T0304 và các hàm đánh giá.

Chạy từ thư mục gốc: python -m scripts.test_runner
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sys

# Đảm bảo import được các module từ thư mục gốc
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluator import RegressionEvaluator
from src.runner import ExperimentRunner


class DummyMLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class TestT0304(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.evaluator = RegressionEvaluator(self.device)
        self.config = {
            "data": {"input_dim": 2, "output_dim": 1},
            "reproducibility": {
                "deterministic": True,
                "deterministic_warn_only": False,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
            },
            "evaluation": {"synchronize_accelerator": False},
        }
        self.runner = ExperimentRunner(self.config, self.device)

        # Dữ liệu giả lập
        x = torch.ones(4, 2)
        y = torch.zeros(4, 1)
        self.loader = DataLoader(TensorDataset(x, y), batch_size=2)

    def test_evaluator_metrics(self):
        """Kiểm tra Evaluator tính đúng MSE, RMSE, MAE bằng PyTorch."""
        model = DummyMLP()
        nn.init.constant_(model.linear.weight, 0.0)
        nn.init.constant_(model.linear.bias, 2.0)

        # Dự đoán luôn là 2.0, nhãn là 0.0 -> Error = 2.0
        # MSE = 4.0, RMSE = 2.0, MAE = 2.0
        metrics = self.evaluator.evaluate(model, self.loader)

        self.assertAlmostEqual(metrics["mse"], 4.0)
        self.assertAlmostEqual(metrics["rmse"], 2.0)
        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertFalse(model.training)

    def test_runner_sets_seeds(self):
        """Kiểm tra Runner thiết lập seed thành công."""
        self.runner.set_seed(42)
        val1 = torch.rand(1).item()

        self.runner.set_seed(42)
        val2 = torch.rand(1).item()

        self.assertEqual(val1, val2)

    @patch("src.runner.FunctionApproximatorMLP", DummyMLP)
    def test_run_single_trial(self):
        """Kiểm tra vòng đời của một Trial có trả về đúng cấu trúc từ điển không."""
        train_config = {
            "epochs": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "sgd_momentum": 0.0,
            "sgd_nesterov": False,
            "early_stopping": {
                "enabled": False,
                "patience": 5,
                "min_delta": 0.0,
                "mode": "min",
            },
            "loss_function": "mse",
            "loss_reduction": "mean",
            "checkpoint": {"restore_best": True},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_best.pt"

            result = self.runner.run_single_trial(
                train_loader=self.loader,
                val_loader=self.loader,
                test_loader=self.loader,
                model_config={},
                train_config=train_config,
                seed=42,
                checkpoint_path=ckpt_path,
            )

            self.assertEqual(result["seed"], 42)
            self.assertIn("train_mse", result)
            self.assertIn("val_rmse", result)
            self.assertIn("test_mae", result)
            self.assertIn("training_time_seconds", result)
            self.assertIn("epochs_run", result)
            self.assertIn("optimizer_steps", result)
            self.assertTrue(ckpt_path.exists())


if __name__ == "__main__":
    unittest.main()
