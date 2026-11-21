"""Kiểm thử tính tái lập và schema xuất MLflow."""

import random
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.train.mlflow_logger import MLflowLogger
from src.train.trainer import Trainer
from src.train.utils import seed_everything


def test_seed_everything_sets_determinism_and_repeats():
    """Xác nhận deterministic flags và chuỗi số ngẫu nhiên tái lập."""
    seed_everything(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    first = (random.random(), float(np.random.rand()), torch.rand(1).item())

    seed_everything(42)
    second = (random.random(), float(np.random.rand()), torch.rand(1).item())

    assert first == second


def test_mlflow_csv_has_exact_ordered_schema(tmp_path):
    """Xuất dữ liệu run giả lập với đúng 24 cột và thứ tự quy định."""
    logger = MLflowLogger.__new__(MLflowLogger)
    logger.experiment_name = "test"
    fake_runs = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "start_time": pd.to_datetime(["2026-11-17T10:00:00Z"]),
            "end_time": pd.to_datetime(["2026-11-17T10:01:00Z"]),
            "status": ["FINISHED"],
            "tags.mlflow.runName": ["test-run"],
            "params.architecture": ["LargeCNN"],
            "metrics.macro_f1": [0.8],
            "extra.unwanted": ["discard"],
        }
    )
    output_path = tmp_path / "summary.csv"

    with patch("src.train.mlflow_logger.mlflow.search_runs", return_value=fake_runs):
        logger.export_summary_csv(str(output_path))

    exported = pd.read_csv(output_path, keep_default_na=False)
    assert exported.columns.tolist() == MLflowLogger.CSV_COLUMNS
    assert len(exported.columns) == 24
    assert exported.loc[0, "Run ID"] == "run-1"
    assert exported.loc[0, "params.architecture"] == "LargeCNN"
    assert exported.loc[0, "metrics.macro_f1"] == 0.8
    assert exported.loc[0, "Duration"] == 60
    assert exported.loc[0, "Source Name"] == "None"
    assert exported.loc[0, "metrics.train_loss"] == "None"
    assert "extra.unwanted" not in exported.columns


def test_trainer_builds_configured_schedulers():
    """Kiểm tra hai scheduler được tạo từ YAML và None không tạo scheduler."""
    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.lr = 0.01
    trainer.weight_decay = 0.0
    trainer.epochs = 5

    trainer.config = {"optimizer": "Adam", "scheduler": "CosineAnnealingLR"}
    trainer.optimizer = trainer._build_optimizer()
    cosine = trainer._build_scheduler()
    assert isinstance(cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
    trainer.optimizer.step()
    cosine.step()

    trainer.config = {"optimizer": "Adam", "scheduler": "ReduceLROnPlateau"}
    trainer.optimizer = trainer._build_optimizer()
    plateau = trainer._build_scheduler()
    assert isinstance(plateau, torch.optim.lr_scheduler.ReduceLROnPlateau)
    trainer.optimizer.step()
    plateau.step(1.0)

    trainer.config = {"optimizer": "Adam", "scheduler": "None"}
    trainer.optimizer = trainer._build_optimizer()
    assert trainer._build_scheduler() is None


def test_trainer_formats_activation_artifact_paths():
    """Xác nhận checkpoint/CSV có tên riêng theo activation trong ablation."""
    path = Trainer._format_artifact_path(
        "/content/drive/MyDrive/BTL02/T0204/checkpoints/large_{activation}.pth",
        "leaky_relu",
    )
    assert str(path).endswith("large_leaky_relu.pth")


def test_trainer_optimizer_uses_configured_weight_decay():
    """Xác nhận Adam nhận đúng weight decay từ YAML ablation."""
    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.lr = 0.001
    trainer.weight_decay = 0.0125
    trainer.config = {"optimizer": "adam"}
    optimizer = trainer._build_optimizer()
    assert optimizer.param_groups[0]["weight_decay"] == 0.0125


def test_empty_mlflow_export_keeps_schema(tmp_path):
    """Experiment chưa có run vẫn phải xuất header đúng hợp đồng 24 cột."""
    logger = MLflowLogger.__new__(MLflowLogger)
    logger.experiment_name = "empty-test"
    with patch("src.train.mlflow_logger.mlflow.search_runs", return_value=pd.DataFrame()):
        logger.export_summary_csv(str(tmp_path / "empty.csv"))
    result = pd.read_csv(tmp_path / "empty.csv")
    assert result.empty
    assert result.columns.tolist() == MLflowLogger.CSV_COLUMNS


def test_mlflow_logger_uses_tracking_uri_environment_variable(monkeypatch):
    """Xác nhận logger dùng URI Drive đã cấu hình trong môi trường."""
    requested_uri = "file:///content/drive/MyDrive/BTL02/T0204/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", requested_uri)
    with patch("src.train.mlflow_logger.mlflow.set_tracking_uri") as set_uri, patch(
        "src.train.mlflow_logger.mlflow.set_experiment"
    ):
        MLflowLogger()
    set_uri.assert_called_once_with(requested_uri)