"""Kiểm thử tính tái lập và schema xuất MLflow."""

import random
import ast
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import mlflow

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


def test_mlflow_csv_sanitizes_numpy_scalars_and_arrays(tmp_path):
    """CSV exporter normalizes NumPy scalar/array values before rebuilding a frame."""
    logger = MLflowLogger.__new__(MLflowLogger)
    logger.experiment_name = "numpy-export-test"
    fake_runs = pd.DataFrame(
        {
            "run_id": ["run-numpy"],
            "status": ["FINISHED"],
            "tags.mlflow.runName": ["numpy-run"],
            "params.seed": [np.int64(42)],
            "metrics.train_acc": [np.float64(91.5)],
            "tags.member": [np.bool_(True)],
            "metrics.extra": [np.array([1.0, 2.0])],
        }
    )
    output_path = tmp_path / "numpy-summary.csv"

    with patch("src.train.mlflow_logger.mlflow.search_runs", return_value=fake_runs), patch(
        "src.train.mlflow_logger.mlflow.set_tracking_uri"
    ), patch("src.train.mlflow_logger.mlflow.set_experiment"), patch(
        "src.train.mlflow_logger.mlflow.get_tracking_uri", return_value="file:///tmp/mlruns"
    ):
        logger.export_summary_csv(str(output_path))

    exported = pd.read_csv(output_path, keep_default_na=False)
    assert exported.columns.tolist() == MLflowLogger.CSV_COLUMNS
    assert exported.loc[0, "params.seed"] == 42
    assert exported.loc[0, "metrics.train_acc"] == 91.5
    assert exported.loc[0, "tags.member"] in ("True", True)
    assert exported.loc[0, "metrics.train_loss"] == "None"


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
    optimizer_path = Trainer._format_artifact_path(
        "outputs/checkpoints/exp08_{optimizer}.pth", "relu", "adamw"
    )
    assert optimizer_path.as_posix().endswith("exp08_adamw.pth")


def test_trainer_optimizer_uses_configured_weight_decay():
    """Xác nhận Adam nhận đúng weight decay từ YAML ablation."""
    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.lr = 0.001
    trainer.weight_decay = 0.0125
    trainer.config = {"optimizer": "adam"}
    optimizer = trainer._build_optimizer()
    assert optimizer.param_groups[0]["weight_decay"] == 0.0125


def test_trainer_builds_exp08_optimizers_from_yaml_and_runs_one_step():
    """Đọc ba YAML EXP-08, kiểm tra optimizer và chạy một bước giả lập."""
    from pathlib import Path

    import yaml

    project_root = Path(__file__).resolve().parents[1]
    config_paths = (
        project_root / "experiments/ablation/exp08_opt_adam.yaml",
        project_root / "experiments/ablation/exp08_opt_adamw.yaml",
        project_root / "experiments/ablation/exp08_opt_sgd_nesterov.yaml",
    )
    expected_types = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        name = config["optimizer"]
        trainer = Trainer.__new__(Trainer)
        trainer.model = torch.nn.Linear(4, 3)
        trainer.lr = float(config["lr"])
        trainer.weight_decay = float(config["weight_decay"])
        trainer.config = config
        optimizer = trainer._build_optimizer()
        assert isinstance(optimizer, expected_types[name])
        assert optimizer.param_groups[0]["lr"] == float(config["lr"])
        assert optimizer.param_groups[0]["weight_decay"] == float(config["weight_decay"])
        if name == "sgd":
            assert optimizer.param_groups[0]["momentum"] == float(config["momentum"])
            assert optimizer.param_groups[0]["nesterov"] is True
        optimizer.zero_grad()
        loss = trainer.model(torch.ones(2, 4)).square().mean()
        loss.backward()
        optimizer.step()


def test_sgd_rejects_nesterov_without_positive_momentum():
    """Nesterov không được bật nếu momentum không dương."""
    trainer = Trainer.__new__(Trainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.lr = 0.01
    trainer.weight_decay = 0.0
    trainer.config = {"optimizer": "sgd", "momentum": 0.0, "nesterov": True}
    with pytest.raises(ValueError, match="Nesterov yêu cầu momentum"):
        trainer._build_optimizer()


def test_exp08_configs_share_controlled_architecture_and_data_settings():
    """Ba cấu hình EXP-08 giữ nguyên kiến trúc, dữ liệu và lịch huấn luyện."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    files = (
        "exp08_opt_adam.yaml",
        "exp08_opt_adamw.yaml",
        "exp08_opt_sgd_nesterov.yaml",
    )
    configs = [
        yaml.safe_load((root / "experiments" / "ablation" / name).read_text(encoding="utf-8"))
        for name in files
    ]
    fixed_fields = (
        "exp_id", "exp_group", "rq_tag", "architecture", "activation", "channels",
        "seed", "image_size", "in_channels", "split_counts", "augmentation",
        "scheduler", "scheduler_params", "batch_size", "epochs", "patience", "dropout",
    )
    for field in fixed_fields:
        assert len({str(config[field]) for config in configs}) == 1, field
    assert [config["optimizer"] for config in configs] == ["adam", "adamw", "sgd"]
    assert configs[2]["momentum"] == 0.9
    assert configs[2]["nesterov"] is True


def test_t0205_colab_notebook_is_valid_json_with_parseable_cells():
    """Kiểm tra JSON notebook và cú pháp Python của các cell thực thi."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks" / "run_t0205_colab.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 6
    for index, cell in enumerate(code_cells, start=1):
        ast.parse("".join(cell["source"]), filename=f"colab-cell-{index}")


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
    ), patch.object(MLflowLogger, "_ensure_file_store_trash_directory"):
        MLflowLogger()
    set_uri.assert_called_once_with(requested_uri)


def test_mlflow_logger_recreates_missing_file_store_trash_directory(tmp_path, monkeypatch):
    """FileStore vẫn ghi được metric nếu mount làm mất thư mục .trash."""
    tracking_root = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_root.as_uri())
    logger = MLflowLogger(experiment_name="trash-directory-regression")
    assert (tracking_root / ".trash").is_dir()

    logger.start_run("missing-trash-regression")
    (tracking_root / ".trash").rmdir()
    logger.log_metrics({"epoch_loss": 1.25}, step=1)

    assert (tracking_root / ".trash").is_dir()
    try:
        from mlflow.tracking import MlflowClient

        run_id = mlflow.active_run().info.run_id
        history = MlflowClient(tracking_uri=tracking_root.as_uri()).get_metric_history(run_id, "epoch_loss")
        assert len(history) == 1
        assert history[0].value == 1.25
    finally:
        logger.end_run()


def test_mlflow_logger_creates_local_file_store_root_and_trash(tmp_path):
    """Logger tạo đồng thời tracking root và trash cho local file URI."""
    tracking_root = tmp_path / "nested" / "mlruns"
    logger = MLflowLogger(
        experiment_name="new-local-store-regression",
        tracking_uri=tracking_root.as_uri(),
    )
    assert tracking_root.is_dir()
    assert (tracking_root / ".trash").is_dir()