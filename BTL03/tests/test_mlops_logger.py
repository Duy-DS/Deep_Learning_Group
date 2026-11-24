"""Kiểm thử schema CSV, TensorBoard và checkpoint manager BTL03."""

import csv
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD

from src.logger import CheckpointManager, ExperimentLogger


def test_logger_writes_exact_schemas_and_tensorboard_event(tmp_path):
    """Logger tạo CSV 18 cột, epoch rows và event file theo cây chuẩn."""
    logger = ExperimentLogger(
        model_group="Group0_SmokeTest",
        model_name="LinearSmoke",
        exp_type="SmokeTest",
        seed=42,
        optimizer="SGD",
        learning_rate=0.01,
        batch_size=4,
        epochs=3,
        member_alias="SmokeTest",
        project_root=tmp_path,
    )
    try:
        for epoch in range(1, 4):
            logger.log_epoch(
                epoch=epoch,
                train_loss=1.0 / epoch,
                val_loss=1.2 / epoch,
                train_metric=0.5 + epoch / 10,
                val_metric=0.4 + epoch / 10,
                learning_rate=0.01,
                epoch_time_sec=0.125,
                metric_name="Accuracy",
            )
        master = logger.finish_run(
            best_epoch=3,
            train_metric=0.8,
            val_metric=0.7,
            test_metric=0.65,
            train_time_sec=0.375,
        )
        event_files = list(logger.run_dir.glob("events.out.tfevents*"))
        assert event_files, "SummaryWriter chưa tạo event file."
        assert logger.run_dir.relative_to(tmp_path).as_posix().startswith(
            "experiments/tensorboard/Group0_SmokeTest/LinearSmoke/SmokeTest_s42_"
        )
    finally:
        logger.close()

    with master.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert tuple(reader.fieldnames or ()) == ExperimentLogger.MASTER_COLUMNS
    assert len(ExperimentLogger.MASTER_COLUMNS) == 18
    assert len(rows) == 1
    assert rows[0]["Exp_Type"] == "SmokeTest"
    assert rows[0]["Seed"] == "42"
    assert rows[0]["Member_Alias"] == "SmokeTest"
    assert rows[0]["Best_Epoch"] == "3"

    with logger.epoch_csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        epoch_rows = list(reader)
        assert tuple(reader.fieldnames or ()) == ExperimentLogger.EPOCH_COLUMNS
    assert len(epoch_rows) == 3
    assert epoch_rows[-1]["Epoch"] == "3"
    assert epoch_rows[-1]["Epoch_Time_Sec"] == "0.125000"


def test_logger_requires_human_alias_and_rejects_duplicate_epoch(tmp_path):
    """Không tự gán tác giả mặc định và không cho ghi trùng epoch."""
    with pytest.raises(ValueError, match="Member_Alias"):
        ExperimentLogger(
            model_group="Group0",
            model_name="Model",
            exp_type="SmokeTest",
            project_root=tmp_path,
        )

    logger = ExperimentLogger(
        model_group="Group0",
        model_name="Model",
        exp_type="SmokeTest",
        member_alias="TestMember",
        project_root=tmp_path,
    )
    try:
        logger.log_epoch(1, 1.0, 1.0)
        with pytest.raises(ValueError, match="tăng nghiêm ngặt"):
            logger.log_epoch(1, 1.0, 1.0)
    finally:
        logger.close()


def test_finish_requires_explicit_best_epoch(tmp_path):
    """Master CSV không được gán Best_Epoch suy diễn từ epoch cuối."""
    logger = ExperimentLogger(
        model_group="Group0",
        model_name="Model",
        exp_type="SmokeTest",
        member_alias="TestMember",
        project_root=tmp_path,
    )
    try:
        logger.log_epoch(1, 0.5, 0.4)
        with pytest.raises(ValueError, match="best_epoch"):
            logger.finish_run()
    finally:
        logger.close()


def test_checkpoint_round_trip_and_drive_mirror(tmp_path):
    """Checkpoint lưu metadata, khôi phục state và mirror Drive giả lập."""
    drive_root = tmp_path / "mounted_drive" / "MyDrive" / "BTL03"
    drive_root.mkdir(parents=True)
    model = nn.Linear(3, 2)
    optimizer = SGD(model.parameters(), lr=0.02)
    manager = CheckpointManager(
        project_root=tmp_path,
        checkpoint_root=tmp_path / "outputs" / "checkpoints",
        drive_root=drive_root,
    )
    path = manager.save(
        model=model,
        model_group="01_mlp",
        model_name="SmokeModel",
        epoch=3,
        optimizer=optimizer,
        best_metric=0.75,
        seed=42,
        config={"exp_type": "SmokeTest", "learning_rate": 0.02},
    )
    mirrored = drive_root / "checkpoints" / "01_mlp" / path.name
    assert path.is_file()
    assert mirrored.is_file()

    restored = nn.Linear(3, 2)
    payload = manager.load(path, model=restored, map_location="cpu")
    assert payload["epoch"] == 3
    assert payload["best_metric"] == pytest.approx(0.75)
    assert payload["seed"] == 42
    assert payload["config"]["exp_type"] == "SmokeTest"
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)


def test_checkpoint_discovers_configured_mounted_drive(tmp_path):
    """Logger tự nhận mount root từ YAML khi đường dẫn cấu hình đã tồn tại."""
    import yaml

    drive_root = tmp_path / "mounted" / "MyDrive" / "BTL03"
    drive_root.mkdir(parents=True)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "paths.yaml").write_text(
        yaml.safe_dump({"project": {"drive_mount_root": str(drive_root)}}),
        encoding="utf-8",
    )
    manager = CheckpointManager(project_root=tmp_path)
    model = nn.Linear(2, 1)
    path = manager.save(model, "02_cnn_mlp", "DriveDiscovery", epoch=1)
    mirror = drive_root / "checkpoints" / "02_cnn_mlp" / path.name
    assert manager.drive_root == drive_root.resolve()
    assert mirror.is_file()
