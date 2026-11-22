"""Chạy smoke-test logger trong sandbox tạm, không lưu artifact vào repo."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD

# Khi script chạy trực tiếp theo đường dẫn, thêm root BTL03 để nạp package src.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import CheckpointManager, ExperimentLogger
from src.utils import seed_everything


class MLOpsSmokeTest:
    """Mô phỏng ba epoch và xác minh CSV, TensorBoard, checkpoint."""

    def run(self) -> dict[str, str | int]:
        """Ghi ba epoch vào sandbox và trả tóm tắt kết quả kiểm tra."""
        seed_everything(42)
        with tempfile.TemporaryDirectory(prefix="btl03_mlops_smoke_") as temporary:
            root = Path(temporary)
            model = nn.Linear(4, 2)
            optimizer = SGD(model.parameters(), lr=0.01)
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
                project_root=root,
            )
            try:
                for epoch in range(1, 4):
                    logger.log_epoch(
                        epoch=epoch,
                        train_loss=1.0 / epoch,
                        val_loss=1.1 / epoch,
                        train_metric=0.5 + epoch * 0.1,
                        val_metric=0.4 + epoch * 0.1,
                        learning_rate=0.01,
                        epoch_time_sec=0.02,
                        metric_name="Accuracy",
                    )
                logger.finish_run(
                    best_epoch=3,
                    train_metric=0.8,
                    val_metric=0.7,
                    test_metric=0.65,
                    train_time_sec=0.06,
                )
                event_files = list(logger.run_dir.glob("events.out.tfevents*"))
                assert event_files, "TensorBoard event file không được tạo."
                master_path = logger.master_csv_path
                epoch_path = logger.epoch_csv_path
                run_dir = logger.run_dir
            finally:
                logger.close()

            with master_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                master_rows = list(reader)
                assert tuple(reader.fieldnames or ()) == ExperimentLogger.MASTER_COLUMNS
            assert len(ExperimentLogger.MASTER_COLUMNS) == 18
            assert len(master_rows) == 1 and master_rows[0]["Exp_Type"] == "SmokeTest"
            assert master_rows[0]["Member_Alias"] == "SmokeTest"

            with epoch_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                epoch_rows = list(reader)
                assert tuple(reader.fieldnames or ()) == ExperimentLogger.EPOCH_COLUMNS
            assert len(epoch_rows) == 3

            checkpoint_manager = CheckpointManager(
                project_root=root,
                checkpoint_root=root / "outputs" / "checkpoints",
            )
            checkpoint_path = checkpoint_manager.save(
                model=model,
                model_group="Group0_SmokeTest",
                model_name="LinearSmoke",
                epoch=3,
                optimizer=optimizer,
                best_metric=0.7,
                seed=42,
                config={"exp_type": "SmokeTest", "lr": 0.01},
            )
            restored = nn.Linear(4, 2)
            metadata = checkpoint_manager.load(checkpoint_path, model=restored)
            assert metadata["epoch"] == 3 and metadata["seed"] == 42
            assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
            assert checkpoint_path.is_file()
            assert run_dir.relative_to(root).as_posix().startswith(
                "experiments/tensorboard/Group0_SmokeTest/LinearSmoke/SmokeTest_s42_"
            )

            result = {
                "epochs": len(epoch_rows),
                "master_columns": len(ExperimentLogger.MASTER_COLUMNS),
                "tensorboard_events": len(event_files),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
            }
        print(
            "MLOps SmokeTest PASS (sandbox tự xóa): "
            f"epochs={result['epochs']}, master_columns={result['master_columns']}, "
            f"events={result['tensorboard_events']}, checkpoint_bytes={result['checkpoint_bytes']}"
        )
        return result


def main() -> None:
    """Điểm vào chạy smoke-test độc lập."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    MLOpsSmokeTest().run()


if __name__ == "__main__":
    main()