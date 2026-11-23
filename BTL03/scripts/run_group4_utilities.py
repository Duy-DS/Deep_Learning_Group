"""Huấn luyện Gender/Emotion CNN với callbacks, TensorBoard và CSV chuẩn."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_group2_cnn import LegacyFERAugmentationDataset
from src.datasets import FEREmotionDataset, WikiGenderDataset
from src.engine import BaseTrainer, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.models import EmotionCNN, HumanGenderCNN
from src.utils import get_torch_generator, seed_everything, seed_worker


class Group4UtilitiesRunner:
    """Nối callbacks, CNN, dataset, logger và biểu đồ hội tụ của WP utilities."""

    CONFIG_FILES = {"gender": "utilities_gender.yaml", "emotion": "utilities_emotion.yaml"}

    def __init__(self, project_root: Path, member_alias: str) -> None:
        """Nạp YAML từng task và kiểm tra alias để bảo vệ provenance."""
        if not member_alias.strip():
            raise ValueError("Cần alias thành viên thật cho nhật ký thực nghiệm.")
        self.project_root = project_root.resolve()
        self.member_alias = member_alias.strip()
        self.configs = {
            key: yaml.safe_load((self.project_root / "configs" / name).read_text(encoding="utf-8"))
            for key, name in self.CONFIG_FILES.items()
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, task: str, epochs: int | None = None, smoke_test: bool = False) -> dict[str, Any]:
        """Chạy một bài toán, chọn checkpoint, ghi epoch logs và vẽ LR staircase."""
        config = dict(self.configs[task])
        seed_everything(int(config["seed"]))
        train_data, validation_data = self._load_datasets(task, config, smoke_test)
        if task == "gender":
            model = HumanGenderCNN(dropout=float(config["dropout"]))
        else:
            model = EmotionCNN(
                dropout_block=float(config["dropout_block"]),
                dropout_dense=float(config["dropout_dense"]),
            )
        model.init_weights_keras_style()
        optimizer = Adam(model.parameters(), lr=float(config["learning_rate"]))
        batch_size = int(config["batch_size"])
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            generator=get_torch_generator(int(config["seed"])),
            worker_init_fn=seed_worker,
            num_workers=int(config["num_workers"]),
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(config["num_workers"]),
        )
        run_epochs = int(epochs if epochs is not None else config["epochs"])
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=self.device,
            metric_fn=lambda logits, labels: (logits.argmax(1) == labels).float().mean(),
            model_group="04_training_utilities",
            model_name=model.model_name,
            seed=int(config["seed"]),
            member_alias=self.member_alias,
            config={**config, "epochs": run_epochs, "project_root": str(self.project_root)},
        )
        callbacks = self._build_callbacks(task, config, optimizer, smoke_test)
        checkpoint_path = callbacks[0].resolved_filepath()
        try:
            from src.logger import ExperimentLogger
        except Exception as error:
            raise RuntimeError(
                "T0209 cần ExperimentLogger/TensorBoard hoạt động; kiểm tra môi trường "
                "tensorboard/protobuf trước khi bắt đầu huấn luyện."
            ) from error
        logger = ExperimentLogger(
            model_group="Group4_Utilities",
            model_name=model.model_name,
            exp_type=f"T0209_{task}_{time.strftime('%Y%m%d_%H%M%S')}",
            seed=int(config["seed"]),
            optimizer=str(config["optimizer"]),
            learning_rate=float(config["learning_rate"]),
            batch_size=batch_size,
            epochs=run_epochs,
            member_alias=self.member_alias,
            project_root=self.project_root,
            csv_root=config["csv_root"],
            tensorboard_root=self.project_root / config["tensorboard_root"],
        )
        try:
            history = trainer.fit(
                train_loader,
                validation_loader,
                run_epochs,
                logger=logger,
                callbacks=callbacks,
            )
        except Exception:
            raise
        finally:
            if not getattr(logger, "_closed", False):
                logger.close()

        early_stopping = callbacks[-1]
        checkpoint_callback = callbacks[0]
        best_epoch = checkpoint_callback.best_epoch
        if best_epoch is None:
            best_epoch = int(np.argmin(history["val_loss"])) + 1
        if not math.isfinite(float(checkpoint_callback.best_score)):
            checkpoint_callback.best_score = history["val_loss"][best_epoch - 1]
        history["learning_rate"] = self._logged_learning_rates(logger, len(history["train_loss"]))
        history_path = self._save_history(task, history, config, callbacks)
        figure_path = self._save_curves(task, history, config, callbacks)
        result = {
            "task": task,
            "model": model.model_name,
            "device": str(self.device),
            "parameters": model.count_parameters()["total_params"],
            "epochs_configured": run_epochs,
            "epochs_completed": len(history["train_loss"]),
            "best_epoch": checkpoint_callback.best_epoch or best_epoch,
            "best_val_loss": float(checkpoint_callback.best_score),
            "best_val_accuracy": history["val_metric"][best_epoch - 1],
            "final_train_accuracy": history["train_metric"][-1],
            "final_val_accuracy": history["val_metric"][-1],
            "overfit_gap_final": history["train_metric"][-1] - history["val_metric"][-1],
            "lr_changes": callbacks[1].lr_changes,
            "early_stopping_epoch": early_stopping.stopped_epoch,
            "checkpoint": str(checkpoint_path.relative_to(self.project_root)),
            "history_csv": str(history_path.relative_to(self.project_root)),
            "figure": str(figure_path.relative_to(self.project_root)),
            "smoke_test": smoke_test,
            "validation_protocol": "WIKI/FER public-test validation, not independent final holdout (legacy protocol).",
        }
        metrics_path = self.project_root / "outputs/figures" / f"t0209_{task}_utilities_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        return result

    def _build_callbacks(
        self, task: str, config: dict[str, Any], optimizer: torch.optim.Optimizer,
        smoke_test: bool,
    ) -> list[Any]:
        """Tạo checkpoint, scheduler, early-stopping theo thứ tự phối hợp an toàn."""
        callback_options = config["early_stopping"]
        scheduler_options = config["reduce_lr"]
        checkpoint_options = config["checkpoint"]
        checkpoint_path = self.project_root / checkpoint_options["filepath"]
        drive_root = self._drive_root() if checkpoint_options["sync_to_drive"] else None
        return [
            # Save checkpoint trước; EarlyStopping là cuối để khôi phục state cuối cùng.
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=checkpoint_options["monitor"],
                mode=checkpoint_options["mode"],
                save_best_only=bool(checkpoint_options["save_best_only"]),
                save_weights_only=bool(checkpoint_options["save_weights_only"]),
                sync_to_drive=bool(checkpoint_options["sync_to_drive"]),
                verbose=not smoke_test,
                drive_root=drive_root,
            ),
            ReduceLROnPlateau(
                optimizer=optimizer,
                monitor=scheduler_options["monitor"],
                factor=float(scheduler_options["factor"]),
                patience=int(scheduler_options["patience"]),
                min_delta=float(scheduler_options["min_delta"]),
                cooldown=int(scheduler_options["cooldown"]),
                min_lr=float(scheduler_options["min_lr"]),
                mode=scheduler_options["mode"],
                verbose=not smoke_test,
            ),
            EarlyStopping(
                monitor=callback_options["monitor"],
                min_delta=float(callback_options["min_delta"]),
                patience=int(callback_options["patience"]),
                mode=callback_options["mode"],
                restore_best_weights=bool(callback_options["restore_best_weights"]),
                verbose=not smoke_test,
            ),
        ]

    def _load_datasets(
        self, task: str, config: dict[str, Any], smoke_test: bool
    ) -> tuple[Dataset, Dataset]:
        """Tạo dataset thật từ YAML hoặc schema-đúng giả lập cho smoke-test."""
        if task == "gender":
            if smoke_test:
                rng = np.random.default_rng(42)
                train_x = rng.integers(0, 256, (8, 64, 64, 1), dtype=np.uint8)
                train_y = np.arange(8, dtype=np.int64) % 2
                val_x, val_y = train_x[:4], train_y[:4]
            else:
                tensorboard_root=(
                    self.project_root / config["tensorboard_root"] / "Group4_Utilities"
                ),
                train_x = np.load(root / config["train_images"], mmap_mode="r")
                train_y = np.load(root / config["train_labels"], mmap_mode="r")
                val_x = np.load(root / config["validation_images"], mmap_mode="r")
                val_y = np.load(root / config["validation_labels"], mmap_mode="r")
            options = {
                "apply_legacy_rotation": bool(config["apply_legacy_rotation"]),
                "normalization": config["normalization"],
            }
            return WikiGenderDataset(train_x, train_y, **options), WikiGenderDataset(val_x, val_y, **options)

        if smoke_test:
            rng = np.random.default_rng(42)
            train_x = rng.random((8, 48, 48), dtype=np.float32)
            train_y = np.arange(8, dtype=np.int64) % 7
            val_x, val_y = train_x[:4], train_y[:4]
        else:
            root = self.project_root / config["data_root"]
            train_x = root / config["train_csv"]
            train_y = None
            val_x = root / config["validation_csv"]
            val_y = None
        train_data = FEREmotionDataset(train_x, train_y, normalization=config["normalization"])
        validation_data = FEREmotionDataset(val_x, val_y, normalization=config["normalization"])
        if not smoke_test:
            train_data = LegacyFERAugmentationDataset(train_data, seed=int(config["seed"]))
        return train_data, validation_data

    def _logged_learning_rates(
        self, logger: Any, epoch_count: int
    ) -> list[float]:
        """Đọc LR epoch ghi trực tiếp bởi ExperimentLogger từ CSV của run."""
        if logger.epoch_csv_path.is_file():
            with logger.epoch_csv_path.open(encoding="utf-8-sig", newline="") as stream:
                values = [float(row["Learning_Rate"]) for row in csv.DictReader(stream)]
            if len(values) == epoch_count:
                return values
        return [float(logger.learning_rate or 0.0)] * epoch_count

    def _save_history(
        self, task: str, history: dict[str, list[float]], config: dict[str, Any], callbacks: list[Any]
    ) -> Path:
        """Ghi Train/Val Loss/Acc/LR và callback events ra CSV chi tiết."""
        destination = self.project_root / config["csv_root"] / f"T0209_{task}_utilities_history.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint, scheduler, early = callbacks[0], callbacks[1], callbacks[2]
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "Epoch", "Train_Loss", "Val_Loss", "Train_Acc", "Val_Acc",
                "Learning_Rate", "Is_Best_Epoch", "Is_LR_Reduction", "Is_Early_Stop",
            ))
            reductions = {epoch for epoch, _, _ in scheduler.lr_changes}
            for index, (train_loss, val_loss, train_acc, val_acc, lr) in enumerate(zip(
                history["train_loss"], history["val_loss"], history["train_metric"],
                history["val_metric"], history["learning_rate"],
            ), start=1):
                writer.writerow((
                    index, train_loss, val_loss, train_acc, val_acc, lr,
                    index == checkpoint.best_epoch, index in reductions,
                    index == early.stopped_epoch,
                ))
        return destination

    def _save_curves(
        self, task: str, history: dict[str, list[float]], config: dict[str, Any], callbacks: list[Any]
    ) -> Path:
        """Vẽ loss, accuracy, LR và đánh dấu best/checkpoint, LR drop, stop."""
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        figure, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
        axes[0].plot(epochs, history["train_loss"], label="Train_Loss")
        axes[0].plot(epochs, history["val_loss"], label="Val_Loss")
        axes[1].plot(epochs, history["train_metric"], label="Train_Acc")
        axes[1].plot(epochs, history["val_metric"], label="Val_Acc")
        axes[2].step(epochs, history["learning_rate"], where="post", label="Learning_Rate")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend()
        best_epoch = callbacks[0].best_epoch
        stopped_epoch = callbacks[2].stopped_epoch
        for axis in axes:
            if best_epoch is not None:
                axis.axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label="Best_Epoch")
            for reduction_epoch, _, _ in callbacks[1].lr_changes:
                axis.axvline(reduction_epoch, color="orange", linestyle=":", alpha=0.6)
            if stopped_epoch is not None:
                axis.axvline(stopped_epoch, color="red", linestyle="-.", alpha=0.7, label="EarlyStopping")
        axes[-1].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-entropy")
        axes[1].set_ylabel("Accuracy")
        axes[2].set_ylabel("LR")
        figure.suptitle(f"{config['model_name']} — callbacks convergence")
        figure.tight_layout()
        destination = self.project_root / config["figure_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return destination

    def _drive_root(self) -> Path | None:
        """Phát hiện mount Drive đã tồn tại qua config, không tự mount."""
        candidate = self.project_root / "configs/paths.yaml"
        if not candidate.is_file():
            return None
        configured = (yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}).get("project", {}).get("drive_mount_root")
        path = Path(configured) if configured else None
        return path if path and path.is_dir() else None

    @staticmethod
    def _read_member_name(path: Path) -> str:
        """Đọc MEMBER_04_NAME từ `.env`, không suy đoán định danh."""
        if not path.is_file():
            raise FileNotFoundError("Thiếu `.env`; không thể tự đặt tác giả báo cáo.")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "MEMBER_04_NAME":
                name = value.strip().strip('"').strip("'")
                if name:
                    return name
        raise ValueError("`.env` thiếu MEMBER_04_NAME.")


def main(argv: list[str] | None = None) -> int:
    """Chạy lựa chọn Gender, Emotion hoặc cả hai với seed cố định từ YAML."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("gender", "emotion", "all"), default="all")
    parser.add_argument("--epochs", type=int, help="Ghi đè epoch để smoke-test.")
    parser.add_argument("--smoke-test", action="store_true", help="Dữ liệu giả schema-đúng; không dùng làm baseline.")
    parser.add_argument("--member-alias", help="Tên thật; mặc định MEMBER_04_NAME từ `.env`.")
    args = parser.parse_args(argv)
    alias = args.member_alias or Group4UtilitiesRunner._read_member_name(PROJECT_ROOT / ".env")
    runner = Group4UtilitiesRunner(PROJECT_ROOT, alias)
    selected = list(runner.CONFIG_FILES) if args.task == "all" else [args.task]
    for task in selected:
        runner.run(task, epochs=args.epochs, smoke_test=args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())