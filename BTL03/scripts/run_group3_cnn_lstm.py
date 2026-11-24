"""Chạy sinh chuỗi MNIST và huấn luyện CNN-LSTM theo cấu hình YAML."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_data import DataBootstrapper
from src.datasets import SequentialMNISTDataset
from src.engine import BaseTrainer
from src.models import MNISTCNNLSTM
from src.utils import get_torch_generator, seed_everything, seed_worker


class Group3CNNLSTMRunner:
    """Kết nối dataset chuỗi, CNN-LSTM, trainer, logger và artifacts."""

    def __init__(self, project_root: Path, member_alias: str) -> None:
        """Đọc YAML và yêu cầu định danh người chạy rõ ràng."""
        if not member_alias.strip():
            raise ValueError("Cần tên thành viên thật để ghi log thực nghiệm.")
        self.project_root = project_root.resolve()
        self.member_alias = member_alias.strip()
        self.config = yaml.safe_load(
            (self.project_root / "configs/cnn_lstm_mnist.yaml").read_text(encoding="utf-8")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, epochs: int | None = None, smoke_test: bool = False) -> dict[str, Any]:
        """Huấn luyện model và đánh giá checkpoint tốt nhất trên split test legacy."""
        config = self.config
        seed_everything(int(config["seed"]))
        if smoke_test:
            train_images, train_labels, test_images, test_labels = self._dummy_arrays()
        else:
            arrays = self._load_mnist_arrays()
            train_images, train_labels = arrays["train_images"], arrays["train_labels"]
            test_images, test_labels = arrays["test_images"], arrays["test_labels"]

        train_dataset = SequentialMNISTDataset(
            train_images, train_labels, seq_len=int(config["seq_len"]),
            seed=int(config["seed"]), is_train=True,
        )
        test_dataset = SequentialMNISTDataset(
            test_images, test_labels, seq_len=int(config["seq_len"]),
            seed=int(config["seed"]), is_train=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=get_torch_generator(int(config["seed"])),
            worker_init_fn=seed_worker,
            num_workers=int(config["num_workers"]),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=int(config["batch_size"]), shuffle=False,
            num_workers=int(config["num_workers"]),
        )
        first_x, first_y = next(iter(test_loader))
        if first_x.ndim != 5 or tuple(first_x.shape[1:]) != (8, 1, 28, 28):
            raise AssertionError(f"Batch shape không đúng B-T-C-H-W: {tuple(first_x.shape)}")
        if tuple(first_y.shape[1:]) != (1,):
            raise AssertionError(f"Target cần shape (B,1), nhận {tuple(first_y.shape)}")

        model = MNISTCNNLSTM(
            seq_len=int(config["seq_len"]),
            frame_dropout=float(config["frame_dropout"]),
            lstm_dropout=float(config["lstm_dropout"]),
        )
        model.init_weights_keras_style()
        optimizer = SGD(model.parameters(), lr=float(config["learning_rate"]))
        run_epochs = int(epochs if epochs is not None else config["epochs"])
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.L1Loss(),
            device=self.device,
            metric_fn=lambda predictions, targets: torch.mean(torch.abs(predictions - targets)),
            model_group="03_cnn_lstm",
            model_name="MNISTCNNLSTM",
            seed=int(config["seed"]),
            member_alias=self.member_alias,
            config={**config, "epochs": run_epochs, "smoke_test": smoke_test},
        )

        # Logger wordt laat geïmporteerd: alleen trainingsruns eisen TensorBoard.
        from src.logger import CheckpointManager, ExperimentLogger

        run_tag = f"{'SmokeTest_' if smoke_test else ''}T0207_MNISTCNNLSTM_{uuid.uuid4().hex[:8]}"
        logger = ExperimentLogger(
            model_group="Group3_CNN_LSTM",
            model_name="MNISTCNNLSTM",
            exp_type=run_tag,
            seed=int(config["seed"]),
            optimizer=str(config["optimizer"]),
            learning_rate=float(config["learning_rate"]),
            batch_size=int(config["batch_size"]),
            epochs=run_epochs,
            member_alias=self.member_alias,
            project_root=self.project_root,
            csv_root=config["csv_root"],
            tensorboard_root=config["tensorboard_root"],
        )
        manager = CheckpointManager(
            project_root=self.project_root,
            checkpoint_root=self.project_root / config["checkpoint_root"],
            drive_root=self._drive_root(),
        )
        deferred_logger = self._DeferredFinishLogger(logger)
        try:
            history = trainer.fit(
                train_loader,
                test_loader,
                run_epochs,
                logger=deferred_logger,
                checkpoint_manager=manager,
            )
            best_epoch = int(np.argmin(history["val_loss"])) + 1
            source_checkpoint = (
                self.project_root / config["checkpoint_root"] / "03_cnn_lstm"
                / "MNISTCNNLSTM_best.pth"
            )
            checkpoint_payload = CheckpointManager.load(
                source_checkpoint,
                model=model,
                optimizer=optimizer,
                map_location=self.device,
            )
            test_mae = self._evaluate_mae(model, test_loader)
            final_checkpoint = manager.save(
                model=model,
                model_group="03_cnn_lstm",
                model_name="MNISTCNNLSTM",
                epoch=int(checkpoint_payload["epoch"]),
                optimizer=optimizer,
                best_metric=float(checkpoint_payload["best_metric"]),
                seed=int(config["seed"]),
                config={**config, "epochs": run_epochs},
                filename=str(config["checkpoint_name"]),
            )
            best_epoch = int(checkpoint_payload["epoch"])
            logger.finish_run(
                best_epoch=best_epoch,
                train_loss=history["train_loss"][best_epoch - 1],
                val_loss=history["val_loss"][best_epoch - 1],
                train_metric=history["train_metric"][best_epoch - 1],
                val_metric=history["val_metric"][best_epoch - 1],
                test_metric=test_mae,
                train_time_sec=sum(history["epoch_time_sec"]),
                metric_name="MAE",
            )
        except Exception:
            logger.close()
            raise
        else:
            logger.close()

        result_figure = self._save_results_figure(model, test_dataset, history, config)
        history_path = self._save_history(history, config)
        target_mae = float(config["target_test_mae"])
        result = {
            "model": model.model_name,
            "parameters": model.count_parameters()["total_params"],
            "train_sequences": len(train_dataset),
            "test_sequences": len(test_dataset),
            "best_epoch": best_epoch,
            "test_mae": test_mae,
            "target_mae": target_mae,
            "target_met": test_mae <= target_mae,
            "device": str(self.device),
            "checkpoint": str(final_checkpoint.relative_to(self.project_root)),
            "history_csv": str(history_path.relative_to(self.project_root)),
            "figure": str(result_figure.relative_to(self.project_root)),
            "legacy_protocol_note": "MNIST legacy dùng test split làm validation_data; đây không phải test holdout độc lập.",
            "smoke_test": smoke_test,
        }
        metrics_path = self.project_root / "outputs/figures/t0207_cnn_lstm_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        return result

    def create_dataset_evidence(self, smoke_test: bool = False) -> Path:
        """Sinh ảnh minh họa chuỗi và nhãn tổng trước khi bắt đầu training."""
        seed_everything(int(self.config["seed"]))
        if smoke_test:
            images, labels, _, _ = self._dummy_arrays()
        else:
            arrays = self._load_mnist_arrays()
            images, labels = arrays["train_images"], arrays["train_labels"]
        dataset = SequentialMNISTDataset(
            images, labels, seq_len=int(self.config["seq_len"]),
            seed=int(self.config["seed"]), is_train=True,
        )
        return self._save_dataset_figure(dataset)

    def _load_mnist_arrays(self) -> dict[str, np.ndarray]:
        """Đọc bốn cache NumPy hoặc dùng MNIST IDX downloader đã có."""
        root = self.project_root / self.config["data_root"]
        names = ("train_images", "train_labels", "test_images", "test_labels")
        paths = {name: root / f"{name}.npy" for name in names}
        if not all(path.is_file() for path in paths.values()):
            DataBootstrapper(self.project_root).download_mnist()
        return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}

    @staticmethod
    def _dummy_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tạo MNIST-like arrays nhỏ để kiểm tra pipeline offline."""
        rng = np.random.default_rng(42)
        train_images = rng.integers(0, 256, (24, 28, 28), dtype=np.uint8)
        test_images = rng.integers(0, 256, (12, 28, 28), dtype=np.uint8)
        return (
            train_images,
            np.arange(len(train_images), dtype=np.uint8) % 10,
            test_images,
            np.arange(len(test_images), dtype=np.uint8) % 10,
        )

    @staticmethod
    def _evaluate_mae(model: MNISTCNNLSTM, loader: DataLoader) -> float:
        """Tính MAE theo tổng số chuỗi và khôi phục trạng thái train/eval."""
        was_training = model.training
        model.eval()
        absolute_error = 0.0
        sample_count = 0
        with torch.inference_mode():
            for sequences, targets in loader:
                predictions = model(sequences.to(next(model.parameters()).device))
                absolute_error += float(torch.abs(predictions - targets.to(predictions.device)).sum())
                sample_count += len(sequences)
        model.train(was_training)
        return absolute_error / sample_count

    def _save_dataset_figure(self, dataset: SequentialMNISTDataset) -> Path:
        """Vẽ tối đa bốn chuỗi tám khung có số khung zero-padded."""
        figure, axes = plt.subplots(4, 8, figsize=(14, 7))
        for row, axis_row in enumerate(axes):
            sequence, target = dataset[row]
            metadata = dataset.metadata(row)
            for frame_index, axis in enumerate(axis_row):
                axis.imshow(sequence[frame_index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
                axis.axis("off")
                if row == 0:
                    axis.set_title(f"t={frame_index}", fontsize=8)
                if frame_index == 0:
                    axis.set_ylabel(
                        f"Tổng={int(target.item())}\n{metadata['frame_count']}/8 ảnh",
                        fontsize=8,
                    )
        figure.suptitle("MNIST: chuỗi tối đa 8 ảnh; khung còn lại được zero-padding")
        figure.tight_layout()
        destination = self.project_root / self.config["figure_root"] / "t0206_seq_mnist_samples.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return destination

    def _save_results_figure(
        self,
        model: MNISTCNNLSTM,
        dataset: SequentialMNISTDataset,
        history: dict[str, list[float]],
        config: dict[str, Any],
    ) -> Path:
        """Vẽ train/validation MAE cùng ví dụ dự đoán tổng trên test split."""
        figure = plt.figure(figsize=(14, 9))
        grid = figure.add_gridspec(2, 1, height_ratios=(1, 1.5))
        curve_axis = figure.add_subplot(grid[0])
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        curve_axis.plot(epochs, history["train_loss"], label="Train MAE")
        curve_axis.plot(epochs, history["val_loss"], label="Validation MAE (legacy test split)")
        curve_axis.set(xlabel="Epoch", ylabel="MAE tổng chữ số", title="CNN-LSTM learning curve")
        curve_axis.grid(True, alpha=0.25)
        curve_axis.legend()

        examples = min(4, len(dataset))
        prediction_axis = figure.add_subplot(grid[1])
        prediction_axis.axis("off")
        table_rows = []
        model.eval()
        with torch.inference_mode():
            for index in range(examples):
                sequence, target = dataset[index]
                prediction = float(model(sequence.unsqueeze(0).to(self.device)).item())
                table_rows.append((index, int(target.item()), f"{prediction:.3f}", dataset.metadata(index)["frame_count"]))
        table = prediction_axis.table(
            cellText=table_rows,
            colLabels=("Mẫu", "Tổng thật", "Dự đoán", "Số ảnh khác 0"),
            loc="center",
        )
        table.scale(1, 1.7)
        prediction_axis.set_title("Ví dụ dự đoán trên split test legacy")
        figure.tight_layout()
        destination = self.project_root / config["figure_root"] / "t0207_cnn_lstm_results.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return destination

    def _save_history(self, history: dict[str, list[float]], config: dict[str, Any]) -> Path:
        """Lưu history nhẹ để đối chiếu train/validation theo từng epoch."""
        destination = self.project_root / config["csv_root"] / "T0207_MNISTCNNLSTM_history.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Epoch", "Train_Loss", "Val_Loss", "Train_MAE", "Val_MAE", "Epoch_Time_Sec"))
            writer.writerows(zip(
                range(1, len(history["train_loss"]) + 1),
                history["train_loss"], history["val_loss"],
                history["train_metric"], history["val_metric"], history["epoch_time_sec"],
            ))
        return destination

    def _drive_root(self) -> Path | None:
        """Trả Drive root chỉ khi YAML chỉ tới thư mục đã mount thực sự."""
        config_path = self.project_root / "configs/paths.yaml"
        if not config_path.is_file():
            return None
        configured = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("project", {}).get("drive_mount_root")
        path = Path(configured) if configured else None
        return path if path and path.is_dir() else None

    @staticmethod
    def _read_member_name(path: Path) -> str:
        """Đọc MEMBER_03_NAME từ `.env`, không cần thư viện dotenv."""
        if not path.is_file():
            raise FileNotFoundError("Thiếu `.env`; không được tự suy đoán định danh tác giả.")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "MEMBER_03_NAME":
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
        raise ValueError("`.env` chưa khai báo MEMBER_03_NAME.")

    class _DeferredFinishLogger:
        """Cho phép BaseTrainer ghi epoch, còn runner chốt summary sau test MAE."""

        def __init__(self, logger: Any) -> None:
            self.logger = logger
            self.member_alias = logger.member_alias

        def log_epoch(self, **kwargs: Any) -> None:
            """Chuyển tiếp log epoch tới ExperimentLogger thật."""
            kwargs.setdefault("metric_name", "MAE")
            self.logger.log_epoch(**kwargs)

        def finish_run(self, **kwargs: Any) -> Path:
            """Hoãn master row đến khi runner có test metric."""
            del kwargs
            return self.logger.master_csv_path


def main(argv: list[str] | None = None) -> int:
    """Điểm vào CLI cho evidence dataset và huấn luyện CNN-LSTM."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, help="Ghi đè số epoch cho lượt kiểm thử.")
    parser.add_argument("--smoke-test", action="store_true", help="Dùng dữ liệu giả nhỏ; không phải kết quả khoa học.")
    parser.add_argument("--dataset-only", action="store_true", help="Chỉ tạo hình minh họa T0206.")
    parser.add_argument("--member-alias", help="Tên thật; mặc định lấy MEMBER_03_NAME từ `.env`.")
    args = parser.parse_args(argv)
    alias = args.member_alias or Group3CNNLSTMRunner._read_member_name(PROJECT_ROOT / ".env")
    runner = Group3CNNLSTMRunner(PROJECT_ROOT, member_alias=alias)
    if args.dataset_only:
        print(runner.create_dataset_evidence(smoke_test=args.smoke_test))
    else:
        runner.create_dataset_evidence(smoke_test=args.smoke_test)
        runner.run(epochs=args.epochs, smoke_test=args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())