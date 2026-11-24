"""Huấn luyện và đánh giá ba mô hình CNN Nhóm 2 từ cấu hình YAML."""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch import nn
from torch.optim import Adam, Optimizer, SGD
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import FEREmotionDataset, MNISTDataset, WikiGenderDataset
from src.engine import BaseTrainer
from src.models import EmotionCNN, HumanGenderCNN, MNISTCNN
from src.utils import get_torch_generator, seed_everything, seed_worker


class LegacyFERAugmentationDataset(Dataset):
    """Áp dụng augmentation FER legacy chỉ khi lấy mẫu train."""

    def __init__(self, dataset: FEREmotionDataset, seed: int = 42) -> None:
        """Giữ dataset chuẩn và generator riêng tái lập cho phép biến đổi."""
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        """Trả số ảnh FER nguồn."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Xoay ±45°, dịch ±20% và lật ngang ngẫu nhiên ảnh train."""
        image, label = self.dataset[index]
        rotation = float(torch.rand((), generator=self.generator)) * 90.0 - 45.0
        angle = rotation * np.pi / 180.0
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        theta = torch.tensor(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0]],
            dtype=image.dtype,
        ).unsqueeze(0)
        # affine_grid dùng tọa độ [-1,1], nên ±0.4 tương ứng ±20% chiều ảnh.
        theta[0, 0, 2] = float(torch.rand((), generator=self.generator)) * 0.8 - 0.4
        theta[0, 1, 2] = float(torch.rand((), generator=self.generator)) * 0.8 - 0.4
        grid = functional.affine_grid(theta, (1, *image.shape), align_corners=False)
        image = functional.grid_sample(
            image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).squeeze(0)
        if bool(torch.rand((), generator=self.generator) < 0.5):
            image = image.flip(-1)
        return image, label


class Group2CNNRunner:
    """Nối YAML, dataset, model, trainer và artifacts của Nhóm 2."""

    CONFIGS = {
        "mnist": "cnn_mnist.yaml",
        "gender": "cnn_gender.yaml",
        "emotion": "cnn_emotion.yaml",
    }

    def __init__(self, project_root: Path, member_alias: str) -> None:
        """Đọc cấu hình ba task và xác thực alias thành viên thật."""
        if not member_alias.strip():
            raise ValueError("Cần alias thành viên thật cho logger và báo cáo.")
        self.project_root = project_root.resolve()
        self.member_alias = member_alias.strip()
        self.configs = {
            key: yaml.safe_load((self.project_root / "configs" / filename).read_text(encoding="utf-8"))
            for key, filename in self.CONFIGS.items()
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, model_key: str, epochs: int | None = None, smoke: bool = False) -> dict[str, Any]:
        """Chạy một model; smoke chỉ lấy lát dữ liệu nhỏ và không suy ra accuracy khoa học."""
        config = dict(self.configs[model_key])
        seed_everything(int(config["seed"]))
        if model_key == "mnist":
            train_dataset, validation_dataset, test_dataset = self._load_mnist(config, smoke)
            model = MNISTCNN(dropout=float(config["dropout"]))
        elif model_key == "gender":
            train_dataset, validation_dataset = self._load_gender(config, smoke)
            test_dataset = None  # Legacy chỉ có validation/public-test split.
            model = HumanGenderCNN(dropout=float(config["dropout"]))
        else:
            train_dataset, validation_dataset = self._load_emotion(config, smoke)
            test_dataset = None  # Legacy tái sử dụng public-test làm validation/evaluate.
            model = EmotionCNN(
                dropout_block=float(config["dropout_block"]),
                dropout_dense=float(config["dropout_dense"]),
            )

        model.init_weights_keras_style()
        batch_size = int(config["batch_size"])
        loader_workers = int(config["num_workers"])
        if model_key == "emotion" and not smoke:
            train_dataset = LegacyFERAugmentationDataset(train_dataset, seed=int(config["seed"]))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            generator=get_torch_generator(int(config["seed"])),
            worker_init_fn=seed_worker, num_workers=loader_workers,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False,
            num_workers=loader_workers,
        )
        run_epochs = int(epochs if epochs is not None else config["epochs"])
        optimizer = self._optimizer(model, config)
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=self.device,
            metric_fn=lambda logits, labels: (logits.argmax(dim=1) == labels).float().mean(),
            model_group="02_cnn_mlp",
            model_name=model.model_name,
            seed=int(config["seed"]),
            member_alias=self.member_alias,
            config={**config, "epochs": run_epochs, "smoke_test": smoke},
        )
        # Logger là bắt buộc cho mọi training; helper model/dataset ở trên vẫn
        # có thể import riêng mà không nạp TensorBoard.
        from src.logger import CheckpointManager, ExperimentLogger

        logger = ExperimentLogger(
            model_group="Group2_CNN",
            model_name=model.model_name,
            exp_type=(
                f"T020{3 + list(self.CONFIGS).index(model_key)}_"
                f"{model.model_name}_{uuid.uuid4().hex[:8]}"
            ),
            seed=int(config["seed"]),
            optimizer=config["optimizer"],
            learning_rate=float(config["learning_rate"]),
            batch_size=batch_size,
            epochs=run_epochs,
            member_alias=self.member_alias,
            project_root=self.project_root,
            csv_root=config["csv_root"],
            tensorboard_root=config["tensorboard_root"],
        )
        checkpoint_manager = CheckpointManager(
            project_root=self.project_root,
            checkpoint_root=self.project_root / config["checkpoint_root"],
            drive_root=self._configured_drive_root(),
        )
        destination = (
            self.project_root / config["checkpoint_root"] / "02_cnn_mlp" / config["checkpoint_name"]
        )
        deferred_logger = self._DeferredFinishLogger(logger)
        try:
            history = trainer.fit(
                train_loader, validation_loader, run_epochs,
                logger=deferred_logger, checkpoint_manager=checkpoint_manager,
            )
            # BaseTrainer checkpoint chọn val_loss thấp nhất. Nạp lại checkpoint đó
            # trước khi test/feature, tránh đánh giá trọng số epoch cuối nhầm lẫn.
            training_checkpoint = (
                self.project_root / config["checkpoint_root"] / "02_cnn_mlp"
                / f"{model.model_name}_best.pth"
            )
            payload = CheckpointManager.load(
                training_checkpoint, model=model, map_location=self.device
            )
            checkpoint_manager.save(
                model=model, model_group="02_cnn_mlp", model_name=model.model_name,
                epoch=int(payload["epoch"]), optimizer=optimizer,
                best_metric=float(payload["best_metric"]), seed=int(config["seed"]),
                config={**config, "epochs": run_epochs},
                filename=str(config["checkpoint_name"]),
            )
            test_accuracy = self._accuracy(model, test_dataset, batch_size) if test_dataset else None
            if model_key == "mnist":
                features, labels = self._extract_pool3(model, test_dataset, batch_size)
                feature_path = self.project_root / config["feature_output"]
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(feature_path, features=features.numpy(), labels=labels.numpy())
            else:
                features = None
            if model_key == "gender":
                self._save_wiki_verification(validation_dataset)
        except Exception:
            logger.close()
            raise
        else:
            best_epoch = int(np.argmin(history["val_loss"])) + 1
            logger.finish_run(
                best_epoch=best_epoch,
                train_loss=history["train_loss"][best_epoch - 1],
                val_loss=history["val_loss"][best_epoch - 1],
                train_metric=history["train_metric"][best_epoch - 1],
                val_metric=history["val_metric"][best_epoch - 1],
                test_metric=test_accuracy,
                train_time_sec=sum(history["epoch_time_sec"]),
            )
            logger.close()

        history["learning_rate"] = self._logged_learning_rates(logger, len(history["train_loss"]))
        self._save_history(model_key, model.model_name, history, config)
        result = {
            "model": model.model_name,
            "parameters": model.count_parameters()["total_params"],
            "best_epoch": int(np.argmin(history["val_loss"])) + 1,
            "train_accuracy_last": history["train_metric"][-1],
            "validation_accuracy_last": history["val_metric"][-1],
            "test_accuracy": test_accuracy,
            "device": str(self.device),
            "checkpoint": str(destination.relative_to(self.project_root)),
            "feature_shape": list(features.shape) if features is not None else None,
            "smoke_test": smoke,
        }
        print(json.dumps(result, ensure_ascii=False))
        if model_key == "mnist" and not smoke and test_accuracy is not None:
            if test_accuracy < float(config["target_test_accuracy"]):
                raise RuntimeError(
                    f"MNIST Test Accuracy {test_accuracy:.4%} thấp hơn ngưỡng "
                    f"{float(config['target_test_accuracy']):.2%}."
                )
        return result

    @staticmethod
    def _optimizer(model: nn.Module, config: dict[str, Any]) -> Optimizer:
        """Tạo optimizer theo YAML, không thay siêu tham số baseline."""
        learning_rate = float(config["learning_rate"])
        if config["optimizer"].upper() == "SGD":
            return SGD(model.parameters(), lr=learning_rate)
        return Adam(model.parameters(), lr=learning_rate)

    def _load_mnist(
        self, config: dict[str, Any], smoke: bool
    ) -> tuple[Subset, Subset, MNISTDataset]:
        """Nạp cache MNIST hoặc tải bằng loader hiện có; giữ test 10.000 ảnh."""
        root = self.project_root / config["data_root"]
        if smoke:
            generator = torch.Generator().manual_seed(42)
            train_x = torch.randint(0, 256, (24, 28, 28), generator=generator).numpy().astype(np.uint8)
            train_y = np.arange(24, dtype=np.int64) % 10
            test_x = train_x[:12].copy()
            test_y = train_y[:12].copy()
        else:
            arrays = self._load_mnist_arrays(root)
            train_x, train_y = arrays["train_images"], arrays["train_labels"]
            test_x, test_y = arrays["test_images"], arrays["test_labels"]
        dataset = MNISTDataset(train_x, train_y)
        test_dataset = MNISTDataset(test_x, test_y)
        val_count = max(1, int(len(dataset) * float(config["validation_fraction"])))
        permutation = torch.randperm(len(dataset), generator=get_torch_generator(42)).tolist()
        return (
            Subset(dataset, permutation[val_count:]),
            Subset(dataset, permutation[:val_count]),
            test_dataset,
        )

    def _load_mnist_arrays(self, root: Path) -> dict[str, np.ndarray]:
        """Nạp bốn mảng MNIST thật hoặc tải MNIST IDX chính thức."""
        names = ("train_images", "train_labels", "test_images", "test_labels")
        files = {name: root / f"{name}.npy" for name in names}
        if not all(path.is_file() for path in files.values()):
            from scripts.download_data import DataBootstrapper
            DataBootstrapper(self.project_root).download_mnist()
        return {name: np.load(path, mmap_mode="r") for name, path in files.items()}

    def _load_gender(
        self, config: dict[str, Any], smoke: bool
    ) -> tuple[WikiGenderDataset, WikiGenderDataset]:
        """Nạp WIKI NHWC; Dataset thực hiện rot90 trước NHWC→NCHW."""
        root = self.project_root / config["data_root"]
        if smoke:
            rng = np.random.default_rng(42)
            train_x = rng.integers(0, 256, (8, 64, 64, 1), dtype=np.uint8)
            train_y = np.eye(2, dtype=np.float32)[np.arange(8) % 2]
            val_x, val_y = train_x[:4], train_y[:4]
        else:
            train_x = np.load(root / config["train_images"], mmap_mode="r")
            train_y = np.load(root / config["train_labels"], mmap_mode="r")
            val_x = np.load(root / config["validation_images"], mmap_mode="r")
            val_y = np.load(root / config["validation_labels"], mmap_mode="r")
        options = {
            "apply_legacy_rotation": bool(config["apply_legacy_rotation"]),
            "normalization": config["normalization"],
        }
        return WikiGenderDataset(train_x, train_y, **options), WikiGenderDataset(val_x, val_y, **options)

    def _load_emotion(
        self, config: dict[str, Any], smoke: bool
    ) -> tuple[FEREmotionDataset, FEREmotionDataset]:
        """Nạp FER CSV gốc hoặc tạo tensor giả đúng schema 48×48, bảy lớp."""
        if smoke:
            rng = np.random.default_rng(42)
            # Ảnh giả dùng unit range để smoke ổn định; preserve mode vẫn
            # không tự chuẩn hóa dữ liệu, đúng preprocessing legacy.
            train_x = rng.random((8, 48, 48), dtype=np.float32)
            train_y = np.eye(7, dtype=np.float32)[np.arange(8) % 7]
            val_x, val_y = train_x[:4], train_y[:4]
        else:
            root = self.project_root / config["data_root"]
            train_x, train_y = root / config["train_csv"], None
            val_x, val_y = root / config["validation_csv"], None
        return (
            FEREmotionDataset(train_x, train_y, normalization=config["normalization"]),
            FEREmotionDataset(val_x, val_y, normalization=config["normalization"]),
        )

    @staticmethod
    def _accuracy(model: nn.Module, dataset: Dataset, batch_size: int) -> float:
        """Tính accuracy của split được truyền, không tự đặt tên split độc lập."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        correct = total = 0
        model.eval()
        with torch.inference_mode():
            for images, labels in loader:
                predictions = model(images.to(next(model.parameters()).device)).argmax(1).cpu()
                correct += int((predictions == labels).sum())
                total += len(labels)
        return correct / total

    @staticmethod
    def _extract_pool3(
        model: MNISTCNN, dataset: Dataset, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook MaxPool `pool3`, làm phẳng (N,256,3,3) thành (N,2304)."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        model.register_feature_hooks(["pool3"])
        feature_batches, label_batches = [], []
        try:
            with torch.inference_mode():
                for images, labels in loader:
                    _, captures = model.extract_features(images.to(next(model.parameters()).device))
                    feature_batches.append(captures["pool3"].flatten(start_dim=1).cpu())
                    label_batches.append(labels.cpu())
        finally:
            model.clear_hooks()
        return torch.cat(feature_batches), torch.cat(label_batches)

    class _DeferredFinishLogger:
        """Ghi từng epoch ngay nhưng hoãn master row đến sau test evaluation."""

        def __init__(self, logger: Any) -> None:
            self.logger = logger
            self.member_alias = logger.member_alias

        def log_epoch(self, **kwargs: Any) -> None:
            """Chuyển tiếp epoch log theo API ExperimentLogger."""
            self.logger.log_epoch(**kwargs)

        def finish_run(self, **kwargs: Any) -> Path:
            """Giữ tương thích BaseTrainer, chưa ghi summary thiếu test metric."""
            del kwargs
            return self.logger.master_csv_path

    @staticmethod
    def _logged_learning_rates(logger: Any, epoch_count: int) -> list[float]:
        """Đọc LR epoch từ CSV run để phản ánh lịch scheduler đã áp dụng."""
        if logger.epoch_csv_path.is_file():
            with logger.epoch_csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rates = [float(row["Learning_Rate"]) for row in csv.DictReader(stream)]
            if len(rates) == epoch_count:
                return rates
        return [float(logger.learning_rate or 0.0)] * epoch_count

    def _save_history(
        self,
        model_key: str,
        model_name: str,
        history: dict[str, list[float]],
        config: dict[str, Any],
    ) -> None:
        """Ghi đường cong epoch CSV và biểu đồ train/validation gitignored."""
        figure_root = self.project_root / config["figure_root"]
        figure_root.mkdir(parents=True, exist_ok=True)
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, history["train_loss"], label="Train")
        axes[0].plot(epochs, history["val_loss"], label="Validation/public-test legacy")
        axes[0].set(title=f"{model_name} — Loss", xlabel="Epoch", ylabel="Cross-entropy")
        axes[1].plot(epochs, history["train_metric"], label="Train")
        axes[1].plot(epochs, history["val_metric"], label="Validation/public-test legacy")
        axes[1].set(title=f"{model_name} — Accuracy", xlabel="Epoch", ylabel="Accuracy")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(figure_root / f"t020{3 + list(self.CONFIGS).index(model_key)}_{model_key}_curves.png", dpi=150)
        plt.close(figure)
        history_path = self.project_root / config["csv_root"] / f"{model_name}_history.csv"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with history_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Epoch", "Train_Loss", "Val_Loss", "Train_Acc", "Val_Acc", "Learning_Rate"))
            writer.writerows(zip(
                epochs, history["train_loss"], history["val_loss"],
                history["train_metric"], history["val_metric"], history["learning_rate"],
            ))

    def _save_wiki_verification(self, dataset: WikiGenderDataset) -> Path:
        """Vẽ lưới ảnh WIKI sau rot90 và chuyển NCHW để kiểm chứng thủ công."""
        figure, axes = plt.subplots(2, 2, figsize=(6, 6))
        for index, axis in enumerate(axes.flat):
            axis.axis("off")
            if index < len(dataset):
                image, label = dataset[index]
                axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
                axis.set_title(f"label={int(label)}; tensor={tuple(image.shape)}")
        figure.suptitle("WIKI sau np.rot90(k=3), rồi NHWC → NCHW")
        figure.tight_layout()
        destination = self.project_root / "outputs/figures/t0204_wiki_rot90_verify.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return destination

    def _configured_drive_root(self) -> Path | None:
        """Mirror artifacts chỉ khi runtime đã mount/configure Drive thật."""
        candidate = self.project_root / "configs/paths.yaml"
        if not candidate.is_file():
            return None
        configured = (yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}).get("project", {}).get("drive_mount_root")
        path = Path(configured) if configured else None
        return path if path and path.is_dir() else None

    @staticmethod
    def _read_member_name(path: Path) -> str:
        """Đọc đúng biến MEMBER_02_NAME mà không phụ thuộc python-dotenv."""
        if not path.is_file():
            raise FileNotFoundError("Thiếu `.env`; không thể ghi log với định danh thành viên.")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "MEMBER_02_NAME":
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
        raise ValueError("`.env` thiếu MEMBER_02_NAME.")


def main(argv: list[str] | None = None) -> int:
    """Chạy chọn lọc CNN; alias Member_02 lấy từ cấu hình cục bộ `.env`."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mnist", "gender", "emotion", "all"), default="all")
    parser.add_argument("--epochs", type=int, help="Override epoch để chạy kiểm thử/smoke.")
    parser.add_argument("--smoke-test", action="store_true", help="Dữ liệu giả; không dùng kết quả làm baseline.")
    parser.add_argument("--member-alias", help="Tên thật; mặc định đọc MEMBER_02_NAME từ `.env`.")
    args = parser.parse_args(argv)
    alias = args.member_alias or Group2CNNRunner._read_member_name(PROJECT_ROOT / ".env")
    runner = Group2CNNRunner(PROJECT_ROOT, member_alias=alias)
    selected = list(runner.CONFIGS) if args.model == "all" else [args.model]
    for model_key in selected:
        runner.run(model_key, epochs=args.epochs, smoke=args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())