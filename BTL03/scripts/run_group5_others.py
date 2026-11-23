"""CLI huấn luyện SegNet và FCN detector theo cấu hình YAML nhóm 5."""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import FCNObjectDetectionDataset, SemanticSegmentationDataset
from src.engine import BaseTrainer
from src.engine.callbacks import ModelCheckpoint
from src.logger import CheckpointManager, ExperimentLogger
from src.models import FCNObjectDetector, SegNet
from src.utils import get_torch_generator, seed_everything, seed_worker


class Group5OthersRunner:
    """Điều phối train/evaluation/visualization của hai bài toán nâng cao."""

    def __init__(
        self,
        project_root: str | Path = PROJECT_ROOT,
        member_alias: str = "CHIVY2005",
        device: str | None = None,
    ) -> None:
        """Khởi tạo đường dẫn, danh tính logger và thiết bị chạy.

        Args:
            project_root: Gốc repository BTL03.
            member_alias: Alias thành viên lấy từ MEMBER_05_NAME.
            device: Thiết bị rõ ràng; None để BaseTrainer tự chọn CUDA/CPU.
        """
        self.project_root = Path(project_root).resolve()
        self.member_alias = member_alias
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.configs = {
            "segnet": self._load_config("configs/segnet.yaml"),
            "detection": self._load_config("configs/fcn_obj_detection.yaml"),
        }

    def _load_config(self, relative_path: str) -> dict[str, Any]:
        """Nạp YAML và giải quyết path tương đối theo gốc project."""
        path = self.project_root / relative_path
        values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values["project_root"] = str(self.project_root)
        return values

    def _path(self, value: str) -> Path:
        """Chuẩn hóa một path cấu hình thành đường dẫn tuyệt đối."""
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    @staticmethod
    def _split(dataset: torch.utils.data.Dataset, seed: int) -> tuple[Subset, Subset]:
        """Chia dữ liệu 80/20 xác định; không trộn mẫu validation vào train."""
        if len(dataset) < 2:
            raise ValueError("Cần ít nhất hai mẫu để tách train/validation.")
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator).tolist()
        train_count = max(1, min(len(indices) - 1, int(len(indices) * 0.8)))
        return Subset(dataset, indices[:train_count]), Subset(dataset, indices[train_count:])

    def _loader(self, dataset: torch.utils.data.Dataset, config: dict[str, Any], shuffle: bool) -> DataLoader:
        """Tạo DataLoader tái lập, với worker seed và generator riêng."""
        return DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=shuffle,
            num_workers=int(config.get("num_workers", 0)),
            worker_init_fn=seed_worker,
            generator=get_torch_generator(int(config["seed"])) if shuffle else None,
            pin_memory=self.device.type == "cuda",
        )

    @staticmethod
    def pixel_accuracy(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Tính tỷ lệ pixel dự đoán đúng từ logits và mask class-index."""
        return (logits.argmax(dim=1) == masks).float().mean()

    @staticmethod
    def mean_iou(logits: torch.Tensor, masks: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Tính mIoU macro; lớp không xuất hiện ở cả hai tensor bị bỏ qua."""
        predictions = logits.argmax(dim=1)
        values: list[torch.Tensor] = []
        for class_index in range(num_classes):
            predicted = predictions == class_index
            target = masks == class_index
            union = (predicted | target).sum()
            if union > 0:
                values.append((predicted & target).sum().float() / union.float())
        return torch.stack(values).mean() if values else logits.new_tensor(0.0)

    @staticmethod
    def score_accuracy(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Metric legacy score sign accuracy; không phải IoU/mAP."""
        return (scores.reshape(-1).sign() == targets.reshape(-1).sign()).float().mean()

    def _logger(self, task: str, config: dict[str, Any], exp_type: str) -> ExperimentLogger:
        """Khởi tạo logger CSV/TensorBoard với schema chuẩn và alias người thật."""
        logging = config["logging"]
        return ExperimentLogger(
            model_group=config["model_group"],
            model_name=config["model_name"],
            exp_type=exp_type,
            seed=int(config["seed"]),
            optimizer="Adam",
            learning_rate=float(config["learning_rate"]),
            batch_size=int(config["batch_size"]),
            epochs=int(config["epochs"]),
            member_alias=self.member_alias,
            project_root=self.project_root,
            csv_root=self._path(logging["csv_root"]),
            tensorboard_root=self._path(logging["tensorboard_root"]) / "Group5_Others",
        )

    def _checkpoint_callback(self, config: dict[str, Any]) -> ModelCheckpoint:
        """Tạo callback lưu file checkpoint đúng tên được yêu cầu."""
        return ModelCheckpoint(
            filepath=self._path(config["checkpoint"]["path"]),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            sync_to_drive=bool(config["checkpoint"].get("sync_to_drive", True)),
            verbose=False,
        )

    def _segnet_datasets(
        self, config: dict[str, Any], synthetic: bool
    ) -> tuple[SemanticSegmentationDataset, SemanticSegmentationDataset, bool]:
        """Nạp split legacy riêng; fallback giả chỉ được bật theo yêu cầu CLI/config."""
        data = config["data"]
        train_images, train_masks = self._path(data["train_images"]), self._path(data["train_masks"])
        val_images, val_masks = self._path(data["validation_images"]), self._path(data["validation_masks"])
        using_synthetic = False
        try:
            train = SemanticSegmentationDataset(
                train_images, train_masks, image_size=config["image_size"],
                num_classes=config["num_classes"], seed=config["seed"], normalize=config["normalize"],
            )
            validation = SemanticSegmentationDataset(
                val_images, val_masks, image_size=config["image_size"],
                num_classes=config["num_classes"], seed=config["seed"] + 1, normalize=config["normalize"],
            )
        except (FileNotFoundError, OSError, ValueError):
            if not synthetic:
                raise
            using_synthetic = True
            train = SemanticSegmentationDataset(
                image_size=config["image_size"], num_classes=config["num_classes"],
                synthetic_fallback=True, synthetic_samples=config["synthetic_samples"],
                seed=config["seed"], normalize=config["normalize"],
            )
            validation = SemanticSegmentationDataset(
                image_size=config["image_size"], num_classes=config["num_classes"],
                synthetic_fallback=True, synthetic_samples=config["synthetic_samples"],
                seed=config["seed"] + 1, normalize=config["normalize"],
            )
        return train, validation, using_synthetic

    def _detection_datasets(
        self, config: dict[str, Any], synthetic: bool
    ) -> tuple[FCNObjectDetectionDataset, FCNObjectDetectionDataset, bool]:
        """Nạp train/validation detection NPY hoặc thư mục ảnh legacy."""
        data = config["data"]
        try:
            train = FCNObjectDetectionDataset(
                self._path(data["train_images"]), self._path(data["train_labels"]),
                image_size=config["image_size"], seed=config["seed"],
            )
            validation = FCNObjectDetectionDataset(
                self._path(data["validation_images"]), self._path(data["validation_labels"]),
                image_size=config["image_size"], seed=config["seed"] + 1,
            )
            return train, validation, False
        except (FileNotFoundError, OSError, ValueError):
            root = self._path(data["root"])
            if root.is_dir():
                train = FCNObjectDetectionDataset(root, image_size=config["image_size"], seed=config["seed"])
                # Khi chỉ có gốc vehicles/non-vehicles, validation tách xác định từ chính pool.
                train_split, validation_split = self._split(train, config["seed"])
                return train_split, validation_split, False
            if not synthetic:
                raise FileNotFoundError(f"Không tìm thấy detection arrays hoặc dataset trong {root}.")
            train = FCNObjectDetectionDataset(
                image_size=config["image_size"], synthetic_fallback=True,
                synthetic_samples=config["synthetic_samples"], seed=config["seed"],
            )
            validation = FCNObjectDetectionDataset(
                image_size=config["image_size"], synthetic_fallback=True,
                synthetic_samples=config["synthetic_samples"], seed=config["seed"] + 1,
            )
            return train, validation, True

    def run_segnet(self, synthetic: bool = False, epochs: int | None = None) -> dict[str, Any]:
        """Huấn luyện SegNet, ghi pixel accuracy/mIoU, checkpoint và preview."""
        config = dict(self.configs["segnet"])
        seed_everything(int(config["seed"]))
        train_data, val_data, using_synthetic = self._segnet_datasets(
            config, synthetic or bool(config.get("synthetic_fallback", False))
        )
        if using_synthetic:
            val_data = Subset(val_data, list(range(max(1, len(val_data)))))
        train_loader = self._loader(train_data, config, shuffle=True)
        val_loader = self._loader(val_data, config, shuffle=False)
        model = SegNet(num_classes=int(config["num_classes"])).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        metric = lambda logits, masks: self.pixel_accuracy(logits, masks)
        trainer = BaseTrainer(
            model, optimizer, nn.CrossEntropyLoss(), device=self.device, metric_fn=metric,
            model_group=config["model_group"], model_name=config["model_name"], seed=config["seed"],
            config={**config, "project_root": str(self.project_root), "synthetic_data": using_synthetic},
        )
        logger = self._logger("segnet", config, "SmokeTest" if using_synthetic else config["logging"]["exp_type"])
        callback = self._checkpoint_callback(config)
        started = time.perf_counter()
        try:
            history = trainer.fit(
                train_loader, val_loader, epochs=int(epochs or config["epochs"]), logger=logger,
                callbacks=[callback],
            )
        finally:
            logger.close()
        elapsed = time.perf_counter() - started
        pixel_acc, miou = self._evaluate_segmentation(model, val_loader, int(config["num_classes"]))
        preview = self._save_segnet_preview(model, val_data, self._path(config["figure_path"]))
        return {
            "task": "segnet", "synthetic_data": using_synthetic, "history": history,
            "pixel_accuracy": pixel_acc, "miou": miou, "elapsed_sec": elapsed,
            "parameters": model.count_parameters(), "checkpoint": str(self._path(config["checkpoint"]["path"])),
            "figure": str(preview),
        }

    def _evaluate_segmentation(
        self, model: SegNet, loader: DataLoader, num_classes: int
    ) -> tuple[float, float]:
        """Đánh giá pixel accuracy và mIoU trên split validation."""
        model.eval()
        correct, total = 0, 0
        intersections = torch.zeros(num_classes, dtype=torch.float64)
        unions = torch.zeros(num_classes, dtype=torch.float64)
        with torch.inference_mode():
            for images, masks in loader:
                logits = model(images.to(self.device)).cpu()
                predictions = logits.argmax(1)
                correct += int((predictions == masks).sum())
                total += masks.numel()
                for class_index in range(num_classes):
                    predicted, target = predictions == class_index, masks == class_index
                    intersections[class_index] += (predicted & target).sum()
                    unions[class_index] += (predicted | target).sum()
        valid = unions > 0
        miou = float((intersections[valid] / unions[valid]).mean()) if valid.any() else 0.0
        return (correct / total if total else 0.0), miou

    def _save_segnet_preview(
        self, model: SegNet, dataset: torch.utils.data.Dataset, path: Path
    ) -> Path:
        """Vẽ ảnh gốc, ground truth và dự đoán cạnh nhau cho một mẫu."""
        image, mask = dataset[0]
        model.eval()
        with torch.inference_mode():
            prediction = model(image.unsqueeze(0).to(self.device)).argmax(1)[0].cpu().numpy()
        source = image[0].cpu().numpy()
        target = mask.cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(source, cmap="gray"); axes[0].set_title("Ảnh depth")
        axes[1].imshow(target, vmin=0, vmax=model.num_classes - 1, cmap="tab10"); axes[1].set_title("Ground truth")
        axes[2].imshow(prediction, vmin=0, vmax=model.num_classes - 1, cmap="tab10"); axes[2].set_title("SegNet")
        for axis in axes: axis.axis("off")
        fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=140); plt.close(fig)
        return path

    def run_detection(self, synthetic: bool = False, epochs: int | None = None) -> dict[str, Any]:
        """Huấn luyện FCN score-regression và sinh heatmap/bounding-box preview."""
        config = dict(self.configs["detection"])
        seed_everything(int(config["seed"]))
        train_data, val_data, using_synthetic = self._detection_datasets(
            config, synthetic or bool(config.get("synthetic_fallback", False))
        )
        if using_synthetic:
            train_data, val_data = self._split(train_data, config["seed"])
        train_loader = self._loader(train_data, config, shuffle=True)
        val_loader = self._loader(val_data, config, shuffle=False)
        model = FCNObjectDetector(int(config["image_size"]), int(config["window_stride"])).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        trainer = BaseTrainer(
            model, optimizer, nn.MSELoss(), device=self.device, metric_fn=self.score_accuracy,
            model_group=config["model_group"], model_name=config["model_name"], seed=config["seed"],
            config={**config, "project_root": str(self.project_root), "synthetic_data": using_synthetic},
        )
        logger = self._logger("detection", config, "SmokeTest" if using_synthetic else config["logging"]["exp_type"])
        callback = self._checkpoint_callback(config)
        started = time.perf_counter()
        try:
            history = trainer.fit(
                train_loader, val_loader, epochs=int(epochs or config["epochs"]), logger=logger,
                callbacks=[callback],
            )
        finally:
            logger.close()
        elapsed = time.perf_counter() - started
        preview, heatmap, boxes = self._save_detection_preview(model, config)
        return {
            "task": "detection", "synthetic_data": using_synthetic, "history": history,
            "score_accuracy": history["val_metric"][-1], "elapsed_sec": elapsed,
            "parameters": model.count_parameters(), "checkpoint": str(self._path(config["checkpoint"]["path"])),
            "figure": str(preview), "heatmap_shape": tuple(heatmap.shape), "boxes": boxes,
        }

    def _save_detection_preview(
        self, model: FCNObjectDetector, config: dict[str, Any]
    ) -> tuple[Path, torch.Tensor, list[tuple[int, int, int, int, float]]]:
        """Chạy ảnh lớn/Nhiều kích thước hoặc fallback tile để hiển thị heatmap."""
        image_path = self._path(config["data"]["inference_image"])
        if image_path.is_file():
            image = Image.open(image_path).convert("RGB")
            source = np.asarray(image)
        else:
            # Hình tổng hợp chỉ minh họa pipeline, không đại diện dữ liệu thật.
            generator = np.random.default_rng(int(config["seed"]))
            source = generator.integers(0, 256, (160, 192, 3), dtype=np.uint8)
        original_height, original_width = source.shape[:2]
        tensor = torch.from_numpy(np.array(source, copy=True)).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 127.5 - 1.0
        # Padding phản chiếu để ảnh nhỏ hơn cửa sổ vẫn có một score-map hợp lệ.
        pad_h = max(0, 64 - original_height); pad_w = max(0, 64 - original_width)
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        model.eval()
        with torch.inference_mode():
            heatmap, boxes = model.detect_objects(
                tensor.to(self.device), threshold=float(config["confidence_threshold"]),
                stride=int(config["window_stride"]), win_size=int(config["window_size"]),
            )
        heatmap_cpu = heatmap.detach().cpu()
        annotated = Image.fromarray(source.copy())
        draw = ImageDraw.Draw(annotated)
        for x1, y1, x2, y2, score in boxes:
            draw.rectangle((x1, y1, min(x2, original_width - 1), min(y2, original_height - 1)), outline="lime", width=2)
            draw.text((x1, y1), f"vehicle {score:.2f}", fill="lime")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(source); axes[0].set_title("Ảnh gốc")
        axes[1].imshow(heatmap_cpu[0, 0].numpy(), cmap="inferno", vmin=0, vmax=1); axes[1].set_title("Heatmap xe")
        axes[2].imshow(annotated); axes[2].set_title(f"Phát hiện ({len(boxes)} boxes)")
        for axis in axes: axis.axis("off")
        fig.tight_layout(); path = self._path(config["figure_path"]); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140); plt.close(fig)
        return path, heatmap_cpu, boxes

    def run(self, task: str = "all", synthetic: bool = False, epochs: int | None = None) -> list[dict[str, Any]]:
        """Thực thi task SegNet, detection hoặc cả hai theo CLI."""
        results: list[dict[str, Any]] = []
        if task in {"segnet", "all"}:
            results.append(self.run_segnet(synthetic=synthetic, epochs=epochs))
        if task in {"detection", "all"}:
            results.append(self.run_detection(synthetic=synthetic, epochs=epochs))
        return results

    def package_artifacts(self, destination: str | Path) -> Path:
        """Nén checkpoint, CSV logs và figure nhẹ để tải từ Kaggle."""
        archive = Path(destination)
        archive.parent.mkdir(parents=True, exist_ok=True)
        candidates = (
            self.project_root / "outputs/checkpoints/05_others",
            self.project_root / "outputs/figures/t0210_segnet_segmentation_preview.png",
            self.project_root / "outputs/figures/t0211_fcn_heatmap_detection.png",
            self.project_root / "experiments/csv_logs",
        )
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for candidate in candidates:
                files = [candidate] if candidate.is_file() else candidate.rglob("*") if candidate.is_dir() else []
                for file_path in files:
                    if file_path.is_file() and file_path.suffix.lower() in {".pt", ".pth", ".png", ".csv"}:
                        bundle.write(file_path, file_path.relative_to(self.project_root))
        return archive


def main() -> None:
    """Đọc đối số CLI và chạy pipeline nhóm 5."""
    parser = argparse.ArgumentParser(description="Huấn luyện SegNet và FCN object detection nhóm 5.")
    parser.add_argument("--task", choices=("segnet", "detection", "all"), default="all")
    parser.add_argument("--segnet-config", default="configs/segnet.yaml")
    parser.add_argument("--detection-config", default="configs/fcn_obj_detection.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Opt-in synthetic fallback chỉ để kiểm tra pipeline.")
    parser.add_argument("--epochs", type=int, help="Ghi đè số epoch; hữu ích cho smoke-test.")
    parser.add_argument("--member-alias", default="CHIVY2005")
    parser.add_argument("--device", default=None)
    parser.add_argument("--package", default=None, help="Tạo ZIP artifacts sau khi chạy.")
    args = parser.parse_args()
    runner = Group5OthersRunner(member_alias=args.member_alias, device=args.device)
    if args.segnet_config != "configs/segnet.yaml":
        runner.configs["segnet"] = runner._load_config(args.segnet_config)
    if args.detection_config != "configs/fcn_obj_detection.yaml":
        runner.configs["detection"] = runner._load_config(args.detection_config)
    for result in runner.run(args.task, synthetic=args.synthetic, epochs=args.epochs):
        print(result)
    if args.package:
        print(f"Artifacts: {runner.package_artifacts(args.package)}")


if __name__ == "__main__":
    main()
