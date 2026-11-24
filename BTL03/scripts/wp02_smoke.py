"""Kiểm toán parity và smoke train một epoch cho mười cấu hình WP02."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from src.datasets import (
    BivariateNormalDataset,
    FCNObjectDetectionDataset,
    FEREmotionDataset,
    MNISTDataset,
    SemanticSegmentationDataset,
    SequentialMNISTDataset,
    WikiGenderDataset,
)
from src.engine import BaseTrainer, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.logger import ExperimentLogger
from src.models import (
    EmotionCNN,
    FCNObjectDetector,
    FuncApxmMLP,
    HumanGenderCNN,
    MNISTCNN,
    MNISTCNNLSTM,
    MNISTMLP,
    SegNet,
)
from src.utils import get_torch_generator, seed_everything
from scripts.run_group5_others import Group5OthersRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EXP-01 chưa lưu model.summary; B=16 trong YAML có 4B+1=65 tham số.
WP02_CASES: tuple[dict[str, Any], ...] = (
    {"key": "func_apxm_mlp", "name": "FuncApxmMLP", "group": "01_mlp", "params": 65, "batch": 8, "input": (2,), "target": (1,)},
    {"key": "mnist_mlp", "name": "MNISTMLP", "group": "01_mlp", "params": 669_706, "batch": 2, "input": (1, 28, 28), "target": ()},
    {"key": "mnist_cnn", "name": "MNISTCNN", "group": "02_cnn_mlp", "params": 1_817_610, "batch": 2, "input": (1, 28, 28), "target": ()},
    {"key": "human_gender_cnn", "name": "HumanGenderCNN", "group": "02_cnn_mlp", "params": 4_063_570, "batch": 2, "input": (1, 64, 64), "target": ()},
    {"key": "emotion_cnn", "name": "EmotionCNN", "group": "02_cnn_mlp", "params": 19_028_287, "batch": 2, "input": (1, 48, 48), "target": ()},
    {"key": "mnist_cnn_lstm", "name": "MNISTCNNLSTM", "group": "03_cnn_lstm", "params": 1_747_073, "batch": 2, "input": (8, 1, 28, 28), "target": (1,)},
    {"key": "gender_cnn_callbacks", "name": "HumanGenderCNN", "group": "04_training_utilities", "params": 4_063_570, "batch": 2, "input": (1, 64, 64), "target": ()},
    {"key": "emotion_cnn_callbacks", "name": "EmotionCNN", "group": "04_training_utilities", "params": 19_028_287, "batch": 2, "input": (1, 48, 48), "target": ()},
    {"key": "segnet", "name": "SegNet", "group": "05_others", "params": 29_458_211, "batch": 2, "input": (1, 32, 32), "target": (32, 32)},
    {"key": "fcn_detector", "name": "FCNObjectDetector", "group": "05_others", "params": 83_367, "batch": 2, "input": (3, 64, 64), "target": (1, 1, 1)},
)


class WP02SmokeHarness:
    """Dựng dữ liệu schema-đúng và train từng cấu hình trong sandbox tạm."""

    def __init__(self, project_root: Path = PROJECT_ROOT, seed: int = 42) -> None:
        """Khởi tạo factory gắn với root repository và seed thống nhất."""
        self.project_root = Path(project_root).resolve()
        self.seed = int(seed)
        self.results: list[dict[str, Any]] = []

    def model(self, case: dict[str, Any]) -> nn.Module:
        """Khởi tạo model theo registry của mười task/cấu hình WP02."""
        factories = {
            "func_apxm_mlp": lambda: FuncApxmMLP(hidden_units=16),
            "mnist_mlp": MNISTMLP,
            "mnist_cnn": MNISTCNN,
            "human_gender_cnn": HumanGenderCNN,
            "emotion_cnn": EmotionCNN,
            "mnist_cnn_lstm": MNISTCNNLSTM,
            "gender_cnn_callbacks": HumanGenderCNN,
            "emotion_cnn_callbacks": EmotionCNN,
            "segnet": SegNet,
            "fcn_detector": FCNObjectDetector,
        }
        model = factories[case["key"]]()
        initializer = getattr(model, "init_weights_keras_style", None)
        if callable(initializer):
            initializer()
        return model

    def loaders(self, case: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
        """Tạo train/validation batches đúng layout, dtype và loss target."""
        key, batch = case["key"], int(case["batch"])
        if key == "func_apxm_mlp":
            train = BivariateNormalDataset(split="train", sample_count=16, seed=self.seed)
            valid = BivariateNormalDataset(split="test", seed=self.seed)
        elif key in {"mnist_mlp", "mnist_cnn"}:
            gen = torch.Generator().manual_seed(self.seed)
            images = torch.randint(0, 256, (batch, 28, 28), generator=gen, dtype=torch.uint8).numpy()
            labels = torch.arange(batch).numpy() % 10
            train = MNISTDataset(images, labels)
            valid = MNISTDataset(images.copy(), labels.copy())
        elif key in {"human_gender_cnn", "gender_cnn_callbacks"}:
            images = np.random.default_rng(self.seed).integers(0, 256, (batch, 64, 64, 1), dtype=np.uint8)
            labels = np.arange(batch, dtype=np.int64) % 2
            train = WikiGenderDataset(images, labels, apply_legacy_rotation=True, normalization="auto")
            valid = WikiGenderDataset(images.copy(), labels.copy(), apply_legacy_rotation=True, normalization="auto")
        elif key in {"emotion_cnn", "emotion_cnn_callbacks"}:
            images = np.random.default_rng(self.seed).random((batch, 48, 48), dtype=np.float32)
            labels = np.arange(batch, dtype=np.int64) % 7
            train = FEREmotionDataset(images, labels, normalization="preserve")
            valid = FEREmotionDataset(images.copy(), labels.copy(), normalization="preserve")
        elif key == "mnist_cnn_lstm":
            rng = np.random.default_rng(self.seed)
            images = rng.integers(0, 256, (batch * 2, 28, 28), dtype=np.uint8)
            labels = np.arange(batch * 2, dtype=np.int64) % 10
            train = SequentialMNISTDataset(images, labels, seq_len=8, seed=self.seed, is_train=True)
            valid = SequentialMNISTDataset(images.copy(), labels.copy(), seq_len=8, seed=self.seed + 1, is_train=False)
        elif key == "segnet":
            train = SemanticSegmentationDataset(
                image_size=32, num_classes=7, synthetic_fallback=True,
                synthetic_samples=batch, seed=self.seed,
            )
            valid = SemanticSegmentationDataset(
                image_size=32, num_classes=7, synthetic_fallback=True,
                synthetic_samples=batch, seed=self.seed + 1,
            )
        elif key == "fcn_detector":
            train = FCNObjectDetectionDataset(
                image_size=64, synthetic_fallback=True, synthetic_samples=batch, seed=self.seed,
            )
            valid = FCNObjectDetectionDataset(
                image_size=64, synthetic_fallback=True, synthetic_samples=batch, seed=self.seed + 1,
            )
        else:
            raise KeyError(f"Không có dataset smoke cho {key}.")
        return (
            DataLoader(train, batch_size=batch, shuffle=False),
            DataLoader(valid, batch_size=batch, shuffle=False),
        )

    @staticmethod
    def objective(case: dict[str, Any]) -> tuple[nn.Module, Any]:
        """Chọn criterion và metric đúng loại bài toán legacy."""
        key = case["key"]
        if key == "func_apxm_mlp":
            return nn.MSELoss(), lambda output, target: nn.functional.mse_loss(output, target)
        if key == "mnist_cnn_lstm":
            return nn.L1Loss(), lambda output, target: (output - target).abs().mean()
        if key == "segnet":
            return nn.CrossEntropyLoss(), lambda output, target: Group5OthersRunner.pixel_accuracy(output, target)
        if key == "fcn_detector":
            return nn.MSELoss(), lambda output, target: (output.reshape(-1).sign() == target.reshape(-1).sign()).float().mean()
        return nn.CrossEntropyLoss(), lambda output, target: (output.argmax(1) == target).float().mean()

    def run_case(self, case: dict[str, Any], temp_root: Path | None = None) -> dict[str, Any]:
        """Kiểm parity/hooks rồi train một epoch hoàn toàn trong thư mục tạm.

        Logger được thử với CSV/TensorBoard riêng cho từng case. Nếu môi trường
        logger không tương thích, trainer vẫn chạy để tách riêng bằng chứng
        training khỏi blocker TensorBoard; trạng thái logger được lưu tường minh.
        """
        if temp_root is None:
            with tempfile.TemporaryDirectory(prefix=f"wp02_{case['key']}_") as temporary:
                return self._run_case(case, Path(temporary))
        return self._run_case(case, Path(temp_root).resolve())

    def _run_case(self, case: dict[str, Any], smoke_root: Path) -> dict[str, Any]:
        """Thực thi một case với mọi output được cô lập dưới temp_root."""
        seed_everything(self.seed)
        started = time.perf_counter()
        model = self.model(case)
        counts = model.count_parameters()
        actual = int(counts["total_params"])
        if actual != int(case["params"]):
            raise AssertionError(f"{case['name']}: PyTorch={actual}, Keras={case['params']}.")
        if isinstance(model, MNISTCNNLSTM):
            assert model.lstm1.batch_first and model.lstm2.batch_first
        train_loader, valid_loader = self.loaders(case)
        criterion, metric = self.objective(case)
        if case["key"] == "func_apxm_mlp":
            optimizer = SGD(model.parameters(), lr=0.01)
        elif case["key"] in {"mnist_cnn_lstm", "mnist_mlp", "mnist_cnn"}:
            optimizer = SGD(model.parameters(), lr=0.01)
        else:
            optimizer = Adam(model.parameters(), lr=0.001)
        smoke_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = smoke_root / "outputs/checkpoints" / case["group"] / f"{case['key']}_best.pt"
        callbacks: list[Any] = []
        if case["group"] == "04_training_utilities":
            callbacks = [
                ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min", verbose=False, sync_to_drive=False),
                ReduceLROnPlateau(optimizer, monitor="val_loss", patience=1, verbose=False),
                EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=False),
            ]
        elif case["group"] == "05_others":
            callbacks = [ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min", verbose=False, sync_to_drive=False)]
        logger_status = "NOT_CONFIGURED"
        logger: ExperimentLogger | None = None
        try:
            logger = ExperimentLogger(
                model_group=case["group"],
                model_name=case["name"],
                exp_type=f"SmokeTest_{case['key']}",
                seed=self.seed,
                optimizer="SGD" if isinstance(optimizer, SGD) else "Adam",
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                batch_size=int(case["batch"]),
                epochs=1,
                member_alias="Duy-DS",
                project_root=smoke_root,
                csv_root=smoke_root / "experiments/csv_logs",
                tensorboard_root=smoke_root / "experiments/tensorboard",
            )
            logger_status = "PASS"
        except RuntimeError as error:
            # SummaryWriter không tương thích trong runtime hiện tại. Không
            # chuyển sang logger giả; báo BLOCKED và vẫn kiểm tra BaseTrainer.
            logger_status = f"BLOCKED: {error}"

        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
            metric_fn=metric,
            model_group=case["group"],
            model_name=case["name"],
            seed=self.seed,
            config={"project_root": str(smoke_root), "smoke_test": True, "task": case["key"]},
        )
        try:
            history = trainer.fit(
                train_loader, valid_loader, epochs=1, logger=logger, callbacks=callbacks
            )
        finally:
            if logger is not None:
                logger.close()
        if len(history["train_loss"]) != 1 or len(history["val_loss"]) != 1:
            raise AssertionError(f"{case['key']}: trainer did not complete one epoch.")

        # Feature-hook gates required by the integration task.
        if case["key"] == "mnist_mlp":
            model.register_feature_hooks(["relu2"])
            images, _ = next(iter(valid_loader))
            _, features = model.extract_features(images)
            assert features["relu2"].shape == (len(images), 512)
            model.clear_hooks()
        elif case["key"] == "mnist_cnn":
            model.register_feature_hooks(["pool3"])
            images, _ = next(iter(valid_loader))
            _, features = model.extract_features(images)
            assert features["pool3"].shape[1:] == (256, 3, 3)
            model.clear_hooks()
        elif case["key"] == "fcn_detector":
            images, _ = next(iter(valid_loader))
            with torch.inference_mode():
                heatmap, boxes = model.detect_objects(
                    images, threshold=0.0, stride=8, win_size=64
                )
            assert heatmap.ndim == 4 and heatmap.shape[0] == len(images)
            assert isinstance(boxes, list)

        if callbacks:
            assert callbacks[0].best_epoch == 1
            assert checkpoint_path.is_file()
        elapsed = time.perf_counter() - started
        result = {
            "model": case["name"],
            "case": case["key"],
            "group": case["group"],
            "pytorch_params": actual,
            "keras_params": case["params"],
            "delta": actual - case["params"],
            "status": "PASS",
            "logger_status": logger_status,
            "seconds": elapsed,
            "history": history,
            "checkpoint": str(checkpoint_path) if callbacks else None,
        }
        self.results.append(result)
        return result

    def print_summary(self) -> None:
        """In bảng parity và thời gian đã chạy ra stdout."""
        print("| Model | Group | PyTorch params | Keras params | Delta | 1-epoch smoke | Logger | Seconds |")
        print("|---|---|---:|---:|---:|---|---|---:|")
        for result in self.results:
            pytorch_params = result["pytorch_params"]
            keras_params = result["keras_params"]
            if isinstance(pytorch_params, int):
                pytorch_params = f"{pytorch_params:,}"
            if isinstance(keras_params, int):
                keras_params = f"{keras_params:,}"
            print(
                f"| {result['model']} (`{result['case']}`) | {result['group']} | "
                f"{pytorch_params} | {keras_params} | "
                f"{result['delta']} | {result['status']} | {result.get('logger_status', 'N/A')} | "
                f"{result['seconds']:.2f} |"
            )


def main(argv: list[str] | None = None, output_json: Path | None = None) -> int:
    """Chạy mười cấu hình, in bảng và trả mã lỗi nếu model hoặc epoch lỗi.

    Args:
        argv: Dành riêng cho lớp CLI thống nhất, không tự parse đối số ở đây.
        output_json: Đường dẫn tùy chọn để xuất bảng kết quả dưới dạng JSON.
    """
    del argv
    harness = WP02SmokeHarness()
    failures: list[tuple[str, str]] = []
    for case in WP02_CASES:
        try:
            with tempfile.TemporaryDirectory(prefix=f"wp02_{case['key']}_") as temporary:
                harness.run_case(case, Path(temporary))
        except Exception as error:  # Bảng phải ghi FAIL rồi tiếp tục các model còn lại.
            failures.append((case["key"], repr(error)))
            harness.results.append({
                "model": case["name"], "case": case["key"], "group": case["group"],
                "pytorch_params": "ERROR", "keras_params": case["params"], "delta": "N/A",
                "status": "FAIL", "logger_status": "N/A", "seconds": 0.0,
            })
    harness.print_summary()
    summary = {
        "seed": 42,
        "overall_status": "FAIL" if failures else (
            "BLOCKED_LOGGER" if any(
                str(result.get("logger_status", "")).startswith("BLOCKED:")
                for result in harness.results
            ) else "PASS"
        ),
        "passed_cases": sum(result["status"] == "PASS" for result in harness.results),
        "total_cases": len(WP02_CASES),
        "logger_blocked_cases": sum(
            str(result.get("logger_status", "")).startswith("BLOCKED:")
            for result in harness.results
        ),
        "results": harness.results,
        "failures": failures,
    }
    if output_json is not None:
        destination = Path(output_json).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Smoke summary JSON: {destination}")
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
        return 1
    if summary["logger_blocked_cases"]:
        print("Training PASS; TensorBoard/CSV logger is BLOCKED in this runtime.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
