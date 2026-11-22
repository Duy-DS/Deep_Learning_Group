"""Chạy thí nghiệm Gaussian MLP và phân loại MNIST của Nhóm 1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset

# Khi chạy ``python scripts/run_group1_mlp.py``, thêm gốc repo để import src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_data import DataBootstrapper
from src.datasets import BivariateNormalDataset, MNISTDataset
from src.engine import BaseTrainer
from src.logger import CheckpointManager, ExperimentLogger
from src.models import FuncApxmMLP, MNISTMLP
from src.utils import get_torch_generator, seed_everything, seed_worker


class Group1MLPRunner:
    """Nối cấu hình, Dataset, mô hình, trainer và log cho Nhóm 1."""

    def __init__(self, project_root: Path, member_alias: str = "Duy-DS") -> None:
        """Khởi tạo runner từ gốc repo và cấu hình YAML."""
        self.project_root = project_root.resolve()
        self.member_alias = member_alias
        self.func_config = yaml.safe_load(
            (self.project_root / "configs/mlp_func_apxm.yaml").read_text(encoding="utf-8")
        )
        self.mnist_config = yaml.safe_load(
            (self.project_root / "configs/mlp_mnist.yaml").read_text(encoding="utf-8")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sweep_csv_path = self.project_root / self.func_config["paths"]["sweep_csv"]

    def _logger(self, **kwargs: Any) -> ExperimentLogger:
        """Tạo logger theo cây chuẩn; model_name riêng tránh va chạm HParams."""
        return ExperimentLogger(
            member_alias=self.member_alias,
            project_root=self.project_root,
            **kwargs,
        )

    class _DeferredFinishLogger:
        """Chuyển tiếp log epoch nhưng hoãn chốt run tới khi có test metric."""

        def __init__(self, logger: ExperimentLogger) -> None:
            self.logger = logger
            self.member_alias = logger.member_alias

        def log_epoch(self, **kwargs: Any) -> None:
            """Ghi epoch ngay qua logger chuẩn."""
            self.logger.log_epoch(**kwargs)

        def finish_run(self, **kwargs: Any) -> Path:
            """Giữ API BaseTrainer và chờ runner đánh giá test độc lập."""
            self.pending_summary = kwargs
            return self.logger.master_csv_path

    def run_gaussian_sweeps(
        self, sweep_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """Chạy sweep A/B/C, ghi epoch metrics và tổng hợp CSV nhẹ."""
        config = self.func_config
        seed_everything(int(config["seed"]))
        all_train = BivariateNormalDataset(split="train", sample_count=1000, seed=42)
        all_test = BivariateNormalDataset(split="test")
        scenarios = [
            ("A", value, int(config["default_B"]), int(config["default_C"]))
            for value in config["sweeps"]["A"]
        ]
        scenarios += [
            ("B", int(config["default_A"]), value, int(config["default_C"]))
            for value in config["sweeps"]["B"]
        ]
        scenarios += [
            # Notebook legacy giữ A=800 và B=128 cho sweep số epoch C.
            ("C", int(config["default_A"]), int(config["default_B_C"]), value)
            for value in config["sweeps"]["C"]
        ]
        rows: list[dict[str, Any]] = []
        self._sweep_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._sweep_csv_path.exists():
            self._sweep_csv_path.write_text(
                "Sweep,A,B,C,Train_MSE_Grid_1024,Test_MSE_Grid_100,Train_Time_Sec\n",
                encoding="utf-8-sig",
            )
        for sweep_name, sample_count, hidden_units, epochs in scenarios:
            if sweep_filter and not (
                sweep_name == sweep_filter
                or (sweep_filter == "C10" and sweep_name == "C" and epochs == 10)
            ):
                continue
            key = (sweep_name, sample_count, hidden_units, epochs)
            existing_run = self._find_complete_sweep_run(*key)
            if existing_run is not None and self._summary_matches_run(key, existing_run):
                print(f"Skip verified sweep: {key}")
                continue
            if existing_run is not None:
                self._append_sweep_csv(
                    self._sweep_csv_path,
                    {
                        "Sweep": sweep_name,
                        "A": sample_count,
                        "B": hidden_units,
                        "C": epochs,
                        "Train_MSE_Grid_1024": float(existing_run["Train_Loss"]),
                        "Test_MSE_Grid_100": float(existing_run["Test_Metric"]),
                        "Train_Time_Sec": float(existing_run["Train_Time_Sec"]),
                        "Run_ID": existing_run["Run_ID"],
                    },
                )
                print(f"Reconciled summary from verified master run: {key}")
                continue
            seed_everything(42)
            subset = BivariateNormalDataset(
                split="train", sample_count=sample_count, seed=42
            )
            train_loader = DataLoader(
                subset,
                batch_size=int(config["batch_size"]),
                shuffle=True,
                generator=get_torch_generator(42),
                worker_init_fn=seed_worker,
                num_workers=0,
            )
            model = FuncApxmMLP(hidden_units=hidden_units)
            model.init_weights_keras_style()
            optimizer = SGD(model.parameters(), lr=float(config["learning_rate"]))
            criterion = nn.MSELoss()
            safe_name = f"FuncApxmMLP_{sweep_name}{sample_count}_B{hidden_units}_C{epochs}"
            run_token = uuid.uuid4().hex[:10]
            # Tên model giữ cấu hình chuẩn để kiểm tra provenance; token duy nhất
            # nằm trong Exp_Type, từ đó Run_ID không va chạm trong cùng giây.
            logger_model_name = f"{safe_name}_{run_token}"
            logger = self._logger(
                model_group="Group1_MLP",
                model_name=logger_model_name,
                exp_type=f"T0201_{sweep_name}_{run_token}",
                seed=42,
                optimizer="SGD",
                learning_rate=float(config["learning_rate"]),
                batch_size=int(config["batch_size"]),
                epochs=epochs,
            )
            started = time.perf_counter()
            model.to(self.device)
            for epoch in range(1, epochs + 1):
                epoch_started = time.perf_counter()
                model.train()
                sum_loss = 0.0
                seen = 0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    optimizer.step()
                    sum_loss += float(loss.detach()) * len(inputs)
                    seen += len(inputs)
                train_loss = sum_loss / seen
                train_mse = self._evaluate_mse(model, all_train.full_inputs, all_train.full_targets)
                test_mse = self._evaluate_mse(model, all_test.inputs, all_test.targets)
                epoch_time = time.perf_counter() - epoch_started
                logger.log_epoch(
                    epoch,
                    train_loss,
                    test_mse,
                    train_metric=train_mse,
                    val_metric=test_mse,
                    learning_rate=float(config["learning_rate"]),
                    epoch_time_sec=epoch_time,
                    metric_name="MSE",
                )
            elapsed = time.perf_counter() - started
            logger.finish_run(
                best_epoch=epochs,
                train_loss=train_mse,
                val_loss=test_mse,
                train_metric=train_mse,
                val_metric=test_mse,
                test_metric=test_mse,
                train_time_sec=elapsed,
                metric_name="MSE",
            )
            logger.close()
            # Sweep logs use the final epoch; explicitly report matching losses.
            final_train_mse = self._evaluate_mse(model, all_train.full_inputs, all_train.full_targets)
            final_test_mse = self._evaluate_mse(model, all_test.inputs, all_test.targets)
            completed_run = self._find_complete_sweep_run(
                *key,
                expected_run_id=logger.run_id,
                expected_model_name=logger_model_name,
            )
            if completed_run is None or completed_run["Run_ID"] != logger.run_id:
                raise RuntimeError(
                    f"Sweep {key} không có Run_ID/master row và epoch trace đầy đủ; "
                    "không chấp nhận kết quả vào CSV tổng hợp."
                )
            rows.append(
                {
                    "Sweep": sweep_name,
                    "A": sample_count,
                    "B": hidden_units,
                    "C": epochs,
                    "Train_MSE_Grid_1024": final_train_mse,
                    "Test_MSE_Grid_100": final_test_mse,
                    "Train_Time_Sec": elapsed,
                    "Run_ID": logger.run_id,
                }
            )
            self._append_sweep_csv(self._sweep_csv_path, rows[-1])
            print(json.dumps(rows[-1], ensure_ascii=False))
        all_rows = []
        with self._sweep_csv_path.open(encoding="utf-8-sig", newline="") as stream:
            all_rows = list(csv.DictReader(stream))
        if all_rows:
            plot_rows = [
                {**row, "A": int(row["A"]), "B": int(row["B"]), "C": int(row["C"]),
                 "Train_MSE_Grid_1024": float(row["Train_MSE_Grid_1024"]),
                 "Test_MSE_Grid_100": float(row["Test_MSE_Grid_100"])}
                for row in all_rows
            ]
            self._plot_sweep_results(plot_rows)
        return rows

    def _plot_sweep_results(self, rows: list[dict[str, Any]]) -> Path:
        """Vẽ MSE train/test cho từng trục khảo sát A, B và C."""
        destination = self.project_root / "outputs/figures/t0201_func_apxm_abc_sweep.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, name, label in zip(
            axes, ("A", "B", "C"), ("Số mẫu A", "Nút ẩn B", "Epoch C")
        ):
            subset = [row for row in rows if row["Sweep"] == name]
            axis.plot(
                [row[name] for row in subset],
                [row["Train_MSE_Grid_1024"] for row in subset],
                marker="o",
                label="Train",
            )
            axis.plot(
                [row[name] for row in subset],
                [row["Test_MSE_Grid_100"] for row in subset],
                marker="s",
                label="Test",
            )
            axis.set_xlabel(label)
            axis.set_ylabel("MSE")
            axis.set_title(f"Khảo sát {name}")
            axis.grid(True, alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return destination

    @staticmethod
    def _append_sweep_csv(path: Path, row: dict[str, Any]) -> None:
        """Ghi/đổi kết quả cùng cấu hình mà không tạo hàng trùng."""
        rows: list[dict[str, Any]] = []
        fields = list(row)
        if path.exists() and path.stat().st_size:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = list(reader.fieldnames or fields)
                rows = list(reader)
        fields = [field for field in fields if field != "Run_ID"] + ["Run_ID"]
        key = tuple(str(row[field]) for field in ("Sweep", "A", "B", "C"))
        replaced = False
        updated: list[dict[str, Any]] = []
        for existing in rows:
            existing.setdefault("Run_ID", "")
            existing_key = tuple(
                str(existing[field]) for field in ("Sweep", "A", "B", "C")
            )
            if existing_key == key:
                if not replaced:
                    updated.append(row)
                    replaced = True
            else:
                updated.append(existing)
        if not replaced:
            updated.append(row)
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(updated)

    def _find_complete_sweep_run(
        self,
        sweep: str,
        sample_count: int,
        hidden_units: int,
        epochs: int,
        expected_run_id: str | None = None,
        expected_model_name: str | None = None,
    ) -> dict[str, str] | None:
        """Tìm master row có epoch trace khớp đầy đủ cấu hình sweep.

        Master CSV là nhật ký append-only nên một dòng summary đơn lẻ chưa đủ
        chứng minh run hoàn chỉnh. Hàm còn xác thực model name, số epoch, metric
        hữu hạn và epoch CSV có đúng số hàng mong đợi.
        """
        master_path = self.project_root / "experiments/csv_logs/master_experiment_runs.csv"
        if not master_path.is_file():
            return None
        if sweep == "A":
            expected_model = f"FuncApxmMLP_A{sample_count}_B{hidden_units}_C{epochs}"
            matcher = lambda name: name == expected_model or name.startswith(expected_model + "_")
        elif sweep == "B":
            expected_model = f"FuncApxmMLP_B{sample_count}_B{hidden_units}_C{epochs}"
            matcher = lambda name: name == expected_model or name.startswith(expected_model + "_")
        else:
            expected_model = f"FuncApxmMLP_C{sample_count}_B{hidden_units}_C{epochs}"
            matcher = lambda name: name == expected_model or name.startswith(expected_model + "_")

        with master_path.open(encoding="utf-8-sig", newline="") as stream:
            master_rows = list(csv.DictReader(stream))
        for master_row in reversed(master_rows):
            if expected_run_id and master_row.get("Run_ID") != expected_run_id:
                continue
            if expected_model_name and master_row.get("Model_Name") != expected_model_name:
                continue
            if not matcher(master_row.get("Model_Name", "")):
                continue
            if master_row.get("Member_Alias") != self.member_alias:
                continue
            if int(master_row.get("Epochs", 0)) != epochs:
                continue
            if int(master_row.get("Best_Epoch", 0)) != epochs:
                continue
            try:
                test_metric = float(master_row["Test_Metric"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(test_metric):
                continue
            epoch_path = self.project_root / "experiments/csv_logs" / (
                master_row["Run_ID"] + "_epochs.csv"
            )
            if not epoch_path.is_file():
                continue
            with epoch_path.open(encoding="utf-8-sig", newline="") as epoch_stream:
                epoch_rows = list(csv.DictReader(epoch_stream))
            if len(epoch_rows) != epochs:
                continue
            if [int(row["Epoch"]) for row in epoch_rows] != list(range(1, epochs + 1)):
                continue
            if any(
                not np.isfinite(float(row[column]))
                for row in epoch_rows
                for column in ("Train_Loss", "Val_Loss")
            ):
                continue
            return master_row
        return None

    def _summary_matches_run(
        self, key: tuple[str, int, int, int], master_row: dict[str, str]
    ) -> bool:
        """Xác nhận CSV tổng hợp đang trỏ đúng run đã kiểm chứng."""
        if not self._sweep_csv_path.is_file():
            return False
        with self._sweep_csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        matching = [
            row
            for row in rows
            if (
                row["Sweep"],
                int(row["A"]),
                int(row["B"]),
                int(row["C"]),
            ) == key
            and row.get("Run_ID") == master_row.get("Run_ID")
        ]
        return any(
            row.get("Run_ID") == master_row.get("Run_ID")
            and np.isclose(
                float(row["Train_MSE_Grid_1024"]),
                float(master_row["Train_Loss"]),
                rtol=1e-4,
                atol=5e-7,
            )
            and np.isclose(
                float(row["Test_MSE_Grid_100"]),
                float(master_row["Test_Metric"]),
                rtol=1e-4,
                atol=5e-7,
            )
            for row in matching
        )

    @staticmethod
    def _evaluate_mse(
        model: FuncApxmMLP, inputs: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Tính MSE theo toàn bộ tensor sau khi chuyển model sang eval."""
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        with torch.inference_mode():
            loss = nn.functional.mse_loss(model(inputs.to(device)), targets.to(device))
        model.train(was_training)
        return float(loss.cpu())

    def run_mnist(self, epochs: int | None = None, download: bool = True) -> dict[str, Any]:
        """Tải MNIST thật, huấn luyện, checkpoint, đánh giá và trích fc2."""
        config = self.mnist_config
        seed_everything(int(config["seed"]))
        data_root = self.project_root / config["data_root"]
        arrays = self._load_mnist_arrays(data_root, download)
        train_data = MNISTDataset(arrays["train_images"], arrays["train_labels"])
        test_data = MNISTDataset(arrays["test_images"], arrays["test_labels"])
        val_count = int(len(train_data) * float(config["validation_fraction"]))
        generator = get_torch_generator(42)
        permutation = torch.randperm(len(train_data), generator=generator).tolist()
        val_indices = permutation[:val_count]
        train_indices = permutation[val_count:]
        train_subset, val_subset = Subset(train_data, train_indices), Subset(train_data, val_indices)
        batch_size = int(config["batch_size"])
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            generator=get_torch_generator(42),
            worker_init_fn=seed_worker,
            num_workers=int(config["num_workers"]),
        )
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = MNISTMLP(dropout=float(config["dropout"]))
        model.init_weights_keras_style()
        optimizer = SGD(model.parameters(), lr=float(config["learning_rate"]))
        run_epochs = int(epochs if epochs is not None else config["epochs"])
        criterion = nn.CrossEntropyLoss()
        trainer = BaseTrainer(
            model,
            optimizer,
            criterion,
            device=self.device,
            metric_fn=lambda logits, labels: (logits.argmax(1) == labels).float().mean(),
            model_group="01_mlp",
            model_name="MNISTMLP",
            seed=42,
            member_alias=self.member_alias,
            config={**config, "epochs": run_epochs},
        )
        training_manager = CheckpointManager(
            project_root=self.project_root,
            checkpoint_root=(
                self.project_root
                / config["checkpoint_root"]
                / f"_training_{uuid.uuid4().hex[:10]}"
            ),
            drive_root=self._configured_drive_root(),
        )
        logger = self._create_history_logger(config, run_epochs, batch_size)
        try:
            history = trainer.fit(
                train_loader,
                val_loader,
                run_epochs,
                logger=self._DeferredFinishLogger(logger),
                checkpoint_manager=training_manager,
            )
        except Exception:
            logger.close()
            raise
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        training_checkpoint = (
            training_manager.checkpoint_root / "01_mlp" / "MNISTMLP_best.pth"
        )
        payload = CheckpointManager.load(
            training_checkpoint, model=model, map_location=self.device
        )
        model.eval()
        train_accuracy = self._evaluate_accuracy(model, DataLoader(train_subset, batch_size=batch_size))
        val_accuracy = self._evaluate_accuracy(model, val_loader)
        test_accuracy = self._evaluate_accuracy(model, test_loader)
        manager = CheckpointManager(
            project_root=self.project_root,
            checkpoint_root=self.project_root / config["checkpoint_root"],
            drive_root=self._configured_drive_root(),
        )
        try:
            logger.finish_run(
                best_epoch=best_epoch,
                train_loss=history["train_loss"][best_epoch - 1],
                val_loss=history["val_loss"][best_epoch - 1],
                train_metric=history["train_metric"][best_epoch - 1],
                val_metric=history["val_metric"][best_epoch - 1],
                test_metric=test_accuracy,
                train_time_sec=sum(history["epoch_time_sec"]),
            )
        finally:
            logger.close()
        checkpoint = manager.save(
            model=model,
            model_group="01_mlp",
            model_name="MNISTMLP",
            epoch=int(payload["epoch"]),
            optimizer=optimizer,
            best_metric=float(payload["best_metric"]),
            seed=42,
            config={**config, "epochs": run_epochs},
            filename=str(config["checkpoint_name"]),
        )
        feature_path = self.project_root / config["feature_output"]
        features, labels = self._extract_features(model, test_loader)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_path,
            features=features.numpy(),
            labels=labels.numpy(),
        )
        tensorboard_root = self.project_root / config["tensorboard_root"] / "Group1_MLP"
        run_dirs = sorted(tensorboard_root.glob("MNISTMLP/T0202_MNISTMLP_s42_*"))
        if run_dirs:
            metrics_writer = torch.utils.tensorboard.SummaryWriter(str(run_dirs[-1]))
            metrics_writer.add_scalar("Metric/Test/Accuracy", test_accuracy, best_epoch)
            metrics_writer.flush()
            metrics_writer.close()
        result = {
            "parameters": model.count_parameters()["total_params"],
            "best_epoch": best_epoch,
            "train_subset_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "test_samples": len(test_data),
            "feature_shape": list(features.shape),
            "checkpoint": str(checkpoint.relative_to(self.project_root)),
            "features": str(feature_path.relative_to(self.project_root)),
            "history": history,
        }
        (self.project_root / "outputs/features").mkdir(parents=True, exist_ok=True)
        (self.project_root / "outputs/features/mnist_mlp_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._plot_mnist_history(history)
        print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False))
        return result

    def _plot_mnist_history(self, history: dict[str, list[float]]) -> Path:
        """Vẽ train/validation loss và accuracy theo epoch."""
        destination = self.project_root / "outputs/figures/t0202_mnist_mlp_learning_curves.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, history["train_loss"], label="Train")
        axes[0].plot(epochs, history["val_loss"], label="Validation")
        axes[0].set(title="MNIST MLP — Loss", xlabel="Epoch", ylabel="Cross-entropy")
        axes[1].plot(epochs, history["train_metric"], label="Train")
        axes[1].plot(epochs, history["val_metric"], label="Validation")
        axes[1].set(title="MNIST MLP — Accuracy", xlabel="Epoch", ylabel="Accuracy")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return destination

    def _load_mnist_arrays(self, data_root: Path, download: bool) -> dict[str, np.ndarray]:
        """Đọc MNIST đã cache hoặc tải bằng torchvision, rồi fallback IDX."""
        names = ("train_images", "train_labels", "test_images", "test_labels")
        cached = {name: data_root / f"{name}.npy" for name in names}
        if all(path.is_file() for path in cached.values()):
            return {name: np.load(path, mmap_mode="r") for name, path in cached.items()}
        if not download:
            raise FileNotFoundError(f"Thiếu MNIST thật trong {data_root}; bật --download.")
        try:
            from torchvision.datasets import MNIST

            train_set = MNIST(str(data_root.parent), train=True, download=True)
            test_set = MNIST(str(data_root.parent), train=False, download=True)
            arrays = {
                "train_images": train_set.data.numpy(),
                "train_labels": train_set.targets.numpy(),
                "test_images": test_set.data.numpy(),
                "test_labels": test_set.targets.numpy(),
            }
        except Exception as error:
            print(f"torchvision MNIST không khả dụng ({error}); dùng bộ tải IDX sẵn có.", file=sys.stderr)
            DataBootstrapper(self.project_root).download_mnist()
            arrays = {name: np.load(path, mmap_mode="r") for name, path in cached.items()}
        for name, array in arrays.items():
            output = cached[name]
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                np.save(output, array, allow_pickle=False)
        return arrays

    def _configured_drive_root(self) -> Path | None:
        """Chỉ bật mirror khi runtime adapter đã mount Drive thật."""
        candidate = self.project_root / "configs/paths.yaml"
        if not candidate.exists():
            return None
        configured = (yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}).get(
            "project", {}
        ).get("drive_mount_root")
        if not configured:
            return None
        path = Path(configured)
        return path if path.is_dir() else None

    def _create_history_logger(
        self, config: dict[str, Any], epochs: int, batch_size: int
    ) -> ExperimentLogger:
        """Tạo logger trainer; riêng Test_Metric được ghi ở bản tổng kết sau evaluate."""
        return self._logger(
            model_group="Group1_MLP",
            model_name="MNISTMLP",
            exp_type="T0202_MNISTMLP",
            seed=42,
            optimizer="SGD",
            learning_rate=float(config["learning_rate"]),
            batch_size=batch_size,
            epochs=epochs,
        )

    @staticmethod
    def _evaluate_accuracy(model: MNISTMLP, loader: DataLoader) -> float:
        """Tính độ chính xác phần trăm trên một DataLoader MNIST."""
        model.eval()
        correct = total = 0
        device = next(model.parameters()).device
        with torch.inference_mode():
            for images, labels in loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += len(labels)
        return correct / total

    @staticmethod
    def _extract_features(model: MNISTMLP, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook activation sau ReLU fc2 và ghép feature/nhãn toàn tập."""
        model.eval()
        model.register_feature_hooks(["relu2"])
        feature_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []
        device = next(model.parameters()).device
        try:
            with torch.inference_mode():
                for images, labels in loader:
                    _, captured = model.extract_features(images.to(device))
                    feature_batches.append(captured["relu2"].cpu())
                    label_batches.append(labels.cpu())
        finally:
            model.clear_hooks()
        return torch.cat(feature_batches), torch.cat(label_batches)

    def plot_gaussian(self) -> Path:
        """Vẽ bề mặt mục tiêu Gaussian 3D để dùng trong báo cáo/notebook."""
        dataset = BivariateNormalDataset(split="test")
        coordinates = dataset.inputs.numpy()
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d")
        axis.plot_trisurf(coordinates[:, 0], coordinates[:, 1], dataset.targets[:, 0].numpy())
        axis.set_title("Mật độ chuẩn hai biến — lưới kiểm thử 10×10")
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.set_zlabel("Mật độ")
        destination = self.project_root / "outputs/figures/t0201_gaussian_surface.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return destination


def _project_root() -> Path:
    """Tìm project root từ vị trí file script."""
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào CLI cho sweep Gaussian, MNIST hoặc cả hai."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "func", "mnist"), default="all")
    parser.add_argument("--mnist-epochs", type=int, default=None)
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--sweep", choices=("A", "B", "C", "C10"), default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--member-alias", default="Duy-DS")
    args = parser.parse_args(argv)
    runner = Group1MLPRunner(_project_root(), member_alias=args.member_alias)
    if args.task in {"all", "func"}:
        if not args.skip_sweeps:
            runner.run_gaussian_sweeps(sweep_filter=args.sweep)
        runner.plot_gaussian()
    if args.task in {"all", "mnist"}:
        runner.run_mnist(epochs=args.mnist_epochs, download=not args.no_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())