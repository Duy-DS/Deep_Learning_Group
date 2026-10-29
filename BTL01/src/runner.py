"""Trình thực thi thực nghiệm (Experiment Runner).

Module này thuộc T0304, chịu trách nhiệm tự động hóa vòng lặp thực nghiệm,
đo lường thời gian (perf_counter), đặt cấu hình tái lập (seed),
và kết xuất dữ liệu phục vụ báo cáo đo đạc.
"""

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import sys

# Đảm bảo import được các module từ thư mục gốc
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluator import RegressionEvaluator
from src.data_generator import GaussianDataGenerator
from src.models import FunctionApproximatorMLP
from src.trainer import Trainer


class ExperimentRunner:
    """Quản lý và thực thi một hoặc nhiều lần chạy thực nghiệm an toàn."""

    def __init__(
        self,
        baseline_config: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        """Khởi tạo trình thực thi thực nghiệm.

        Tham số:
            baseline_config: Toàn bộ cấu hình đọc từ baseline.yaml.
            device: Thiết bị tính toán (CPU/CUDA/MPS).
        """
        self.config = baseline_config
        self.device = device
        self.evaluator = RegressionEvaluator(device)

    def set_seed(self, seed: int) -> None:
        """Cố định các bộ sinh số ngẫu nhiên để đảm bảo tính tái lập chuẩn (Reproducibility).

        Tham số:
            seed: Giá trị seed nguyên.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Áp dụng cấu hình determinism từ baseline.yaml
        if self.config["reproducibility"]["deterministic"]:
            torch.use_deterministic_algorithms(
                True,
                warn_only=self.config["reproducibility"]["deterministic_warn_only"],
            )
            torch.backends.cudnn.deterministic = self.config["reproducibility"][
                "cudnn_deterministic"
            ]
            torch.backends.cudnn.benchmark = self.config["reproducibility"][
                "cudnn_benchmark"
            ]

    def build_width_trials(
        self, experiment_config: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Resolve the complete Experiment B width/seed grid.

        Args:
            experiment_config: The ``exp_B`` mapping from ``exp_specs.yaml``.

        Returns:
            Trial mappings with ``width``, ``hidden_dims``, ``train_size``,
            and ``seed`` for every registered run.
        """
        widths = [
            *experiment_config["neuron_widths"],
            *experiment_config.get("extended_widths", []),
        ]
        return [
            {
                "width": int(width),
                "hidden_dims": [int(width)],
                "train_size": int(experiment_config["train_size"]),
                "seed": int(seed),
            }
            for width in widths
            for seed in experiment_config["multi_seeds"]
        ]

    @staticmethod
    def _parameter_count(
        input_dim: int, hidden_dims: list[int], output_dim: int
    ) -> int:
        """Count parameters for bias-enabled fully connected layers."""
        widths = [input_dim, *hidden_dims, output_dim]
        return sum(
            (left * right) + right
            for left, right in zip(widths, widths[1:])
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        """Return the SHA-256 digest of a persisted input artifact."""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def run_width_experiment(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        experiment_config: Mapping[str, Any],
        model_config: Mapping[str, Any],
        train_config: Mapping[str, Any],
        output_root: Path | str = REPO_ROOT / "outputs",
        epochs: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run and aggregate the registered Experiment B width trials.

        Args:
            train_dataset: Full training dataset with ``split='train'`` metadata.
            val_dataset: Validation dataset with ``split='val'`` metadata.
            test_dataset: Independent test dataset used only after fitting.
            experiment_config: The ``exp_B`` mapping from ``exp_specs.yaml``.
            model_config: Baseline model configuration.
            train_config: Baseline training configuration.
            output_root: Root directory for checkpoints and tables.
            epochs: Optional smoke-test override; ``None`` uses the baseline.

        Returns:
            Per-run and mean/std summary DataFrames.
        """
        root = Path(output_root)
        checkpoints = root / "checkpoints"
        tables = root / "tables"
        checkpoints.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        trials = self.build_width_trials(experiment_config)
        data_config = self.config["data"]
        generator = GaussianDataGenerator(
            mu=data_config["mu"],
            cov=data_config["cov"],
            x1_range=tuple(data_config["x1_range"]),
            x2_range=tuple(data_config["x2_range"]),
        )
        resolved_train_config = dict(train_config)
        if epochs is not None:
            resolved_train_config["epochs"] = epochs
        rows: list[dict[str, Any]] = []
        metric_names = [
            "train_mse", "train_rmse", "train_mae",
            "val_mse", "val_rmse", "val_mae",
            "test_mse", "test_rmse", "test_mae",
            "training_time_seconds", "best_epoch", "epochs_run",
            "optimizer_steps",
        ]
        for trial in trials:
            width = trial["width"]
            seed = trial["seed"]
            run_id = f"exp_B_hidden_width{width}_seed{seed}"
            checkpoint_path = checkpoints / f"best_model_{run_id}.pt"
            subset_indices = generator.fixed_train_subset_indices(
                len(train_dataset), trial["train_size"], data_config["subset_seed"]
            )
            loader_generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                Subset(train_dataset, subset_indices.tolist()),
                batch_size=train_config["batch_size"],
                shuffle=train_config["shuffle_train"],
                drop_last=resolved_train_config["drop_last"],
                generator=loader_generator,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=train_config["batch_size"],
                shuffle=False,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=train_config["batch_size"],
                shuffle=False,
            )
            try:
                result = self.run_single_trial(
                    train_loader,
                    val_loader,
                    test_loader,
                    {**model_config, "hidden_dims": [width], "hidden_depth": 1},
                    resolved_train_config,
                    seed,
                    checkpoint_path,
                )
                history = result.pop("history", {})
                history_frame = pd.DataFrame(history)
                history_frame.insert(0, "epoch", range(1, len(history_frame) + 1))
                history_frame.to_csv(
                    tables / f"history_{run_id}.csv", index=False
                )
                manifest = {
                    "experiment": "exp_B",
                    "run_id": run_id,
                    "factor": {"name": "hidden_width", "value": width},
                    "seed": seed,
                    "device": str(self.device),
                    "resolved_config": {
                        "model": {
                            **model_config,
                            "hidden_dims": [width],
                            "hidden_depth": 1,
                        },
                        "training": resolved_train_config,
                    },
                    "data_hashes": {
                        "train_full": self._sha256(train_dataset.csv_path),
                        "val": self._sha256(val_dataset.csv_path),
                        "test": self._sha256(test_dataset.csv_path),
                    },
                    "config_hashes": {
                        "baseline": self._sha256(REPO_ROOT / "configs/baseline.yaml"),
                        "experiment_specs": self._sha256(REPO_ROOT / "configs/exp_specs.yaml"),
                    },
                    "checkpoint_path": str(checkpoint_path),
                }
                with (tables / f"manifest_{run_id}.json").open(
                    "w", encoding="utf-8"
                ) as stream:
                    json.dump(manifest, stream, indent=2, sort_keys=True)
                row = {
                    "run_id": run_id,
                    "experiment": "exp_B",
                    "factor": "hidden_width",
                    "width": width,
                    "hidden_dims": str([width]),
                    "train_size": trial["train_size"],
                    "parameter_count": self._parameter_count(2, [width], 1),
                    "seed": seed,
                    "status": "completed",
                    **{name: result.get(name, float("nan")) for name in metric_names},
                }
            except (RuntimeError, ValueError, OSError, KeyError) as error:
                row = {
                    "run_id": run_id,
                    "experiment": "exp_B",
                    "factor": "hidden_width",
                    "width": width,
                    "hidden_dims": str([width]),
                    "train_size": trial["train_size"],
                    "parameter_count": self._parameter_count(2, [width], 1),
                    "seed": seed,
                    "status": "failed",
                    "error": str(error),
                }
            rows.append(row)
        runs = pd.DataFrame(rows)
        runs.to_csv(tables / "runs_exp_B.csv", index=False)
        valid = runs[runs["status"] == "completed"]
        summary_rows: list[dict[str, Any]] = []
        for width, group in valid.groupby("width", sort=True):
            summary: dict[str, Any] = {
                "width": int(width),
                "hidden_dims": str([int(width)]),
                "n_runs": int(len(group)),
            }
            for name in metric_names:
                values = group[name].astype(float)
                summary[f"{name}_mean"] = float(values.mean())
                summary[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            summary_rows.append(summary)
        summary_frame = pd.DataFrame(summary_rows)
        summary_frame.to_csv(tables / "summary_exp_B.csv", index=False)
        return runs, summary_frame

    def build_train_size_trials(
        self, experiment_config: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Resolve the Experiment A train-size/seed grid."""
        return [
            {
                "train_size": int(train_size),
                "hidden_dims": [10],
                "seed": int(seed),
            }
            for train_size in experiment_config["sample_sizes"]
            for seed in experiment_config["multi_seeds"]
        ]

    def run_train_size_experiment(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        experiment_config: Mapping[str, Any],
        model_config: Mapping[str, Any],
        train_config: Mapping[str, Any],
        output_root: Path | str = REPO_ROOT / "outputs",
        epochs: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run and aggregate the registered Experiment A train-size trials."""
        root = Path(output_root)
        checkpoints = root / "checkpoints"
        tables = root / "tables"
        checkpoints.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        data_config = self.config["data"]
        generator = GaussianDataGenerator(
            mu=data_config["mu"],
            cov=data_config["cov"],
            x1_range=tuple(data_config["x1_range"]),
            x2_range=tuple(data_config["x2_range"]),
        )
        resolved_train_config = dict(train_config)
        if epochs is not None:
            resolved_train_config["epochs"] = epochs
        metric_names = [
            "train_mse", "train_rmse", "train_mae",
            "val_mse", "val_rmse", "val_mae",
            "test_mse", "test_rmse", "test_mae",
            "training_time_seconds", "best_epoch", "epochs_run",
            "optimizer_steps",
        ]
        rows: list[dict[str, Any]] = []
        for trial in self.build_train_size_trials(experiment_config):
            train_size = trial["train_size"]
            seed = trial["seed"]
            run_id = f"exp_A_train_size{train_size}_seed{seed}"
            checkpoint_path = checkpoints / f"best_model_{run_id}.pt"
            indices = generator.fixed_train_subset_indices(
                len(train_dataset), train_size, data_config["subset_seed"]
            )
            train_loader = DataLoader(
                Subset(train_dataset, indices.tolist()),
                batch_size=resolved_train_config["batch_size"],
                shuffle=resolved_train_config["shuffle_train"],
                drop_last=resolved_train_config["drop_last"],
                generator=torch.Generator().manual_seed(seed),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=resolved_train_config["batch_size"]
            )
            test_loader = DataLoader(
                test_dataset, batch_size=resolved_train_config["batch_size"]
            )
            try:
                result = self.run_single_trial(
                    train_loader,
                    val_loader,
                    test_loader,
                    {**model_config, "hidden_dims": [10], "hidden_depth": 1},
                    resolved_train_config,
                    seed,
                    checkpoint_path,
                )
                history = result.pop("history", {})
                history_frame = pd.DataFrame(history)
                history_frame.insert(0, "epoch", range(1, len(history_frame) + 1))
                history_frame.to_csv(tables / f"history_{run_id}.csv", index=False)
                manifest = {
                    "experiment": "exp_A",
                    "run_id": run_id,
                    "factor": {"name": "train_size", "value": train_size},
                    "seed": seed,
                    "device": str(self.device),
                    "resolved_config": {
                        "model": {
                            **model_config, "hidden_dims": [10], "hidden_depth": 1
                        },
                        "training": resolved_train_config,
                    },
                    "data_hashes": {
                        "train_full": self._sha256(train_dataset.csv_path),
                        "val": self._sha256(val_dataset.csv_path),
                        "test": self._sha256(test_dataset.csv_path),
                    },
                    "config_hashes": {
                        "baseline": self._sha256(REPO_ROOT / "configs/baseline.yaml"),
                        "experiment_specs": self._sha256(REPO_ROOT / "configs/exp_specs.yaml"),
                    },
                    "checkpoint_path": str(checkpoint_path),
                }
                with (tables / f"manifest_{run_id}.json").open(
                    "w", encoding="utf-8"
                ) as stream:
                    json.dump(manifest, stream, indent=2, sort_keys=True)
                row = {
                    "run_id": run_id,
                    "experiment": "exp_A",
                    "factor": "train_size",
                    "train_size": train_size,
                    "hidden_dims": str([10]),
                    "parameter_count": self._parameter_count(2, [10], 1),
                    "seed": seed,
                    "status": "completed",
                    **{name: result.get(name, float("nan")) for name in metric_names},
                }
            except (RuntimeError, ValueError, OSError, KeyError) as error:
                row = {
                    "run_id": run_id,
                    "experiment": "exp_A",
                    "factor": "train_size",
                    "train_size": train_size,
                    "hidden_dims": str([10]),
                    "parameter_count": self._parameter_count(2, [10], 1),
                    "seed": seed,
                    "status": "failed",
                    "error": str(error),
                }
            rows.append(row)
        runs = pd.DataFrame(rows)
        runs.to_csv(tables / "runs_exp_A.csv", index=False)
        valid = runs[runs["status"] == "completed"]
        summary_rows: list[dict[str, Any]] = []
        for train_size, group in valid.groupby("train_size", sort=True):
            summary: dict[str, Any] = {
                "train_size": int(train_size),
                "hidden_dims": str([10]),
                "parameter_count": self._parameter_count(2, [10], 1),
                "n_runs": int(len(group)),
            }
            for name in metric_names:
                values = group[name].astype(float)
                summary[f"{name}_mean"] = float(values.mean())
                summary[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            summary_rows.append(summary)
        summary_frame = pd.DataFrame(summary_rows)
        summary_frame.to_csv(tables / "summary_exp_A.csv", index=False)
        return runs, summary_frame

    def build_depth_trials(
        self, experiment_config: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Resolve the fixed-width Experiment C depth/seed grid."""
        width = int(experiment_config["width_per_layer"])
        trials: list[dict[str, Any]] = []
        for depth in experiment_config["depths"]:
            hidden_dims = [width] * int(depth)
            for seed in experiment_config["multi_seeds"]:
                trials.append({
                    "depth": int(depth),
                    "hidden_dims": hidden_dims,
                    "train_size": int(experiment_config["train_size"]),
                    "seed": int(seed),
                    "parameter_count": self._parameter_count(2, hidden_dims, 1),
                })
        return trials

    def run_depth_experiment(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        experiment_config: Mapping[str, Any],
        model_config: Mapping[str, Any],
        train_config: Mapping[str, Any],
        output_root: Path | str = REPO_ROOT / "outputs",
        epochs: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run and aggregate the registered Experiment C depth trials."""
        root = Path(output_root)
        checkpoints = root / "checkpoints"
        tables = root / "tables"
        checkpoints.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        data_config = self.config["data"]
        generator = GaussianDataGenerator(
            mu=data_config["mu"],
            cov=data_config["cov"],
            x1_range=tuple(data_config["x1_range"]),
            x2_range=tuple(data_config["x2_range"]),
        )
        resolved_train_config = dict(train_config)
        if epochs is not None:
            resolved_train_config["epochs"] = epochs
        metric_names = [
            "train_mse", "train_rmse", "train_mae",
            "val_mse", "val_rmse", "val_mae",
            "test_mse", "test_rmse", "test_mae",
            "training_time_seconds", "best_epoch", "epochs_run",
            "optimizer_steps",
        ]
        rows: list[dict[str, Any]] = []
        for trial in self.build_depth_trials(experiment_config):
            depth = trial["depth"]
            seed = trial["seed"]
            run_id = f"exp_C_hidden_depth{depth}_seed{seed}"
            checkpoint_path = checkpoints / f"best_model_{run_id}.pt"
            indices = generator.fixed_train_subset_indices(
                len(train_dataset), trial["train_size"], data_config["subset_seed"]
            )
            train_loader = DataLoader(
                Subset(train_dataset, indices.tolist()),
                batch_size=resolved_train_config["batch_size"],
                shuffle=resolved_train_config["shuffle_train"],
                drop_last=resolved_train_config["drop_last"],
                generator=torch.Generator().manual_seed(seed),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=resolved_train_config["batch_size"]
            )
            test_loader = DataLoader(
                test_dataset, batch_size=resolved_train_config["batch_size"]
            )
            try:
                result = self.run_single_trial(
                    train_loader,
                    val_loader,
                    test_loader,
                    {**model_config, "hidden_dims": trial["hidden_dims"], "hidden_depth": depth},
                    resolved_train_config,
                    seed,
                    checkpoint_path,
                )
                history = result.pop("history", {})
                history_frame = pd.DataFrame(history)
                history_frame.insert(0, "epoch", range(1, len(history_frame) + 1))
                history_frame.to_csv(tables / f"history_{run_id}.csv", index=False)
                manifest = {
                    "experiment": "exp_C",
                    "run_id": run_id,
                    "factor": {"name": "hidden_depth", "value": depth},
                    "seed": seed,
                    "device": str(self.device),
                    "comparison_mode": "fixed_width_not_parameter_matched",
                    "resolved_config": {
                        "model": {
                            **model_config,
                            "hidden_dims": trial["hidden_dims"],
                            "hidden_depth": depth,
                        },
                        "training": resolved_train_config,
                    },
                    "data_hashes": {
                        "train_full": self._sha256(train_dataset.csv_path),
                        "val": self._sha256(val_dataset.csv_path),
                        "test": self._sha256(test_dataset.csv_path),
                    },
                    "config_hashes": {
                        "baseline": self._sha256(REPO_ROOT / "configs/baseline.yaml"),
                        "experiment_specs": self._sha256(REPO_ROOT / "configs/exp_specs.yaml"),
                    },
                    "checkpoint_path": str(checkpoint_path),
                }
                with (tables / f"manifest_{run_id}.json").open(
                    "w", encoding="utf-8"
                ) as stream:
                    json.dump(manifest, stream, indent=2, sort_keys=True)
                row = {
                    "run_id": run_id,
                    "experiment": "exp_C",
                    "factor": "hidden_depth",
                    "depth": depth,
                    "hidden_dims": str(trial["hidden_dims"]),
                    "train_size": trial["train_size"],
                    "parameter_count": trial["parameter_count"],
                    "seed": seed,
                    "status": "completed",
                    **{name: result.get(name, float("nan")) for name in metric_names},
                }
            except (RuntimeError, ValueError, OSError, KeyError) as error:
                row = {
                    "run_id": run_id,
                    "experiment": "exp_C",
                    "factor": "hidden_depth",
                    "depth": depth,
                    "hidden_dims": str(trial["hidden_dims"]),
                    "train_size": trial["train_size"],
                    "parameter_count": trial["parameter_count"],
                    "seed": seed,
                    "status": "failed",
                    "error": str(error),
                }
            rows.append(row)
        runs = pd.DataFrame(rows)
        runs.to_csv(tables / "runs_exp_C.csv", index=False)
        valid = runs[runs["status"] == "completed"]
        summary_rows: list[dict[str, Any]] = []
        for depth, group in valid.groupby("depth", sort=True):
            summary: dict[str, Any] = {
                "depth": int(depth),
                "hidden_dims": group.iloc[0]["hidden_dims"],
                "parameter_count": int(group.iloc[0]["parameter_count"]),
                "n_runs": int(len(group)),
            }
            for name in metric_names:
                values = group[name].astype(float)
                summary[f"{name}_mean"] = float(values.mean())
                summary[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            summary_rows.append(summary)
        summary_frame = pd.DataFrame(summary_rows)
        summary_frame.to_csv(tables / "summary_exp_C.csv", index=False)
        return runs, summary_frame

    def run_single_trial(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model_config: Mapping[str, Any],
        train_config: Mapping[str, Any],
        seed: int,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        """Thực thi vòng đời huấn luyện và đánh giá trên một tổ hợp siêu tham số.

        Tham số:
            train_loader: DataLoader cho tập huấn luyện.
            val_loader: DataLoader cho tập xác thực.
            test_loader: DataLoader cho tập kiểm tra (chỉ dùng đánh giá test cuối).
            model_config: Cấu hình kiến trúc mô hình (dict).
            train_config: Cấu hình vòng lặp huấn luyện (dict).
            seed: Seed tái lập.
            checkpoint_path: Đường dẫn lưu file .pt tốt nhất.

        Trả về:
            Từ điển chứa kết quả đo đạc (metrics), thời gian, và đối tượng lịch sử.
        """
        self.set_seed(seed)

        # 1. Khởi tạo Mô hình từ đầu cho mỗi seed (không Warm-start)
        model = FunctionApproximatorMLP(model_config, self.config["data"]).to(
            self.device
        )

        # 2. Khởi tạo Trainer an toàn từ T0303
        trainer = Trainer(
            model=model,
            train_config=train_config,
            device=self.device,
            checkpoint_dir=checkpoint_path.parent,
            checkpoint_path=checkpoint_path,
        )

        # 3. Đồng bộ GPU (nếu dùng) trước khi bấm giờ huấn luyện
        sync_accelerator = self.config["evaluation"]["synchronize_accelerator"]
        if sync_accelerator and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        start_time = time.perf_counter()

        # 4. Huấn luyện (gắn Early Stopping & Atomic Checkpointing)
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_config["epochs"],
            checkpoint_name=checkpoint_path.name,
            verbose=False,
        )

        if sync_accelerator and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        training_time = time.perf_counter() - start_time

        # 5. Khôi phục best checkpoint (được Trainer tự xử lý) và đánh giá metrics
        train_metrics = self.evaluator.evaluate(trainer.model, train_loader)
        val_metrics = self.evaluator.evaluate(trainer.model, val_loader)
        test_metrics = self.evaluator.evaluate(trainer.model, test_loader)

        return {
            "seed": seed,
            "best_epoch": trainer.best_epoch,
            "epochs_run": trainer.epochs_run,
            "optimizer_steps": trainer.epochs_run * len(train_loader),
            "training_time_seconds": training_time,
            "training_time_sec": training_time,
            "train_mse": train_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "test_mse": test_metrics["mse"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "history": history,
        }
