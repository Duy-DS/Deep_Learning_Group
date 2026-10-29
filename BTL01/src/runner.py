"""Trình thực thi thực nghiệm (Experiment Runner).

Module này thuộc T0304, chịu trách nhiệm tự động hóa vòng lặp thực nghiệm,
đo lường thời gian (perf_counter), đặt cấu hình tái lập (seed),
và kết xuất dữ liệu phục vụ báo cáo đo đạc.
"""

import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

# Đảm bảo import được các module từ thư mục gốc
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluator import RegressionEvaluator
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
