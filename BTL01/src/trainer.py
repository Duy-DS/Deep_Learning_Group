"""Vòng lặp huấn luyện tùy chỉnh và các Callback.

Module này triển khai thành phần T0302: logic của Trainer, Dừng sớm (EarlyStopping) và Lưu điểm kiểm tra (Checkpoint).
"""

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class EarlyStopping:
    """Dừng sớm (Early stopping) để chấm dứt quá trình huấn luyện khi loss trên tập xác thực không còn cải thiện."""

    def __init__(self, patience: int, min_delta: float, mode: str = "min") -> None:
        """Khởi tạo EarlyStopping.

        Tham số:
            patience: Số epoch chờ đợi sau lần cuối loss trên tập xác thực được cải thiện.
            min_delta: Sự thay đổi tối thiểu của đại lượng được theo dõi để được coi là một sự cải thiện.
            mode: Một trong hai giá trị 'min' hoặc 'max'.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.early_stop = False

    def __call__(self, current_loss: float) -> bool:
        """Kiểm tra xem có nên dừng quá trình huấn luyện hay không.

        Tham số:
            current_loss: Loss trên tập xác thực của epoch hiện tại.

        Trả về:
            True nếu kích hoạt dừng sớm, False nếu ngược lại.
        """
        if self.mode == "min":
            improvement = self.best_score - current_loss > self.min_delta
        else:
            improvement = current_loss - self.best_score > self.min_delta

        if improvement:
            self.best_score = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improvement


class Trainer:
    """Xử lý vòng lặp huấn luyện và đánh giá mô hình."""

    def __init__(
        self,
        model: nn.Module,
        train_config: Mapping[str, Any],
        device: torch.device,
        checkpoint_path: Path,
    ) -> None:
        """Khởi tạo Trainer.

        Tham số:
            model: Mô hình PyTorch cần huấn luyện.
            train_config: Mapping cấu hình huấn luyện (ví dụ: từ `training_baseline`).
            device: Thiết bị tính toán (CPU/CUDA/MPS).
            checkpoint_path: Đường dẫn để lưu trọng số mô hình tốt nhất.
        """
        self.model = model.to(device)
        self.device = device
        self.config = train_config
        self.checkpoint_path = checkpoint_path

        self.epochs = self.config["epochs"]
        self.criterion = self._get_loss_function()
        self.optimizer = self._get_optimizer()

        early_stop_cfg = self.config["early_stopping"]
        self.use_early_stopping = early_stop_cfg["enabled"]
        if self.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=early_stop_cfg["patience"],
                min_delta=early_stop_cfg["min_delta"],
                mode=early_stop_cfg["mode"],
            )

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")

    def _get_loss_function(self) -> nn.Module:
        """Khởi tạo hàm loss dựa trên cấu hình."""
        if self.config["loss_function"] == "mse":
            return nn.MSELoss(reduction=self.config["loss_reduction"])
        raise ValueError(f"Unsupported loss function: {self.config['loss_function']}")

    def _get_optimizer(self) -> optim.Optimizer:
        """Khởi tạo bộ tối ưu hóa (optimizer) dựa trên cấu hình."""
        opt_name = self.config["optimizer"]
        lr = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]

        if opt_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(self.config["adam_betas"]),
                eps=self.config["adam_eps"],
                amsgrad=self.config["adam_amsgrad"],
            )
        elif opt_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=self.config["sgd_momentum"],
                nesterov=self.config["sgd_nesterov"],
            )
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Chạy một epoch huấn luyện."""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()

            if self.config.get("gradient_clipping", False):
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        return total_loss / len(dataloader.dataset)  # type: ignore

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Chạy đánh giá trên tập xác thực hoặc tập kiểm tra."""
        self.model.eval()
        total_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            total_loss += loss.item() * X_batch.size(0)

        return total_loss / len(dataloader.dataset)  # type: ignore

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict[str, list[float]]:
        """Thực thi toàn bộ vòng lặp huấn luyện.

        Tham số:
            train_loader: DataLoader cho tập huấn luyện.
            val_loader: DataLoader cho tập xác thực.

        Trả về:
            Dictionary chứa lịch sử 'train_loss' và 'val_loss'.
        """
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Checkpoint logic (Save on strict improvement)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint_path)

            # Early stopping logic
            if self.use_early_stopping:
                is_improvement = self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Restore best weights if required
        if self.config["checkpoint"]["restore_best"] and self.checkpoint_path.exists():
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )

        return self.history
