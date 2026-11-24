"""Vòng lặp huấn luyện PyTorch dùng chung cho các mô hình BTL03."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .callbacks import CallbackList


class TrainerCallback(ABC):
    """Giao diện callback cuối epoch cho scheduler/early stopping.

    Callback được gọi sau validation và logger, trước checkpoint lưu epoch tốt
    nhất. Trả về True để yêu cầu dừng sớm.
    """

    @abstractmethod
    def on_epoch_end(
        self, trainer: "BaseTrainer", epoch: int, logs: Mapping[str, float]
    ) -> bool | None:
        """Xử lý log cuối epoch; trả True nếu muốn dừng quá trình fit."""
        raise NotImplementedError


class BaseTrainer:
    """Đóng gói train/validation, logging, checkpoint và callback theo OOP."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None,
        *,
        model_group: str = "02_cnn_mlp",
        model_name: str | None = None,
        seed: int = 42,
        member_alias: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Khởi tạo trainer và chuyển mô hình lên thiết bị được chọn.

        Args:
            model: Mô hình PyTorch; tương thích BaseMigratedModel hoặc nn.Module.
            optimizer: Optimizer gắn với tham số của model.
            criterion: Hàm loss PyTorch.
            device: Thiết bị tường minh; mặc định CUDA nếu có, ngược lại CPU.
            metric_fn: Hàm ``(dự đoán, target) -> scalar``; có thể bỏ qua.
            model_group: Nhóm dùng khi lưu checkpoint.
            model_name: Tên mô hình; mặc định lấy thuộc tính model_name hoặc class.
            seed: Seed của run.
            member_alias: Alias thành viên thật dùng cho checkpoint/logger.
            config: Cấu hình được ghi vào metadata checkpoint.
        """
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_fn = metric_fn
        self.model_group = model_group
        self.model_name = model_name or getattr(model, "model_name", type(model).__name__)
        self.seed = int(seed)
        self.member_alias = member_alias
        self.config = dict(config or {})
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_metric": [], "val_metric": [], "epoch_time_sec": [],
            "learning_rate": [],
        }
        self.callbacks: list[Any] = []
        self.stop_training = False

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Lấy (input,target) từ tuple/list DataLoader và xác thực tensor."""
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Mỗi batch phải là tuple/list gồm input và target.")
        inputs, targets = batch[0], batch[1]
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("Input và target của batch phải là torch.Tensor.")
        return inputs, targets

    @staticmethod
    def _loss_for_batch(loss: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Chuẩn hóa loss về scalar để backward ổn định."""
        if loss.ndim == 0:
            return loss
        return loss.mean()

    def _validate_model_input(self, inputs: torch.Tensor) -> None:
        """Gọi validator NCHW nếu mô hình cơ sở có cung cấp giao diện đó."""
        validator = getattr(self.model, "validate_input_shape", None)
        if callable(validator) and inputs.ndim == 4:
            validator(inputs)

    def _metric_value(self, outputs: torch.Tensor, targets: torch.Tensor) -> float | None:
        """Tính metric scalar hoặc trả None khi metric_fn không được cấu hình."""
        if self.metric_fn is None:
            return None
        value = self.metric_fn(outputs.detach(), targets.detach())
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("metric_fn phải trả về scalar.")
            value = value.item()
        return float(value)

    def train_one_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Chạy đúng một epoch train và trả loss/metric trung bình theo mẫu.

        Args:
            train_loader: DataLoader cung cấp batch `(input, target)`.

        Returns:
            Tuple `(mean_loss, mean_metric)`; metric là NaN nếu metric_fn vắng mặt.
        """
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        metric_samples = 0
        sample_count = 0
        for batch in train_loader:
            inputs, targets = self._unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self._validate_model_input(inputs)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self._loss_for_batch(self.criterion(outputs, targets), len(inputs))
            loss.backward()
            self.optimizer.step()
            batch_count = int(inputs.shape[0])
            total_loss += float(loss.detach().item()) * batch_count
            metric = self._metric_value(outputs, targets)
            if metric is not None:
                total_metric += metric * batch_count
                metric_samples += batch_count
            sample_count += batch_count
        if sample_count == 0:
            raise ValueError("train_loader không chứa sample nào.")
        mean_metric = total_metric / metric_samples if metric_samples else float("nan")
        return total_loss / sample_count, mean_metric

    def validate_one_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Chạy validation không gradient và trả loss/metric trung bình theo mẫu.

        Args:
            val_loader: DataLoader validation; không được áp augmentation train.

        Returns:
            Tuple `(mean_loss, mean_metric)`; metric là NaN nếu metric_fn vắng mặt.
        """
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        metric_samples = 0
        sample_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self._validate_model_input(inputs)
                outputs = self.model(inputs)
                loss = self._loss_for_batch(self.criterion(outputs, targets), len(inputs))
                batch_count = int(inputs.shape[0])
                total_loss += float(loss.item()) * batch_count
                metric = self._metric_value(outputs, targets)
                if metric is not None:
                    total_metric += metric * batch_count
                    metric_samples += batch_count
                sample_count += batch_count
        if sample_count == 0:
            raise ValueError("val_loader không chứa sample nào.")
        mean_metric = total_metric / metric_samples if metric_samples else float("nan")
        return total_loss / sample_count, mean_metric

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        logger: Any = None,
        checkpoint_manager: Any = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, List[float]]:
        """Huấn luyện nhiều epoch, ghi logs, checkpoint tốt nhất và callback.

        Args:
            train_loader: Dữ liệu train.
            val_loader: Dữ liệu validation độc lập theo protocol dự án.
            epochs: Số epoch dương.
            logger: Đối tượng tương thích ExperimentLogger của T0104.
            checkpoint_manager: Đối tượng tương thích CheckpointManager T0104.
            callbacks: Callback lifecycle mới hoặc API legacy
                `on_epoch_end(trainer, epoch, logs)`; callback trả True để dừng.

        Returns:
            Dict lịch sử gồm loss, metric (NaN nếu bỏ qua) và epoch time.
        """
        if epochs <= 0:
            raise ValueError("epochs phải là số nguyên dương.")
        if logger is not None and not getattr(logger, "member_alias", None):
            raise ValueError(
                "Logger phải được khởi tạo với member_alias thật theo quy định định danh BTL03."
            )
        callback_list = CallbackList(callbacks)
        callback_list.set_trainer(self)
        self.callbacks = callback_list.callbacks
        self.stop_training = False
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_metric": [], "val_metric": [], "epoch_time_sec": [],
            "learning_rate": [],
        }
        best_val_loss = float("inf")
        best_epoch: int | None = None
        callback_list.on_train_begin({"model": self.model, "optimizer": self.optimizer})
        try:
            for epoch in range(1, int(epochs) + 1):
                callback_list.on_epoch_begin(
                    epoch, {"model": self.model, "optimizer": self.optimizer}
                )
                started = time.perf_counter()
                train_loss, train_metric = self.train_one_epoch(train_loader)
                val_loss, val_metric = self.validate_one_epoch(val_loader)
                epoch_time = time.perf_counter() - started
                epoch_lr = float(self.optimizer.param_groups[0]["lr"])
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["train_metric"].append(train_metric)
                self.history["val_metric"].append(val_metric)
                self.history["epoch_time_sec"].append(epoch_time)
                self.history["learning_rate"].append(epoch_lr)

                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    if checkpoint_manager is not None:
                        checkpoint_manager.save(
                            model=self.model,
                            model_group=self.model_group,
                            model_name=self.model_name,
                            epoch=epoch,
                            optimizer=self.optimizer,
                            best_metric=val_loss,
                            seed=self.seed,
                            config=self.config,
                        )

                # Logs gồm cả số đo và tham chiếu runtime để callbacks có thể
                # lưu checkpoint/khôi phục state mà không phụ thuộc biến toàn cục.
                logs: dict[str, Any] = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metric": train_metric,
                    "val_metric": val_metric,
                    "train_acc": train_metric,
                    "val_acc": val_metric,
                    "lr": epoch_lr,
                    "learning_rate": epoch_lr,
                    "epoch_time_sec": epoch_time,
                    "model": self.model,
                    "optimizer": self.optimizer,
                }
                # Ghi loss/accuracy epoch hiện tại trước scheduler để không
                # làm sai LR ứng với metric trong CSV/TensorBoard.
                if logger is not None:
                    logger.log_epoch(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_metric=train_metric if math.isfinite(train_metric) else None,
                        val_metric=val_metric if math.isfinite(val_metric) else None,
                        learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                        epoch_time_sec=epoch_time,
                    )
                should_stop = callback_list.on_epoch_end(epoch, logs, self)
                self.stop_training = self.stop_training or should_stop
                # EarlyStopping có thể đã khôi phục best weights; duy trì invariant
                # rằng trainer checkpoint chung cũng trỏ tới cùng model state.
                if should_stop and checkpoint_manager is not None and best_epoch is not None:
                    checkpoint_manager.save(
                        model=self.model,
                        model_group=self.model_group,
                        model_name=self.model_name,
                        epoch=best_epoch,
                        optimizer=self.optimizer,
                        best_metric=best_val_loss,
                        seed=self.seed,
                        config=self.config,
                    )
                # Scheduler có thể vừa đổi LR; logger ghi LR sẽ dùng epoch sau.
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                logs["lr"] = current_lr
                logs["learning_rate"] = current_lr
                if self.stop_training:
                    break
        except Exception:
            # Không tổng kết run lỗi; vẫn đóng lifecycle callback để khôi phục
            # weights tốt nhất, nhưng không tạo master CSV có vẻ như run hoàn tất.
            try:
                callback_list.on_train_end({
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "stopped_epoch": len(self.history["train_loss"]),
                    "model": self.model,
                    "optimizer": self.optimizer,
                    "failed": True,
                })
            except Exception:
                # Không để lỗi cleanup callback che khuất exception train gốc.
                pass
            raise
        else:
            # Snapshot/checkpoint best được ghi trước khi callback train-end
            # khôi phục weights; callback save không được mutate model đang chạy.
            for callback in self.callbacks:
                save_best = getattr(callback, "save_best_checkpoint", None)
                if callable(save_best):
                    save_best()
            callback_list.on_train_end({
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "stopped_epoch": len(self.history["train_loss"]),
                "model": self.model,
                "optimizer": self.optimizer,
            })

        if logger is not None and self.history["train_loss"]:
            if best_epoch is None:
                raise RuntimeError("Không xác định được best_epoch để tổng kết logger.")
            safe_metric = lambda value: value if math.isfinite(value) else None
            logger.finish_run(
                best_epoch=best_epoch,
                train_loss=self.history["train_loss"][-1],
                val_loss=best_val_loss,
                train_metric=safe_metric(self.history["train_metric"][-1]),
                val_metric=safe_metric(self.history["val_metric"][-1]),
                train_time_sec=sum(self.history["epoch_time_sec"]),
            )
        elif logger is not None:
            logger.close()
        return self.history