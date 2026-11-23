"""Callbacks OOP cho quản lý quá trình huấn luyện PyTorch BTL03."""

from __future__ import annotations

import copy
import math
from abc import ABC
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.optim import Optimizer


class _StateDictSnapshot:
    """Adapter chỉ ghi best model state mà không mutate model đang train."""

    def __init__(self, model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
        """Giữ module schema và state tốt nhất đã deep-copy."""
        self._model = model
        self._state_dict = state_dict

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Trả snapshot để CheckpointManager.serialize đúng weights tốt nhất."""
        return self._state_dict

    def parameters(self):
        """Chuyển tiếp iterator tham số để adapter tương thích typing manager."""
        return self._model.parameters()


class _OptimizerStateSnapshot:
    """Adapter cung cấp snapshot optimizer đã lưu mà không sửa optimizer đang chạy."""

    def __init__(self, optimizer: Optimizer, state_dict: dict[str, Any] | None) -> None:
        """Giữ optimizer schema và state dict snapshot tốt nhất."""
        self._optimizer = optimizer
        self._state_dict = state_dict or optimizer.state_dict()

    def state_dict(self) -> dict[str, Any]:
        """Trả state snapshot cho CheckpointManager."""
        return self._state_dict


class BaseCallback(ABC):
    """Lớp cơ sở lifecycle Keras-like, được gắn với trainer trước mỗi lượt fit."""

    def __init__(self) -> None:
        """Khởi tạo tham chiếu runtime; callback có thể tái sử dụng giữa các run."""
        self.trainer: Any | None = None
        self.model: nn.Module | None = None
        self.optimizer: Optimizer | None = None

    def set_trainer(self, trainer: Any) -> None:
        """Gắn trainer/model/optimizer hiện tại vào callback."""
        self.trainer = trainer
        self.model = trainer.model
        self.optimizer = trainer.optimizer

    def reset_state(self) -> None:
        """Đặt lại trạng thái theo run; lớp con ghi đè khi có bộ đếm/best score."""

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Được gọi một lần trước epoch đầu tiên."""
        del logs

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Được gọi trước mỗi epoch."""
        del epoch, logs

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Xử lý metrics cuối epoch; trả True để dừng huấn luyện."""
        del epoch, logs
        return False

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Được gọi một lần khi fit kết thúc bình thường hoặc dừng sớm."""
        del logs


class CallbackList:
    """Bộ điều phối lifecycle, tương thích callback legacy của `BaseTrainer`."""

    def __init__(self, callbacks: Sequence[Any] | None = None) -> None:
        """Lưu danh sách callback theo đúng thứ tự cấu hình."""
        self.callbacks = list(callbacks or [])

    def set_trainer(self, trainer: Any) -> None:
        """Gắn callback OOP với trainer và reset bộ đếm cho lần fit hiện tại."""
        trainer.callbacks = self.callbacks
        for callback in self.callbacks:
            if isinstance(callback, BaseCallback):
                callback.set_trainer(trainer)
            reset = getattr(callback, "reset_state", None)
            if callable(reset):
                reset()

    def on_train_begin(self, logs: dict[str, Any]) -> None:
        """Phát lifecycle train-begin theo thứ tự callback."""
        for callback in self.callbacks:
            handler = getattr(callback, "on_train_begin", None)
            if callable(handler):
                handler(dict(logs))

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any]) -> None:
        """Phát lifecycle epoch-begin theo thứ tự callback."""
        for callback in self.callbacks:
            handler = getattr(callback, "on_epoch_begin", None)
            if callable(handler):
                handler(epoch, dict(logs))

    def on_epoch_end(self, epoch: int, logs: dict[str, Any], trainer: Any) -> bool:
        """Phát callback mới hoặc adapter cũ và hợp nhất cờ dừng sớm."""
        stop = False
        for callback in self.callbacks:
            handler = getattr(callback, "on_epoch_end", None)
            if not callable(handler):
                raise TypeError("Callback cần định nghĩa on_epoch_end.")
            if isinstance(callback, BaseCallback):
                requested = handler(epoch, logs)
            else:
                # Tương thích API lịch sử: on_epoch_end(trainer, epoch, logs).
                requested = handler(trainer, epoch, logs)
            stop = stop or bool(requested)
        return stop

    def on_train_end(self, logs: dict[str, Any]) -> None:
        """Phát lifecycle train-end cho mọi callback sau fit."""
        for callback in self.callbacks:
            handler = getattr(callback, "on_train_end", None)
            if callable(handler):
                handler(dict(logs))


class _MonitorCallback(BaseCallback):
    """Cơ sở dùng chung để xác thực monitor, mode và ngưỡng cải thiện."""

    def __init__(self, monitor: str, mode: str, min_delta: float = 0.0) -> None:
        """Khởi tạo tiêu chí cực tiểu/cực đại và sentinel chưa có best."""
        super().__init__()
        if not monitor:
            raise ValueError("monitor không được để trống.")
        if mode not in {"min", "max", "auto"}:
            raise ValueError("mode phải là 'min', 'max' hoặc 'auto'.")
        if min_delta < 0 or not math.isfinite(min_delta):
            raise ValueError("min_delta phải hữu hạn và không âm.")
        self.monitor = monitor
        self.mode = ("max" if any(key in monitor.lower() for key in ("acc", "auc", "precision", "recall")) else "min") if mode == "auto" else mode
        self.min_delta = float(min_delta)
        self.best_score = math.inf if self.mode == "min" else -math.inf

    def is_improved(self, value: float) -> bool:
        """So sánh monitor theo mode và min_delta, bỏ qua NaN/Inf."""
        if not math.isfinite(value):
            return False
        if self.mode == "min":
            return value < self.best_score - self.min_delta
        return value > self.best_score + self.min_delta

    def monitored_value(self, logs: Mapping[str, Any] | None) -> float | None:
        """Lấy monitor từ logs và chuyển về float hữu hạn nếu có."""
        if logs is None or self.monitor not in logs:
            return None
        try:
            value = float(logs[self.monitor])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None


class EarlyStopping(_MonitorCallback):
    """Dừng khi monitor không cải thiện đủ lâu và tùy chọn khôi phục weights."""

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 5,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ) -> None:
        """Cấu hình early stopping theo ngưỡng và số epoch kiên nhẫn."""
        super().__init__(monitor, mode, min_delta)
        if patience < 0:
            raise ValueError("patience phải không âm.")
        self.patience = int(patience)
        self._patience_floor = 1 if self.patience == 0 else self.patience
        self.restore_best_weights = bool(restore_best_weights)
        self.verbose = bool(verbose)
        self.wait = 0
        self.stopped_epoch: int | None = None
        self.best_weights: dict[str, torch.Tensor] | None = None
        self.best_epoch: int | None = None
        self.best_logs: dict[str, Any] | None = None
        self.best_optimizer_state: dict[str, Any] | None = None

    def reset_state(self) -> None:
        """Xóa bộ đếm/best state để cùng callback chạy độc lập ở lượt fit mới."""
        self.best_score = math.inf if self.mode == "min" else -math.inf
        self.wait = 0
        self.stopped_epoch = None
        self.best_weights = None
        self.best_epoch = None
        self.best_logs = None
        self.best_optimizer_state = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Lưu weights tốt nhất hoặc dừng khi wait đạt patience."""
        value = self.monitored_value(logs)
        if value is None:
            return False
        if self.is_improved(value):
            self.best_score = value
            self.wait = 0
            self.best_epoch = epoch
            self.best_logs = dict(logs or {})
            if self.model is None:
                raise RuntimeError("EarlyStopping chưa được gắn với model.")
            self.best_weights = copy.deepcopy(self.model.state_dict())
            if self.optimizer is not None:
                self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            return False

        self.wait += 1
        if self.wait < self._patience_floor:
            return False
        self.stopped_epoch = epoch
        if self.restore_best_weights and self.best_weights is not None and self.model is not None:
            self.model.load_state_dict(self.best_weights)
        if self.trainer is not None:
            self.trainer.stop_training = True
        if self.verbose:
            print(
                f"Dừng sớm tại epoch {epoch}: '{self.monitor}' không cải thiện "
                f"{self.wait} epoch; best={self.best_score:.6g} (epoch {self.best_epoch})."
            )
        return True


class ModelCheckpoint(_MonitorCallback):
    """Lưu checkpoint `.pt` khi monitor cải thiện hoặc tại mỗi epoch."""

    def __init__(
        self,
        filepath: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_weights_only: bool = False,
        sync_to_drive: bool = True,
        verbose: bool = True,
        drive_root: str | Path | None = None,
    ) -> None:
        """Cấu hình đường dẫn checkpoint, tiêu chí lưu và mirror Drive tùy chọn."""
        super().__init__(monitor, mode)
        self.filepath = Path(filepath)
        if self.filepath.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("filepath checkpoint phải có đuôi .pt hoặc .pth.")
        self.save_best_only = bool(save_best_only)
        self.save_weights_only = bool(save_weights_only)
        self.sync_to_drive = bool(sync_to_drive)
        self.verbose = bool(verbose)
        self.drive_root = Path(drive_root) if drive_root is not None else None
        self.best_epoch: int | None = None
        self.saved_epochs: list[int] = []
        self._resolved_filepath: Path | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.best_optimizer_state: dict[str, Any] | None = None

    def reset_state(self) -> None:
        """Đặt lại best score cho run mới nhưng giữ danh sách lưu lịch sử test."""
        self.best_score = math.inf if self.mode == "min" else -math.inf
        self.best_epoch = None
        self.saved_epochs = []
        self._resolved_filepath = None
        self.best_model_state = None
        self.best_optimizer_state = None

    def resolved_filepath(self) -> Path:
        """Trả checkpoint path tuyệt đối sau khi gắn trainer/project root."""
        if self._resolved_filepath is not None:
            return self._resolved_filepath
        if self.filepath.is_absolute():
            return self.filepath
        root = (
            Path(self.trainer.config.get("project_root", Path(__file__).resolve().parents[2]))
            if self.trainer is not None else Path(__file__).resolve().parents[2]
        )
        return root.resolve() / self.filepath

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Lưu model/optimizer metadata qua CheckpointManager vào filepath đích."""
        value = self.monitored_value(logs)
        if value is None:
            return False
        improved = self.is_improved(value)
        if self.save_best_only and not improved:
            return False
        if self.trainer is None or self.model is None or self.optimizer is None:
            raise RuntimeError("ModelCheckpoint chưa được gắn với trainer.")

        from ..logger import CheckpointManager

        project_root = Path(
            self.trainer.config.get("project_root", Path(__file__).resolve().parents[2])
        ).resolve()
        filepath = self.resolved_filepath().resolve()
        self._resolved_filepath = filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)
        manager = CheckpointManager(
            project_root=project_root,
            # CheckpointManager tự nối model_group, vì vậy root là cha của group.
            checkpoint_root=filepath.resolve().parent.parent,
            drive_root=self.drive_root if self.sync_to_drive else None,
        )
        if self.save_weights_only:
            # Cùng schema manager để CheckpointManager.load() vẫn hoạt động.
            optimizer = None
        else:
            optimizer = self.optimizer
        manager.save(
            model=self.model,
            model_group=filepath.parent.name,
            model_name=self.trainer.model_name,
            epoch=epoch,
            optimizer=optimizer,
            best_metric=value,
            seed=self.trainer.seed,
            config={
                **self.trainer.config,
                "monitor": self.monitor,
                "best_score": value,
                "mode": self.mode,
            },
            filename=filepath.name,
            metadata={"monitor": self.monitor, "best_score": value},
        )
        if improved:
            self.best_score = value
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.saved_epochs.append(epoch)
        if self.verbose:
            print(f"Đã lưu checkpoint epoch {epoch}: {filepath} ({self.monitor}={value:.6g}).")
        return False

    def save_best_checkpoint(self) -> None:
        """Ghi best model và optimizer state snapshot mà không mutate model hiện hành."""
        if self.best_model_state is None or self.model is None or self.optimizer is None:
            return
        from ..logger import CheckpointManager

        path = self.resolved_filepath()
        root = Path(
            self.trainer.config.get("project_root", Path(__file__).resolve().parents[2])
        ).resolve()
        manager = CheckpointManager(
            project_root=root,
            checkpoint_root=path.parent.parent,
            drive_root=self.drive_root if self.sync_to_drive else None,
        )
        optimizer_snapshot = copy.deepcopy(
            self.best_optimizer_state or self.optimizer.state_dict()
        )
        # Recurrent/Adam state giữ best epoch; learning-rate group theo scheduler
        # đến cuối run để checkpoint phản ánh schedule thực sự đã áp dụng.
        for saved_group, current_group in zip(
            optimizer_snapshot["param_groups"], self.optimizer.param_groups
        ):
            saved_group["lr"] = current_group["lr"]
        manager.save(
            model=_StateDictSnapshot(self.model, self.best_model_state),
            model_group=path.parent.name,
            model_name=self.trainer.model_name,
            epoch=int(self.best_epoch or 0),
            optimizer=_OptimizerStateSnapshot(self.optimizer, optimizer_snapshot),
            best_metric=self.best_score,
            seed=self.trainer.seed,
            config={**self.trainer.config, "monitor": self.monitor, "best_score": self.best_score, "mode": self.mode},
            filename=path.name,
            metadata={"monitor": self.monitor, "best_score": self.best_score},
        )

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Khôi phục model weights tốt nhất; optimizer/LR hiện hành không đổi."""
        del logs
        if self.best_model_state is None or self.model is None or self.optimizer is None:
            return
        self.model.load_state_dict(self.best_model_state)


class ReduceLROnPlateau(_MonitorCallback):
    """Giảm learning rate theo monitor, patience và cooldown."""

    def __init__(
        self,
        optimizer: Optimizer,
        monitor: str = "val_loss",
        factor: float = 0.5,
        patience: int = 3,
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 1e-6,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        """Cấu hình giảm LR; optimizer được truyền tường minh hoặc khớp trainer."""
        super().__init__(monitor, mode, min_delta)
        if not 0.0 < factor < 1.0:
            raise ValueError("factor phải nằm nghiêm ngặt trong (0,1).")
        if patience < 0 or cooldown < 0:
            raise ValueError("patience và cooldown phải không âm.")
        if min_lr < 0 or not math.isfinite(min_lr):
            raise ValueError("min_lr phải hữu hạn và không âm.")
        self.configured_optimizer = optimizer
        self.factor = float(factor)
        self.patience = int(patience)
        self._patience_floor = 1 if self.patience == 0 else self.patience
        self.cooldown = int(cooldown)
        self.min_lr = float(min_lr)
        self.verbose = bool(verbose)
        self.wait = 0
        self.cooldown_counter = 0
        self.lr_changes: list[tuple[int, float, float]] = []

    def reset_state(self) -> None:
        """Reset plateau/cooldown counters khi bắt đầu fit mới."""
        self.best_score = math.inf if self.mode == "min" else -math.inf
        self.wait = 0
        self.cooldown_counter = 0
        self.lr_changes = []

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Giảm mọi param group khi chuỗi plateau vượt ngưỡng patience."""
        optimizer = self.configured_optimizer
        if self.optimizer is not None and optimizer is not self.optimizer:
            raise ValueError("Optimizer callback không trùng optimizer của BaseTrainer.")
        value = self.monitored_value(logs)
        if value is None:
            return False
        if self.is_improved(value):
            self.best_score = value
            self.wait = 0
            return False

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
            return False
        self.wait += 1
        if self.wait < self._patience_floor:
            return False

        self.wait = 0
        changed: list[tuple[float, float]] = []
        for group in optimizer.param_groups:
            old_lr = float(group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                group["lr"] = new_lr
                changed.append((old_lr, new_lr))
        if changed:
            self.cooldown_counter = self.cooldown
            for old_lr, new_lr in changed:
                self.lr_changes.append((epoch, old_lr, new_lr))
            if self.verbose:
                print(
                    f"Giảm learning rate tại epoch {epoch}: "
                    + ", ".join(f"{old:.6g} → {new:.6g}" for old, new in changed)
                )
        return False