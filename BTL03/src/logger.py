"""Ghi log thực nghiệm PyTorch ra CSV, TensorBoard và checkpoint."""

from __future__ import annotations

import csv
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Ghi log từng epoch và tổng kết run theo schema BTL03."""

    MASTER_COLUMNS = (
        "Timestamp", "Run_ID", "Model_Group", "Model_Name", "Exp_Type", "Seed",
        "Optimizer", "LR", "Batch_Size", "Epochs", "Best_Epoch", "Train_Loss",
        "Val_Loss", "Train_Metric", "Val_Metric", "Test_Metric", "Train_Time_Sec",
        "Member_Alias",
    )
    EPOCH_COLUMNS = (
        "Epoch", "Train_Loss", "Val_Loss", "Train_Metric", "Val_Metric",
        "Learning_Rate", "Epoch_Time_Sec",
    )
    _CSV_LOCK = threading.Lock()

    def __init__(
        self,
        model_group: str,
        model_name: str,
        exp_type: str,
        seed: int = 42,
        optimizer: str = "NOT_RECORDED",
        learning_rate: float | None = None,
        batch_size: int | None = None,
        epochs: int | None = None,
        member_alias: str | None = None,
        member_id: str | None = None,
        project_root: str | Path | None = None,
        csv_root: str | Path | None = None,
        tensorboard_root: str | Path | None = None,
    ) -> None:
        """Tạo run directory, writer và CSV epoch.

        Args:
            model_group: Nhóm mô hình, ví dụ ``02_cnn_mlp``.
            model_name: Tên mô hình, ví dụ ``EmotionCNN``.
            exp_type: Loại thực nghiệm; smoke-test phải ghi ``SmokeTest``.
            seed: Seed của lần chạy; baseline chuẩn của dự án là 42.
            optimizer: Tên optimizer để đưa vào master CSV/HParams.
            learning_rate: Learning rate; dùng ``None`` nếu không áp dụng.
            batch_size: Batch size; dùng ``None`` nếu không áp dụng.
            epochs: Epoch tối đa được cấu hình.
            member_alias: Alias thành viên tường minh; ưu tiên cao nhất.
            member_id: Khóa thành viên trong `.env`, dạng ``MEMBER_04``.
            project_root: Gốc dự án BTL03; mặc định dò từ file này.
            csv_root: Thư mục CSV; mặc định ``experiments/csv_logs``.
            tensorboard_root: Gốc TensorBoard; mặc định ``experiments/tensorboard``.

        Raises:
            ValueError: Metadata không hợp lệ hoặc thiếu định danh thành viên.
            FileExistsError: Run directory cùng timestamp đã tồn tại.
        """
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
        self.model_group = self._safe_component(model_group, "model_group")
        self.model_name = self._safe_component(model_name, "model_name")
        self.exp_type = self._safe_component(exp_type, "exp_type")
        self.seed = int(seed)
        self.optimizer = str(optimizer)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.member_alias = self._resolve_member_alias(member_alias, member_id)

        self.csv_root = self._resolve_path(csv_root, "experiments/csv_logs")
        self.tensorboard_root = self._resolve_path(tensorboard_root, "experiments/tensorboard")
        self.timestamp = datetime.now(timezone.utc).replace(microsecond=0)
        self.timestamp_text = self.timestamp.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.exp_type}_s{self.seed}_{self.timestamp_text}"
        self.run_dir = (
            self.tensorboard_root / self.model_group / self.model_name / self.run_id
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.epoch_csv_path = self.csv_root / f"{self.run_id}_epochs.csv"
        self.master_csv_path = self.csv_root / "master_experiment_runs.csv"
        self.csv_root.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header(self.epoch_csv_path, self.EPOCH_COLUMNS)
        self._ensure_csv_header(self.master_csv_path, self.MASTER_COLUMNS)
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        self._last_epoch = 0
        self._last_values: dict[str, float | None] = {
            "train_loss": None, "val_loss": None,
            "train_metric": None, "val_metric": None,
            "learning_rate": learning_rate,
        }
        self._epoch_times: list[float] = []
        self._finished = False
        self._closed = False

    @staticmethod
    def _safe_component(value: str, field_name: str) -> str:
        """Chỉ chấp nhận thành phần đường dẫn an toàn, không chứa traversal."""
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
        if not normalized or normalized in {".", ".."}:
            raise ValueError(f"{field_name} phải là tên định danh đường dẫn hợp lệ.")
        return normalized

    def _resolve_path(self, value: str | Path | None, default: str) -> Path:
        """Đổi đường dẫn cấu hình tương đối thành đường dẫn tuyệt đối dự án."""
        path = Path(value) if value is not None else Path(default)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def _resolve_member_alias(self, alias: str | None, member_id: str | None) -> str:
        """Lấy alias tường minh, biến môi trường hoặc NAME từ `.env`; không đoán."""
        value = alias or os.environ.get("MEMBER_ALIAS")
        if not value and member_id:
            key = member_id if member_id.startswith("MEMBER_") else f"MEMBER_{member_id}"
            env_path = self.project_root / ".env"
            if env_path.is_file():
                for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                    match = re.match(r"\s*([^#=]+)\s*=\s*(.*)\s*$", line)
                    if match and match.group(1).strip() == f"{key}_NAME":
                        value = match.group(2).strip().strip('"').strip("'")
                        break
        if not value or re.search(r"(AI Agent|Copilot|Claude|Codex|Cursor)", value, re.I):
            raise ValueError(
                "Thiếu Member_Alias thật. Truyền member_alias, đặt MEMBER_ALIAS, "
                "hoặc cung cấp member_id có NAME hợp lệ trong .env."
            )
        return value.strip()

    @staticmethod
    def _ensure_csv_header(path: Path, columns: tuple[str, ...]) -> None:
        """Tạo CSV có header hoặc từ chối file đã có schema không khớp."""
        if path.exists() and path.stat().st_size:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
            if tuple(header) != columns:
                raise ValueError(
                    f"Schema CSV không khớp tại {path}: expected={columns}, actual={header}."
                )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerow(columns)

    @staticmethod
    def _number(value: float | int | None) -> str:
        """Định dạng số ổn định; ô chưa có dữ liệu để trống."""
        return "" if value is None else f"{float(value):.6f}"

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metric: float | None = None,
        val_metric: float | None = None,
        learning_rate: float | None = None,
        epoch_time_sec: float | None = None,
        metric_name: str = "Accuracy",
    ) -> None:
        """Ghi scalar TensorBoard và một hàng CSV cho epoch.

        Args:
            epoch: Số epoch nguyên dương, tăng nghiêm ngặt.
            train_loss: Loss trên tập huấn luyện.
            val_loss: Loss trên tập validation.
            train_metric: Metric train tùy chọn.
            val_metric: Metric validation tùy chọn.
            learning_rate: Learning rate tại epoch; mặc định metadata run.
            epoch_time_sec: Thời gian epoch tính bằng giây.
            metric_name: Tên độ đo như Accuracy, MAE hoặc Pixel_Acc.
        """
        if self._closed or self._finished:
            raise RuntimeError("Không thể ghi epoch sau khi run đã kết thúc/đóng.")
        if epoch <= self._last_epoch:
            raise ValueError("epoch phải là số nguyên dương và tăng nghiêm ngặt.")
        metric_tag = self._safe_component(metric_name, "metric_name")
        lr = self.learning_rate if learning_rate is None else learning_rate
        values = {
            "train_loss": float(train_loss), "val_loss": float(val_loss),
            "train_metric": None if train_metric is None else float(train_metric),
            "val_metric": None if val_metric is None else float(val_metric),
            "learning_rate": None if lr is None else float(lr),
        }

        self.writer.add_scalar("Loss/Train", values["train_loss"], epoch)
        self.writer.add_scalar("Loss/Validation", values["val_loss"], epoch)
        if values["train_metric"] is not None:
            self.writer.add_scalar(f"Metric/Train/{metric_tag}", values["train_metric"], epoch)
        if values["val_metric"] is not None:
            self.writer.add_scalar(f"Metric/Validation/{metric_tag}", values["val_metric"], epoch)
        if values["learning_rate"] is not None:
            self.writer.add_scalar("Hyperparameters/Learning_Rate", values["learning_rate"], epoch)
        if epoch_time_sec is not None:
            self.writer.add_scalar("Timing/Epoch_Time_Sec", float(epoch_time_sec), epoch)

        row = {
            "Epoch": int(epoch),
            "Train_Loss": self._number(train_loss),
            "Val_Loss": self._number(val_loss),
            "Train_Metric": self._number(train_metric),
            "Val_Metric": self._number(val_metric),
            "Learning_Rate": self._number(lr),
            "Epoch_Time_Sec": self._number(epoch_time_sec),
        }
        with self._CSV_LOCK, self.epoch_csv_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.EPOCH_COLUMNS).writerow(row)
        self._last_epoch = int(epoch)
        self._last_values = values
        if epoch_time_sec is not None:
            self._epoch_times.append(float(epoch_time_sec))

    def finish_run(
        self,
        best_epoch: int | None = None,
        train_loss: float | None = None,
        val_loss: float | None = None,
        train_metric: float | None = None,
        val_metric: float | None = None,
        test_metric: float | None = None,
        train_time_sec: float | None = None,
        metric_name: str = "Accuracy",
    ) -> Path:
        """Ghi HParams và thêm đúng một dòng tổng kết 18 cột vào master CSV."""
        if self._closed:
            raise RuntimeError("Logger đã đóng.")
        if self._finished:
            raise RuntimeError("Run đã được tổng kết; không ghi trùng master row.")
        if self._last_epoch < 1:
            raise RuntimeError("Không thể tổng kết run chưa ghi epoch nào.")
        if best_epoch is None:
            raise ValueError("best_epoch phải được truyền tường minh; không suy diễn từ epoch cuối.")
        metrics = {
            "Train_Loss": self._last_values["train_loss"] if train_loss is None else float(train_loss),
            "Val_Loss": self._last_values["val_loss"] if val_loss is None else float(val_loss),
            "Test_Metric": None if test_metric is None else float(test_metric),
        }
        if train_metric is None:
            train_metric = self._last_values["train_metric"]
        if val_metric is None:
            val_metric = self._last_values["val_metric"]
        if train_time_sec is None and self._epoch_times:
            train_time_sec = sum(self._epoch_times)

        hparams: dict[str, str | int | float] = {
            "Model_Group": self.model_group,
            "Model_Name": self.model_name,
            "Exp_Type": self.exp_type,
            "Seed": self.seed,
            "Optimizer": self.optimizer,
            "LR": self.learning_rate if self.learning_rate is not None else "NOT_RECORDED",
            "Batch_Size": self.batch_size if self.batch_size is not None else "NOT_RECORDED",
            "Epochs": self.epochs if self.epochs is not None else self._last_epoch,
        }
        hparam_metrics = {
            "hparam/Train_Loss": float(metrics["Train_Loss"]),
            "hparam/Val_Loss": float(metrics["Val_Loss"]),
        }
        for key, value in (("Train_Metric", train_metric), ("Val_Metric", val_metric), ("Test_Metric", test_metric)):
            if value is not None:
                hparam_metrics[f"hparam/{key}"] = float(value)
        self.writer.add_hparams(hparams, hparam_metrics)
        self.writer.flush()

        row = {
            "Timestamp": self.timestamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "Run_ID": self.run_id,
            "Model_Group": self.model_group,
            "Model_Name": self.model_name,
            "Exp_Type": self.exp_type,
            "Seed": self.seed,
            "Optimizer": self.optimizer,
            "LR": self._number(self.learning_rate),
            "Batch_Size": self.batch_size if self.batch_size is not None else "",
            "Epochs": self.epochs if self.epochs is not None else self._last_epoch,
            "Best_Epoch": int(best_epoch),
            "Train_Loss": self._number(metrics["Train_Loss"]),
            "Val_Loss": self._number(metrics["Val_Loss"]),
            "Train_Metric": self._number(train_metric),
            "Val_Metric": self._number(val_metric),
            "Test_Metric": self._number(test_metric),
            "Train_Time_Sec": self._number(train_time_sec),
            "Member_Alias": self.member_alias,
        }
        with self._CSV_LOCK:
            with self.master_csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                existing_rows = list(csv.DictReader(stream))
            if any(existing.get("Run_ID") == self.run_id for existing in existing_rows):
                raise RuntimeError(f"Run_ID đã tồn tại trong master CSV: {self.run_id}")
            with self.master_csv_path.open("a", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=self.MASTER_COLUMNS).writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
        self._finished = True
        return self.master_csv_path

    def close(self) -> None:
        """Flush và đóng SummaryWriter đúng một lần."""
        if not self._closed:
            self.writer.flush()
            self.writer.close()
            self._closed = True

    def __enter__(self) -> "ExperimentLogger":
        """Cho phép quản lý vòng đời logger bằng context manager."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Đóng writer an toàn khi thoát khối context manager."""
        self.close()


class CheckpointManager:
    """Lưu checkpoint PyTorch cục bộ và mirror tùy chọn lên Drive đã mount."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        checkpoint_root: str | Path | None = None,
        drive_root: str | Path | None = None,
    ) -> None:
        """Khởi tạo thư mục checkpoint; Drive root có thể truyền từ adapter.

        Args:
            project_root: Gốc dự án BTL03; mặc định dò theo vị trí module.
            checkpoint_root: Gốc cục bộ; mặc định `outputs/checkpoints`.
            drive_root: Thư mục MyDrive/BTL03 đã mount; có thể lấy từ biến
                `BTL03_GOOGLE_DRIVE_ROOT`. Không tự mount hoặc truy cập mạng.
        """
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
        local = Path(checkpoint_root) if checkpoint_root is not None else Path("outputs/checkpoints")
        self.checkpoint_root = local.resolve() if local.is_absolute() else (self.project_root / local).resolve()
        configured_drive = drive_root or os.environ.get("BTL03_GOOGLE_DRIVE_ROOT")
        if configured_drive is None:
            configured_drive = self._discover_configured_drive_root()
        self.drive_root = Path(configured_drive).resolve() if configured_drive else None

    def _discover_configured_drive_root(self) -> Path | None:
        """Đọc mount root từ YAML; chỉ kích hoạt mirror khi mount đã tồn tại.

        Đường dẫn nền tảng không được hardcode trong lớp logger. Colab adapter
        có thể mount Drive; Kaggle/local runtime không có mount thì trả None.
        """
        config_path = self.project_root / "configs" / "paths.yaml"
        if not config_path.is_file():
            return None
        try:
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            configured = config.get("project", {}).get("drive_mount_root")
            if not configured:
                return None
            candidate = Path(configured).expanduser()
            if not candidate.is_absolute():
                candidate = self.project_root / candidate
            return candidate.resolve() if candidate.is_dir() else None
        except (OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _safe_component(value: str, field_name: str) -> str:
        """Lọc tên nhóm/file để không cho phép traversal đường dẫn."""
        component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
        if not component or component in {".", ".."}:
            raise ValueError(f"{field_name} không phải thành phần đường dẫn hợp lệ.")
        return component

    def save(
        self,
        model: nn.Module,
        model_group: str,
        model_name: str,
        epoch: int,
        optimizer: Optimizer | None = None,
        best_metric: float | None = None,
        seed: int = 42,
        config: Mapping[str, Any] | None = None,
        filename: str | None = None,
    ) -> Path:
        """Lưu state dict và metadata, rồi mirror nếu Drive đã tồn tại.

        Args:
            model: Mô hình PyTorch cần lưu.
            model_group: Nhóm mô hình dùng làm thư mục con.
            model_name: Tên mô hình dùng tạo tên checkpoint mặc định.
            epoch: Epoch tạo checkpoint.
            optimizer: Optimizer tùy chọn để lưu state.
            best_metric: Metric tốt nhất đến thời điểm lưu.
            seed: Seed của run, mặc định 42.
            config: Cấu hình primitive/serializable của run.
            filename: Tên file tùy chọn; đuôi `.pt` hoặc `.pth`.

        Returns:
            Path: Đường dẫn checkpoint cục bộ.

        Raises:
            ValueError: Metadata hoặc filename không hợp lệ.
        """
        if epoch < 0:
            raise ValueError("epoch phải không âm.")
        group = self._safe_component(model_group, "model_group")
        name = self._safe_component(model_name, "model_name")
        checkpoint_name = filename or f"{name}_best.pth"
        if Path(checkpoint_name).name != checkpoint_name or Path(checkpoint_name).suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("filename phải là tên file đơn có đuôi .pt hoặc .pth.")

        destination = self.checkpoint_root / group / checkpoint_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "best_metric": None if best_metric is None else float(best_metric),
            "seed": int(seed),
            "config": dict(config or {}),
        }
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        self._mirror_to_drive(destination, group)
        return destination

    def _mirror_to_drive(self, source: Path, model_group: str) -> Path | None:
        """Sao chép checkpoint vào Drive đã mount; không tự mount Drive."""
        if self.drive_root is None or not self.drive_root.is_dir():
            return None
        destination = self.drive_root / "checkpoints" / model_group / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @staticmethod
    def load(
        path: str | Path,
        model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Nạp checkpoint tin cậy và tùy chọn khôi phục model/optimizer."""
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        required = {"epoch", "model_state_dict", "optimizer_state_dict", "best_metric", "seed", "config"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Checkpoint thiếu metadata bắt buộc: {sorted(missing)}")
        if model is not None:
            model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and payload["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload