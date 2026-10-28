"""Vòng lặp huấn luyện PyTorch thủ công và chọn mô hình chỉ dựa trên tập validation (T0303)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset, DataLoader, Subset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EarlyStopping:
    """Theo dõi sự cải thiện đáng kể của validation loss độc lập với việc lưu checkpoint tốt nhất."""

    def __init__(
        self, patience: int, min_delta: float, mode: str = "min"
    ) -> None:
        """Khởi tạo mốc theo dõi validation loss.

        Tham số:
            patience: Số epoch liên tiếp không cải thiện để kích hoạt dừng sớm.
            min_delta: Ngưỡng cải thiện tối thiểu (không âm, hữu hạn).
            mode: Chỉ chấp nhận ``min`` cho validation loss.

        Ngoại lệ:
            ValueError: Nếu bất kỳ thiết lập nào vi phạm giao thức đánh giá loss.
        """
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise ValueError("patience phải là một số nguyên dương.")
        if patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("patience hoặc min_delta không hợp lệ.")
        if mode != "min":
            raise ValueError("Chỉ hỗ trợ mode='min' cho validation loss.")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = math.inf
        self.early_stop = False

    def __call__(self, current_loss: float) -> bool:
        """Cập nhật bộ đếm dựa trên mức độ cải thiện so với điểm số tốt nhất.

        Tham số:
            current_loss: Giá trị validation loss trung bình (hữu hạn) của một epoch.

        Trả về:
            True nếu loss cải thiện nhiều hơn min_delta, ngược lại False.
            Đọc thuộc tính early_stop để nhận biết quyết định dừng sớm sau patience epoch.

        Ngoại lệ:
            ValueError: Nếu current_loss không phải số hữu hạn.
        """
        if not math.isfinite(current_loss):
            raise ValueError("Validation loss phải là một số hữu hạn.")
        improvement = current_loss < self.best_score - self.min_delta
        if improvement:
            self.best_score = current_loss
            self.counter = 0
        else:
            self.counter += 1
        self.early_stop = self.counter >= self.patience
        return improvement


class Trainer:
    """Huấn luyện mô hình hồi quy đầu ra vô hướng, tuyệt đối không nhận test loader.

    Các batch dữ liệu chứa đầu vào (B, 2) và nhãn target (B, 1). Các tập dữ liệu
    có gắn metadata là test sẽ bị từ chối, bao gồm cả khi bọc qua Subset hay ConcatDataset.
    Phương thức fit sẽ reset lại lịch sử theo dõi, không reset trọng số khởi tạo của mô hình
    hay trạng thái optimizer; cần khởi tạo đối tượng mới cho các lần thực nghiệm độc lập.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        criterion: nn.Module | None = None,
        device: torch.device | None = None,
        checkpoint_dir: Path | str = "outputs/checkpoints",
        early_stopping_patience: int | None = None,
        min_delta: float | None = None,
        *,
        train_config: Mapping[str, Any] | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        """Cấu hình bộ huấn luyện (Trainer), hỗ trợ tương thích API cũ.

        Tham số:
            model: Mô hình hồi quy có kích thước đầu ra (B, 1).
            optimizer: Optimizer cho mô hình, hoặc None để tự tạo từ cấu hình config.
                Cần chuyển mô hình lên device trước khi khởi tạo optimizer.
            criterion: Hàm loss vô hướng (mean), mặc định là MSELoss.
            device: Thiết bị tính toán; mặc định là CPU.
            checkpoint_dir: Thư mục lưu checkpoint (Path hoặc chuỗi đường dẫn tương đối).
            early_stopping_patience: Ghi đè giá trị patience trong config (nếu truyền).
            min_delta: Ghi đè ngưỡng cải thiện tối thiểu trong config (nếu truyền).
            train_config: Từ điển cấu hình huấn luyện; mặc định đọc từ baseline YAML.
            checkpoint_path: Đường dẫn checkpoint cụ thể tùy chọn (tương thích cũ).

        Ngoại lệ:
            ValueError: Nếu reduction của hàm loss, optimizer hoặc các thiết lập không hợp lệ.
        """
        if train_config is None:
            with (PROJECT_ROOT / "configs/baseline.yaml").open(
                encoding="utf-8"
            ) as stream:
                train_config = yaml.safe_load(stream)["training_baseline"]
        self.config = dict(train_config)
        self.device = torch.device("cpu") if device is None else device
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss() if criterion is None else criterion
        self.criterion = self.criterion.to(self.device)
        if getattr(self.criterion, "reduction", "mean") != "mean":
            raise ValueError("criterion phải trả về giá trị loss trung bình vô hướng (mean).")
        self.optimizer = optimizer if optimizer is not None else (
            self._get_optimizer()
        )
        model_ids = {id(p) for p in self.model.parameters()}
        optimizer_ids = {
            id(p) for group in self.optimizer.param_groups
            for p in group["params"]
        }
        if not optimizer_ids or not optimizer_ids <= model_ids:
            raise ValueError("Optimizer phải tham chiếu đúng các tham số của mô hình được truyền vào.")
        early = self.config["early_stopping"]
        self.patience = (
            early["patience"] if early_stopping_patience is None
            else early_stopping_patience
        )
        self.min_delta = early["min_delta"] if min_delta is None else min_delta
        self.use_early_stopping = early["enabled"]
        if early.get("monitor", "val_loss") != "val_loss":
            raise ValueError("Early stopping bắt buộc phải theo dõi val_loss.")
        self.early_stopping = EarlyStopping(
            self.patience, self.min_delta, early.get("mode", "min")
        )
        self.epochs = self.config["epochs"]
        directory = Path(checkpoint_dir)
        self.checkpoint_dir = (
            directory if directory.is_absolute() else PROJECT_ROOT / directory
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "best_model.pt"
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            self.checkpoint_path = (
                path if path.is_absolute() else PROJECT_ROOT / path
            )
        self._default_checkpoint_path = self.checkpoint_path
        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "lr": []
        }
        self.best_val_loss = math.inf
        self.best_epoch = 0
        self.epochs_run = 0
        self.stopped_early = False

    def _get_optimizer(self) -> Optimizer:
        """Khởi tạo optimizer dựa trên cấu hình huấn luyện đã giải nạp.

        Trả về:
            Optimizer Adam hoặc SGD gắn với tham số mô hình trên device đã chọn.

        Ngoại lệ:
            ValueError: Nếu tên optimizer trong cấu hình không được hỗ trợ.
        """
        cfg = self.config
        kwargs = {"lr": cfg["learning_rate"],
                  "weight_decay": cfg["weight_decay"]}
        if cfg["optimizer"] == "adam":
            return torch.optim.Adam(
                self.model.parameters(), **kwargs,
                betas=tuple(cfg["adam_betas"]), eps=cfg["adam_eps"],
                amsgrad=cfg["adam_amsgrad"],
            )
        if cfg["optimizer"] == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), **kwargs,
                momentum=cfg["sgd_momentum"], nesterov=cfg["sgd_nesterov"],
            )
        raise ValueError(f"Không hỗ trợ optimizer: {cfg['optimizer']}")

    def _check_loader(self, dataloader: DataLoader, role: str) -> None:
        """Từ chối dữ liệu test hoặc các tập dữ liệu bị gắn nhãn sai vai trò.

        Tham số:
            dataloader: DataLoader cung cấp các cặp Tensor.
            role: Vai trò phân chia kỳ vọng ('train' hoặc 'val').

        Ngoại lệ:
            ValueError: Nếu metadata phát hiện tập dữ liệu không tương thích.
        """
        pending = [dataloader.dataset]
        while pending:
            dataset = pending.pop()
            split = getattr(dataset, "split", None)
            source = getattr(dataset, "csv_path", None)
            if source is not None and Path(source).stem.lower() == "test":
                raise ValueError("Nghiêm cấm đưa dữ liệu tập test vào Trainer.")
            if split is not None and split != role:
                raise ValueError(f"Kỳ vọng split '{role}', nhưng nhận được '{split}'.")
            if isinstance(dataset, Subset):
                pending.append(dataset.dataset)
            elif isinstance(dataset, ConcatDataset):
                pending.extend(dataset.datasets)

    def _batch_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Tính giá trị loss trung bình vô hướng hữu hạn, tránh lỗi broadcasting sai chiều.

        Tham số:
            inputs: Tensor đầu vào trên device, kích thước (B, 2).
            targets: Tensor nhãn target trên device, kích thước (B, 1).

        Trả về:
            Giá trị loss vô hướng có thể tính đạo hàm (differentiable).

        Ngoại lệ:
            ValueError: Nếu kích thước Tensor hoặc giá trị số học không hợp lệ.
        """
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError("inputs phải có kích thước (batch_size, 2).")
        if targets.shape != (inputs.shape[0], 1):
            raise ValueError("targets phải có kích thước (batch_size, 1).")
        if (not torch.isfinite(inputs).all()
                or not torch.isfinite(targets).all()):
            raise ValueError("inputs và targets phải chứa các giá trị hữu hạn.")
        outputs = self.model(inputs)
        if (outputs.shape != targets.shape
                or not torch.isfinite(outputs).all()):
            raise ValueError("Kích thước đầu ra hoặc giá trị dự đoán không hợp lệ.")
        loss = self.criterion(outputs, targets)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("Hàm loss phải trả về một giá trị vô hướng hữu hạn.")
        return loss

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        """Thực hiện một epoch huấn luyện với các bước forward và backward tường minh.

        Tham số:
            dataloader: DataLoader tập huấn luyện, kích thước các batch (B, 2) và (B, 1).

        Trả về:
            Giá trị loss trung bình tính theo trọng số số lượng mẫu thực tế.

        Ngoại lệ:
            ValueError: Nếu gặp dữ liệu test, dataloader rỗng hoặc gradient không hữu hạn.
        """
        self._check_loader(dataloader, "train")
        self.model.train()
        total, count = 0.0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            loss = self._batch_loss(inputs, targets)
            loss.backward()
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise ValueError("Gradient không hữu hạn (NaN hoặc Inf); hủy quá trình.")
            if self.config.get("gradient_clipping", False):
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item() * inputs.shape[0]
            count += inputs.shape[0]
        if count == 0:
            raise ValueError("DataLoader huấn luyện không cung cấp mẫu dữ liệu nào.")
        return total / count

    def validate(self, dataloader: DataLoader) -> float:
        """Tính toán loss trên tập validation mà không cập nhật gradient.

        Tham số:
            dataloader: DataLoader tập validation, kích thước các batch (B, 2) và (B, 1).
                Tuyệt đối không được chứa dữ liệu của tập test độc lập.

        Trả về:
            Giá trị validation loss trung bình tính theo trọng số số lượng mẫu.

        Ngoại lệ:
            ValueError: Nếu split không tương thích, batch rỗng hoặc không hợp lệ.
        """
        self._check_loader(dataloader, "val")
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                loss = self._batch_loss(inputs, targets)
                total += loss.item() * inputs.shape[0]
                count += inputs.shape[0]
        if count == 0:
            raise ValueError("DataLoader validation không cung cấp mẫu dữ liệu nào.")
        return total / count

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Bí danh (alias) tương thích với tên phương thức huấn luyện trước đây.

        Tham số:
            dataloader: DataLoader tập huấn luyện với các cặp kích thước (B, 2) và (B, 1).

        Trả về:
            Loss trung bình trên tập huấn luyện.
        """
        return self.train_one_epoch(dataloader)

    def evaluate(self, dataloader: DataLoader) -> float:
        """Bí danh tương thích cũ chỉ chạy trên tập validation, tuyệt đối không dùng cho test.

        Tham số:
            dataloader: DataLoader tập xác thực với các cặp kích thước (B, 2) và (B, 1).

        Trả về:
            Loss trung bình trên tập xác thực.
        """
        return self.validate(dataloader)

    def save_checkpoint(self, epoch: int) -> Path:
        """Lưu trữ an toàn (atomic) trọng số mô hình, optimizer và metadata validation.

        Tham số:
            epoch: Epoch tốt nhất tính từ 1; best_val_loss phải là số hữu hạn.

        Trả về:
            Path của file checkpoint tốt nhất (.pt) đã lưu.
        """
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(".pt.tmp")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "history": {k: list(v) for k, v in self.history.items()},
        }
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(self.checkpoint_path)
        finally:
            temporary.unlink(missing_ok=True)
        return self.checkpoint_path

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int | None = None,
        checkpoint_name: str = "best_model.pt",
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Huấn luyện và khôi phục trọng số tại epoch có validation loss nhỏ nhất nghiêm ngặt.

        Tham số:
            train_loader: DataLoader tập train, kích thước batch (B, 2) và (B, 1).
            val_loader: DataLoader tập validation độc lập, tuyệt đối không phải tập test.
            epochs: Giới hạn số epoch, mặc định lấy theo cấu hình config.
            checkpoint_name: Tên file .pt đơn thuần nằm trong checkpoint_dir.
                Đường dẫn checkpoint_path cũ được giữ nếu tên này là mặc định.
            verbose: Có in thông tin tiến trình loss, learning rate từng epoch hay không.

        Trả về:
            Lịch sử huấn luyện gồm train_loss, val_loss và learning rate mỗi epoch.
            Mô hình và optimizer đều được phục hồi về trạng thái epoch tốt nhất.

        Ngoại lệ:
            ValueError: Nếu số epoch, tên file, tập dữ liệu hoặc batch không hợp lệ.
        """
        limit = self.epochs if epochs is None else epochs
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("epochs phải là số nguyên dương.")
        name = Path(checkpoint_name)
        if (name.name != checkpoint_name or "\\" in checkpoint_name
                or name.suffix != ".pt"):
            raise ValueError("checkpoint_name phải là tên file có phần mở rộng .pt.")
        self.checkpoint_path = (
            self._default_checkpoint_path if checkpoint_name == "best_model.pt"
            else self.checkpoint_dir / name
        )
        self._check_loader(train_loader, "train")
        self._check_loader(val_loader, "val")
        self.history = {"train_loss": [], "val_loss": [], "lr": []}
        self.best_val_loss, self.best_epoch = math.inf, 0
        self.epochs_run, self.stopped_early = 0, False
        self.early_stopping = EarlyStopping(self.patience, self.min_delta)
        for epoch in range(1, limit + 1):
            lr = float(self.optimizer.param_groups[0]["lr"])
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise ValueError("Loss trong epoch không hữu hạn; hủy quá trình huấn luyện.")
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)
            self.epochs_run = epoch
            if val_loss < self.best_val_loss:
                self.best_val_loss, self.best_epoch = val_loss, epoch
                self.save_checkpoint(epoch)
            self.early_stopping(val_loss)
            if verbose:
                print(f"Epoch {epoch:04d}: train_loss={train_loss:.10f} "
                      f"val_loss={val_loss:.10f} lr={lr:g}")
            if self.use_early_stopping and self.early_stopping.early_stop:
                self.stopped_early = True
                if verbose:
                    print(f"Early stopping kích hoạt tại epoch {epoch}.")
                break
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.eval()
        return self.history