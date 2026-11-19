"""Module Trainer huấn luyện và đánh giá mô hình chuẩn PyTorch cho FER2013.

Tuân thủ nghiêm ngặt hiến pháp AGENTS.md:
- Thiết kế 100% hướng đối tượng (OOP) với Docstring Tiếng Việt đầy đủ.
- Tự động khóa cứng Random Seed=42 với `seed_everything(42)`.
- Tích hợp MLflowLogger chuẩn 24 cột theo dõi quá trình huấn luyện và đánh giá.
- Cơ chế EarlyStopping chống học vẹt và lưu giữ Checkpoint tốt nhất (Best Checkpoint).
- Đánh giá đa chiều với Accuracy và Macro-F1 (đặc biệt quan trọng với lớp Disgust).
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import get_dataloaders
from src.models import LargeCNN, MediumCNN, SmallCNN
from src.train.mlflow_logger import MLflowLogger
from src.train.utils import count_parameters, get_device, seed_everything


class Trainer:
    """Lớp điều phối toàn bộ quy trình huấn luyện, đánh giá và lưu trữ thực nghiệm."""

    def __init__(self, config: Union[str, Path, Dict[str, Any]]) -> None:
        """Khởi tạo Trainer với tệp cấu hình YAML hoặc từ điển tham số.

        Args:
            config: Đường dẫn tệp cấu hình YAML hoặc Dict chứa siêu tham số.
        """
        # 1. Đọc và nạp cấu hình
        if isinstance(config, (str, Path)):
            self.config_path = Path(config)
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config_path = None
            self.config = config

        # 2. Khóa cứng Random Seed = 42 bảo đảm tính tái lập
        self.seed = int(self.config.get("seed", 42))
        seed_everything(self.seed)

        # 3. Thiết lập thiết bị tính toán (GPU / CPU)
        self.device = get_device()

        # 4. Trích xuất siêu tham số huấn luyện
        self.exp_id = str(self.config.get("exp_id", "EXP-01"))
        self.architecture = str(self.config.get("architecture", "SmallCNN"))
        self.exp_group = str(self.config.get("exp_group", "Baseline"))
        self.rq_tag = str(self.config.get("rq_tag", "RQ1"))
        self.image_size = int(self.config.get("image_size", 48))
        self.in_channels = int(self.config.get("in_channels", 1))
        self.augmentation = str(self.config.get("augmentation", "None"))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.lr = float(self.config.get("lr", 1e-3))
        self.epochs = int(self.config.get("epochs", 50))
        self.dropout = float(self.config.get("dropout", 0.5))
        self.activation = str(self.config.get("activation", "relu"))
        self.channels = self.config.get("channels")
        self.num_workers = int(self.config.get("num_workers", 0))
        self.num_classes = int(self.config.get("num_classes", 7))
        self.normalization = str(self.config.get("normalization", "fer2013"))
        self.patience = int(self.config.get("patience", 10))
        self.member = str(self.config.get("member", "Thành viên 2"))

        # Đường dẫn lưu checkpoint
        raw_checkpoint = self.config.get("checkpoint_path", "outputs/checkpoints/best_small_cnn.pth")
        self.checkpoint_path = Path(raw_checkpoint)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # 5. Khởi tạo DataLoaders cho 3 phân tập
        csv_path = self.config.get("csv_path", "data/raw/fer2013.csv")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            csv_path=csv_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=self.image_size,
            in_channels=self.in_channels,
            augmentation=self.augmentation,
            normalization=self.normalization,
            seed=self.seed,
            pin_memory=torch.cuda.is_available(),
        )

        # 6. Khởi tạo mô hình
        self.model = self._build_model().to(self.device)
        print(f"[*] Khởi tạo thành công mô hình {self.architecture}: {count_parameters(self.model):,} tham số.")

        # 7. Khởi tạo Hàm mất mát và Bộ tối ưu hóa (Optimizer)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_scheduler()

        # 8. Khởi tạo MLflowLogger
        self.logger = MLflowLogger()

    def _build_model(self) -> nn.Module:
        """Khởi tạo kiến trúc mô hình tương ứng với tham số architecture."""
        if self.architecture == "SmallCNN":
            return SmallCNN(
                num_classes=self.num_classes,
                in_channels=self.in_channels,
                dropout=self.dropout,
                activation=self.activation,
            )
        elif self.architecture == "MediumCNN":
            return MediumCNN(
                num_classes=self.num_classes,
                in_channels=self.in_channels,
                dropout=self.dropout,
                activation=self.activation,
                channels=self.channels if self.channels is not None else (64, 128, 256),
                dropout_rate=self.dropout,
            )
        elif self.architecture == "LargeCNN":
            return LargeCNN(
                num_classes=self.num_classes,
                in_channels=self.in_channels,
                dropout=self.dropout,
                activation=self.activation,
                channels=self.channels if self.channels is not None else (64, 128, 256, 512),
                dropout_rate=self.dropout,
            )
        raise ValueError(f"Kiến trúc mô hình '{self.architecture}' chưa được hỗ trợ trong Trainer.")

    def _build_optimizer(self) -> optim.Optimizer:
        """Khởi tạo bộ tối ưu hóa theo cấu hình."""
        opt_name = str(self.config.get("optimizer", "Adam")).lower()
        weight_decay = self.config.get("weight_decay")
        if opt_name == "adam":
            wd = float(weight_decay) if weight_decay is not None else 0.0
            return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=wd)
        elif opt_name == "adamw":
            wd = float(weight_decay) if weight_decay is not None else 1e-4
            return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=wd)
        elif opt_name in ("sgd", "sgd_nesterov"):
            wd = float(weight_decay) if weight_decay is not None else 0.0
            return optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, nesterov=True, weight_decay=wd)
        else:
            raise ValueError(f"Không hỗ trợ bộ tối ưu hóa: '{opt_name}'")

    def _build_scheduler(self) -> Optional[optim.lr_scheduler.LRScheduler]:
        """Khởi tạo bộ điều chỉnh learning rate theo cấu hình YAML."""
        scheduler_name = str(self.config.get("scheduler", "None")).strip().lower()
        scheduler_params = self.config.get("scheduler_params", {}) or {}
        if scheduler_name in ("none", "null", ""):
            return None
        if scheduler_name == "reducelronplateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=str(scheduler_params.get("mode", "min")),
                factor=float(scheduler_params.get("factor", 0.1)),
                patience=int(scheduler_params.get("patience", 3)),
                min_lr=float(scheduler_params.get("min_lr", 1e-6)),
            )
        if scheduler_name == "cosineannealinglr":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(scheduler_params.get("T_max", self.epochs)),
                eta_min=float(scheduler_params.get("eta_min", 0.0)),
            )
        raise ValueError(
            f"Scheduler '{self.config.get('scheduler')}' không được hỗ trợ; "
            "chọn ReduceLROnPlateau, CosineAnnealingLR hoặc None."
        )

    def train_one_epoch(self) -> Tuple[float, float]:
        """Thực hiện lan truyền tiến, tính gradient và cập nhật trọng số cho 1 epoch tập Train.

        Returns:
            Tuple[float, float]: (train_loss trung bình, train_accuracy %)
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += images.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = (correct_predictions / total_samples) * 100.0
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Đánh giá hiệu năng mô hình trên phân tập chỉ định (Validation hoặc Test).

        Args:
            dataloader: DataLoader của phân tập cần đánh giá.

        Returns:
            Tuple[float, float, float]: (loss trung bình, accuracy %, macro_f1 trong khoảng [0, 1])
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        total_samples = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                total_samples += images.size(0)

        eval_loss = running_loss / total_samples
        eval_acc = (np.sum(np.array(all_preds) == np.array(all_targets)) / total_samples) * 100.0
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

        return eval_loss, eval_acc, macro_f1

    def fit(self) -> Dict[str, Any]:
        """Thực thi toàn bộ vòng đời huấn luyện (Training loop) với EarlyStopping và MLflow.

        Returns:
            Dict[str, Any]: Từ điển chứa kết quả thực nghiệm tổng kết.
        """
        run_name = f"{self.exp_id}_{self.architecture}"
        print(f"\n{'='*70}")
        print(f"🚀 BẮT ĐẦU THỰC NGHIỆM: {run_name} (Nhóm: {self.exp_group} | Câu hỏi: {self.rq_tag})")
        print(f"{'='*70}")

        # 1. Khởi động MLflow Run
        self.logger.start_run(run_name=run_name, exp_group=self.exp_group, tags={"member": self.member})

        # 2. Ghi nhận tham số khởi tạo (Params)
        init_params = {
            "architecture": self.architecture,
            "exp_group": self.exp_group,
            "rq_tag": self.rq_tag,
            "seed": self.seed,
            "image_size": self.image_size,
            "augmentation": self.augmentation,
            "optimizer": str(self.config.get("optimizer", "Adam")),
            "lr": self.lr,
            "scheduler": str(self.config.get("scheduler", "None")),
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "activation": self.activation,
            "channels": str(self.channels) if self.channels is not None else "default",
            "member": self.member,
        }
        self.logger.log_params(init_params)

        best_val_acc = float("-inf")
        best_val_macro_f1 = 0.0
        best_epoch = 0
        patience_counter = 0

        # 3. Vòng lặp từng Epoch
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc, val_f1 = self.evaluate(self.val_loader)

            # Scheduler giảm LR theo validation loss hoặc theo lịch cosine.
            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_acc if self.lr_scheduler.mode == "max" else val_loss)
            elif self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}"
            )

            # Ghi nhận metrics của từng epoch vào MLflow
            self.logger.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "lr": self.optimizer.param_groups[0]["lr"],
            }, step=epoch)

            # Cơ chế EarlyStopping & Lưu trữ Checkpoint tốt nhất dựa trên Val Acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_macro_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0

                # Lưu trọng số mô hình tốt nhất
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  --> [*] Lưu best checkpoint tại Epoch {epoch} (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n[!] EarlyStopping kích hoạt sau {self.patience} epoch không cải thiện độ chính xác.")
                    break

        print(f"\n{'='*70}")
        print(f"🏆 HUẤN LUYỆN HOÀN TẤT! Đang đánh giá Checkpoint tốt nhất trên tập Private Test...")

        # 4. Nạp checkpoint tốt nhất để đánh giá trên tập Private Test độc lập
        if self.checkpoint_path.exists():
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

        test_loss, private_test_acc, private_macro_f1 = self.evaluate(self.test_loader)
        print(f"[*] KẾT QUẢ ĐÁNH GIÁ CUỐI KỲ (Private Test):")
        print(f"    - Private Test Acc : {private_test_acc:.2f}%")
        print(f"    - Private Macro-F1 : {private_macro_f1:.4f}")
        print(f"    - Best Val Acc     : {best_val_acc:.2f}% (tại Epoch {best_epoch})")
        print(f"{'='*70}")

        # 5. Ghi nhận các chỉ số tổng kết vào MLflow theo đúng định dạng 24 cột
        final_summary_metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "best_val_acc": best_val_acc,
            "private_test_acc": private_test_acc,
            "macro_f1": private_macro_f1,
            "best_epoch": best_epoch,
        }
        self.logger.log_metrics(final_summary_metrics)

        # Lưu checkpoint làm artifact trong MLflow
        if self.checkpoint_path.exists():
            self.logger.log_artifact(str(self.checkpoint_path))

        # Đóng phiên chạy MLflow và xuất file CSV tổng kết
        self.logger.end_run()
        summary_csv = self.logger.export_summary_csv()
        print(f"[*] Đã xuất bảng tổng kết 24 cột tại: {summary_csv}")

        return {
            "exp_id": self.exp_id,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "private_test_acc": private_test_acc,
            "macro_f1": private_macro_f1,
        }


def parse_args() -> argparse.Namespace:
    """Xử lý các đối số dòng lệnh khi chạy script."""
    parser = argparse.ArgumentParser(description="Chương trình huấn luyện mô hình FER2013.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/baseline/exp01_small_cnn.yaml",
        help="Đường dẫn đến tệp tin cấu hình YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(config=args.config)
    trainer.fit()
