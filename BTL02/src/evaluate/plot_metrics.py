"""Đọc lịch sử huấn luyện và trực quan hóa metrics FER2013 bằng PyTorch."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix


class MetricsVisualizer:
    """Tính báo cáo phân loại, confusion matrix và biểu đồ lịch sử huấn luyện.

    Tập dữ liệu được truyền vào qua DataLoader evaluation đã dựng sẵn; module
    không tạo augmentation và không sửa đổi pipeline tiền xử lý của dự án.
    """

    CLASS_NAMES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")

    def __init__(
        self,
        output_dir: Path = Path("outputs/figures/t0501"),
        class_names: Sequence[str] = CLASS_NAMES,
        device: Optional[torch.device] = None,
        dpi: int = 300,
    ) -> None:
        """Khởi tạo trực quan hóa.

        Args:
            output_dir: Thư mục đầu ra; mặc định nằm trong outputs bị gitignore.
            class_names: Tên lớp theo thứ tự chỉ số nhãn.
            device: Thiết bị suy luận; mặc định ưu tiên CUDA nếu khả dụng.
            dpi: Độ phân giải ảnh xuất.
        """
        if not class_names or dpi <= 0:
            raise ValueError("Cần ít nhất một lớp và dpi phải dương.")
        self.output_dir = Path(output_dir)
        self.class_names = tuple(class_names)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dpi = dpi

    @staticmethod
    def synthetic_history(epochs: int = 10) -> pd.DataFrame:
        """Tạo lịch sử xác định để kiểm tra offline, không đại diện số liệu thật."""
        if epochs <= 0:
            raise ValueError("epochs phải dương.")
        return pd.DataFrame(
            {
                "epoch": np.arange(1, epochs + 1),
                "train_loss": np.linspace(1.8, 0.5, epochs),
                "val_loss": np.linspace(1.9, 0.8, epochs),
                "train_acc": np.linspace(25.0, 82.0, epochs),
                "val_acc": np.linspace(24.0, 65.0, epochs),
                "lr": np.linspace(1e-3, 1e-4, epochs),
            }
        )

    @classmethod
    def load_history(cls, csv_path: Path) -> pd.DataFrame:
        """Đọc CSV epoch-level và chuẩn hóa các bí danh cột thông dụng.

        CSV summary MLflow 24 cột không chứa epoch history nên bị từ chối thay
        vì tạo biểu đồ gây hiểu nhầm từ một giá trị cuối run.
        """
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy CSV lịch sử: {path}")
        frame = pd.read_csv(path)
        aliases = {
            "metrics.train_loss": "train_loss",
            "metrics.train_acc": "train_acc",
            "metrics.val_loss": "val_loss",
            "metrics.val_acc": "val_acc",
            "learning_rate": "lr",
            "metrics.lr": "lr",
        }
        frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
        required = {"epoch", "train_loss", "train_acc", "lr"}
        if not required.issubset(frame.columns):
            missing = ", ".join(sorted(required - set(frame.columns)))
            raise ValueError(f"CSV không có lịch sử theo epoch; thiếu cột: {missing}")
        if frame.empty or not np.isfinite(frame[list(required)].to_numpy(dtype=float)).all():
            raise ValueError("Lịch sử rỗng hoặc chứa giá trị không hữu hạn.")
        return frame.sort_values("epoch").reset_index(drop=True)

    def plot_training_curves(self, history: pd.DataFrame, name: str = "training_curves") -> Path:
        """Vẽ loss, accuracy và learning rate theo epoch rồi lưu ảnh độ phân giải cao."""
        required = {"epoch", "train_loss", "train_acc", "lr"}
        if not required.issubset(history.columns) or history.empty:
            raise ValueError("Lịch sử phải có epoch, train_loss, train_acc, lr và không rỗng.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for key, label, axis in (("loss", "Loss", axes[0]), ("acc", "Accuracy (%)", axes[1])):
            for prefix, title in (("train", "Train"), ("val", "Validation")):
                column = f"{prefix}_{key}"
                if column in history:
                    axis.plot(history["epoch"], history[column], marker="o", label=title)
            axis.set(title=label, xlabel="Epoch")
            axis.grid(alpha=0.25)
            axis.legend()
        axes[2].plot(history["epoch"], history["lr"], color="tab:green", marker="o")
        axes[2].set(title="Learning Rate", xlabel="Epoch", yscale="log")
        axes[2].grid(alpha=0.25)
        destination = self.output_dir / f"{name}.png"
        fig.savefig(destination, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return destination

    def infer(self, model: nn.Module, dataloader: Iterable) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chạy suy luận không gradient và trả nhãn thật/dự đoán trên CPU."""
        model = model.to(self.device)
        model.eval()
        targets: List[torch.Tensor] = []
        predictions: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch[:2]
                logits = model(images.to(self.device))
                targets.append(labels.detach().cpu().long())
                predictions.append(logits.argmax(dim=1).detach().cpu().long())
        if not targets:
            raise ValueError("DataLoader đánh giá rỗng.")
        return torch.cat(targets), torch.cat(predictions)

    def build_report(self, targets: torch.Tensor, predictions: torch.Tensor) -> Dict[str, object]:
        """Tạo classification report và confusion matrix đủ cả bảy lớp."""
        y_true = torch.as_tensor(targets).detach().cpu().numpy().astype(int)
        y_pred = torch.as_tensor(predictions).detach().cpu().numpy().astype(int)
        if y_true.ndim != 1 or y_pred.shape != y_true.shape or y_true.size == 0:
            raise ValueError("Nhãn thật/dự đoán phải là hai vector không rỗng cùng shape.")
        labels = list(range(len(self.class_names)))
        if np.any((y_true < 0) | (y_true >= len(labels))) or np.any((y_pred < 0) | (y_pred >= len(labels))):
            raise ValueError("Phát hiện nhãn nằm ngoài miền lớp đã cấu hình.")
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(
            y_true, y_pred, labels=labels, target_names=self.class_names,
            output_dict=True, zero_division=0,
        )
        return {"confusion_matrix": matrix, "classification_report": report}

    def plot_confusion_matrix(self, matrix: np.ndarray, split: str, normalized: bool = False) -> Path:
        """Vẽ confusion matrix số lượng hoặc chuẩn hóa theo hàng phần trăm."""
        matrix = np.asarray(matrix)
        if matrix.shape != (len(self.class_names), len(self.class_names)):
            raise ValueError("Kích thước confusion matrix không khớp số lớp.")
        values = matrix.astype(float)
        if normalized:
            totals = values.sum(axis=1, keepdims=True)
            values = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0) * 100.0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        image = axis.imshow(values, cmap="Blues")
        fig.colorbar(image, ax=axis, label="Tỷ lệ (%)" if normalized else "Số ảnh")
        axis.set(xticks=range(len(self.class_names)), yticks=range(len(self.class_names)),
                 xticklabels=self.class_names, yticklabels=self.class_names,
                 xlabel="Nhãn dự đoán", ylabel="Nhãn thật",
                 title=f"Confusion Matrix — {split}" + (" (chuẩn hóa theo hàng)" if normalized else ""))
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
        threshold = values.max() / 2 if values.size else 0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                text = f"{values[row, col]:.1f}%" if normalized else f"{int(matrix[row, col])}"
                axis.text(col, row, text, ha="center", va="center",
                          color="white" if values[row, col] > threshold else "black", fontsize=8)
        destination = self.output_dir / f"confusion_{split}_{'normalized' if normalized else 'counts'}.png"
        fig.savefig(destination, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return destination

    def save_report(self, report: Dict[str, object], split: str) -> Dict[str, Path]:
        """Xuất report định lượng sang JSON, CSV và Markdown."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        matrix = np.asarray(report["confusion_matrix"])
        metrics = report["classification_report"]
        paths = {
            "json": self.output_dir / f"classification_{split}.json",
            "csv": self.output_dir / f"classification_{split}.csv",
            "markdown": self.output_dir / f"classification_{split}.md",
        }
        payload = {"classes": list(self.class_names), "confusion_matrix": matrix.tolist(),
                   "classification_report": metrics}
        paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
        pd.DataFrame(metrics).T.to_csv(paths["csv"], index_label="class", encoding="utf-8")
        rows = ["| Lớp | Precision | Recall | F1-score | Support |", "|:---|---:|---:|---:|---:|"]
        for label, values in metrics.items():
            if not isinstance(values, dict):
                values = {"precision": values, "recall": values, "f1-score": values, "support": len(matrix)}
            rows.append(f"| {label} | {values.get('precision', 0):.4f} | {values.get('recall', 0):.4f} | "
                        f"{values.get('f1-score', 0):.4f} | {values.get('support', 0)} |")
        paths["markdown"].write_text("\n".join([f"# Classification Report — {split}", "", *rows, ""]), encoding="utf-8")
        self.plot_confusion_matrix(matrix, split, normalized=False)
        self.plot_confusion_matrix(matrix, split, normalized=True)
        return paths

    def evaluate_split(self, model: nn.Module, dataloader: Iterable, split: str) -> Dict[str, object]:
        """Suy luận một split, rồi lưu report và hai confusion matrix."""
        targets, predictions = self.infer(model, dataloader)
        report = self.build_report(targets, predictions)
        report["artifacts"] = self.save_report(report, split)
        return report

    @classmethod
    def cli(cls, arguments: Optional[Sequence[str]] = None) -> int:
        """Vẽ lịch sử CSV; hỗ trợ mock rõ ràng cho kiểm thử offline."""
        parser = argparse.ArgumentParser(description="Trực quan hóa metrics huấn luyện FER2013.")
        parser.add_argument("--history", type=Path)
        parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/t0501"))
        parser.add_argument("--mock", action="store_true", help="Dùng dữ liệu tổng hợp, không phải kết quả thực nghiệm.")
        args = parser.parse_args(arguments)
        if bool(args.history) == args.mock:
            parser.error("Chọn đúng một trong --history hoặc --mock.")
        visualizer = cls(args.output_dir)
        history = visualizer.synthetic_history() if args.mock else visualizer.load_history(args.history)
        print(visualizer.plot_training_curves(history))
        return 0


if __name__ == "__main__":
    raise SystemExit(MetricsVisualizer.cli())