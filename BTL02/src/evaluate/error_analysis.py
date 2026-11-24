"""Phân tích các dự đoán đúng/sai và xác suất top-k trên FER2013."""

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class ErrorAnalyzer:
    """Suy luận evaluation split, xếp hạng nhầm lẫn và xuất bảng/lưới mẫu."""

    CLASS_NAMES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")

    def __init__(self, model: nn.Module, output_dir: Path = Path("outputs/figures/t0503"),
                 class_names: Sequence[str] = CLASS_NAMES,
                 device: Optional[torch.device] = None, top_k: int = 3) -> None:
        """Khởi tạo analyzer; top-k mặc định theo tiêu chí T0503."""
        if not class_names or top_k <= 0:
            raise ValueError("Cần tên lớp và top_k dương.")
        self.model = model
        self.output_dir = Path(output_dir)
        self.class_names = tuple(class_names)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = min(top_k, len(self.class_names))

    def collect(self, dataloader: Iterable) -> Dict[str, torch.Tensor]:
        """Chạy model.eval()/torch.no_grad và thu thập labels, logits, softmax, top-k."""
        self.model.to(self.device).eval()
        labels_all: List[torch.Tensor] = []
        logits_all: List[torch.Tensor] = []
        images_all: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch[:2]
                logits = self.model(images.to(self.device))
                if logits.ndim != 2 or logits.shape[1] != len(self.class_names):
                    raise ValueError("Model logits phải có shape [B, số_lớp].")
                images_all.append(images.detach().cpu())
                labels_all.append(labels.detach().cpu().long())
                logits_all.append(logits.detach().cpu())
        if not labels_all:
            raise ValueError("DataLoader đánh giá rỗng.")
        logits = torch.cat(logits_all)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = probabilities.topk(self.top_k, dim=1)
        labels = torch.cat(labels_all)
        if torch.any((labels < 0) | (labels >= len(self.class_names))):
            raise ValueError("Nhãn thật nằm ngoài miền lớp.")
        predictions = probabilities.argmax(dim=1)
        return {"images": torch.cat(images_all), "targets": labels, "logits": logits,
                "probabilities": probabilities, "predictions": predictions,
                "top_probabilities": top_probs, "top_indices": top_indices,
                "correct": predictions.eq(labels)}

    def confusion_pairs(self, results: Dict[str, torch.Tensor]) -> List[Dict[str, object]]:
        """Xếp hạng cặp nhầm lẫn có hướng true→pred, loại mẫu dự đoán đúng."""
        targets = results["targets"].tolist()
        predictions = results["predictions"].tolist()
        counts = Counter((true, pred) for true, pred in zip(targets, predictions) if true != pred)
        pairs = [{"true": self.class_names[true], "predicted": self.class_names[pred], "count": count}
                 for (true, pred), count in counts.items()]
        return sorted(pairs, key=lambda item: (-item["count"], item["true"], item["predicted"]))

    def summarize(self, results: Dict[str, torch.Tensor], confidence_threshold: float = 0.7) -> Dict[str, object]:
        """Phân nhóm đúng tự tin cao/thấp và sai; ngưỡng được ghi nhận rõ ràng."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold phải thuộc [0, 1].")
        confidence = results["probabilities"].max(dim=1).values
        correct = results["correct"]
        return {
            "true_high_confidence": torch.where(correct & (confidence >= confidence_threshold))[0].tolist(),
            "true_low_confidence": torch.where(correct & (confidence < confidence_threshold))[0].tolist(),
            "fail_cases": torch.where(~correct)[0].tolist(),
            "confusion_pairs": self.confusion_pairs(results),
            "confidence_threshold": confidence_threshold,
        }

    def _sample_row(self, results: Dict[str, torch.Tensor], index: int, status: str) -> Dict[str, object]:
        """Tạo một dòng bảng với nhãn thật/dự đoán, confidence và top-k."""
        top_indices = results["top_indices"][index].tolist()
        top_probabilities = results["top_probabilities"][index].tolist()
        return {
            "index": index,
            "status": status,
            "true": self.class_names[int(results["targets"][index])],
            "predicted": self.class_names[int(results["predictions"][index])],
            "confidence": float(results["probabilities"][index].max()),
            "top_k": " | ".join(f"{self.class_names[label]}: {probability:.1%}"
                                 for label, probability in zip(top_indices, top_probabilities)),
        }

    def save_tables(self, results: Dict[str, torch.Tensor], summary: Dict[str, object],
                    split: str) -> Dict[str, Path]:
        """Lưu mọi case, cặp nhầm lẫn và khung phân tích nguyên nhân dưới dạng CSV/Markdown."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        high = set(summary["true_high_confidence"])
        low = set(summary["true_low_confidence"])
        for index in range(len(results["targets"])):
            if index in high:
                status = "True-high"
            elif index in low:
                status = "True-low"
            else:
                status = "Fail"
            rows.append(self._sample_row(results, index, status))
        csv_path = self.output_dir / f"cases_{split}.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        pair_path = self.output_dir / f"confusion_pairs_{split}.csv"
        pairs = summary["confusion_pairs"]
        with pair_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["true", "predicted", "count"])
            writer.writeheader()
            writer.writerows(pairs)
        analysis_path = self.output_dir / f"analysis_{split}.md"
        pair_lines = ["| Nhãn thật | Dự đoán nhầm | Số ảnh |", "|:---|:---|---:|"]
        pair_lines.extend(f"| {pair['true']} | {pair['predicted']} | {pair['count']} |" for pair in pairs)
        causes = [
            "### Khung giả thuyết cần kiểm chứng trên ảnh gốc", "- Che khuất mắt, lông mày hoặc miệng.",
            "- Góc nghiêng, ánh sáng yếu/gắt hoặc chất lượng ảnh thấp.",
            "- Biểu cảm mơ hồ, nhãn nhiễu/chủ quan; cần người gán nhãn rà soát.",
            "- Lớp Disgust ít mẫu; kiểm tra support/per-class recall trước khi kết luận.",
            "Các ý trên là giả thuyết phân tích, không tự động suy ra chỉ từ logits.",
        ]
        content = [f"# Phân tích lỗi — {split}", "",
                   f"- True, tự tin cao: {len(high)}; True, tự tin thấp: {len(low)}; Fail: {len(summary['fail_cases'])}.",
                   f"- Ngưỡng tự tin: {summary['confidence_threshold']:.0%}.", "",
                   "## Cặp nhầm lẫn nhiều nhất", *pair_lines, "", *causes, ""]
        analysis_path.write_text("\n".join(content), encoding="utf-8")
        return {"cases_csv": csv_path, "pairs_csv": pair_path, "analysis_markdown": analysis_path}

    def plot_cases(self, results: Dict[str, torch.Tensor], indices: Sequence[int], split: str,
                   status: str, max_cases: int = 24) -> Optional[Path]:
        """Vẽ grid với True/Pred/top-k; ảnh đầu vào hiển thị theo tensor evaluation có sẵn."""
        selected = list(indices)[:max_cases]
        if not selected:
            return None
        columns = min(4, len(selected))
        rows_count = (len(selected) + columns - 1) // columns
        fig, axes = plt.subplots(rows_count, columns, figsize=(columns * 3, rows_count * 3), squeeze=False)
        for axis, index in zip(axes.flat, selected):
            image = results["images"][index].permute(1, 2, 0).numpy()
            low_value, high_value = float(image.min()), float(image.max())
            image = (image - low_value) / (high_value - low_value) if high_value > low_value else np.zeros_like(image)
            if image.shape[-1] == 1:
                image = image[..., 0]
            row = self._sample_row(results, index, status)
            axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
            axis.set_title(f"True: {row['true']} | Pred: {row['predicted']}\n{row['top_k']}", fontsize=8)
            axis.axis("off")
        for axis in axes.flat[len(selected):]:
            axis.axis("off")
        fig.suptitle(f"{split} — {status} cases (top-{self.top_k})")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"grid_{split}_{status.lower()}.png"
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return destination

    def analyze(self, dataloader: Iterable, split: str = "PrivateTest",
                confidence_threshold: float = 0.7) -> Dict[str, object]:
        """Chạy trọn luồng suy luận, thống kê, bảng lỗi và lưới đúng/sai."""
        results = self.collect(dataloader)
        summary = self.summarize(results, confidence_threshold)
        artifacts = self.save_tables(results, summary, split)
        artifacts["true_grid"] = self.plot_cases(results, summary["true_high_confidence"], split, "True")
        artifacts["fail_grid"] = self.plot_cases(results, summary["fail_cases"], split, "Fail")
        return {"results": results, "summary": summary, "artifacts": artifacts}