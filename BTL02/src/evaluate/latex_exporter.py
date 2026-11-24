"""Xuất bảng và đồ thị TeX từ bằng chứng thực nghiệm BTL02 đã có."""

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class ThesisArtifactExporter:
    """Đọc CSV summary/benchmark và sinh bảng longtable, không bịa số liệu."""

    SUMMARY_COLUMNS = (
        "Start Time", "Duration", "Run ID", "Name", "Source Name", "Status",
        "params.architecture", "params.exp_group", "params.rq_tag", "params.seed",
        "params.image_size", "params.augmentation", "params.optimizer", "params.lr",
        "params.scheduler", "params.batch_size", "params.dropout", "metrics.train_loss",
        "metrics.train_acc", "metrics.best_val_acc", "metrics.private_test_acc",
        "metrics.macro_f1", "metrics.best_epoch", "tags.member",
    )
    SUMMARY_FIELDS = (
        ("Run ID", "Run ID"), ("Name", "Tên run"), ("params.architecture", "Kiến trúc"),
        ("params.rq_tag", "RQ"), ("params.seed", "Seed"), ("params.image_size", "Kích thước ảnh"),
        ("metrics.best_epoch", "Best epoch"), ("metrics.train_acc", "Train Acc"),
        ("metrics.best_val_acc", "Public/Val Acc"), ("metrics.private_test_acc", "Private Acc"),
        ("metrics.macro_f1", "Macro-F1"),
    )
    BENCHMARK_COLUMNS = (
        "architecture", "image_size", "device", "warmup_runs", "measure_runs",
        "total_params", "trainable_params", "frozen_params", "model_tensor_mb",
        "checkpoint_size_mb", "macs", "flops", "latency_ms_bs1",
        "latency_std_ms_bs1", "throughput_batch_size", "throughput_images_per_sec",
    )
    LATEX_SPECIAL = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }

    def __init__(self, logs_dir: Path, generated_dir: Path) -> None:
        """Khởi tạo thư mục nguồn CSV và đích sinh mã TeX."""
        self.logs_dir = Path(logs_dir)
        self.generated_dir = Path(generated_dir)

    @classmethod
    def escape_latex(cls, value: object) -> str:
        """Thoát từng ký tự đặc biệt mà không xử lý đệ quy ký tự vừa sinh."""
        return "".join(cls.LATEX_SPECIAL.get(character, character) for character in str(value))

    @staticmethod
    def _read_csv(path: Path, expected: Sequence[str]) -> List[Dict[str, str]]:
        """Đọc CSV chỉ khi header đúng hợp đồng yêu cầu."""
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(expected):
                raise ValueError(f"Schema CSV không đúng tại {path.name}.")
            return list(reader)

    def _summary_rows(self) -> List[Dict[str, str]]:
        """Đọc và hợp nhất các CSV có đúng schema MLflow summary 24 cột."""
        if not self.logs_dir.is_dir():
            return []
        merged: Dict[str, Dict[str, str]] = {}
        for path in sorted(self.logs_dir.glob("*.csv")):
            with path.open("r", newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != self.SUMMARY_COLUMNS:
                    continue
                for row in reader:
                    run_id = (row.get("Run ID") or "").strip()
                    if run_id:
                        merged[run_id] = row
        return list(merged.values())

    def _write_table(self, destination: Path, headers: Sequence[str],
                     rows: Sequence[Sequence[str]], warning: str) -> Path:
        """Ghi bảng longtable booktabs với header lặp và placeholder an toàn."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        width = 0.90 / len(headers)
        column_spec = "".join(f"p{{{width:.4f}\\textwidth}}" for _ in headers)
        row_end = " " + chr(92) * 2
        lines = [
            "% Tự động sinh bởi ThesisArtifactExporter; không chỉnh sửa thủ công.",
            r"\begin{longtable}{" + column_spec + "}", r"\scriptsize", r"\toprule",
            " & ".join(self.escape_latex(item) for item in headers) + row_end,
            r"\midrule", r"\endfirsthead", r"\toprule",
            " & ".join(self.escape_latex(item) for item in headers) + row_end,
            r"\midrule", r"\endhead", r"\midrule", r"\multicolumn{" + str(len(headers))
            + r"}{r}{(Tiếp theo)}" + row_end, r"\endfoot", r"\bottomrule", r"\endlastfoot",
        ]
        if rows:
            lines.extend(" & ".join(row) + row_end for row in rows)
        else:
            lines.append(r"\multicolumn{" + str(len(headers)) + r"}{p{0.9\textwidth}}{\textit{"
                         + self.escape_latex(warning) + "}}" + row_end)
        lines.extend([r"\end{longtable}", ""])
        destination.write_text("\n".join(lines), encoding="utf-8")
        return destination

    @staticmethod
    def _numeric(value: str) -> Optional[float]:
        """Đọc số hữu hạn; giá trị thiếu/không hợp lệ trả None."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number and abs(number) != float("inf") else None

    def export_summary_tables(self) -> Dict[str, Path]:
        """Sinh bảng toàn run và bảng best theo Private Acc/Macro-F1."""
        rows = self._summary_rows()
        outputs: Dict[str, Path] = {}
        for name, selected in (("all_runs", rows), ("rq1_depth", [r for r in rows if "RQ1" in r.get("params.rq_tag", "")]),
                               ("rq2_transfer", [r for r in rows if "RQ2" in r.get("params.rq_tag", "")]),
                               ("rq3_ablation", [r for r in rows if "RQ3" in r.get("params.rq_tag", "") or "Ablation" in r.get("params.exp_group", "")])):
            rendered = []
            for row in selected:
                cells = [self.escape_latex(row.get(key, "") or "--") for key, _ in self.SUMMARY_FIELDS]
                for metric_index, metric_key in ((9, "metrics.private_test_acc"), (10, "metrics.macro_f1")):
                    candidates = [self._numeric(item.get(metric_key, "")) for item in selected]
                    valid = [value for value in candidates if value is not None]
                    current = self._numeric(row.get(metric_key, ""))
                    if current is not None and valid and current == max(valid):
                        cells[metric_index] = r"\textbf{" + cells[metric_index] + "}"
                rendered.append(cells)
            outputs[name] = self._write_table(
                self.generated_dir / "tables" / f"{name}.tex",
                [title for _, title in self.SUMMARY_FIELDS], rendered,
                "Chưa có CSV 24 cột hợp lệ cho nhóm này; chưa có số liệu để báo cáo.",
            )
        return outputs

    def export_compute_table(self) -> Path:
        """Xuất bảng chi phí chỉ từ CSV benchmark có schema riêng đã xác nhận."""
        rows: List[Dict[str, str]] = []
        if self.logs_dir.is_dir():
            for path in sorted(self.logs_dir.glob("*benchmark*.csv")):
                rows.extend(self._read_csv(path, self.BENCHMARK_COLUMNS))
        fields = ("architecture", "image_size", "total_params", "flops", "latency_ms_bs1", "throughput_images_per_sec")
        labels = ("Kiến trúc", "Ảnh", "Params", "FLOPs", "Latency (ms)", "Ảnh/giây")
        rendered = [[self.escape_latex(row.get(key, "") or "--") for key in fields] for row in rows]
        return self._write_table(self.generated_dir / "tables" / "compute_cost.tex", labels, rendered,
                                 "Chưa có CSV benchmark SOTA hợp lệ; Params/FLOPs chưa được xác nhận.")

    def export_vector_plot(self) -> Path:
        """Sinh PGFPlots so sánh các run thật; không tạo series giả khi thiếu metrics."""
        rows = self._summary_rows()
        valid = [(row, self._numeric(row.get("metrics.private_test_acc", "")),
                  self._numeric(row.get("metrics.macro_f1", ""))) for row in rows]
        valid = [item for item in valid if item[1] is not None and item[2] is not None]
        destination = self.generated_dir / "plots" / "private_macro_comparison.tex"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not valid:
            destination.write_text(
                r"\todofigure{Chưa có metrics hợp lệ để sinh biểu đồ vector so sánh.}" + "\n",
                encoding="utf-8",
            )
            return destination
        labels = [self.escape_latex(item[0].get("Name", "Run")) for item in valid]
        indices = list(range(1, len(valid) + 1))
        content = (
            "% Tự động sinh từ metrics có trong CSV summary; không chỉnh sửa thủ công.\n"
            r"\begin{tikzpicture}" + "\n" + r"\begin{axis}[ybar,bar width=7pt,width=\textwidth,height=7cm," + "\n"
            + r"  ylabel={Metric}, xtick={" + ",".join(map(str, indices)) + "}, xticklabels={"
            + ",".join("{" + label + "}" for label in labels) + "}, x tick label style={rotate=60,anchor=east}," + "\n"
            + r"  legend style={at={(0.5,1.02)},anchor=south,legend columns=2}, ymin=0]" + "\n"
            + r"\addplot coordinates {" + " ".join(f"({index},{item[1]:.6g})" for index, item in zip(indices, valid)) + "};\n"
            + r"\addplot coordinates {" + " ".join(f"({index},{item[2] * 100.0:.6g})" for index, item in zip(indices, valid)) + "};\n"
            + r"\legend{Private Test Acc,Macro-F1}" + "\n" + r"\end{axis}" + "\n" + r"\end{tikzpicture}" + "\n"
        )
        destination.write_text(content, encoding="utf-8")
        return destination

    def export_all(self) -> Dict[str, object]:
        """Sinh mọi bảng chuẩn và vector plot nếu dữ liệu nguồn hợp lệ."""
        outputs: Dict[str, object] = {"tables": self.export_summary_tables()}
        outputs["tables"]["compute_cost"] = self.export_compute_table()
        outputs["plot"] = self.export_vector_plot()
        return outputs