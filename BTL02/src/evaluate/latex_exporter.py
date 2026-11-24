"""Xuất bảng và đồ thị TeX từ bằng chứng thực nghiệm BTL02 đã có."""

import csv
import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
        ("Run ID", "Run ID"), ("params.architecture", "Kiến trúc"),
        ("params.rq_tag", "RQ/nhóm"), ("params.image_size", "Ảnh"),
        ("metrics.best_epoch", "Best epoch"), ("metrics.best_val_acc", "Public/Val Acc (%)"),
        ("metrics.private_test_acc", "Private Acc (%)"), ("metrics.macro_f1", "Macro-F1"),
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

    def __init__(self, logs_dir: Path, generated_dir: Path, figure_dir: Optional[Path] = None,
                 evidence_dir: Optional[Path] = None) -> None:
        """Khởi tạo thư mục nguồn CSV và đích sinh mã TeX."""
        self.logs_dir = Path(logs_dir)
        self.generated_dir = Path(generated_dir)
        self.figure_dir = Path(figure_dir) if figure_dir else self.logs_dir.parent / "figures"
        self.evidence_dir = Path(evidence_dir) if evidence_dir else self.logs_dir.parent / "docs" / "experiences"

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
                        # Giữ cả tên run và thời điểm nguồn để phân biệt EXP-01 run với
                        # run ReLU trong ablation có cùng architecture/checkpoint metrics.
                        identity = f"{run_id}|{row.get('Name', '')}|{row.get('Start Time', '')}"
                        merged[identity] = row
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
        groups = (
            ("all_runs", rows),
            ("rq1_depth", [row for row in rows if row.get("params.rq_tag", "") == "RQ1"]),
            ("rq2_transfer", [row for row in rows if row.get("params.rq_tag", "") == "RQ2_SOTA_Pretrained_128"]),
            ("rq3_ablation", [row for row in rows if str(row.get("params.rq_tag", "")).startswith("RQ3_")
                               or str(row.get("params.rq_tag", "")).startswith("Ablation_")
                               or row.get("params.rq_tag") == "RQ1_Ablation_Activation"]),
        )
        for name, selected in groups:
            rendered = []
            for row in selected:
                cells = [self.escape_latex(row.get(key, "") or "--") for key, _ in self.SUMMARY_FIELDS]
                for metric_index, metric_key in ((6, "metrics.private_test_acc"), (7, "metrics.macro_f1")):
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
        if not rows and self.evidence_dir.is_dir():
            rows = self._benchmark_rows_from_experience()
        fields = ("architecture", "image_size", "total_params", "flops", "latency_ms_bs1", "throughput_images_per_sec")
        labels = ("Kiến trúc", "Ảnh", "Params", "FLOPs", "Latency (ms)", "Ảnh/giây")
        rendered = [[self.escape_latex(row.get(key, "") or "--") for key in fields] for row in rows]
        return self._write_table(self.generated_dir / "tables" / "compute_cost.tex", labels, rendered,
                                 "Chưa có CSV benchmark SOTA hợp lệ; Params/FLOPs chưa được xác nhận.")

    def _benchmark_rows_from_experience(self) -> List[Dict[str, str]]:
        """Đọc đúng bảng benchmark đã nghiệm thu khi CSV benchmark không còn local."""
        evidence_files = sorted(self.evidence_dir.glob("*COST*.md"))
        rows: List[Dict[str, str]] = []
        for path in sorted(self.evidence_dir.glob("*COST*.md")):
            lines = path.read_text(encoding="utf-8").splitlines()
            parameter_counts: Dict[str, str] = {}
            for line in lines:
                cells = [cell.strip().replace(",", "") for cell in line.strip().strip("|").split("|")]
                # Bảng accuracy T0304 ghi số Params theo topology ở cột cuối.
                if len(cells) == 8 and cells[0] in {"VGG16", "ResNet50", "EfficientNet-B0"}:
                    if cells[-1].isdigit():
                        parameter_counts[cells[0]] = cells[-1]
            for line in lines:
                cells = [cell.strip().replace(",", "") for cell in line.strip().strip("|").split("|")]
                # Bảng T0305 có 9 cột, gồm cột Status cuối; hàng phải có marker CPU.
                if (len(cells) != 9 or cells[0] not in {"VGG16", "ResNet50", "EfficientNet-B0"}
                        or ("CPU" not in cells[8] and cells[8] != "OK")):
                    continue
                try:
                    int(cells[1]); int(cells[2]); int(cells[3])
                    float(cells[4]); float(cells[5]); float(cells[6]); float(cells[7])
                except ValueError:
                    continue
                rows.append({
                    "architecture": cells[0], "image_size": cells[1],
                    "total_params": parameter_counts.get(cells[0], "--"),
                    "macs": cells[2], "flops": cells[3],
                    "model_tensor_mb": cells[4], "checkpoint_size_mb": cells[5],
                    "latency_ms_bs1": cells[6], "throughput_images_per_sec": cells[7],
                })
        return rows

    def export_vector_plot(self) -> Path:
        """Sinh PGFPlots cho ba cấu hình RQ1 từ CSV; không nội suy theo epoch."""
        rows = [row for row in self._summary_rows() if row.get("params.rq_tag") == "RQ1"]
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
        public_points = " ".join(
            f"({index},{float(item[0].get('metrics.best_val_acc', 0.0)):.6g})"
            for index, item in zip(indices, valid)
        )
        private_points = " ".join(f"({index},{item[1]:.6g})" for index, item in zip(indices, valid))
        macro_points = " ".join(f"({index},{item[2] * 100.0:.6g})" for index, item in zip(indices, valid))
        labels_tex = ",".join("{" + label + "}" for label in labels)
        lines = [
            "% Tự động sinh từ metrics CSV summary; không chỉnh sửa thủ công.",
            r"\begin{tikzpicture}",
            r"\begin{axis}[ybar,bar width=5pt,width=\textwidth,height=7cm,",
            "ylabel={Accuracy / F1 (\\%)},",
            "xtick={" + ",".join(map(str, indices)) + "},",
            "xticklabels={" + labels_tex + "},",
            r"x tick label style={rotate=55,anchor=east},",
            r"legend style={at={(0.5,1.02)},anchor=south,legend columns=3},",
            "ymin=0, ymax=100]",
            r"\addplot coordinates {" + public_points + "};",
            r"\addplot coordinates {" + private_points + "};",
            r"\addplot coordinates {" + macro_points + "};",
            r"\legend{PublicTest validation Acc,PrivateTest Acc,Macro-F1}",
            r"\end{axis}",
            r"\end{tikzpicture}",
            "",
        ]
        content = "\n".join(lines)
        destination.write_text(content, encoding="utf-8")
        return destination

    def export_comparison_figures(self, dpi: int = 300) -> Dict[str, Path]:
        """Vẽ các so sánh Public/Private/Macro-F1 từ run summary đã khử trùng.

        Các hình là đối chiếu metrics cuối/best từ CSV, không phải training curves.
        RQ2 giới hạn ở pretrained 128; RQ3 tách các nhóm recipe để không trộn
        thí nghiệm khác kiến trúc hoặc mục tiêu vào một phép so sánh duy nhất.
        """
        if dpi <= 0:
            raise ValueError("dpi phải là số nguyên dương.")
        rows = self._summary_rows()
        figure_dir = self.figure_dir / "t0602"
        groups = {
            "t0602_rq1_metrics": [row for row in rows if row.get("params.rq_tag") == "RQ1"],
            "t0602_rq2_metrics": [row for row in rows if row.get("params.rq_tag") == "RQ2_SOTA_Pretrained_128"],
            "t0602_rq3_metrics": [row for row in rows
                                  if str(row.get("params.rq_tag", "")).startswith("RQ3_")
                                  or str(row.get("params.rq_tag", "")).startswith("Ablation_")
                                  or row.get("params.rq_tag") == "RQ1_Ablation_Activation"],
        }
        outputs: Dict[str, Path] = {}
        figure_dir.mkdir(parents=True, exist_ok=True)
        for name, selected in groups.items():
            valid = [row for row in selected if self._numeric(row.get("metrics.private_test_acc", "")) is not None
                     and self._numeric(row.get("metrics.macro_f1", "")) is not None]
            if not valid:
                continue
            labels = [row.get("Name", "Run") for row in valid]
            public = [self._numeric(row["metrics.best_val_acc"]) for row in valid]
            private = [self._numeric(row["metrics.private_test_acc"]) for row in valid]
            macro = [self._numeric(row["metrics.macro_f1"]) * 100.0 for row in valid]
            x = np.arange(len(valid), dtype=float)
            width = 0.25
            fig, axis = plt.subplots(figsize=(max(9, len(valid) * 0.72), 5.5), constrained_layout=True)
            axis.bar(x - width, public, width, label="PublicTest validation Accuracy (%)")
            axis.bar(x, private, width, label="PrivateTest Accuracy (%)")
            axis.bar(x + width, macro, width, label="Private Macro-F1 (%)")
            axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
            axis.set_ylabel("Phần trăm (%)")
            axis.set_title(f"Metrics thực nghiệm — {name.upper()} (CSV summary; một run/seed)")
            axis.grid(axis="y", alpha=0.25)
            axis.legend()
            destination = figure_dir / f"{name}.png"
            fig.savefig(destination, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            outputs[name] = destination
        return outputs

    def export_all(self) -> Dict[str, object]:
        """Sinh mọi bảng chuẩn và vector plot nếu dữ liệu nguồn hợp lệ."""
        outputs: Dict[str, object] = {"tables": self.export_summary_tables()}
        outputs["tables"]["compute_cost"] = self.export_compute_table()
        outputs["plot"] = self.export_vector_plot()
        outputs["figures"] = self.export_comparison_figures()
        return outputs

    @classmethod
    def cli(cls, argv: Optional[Sequence[str]] = None) -> int:
        """Chạy exporter trực tiếp với thư mục gốc BTL02 được cấu hình."""
        parser = argparse.ArgumentParser(description="Xuất artifacts LaTeX từ logs BTL02.")
        parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
        parser.add_argument("--dpi", type=int, default=300)
        args = parser.parse_args(argv)
        if args.dpi <= 0:
            parser.error("--dpi phải là số nguyên dương.")
        root = args.project_root.resolve()
        exporter = cls(root / "outputs" / "logs", root / "docs" / "thesis" / "generated",
                       root / "outputs" / "figures")
        artifacts = exporter.export_all()
        exporter.export_comparison_figures(dpi=args.dpi)
        print(f"Export complete: {len(artifacts['tables'])} TeX tables; comparison figures saved under outputs/figures/t0602.")
        return 0


if __name__ == "__main__":
    raise SystemExit(ThesisArtifactExporter.cli())