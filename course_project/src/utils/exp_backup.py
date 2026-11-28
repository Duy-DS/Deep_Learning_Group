"""
Mô-đun tiện ích ghi nhận MLflow, xuất CSV chuẩn `Raw_MLflow_Logs` và đóng gói
sao lưu nhật ký thực nghiệm cho dự án DL-PM25-BJ (Colab / Kaggle / Local).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import zipfile

import pandas as pd
import yaml


class MLflowRunLogger:
    """
    Lớp bao bọc ghi nhận thực nghiệm MLflow an toàn theo biến môi trường `MLFLOW_TRACKING_URI`.

    Nếu thư viện `mlflow` chưa được cài đặt hoặc cấu hình URI gặp sự cố, lớp tự động
    bỏ qua mà không làm gián đoạn quá trình huấn luyện.
    """

    def __init__(
        self,
        experiment_name: str = "DL-PM25-BJ",
        tracking_uri: Optional[str] = None,
    ) -> None:
        """
        Khởi tạo logger MLflow với tên thí nghiệm và Tracking URI động.

        Tham số:
            experiment_name (str): Tên experiment trên MLflow.
            tracking_uri (Optional[str]): URI lưu trữ MLflow; mặc định đọc từ `MLFLOW_TRACKING_URI` hoặc `mlruns`.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        self._mlflow: Any = None
        try:
            import mlflow  # type: ignore

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
        except Exception:
            self._mlflow = None

    def log_run(
        self,
        run_name: str,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Ghi nhận một run hoàn chỉnh (params, metrics, tags) lên MLflow.

        Tham số:
            run_name (str): Tên định danh run.
            params (Optional[Dict[str, Any]]): Từ điển siêu tham số.
            metrics (Optional[Dict[str, float]]): Từ điển độ đo đánh giá.
            tags (Optional[Dict[str, str]]): Từ điển thẻ phân loại (ví dụ `mlflow.source.name`).

        Trả về:
            Optional[str]: Run ID nếu ghi thành công, ngược lại trả về `None`.
        """
        if self._mlflow is None:
            return None
        try:
            with self._mlflow.start_run(run_name=run_name) as active_run:
                if tags:
                    clean_tags = {str(k): str(v) for k, v in tags.items() if v is not None}
                    if clean_tags:
                        self._mlflow.set_tags(clean_tags)
                if params:
                    clean_params = {
                        str(k): str(v) for k, v in params.items() if v is not None
                    }
                    if clean_params:
                        self._mlflow.log_params(clean_params)
                if metrics:
                    clean_metrics: Dict[str, float] = {}
                    for k, v in metrics.items():
                        try:
                            val = float(v)
                            if val == val:  # Loại bỏ NaN
                                clean_metrics[str(k)] = val
                        except (TypeError, ValueError):
                            continue
                    if clean_metrics:
                        self._mlflow.log_metrics(clean_metrics)
                return str(active_run.info.run_id)
        except Exception:
            return None


class MLflowLogExporter:
    """
    Lớp quét thư mục `mlruns/` và xuất toàn bộ nhật ký thực nghiệm sang CSV
    khớp 100% cấu trúc cột của sheet `Raw_MLflow_Logs`.
    """

    BASE_COLUMNS: List[str] = [
        "Start Time",
        "Duration",
        "Run ID",
        "Name",
        "Source Name",
        "Status",
    ]
    _STATUS_MAP: Dict[int, str] = {
        1: "RUNNING",
        2: "SCHEDULED",
        3: "FINISHED",
        4: "FAILED",
        5: "KILLED",
    }

    def __init__(self, mlruns_dir: str | Path = "mlruns") -> None:
        """
        Khởi tạo bộ xuất log MLflow từ thư mục file-store `mlruns/`.

        Tham số:
            mlruns_dir (str | Path): Đường dẫn thư mục gốc chứa các experiment MLflow.
        """
        self.mlruns_dir = Path(mlruns_dir)

    @staticmethod
    def _format_timestamp_ms(ts_ms: Any) -> str:
        """Chuyển đổi timestamp mili-giây sang chuỗi UTC `YYYY-MM-DD HH:MM:SS`."""
        try:
            if ts_ms is None:
                return ""
            dt = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError, OSError):
            return ""

    @staticmethod
    def _format_duration(start_ms: Any, end_ms: Any) -> str:
        """Tính thời lượng chạy theo định dạng dễ đọc (giây hoặc phút)."""
        try:
            if start_ms is None or end_ms is None:
                return ""
            diff_sec = max(0.0, (float(end_ms) - float(start_ms)) / 1000.0)
            if diff_sec < 60.0:
                return f"{diff_sec:.1f}s"
            return f"{diff_sec / 60.0:.2f}min"
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _read_key_value_dir(folder: Path, prefix: str, is_metric: bool = False) -> Dict[str, Any]:
        """Đọc các tệp tham số, độ đo hoặc thẻ bên trong một thư mục run MLflow."""
        result: Dict[str, Any] = {}
        if not folder.exists() or not folder.is_dir():
            return result
        for item in sorted(folder.iterdir()):
            if not item.is_file():
                continue
            try:
                content = item.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if is_metric:
                # Mỗi dòng metric MLflow có dạng: "<timestamp> <value> <step>"
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                if lines:
                    parts = lines[-1].split()
                    if len(parts) >= 2:
                        try:
                            result[f"{prefix}{item.name}"] = float(parts[1])
                        except ValueError:
                            result[f"{prefix}{item.name}"] = parts[1]
            else:
                result[f"{prefix}{item.name}"] = content
        return result

    def collect_runs(self) -> pd.DataFrame:
        """
        Quét đệ quy thư mục `mlruns/` để tổng hợp tất cả run thành DataFrame chuẩn `Raw_MLflow_Logs`.

        Trả về:
            pd.DataFrame: Bảng dữ liệu gồm `Start Time`, `Duration`, `Run ID`, `Name`,
            `Source Name`, `Status`, `params.*`, `metrics.*`, `tags.*`.
        """
        if not self.mlruns_dir.exists():
            return pd.DataFrame(columns=self.BASE_COLUMNS)

        rows: List[Dict[str, Any]] = []
        param_cols: set[str] = set()
        metric_cols: set[str] = set()
        tag_cols: set[str] = set()

        for meta_path in sorted(self.mlruns_dir.glob("*/*/meta.yaml")):
            run_dir = meta_path.parent
            if run_dir.name.startswith(".") or run_dir.parent.name.startswith("."):
                continue
            try:
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if not isinstance(meta, dict) or "run_id" not in meta:
                continue

            params = self._read_key_value_dir(run_dir / "params", "params.", is_metric=False)
            metrics = self._read_key_value_dir(run_dir / "metrics", "metrics.", is_metric=True)
            tags = self._read_key_value_dir(run_dir / "tags", "tags.", is_metric=False)

            param_cols.update(params.keys())
            metric_cols.update(metrics.keys())
            tag_cols.update(tags.keys())

            raw_status = meta.get("status", "")
            status_str = (
                self._STATUS_MAP.get(raw_status, str(raw_status))
                if isinstance(raw_status, int)
                else str(raw_status)
            )
            run_name = meta.get("run_name") or tags.get("tags.mlflow.runName", "")
            source_name = tags.get("tags.mlflow.source.name", "")

            row: Dict[str, Any] = {
                "Start Time": self._format_timestamp_ms(meta.get("start_time")),
                "Duration": self._format_duration(meta.get("start_time"), meta.get("end_time")),
                "Run ID": str(meta.get("run_id", run_dir.name)),
                "Name": str(run_name),
                "Source Name": str(source_name),
                "Status": status_str,
            }
            row.update(params)
            row.update(metrics)
            row.update(tags)
            rows.append(row)

        ordered_cols = (
            self.BASE_COLUMNS
            + sorted(param_cols)
            + sorted(metric_cols)
            + sorted(tag_cols)
        )
        if not rows:
            return pd.DataFrame(columns=ordered_cols)
        df = pd.DataFrame(rows)
        for col in ordered_cols:
            if col not in df.columns:
                df[col] = ""
        return df[ordered_cols]

    def export_csv(self, csv_path: str | Path = "outputs/mlflow_runs_export.csv") -> Path:
        """
        Xuất bảng log MLflow ra tệp CSV chuẩn `Raw_MLflow_Logs`.

        Tham số:
            csv_path (str | Path): Đường dẫn tệp CSV đầu ra.

        Trả về:
            Path: Đường dẫn tệp CSV vừa được ghi.
        """
        out_path = Path(csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.collect_runs()
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[EXPORTED] Đã xuất {len(df)} MLflow runs ra: {out_path}")
        return out_path


class ExperimentBackupManager:
    """
    Lớp đóng gói an toàn toàn bộ nhật ký thực nghiệm (`mlruns/`, `outputs/logs/`,
    `outputs/ablation/`, `outputs/reports/`, `docs/` và tùy chọn `outputs/checkpoints/`)
    thành tệp nén `.zip` gắn timestamp `dl_pm25_bj_backup_[YYMMDD]_[HHMM].zip`.
    """

    def __init__(
        self,
        project_root: str | Path = ".",
        mlruns_dir: str | Path = "mlruns",
        output_dir: str | Path = "outputs",
        backup_dir: str | Path = "backups",
    ) -> None:
        """
        Khởi tạo trình quản lý sao lưu thực nghiệm.

        Tham số:
            project_root (str | Path): Thư mục gốc dự án (`course_project/`).
            mlruns_dir (str | Path): Thư mục lưu trữ MLflow runs.
            output_dir (str | Path): Thư mục đầu ra thực nghiệm (`outputs/`).
            backup_dir (str | Path): Thư mục chứa gói nén `.zip` đầu ra.
        """
        self.project_root = Path(project_root).resolve()
        self.mlruns_dir = self._resolve_path(mlruns_dir)
        self.output_dir = self._resolve_path(output_dir)
        self.backup_dir = self._resolve_path(backup_dir)

    def _resolve_path(self, target: str | Path) -> Path:
        """Chuẩn hóa đường dẫn tương đối dựa trên `project_root`."""
        p = Path(target)
        return p if p.is_absolute() else (self.project_root / p)

    def create_backup_archive(
        self,
        include_checkpoints: bool = False,
        include_docs: bool = True,
        export_mlflow_csv: bool = True,
        timestamp: Optional[str] = None,
    ) -> Path:
        """
        Tạo tệp `.zip` sao lưu toàn bộ kết quả thực nghiệm và nhật ký MLflow.

        Tham số:
            include_checkpoints (bool): Có đính kèm `outputs/checkpoints/` vào gói nén hay không.
            include_docs (bool): Có đính kèm thư mục `docs/` hay không.
            export_mlflow_csv (bool): Có tự động xuất `mlflow_runs_export.csv` trước khi nén hay không.
            timestamp (Optional[str]): Chuỗi thời gian `[YYMMDD]_[HHMM]` tùy chỉnh.

        Trả về:
            Path: Đường dẫn tuyệt đối tới tệp `.zip` vừa tạo.
        """
        ts = timestamp or datetime.now().strftime("%y%m%d_%H%M")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.backup_dir / f"dl_pm25_bj_backup_{ts}.zip"

        csv_path: Optional[Path] = None
        if export_mlflow_csv:
            exporter = MLflowLogExporter(self.mlruns_dir)
            csv_path = exporter.export_csv(self.output_dir / "mlflow_runs_export.csv")

        target_dirs: List[tuple[Path, str]] = [
            (self.mlruns_dir, "mlruns"),
            (self.output_dir / "logs", "outputs/logs"),
            (self.output_dir / "ablation", "outputs/ablation"),
            (self.output_dir / "reports", "outputs/reports"),
        ]
        if include_checkpoints:
            target_dirs.append((self.output_dir / "checkpoints", "outputs/checkpoints"))
        if include_docs:
            target_dirs.append((self.project_root / "docs", "docs"))

        packed_files = 0
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if csv_path is not None and csv_path.exists():
                zf.write(csv_path, arcname="outputs/mlflow_runs_export.csv")
                packed_files += 1

            for src_dir, arc_prefix in target_dirs:
                if not src_dir.exists():
                    continue
                for file_path in sorted(src_dir.rglob("*")):
                    if not file_path.is_file():
                        continue
                    rel = file_path.relative_to(src_dir)
                    zf.write(file_path, arcname=str(Path(arc_prefix) / rel))
                    packed_files += 1

        print(f"[BACKUP] Đã đóng gói {packed_files} tệp tại: {zip_path}")
        return zip_path


def parse_args() -> argparse.Namespace:
    """Phân tích tham số dòng lệnh cho tác vụ xuất log MLflow và đóng gói sao lưu."""
    parser = argparse.ArgumentParser(
        description="Xuất mlflow_runs_export.csv và đóng gói backup thực nghiệm DL-PM25-BJ."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--mlruns-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--backup-dir", type=Path, default=Path("backups"))
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Đính kèm thư mục outputs/checkpoints/ vào tệp zip sao lưu.",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Bỏ qua thư mục docs/ khi đóng gói.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manager = ExperimentBackupManager(
        project_root=args.project_root,
        mlruns_dir=args.mlruns_dir,
        output_dir=args.output_dir,
        backup_dir=args.backup_dir,
    )
    manager.create_backup_archive(
        include_checkpoints=args.include_checkpoints,
        include_docs=(not args.skip_docs),
        export_mlflow_csv=True,
    )

