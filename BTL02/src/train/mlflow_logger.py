import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.uri import is_local_uri


class MLflowLogger:
    """Lớp quản lý ghi nhận tiến trình huấn luyện và chỉ số thực nghiệm qua MLflow."""

    # Schema bất biến đồng bộ với tab Raw_MLflow_Logs trên Google Sheets.
    CSV_COLUMNS = [
        "Start Time", "Duration", "Run ID", "Name", "Source Name", "Status",
        "params.architecture", "params.exp_group", "params.rq_tag", "params.seed",
        "params.image_size", "params.augmentation", "params.optimizer", "params.lr",
        "params.scheduler", "params.batch_size", "params.dropout", "metrics.train_loss",
        "metrics.train_acc", "metrics.best_val_acc", "metrics.private_test_acc",
        "metrics.macro_f1", "metrics.best_epoch", "tags.member",
    ]

    def __init__(
        self,
        experiment_name: str = "BTL02_FER2013_PyTorch",
        tracking_uri: Optional[str] = None,
    ) -> None:
        """Khởi tạo logger, ưu tiên URI tường minh rồi đến biến môi trường.

        Args:
            experiment_name: Tên experiment trong MLflow.
            tracking_uri: URI lưu trữ tùy chọn; nếu bỏ trống dùng biến môi trường
                ``MLFLOW_TRACKING_URI`` hoặc mặc định ``mlruns``.
        """
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        resolved_tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        self.tracking_uri = resolved_tracking_uri
        mlflow.set_tracking_uri(resolved_tracking_uri)
        self._ensure_file_store_trash_directory()
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def _ensure_file_store_trash_directory(self) -> None:
        """Khôi phục thư mục trash mà MLflow FileStore cần, kể cả trên Drive mount."""
        if not is_local_uri(self.tracking_uri):
            return
        tracking_root = Path(local_file_uri_to_path(self.tracking_uri))
        os.makedirs(tracking_root, exist_ok=True)
        os.makedirs(tracking_root / ".trash", exist_ok=True)

    @staticmethod
    def _sanitize_python_scalar(value: Any) -> Any:
        """Chuyển giá trị NumPy/lồng nhau thành scalar Python an toàn cho CSV."""
        if isinstance(value, np.generic):
            return MLflowLogger._sanitize_python_scalar(value.item())
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return MLflowLogger._sanitize_python_scalar(value.item())
            return str(MLflowLogger._sanitize_python_scalar(value.tolist()))
        if isinstance(value, (list, tuple, dict)):
            if isinstance(value, dict):
                value = {
                    str(key): MLflowLogger._sanitize_python_scalar(item)
                    for key, item in value.items()
                }
            else:
                value = [MLflowLogger._sanitize_python_scalar(item) for item in value]
            return str(value)
        if value is None or type(value) in (bool, int, float, str):
            return value
        return str(value)

    @classmethod
    def normalize_records(cls, records: Any) -> list[dict[str, Any]]:
        """Chuẩn hóa các record NumPy trước khi tạo DataFrame hoặc ghi CSV."""
        normalized = []
        for record in records:
            if isinstance(record, pd.Series):
                record = record.to_dict()
            if not isinstance(record, dict):
                raise TypeError("Mỗi record cần chuẩn hóa phải là dictionary.")
            normalized.append(
                {
                    str(key): cls._sanitize_python_scalar(value)
                    for key, value in record.items()
                }
            )
        return normalized

    def start_run(
        self,
        run_name: str,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        exp_group: Optional[str] = None,
    ) -> None:
        """Bắt đầu một phiên ghi nhận (MLflow run)."""
        all_tags = dict(tags or {})
        if exp_group:
            all_tags["exp_group"] = exp_group

        mlflow.start_run(run_name=run_name)
        if all_tags:
            mlflow.set_tags(all_tags)
        if params:
            mlflow.log_params(params)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Ghi nhận tập tham số cấu hình."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Ghi nhận các chỉ số mất mát và độ chính xác."""
        self._ensure_file_store_trash_directory()
        if step is not None:
            mlflow.log_metrics(metrics, step=step)
        else:
            mlflow.log_metrics(metrics)

    def log_artifact(self, file_path: str) -> None:
        """Đính kèm tệp tin trọng số hoặc đồ thị vào phiên chạy."""
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path)

    def end_run(self) -> None:
        """Kết thúc phiên chạy hiện tại an toàn."""
        mlflow.end_run()

    def export_summary_csv(self, output_path: str = "outputs/logs/mlflow_summary_24col.csv") -> str:
        """Xuất MLflow runs với đúng 24 cột theo thứ tự của Google Sheets."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        raw_runs = mlflow.search_runs(experiment_names=[self.experiment_name])
        safe_runs = self.normalize_records(raw_runs.to_dict(orient="records"))
        runs_df = pd.DataFrame(safe_runs)
        column_mapping = {
            "Start Time": "start_time",
            "Run ID": "run_id",
            "Name": "tags.mlflow.runName",
            "Source Name": "tags.mlflow.source.name",
            "Status": "status",
            **{column: column for column in self.CSV_COLUMNS if column.startswith(("params.", "metrics.", "tags."))},
        }
        summary_df = pd.DataFrame(index=runs_df.index)
        for output_column, source_column in column_mapping.items():
            if source_column in runs_df.columns:
                summary_df[output_column] = runs_df[source_column]
            else:
                summary_df[output_column] = "None"

        if "end_time" in runs_df.columns and "start_time" in runs_df.columns:
            start_time = pd.to_datetime(runs_df["start_time"], errors="coerce", utc=True)
            end_time = pd.to_datetime(runs_df["end_time"], errors="coerce", utc=True)
            duration = (end_time - start_time).dt.total_seconds()
            summary_df["Duration"] = duration.where(duration.notna(), "None")
        else:
            summary_df["Duration"] = "None"

        summary_df = summary_df.reindex(columns=self.CSV_COLUMNS)
        assert len(summary_df.columns) == 24, "Schema CSV MLflow phải có đúng 24 cột."
        safe_records = self.normalize_records(summary_df.to_dict(orient="records"))
        summary_df = pd.DataFrame(safe_records, columns=self.CSV_COLUMNS)
        summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path
