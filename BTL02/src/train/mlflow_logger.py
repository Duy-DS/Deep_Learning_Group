import os
from typing import Any, Dict, Optional
import mlflow
import pandas as pd


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
        mlflow.set_tracking_uri(resolved_tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

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
        runs_df = mlflow.search_runs(experiment_names=[self.experiment_name])
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
        summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path
