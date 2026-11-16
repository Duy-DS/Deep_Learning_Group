import os
from typing import Any, Dict, Optional
import mlflow
import pandas as pd


class MLflowLogger:
    """Lớp quản lý ghi nhận tiến trình huấn luyện và chỉ số thực nghiệm qua MLflow."""

    def __init__(self, experiment_name: str = "BTL02_FER2013_PyTorch", tracking_uri: str = "mlruns") -> None:
        """Khởi tạo logger với tên thực nghiệm và thư mục lưu trữ mlruns."""
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        mlflow.set_tracking_uri(tracking_uri)
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
        """Trích xuất tất cả các runs thành file CSV đồng bộ với Tab 4 Google Sheets."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        runs_df = mlflow.search_runs(experiment_names=[self.experiment_name])
        runs_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path
