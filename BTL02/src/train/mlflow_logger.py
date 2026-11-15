import mlflow
import os 
from typing import Dict, Any, Optional


class MLflowLogger:
    def __init__(self, experiment_name: str = "BTL02_FER2013_PyTorch", tracking_uri: str = "mlruns"):
        # thụt vào 1 Tab (4 dấu cách)
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def start_run(self, run_name: str, params: Optional[Dict[str, Any]] = None, tags: Optional[Dict[str, str]] = None):
        # 1. Gọi mlflow.start_run(run_name=run_name)
        # 2. Nếu có tags thì gọi mlflow.set_tags(tags)
        # 3. Nếu có params thì gọi mlflow.log_params(params)
        mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)


    def log_metrics(self, metrics: Dict[str,float], step: int):
        # Chặng 4: hàm mới nằm ở đây
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, file_path: str):
        # Chặng 5a: hàm mới nằm ở đây
        mlflow.log_artifact(file_path)
    def end_run(self):
        # Chặng 5b: hàm mới nằm ở đây
        mlflow.end_run()
    def export_summary_csv(self, output_path: str= "outputs/logs/mlflow_summary_24col.csv") -> None:
        """
        Trích xuất tất cả các runs thành file CSV đồng bộ với Tab 4 Google Sheets.
        """
        import pandas as pd
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        runs_df = mlflow.search_runs(experiment_names=[self.experiment_name])
        runs_df.to_csv(output_path,index = False, encoding = "utf-8-sig")
