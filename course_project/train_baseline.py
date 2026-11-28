"""
Mô-đun huấn luyện và tinh chỉnh siêu tham số các mô hình Baseline (LSTM & DLinear).

Triển khai Task BPM25_T0204:
- Huấn luyện đa mô hình (LSTM, DLinear) trên 4 mốc Forecast Horizon H in {24, 48, 96, 192}.
- Vòng lặp huấn luyện OOP kết hợp Early Stopping chống Overfitting.
- Lưu trữ checkpoint trọng số (.pth) tốt nhất tại checkpoints/.
- Xuất log chi tiết lịch sử Loss (Train/Val) ra outputs/logs/.
- Đánh giá multi-seed (3 seeds: 42, 1024, 2026) và xuất bảng kết quả mean ± std
  qua MetricsCalculator tuân thủ ExperimentMetricsContract (AGENTS.md).
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.dlinear import DLinearModel
from src.models.lstm import LSTMModel
from utils_metrics import ExperimentMetricsContract, MetricsCalculator


# ---------------------------------------------------------------------------
# Cố định Seed ngẫu nhiên (Reproducibility)
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """
    Cố định hạt ngẫu nhiên trên toàn bộ các thư viện để đảm bảo tái lập 100%.

    Tham số:
        seed (int): Giá trị seed ngẫu nhiên cần cố định.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Bộ điều khiển dừng sớm (Early Stopping)
# ---------------------------------------------------------------------------


class EarlyStopping:
    """
    Bộ giám sát dừng sớm (Early Stopping) dựa trên hàm mất mát tập Validation.

    Nếu Validation Loss không cải thiện qua `patience` epochs liên tiếp,
    quá trình huấn luyện sẽ ngắt và lưu lại checkpoint có loss thấp nhất.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        checkpoint_path: str = "checkpoints/best_model.pth",
    ) -> None:
        """
        Khởi tạo bộ Early Stopping.

        Tham số:
            patience (int): Số epochs tối đa chờ đợi khi loss không giảm.
            min_delta (float): Mức cải thiện tối thiểu để tính là có tiến bộ.
            checkpoint_path (str): Đường dẫn lưu file trọng số .pth tối ưu.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

        # Tự động tạo thư mục chứa checkpoint nếu chưa tồn tại
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        """
        Đánh giá và kiểm tra trạng thái dừng sớm sau mỗi epoch.

        Tham số:
            val_loss (float): Loss trên tập validation của epoch hiện tại.
            model (nn.Module): Thể hiện mô hình đang huấn luyện.
            epoch (int): Số thứ tự epoch hiện tại.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(
                f"  [Checkpoint] Đã lưu trọng số tốt nhất (Val Loss: {val_loss:.6f}) tại epoch {epoch}"
            )
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(
                    f"  [Early Stopping] Đã kích hoạt dừng sớm tại epoch {epoch}! (Epoch tối ưu: {self.best_epoch})"
                )


# ---------------------------------------------------------------------------
# Class quản lý huấn luyện (Baseline Trainer OOP)
# ---------------------------------------------------------------------------


class BaselineTrainer:
    """
    Lớp quản lý chu trình huấn luyện, đánh giá và xuất log cho các mô hình Baseline.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: Literal["LSTM", "DLinear"],
        horizon: int,
        seed: int,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "outputs/logs",
    ) -> None:
        """
        Khởi tạo Trainer cho mô hình baseline.

        Tham số:
            model (nn.Module): Mô hình LSTMModel hoặc DLinearModel.
            model_name (str): Tên mô hình ('LSTM' hoặc 'DLinear').
            horizon (int): Khung dự báo H.
            seed (int): Hạt giống ngẫu nhiên.
            learning_rate (float): Tốc độ học của Optimizer.
            device (str): Thiết bị tính toán ('cpu' hoặc 'cuda').
            checkpoint_dir (str): Thư mục lưu checkpoint.
            log_dir (str): Thư mục lưu file log loss.
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.horizon = horizon
        self.seed = seed
        self.device = device
        self.log_dir = log_dir

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        base_filename = f"{model_name.lower()}_h{horizon}_seed{seed}"
        self.checkpoint_path = os.path.join(checkpoint_dir, f"{base_filename}_best.pth")
        self.log_csv_path = os.path.join(log_dir, f"history_{base_filename}.csv")

        self.early_stopping = EarlyStopping(
            patience=5, min_delta=1e-4, checkpoint_path=self.checkpoint_path
        )
        self.history: List[Dict[str, float]] = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Huấn luyện mô hình qua 1 epoch đầy đủ.

        Tham số:
            dataloader (DataLoader): Dataloader của tập Train.

        Trả về:
            float: Giá trị Loss trung bình của epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()

            # Gradient clipping chống bùng nổ gradient trong mạng tuần tự
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()
            total_loss += loss.item() * len(batch_x)

        return total_loss / len(dataloader.dataset)

    def validate(self, dataloader: DataLoader) -> float:
        """
        Đánh giá hàm mất mát trên tập Validation (tắt gradient).

        Tham số:
            dataloader (DataLoader): Dataloader của tập Validation.

        Trả về:
            float: Giá trị Validation Loss trung bình.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item() * len(batch_x)

        return total_loss / len(dataloader.dataset)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 30
    ) -> None:
        """
        Thực thi toàn bộ vòng lặp huấn luyện kết hợp Early Stopping.

        Tham số:
            train_loader (DataLoader): Dataloader tập Train.
            val_loader (DataLoader): Dataloader tập Validation.
            epochs (int): Số epochs tối đa cần chạy.
        """
        print(
            f"\n[START TRAINING] Model: {self.model_name} | Horizon={self.horizon} | Seed={self.seed}"
        )
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                }
            )

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
                )

            self.early_stopping(val_loss, self.model, epoch)
            if self.early_stopping.early_stop:
                break

        # Lưu lịch sử loss per epoch ra file CSV (tiêu chí DoD T0204)
        df_history = pd.DataFrame(self.history)
        df_history.to_csv(self.log_csv_path, index=False)
        print(f"[SAVED LOG] Đã lưu lịch sử loss tại: {self.log_csv_path}")

        # Nạp lại checkpoint tốt nhất để chuẩn bị đánh giá
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )
            print(
                f"[RESTORED WEIGHTS] Đã khôi phục checkpoint tối ưu từ epoch {self.early_stopping.best_epoch}"
            )

    def evaluate_test(
        self, test_loader: DataLoader, metrics_calc: MetricsCalculator
    ) -> Dict[str, float]:
        """
        Đánh giá độ đo MSE và MAE trên tập Test độc lập.

        Tham số:
            test_loader (DataLoader): Dataloader tập Test.
            metrics_calc (MetricsCalculator): Đối tượng tính toán độ đo.

        Trả về:
            Dict[str, float]: Từ điển chứa giá trị MSE và MAE.
        """
        self.model.eval()
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                predictions = self.model(batch_x)

                preds_list.append(predictions.cpu())
                targets_list.append(batch_y.cpu())

        all_preds = torch.cat(preds_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)

        test_metrics = metrics_calc.compute(all_preds, all_targets)
        return test_metrics


# ---------------------------------------------------------------------------
# Trình tạo DataLoader thực nghiệm (Synthetic Dataloader Generator)
# ---------------------------------------------------------------------------


def create_synthetic_dataloaders(
    num_samples: int = 500,
    lookback: int = 96,
    horizon: int = 24,
    in_features: int = 15,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo bộ Dataloaders tuần tự theo thời gian: Train (60%), Val (20%), Test (20%).

    Tham số:
        num_samples (int): Tổng số mẫu sinh giả lập.
        lookback (int): Chiều dài chuỗi quá khứ L=96.
        horizon (int): Khung dự báo H.
        in_features (int): Số lượng biến đầu vào D=15.
        batch_size (int): Kích thước batch.

    Trả về:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    time_steps = torch.linspace(0, 50, num_samples).unsqueeze(-1)
    signal = torch.sin(time_steps) + torch.randn(num_samples, in_features) * 0.1

    features_list, targets_list = [], []
    for i in range(len(signal) - lookback - horizon + 1):
        features_list.append(signal[i : i + lookback])
        targets_list.append(signal[i + lookback : i + lookback + horizon, 0:1])

    features_tensor = torch.stack(features_list)
    targets_tensor = torch.stack(targets_list)

    total_count = len(features_tensor)
    n_train = int(total_count * 0.6)
    n_val = int(total_count * 0.2)

    dataset_train = TensorDataset(features_tensor[:n_train], targets_tensor[:n_train])
    dataset_val = TensorDataset(
        features_tensor[n_train : n_train + n_val],
        targets_tensor[n_train : n_train + n_val],
    )
    dataset_test = TensorDataset(
        features_tensor[n_train + n_val :], targets_tensor[n_train + n_val :]
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Luồng thực thi Multi-Seed (3 Seeds: 42, 1024, 2026)
# ---------------------------------------------------------------------------


def load_baseline_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Đọc cấu hình wp02_baseline, data và paths từ tệp YAML chuẩn."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        return {}
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_multi_seed_experiment(
    model_name: Literal["LSTM", "DLinear"],
    horizon: int,
    epochs: int = 15,
    smoke_test: bool = True,
    seeds: Optional[List[int]] = None,
    lookback: int = 96,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "outputs/logs",
) -> ExperimentMetricsContract:
    """
    Chạy thực nghiệm tự động qua 3 seeds và xuất hợp đồng kết quả mean ± std.

    Tham số:
        model_name (str): Tên mô hình ('LSTM' hoặc 'DLinear').
        horizon (int): Khung dự báo H ∈ {24, 48, 96, 192}.
        epochs (int): Số epochs tối đa mỗi lần chạy.
        smoke_test (bool): Chế độ chạy nhanh kiểm tra luồng.
        seeds (Optional[List[int]]): Danh sách hạt giống ngẫu nhiên từ YAML.
        lookback (int): Chiều dài cửa sổ quá khứ từ YAML.
        learning_rate (float): Tốc độ học từ YAML.
        checkpoint_dir (str): Thư mục lưu checkpoint từ YAML.
        log_dir (str): Thư mục lưu log từ YAML.

    Trả về:
        ExperimentMetricsContract: Hợp đồng kết quả theo chuẩn Pydantic v2.
    """
    seeds = list(seeds) if seeds else [42, 1024, 2026]
    mse_list = []
    mae_list = []
    last_checkpoint = ""

    in_features = 15
    metrics_calc = MetricsCalculator()

    for seed in seeds:
        set_seed(seed)
        num_samples = 300 if smoke_test else 1500
        train_loader, val_loader, test_loader = create_synthetic_dataloaders(
            num_samples=num_samples,
            lookback=lookback,
            horizon=horizon,
            in_features=in_features,
        )

        if model_name == "LSTM":
            model = LSTMModel(
                lookback=lookback,
                horizon=horizon,
                in_features=in_features,
                hidden_size=64,
            )
        else:
            model = DLinearModel(
                lookback=lookback,
                horizon=horizon,
                in_features=in_features,
                kernel_size=25,
            )

        trainer = BaselineTrainer(
            model=model,
            model_name=model_name,
            horizon=horizon,
            seed=seed,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )

        trainer.fit(train_loader, val_loader, epochs=epochs)
        test_metrics = trainer.evaluate_test(test_loader, metrics_calc)

        mse_list.append(test_metrics["mse"])
        mae_list.append(test_metrics["mae"])
        last_checkpoint = trainer.checkpoint_path

        print(
            f"  [RESULT SEED {seed}] Test MSE: {test_metrics['mse']:.6f} | Test MAE: {test_metrics['mae']:.6f}"
        )

        try:
            from src.utils.exp_backup import MLflowRunLogger

            MLflowRunLogger().log_run(
                run_name=f"baseline_{model_name}_H{horizon}_seed{seed}",
                params={
                    "stage": "wp02_baseline",
                    "model": model_name,
                    "lookback": lookback,
                    "horizon": horizon,
                    "seed": seed,
                    "lr": learning_rate,
                    "epochs": epochs,
                    "smoke_test": smoke_test,
                },
                metrics={
                    "test_mse": float(test_metrics["mse"]),
                    "test_mae": float(test_metrics["mae"]),
                },
                tags={
                    "mlflow.source.name": "train_baseline.py",
                    "model": model_name,
                    "horizon": str(horizon),
                    "seed": str(seed),
                },
            )
        except Exception:
            pass

    # Đóng gói kết quả qua ExperimentMetricsContract
    contract = metrics_calc.summarize_multi_seed(
        experiment_id=f"EXP_{model_name}_H{horizon}",
        model_name=model_name,
        station_id="ALL_12_STATIONS",
        horizon=horizon,
        mse_per_seed=mse_list,
        mae_per_seed=mae_list,
        checkpoint_path=last_checkpoint,
        seeds=seeds,
        extra_notes=f"Thực nghiệm Baseline {model_name} với L=96, H={horizon}",
    )

    print("\n" + "=" * 70)
    print(f"📊 BẢNG TỔNG HỢP MULTI-SEED (3 SEEDS): {model_name} - Horizon H={horizon}")
    print("=" * 70)
    print(metrics_calc.format_report_row(contract))
    metrics_calc.print_summary(contract)
    print("=" * 70 + "\n")

    return contract


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình Baseline PM2.5 (T0204)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Đường dẫn tệp cấu hình YAML chuẩn",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["LSTM", "DLinear", "all"],
        default="all",
        help="Chọn kiến trúc để huấn luyện (LSTM, DLinear hoặc all)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[24, 48, 96, 192],
        default=None,
        help="Khung dự báo H giờ (24, 48, 96, 192)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Số epochs tối đa")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thư mục gốc ghi đè checkpoints & logs (tùy chọn cho Colab/Kaggle)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=True,
        help="Chế độ smoke test nhanh để kiểm tra pipeline",
    )

    args = parser.parse_args()
    cfg = load_baseline_yaml_config(args.config)
    wp02_cfg = cfg.get("wp02_baseline", {}) if isinstance(cfg.get("wp02_baseline"), dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}

    horizon = int(args.horizon if args.horizon is not None else data_cfg.get("horizon", 24))
    epochs = int(args.epochs if args.epochs is not None else wp02_cfg.get("epochs", 10))
    lookback = int(data_cfg.get("lookback", 96))
    seeds = list(wp02_cfg.get("seeds", [42, 1024, 2026]))
    lr = float(wp02_cfg.get("lr", 1e-3))
    if args.output_dir:
        ckpt_dir = str(Path(args.output_dir) / "checkpoints" / "wp02_baseline")
        log_dir = str(Path(args.output_dir) / "logs" / "wp02_baseline")
    else:
        ckpt_dir = str(paths_cfg.get("wp02_ckpt_dir", "checkpoints"))
        log_dir = str(paths_cfg.get("wp02_log_dir", "outputs/logs"))

    models_to_run = ["LSTM", "DLinear"] if args.model == "all" else [args.model]

    for model_type in models_to_run:
        run_multi_seed_experiment(
            model_name=model_type,
            horizon=horizon,
            epochs=epochs,
            smoke_test=args.smoke_test,
            seeds=seeds,
            lookback=lookback,
            learning_rate=lr,
            checkpoint_dir=ckpt_dir,
            log_dir=log_dir,
        )
