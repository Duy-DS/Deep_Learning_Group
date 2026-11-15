"""
Pipeline huấn luyện toàn cục (Global Pre-training) cho 4 mô hình dự báo PM2.5 Bắc Kinh
trên dữ liệu 12 trạm quan trắc.
Hỗ trợ:
- Huấn luyện đa hạt giống (3 seeds mặc định: 42, 1024, 2026) và tổng hợp mean ± std.
- Đo thời gian huấn luyện thực tế (train_time_sec) và VRAM đỉnh (peak_vram_mb).
- Tăng tốc với train_stride, Mixed Precision (AMP) và truyền dữ liệu bất đồng bộ.
"""

import argparse
import gc
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.dlinear import DLinearModel
from src.models.informer import InformerModel
from src.models.lstm import LSTMModel
from src.models.patchtst import PatchTSTModel

STATIONS_12 = [
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
]

TARGET_COL = "PM2.5"
FEATURE_COLS_MULTIVARIATE = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEW",
    "RAIN",
    "WSPM",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]
FEATURE_COLS_UNIVARIATE = ["PM2.5"]


def set_seed(seed: int) -> None:
    """Cố định hạt giống ngẫu nhiên cho toàn bộ thư viện để đảm bảo tái lập thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_feature_columns(mode: str) -> List[str]:
    """Trả về danh sách cột đặc trưng theo chế độ multivariate hoặc univariate."""
    if mode == "multivariate":
        return list(FEATURE_COLS_MULTIVARIATE)
    if mode == "univariate":
        return list(FEATURE_COLS_UNIVARIATE)
    raise ValueError(f"Chế độ feature_mode không hợp lệ: {mode}")


def preprocess_station_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu chuỗi thời gian cho một trạm:
    sắp xếp thời gian, nội suy tuyến tính + forward fill và tạo đặc trưng chu kỳ sin/cos.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "DEWP" in df.columns and "DEW" not in df.columns:
        df = df.rename(columns={"DEWP": "DEW"})
    if {"year", "month", "day", "hour"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df = df.sort_values("datetime").reset_index(drop=True)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

    numeric_cols = [
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEW",
        "RAIN",
        "WSPM",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            df[col] = df[col].ffill().bfill()

    if "datetime" in df.columns:
        hours = df["datetime"].dt.hour.to_numpy(dtype=np.float32)
        months = df["datetime"].dt.month.to_numpy(dtype=np.float32)
    else:
        hours = df["hour"].to_numpy(dtype=np.float32)
        months = df["month"].to_numpy(dtype=np.float32)

    df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    df["month_sin"] = np.sin(2.0 * np.pi * (months - 1.0) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (months - 1.0) / 12.0)

    return df


def split_chronological(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float, float] = (0.6, 0.1, 0.1, 0.2),
) -> Dict[str, pd.DataFrame]:
    """Chia tuần tự theo thời gian tuyến tính 60/10/10/20 (train, train_dev, dev, test)."""
    n = len(df)
    n_train = int(n * ratios[0])
    n_train_dev = int(n * (ratios[0] + ratios[1]))
    n_dev = int(n * (ratios[0] + ratios[1] + ratios[2]))

    return {
        "train": df.iloc[:n_train].reset_index(drop=True),
        "train_dev": df.iloc[n_train:n_train_dev].reset_index(drop=True),
        "dev": df.iloc[n_train_dev:n_dev].reset_index(drop=True),
        "test": df.iloc[n_dev:].reset_index(drop=True),
    }


def make_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = TARGET_COL,
    lookback: int = 96,
    horizon: int = 24,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tạo các cặp cửa sổ trượt (X: [N, L, D], y: [N, H]) nội trạm."""
    values_x = df[feature_cols].to_numpy(dtype=np.float32)
    values_y = df[target_col].to_numpy(dtype=np.float32)

    total_len = len(df)
    max_start = total_len - lookback - horizon + 1
    if max_start <= 0:
        return (
            np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )

    indices = range(0, max_start, stride)
    X_list = [values_x[i : i + lookback] for i in indices]
    y_list = [values_y[i + lookback : i + lookback + horizon] for i in indices]

    return np.stack(X_list, axis=0), np.stack(y_list, axis=0)


def load_and_split_stations(
    data_dir: Path,
    stations: List[str] = STATIONS_12,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Đọc dữ liệu CSV của 12 trạm từ data/raw và chia tập theo thời gian."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu CSV trong thư mục: {data_dir}"
        )

    station_splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    for station in stations:
        matched = [f for f in csv_files if station.lower() in f.name.lower()]
        if not matched:
            raise FileNotFoundError(f"Không tìm thấy tệp CSV cho trạm: {station}")
        raw_df = pd.read_csv(matched[0])
        clean_df = preprocess_station_df(raw_df)
        station_splits[station] = split_chronological(clean_df)

    return station_splits


def prepare_global_12stations_data(
    data_dir: Path,
    mode: str = "multivariate",
    lookback: int = 96,
    horizon: int = 24,
    batch_size: int = 256,
    train_stride: int = 3,
    scaler_save_path: Optional[Path] = None,
) -> Tuple[Dict[str, DataLoader], StandardScaler, List[str]]:
    """
    Khớp StandardScaler duy nhất trên tập train gộp 12 trạm (chống Data Leakage),
    sau đó tạo DataLoader cho 4 tập: train, train_dev, dev, test.
    """
    feature_cols = get_feature_columns(mode)
    station_splits = load_and_split_stations(data_dir, STATIONS_12)

    # Fit StandardScaler CHỈ trên tập train của 12 trạm
    train_concat = pd.concat(
        [splits["train"][feature_cols] for splits in station_splits.values()],
        axis=0,
        ignore_index=True,
    )
    scaler = StandardScaler()
    scaler.fit(train_concat)

    if scaler_save_path is not None:
        scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_save_path)

    agg_X: Dict[str, List[np.ndarray]] = {
        k: [] for k in ["train", "train_dev", "dev", "test"]
    }
    agg_y: Dict[str, List[np.ndarray]] = {
        k: [] for k in ["train", "train_dev", "dev", "test"]
    }

    for _, splits in station_splits.items():
        for split_name in ["train", "train_dev", "dev", "test"]:
            scaled_df = splits[split_name].copy()
            scaled_df[feature_cols] = scaler.transform(scaled_df[feature_cols])

            current_stride = train_stride if split_name == "train" else 1
            X, y = make_sliding_windows(
                scaled_df,
                feature_cols=feature_cols,
                target_col=TARGET_COL,
                lookback=lookback,
                horizon=horizon,
                stride=current_stride,
            )
            agg_X[split_name].append(X)
            agg_y[split_name].append(y)

    use_pin_memory = torch.cuda.is_available()
    global_loaders: Dict[str, DataLoader] = {}
    for split_name in ["train", "train_dev", "dev", "test"]:
        X_all = np.concatenate(agg_X[split_name], axis=0)
        y_all = np.concatenate(agg_y[split_name], axis=0)
        ds = TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all))
        global_loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            drop_last=False,
            pin_memory=use_pin_memory,
        )

    return global_loaders, scaler, feature_cols


def build_model(
    model_name: str,
    lookback: int,
    horizon: int,
    in_features: int,
    device: torch.device,
) -> nn.Module:
    """Khởi tạo kiến trúc mô hình theo tên và chuyển sang thiết bị tính toán."""
    name = model_name.lower()
    if name == "lstm":
        model = LSTMModel(lookback=lookback, horizon=horizon, in_features=in_features)
    elif name == "dlinear":
        model = DLinearModel(
            lookback=lookback, horizon=horizon, in_features=in_features
        )
    elif name == "informer":
        model = InformerModel(
            lookback=lookback, horizon=horizon, in_features=in_features
        )
    elif name == "patchtst":
        model = PatchTSTModel(
            lookback=lookback, horizon=horizon, in_features=in_features
        )
    else:
        raise ValueError(f"Kiến trúc mô hình không được hỗ trợ: {model_name}")

    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """Huấn luyện 1 epoch với Mixed Precision (AMP) và truyền dữ liệu bất đồng bộ."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_amp = (scaler is not None) and (device.type == "cuda")

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(batch_x).squeeze(-1)
            loss = criterion(preds, batch_y)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Đánh giá mô hình trên tập dữ liệu và trả về Loss, MSE, RMSE, MAE."""
    model.eval()
    total_loss = 0.0
    total_se = 0.0
    total_ae = 0.0
    total_elements = 0
    total_samples = 0
    use_amp = device.type == "cuda"

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(batch_x).squeeze(-1)
            loss = criterion(preds, batch_y)

        preds_f32 = preds.float()
        diff = preds_f32 - batch_y.float()
        total_se += torch.sum(diff**2).item()
        total_ae += torch.sum(torch.abs(diff)).item()
        total_elements += batch_y.numel()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mse = total_se / max(total_elements, 1)
    mae = total_ae / max(total_elements, 1)
    rmse = float(np.sqrt(mse))
    return {
        "loss": total_loss / max(total_samples, 1),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


class TrainingPipeline:
    """
    Lớp điều phối toàn bộ quy trình Pre-training đa hạt giống (Multi-Seed),
    đo thời gian, VRAM và xuất báo cáo định dạng mean ± std.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        mode: str = "multivariate",
        lookback: int = 96,
        horizon: int = 24,
        batch_size: int = 256,
        train_stride: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        patience: int = 5,
        seeds: Optional[List[int]] = None,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.mode = mode
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.train_stride = train_stride
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.seeds = seeds if seeds is not None else [42, 1024, 2026]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = self.output_dir / "checkpoints" / "global"
        self.log_dir = self.output_dir / "logs"
        self.report_dir = self.output_dir / "reports"

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def train_single_seed(
        self,
        model_name: str,
        seed: int,
        loaders: Dict[str, DataLoader],
        in_features: int,
    ) -> Dict[str, float]:
        """Huấn luyện một mô hình trên một hạt giống cụ thể, đo thời gian và VRAM."""
        set_seed(seed)
        model = build_model(
            model_name=model_name,
            lookback=self.lookback,
            horizon=self.horizon,
            in_features=in_features,
            device=self.device,
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        grad_scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        seed_ckpt_path = (
            self.ckpt_dir
            / f"{model_name}_{self.mode}_H{self.horizon}_seed{seed}_pretrained.pt"
        )
        log_csv_path = (
            self.log_dir
            / f"pretrain_{model_name}_{self.mode}_H{self.horizon}_seed{seed}.csv"
        )

        best_dev_loss = float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        history_rows = []

        # Bắt đầu đo VRAM đỉnh và thời gian huấn luyện thực tế
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        t_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                dataloader=loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                scaler=grad_scaler,
            )
            train_dev_metrics = evaluate(
                model, loaders["train_dev"], criterion, self.device
            )
            dev_metrics = evaluate(model, loaders["dev"], criterion, self.device)
            dev_loss = dev_metrics["loss"]
            scheduler.step(dev_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "train_dev_loss": round(train_dev_metrics["loss"], 6),
                    "train_dev_mae": round(train_dev_metrics["mae"], 6),
                    "dev_loss": round(dev_loss, 6),
                    "dev_rmse": round(dev_metrics["rmse"], 6),
                    "dev_mae": round(dev_metrics["mae"], 6),
                    "lr": current_lr,
                }
            )

            print(
                f"[{model_name.upper()} | seed={seed}] Epoch {epoch:02d}/{self.epochs:02d} "
                f"| train_loss={train_loss:.5f} | dev_loss={dev_loss:.5f} "
                f"| dev_mae={dev_metrics['mae']:.5f} | lr={current_lr:.2e}"
            )

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_name": model_name,
                        "mode": self.mode,
                        "lookback": self.lookback,
                        "horizon": self.horizon,
                        "in_features": in_features,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_dev_loss": best_dev_loss,
                        "model_state_dict": model.state_dict(),
                    },
                    seed_ckpt_path,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(
                        f"--> Early stopping tại epoch {epoch} (tốt nhất: epoch {best_epoch})"
                    )
                    break

        # Chốt thời gian huấn luyện và mức VRAM cao nhất đã cấp phát
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
        else:
            peak_vram_mb = 0.0
        train_time_sec = time.perf_counter() - t_start

        pd.DataFrame(history_rows).to_csv(log_csv_path, index=False)

        # Nạp lại trọng số tốt nhất trên dev để đánh giá độc lập trên tập test
        ckpt = torch.load(seed_ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, loaders["test"], criterion, self.device)
        train_dev_final = evaluate(model, loaders["train_dev"], criterion, self.device)
        del model, optimizer, scheduler, grad_scaler
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "model": model_name,
            "mode": self.mode,
            "horizon": self.horizon,
            "seed": seed,
            "trainable_params": trainable_params,
            "best_epoch": best_epoch,
            "best_dev_loss": round(best_dev_loss, 6),
            "train_dev_mse": round(train_dev_final["mse"], 6),
            "train_dev_mae": round(train_dev_final["mae"], 6),
            "test_mse": round(test_metrics["mse"], 6),
            "test_rmse": round(test_metrics["rmse"], 6),
            "test_mae": round(test_metrics["mae"], 6),
            "train_time_sec": round(train_time_sec, 3),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "checkpoint_path": str(seed_ckpt_path),
        }

    def run(self, models: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chạy toàn bộ các mô hình qua danh sách seeds và xuất báo cáo tổng hợp."""
        scaler_path = self.ckpt_dir / f"scaler_12stations_{self.mode}.joblib"
        loaders, _, feature_cols = prepare_global_12stations_data(
            data_dir=self.data_dir,
            mode=self.mode,
            lookback=self.lookback,
            horizon=self.horizon,
            batch_size=self.batch_size,
            train_stride=self.train_stride,
            scaler_save_path=scaler_path,
        )
        in_features = len(feature_cols)
        print(
            f"Thiết bị: {self.device} | Mode: {self.mode} (D={in_features}) "
            f"| L={self.lookback}, H={self.horizon} | Seeds: {self.seeds}"
        )

        all_run_records = []
        summary_records = []
        ddof = 1 if len(self.seeds) > 1 else 0

        for model_name in models:
            print(f"\n{'=' * 70}\nHUẤN LUYỆN MÔ HÌNH: {model_name.upper()}\n{'=' * 70}")
            model_runs = []
            best_overall_dev_loss = float("inf")
            best_overall_ckpt_src: Optional[Path] = None

            for seed in self.seeds:
                rec = self.train_single_seed(
                    model_name=model_name,
                    seed=seed,
                    loaders=loaders,
                    in_features=in_features,
                )
                model_runs.append(rec)
                all_run_records.append(rec)

                if rec["best_dev_loss"] < best_overall_dev_loss:
                    best_overall_dev_loss = rec["best_dev_loss"]
                    best_overall_ckpt_src = Path(rec["checkpoint_path"])

            # Lưu bản checkpoint chuẩn *_pretrained.pt từ seed tốt nhất cho fine_tune.py
            if best_overall_ckpt_src is not None and best_overall_ckpt_src.exists():
                canonical_ckpt = (
                    self.ckpt_dir
                    / f"{model_name}_{self.mode}_H{self.horizon}_pretrained.pt"
                )
                canonical_ckpt.write_bytes(best_overall_ckpt_src.read_bytes())

            df_m = pd.DataFrame(model_runs)
            mse_mean = float(df_m["test_mse"].to_numpy().mean())
            mse_std = float(df_m["test_mse"].to_numpy().std(ddof=ddof))
            rmse_mean = float(df_m["test_rmse"].to_numpy().mean())
            rmse_std = float(df_m["test_rmse"].to_numpy().std(ddof=ddof))
            mae_mean = float(df_m["test_mae"].to_numpy().mean())
            mae_std = float(df_m["test_mae"].to_numpy().std(ddof=ddof))
            time_mean = float(df_m["train_time_sec"].to_numpy().mean())
            time_std = float(df_m["train_time_sec"].to_numpy().std(ddof=ddof))
            vram_mean = float(df_m["peak_vram_mb"].to_numpy().mean())
            vram_max = float(df_m["peak_vram_mb"].to_numpy().max())

            summary_records.append(
                {
                    "model": model_name,
                    "mode": self.mode,
                    "horizon": self.horizon,
                    "num_seeds": len(self.seeds),
                    "seeds": ",".join(map(str, self.seeds)),
                    "trainable_params": int(df_m["trainable_params"].iloc[0]),
                    "mse_mean": round(mse_mean, 6),
                    "mse_std": round(mse_std, 6),
                    "mse_mean_std": f"{mse_mean:.6f} ± {mse_std:.6f}",
                    "rmse_mean": round(rmse_mean, 6),
                    "rmse_std": round(rmse_std, 6),
                    "rmse_mean_std": f"{rmse_mean:.6f} ± {rmse_std:.6f}",
                    "mae_mean": round(mae_mean, 6),
                    "mae_std": round(mae_std, 6),
                    "mae_mean_std": f"{mae_mean:.6f} ± {mae_std:.6f}",
                    "train_time_mean_sec": round(time_mean, 3),
                    "train_time_std_sec": round(time_std, 3),
                    "peak_vram_mean_mb": round(vram_mean, 2),
                    "peak_vram_max_mb": round(vram_max, 2),
                }
            )

        runs_df = pd.DataFrame(all_run_records)
        summary_df = pd.DataFrame(summary_records)

        runs_csv = (
            self.report_dir / f"global_pretrain_runs_{self.mode}_H{self.horizon}.csv"
        )
        summary_csv = (
            self.report_dir / f"global_pretrain_summary_{self.mode}_H{self.horizon}.csv"
        )
        runs_df.to_csv(runs_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)

        print(f"\nĐã lưu chi tiết từng seed tại: {runs_csv}")
        print(f"Đã lưu tổng hợp mean ± std tại: {summary_csv}")
        return runs_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện Pre-training 4 mô hình PM2.5 đa hạt giống (3 seeds)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "dlinear", "informer", "patchtst"],
        choices=["lstm", "dlinear", "informer", "patchtst"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multivariate",
        choices=["multivariate", "univariate"],
    )
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 1024, 2026],
        help="Danh sách hạt giống ngẫu nhiên (mặc định 3 seeds: 42 1024 2026).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = TrainingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        train_stride=args.train_stride,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        seeds=args.seeds,
    )
    pipeline.run(models=args.models)
