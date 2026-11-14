"""
T0301 — Pipeline tổng thể (Training Loop):
- Cung cấp các hàm lõi dùng chung: build_model, train_one_epoch, evaluate, get_feature_columns.
- Huấn luyện Pre-trained Models trên toàn bộ dữ liệu 12 trạm (an toàn ranh giới chuỗi thời gian).
- Lưu best weights (.pt), Global StandardScaler (.joblib) và log quá trình huấn luyện.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.data_pipeline.preprocess import (
        TARGET_COL,
        chronological_split,
        clean_missing_values,
        encode_features,
        load_pm25,
    )
    from src.data_pipeline.dataset import make_sliding_windows
except ImportError:
    from src.data_pipeline.preprocess import (
        TARGET_COL,
        chronological_split,
        clean_missing_values,
        encode_features,
        load_pm25,
    )

from src.models.dlinear import DLinearModel
from src.models.informer import InformerModel
from src.models.lstm import LSTMModel
from src.models.patchtst import PatchTSTModel


ALL_MODELS: List[Literal["lstm", "dlinear", "informer", "patchtst"]] = [
    "lstm",
    "dlinear",
    "informer",
    "patchtst",
]
NON_FEATURE_COLS = {"no", "year", "month", "day", "hour", "station", "datetime", "wd"}


def set_seed(seed: int = 42) -> None:
    """Cố định random seed cho Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sort_station_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sắp xếp dữ liệu một trạm theo trục thời gian tăng dần."""
    if "datetime" in df.columns:
        return df.sort_values("datetime").reset_index(drop=True)
    return df.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)


def get_feature_columns(
    df: pd.DataFrame,
    mode: Literal["multivariate", "univariate"] = "multivariate",
    target_col: str = TARGET_COL,
) -> List[str]:
    """Xác định danh sách cột đặc trưng đầu vào sau bước encode_features()."""
    if mode == "univariate":
        return [target_col]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in NON_FEATURE_COLS]
    if target_col in feature_cols:
        feature_cols = [target_col] + [c for c in feature_cols if c != target_col]
    return feature_cols


def build_model(
    model_name: Literal["lstm", "dlinear", "informer", "patchtst"],
    lookback: int,
    horizon: int,
    in_features: int,
) -> nn.Module:
    """Khởi tạo kiến trúc mô hình theo tên."""
    name = model_name.strip().lower()
    if name == "lstm":
        return LSTMModel(lookback=lookback, horizon=horizon, in_features=in_features)
    if name == "dlinear":
        return DLinearModel(lookback=lookback, horizon=horizon, in_features=in_features)
    if name == "informer":
        return InformerModel(
            lookback=lookback, horizon=horizon, in_features=in_features
        )
    if name == "patchtst":
        return PatchTSTModel(
            lookback=lookback, horizon=horizon, in_features=in_features
        )
    raise ValueError(f"Mô hình không hợp lệ: '{model_name}'. Chọn trong: {ALL_MODELS}.")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Huấn luyện mô hình qua 1 epoch và trả về loss trung bình."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x).squeeze(-1)  # [B, H, 1] -> [B, H]
        loss = criterion(preds, batch_y)
        loss.backward()
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
    """Đánh giá mô hình trên tập validation (dev) hoặc test."""
    model.eval()
    total_loss = 0.0
    total_se = 0.0
    total_ae = 0.0
    total_elements = 0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        preds = model(batch_x).squeeze(-1)  # [B, H, 1] -> [B, H]
        loss = criterion(preds, batch_y)

        diff = preds - batch_y
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


def prepare_global_12stations_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int = 96,
    horizon: int = 24,
    stride: int = 1,
    batch_size: int = 64,
    scaler_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, DataLoader], StandardScaler, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Chuẩn bị DataLoader toàn cục từ 12 trạm:
    - Chia tuyến tính 60/10/10/20 trong nội bộ từng trạm.
    - Fit StandardScaler duy nhất trên tập Train gộp của 12 trạm (tránh Data Leakage).
    - Tạo sliding window độc lập trên từng trạm rồi gộp lại (tránh cửa sổ xuyên trạm).
    """
    station_splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    train_frames: List[pd.DataFrame] = []

    for station, s_df in df.groupby("station", sort=True):
        s_sorted = sort_station_df(s_df.copy())
        splits = chronological_split(s_sorted)
        station_splits[str(station)] = splits
        train_frames.append(splits["train"][feature_cols])

    scaler = StandardScaler()
    scaler.fit(pd.concat(train_frames, axis=0, ignore_index=True))

    if scaler_path is not None:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

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

            X, y = make_sliding_windows(
                scaled_df,
                feature_cols=feature_cols,
                target_col=TARGET_COL,
                lookback=lookback,
                horizon=horizon,
                stride=stride,
            )
            agg_X[split_name].append(X)
            agg_y[split_name].append(y)

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
        )

    return global_loaders, scaler, station_splits


def train_single_global_model(
    model_name: Literal["lstm", "dlinear", "informer", "patchtst"],
    global_loaders: Dict[str, DataLoader],
    in_features: int,
    lookback: int = 96,
    horizon: int = 24,
    mode: str = "multivariate",
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    checkpoint_dir: str | Path = "outputs/checkpoints/global",
    log_dir: str | Path = "outputs/logs",
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Huấn luyện và đánh giá 1 mô hình trên tập dữ liệu gộp 12 trạm."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / f"{model_name}_{mode}_H{horizon}_pretrained.pt"
    log_path = log_dir / f"pretrain_{model_name}_{mode}_H{horizon}.csv"

    model = build_model(
        model_name=model_name,
        lookback=lookback,
        horizon=horizon,
        in_features=in_features,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_dev_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    print("\n" + "=" * 80)
    print(
        f"[PRE-TRAIN 12 TRẠM] Model={model_name.upper()} | Mode={mode} | "
        f"L={lookback}, H={horizon} | D={in_features} | "
        f"Params={model.count_parameters():,} | Device={device}"
    )
    print("=" * 80)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, global_loaders["train"], criterion, optimizer, device
        )
        train_dev_metrics = evaluate(
            model, global_loaders["train_dev"], criterion, device
        )
        dev_metrics = evaluate(model, global_loaders["dev"], criterion, device)
        scheduler.step(dev_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_dev_loss": train_dev_metrics["loss"],
                "train_dev_rmse": train_dev_metrics["rmse"],
                "train_dev_mae": train_dev_metrics["mae"],
                "dev_loss": dev_metrics["loss"],
                "dev_rmse": dev_metrics["rmse"],
                "dev_mae": dev_metrics["mae"],
            }
        )

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | "
            f"train_dev_rmse={train_dev_metrics['rmse']:.4f} | "
            f"dev_rmse={dev_metrics['rmse']:.4f} | dev_mae={dev_metrics['mae']:.4f}"
        )

        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [INFO] Early stopping kích hoạt tại epoch {epoch}.")
                break

    pd.DataFrame(history).to_csv(log_path, index=False)

    # Đánh giá checkpoint tốt nhất trên cả train_dev, dev và test
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    train_dev_final = evaluate(model, global_loaders["train_dev"], criterion, device)
    dev_final = evaluate(model, global_loaders["dev"], criterion, device)
    test_metrics = evaluate(model, global_loaders["test"], criterion, device)

    print(
        f"[GLOBAL TEST] {model_name.upper()} (best_ep={best_epoch}): "
        f"MSE={test_metrics['mse']:.4f} | "
        f"RMSE={test_metrics['rmse']:.4f} | "
        f"MAE={test_metrics['mae']:.4f}"
    )

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "mode": mode,
        "horizon": horizon,
        "best_epoch": best_epoch,
        "global_train_dev_rmse": round(train_dev_final["rmse"], 6),
        "global_dev_rmse": round(dev_final["rmse"], 6),
        "global_test_mse": round(test_metrics["mse"], 6),
        "global_test_rmse": round(test_metrics["rmse"], 6),
        "global_test_mae": round(test_metrics["mae"], 6),
        "checkpoint": str(ckpt_path),
        "log_file": str(log_path),
    }


def run_pretraining_pipeline(
    data_dir: str | Path = "data/raw/PRSA_Data_20130301-20170228",
    models: Sequence[Literal["lstm", "dlinear", "informer", "patchtst"]] = ALL_MODELS,
    mode: Literal["multivariate", "univariate"] = "multivariate",
    lookback: int = 96,
    horizon: int = 24,
    stride: int = 1,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    output_root: str | Path = "outputs",
    seed: int = 42,
) -> List[Dict[str, object]]:
    """Chạy huấn luyện Pre-training cho danh sách mô hình trên toàn bộ 12 trạm."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = Path(output_root)
    global_ckpt_dir = output_root / "checkpoints" / "global"
    log_dir = output_root / "logs"
    report_dir = output_root / "reports"
    global_ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = global_ckpt_dir / f"scaler_12stations_{mode}.joblib"

    df = load_pm25(data_dir)
    df = clean_missing_values(df, target_col=TARGET_COL)
    df = encode_features(df)

    feature_cols = get_feature_columns(df, mode=mode, target_col=TARGET_COL)
    global_loaders, _, station_splits = prepare_global_12stations_data(
        df=df,
        feature_cols=feature_cols,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        batch_size=batch_size,
        scaler_path=scaler_path,
    )

    print(
        f"[INFO] Sẵn sàng {len(station_splits)} trạm | D={len(feature_cols)} | "
        f"Global Train Windows={len(global_loaders['train'].dataset):,}"
    )

    summary_rows: List[Dict[str, object]] = []
    for model_name in models:
        res = train_single_global_model(
            model_name=model_name,
            global_loaders=global_loaders,
            in_features=len(feature_cols),
            lookback=lookback,
            horizon=horizon,
            mode=mode,
            epochs=epochs,
            lr=lr,
            patience=patience,
            checkpoint_dir=global_ckpt_dir,
            log_dir=log_dir,
            device=device,
        )
        summary_rows.append(res)
        gc.collect()

    summary_csv = report_dir / f"global_pretrain_summary_{mode}_H{horizon}.csv"
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"TỔNG KẾT PRE-TRAINING 12 TRẠM (MODE={mode}, H={horizon})")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\n[SAVED] Scaler toàn cục: {scaler_path}")
    print(f"[SAVED] Bảng kết quả Pre-train: {summary_csv}")

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T0301: Pre-training 4 mô hình trên 12 trạm"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw/PRSA_Data_20130301-20170228"
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument(
        "--mode",
        type=str,
        default="multivariate",
        choices=["multivariate", "univariate"],
    )
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24, choices=[24, 48, 96, 192])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pretraining_pipeline(
        data_dir=args.data_dir,
        models=args.models,
        mode=args.mode,
        lookback=args.lookback,
        horizon=args.horizon,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        output_root=args.output_root,
        seed=args.seed,
    )
