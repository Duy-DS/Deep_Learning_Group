"""
T0304 — Fine-Tuning mô hình Pre-trained cho từng trạm quan trắc PM2.5 Bắc Kinh:
- Nạp trọng số nền (từ T0301) và Global StandardScaler.
- Đánh giá hiệu năng Trước (Zero-shot) và Sau khi Fine-tuning trên từng trạm.
- Lưu 48 bộ trọng số riêng (4 mô hình x 12 trạm) cùng bảng báo cáo mức cải thiện.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Tương thích linh hoạt các cấu trúc thư mục trong repository
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
    from src.data_pipeline.dataset import make_sliding_windows
# Kế thừa trực tiếp từ T0301 (training_loop.py)
from training_loop import (
    ALL_MODELS,
    build_model,
    evaluate,
    get_feature_columns,
    set_seed,
    sort_station_df,
    train_one_epoch,
)


def load_global_scaler(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: str | Path,
) -> StandardScaler:
    """
    Nạp Global StandardScaler đã lưu ở bước Pre-training (T0301).
    Nếu chưa tồn tại, tự động khớp trên tập train (60%) gộp của 12 trạm.
    """
    scaler_file = Path(scaler_path)
    if scaler_file.exists():
        return joblib.load(scaler_file)

    train_frames = []
    for _, s_df in df.groupby("station", sort=True):
        splits = chronological_split(sort_station_df(s_df.copy()))
        train_frames.append(splits["train"][feature_cols])

    scaler = StandardScaler()
    scaler.fit(pd.concat(train_frames, axis=0, ignore_index=True))
    scaler_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_file)
    return scaler


def build_single_station_loaders(
    station_df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    lookback: int = 96,
    horizon: int = 24,
    stride: int = 1,
    batch_size: int = 64,
) -> Dict[str, DataLoader]:
    """Tạo bộ DataLoader (train, train_dev, dev, test) cho riêng 1 trạm."""
    sorted_df = sort_station_df(station_df.copy())
    splits = chronological_split(sorted_df)

    loaders: Dict[str, DataLoader] = {}
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
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            drop_last=False,
        )
    return loaders


def configure_trainable_layers(
    model: nn.Module,
    model_name: str,
    freeze_backbone: bool = False,
) -> None:
    """
    Cấu hình các lớp được phép cập nhật khi Fine-tuning:
    - freeze_backbone=False: Tinh chỉnh toàn bộ mạng với learning rate nhỏ.
    - freeze_backbone=True : Đóng băng encoder/backbone, chỉ tinh chỉnh lớp dự báo đầu ra.
    """
    for param in model.parameters():
        param.requires_grad = True

    if not freeze_backbone:
        return

    for param in model.parameters():
        param.requires_grad = False

    head_keywords = {
        "lstm": ("fc", "head", "proj", "linear"),
        "dlinear": ("linear_seasonal", "linear_trend", "linear"),
        "informer": ("decoder", "projection", "proj", "head"),
        "patchtst": ("head", "projection", "proj", "linear"),
    }.get(model_name.strip().lower(), ("head", "fc", "projection", "decoder"))

    unfrozen = 0
    for param_name, param in model.named_parameters():
        if any(key in param_name.lower() for key in head_keywords):
            param.requires_grad = True
            unfrozen += 1

    if unfrozen == 0:
        for param in model.parameters():
            param.requires_grad = True


def finetune_single_station(
    model_name: Literal["lstm", "dlinear", "informer", "patchtst"],
    station: str,
    station_loaders: Dict[str, DataLoader],
    pretrained_ckpt: str | Path,
    in_features: int,
    lookback: int = 96,
    horizon: int = 24,
    mode: str = "multivariate",
    epochs: int = 15,
    lr: float = 1e-4,
    patience: int = 4,
    freeze_backbone: bool = False,
    station_ckpt_dir: str | Path = "outputs/checkpoints/stations",
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Nạp trọng số nền của 1 mô hình, đo Zero-shot (Trước), Fine-tune trên 1 trạm
    và đo lại (Sau) để lưu trọng số trạm cùng mức cải thiện.
    """
    pretrained_ckpt = Path(pretrained_ckpt)
    if not pretrained_ckpt.exists():
        raise FileNotFoundError(
            f"Không tìm thấy trọng số Pre-trained tại: {pretrained_ckpt}. "
            "Hãy chạy training_loop.py (T0301) trước."
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(station_ckpt_dir) / station
    out_dir.mkdir(parents=True, exist_ok=True)
    ft_ckpt_path = out_dir / f"{model_name}_{mode}_H{horizon}_finetuned.pt"

    model = build_model(
        model_name=model_name,
        lookback=lookback,
        horizon=horizon,
        in_features=in_features,
    ).to(device)
    model.load_state_dict(torch.load(pretrained_ckpt, map_location=device))

    criterion = nn.MSELoss()
    before_train_dev = evaluate(model, station_loaders["train_dev"], criterion, device)
    before_dev = evaluate(model, station_loaders["dev"], criterion, device)
    before_test = evaluate(model, station_loaders["test"], criterion, device)

    best_dev_loss = before_dev["loss"]
    best_epoch = 0
    torch.save(model.state_dict(), ft_ckpt_path)

    # 2. Thiết lập lớp huấn luyện & Optimizer
    configure_trainable_layers(
        model, model_name=model_name, freeze_backbone=freeze_backbone
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        _ = train_one_epoch(
            model, station_loaders["train"], criterion, optimizer, device
        )
        dev_metrics = evaluate(model, station_loaders["dev"], criterion, device)
        scheduler.step(dev_metrics["loss"])

        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ft_ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(torch.load(ft_ckpt_path, map_location=device))
    after_train_dev = evaluate(model, station_loaders["train_dev"], criterion, device)
    after_dev = evaluate(model, station_loaders["dev"], criterion, device)
    after_test = evaluate(model, station_loaders["test"], criterion, device)

    mse_gain_pct = (
        ((before_test["mse"] - after_test["mse"]) / before_test["mse"]) * 100.0
        if before_test["mse"] > 0
        else 0.0
    )
    rmse_gain_pct = (
        ((before_test["rmse"] - after_test["rmse"]) / before_test["rmse"]) * 100.0
        if before_test["rmse"] > 0
        else 0.0
    )
    mae_gain_pct = (
        ((before_test["mae"] - after_test["mae"]) / before_test["mae"]) * 100.0
        if before_test["mae"] > 0
        else 0.0
    )

    print(
        f"  [{model_name:<8} | {station:<14}] "
        f"MSE: {before_test['mse']:.4f} -> {after_test['mse']:.4f} ({mse_gain_pct:+.2f}%) | "
        f"RMSE: {before_test['rmse']:.4f} -> {after_test['rmse']:.4f} ({rmse_gain_pct:+.2f}%) | "
        f"MAE: {before_test['mae']:.4f} -> {after_test['mae']:.4f} ({mae_gain_pct:+.2f}%) "
        f"[best_ep={best_epoch}]"
    )

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "station": station,
        "mode": mode,
        "horizon": horizon,
        "freeze_backbone": freeze_backbone,
        "best_epoch": best_epoch,
        "before_train_dev_rmse": round(before_train_dev["rmse"], 6),
        "after_train_dev_rmse": round(after_train_dev["rmse"], 6),
        "before_dev_rmse": round(before_dev["rmse"], 6),
        "after_dev_rmse": round(after_dev["rmse"], 6),
        "before_mse": round(before_test["mse"], 6),
        "after_mse": round(after_test["mse"], 6),
        "mse_improvement_pct": round(mse_gain_pct, 3),
        "before_rmse": round(before_test["rmse"], 6),
        "after_rmse": round(after_test["rmse"], 6),
        "rmse_improvement_pct": round(rmse_gain_pct, 3),
        "before_mae": round(before_test["mae"], 6),
        "after_mae": round(after_test["mae"], 6),
        "mae_improvement_pct": round(mae_gain_pct, 3),
        "improved": bool(after_test["mse"] < before_test["mse"]),
        "checkpoint": str(ft_ckpt_path),
    }


def run_fine_tuning_pipeline(
    data_dir: str | Path = "data/raw/PRSA_Data_20130301-20170228",
    models: Sequence[Literal["lstm", "dlinear", "informer", "patchtst"]] = ALL_MODELS,
    stations: Optional[Sequence[str]] = None,
    mode: Literal["multivariate", "univariate"] = "multivariate",
    lookback: int = 96,
    horizon: int = 24,
    stride: int = 1,
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-4,
    patience: int = 4,
    freeze_backbone: bool = False,
    output_root: str | Path = "outputs",
    seed: int = 42,
) -> Dict[str, object]:
    """
    Chạy Fine-tuning toàn bộ các mô hình (mặc định 4 mô hình) trên từng trạm (mặc định 12 trạm)
    và xuất bảng đối chứng Trước vs. Sau.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = Path(output_root)
    global_ckpt_dir = output_root / "checkpoints" / "global"
    station_ckpt_dir = output_root / "checkpoints" / "stations"
    report_dir = output_root / "reports"
    station_ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = global_ckpt_dir / f"scaler_12stations_{mode}.joblib"

    df = load_pm25(data_dir)
    df = clean_missing_values(df, target_col=TARGET_COL)
    df = encode_features(df)

    feature_cols = get_feature_columns(df, mode=mode, target_col=TARGET_COL)
    in_features = len(feature_cols)
    scaler = load_global_scaler(df, feature_cols=feature_cols, scaler_path=scaler_path)

    all_stations = sorted(df["station"].dropna().unique().tolist())
    target_stations = list(stations) if stations is not None else all_stations

    per_station_loaders: Dict[str, Dict[str, DataLoader]] = {
        st: build_single_station_loaders(
            station_df=df[df["station"] == st],
            feature_cols=feature_cols,
            scaler=scaler,
            lookback=lookback,
            horizon=horizon,
            stride=stride,
            batch_size=batch_size,
        )
        for st in target_stations
    }

    all_station_records: List[Dict[str, object]] = []
    model_summary_records: List[Dict[str, object]] = []

    for model_name in models:
        pretrained_ckpt = (
            global_ckpt_dir / f"{model_name}_{mode}_H{horizon}_pretrained.pt"
        )
        print("\n" + "=" * 80)
        print(
            f"[FINE-TUNE TỪNG TRẠM] Model={model_name.upper()} | Mode={mode} | "
            f"H={horizon} | Số trạm={len(target_stations)}"
        )
        print("=" * 80)

        model_records: List[Dict[str, object]] = []
        for st in target_stations:
            rec = finetune_single_station(
                model_name=model_name,
                station=st,
                station_loaders=per_station_loaders[st],
                pretrained_ckpt=pretrained_ckpt,
                in_features=in_features,
                lookback=lookback,
                horizon=horizon,
                mode=mode,
                epochs=epochs,
                lr=lr,
                patience=patience,
                freeze_backbone=freeze_backbone,
                station_ckpt_dir=station_ckpt_dir,
                device=device,
            )
            model_records.append(rec)
            all_station_records.append(rec)

        df_m = pd.DataFrame(model_records)
        win_count = int(df_m["improved"].to_numpy().sum())
        model_summary_records.append(
            {
                "model": model_name,
                "mode": mode,
                "horizon": horizon,
                "mean_before_mse": round(
                    float(df_m["before_mse"].to_numpy().mean()), 6
                ),
                "mean_after_mse": round(float(df_m["after_mse"].to_numpy().mean()), 6),
                "avg_mse_gain_pct": round(
                    float(df_m["mse_improvement_pct"].to_numpy().mean()), 3
                ),
                "mean_before_rmse": round(
                    float(df_m["before_rmse"].to_numpy().mean()), 6
                ),
                "mean_after_rmse": round(
                    float(df_m["after_rmse"].to_numpy().mean()), 6
                ),
                "avg_rmse_gain_pct": round(
                    float(df_m["rmse_improvement_pct"].to_numpy().mean()), 3
                ),
                "mean_before_mae": round(
                    float(df_m["before_mae"].to_numpy().mean()), 6
                ),
                "mean_after_mae": round(float(df_m["after_mae"].to_numpy().mean()), 6),
                "avg_mae_gain_pct": round(
                    float(df_m["mae_improvement_pct"].to_numpy().mean()), 3
                ),
                "stations_improved": f"{win_count}/{len(target_stations)}",
            }
        )
        gc.collect()

    detail_csv = report_dir / f"before_vs_after_by_station_{mode}_H{horizon}.csv"
    summary_csv = report_dir / f"before_vs_after_by_model_{mode}_H{horizon}.csv"
    summary_json = report_dir / f"before_vs_after_full_{mode}_H{horizon}.json"

    pd.DataFrame(all_station_records).to_csv(
        detail_csv, index=False, encoding="utf-8-sig"
    )
    summary_df = pd.DataFrame(model_summary_records)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "mode": mode,
                    "lookback": lookback,
                    "horizon": horizon,
                    "epochs": epochs,
                    "lr": lr,
                    "freeze_backbone": freeze_backbone,
                    "seed": seed,
                },
                "model_summary": model_summary_records,
                "station_details": all_station_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 90)
    print(f"TỔNG HỢP TRƯỚC VS. SAU FINE-TUNING (MODE={mode}, H={horizon})")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print(f"\n[SAVED] Chi tiết từng trạm (48 dòng): {detail_csv}")
    print(f"[SAVED] Tổng hợp theo mô hình (4 dòng): {summary_csv}")

    return {
        "model_summary": model_summary_records,
        "station_details": all_station_records,
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T0304: Fine-tuning mô hình theo từng trạm"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw/PRSA_Data_20130301-20170228"
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Danh sách trạm (trống = cả 12 trạm)",
    )
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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fine_tuning_pipeline(
        data_dir=args.data_dir,
        models=args.models,
        stations=args.stations,
        mode=args.mode,
        lookback=args.lookback,
        horizon=args.horizon,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        freeze_backbone=args.freeze_backbone,
        output_root=args.output_root,
        seed=args.seed,
    )
