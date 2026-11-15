"""
T0301 — Pipeline huấn luyện toàn cục (Global Pre-training) cho dự báo PM2.5 Bắc Kinh
trên dữ liệu 12 trạm quan trắc.
- Tái sử dụng 100% pipeline tiền xử lý và tạo cửa sổ của WP01
  (src/data_pipeline/preprocess.py & src/data_pipeline/dataset.py).
- Tích hợp RevINModelWrapper + HuberLoss + Warmup/CosineLR + Weight Decay mạnh
  để triệt tiêu lệch phân phối và đảm bảo dev_loss giảm ổn định qua các epoch.
- Hỗ trợ đầy đủ Ablation Studies: multivariate vs. univariate, H in {24, 48, 96, 192},
  và biến thể informer_full (Full Attention O(L^2)).
"""

from __future__ import annotations

import argparse
import gc
import inspect
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data_pipeline.dataset import make_sliding_windows
from src.data_pipeline.preprocess import (
    TARGET_COL,
    chronological_split,
    clean_missing_values,
    encode_features,
    load_pm25,
    scale_splits,
)
from src.models.dlinear import DLinearModel
from src.models.informer import InformerModel
from src.models.lstm import LSTMModel
from src.models.patchtst import PatchTSTModel

DEFAULT_MODELS: List[str] = ["lstm", "dlinear", "informer", "patchtst"]
ALL_SUPPORTED_MODELS: List[str] = [
    "lstm",
    "dlinear",
    "informer",
    "patchtst",
    "informer_full",
]

CANONICAL_MULTIVARIATE_COLS: List[str] = [
    "pm2.5",
    "pm10",
    "so2",
    "no2",
    "co",
    "o3",
    "temp",
    "pres",
    "dewp",
    "rain",
    "wspm",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


class RevINModelWrapper(nn.Module):
    """
    Bọc kiến trúc dự báo với Reversible Instance Normalization (RevIN):
    - Chuẩn hóa các kênh vật lý/ô nhiễm liên tục theo từng cửa sổ (B, L, C_cont)
      để loại bỏ dịch chuyển phân phối giữa các năm (2013-2015 vs 2016-2017).
    - Giữ nguyên các kênh mã hóa chu kỳ thời gian (hour_sin/cos, month_sin/cos).
    - Giải chuẩn hóa đầu ra dự báo (B, H) theo thống kê cục bộ của kênh mục tiêu (index=0).
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_continuous_features: Optional[int] = None,
        target_idx: int = 0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.in_features = in_features
        if num_continuous_features is None:
            # Trong bộ 15 đặc trưng chuẩn, 11 cột đầu là biến khí tượng/ô nhiễm liên tục,
            # 4 cột cuối là đặc trưng chu kỳ sin/cos.
            self.num_cont = 11 if in_features >= 15 else in_features
        else:
            self.num_cont = min(num_continuous_features, in_features)

        self.target_idx = target_idx
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_cont))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_cont))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x_cont = x[:, :, : self.num_cont]
        mean = x_cont.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x_cont, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

        x_cont_norm = ((x_cont - mean) / stdev) * self.affine_weight + self.affine_bias

        if self.num_cont < x.size(-1):
            x_in = torch.cat([x_cont_norm, x[:, :, self.num_cont :]], dim=-1)
        else:
            x_in = x_cont_norm

        y_norm = self.backbone(x_in).squeeze(-1)  # (B, H)

        idx = self.target_idx
        y_unscaled = (y_norm - self.affine_bias[:, :, idx]) / (
            self.affine_weight[:, :, idx] + self.eps
        )
        y_out = y_unscaled * stdev[:, :, idx] + mean[:, :, idx]
        return y_out


def set_seed(seed: int) -> None:
    """Cố định hạt giống ngẫu nhiên cho toàn bộ thư viện để đảm bảo tái lập thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_data_dir(data_dir: str | Path) -> Path:
    """Tự động định vị thư mục chứa các file CSV PRSA."""
    path = Path(data_dir)
    if path.is_file() or any(path.glob("*.csv")):
        return path
    nested = path / "PRSA_Data_20130301-20170228"
    if nested.exists() and any(nested.glob("*.csv")):
        return nested
    return path


def get_feature_columns(
    df: pd.DataFrame,
    mode: str = "multivariate",
    target_col: str = TARGET_COL,
) -> List[str]:
    """
    Xác định danh sách cột đặc trưng từ DataFrame đã qua WP01 encode_features().
    Luôn đặt target_col ('pm2.5') ở vị trí đầu tiên (index=0).
    """
    if mode == "univariate":
        if target_col not in df.columns:
            raise ValueError(
                f"Không tìm thấy cột mục tiêu '{target_col}' trong dữ liệu."
            )
        return [target_col]

    if mode != "multivariate":
        raise ValueError(f"Chế độ feature_mode không hợp lệ: {mode}")

    if all(col in df.columns for col in CANONICAL_MULTIVARIATE_COLS):
        return list(CANONICAL_MULTIVARIATE_COLS)

    excluded = {
        "no",
        "year",
        "month",
        "day",
        "hour",
        "datetime",
        "station",
        "wd",
        "cbwd",
    }
    numeric_cols = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if target_col in numeric_cols:
        numeric_cols = [target_col] + [c for c in numeric_cols if c != target_col]
    return numeric_cols


def prepare_global_12stations_data(
    data_dir: str | Path,
    mode: str = "multivariate",
    lookback: int = 96,
    horizon: int = 24,
    batch_size: int = 512,
    train_stride: int = 6,
    scaler_save_path: Optional[Path] = None,
) -> Tuple[Dict[str, DataLoader], StandardScaler, List[str]]:
    """
    Gọi trực tiếp pipeline của WP01:
    load_pm25 -> clean_missing_values -> encode_features -> chronological_split
    -> scale_splits -> make_sliding_windows.
    """
    resolved_dir = resolve_data_dir(data_dir)
    df = load_pm25(resolved_dir)
    df = clean_missing_values(df, target_col=TARGET_COL)
    df = encode_features(df)

    feature_cols = get_feature_columns(df, mode=mode, target_col=TARGET_COL)
    splits = chronological_split(df)
    scaled_splits, scaler = scale_splits(
        splits, feature_cols=feature_cols, scaler_path=scaler_save_path
    )

    use_pin_memory = torch.cuda.is_available()
    num_workers = 4 if use_pin_memory else 0
    global_loaders: Dict[str, DataLoader] = {}

    for split_name in ["train", "train_dev", "dev", "test"]:
        current_stride = train_stride if split_name == "train" else 1
        X_all, y_all = make_sliding_windows(
            scaled_splits[split_name],
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            lookback=lookback,
            horizon=horizon,
            stride=current_stride,
        )
        ds = TensorDataset(torch.from_numpy(X_all), torch.from_numpy(y_all))
        global_loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            drop_last=False,
            pin_memory=use_pin_memory,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    return global_loaders, scaler, feature_cols


def _instantiate_with_regularization(
    cls: type[nn.Module],
    base_kwargs: Dict[str, object],
    dropout: float = 0.3,
) -> nn.Module:
    """Truyền an toàn các tham số điều chuẩn (dropout) nếu kiến trúc hỗ trợ."""
    sig = inspect.signature(cls.__init__)
    kwargs = dict(base_kwargs)
    if "dropout" in sig.parameters:
        kwargs["dropout"] = dropout
    return cls(**kwargs)


def build_model(
    model_name: str,
    lookback: int,
    horizon: int,
    in_features: int,
    device: torch.device,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Khởi tạo kiến trúc mô hình và bọc RevINModelWrapper cho các kiến trúc nhạy cảm
    với dịch chuyển phân phối (LSTM, Informer) để đảm bảo hội tụ ổn định.
    """
    name = model_name.strip().lower()
    base_kwargs: Dict[str, object] = {
        "lookback": lookback,
        "horizon": horizon,
        "in_features": in_features,
    }

    if name == "lstm":
        backbone = _instantiate_with_regularization(
            LSTMModel, base_kwargs, dropout=dropout
        )
        model: nn.Module = RevINModelWrapper(
            backbone=backbone, in_features=in_features, target_idx=0
        )
    elif name == "dlinear":
        # DLinear đã có phân rã trung bình trượt cục bộ (Trend/Seasonal)
        model = DLinearModel(**base_kwargs)
    elif name == "informer":
        backbone = _instantiate_with_regularization(
            InformerModel, base_kwargs, dropout=dropout
        )
        model = RevINModelWrapper(
            backbone=backbone, in_features=in_features, target_idx=0
        )
    elif name in {"informer_full", "informer_fullattention"}:
        sig = inspect.signature(InformerModel.__init__)
        full_kwargs = dict(base_kwargs)
        if "dropout" in sig.parameters:
            full_kwargs["dropout"] = dropout
        if "attn" in sig.parameters:
            full_kwargs["attn"] = "full"
        elif "attention_type" in sig.parameters:
            full_kwargs["attention_type"] = "full"
        elif "use_probsparse" in sig.parameters:
            full_kwargs["use_probsparse"] = False
        elif "factor" in sig.parameters:
            full_kwargs["factor"] = lookback
        backbone = InformerModel(**full_kwargs)
        model = RevINModelWrapper(
            backbone=backbone, in_features=in_features, target_idx=0
        )
    elif name == "patchtst":
        # PatchTSTModel đã tích hợp sẵn RevIN + FlashAttention
        model = _instantiate_with_regularization(
            PatchTSTModel, base_kwargs, dropout=dropout
        )
    else:
        raise ValueError(f"Kiến trúc mô hình không được hỗ trợ: {model_name}")

    return model.to(device)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 2,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Lịch tốc độ học tăng ấm tuyến tính (Warmup) rồi giảm mượt theo hàm Cosine."""
    warmup = max(1, min(warmup_epochs, max(1, total_epochs - 1)))

    def lr_lambda(epoch_idx: int) -> float:
        # epoch_idx bắt đầu từ 0
        if epoch_idx < warmup:
            return float(epoch_idx + 1) / float(warmup)
        progress = float(epoch_idx - warmup) / float(max(1, total_epochs - warmup))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """Huấn luyện 1 epoch với Mixed Precision (AMP) và Gradient Clipping."""
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
        batch_size: int = 512,
        train_stride: int = 6,
        lr: float = 5e-5,
        weight_decay: float = 1e-2,
        dropout: float = 0.3,
        epochs: int = 20,
        patience: int = 6,
        seeds: Optional[List[int]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.mode = mode
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.train_stride = train_stride
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
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
    ) -> Dict[str, object]:
        """Huấn luyện một mô hình trên một hạt giống cụ thể, đo thời gian và VRAM."""
        set_seed(seed)
        model = build_model(
            model_name=model_name,
            lookback=self.lookback,
            horizon=self.horizon,
            in_features=in_features,
            device=self.device,
            dropout=self.dropout,
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # DLinear dùng lr nhỉnh hơn vì mạng tuyến tính ít tham số
        effective_lr = (
            max(self.lr, 2e-4) if model_name.lower() == "dlinear" else self.lr
        )

        train_criterion = nn.HuberLoss(delta=1.0)
        eval_criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=effective_lr, weight_decay=self.weight_decay
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer=optimizer, total_epochs=self.epochs, warmup_epochs=2
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

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        t_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(
                model=model,
                dataloader=loaders["train"],
                criterion=train_criterion,
                optimizer=optimizer,
                device=self.device,
                scaler=grad_scaler,
            )
            train_dev_metrics = evaluate(
                model, loaders["train_dev"], eval_criterion, self.device
            )
            dev_metrics = evaluate(model, loaders["dev"], eval_criterion, self.device)
            dev_loss = dev_metrics["loss"]
            scheduler.step()

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_huber_loss": round(train_loss, 6),
                    "train_dev_mse": round(train_dev_metrics["mse"], 6),
                    "train_dev_mae": round(train_dev_metrics["mae"], 6),
                    "dev_mse": round(dev_loss, 6),
                    "dev_rmse": round(dev_metrics["rmse"], 6),
                    "dev_mae": round(dev_metrics["mae"], 6),
                    "lr": current_lr,
                }
            )

            print(
                f"[{model_name.upper()} | seed={seed}] Epoch {epoch:02d}/{self.epochs:02d} "
                f"| train_loss={train_loss:.5f} | train_dev_mse={train_dev_metrics['mse']:.5f} "
                f"| dev_mse={dev_loss:.5f} | dev_mae={dev_metrics['mae']:.5f} | lr={current_lr:.2e}"
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
                        "dropout": self.dropout,
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

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
        else:
            peak_vram_mb = 0.0
        train_time_sec = time.perf_counter() - t_start

        pd.DataFrame(history_rows).to_csv(log_csv_path, index=False)

        ckpt = torch.load(seed_ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        train_dev_final = evaluate(
            model, loaders["train_dev"], eval_criterion, self.device
        )
        test_metrics = evaluate(model, loaders["test"], eval_criterion, self.device)

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

    def run(self, models: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            f"| L={self.lookback}, H={self.horizon} | Stride={self.train_stride} | Seeds: {self.seeds}"
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

                if float(rec["best_dev_loss"]) < best_overall_dev_loss:
                    best_overall_dev_loss = float(rec["best_dev_loss"])
                    best_overall_ckpt_src = Path(str(rec["checkpoint_path"]))

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
        description="T0301: Huấn luyện Pre-training đa hạt giống (3 seeds) dùng WP01 pipeline."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=ALL_SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multivariate",
        choices=["multivariate", "univariate"],
    )
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24, choices=[24, 48, 96, 192])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-stride", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
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
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        seeds=args.seeds,
    )
    pipeline.run(models=args.models)
