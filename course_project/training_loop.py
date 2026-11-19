"""
T0301 — Pipeline huấn luyện toàn cục (Global Pre-training) cho dự báo PM2.5 Bắc Kinh
trên dữ liệu 12 trạm quan trắc, tích hợp chuẩn đánh giá `utils_metrics.py` và
mô hình Baseline từ WP02 (`train_baseline.py`, `src/models/*`).
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data_pipeline.dataset import make_sliding_windows
from src.data_pipeline.preprocess import TARGET_COL, scale_splits
from src.models.dlinear import DLinearModel
from src.models.informer import InformerModel
from src.models.lstm import LSTMModel
from src.models.patchtst import PatchTSTModel
from train_baseline import BaselineTrainer
from utils_metrics import ExperimentMetricsContract, MetricsCalculator

ALL_SPLITS: Tuple[str, ...] = ("train", "train_dev", "dev", "test")
DEFAULT_MODELS: List[str] = ["lstm", "dlinear", "informer", "patchtst"]
ALL_SUPPORTED_MODELS: List[str] = [
    "lstm",
    "dlinear",
    "informer",
    "patchtst",
    "informer_full",
]

ContractModelLiteral = Literal[
    "LSTM", "DLinear", "Informer", "Informer_NoProbSparse", "PatchTST"
]

CONTRACT_MODEL_NAME_MAP: Dict[str, ContractModelLiteral] = {
    "lstm": "LSTM",
    "dlinear": "DLinear",
    "informer": "Informer",
    "informer_full": "Informer_NoProbSparse",
    "informer_fullattention": "Informer_NoProbSparse",
    "informer_noprobsparse": "Informer_NoProbSparse",
    "patchtst": "PatchTST",
}

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


def to_contract_model_name(model_name: str) -> ContractModelLiteral:
    """Ánh xạ tên mô hình nội bộ WP03 sang chuẩn Literal của ExperimentMetricsContract."""
    key = model_name.strip().lower()
    if key not in CONTRACT_MODEL_NAME_MAP:
        raise ValueError(
            f"Tên mô hình '{model_name}' chưa được ánh xạ trong ExperimentMetricsContract."
        )
    return CONTRACT_MODEL_NAME_MAP[key]


class RevINModelWrapper(nn.Module):
    """
    Bọc kiến trúc dự báo với Reversible Instance Normalization (RevIN):
    - Chuẩn hóa các kênh vật lý/ô nhiễm liên tục theo từng cửa sổ (B, L, C_cont)
      với chặn dưới stdev >= min_std để tránh khuếch đại nhiễu trên kênh thưa (rain).
    - Giữ nguyên các kênh mã hóa chu kỳ thời gian và One-Hot hướng gió.
    - Giải chuẩn hóa đầu ra dự báo (B, H) theo thống kê cục bộ của kênh mục tiêu.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_continuous_features: Optional[int] = None,
        target_idx: int = 0,
        eps: float = 1e-5,
        min_std: float = 0.1,
        affine: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.in_features = in_features
        if num_continuous_features is None:
            self.num_cont = 11 if in_features >= 15 else in_features
        else:
            self.num_cont = min(num_continuous_features, in_features)

        self.target_idx = target_idx
        self.eps = eps
        self.min_std = min_std
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_cont))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_cont))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cont = x[:, :, : self.num_cont]
        mean = x_cont.mean(dim=1, keepdim=True).detach()
        stdev = (
            torch.sqrt(
                torch.var(x_cont, dim=1, keepdim=True, unbiased=False) + self.eps
            )
            .clamp(min=self.min_std)
            .detach()
        )

        x_cont_norm = (x_cont - mean) / stdev
        if self.affine:
            x_cont_norm = x_cont_norm * self.affine_weight + self.affine_bias

        if self.num_cont < x.size(-1):
            x_in = torch.cat([x_cont_norm, x[:, :, self.num_cont :]], dim=-1)
        else:
            x_in = x_cont_norm

        y_norm = self.backbone(x_in).squeeze(-1)

        idx = self.target_idx
        if self.affine:
            weight = self.affine_weight[:, :, idx]
            safe_weight = torch.where(
                weight >= 0,
                weight.clamp(min=self.eps),
                weight.clamp(max=-self.eps),
            )
            y_unscaled = (y_norm - self.affine_bias[:, :, idx]) / safe_weight
        else:
            y_unscaled = y_norm

        y_out = y_unscaled * stdev[:, :, idx]
        return y_out


class ResidualAnchorWrapper(nn.Module):
    """Neo dự báo vào giá trị PM2.5 giờ cuối cùng của cửa sổ và học phần dư delta."""

    def __init__(
        self,
        model: nn.Module,
        horizon: int,
        in_features: int,
        target_idx: int = 0,
        input_noise_std: float = 0.05,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_idx = target_idx
        self.num_cont = 11 if in_features >= 15 else in_features
        self.input_noise_std = input_noise_std
        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.station_bias = nn.Parameter(torch.zeros(1, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_val = x[:, -1:, self.target_idx]  # (B, 1)
        if self.training and self.input_noise_std > 0.0:
            noise = torch.randn_like(x[:, :, : self.num_cont]) * self.input_noise_std
            x_in = torch.cat(
                [x[:, :, : self.num_cont] + noise, x[:, :, self.num_cont :]], dim=-1
            )
        else:
            x_in = x

        delta = self.model(x_in).squeeze(-1)  # (B, H)
        return last_val + self.res_scale * delta + self.station_bias


def set_seed(seed: int) -> None:
    """Cố định hạt giống ngẫu nhiên cho toàn bộ thư viện để đảm bảo tái lập thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def apply_yaml_config(parser: argparse.ArgumentParser, sections: list[str]) -> None:
    """Nạp tham số mặc định từ tệp cấu hình YAML nếu tồn tại."""
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    pre_args, _ = parser.parse_known_args()
    cfg_path = Path(pre_args.config)
    if not cfg_path.exists():
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    merged: dict = {}
    for sec in sections:
        merged.update(cfg.get(sec, {}))

    valid_dests = {action.dest for action in parser._actions}
    parser.set_defaults(**{k: v for k, v in merged.items() if k in valid_dests})


def build_criterion(loss_fn: str = "MSELoss", huber_delta: float = 2.5) -> nn.Module:
    """Khởi tạo hàm mất mát đồng nhất với tiêu chí đánh giá."""
    name = loss_fn.strip().lower()
    if name in {"mse", "mseloss", "l2"}:
        return nn.MSELoss()
    if name in {"huber", "huberloss", "smoothl1", "smoothl1loss"}:
        return nn.HuberLoss(delta=huber_delta)
    if name in {"mae", "l1", "l1loss"}:
        return nn.L1Loss()
    raise ValueError(f"Hàm mất mát không được hỗ trợ: {loss_fn}")


def get_feature_columns(
    df: pd.DataFrame,
    mode: str = "multivariate",
    target_col: str = TARGET_COL,
) -> List[str]:
    """Lấy danh sách đặc trưng đã mã hóa của WP01 (15 biến cơ sở + các cột One-Hot hướng gió wd_*)."""
    if mode == "univariate":
        if target_col not in df.columns:
            raise ValueError(
                f"Không tìm thấy cột mục tiêu '{target_col}' trong dữ liệu."
            )
        return [target_col]

    if mode != "multivariate":
        raise ValueError(f"Chế độ feature_mode không hợp lệ: {mode}")

    base_cols = [c for c in CANONICAL_MULTIVARIATE_COLS if c in df.columns]
    wd_cols = sorted(
        c
        for c in df.columns
        if c.startswith(("wd_", "cbwd_")) and pd.api.types.is_numeric_dtype(df[c])
    )
    if base_cols:
        return base_cols + [c for c in wd_cols if c not in base_cols]

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


def load_wp01_splits(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Nạp đầy đủ cả 4 tập (train, train_dev, dev, test) đã chia sẵn bởi WP01."""
    base_dir = Path(data_dir)
    split_patterns = {
        "train": ["train.csv", "*_train.csv", "train_*.csv"],
        "train_dev": ["train_dev.csv", "*_train_dev.csv", "train_dev_*.csv"],
        "dev": ["dev.csv", "val.csv", "*_dev.csv", "*_val.csv"],
        "test": ["test.csv", "*_test.csv", "test_*.csv"],
    }
    splits: Dict[str, pd.DataFrame] = {}
    for split_name in ALL_SPLITS:
        patterns = split_patterns[split_name]
        matched: Optional[Path] = None
        for pat in patterns:
            candidates = sorted(base_dir.glob(pat))
            if split_name == "train":
                candidates = [p for p in candidates if "train_dev" not in p.name]
            if candidates:
                matched = candidates[0]
                break
        if matched is None:
            raise FileNotFoundError(
                f"Không tìm thấy tệp '{split_name}' của WP01 trong thư mục: {base_dir}"
            )
        df_part = pd.read_csv(matched)
        if df_part.empty:
            raise ValueError(f"Tệp '{matched}' cho tập '{split_name}' đang rỗng.")
        splits[split_name] = df_part
    return splits


def load_and_prepare_wp01_splits(
    data_dir: str | Path,
    mode: str = "multivariate",
    scaler_save_path: Optional[Path] = None,
) -> Tuple[Dict[str, pd.DataFrame], StandardScaler, List[str]]:
    """
    Nạp đầy đủ 4 tập (train, train_dev, dev, test) từ WP01 và đảm bảo không Double-Scaling.
    """
    splits = load_wp01_splits(data_dir)
    feature_cols = get_feature_columns(
        splits["train"], mode=mode, target_col=TARGET_COL
    )

    train_target = splits["train"][TARGET_COL].to_numpy(dtype=np.float64)
    already_scaled = (abs(float(np.nanmean(train_target))) < 0.05) and (
        0.95 < float(np.nanstd(train_target)) < 1.05
    )
    wp01_scalers = sorted(Path(data_dir).glob("*scaler*.joblib")) + sorted(
        Path(data_dir).glob("*scaler*.pkl")
    )

    if already_scaled:
        scaled_splits = splits
        if scaler_save_path is not None and scaler_save_path.exists():
            scaler: StandardScaler = joblib.load(scaler_save_path)
        elif wp01_scalers:
            scaler = joblib.load(wp01_scalers[0])
            if scaler_save_path is not None:
                scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(scaler, scaler_save_path)
        else:
            scaler = StandardScaler()
        return scaled_splits, scaler, feature_cols

    if scaler_save_path is not None and scaler_save_path.exists():
        scaler = joblib.load(scaler_save_path)
        if (
            hasattr(scaler, "feature_names_in_")
            and list(scaler.feature_names_in_) != feature_cols
        ):
            scaled_splits, scaler = scale_splits(
                splits, feature_cols=feature_cols, scaler_path=scaler_save_path
            )
        else:
            scaled_splits = {}
            for split_name in ALL_SPLITS:
                part = splits[split_name].copy()
                part[feature_cols] = scaler.transform(part[feature_cols])
                scaled_splits[split_name] = part
    else:
        scaled_splits, scaler = scale_splits(
            splits, feature_cols=feature_cols, scaler_path=scaler_save_path
        )

    return scaled_splits, scaler, feature_cols


def prepare_global_12stations_data(
    data_dir: str | Path,
    mode: str = "multivariate",
    lookback: int = 96,
    horizon: int = 24,
    batch_size: int = 512,
    train_stride: int = 12,
    scaler_save_path: Optional[Path] = None,
) -> Tuple[Dict[str, DataLoader], StandardScaler, List[str]]:
    """Tạo đầy đủ 4 DataLoader (train, train_dev, dev, test) từ 12 trạm (tách cửa sổ theo từng trạm)."""
    scaled_splits, scaler, feature_cols = load_and_prepare_wp01_splits(
        data_dir=data_dir,
        mode=mode,
        scaler_save_path=scaler_save_path,
    )

    use_pin_memory = torch.cuda.is_available()
    num_workers = 4 if use_pin_memory else 0
    global_loaders: Dict[str, DataLoader] = {}

    for split_name in ALL_SPLITS:
        current_stride = train_stride if split_name == "train" else 1
        df_split = scaled_splits[split_name]

        if "station" in df_split.columns:
            x_parts: List[np.ndarray] = []
            y_parts: List[np.ndarray] = []
            for _, st_df in df_split.groupby("station", sort=True):
                st_clean = st_df.reset_index(drop=True)
                X_st, y_st = make_sliding_windows(
                    st_clean,
                    feature_cols=feature_cols,
                    target_col=TARGET_COL,
                    lookback=lookback,
                    horizon=horizon,
                    stride=current_stride,
                )
                if len(X_st) > 0:
                    x_parts.append(X_st)
                    y_parts.append(y_st)
            if not x_parts:
                raise ValueError(
                    f"Không tạo được cửa sổ trượt nào cho tập '{split_name}' (L={lookback}, H={horizon})."
                )
            X_all = np.concatenate(x_parts, axis=0)
            y_all = np.concatenate(y_parts, axis=0)
        else:
            X_all, y_all = make_sliding_windows(
                df_split,
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
    dropout: float = 0.35,
) -> nn.Module:
    """Truyền an toàn các tham số điều chuẩn (dropout) nếu kiến trúc hỗ trợ."""
    sig = inspect.signature(cls.__init__)
    kwargs = dict(base_kwargs)
    if "dropout" in sig.parameters:
        kwargs["dropout"] = dropout
    return cls(**kwargs)


def build_raw_wp02_backbone(
    model_name: str,
    lookback: int,
    horizon: int,
    in_features: int,
    dropout: float = 0.35,
) -> nn.Module:
    """Khởi tạo kiến trúc gốc từ WP02 (`src/models/*`) chưa bọc RevIN/ResidualAnchor."""
    name = model_name.strip().lower()
    base_kwargs: Dict[str, object] = {
        "lookback": lookback,
        "horizon": horizon,
        "in_features": in_features,
    }
    if name == "lstm":
        return _instantiate_with_regularization(LSTMModel, base_kwargs, dropout=dropout)
    if name == "dlinear":
        return DLinearModel(**base_kwargs)
    if name == "informer":
        return _instantiate_with_regularization(
            InformerModel, base_kwargs, dropout=dropout
        )
    if name in {"informer_full", "informer_fullattention"}:
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
        return InformerModel(**full_kwargs)
    if name == "patchtst":
        return _instantiate_with_regularization(
            PatchTSTModel, base_kwargs, dropout=dropout
        )
    raise ValueError(f"Kiến trúc mô hình không được hỗ trợ: {model_name}")


def build_model(
    model_name: str,
    lookback: int,
    horizon: int,
    in_features: int,
    device: torch.device,
    dropout: float = 0.35,
    use_wrappers: bool = True,
) -> nn.Module:
    """
    Khởi tạo mô hình:
    - `use_wrappers=True` (WP03): Bọc thêm RevINModelWrapper + ResidualAnchorWrapper.
    - `use_wrappers=False` (WP02 Baseline): Giữ nguyên kiến trúc gốc để nạp/huấn luyện trọng số WP02.
    """
    name = model_name.strip().lower()
    backbone = build_raw_wp02_backbone(
        model_name=name,
        lookback=lookback,
        horizon=horizon,
        in_features=in_features,
        dropout=dropout,
    )
    if not use_wrappers:
        return backbone.to(device)

    if name in {"lstm", "informer", "informer_full", "informer_fullattention"}:
        model: nn.Module = RevINModelWrapper(
            backbone=backbone, in_features=in_features, target_idx=0, affine=False
        )
    else:
        model = backbone

    wrapped = ResidualAnchorWrapper(
        model=model,
        horizon=horizon,
        in_features=in_features,
        target_idx=0,
        input_noise_std=0.05,
        init_scale=0.1,
    )
    return wrapped.to(device)


def build_stepwise_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Lịch tốc độ học tăng ấm mượt theo từng batch (Step-wise Warmup + Cosine Decay)."""
    total_steps = max(1, total_steps)
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
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
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """Huấn luyện 1 epoch với Mixed Precision (AMP), Gradient Clipping và Step-wise LR."""
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
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= old_scale:
                scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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
    metrics_calc: Optional[MetricsCalculator] = None,
) -> Dict[str, float]:
    """
    Đánh giá mô hình bằng `MetricsCalculator` từ `utils_metrics.py` trên toàn bộ tập dữ liệu
    (gom toàn bộ batch về shape [N, H, 1] trước khi tính MSE, RMSE, MAE để không lệch batch cuối).
    """
    calc = metrics_calc or MetricsCalculator()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    use_amp = device.type == "cuda"

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(batch_x).squeeze(-1)
            loss = criterion(preds, batch_y)

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds_list.append(preds.float().cpu())
        targets_list.append(batch_y.float().cpu())

    all_preds = torch.cat(preds_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    if all_preds.dim() == 2:
        all_preds = all_preds.unsqueeze(-1)
    if all_targets.dim() == 2:
        all_targets = all_targets.unsqueeze(-1)

    raw_metrics = calc.compute(all_preds, all_targets)
    mse = float(raw_metrics["mse"])
    mae = float(raw_metrics["mae"])
    rmse = float(raw_metrics.get("rmse", math.sqrt(max(0.0, mse))))

    return {
        "loss": total_loss / max(total_samples, 1),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def _to_3d_target_loader(loader: DataLoader, shuffle: bool) -> DataLoader:
    """Chuyển DataLoader (X, y[N, H]) sang (X, y[N, H, 1]) tương thích với WP02 BaselineTrainer."""
    ds = loader.dataset
    if not isinstance(ds, TensorDataset) or len(ds.tensors) < 2:
        return loader
    x_tensor, y_tensor = ds.tensors[0], ds.tensors[1]
    if y_tensor.dim() == 2:
        y_tensor = y_tensor.unsqueeze(-1)
    return DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=loader.batch_size or 256,
        shuffle=shuffle,
        drop_last=False,
    )


def run_wp02_baseline_pipeline(
    data_dir: Path,
    output_dir: Path,
    mode: str = "multivariate",
    lookback: int = 96,
    horizon: int = 24,
    batch_size: int = 512,
    train_stride: int = 12,
    epochs: int = 20,
    lr: float = 1e-3,
    seeds: Optional[List[int]] = None,
    models: Sequence[str] = ("lstm", "dlinear"),
) -> Tuple[pd.DataFrame, List[ExperimentMetricsContract]]:
    """
    Chạy thực nghiệm các mô hình Baseline của WP02 (`BaselineTrainer` trong `train_baseline.py`)
    trên chính bộ dữ liệu thật 12 trạm từ WP01 và đóng gói qua `ExperimentMetricsContract`.
    """
    seeds = seeds if seeds is not None else [42, 1024, 2026]
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    metrics_calc = MetricsCalculator()

    ckpt_dir = output_dir / "checkpoints" / "wp02_baseline"
    log_dir = output_dir / "logs" / "wp02_baseline"
    report_dir = output_dir / "reports"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = (
        output_dir / "checkpoints" / "global" / f"scaler_12stations_{mode}.joblib"
    )
    loaders, _, feature_cols = prepare_global_12stations_data(
        data_dir=data_dir,
        mode=mode,
        lookback=lookback,
        horizon=horizon,
        batch_size=batch_size,
        train_stride=train_stride,
        scaler_save_path=scaler_path,
    )
    in_features = len(feature_cols)

    train_loader_3d = _to_3d_target_loader(loaders["train"], shuffle=True)
    val_loader_3d = _to_3d_target_loader(loaders["dev"], shuffle=False)
    test_loader_3d = _to_3d_target_loader(loaders["test"], shuffle=False)

    baseline_models = [m.lower() for m in models if m.lower() in {"lstm", "dlinear"}]
    summary_rows: List[Dict[str, object]] = []
    contracts: List[ExperimentMetricsContract] = []
    ddof = 1 if len(seeds) > 1 else 0

    for m_key in baseline_models:
        wp02_name: Literal["LSTM", "DLinear"] = "LSTM" if m_key == "lstm" else "DLinear"
        mse_list: List[float] = []
        rmse_list: List[float] = []
        mae_list: List[float] = []
        time_list: List[float] = []
        best_ckpt = ""
        best_val = float("inf")

        for seed in seeds:
            set_seed(seed)
            raw_model = build_raw_wp02_backbone(
                model_name=m_key,
                lookback=lookback,
                horizon=horizon,
                in_features=in_features,
            )
            trainer = BaselineTrainer(
                model=raw_model,
                model_name=wp02_name,
                horizon=horizon,
                seed=seed,
                learning_rate=lr,
                device=device_str,
                checkpoint_dir=str(ckpt_dir),
                log_dir=str(log_dir),
            )
            t0 = time.perf_counter()
            trainer.fit(train_loader_3d, val_loader_3d, epochs=epochs)
            elapsed = time.perf_counter() - t0
            test_res = trainer.evaluate_test(test_loader_3d, metrics_calc)

            mse_val = float(test_res["mse"])
            mae_val = float(test_res["mae"])
            rmse_val = float(test_res.get("rmse", math.sqrt(max(0.0, mse_val))))

            mse_list.append(mse_val)
            rmse_list.append(rmse_val)
            mae_list.append(mae_val)
            time_list.append(elapsed)

            if trainer.early_stopping.best_loss < best_val:
                best_val = trainer.early_stopping.best_loss
                best_ckpt = trainer.checkpoint_path

        mse_arr = np.array(mse_list, dtype=np.float64)
        rmse_arr = np.array(rmse_list, dtype=np.float64)
        mae_arr = np.array(mae_list, dtype=np.float64)

        summary_rows.append(
            {
                "source": "WP02_BaselineTrainer",
                "model": m_key,
                "mode": mode,
                "horizon": horizon,
                "in_features": in_features,
                "num_seeds": len(seeds),
                "mse_mean": round(float(np.mean(mse_arr)), 6),
                "mse_std": round(float(np.std(mse_arr, ddof=ddof)), 6),
                "mse_mean_std": f"{np.mean(mse_arr):.6f} ± {np.std(mse_arr, ddof=ddof):.6f}",
                "rmse_mean": round(float(np.mean(rmse_arr)), 6),
                "rmse_std": round(float(np.std(rmse_arr, ddof=ddof)), 6),
                "rmse_mean_std": f"{np.mean(rmse_arr):.6f} ± {np.std(rmse_arr, ddof=ddof):.6f}",
                "mae_mean": round(float(np.mean(mae_arr)), 6),
                "mae_std": round(float(np.std(mae_arr, ddof=ddof)), 6),
                "mae_mean_std": f"{np.mean(mae_arr):.6f} ± {np.std(mae_arr, ddof=ddof):.6f}",
                "train_time_mean_sec": round(float(np.mean(time_list)), 3),
                "best_checkpoint": best_ckpt,
            }
        )

        if len(seeds) >= 3 and horizon in {24, 48, 96, 192}:
            contract = metrics_calc.summarize_multi_seed(
                experiment_id=f"WP02_BASE_{wp02_name}_{mode.upper()}_H{horizon}",
                model_name=wp02_name,
                station_id="ALL_12_STATIONS",
                horizon=horizon,  # type: ignore[arg-type]
                mse_per_seed=mse_list,
                mae_per_seed=mae_list,
                checkpoint_path=best_ckpt,
                seeds=seeds,
                extra_notes=f"WP02 BaselineTrainer chạy trên dữ liệu thật WP01 ({mode}, D={in_features}, H={horizon})",
            )
            contracts.append(contract)
            metrics_calc.print_summary(contract)

    base_df = pd.DataFrame(summary_rows)
    base_csv = report_dir / f"wp02_baseline_summary_{mode}_H{horizon}.csv"
    base_df.to_csv(base_csv, index=False)

    if contracts:
        contracts_json = report_dir / f"wp02_baseline_contracts_{mode}_H{horizon}.json"
        contracts_json.write_text(
            json.dumps(
                [c.model_dump() for c in contracts], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        print(f"[SAVED] Hợp đồng WP02 Baseline (Pydantic v2): {contracts_json}")

    return base_df, contracts


class TrainingPipeline:
    """
    Lớp điều phối toàn bộ quy trình Pre-training đa hạt giống (Multi-Seed) của WP03,
    tích hợp đánh giá chuẩn `MetricsCalculator` và xuất hợp đồng `ExperimentMetricsContract`.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        mode: str = "multivariate",
        lookback: int = 96,
        horizon: int = 24,
        batch_size: int = 512,
        train_stride: int = 12,
        lr: float = 3e-5,
        dlinear_lr: float = 1e-4,
        weight_decay: float = 3e-2,
        dropout: float = 0.35,
        epochs: int = 20,
        patience: int = 6,
        loss_fn: str = "MSELoss",
        huber_delta: float = 2.5,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.01,
        max_grad_norm: float = 1.0,
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
        self.dlinear_lr = dlinear_lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.loss_fn = loss_fn
        self.huber_delta = huber_delta
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.max_grad_norm = max_grad_norm
        self.seeds = seeds if seeds is not None else [42, 1024, 2026]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_calc = MetricsCalculator()
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
        feature_cols: List[str],
        target_scale: Optional[float] = None,
    ) -> Dict[str, object]:
        """Huấn luyện 1 mô hình trên 1 seed, theo dõi train/train_dev/dev và đánh giá cả 4 tập."""
        set_seed(seed)
        model = build_model(
            model_name=model_name,
            lookback=self.lookback,
            horizon=self.horizon,
            in_features=in_features,
            device=self.device,
            dropout=self.dropout,
            use_wrappers=True,
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        effective_lr = self.dlinear_lr if model_name.lower() == "dlinear" else self.lr

        train_criterion = build_criterion(
            loss_fn=self.loss_fn, huber_delta=self.huber_delta
        )
        eval_criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=effective_lr, weight_decay=self.weight_decay
        )
        total_steps = self.epochs * max(1, len(loaders["train"]))
        scheduler = build_stepwise_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_ratio=self.warmup_ratio,
            min_lr_ratio=self.min_lr_ratio,
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
            train_loss = train_one_epoch(
                model=model,
                dataloader=loaders["train"],
                criterion=train_criterion,
                optimizer=optimizer,
                device=self.device,
                scaler=grad_scaler,
                scheduler=scheduler,
                max_grad_norm=self.max_grad_norm,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            train_dev_metrics = evaluate(
                model,
                loaders["train_dev"],
                eval_criterion,
                self.device,
                self.metrics_calc,
            )
            dev_metrics = evaluate(
                model, loaders["dev"], eval_criterion, self.device, self.metrics_calc
            )
            dev_loss = dev_metrics["loss"]

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "train_dev_mse": round(train_dev_metrics["mse"], 6),
                    "train_dev_rmse": round(train_dev_metrics["rmse"], 6),
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
                f"| dev_mse={dev_loss:.5f} | dev_rmse={dev_metrics['rmse']:.5f} "
                f"| dev_mae={dev_metrics['mae']:.5f} | lr={current_lr:.2e}"
            )

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "source": "WP03_TrainingPipeline",
                        "use_wrappers": True,
                        "model_name": model_name,
                        "mode": self.mode,
                        "lookback": self.lookback,
                        "horizon": self.horizon,
                        "in_features": in_features,
                        "feature_cols": list(feature_cols),
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

        # Nạp lại trọng số tốt nhất (theo dev) và đánh giá đầy đủ trên cả 4 tập
        ckpt = torch.load(seed_ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        train_final = evaluate(
            model, loaders["train"], eval_criterion, self.device, self.metrics_calc
        )
        train_dev_final = evaluate(
            model, loaders["train_dev"], eval_criterion, self.device, self.metrics_calc
        )
        dev_final = evaluate(
            model, loaders["dev"], eval_criterion, self.device, self.metrics_calc
        )
        test_metrics = evaluate(
            model, loaders["test"], eval_criterion, self.device, self.metrics_calc
        )

        orig_msg = ""
        test_rmse_ugm3: Optional[float] = None
        test_mae_ugm3: Optional[float] = None
        if target_scale is not None and target_scale > 0:
            test_rmse_ugm3 = round(test_metrics["rmse"] * target_scale, 4)
            test_mae_ugm3 = round(test_metrics["mae"] * target_scale, 4)
            orig_msg = (
                f" | Thực tế: RMSE={test_rmse_ugm3:.2f}, MAE={test_mae_ugm3:.2f} ug/m3"
            )

        print(
            f"  --> [BEST EP={best_epoch:02d}] "
            f"Train MSE={train_final['mse']:.5f} | "
            f"Train-Dev MSE={train_dev_final['mse']:.5f} | "
            f"Dev MSE={dev_final['mse']:.5f} (RMSE={dev_final['rmse']:.5f}) | "
            f"Test MSE={test_metrics['mse']:.5f} (RMSE={test_metrics['rmse']:.5f}, MAE={test_metrics['mae']:.5f})"
            f"{orig_msg}"
        )

        del model, optimizer, scheduler, grad_scaler
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        record: Dict[str, object] = {
            "model": model_name,
            "mode": self.mode,
            "horizon": self.horizon,
            "seed": seed,
            "trainable_params": trainable_params,
            "best_epoch": best_epoch,
            "best_dev_loss": round(best_dev_loss, 6),
            "train_mse": round(train_final["mse"], 6),
            "train_rmse": round(train_final["rmse"], 6),
            "train_mae": round(train_final["mae"], 6),
            "train_dev_mse": round(train_dev_final["mse"], 6),
            "train_dev_rmse": round(train_dev_final["rmse"], 6),
            "train_dev_mae": round(train_dev_final["mae"], 6),
            "dev_mse": round(dev_final["mse"], 6),
            "dev_rmse": round(dev_final["rmse"], 6),
            "dev_mae": round(dev_final["mae"], 6),
            "test_mse": round(test_metrics["mse"], 6),
            "test_rmse": round(test_metrics["rmse"], 6),
            "test_mae": round(test_metrics["mae"], 6),
            "train_time_sec": round(train_time_sec, 3),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "checkpoint_path": str(seed_ckpt_path),
        }
        if test_rmse_ugm3 is not None and test_mae_ugm3 is not None:
            record["test_rmse_ugm3"] = test_rmse_ugm3
            record["test_mae_ugm3"] = test_mae_ugm3
        return record

    def run(
        self, models: Sequence[str], include_wp02_baseline: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chạy toàn bộ các mô hình qua danh sách seeds và xuất báo cáo + ExperimentMetricsContract."""
        scaler_path = self.ckpt_dir / f"scaler_12stations_{self.mode}.joblib"
        loaders, scaler, feature_cols = prepare_global_12stations_data(
            data_dir=self.data_dir,
            mode=self.mode,
            lookback=self.lookback,
            horizon=self.horizon,
            batch_size=self.batch_size,
            train_stride=self.train_stride,
            scaler_save_path=scaler_path,
        )
        target_scale: Optional[float] = (
            float(scaler.scale_[0])
            if hasattr(scaler, "scale_") and len(scaler.scale_) > 0
            else None
        )
        in_features = len(feature_cols)
        split_counts = {k: len(v.dataset) for k, v in loaders.items()}
        print(
            f"Thiết bị: {self.device} | Mode: {self.mode} (D={in_features}) "
            f"| L={self.lookback}, H={self.horizon} | Stride={self.train_stride} "
            f"| Loss={self.loss_fn} | Seeds: {self.seeds}"
        )
        print(f"Số cửa sổ dữ liệu 4 tập: {split_counts}")

        all_run_records = []
        summary_records = []
        contracts: List[ExperimentMetricsContract] = []
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
                    feature_cols=feature_cols,
                    target_scale=target_scale,
                )
                model_runs.append(rec)
                all_run_records.append(rec)

                if float(rec["best_dev_loss"]) < best_overall_dev_loss:
                    best_overall_dev_loss = float(rec["best_dev_loss"])
                    best_overall_ckpt_src = Path(str(rec["checkpoint_path"]))

            canonical_ckpt = (
                self.ckpt_dir
                / f"{model_name}_{self.mode}_H{self.horizon}_pretrained.pt"
            )
            if best_overall_ckpt_src is not None and best_overall_ckpt_src.exists():
                canonical_ckpt.write_bytes(best_overall_ckpt_src.read_bytes())

            df_m = pd.DataFrame(model_runs)
            train_mse_mean = float(df_m["train_mse"].to_numpy().mean())
            train_rmse_mean = float(df_m["train_rmse"].to_numpy().mean())
            train_dev_mse_mean = float(df_m["train_dev_mse"].to_numpy().mean())
            train_dev_rmse_mean = float(df_m["train_dev_rmse"].to_numpy().mean())
            dev_mse_mean = float(df_m["dev_mse"].to_numpy().mean())
            dev_rmse_mean = float(df_m["dev_rmse"].to_numpy().mean())

            mse_per_seed = [float(x) for x in df_m["test_mse"].tolist()]
            rmse_per_seed = [float(x) for x in df_m["test_rmse"].tolist()]
            mae_per_seed = [float(x) for x in df_m["test_mae"].tolist()]

            mse_mean = float(np.mean(mse_per_seed))
            mse_std = float(np.std(mse_per_seed, ddof=ddof))
            rmse_mean = float(np.mean(rmse_per_seed))
            rmse_std = float(np.std(rmse_per_seed, ddof=ddof))
            mae_mean = float(np.mean(mae_per_seed))
            mae_std = float(np.std(mae_per_seed, ddof=ddof))
            time_mean = float(df_m["train_time_sec"].to_numpy().mean())
            time_std = float(df_m["train_time_sec"].to_numpy().std(ddof=ddof))
            vram_mean = float(df_m["peak_vram_mb"].to_numpy().mean())
            vram_max = float(df_m["peak_vram_mb"].to_numpy().max())

            # Đóng gói theo ExperimentMetricsContract (utils_metrics.py) khi đủ >= 3 seeds
            if len(self.seeds) >= 3 and self.horizon in {24, 48, 96, 192}:
                contract = self.metrics_calc.summarize_multi_seed(
                    experiment_id=f"WP03_PRETRAIN_{model_name.upper()}_{self.mode.upper()}_H{self.horizon}",
                    model_name=to_contract_model_name(model_name),
                    station_id="ALL_12_STATIONS",
                    horizon=self.horizon,  # type: ignore[arg-type]
                    mse_per_seed=mse_per_seed,
                    mae_per_seed=mae_per_seed,
                    checkpoint_path=str(canonical_ckpt),
                    seeds=list(self.seeds),
                    extra_notes=(
                        f"WP03 Pre-training ({self.mode}, D={in_features}, L={self.lookback}, H={self.horizon}) | "
                        f"RMSE={rmse_mean:.6f}±{rmse_std:.6f}"
                    ),
                )
                contracts.append(contract)
                print(f"[CONTRACT ROW] {self.metrics_calc.format_report_row(contract)}")
            else:
                print(
                    f"[WARN] Số seeds hiện tại ({len(self.seeds)}) < 3 nên bỏ qua xuất ExperimentMetricsContract JSON."
                )

            summary_row: Dict[str, object] = {
                "model": model_name,
                "mode": self.mode,
                "horizon": self.horizon,
                "in_features": in_features,
                "num_seeds": len(self.seeds),
                "seeds": ",".join(map(str, self.seeds)),
                "trainable_params": int(df_m["trainable_params"].iloc[0]),
                "train_mse_mean": round(train_mse_mean, 6),
                "train_rmse_mean": round(train_rmse_mean, 6),
                "train_dev_mse_mean": round(train_dev_mse_mean, 6),
                "train_dev_rmse_mean": round(train_dev_rmse_mean, 6),
                "dev_mse_mean": round(dev_mse_mean, 6),
                "dev_rmse_mean": round(dev_rmse_mean, 6),
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
                "checkpoint_path": str(canonical_ckpt),
            }
            if "test_rmse_ugm3" in df_m.columns:
                rmse_ug_mean = float(df_m["test_rmse_ugm3"].to_numpy().mean())
                rmse_ug_std = float(df_m["test_rmse_ugm3"].to_numpy().std(ddof=ddof))
                mae_ug_mean = float(df_m["test_mae_ugm3"].to_numpy().mean())
                mae_ug_std = float(df_m["test_mae_ugm3"].to_numpy().std(ddof=ddof))
                summary_row["rmse_ugm3_mean_std"] = (
                    f"{rmse_ug_mean:.2f} ± {rmse_ug_std:.2f}"
                )
                summary_row["mae_ugm3_mean_std"] = (
                    f"{mae_ug_mean:.2f} ± {mae_ug_std:.2f}"
                )

            summary_records.append(summary_row)

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

        if contracts:
            contracts_json = (
                self.report_dir
                / f"global_pretrain_contracts_{self.mode}_H{self.horizon}.json"
            )
            contracts_json.write_text(
                json.dumps(
                    [c.model_dump() for c in contracts], ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            print(
                f"Đã lưu hợp đồng ExperimentMetricsContract JSON tại: {contracts_json}"
            )

        print(f"\nĐã lưu chi tiết từng seed (đủ 4 tập) tại: {runs_csv}")
        print(f"Đã lưu tổng hợp mean ± std tại: {summary_csv}")

        if include_wp02_baseline:
            print(
                "\n[WP02 INTEGRATION] Chạy đối chứng WP02 BaselineTrainer trên dữ liệu WP01..."
            )
            run_wp02_baseline_pipeline(
                data_dir=self.data_dir,
                output_dir=self.output_dir,
                mode=self.mode,
                lookback=self.lookback,
                horizon=self.horizon,
                batch_size=self.batch_size,
                train_stride=self.train_stride,
                epochs=self.epochs,
                seeds=self.seeds,
                models=models,
            )

        return runs_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T0301: Huấn luyện Pre-training đa hạt giống (3 seeds) tích hợp WP02 & utils_metrics."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/preprocess"))
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
    parser.add_argument("--train-stride", type=int, default=12)
    parser.add_argument("--train-dev-stride", type=int, default=None)
    parser.add_argument("--dev-stride", type=int, default=None)
    parser.add_argument("--test-stride", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dlinear-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--loss-fn", type=str, default="MSELoss")
    parser.add_argument("--huber-delta", type=float, default=2.5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 1024, 2026],
        help="Danh sách hạt giống ngẫu nhiên (mặc định 3 seeds: 42 1024 2026).",
    )
    parser.add_argument(
        "--include-wp02-baseline",
        action="store_true",
        help="Chạy kèm WP02 BaselineTrainer (LSTM, DLinear) trên cùng dữ liệu WP01 để đối chứng.",
    )
    apply_yaml_config(parser, ["paths", "data", "pretrain"])
    args = parser.parse_args()
    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)
    return args


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
        dlinear_lr=args.dlinear_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        loss_fn=args.loss_fn,
        huber_delta=args.huber_delta,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        max_grad_norm=args.max_grad_norm,
        seeds=args.seeds,
    )
    pipeline.run(
        models=args.models,
        include_wp02_baseline=args.include_wp02_baseline,
    )
