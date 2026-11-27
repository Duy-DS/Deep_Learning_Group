"""
T0301 — Pipeline huấn luyện toàn cục (Global Pre-training) cho dự báo PM2.5 Bắc Kinh
trên dữ liệu 12 trạm quan trắc, tích hợp chuẩn đánh giá `utils_metrics.py` và
mô hình Baseline từ WP02 (`train_baseline.py`, `src/models/*`).

Bản nâng cấp khắc phục lỗi SOTA thua Baseline & đường dự báo phẳng:
- Gỡ `ResidualAnchorWrapper` khỏi mặc định (giữ lại class cho tương thích ngược checkpoint cũ).
- Khôi phục `RevINModelWrapper` chuẩn (cộng lại `+ mean`, bật `affine=True`, `min_std=0.1`
  chỉ trên 11 kênh vật lý liên tục).
- Nạp đầy đủ cấu hình `models:` từ `configs/default.yaml` (xóa dead-config) và hỗ trợ LR/Dropout
  công bằng theo kiến trúc.
- Tách nhóm tham số AdamW decoupled weight decay (`weight_decay=0.0` cho bias/norm/RevIN/pos_emb).
- Cân bằng lựa chọn checkpoint/Early Stopping theo mùa (`balanced_dev`: 50% `train_dev` Đông-Xuân
  + 50% `dev` Hè chuẩn hóa theo phương sai — 0% data leakage, giữ nguyên chia tập WP01).
- Hỗ trợ `PeakShapeLoss` (Peak-Weighted MSE + First-Order Temporal Slope Loss) và ghi nhận
  thêm chỉ số chẩn đoán đỉnh/độ phẳng (`spike_mae_ugm3`, `peak_capture_ratio`, `range_ratio`)
  song song với đánh giá chuẩn không trọng số qua `ExperimentMetricsContract`.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

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
    Bọc kiến trúc dự báo với Reversible Instance Normalization (RevIN) chuẩn:
    - Chuẩn hóa các kênh vật lý/ô nhiễm liên tục theo từng cửa sổ `(B, L, C_cont)`
      với chặn dưới `stdev >= min_std` để tránh khuếch đại nhiễu trên kênh thưa (`rain`).
    - Giữ nguyên các kênh mã hóa chu kỳ thời gian và One-Hot hướng gió.
    - Giải chuẩn hóa đầy đủ đầu ra dự báo (`y_unscaled * stdev + mean`) về lại thang đo gốc.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_continuous_features: Optional[int] = None,
        target_idx: int = 0,
        eps: float = 1e-5,
        min_std: float = 0.1,
        affine: bool = True,
        restore_mean: bool = True,
    ) -> None:
        """
        Khởi tạo lớp bọc RevIN cho các kiến trúc nhạy cảm với dịch chuyển phân phối (LSTM, Informer).

        Tham số:
            backbone (nn.Module): Mô hình nền dự báo chuỗi thời gian.
            in_features (int): Tổng số đặc trưng đầu vào D.
            num_continuous_features (Optional[int]): Số kênh liên tục đầu tiên cần chuẩn hóa.
            target_idx (int): Chỉ số cột mục tiêu PM2.5 (mặc định 0).
            eps (float): Hằng số ổn định số học.
            min_std (float): Ngưỡng chặn dưới độ lệch chuẩn.
            affine (bool): Bật tham số co giãn/tịnh tiến học được (mặc định True).
            restore_mean (bool): Cộng lại trung bình cửa sổ khi giải chuẩn hóa (mặc định True).
        """
        super().__init__()
        self.backbone = backbone
        self.in_features = int(in_features)
        if num_continuous_features is None:
            self.num_cont = 11 if self.in_features >= 15 else self.in_features
        else:
            self.num_cont = min(int(num_continuous_features), self.in_features)

        self.target_idx = int(target_idx)
        self.eps = float(eps)
        self.min_std = float(min_std)
        self.affine = bool(affine)
        self.restore_mean = bool(restore_mean)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_cont))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_cont))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền tiến chuẩn hóa -> backbone -> giải chuẩn hóa, trả về `[B, H, 1]`."""
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
        if self.restore_mean:
            y_out = y_out + mean[:, :, idx]
        return y_out.unsqueeze(-1)


class ResidualAnchorWrapper(nn.Module):
    """
    Lớp bọc phần dư neo vào giá trị cuối cùng (chỉ giữ cho tương thích ngược checkpoint cũ
    hoặc thí nghiệm ablation tùy chọn; TẮT mặc định vì gây phẳng đường dự báo & xung đột RevIN).
    """

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


class PeakShapeLoss(nn.Module):
    """
    Hàm mất mát kết hợp bảo toàn đỉnh ô nhiễm (Peak-Weighted MSE) và
    bảo toàn độ dốc biến thiên theo thời gian (First-Order Temporal Difference Loss).

    Công thức:
        L(y_hat, y) = mean(w(y) * (y_hat - y)^2) + diff_weight * MSE(Δy_hat, Δy)
    trong đó:
        w(y) = clamp(1 + peak_alpha * ReLU(y - peak_threshold_z), max=max_weight)
        Δy_t = y_{t} - y_{t-1}

    Mục đích: Giảm hiện tượng suy biến về đường phẳng (regression-to-the-mean) và tăng khả năng
    bắt đỉnh trong các đợt ô nhiễm nặng PM2.5 >= 150 ug/m3 mà không làm méo thang đo z-score.
    """

    def __init__(
        self,
        peak_threshold_z: float = 0.84,
        peak_alpha: float = 0.25,
        diff_weight: float = 0.10,
        max_weight: float = 3.5,
    ) -> None:
        """
        Khởi tạo hàm mất mát PeakShapeLoss.

        Tham số:
            peak_threshold_z (float): Ngưỡng z-score bắt đầu tăng trọng số đỉnh (~150 ug/m3).
            peak_alpha (float): Hệ số tăng trọng số tuyến tính theo mức vượt ngưỡng đỉnh.
            diff_weight (float): Trọng số thành phần phạt sai lệch độ dốc bậc 1 theo thời gian.
            max_weight (float): Giới hạn trên trọng số mỗi điểm tránh bùng nổ do ngoại lai cực đoan.
        """
        super().__init__()
        self.peak_threshold_z = float(peak_threshold_z)
        self.peak_alpha = float(peak_alpha)
        self.diff_weight = float(diff_weight)
        self.max_weight = float(max_weight)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Tính giá trị hàm mất mát PeakShapeLoss trên cặp `(preds [B, H], targets [B, H])`."""
        sq_err = (preds - targets) ** 2
        if self.peak_alpha > 0.0:
            excess = torch.relu(targets - self.peak_threshold_z)
            weights = (1.0 + self.peak_alpha * excess).clamp(max=self.max_weight)
            base_loss = torch.mean(weights * sq_err)
        else:
            base_loss = torch.mean(sq_err)

        if self.diff_weight > 0.0 and preds.size(-1) > 1:
            pred_diff = preds[:, 1:] - preds[:, :-1]
            target_diff = targets[:, 1:] - targets[:, :-1]
            slope_loss = torch.mean((pred_diff - target_diff) ** 2)
            return base_loss + self.diff_weight * slope_loss
        return base_loss


def set_seed(seed: int) -> None:
    """Cố định hạt giống ngẫu nhiên cho toàn bộ thư viện để đảm bảo tái lập thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_yaml_file(config_path: str | Path) -> Dict[str, Any]:
    """Đọc toàn bộ nội dung tệp cấu hình YAML nếu tồn tại."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_configs(config_path: str | Path = "configs/default.yaml") -> Dict[str, Dict[str, Any]]:
    """Trích xuất mục `models:` từ tệp YAML làm nguồn sự thật duy nhất cho kiến trúc."""
    cfg = load_yaml_file(config_path)
    raw_models = cfg.get("models", {})
    if not isinstance(raw_models, dict):
        return {}
    return {
        str(k).strip().lower(): dict(v)
        for k, v in raw_models.items()
        if isinstance(v, dict)
    }


def apply_yaml_config(parser: argparse.ArgumentParser, sections: list[str]) -> None:
    """Nạp tham số mặc định từ tệp cấu hình YAML nếu tồn tại."""
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    pre_args, _ = parser.parse_known_args()
    cfg = load_yaml_file(pre_args.config)
    if not cfg:
        return

    merged: dict = {}
    for sec in sections:
        sec_data = cfg.get(sec, {})
        if isinstance(sec_data, dict):
            merged.update(sec_data)

    valid_dests = {action.dest for action in parser._actions}
    parser.set_defaults(**{k: v for k, v in merged.items() if k in valid_dests})


def build_criterion(
    loss_fn: str = "PeakShapeLoss",
    huber_delta: float = 2.5,
    peak_threshold_z: float = 0.84,
    peak_alpha: float = 0.25,
    diff_weight: float = 0.10,
    peak_max_weight: float = 3.5,
) -> nn.Module:
    """Khởi tạo hàm mất mát huấn luyện theo tên cấu hình."""
    name = loss_fn.strip().lower()
    if name in {"mse", "mseloss", "l2"}:
        return nn.MSELoss()
    if name in {"huber", "huberloss", "smoothl1", "smoothl1loss"}:
        return nn.HuberLoss(delta=huber_delta)
    if name in {"mae", "l1", "l1loss"}:
        return nn.L1Loss()
    if name in {"peakshape", "peakshapeloss", "peak_shape", "peak_mse"}:
        return PeakShapeLoss(
            peak_threshold_z=peak_threshold_z,
            peak_alpha=peak_alpha,
            diff_weight=diff_weight,
            max_weight=peak_max_weight,
        )
    raise ValueError(f"Hàm mất mát không được hỗ trợ: {loss_fn}")


def build_optimizer_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """
    Tách nhóm tham số cho AdamW decoupled weight decay:
    - Không áp weight decay lên bias, LayerNorm/BatchNorm, RevIN, positional/horizon embeddings
      và các tensor 1 chiều để không bóp nghẹt khả năng căn chỉnh biên độ và dịch mức.
    """
    no_decay_tokens = (
        "bias",
        "norm",
        "revin",
        "affine_weight",
        "affine_bias",
        "pos_embedding",
        "horizon_queries",
        "res_scale",
        "station_bias",
    )
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = param_name.lower()
        if param.ndim < 2 or any(tok in lower_name for tok in no_decay_tokens):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    return groups


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
    train_stride: int = 2,
    scaler_save_path: Optional[Path] = None,
) -> Tuple[Dict[str, DataLoader], StandardScaler, List[str]]:
    """Tạo đầy đủ 4 DataLoader (train, train_dev, dev, test) từ 12 trạm (tách cửa sổ theo từng trạm)."""
    scaled_splits, scaler, feature_cols = load_and_prepare_wp01_splits(
        data_dir=data_dir,
        mode=mode,
        scaler_save_path=scaler_save_path,
    )

    use_pin_memory = torch.cuda.is_available()
    # TensorDataset nằm hoàn toàn trong RAM nên num_workers=0 tránh nhân bản RAM qua fork() gây OOM ở H=192
    num_workers = 0
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


def _filter_init_kwargs(
    cls: type[nn.Module],
    base_kwargs: Dict[str, Any],
    extra_kwargs: Optional[Dict[str, Any]] = None,
    dropout: Optional[float] = None,
) -> Dict[str, Any]:
    """Lọc an toàn các tham số hợp lệ với `__init__` của lớp mô hình từ YAML và tham số gọi."""
    sig = inspect.signature(cls.__init__)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    meta_only_keys = {"use_revin_wrapper", "lr"}

    merged = dict(base_kwargs)
    if extra_kwargs:
        for k, v in extra_kwargs.items():
            if k in meta_only_keys or v is None:
                continue
            if k in sig.parameters or accepts_var_kw:
                merged[k] = v

    if dropout is not None and "dropout" in sig.parameters:
        if not extra_kwargs or "dropout" not in extra_kwargs:
            merged["dropout"] = dropout

    return {
        k: v
        for k, v in merged.items()
        if (k in sig.parameters or accepts_var_kw) and k not in meta_only_keys
    }


def _instantiate_with_regularization(
    cls: type[nn.Module],
    base_kwargs: Dict[str, object],
    dropout: float = 0.20,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Truyền an toàn các tham số cấu hình và điều chuẩn (dropout) nếu kiến trúc hỗ trợ."""
    kwargs = _filter_init_kwargs(
        cls=cls, base_kwargs=dict(base_kwargs), extra_kwargs=model_kwargs, dropout=dropout
    )
    return cls(**kwargs)


def build_raw_wp02_backbone(
    model_name: str,
    lookback: int,
    horizon: int,
    in_features: int,
    dropout: float = 0.20,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Khởi tạo kiến trúc gốc (`src/models/*`) theo đúng cấu hình YAML trước khi bọc wrapper."""
    name = model_name.strip().lower()
    base_kwargs: Dict[str, object] = {
        "lookback": lookback,
        "horizon": horizon,
        "in_features": in_features,
    }
    extra = dict(model_kwargs) if model_kwargs else {}

    if name == "lstm":
        return _instantiate_with_regularization(
            LSTMModel, base_kwargs, dropout=dropout, model_kwargs=extra
        )
    if name == "dlinear":
        kwargs = _filter_init_kwargs(
            DLinearModel, dict(base_kwargs), extra_kwargs=extra, dropout=None
        )
        return DLinearModel(**kwargs)
    if name == "informer":
        extra.setdefault("attention_type", "probsparse")
        return _instantiate_with_regularization(
            InformerModel, base_kwargs, dropout=dropout, model_kwargs=extra
        )
    if name in {"informer_full", "informer_fullattention"}:
        extra["attention_type"] = "full"
        extra.setdefault("factor", lookback)
        return _instantiate_with_regularization(
            InformerModel, base_kwargs, dropout=dropout, model_kwargs=extra
        )
    if name == "patchtst":
        return _instantiate_with_regularization(
            PatchTSTModel, base_kwargs, dropout=dropout, model_kwargs=extra
        )
    raise ValueError(f"Kiến trúc mô hình không được hỗ trợ: {model_name}")


def build_model(
    model_name: str,
    lookback: int,
    horizon: int,
    in_features: int,
    device: torch.device,
    dropout: float = 0.20,
    use_wrappers: bool = True,
    use_residual_anchor: bool = False,
    use_revin_wrapper: Optional[bool] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Khởi tạo mô hình chuẩn WP03/WP04/WP05:
    - `use_wrappers=True, use_residual_anchor=False, use_revin_wrapper=False` (Mặc định v2):
      * Chạy trực tiếp kiến trúc chuẩn trên không gian z-score toàn cục (không bọc `RevINModelWrapper`
        hay `ResidualAnchorWrapper`) để bảo toàn tính hồi quy về trung bình ở H=96/192 và tránh
        khuếch đại sai số đỉnh trên chuỗi PM2.5 lệch phải.
      * Đảm bảo 100% tương thích `strict=True` với WP04 (`ig_explainer.py`) và WP05 (`PyTorchCheckpointLoader`).
    - `use_revin_wrapper=True`: tùy chọn bật `RevINModelWrapper` cho thực nghiệm ablation hoặc nạp checkpoint cũ.
    - `use_residual_anchor=True`: chế độ tương thích ngược cho checkpoint cũ có `res_scale`.
    - `use_wrappers=False`: giữ nguyên kiến trúc gốc WP02 Baseline.
    """
    name = model_name.strip().lower()
    extra = dict(model_kwargs) if model_kwargs else {}

    backbone = build_raw_wp02_backbone(
        model_name=name,
        lookback=lookback,
        horizon=horizon,
        in_features=in_features,
        dropout=dropout,
        model_kwargs=extra,
    )
    if not use_wrappers:
        return backbone.to(device)

    legacy_revin_models = {
        "lstm",
        "informer",
        "informer_full",
        "informer_fullattention",
    }
    if use_revin_wrapper is not None:
        want_revin_wrapper = bool(use_revin_wrapper)
    elif "use_revin_wrapper" in extra:
        want_revin_wrapper = bool(extra["use_revin_wrapper"])
    elif use_residual_anchor:
        want_revin_wrapper = name in legacy_revin_models
    else:
        want_revin_wrapper = False

    if want_revin_wrapper:
        # Nếu bật ResidualAnchorWrapper cũ, RevIN không cộng mean và tắt affine như bản legacy
        model: nn.Module = RevINModelWrapper(
            backbone=backbone,
            in_features=in_features,
            num_continuous_features=extra.get("num_continuous_features"),
            target_idx=int(extra.get("target_idx", 0)),
            min_std=float(extra.get("min_std", 0.1)),
            affine=(not use_residual_anchor),
            restore_mean=(not use_residual_anchor),
        )
    else:
        model = backbone

    if use_residual_anchor:
        model = ResidualAnchorWrapper(
            model=model,
            horizon=horizon,
            in_features=in_features,
            target_idx=int(extra.get("target_idx", 0)),
            input_noise_std=0.05,
            init_scale=0.1,
        )

    return model.to(device)


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


def compute_loader_target_variance(dataloader: DataLoader) -> float:
    """Tính phương sai mục tiêu `y` trên một DataLoader để chuẩn hóa điểm validation liên mùa."""
    ds = dataloader.dataset
    if isinstance(ds, TensorDataset) and len(ds.tensors) >= 2:
        y_t = ds.tensors[1].float()
        return max(float(torch.var(y_t, unbiased=False).item()), 1e-6)
    ys = [batch_y.float() for _, batch_y in dataloader]
    if not ys:
        return 1.0
    all_y = torch.cat(ys, dim=0)
    return max(float(torch.var(all_y, unbiased=False).item()), 1e-6)


def compute_selection_score(
    train_dev_mse: float,
    dev_mse: float,
    var_train_dev: float,
    var_dev: float,
    mode: str = "balanced_dev",
) -> float:
    """
    Tính điểm chọn checkpoint / Early Stopping:
    - `balanced_dev` (Khuyến nghị): 0.5 * (train_dev_mse / var_train_dev) + 0.5 * (dev_mse / var_dev)
      giúp tập Đông-Xuân (`train_dev`) và tập Hè (`dev`) đóng góp công bằng 50/50.
    - `dev`: chỉ dùng `dev_mse` (giống pipeline cũ).
    - `train_dev`: chỉ dùng `train_dev_mse`.
    """
    mode_clean = str(mode).strip().lower()
    if mode_clean == "dev":
        return float(dev_mse)
    if mode_clean == "train_dev":
        return float(train_dev_mse)
    return 0.5 * (float(train_dev_mse) / max(var_train_dev, 1e-6)) + 0.5 * (
        float(dev_mse) / max(var_dev, 1e-6)
    )


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
    target_mean: Optional[float] = None,
    target_scale: Optional[float] = None,
    spike_threshold_ugm3: float = 150.0,
) -> Dict[str, float]:
    """
    Đánh giá mô hình bằng `MetricsCalculator` từ `utils_metrics.py` trên toàn bộ tập dữ liệu
    kèm các chỉ số chẩn đoán đỉnh ô nhiễm & độ phẳng (`spike_mae_ugm3`, `peak_capture_ratio`,
    `range_ratio`) phục vụ kiểm toán trực quan biểu đồ.
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

    all_preds_2d = torch.cat(preds_list, dim=0)
    all_targets_2d = torch.cat(targets_list, dim=0)
    all_preds = all_preds_2d.unsqueeze(-1) if all_preds_2d.dim() == 2 else all_preds_2d
    all_targets = (
        all_targets_2d.unsqueeze(-1) if all_targets_2d.dim() == 2 else all_targets_2d
    )

    raw_metrics = calc.compute(all_preds, all_targets)
    mse = float(raw_metrics["mse"])
    mae = float(raw_metrics["mae"])
    rmse = float(raw_metrics.get("rmse", math.sqrt(max(0.0, mse))))

    # Chẩn đoán độ phẳng (tỷ lệ biên độ dao động nội khung dự báo so với thực tế)
    p_np = all_preds.squeeze(-1).numpy()
    t_np = all_targets.squeeze(-1).numpy()
    true_ranges = t_np.max(axis=1) - t_np.min(axis=1)
    pred_ranges = p_np.max(axis=1) - p_np.min(axis=1)
    mean_true_range = float(np.mean(true_ranges))
    range_ratio = (
        float(np.mean(pred_ranges) / mean_true_range) if mean_true_range > 1e-8 else 0.0
    )

    result: Dict[str, float] = {
        "loss": total_loss / max(total_samples, 1),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "range_ratio": range_ratio,
    }

    # Chẩn đoán khả năng bắt đỉnh trên vùng ô nhiễm nặng (>= 150 ug/m3)
    if target_mean is not None and target_scale is not None and target_scale > 0:
        p_ug = np.clip(p_np * target_scale + target_mean, 0.0, None)
        t_ug = t_np * target_scale + target_mean
        spike_pts = t_ug >= spike_threshold_ugm3
        if np.any(spike_pts):
            diff_spike = p_ug[spike_pts] - t_ug[spike_pts]
            result["spike_mae_ugm3"] = float(np.mean(np.abs(diff_spike)))
            result["spike_rmse_ugm3"] = float(math.sqrt(float(np.mean(diff_spike**2))))
        else:
            result["spike_mae_ugm3"] = float("nan")
            result["spike_rmse_ugm3"] = float("nan")

        t_win_max = t_ug.max(axis=1)
        p_win_max = p_ug.max(axis=1)
        spike_wins = t_win_max >= spike_threshold_ugm3
        if np.any(spike_wins):
            result["peak_capture_ratio"] = float(
                np.mean(p_win_max[spike_wins] / np.maximum(t_win_max[spike_wins], 1e-6))
            )
        else:
            result["peak_capture_ratio"] = float("nan")

    return result


def _to_3d_target_loader(loader: DataLoader, shuffle: bool) -> DataLoader:
    """Chuyển DataLoader `(X, y[N, H])` sang `(X, y[N, H, 1])` tương thích với WP02 BaselineTrainer."""
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
    train_stride: int = 2,
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
        train_stride: int = 4,
        lr: float = 2e-4,
        dlinear_lr: float = 1e-3,
        weight_decay: float = 3e-2,
        dropout: float = 0.30,
        epochs: int = 25,
        patience: int = 5,
        loss_fn: str = "PeakShapeLoss",
        huber_delta: float = 2.5,
        peak_threshold_z: float = 0.84,
        peak_alpha: float = 0.15,
        diff_weight: float = 0.05,
        peak_max_weight: float = 3.0,
        use_residual_anchor: bool = False,
        val_selection_mode: str = "balanced_dev",
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        warmup_ratio: float = 0.15,
        min_lr_ratio: float = 0.01,
        max_grad_norm: float = 1.0,
        seeds: Optional[List[int]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
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
        self.peak_threshold_z = peak_threshold_z
        self.peak_alpha = peak_alpha
        self.diff_weight = diff_weight
        self.peak_max_weight = peak_max_weight
        self.use_residual_anchor = use_residual_anchor
        self.val_selection_mode = val_selection_mode
        self.model_configs = dict(model_configs) if model_configs else {}
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

    def _resolve_model_hyperparams(
        self, model_name: str
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Trích xuất (effective_lr, effective_dropout, model_kwargs) từ cấu hình YAML."""
        m_key = model_name.strip().lower()
        m_cfg = dict(self.model_configs.get(m_key, {}))
        if m_key == "dlinear":
            effective_lr = float(m_cfg.get("lr", self.dlinear_lr))
        else:
            effective_lr = float(m_cfg.get("lr", self.lr))
        effective_dropout = float(m_cfg.get("dropout", self.dropout))
        return effective_lr, effective_dropout, m_cfg

    def train_single_seed(
        self,
        model_name: str,
        seed: int,
        loaders: Dict[str, DataLoader],
        in_features: int,
        feature_cols: List[str],
        target_mean: Optional[float] = None,
        target_scale: Optional[float] = None,
        var_train_dev: float = 1.0,
        var_dev: float = 1.0,
    ) -> Dict[str, object]:
        """Huấn luyện 1 mô hình trên 1 seed, chọn checkpoint cân bằng mùa và đánh giá cả 4 tập."""
        set_seed(seed)
        effective_lr, effective_dropout, model_kwargs = self._resolve_model_hyperparams(
            model_name
        )
        model = build_model(
            model_name=model_name,
            lookback=self.lookback,
            horizon=self.horizon,
            in_features=in_features,
            device=self.device,
            dropout=effective_dropout,
            use_wrappers=True,
            use_residual_anchor=self.use_residual_anchor,
            model_kwargs=model_kwargs,
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        train_criterion = build_criterion(
            loss_fn=self.loss_fn,
            huber_delta=self.huber_delta,
            peak_threshold_z=self.peak_threshold_z,
            peak_alpha=self.peak_alpha,
            diff_weight=self.diff_weight,
            peak_max_weight=self.peak_max_weight,
        )
        eval_criterion = nn.MSELoss()

        param_groups = build_optimizer_param_groups(
            model=model, lr=effective_lr, weight_decay=self.weight_decay
        )
        optimizer = torch.optim.AdamW(param_groups)
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

        best_val_score = float("inf")
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
            val_score = compute_selection_score(
                train_dev_mse=train_dev_metrics["mse"],
                dev_mse=dev_metrics["mse"],
                var_train_dev=var_train_dev,
                var_dev=var_dev,
                mode=self.val_selection_mode,
            )

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
                    "val_score": round(val_score, 6),
                    "lr": current_lr,
                }
            )

            print(
                f"[{model_name.upper()} | seed={seed}] Epoch {epoch:02d}/{self.epochs:02d} "
                f"| train_loss={train_loss:.5f} | train_dev_mse={train_dev_metrics['mse']:.5f} "
                f"| dev_mse={dev_loss:.5f} | val_score={val_score:.5f} | lr={current_lr:.2e}"
            )

            if val_score < best_val_score:
                best_val_score = val_score
                best_dev_loss = dev_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "source": "WP03_TrainingPipeline_v2",
                        "use_wrappers": True,
                        "use_residual_anchor": self.use_residual_anchor,
                        "use_revin_wrapper": bool(
                            model_kwargs.get("use_revin_wrapper", False)
                        ),
                        "val_selection_mode": self.val_selection_mode,
                        "loss_fn": self.loss_fn,
                        "model_name": model_name,
                        "model_kwargs": model_kwargs,
                        "mode": self.mode,
                        "lookback": self.lookback,
                        "horizon": self.horizon,
                        "in_features": in_features,
                        "feature_cols": list(feature_cols),
                        "dropout": effective_dropout,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_dev_loss": best_dev_loss,
                        "best_val_score": best_val_score,
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

        # Nạp lại trọng số tốt nhất và đánh giá đầy đủ trên cả 4 tập (luôn dùng MSE/MAE chuẩn)
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
            model,
            loaders["test"],
            eval_criterion,
            self.device,
            self.metrics_calc,
            target_mean=target_mean,
            target_scale=target_scale,
        )

        orig_msg = ""
        test_rmse_ugm3: Optional[float] = None
        test_mae_ugm3: Optional[float] = None
        if target_scale is not None and target_scale > 0:
            test_rmse_ugm3 = round(test_metrics["rmse"] * target_scale, 4)
            test_mae_ugm3 = round(test_metrics["mae"] * target_scale, 4)
            spike_mae_str = (
                f", SpikeMAE={test_metrics['spike_mae_ugm3']:.1f}"
                if not math.isnan(test_metrics.get("spike_mae_ugm3", float("nan")))
                else ""
            )
            peak_ratio_str = (
                f", PeakCap={test_metrics['peak_capture_ratio']*100:.1f}%"
                if not math.isnan(test_metrics.get("peak_capture_ratio", float("nan")))
                else ""
            )
            orig_msg = (
                f" | Thực tế: RMSE={test_rmse_ugm3:.2f}, MAE={test_mae_ugm3:.2f} ug/m3"
                f"{spike_mae_str}{peak_ratio_str}, RangeRatio={test_metrics['range_ratio']*100:.1f}%"
            )

        print(
            f"  --> [BEST EP={best_epoch:02d}] "
            f"Train MSE={train_final['mse']:.5f} | "
            f"Train-Dev MSE={train_dev_final['mse']:.5f} | "
            f"Dev MSE={dev_final['mse']:.5f} | "
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
            "best_val_score": round(best_val_score, 6),
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
            "test_range_ratio": round(test_metrics["range_ratio"], 4),
            "train_time_sec": round(train_time_sec, 3),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "checkpoint_path": str(seed_ckpt_path),
        }
        if test_rmse_ugm3 is not None and test_mae_ugm3 is not None:
            record["test_rmse_ugm3"] = test_rmse_ugm3
            record["test_mae_ugm3"] = test_mae_ugm3
        if "spike_mae_ugm3" in test_metrics and not math.isnan(
            test_metrics["spike_mae_ugm3"]
        ):
            record["test_spike_mae_ugm3"] = round(test_metrics["spike_mae_ugm3"], 4)
            record["test_spike_rmse_ugm3"] = round(test_metrics["spike_rmse_ugm3"], 4)
        if "peak_capture_ratio" in test_metrics and not math.isnan(
            test_metrics["peak_capture_ratio"]
        ):
            record["test_peak_capture_ratio"] = round(
                test_metrics["peak_capture_ratio"], 4
            )
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
        target_mean: Optional[float] = (
            float(scaler.mean_[0])
            if hasattr(scaler, "mean_") and len(scaler.mean_) > 0
            else None
        )
        target_scale: Optional[float] = (
            float(scaler.scale_[0])
            if hasattr(scaler, "scale_") and len(scaler.scale_) > 0
            else None
        )
        var_train_dev = compute_loader_target_variance(loaders["train_dev"])
        var_dev = compute_loader_target_variance(loaders["dev"])
        in_features = len(feature_cols)
        split_counts = {k: len(v.dataset) for k, v in loaders.items()}
        print(
            f"Thiết bị: {self.device} | Mode: {self.mode} (D={in_features}) "
            f"| L={self.lookback}, H={self.horizon} | Stride={self.train_stride} "
            f"| Loss={self.loss_fn} | ValSel={self.val_selection_mode} "
            f"| ResAnchor={self.use_residual_anchor} | Seeds: {self.seeds}"
        )
        print(
            f"Số cửa sổ dữ liệu 4 tập: {split_counts} "
            f"(var_train_dev={var_train_dev:.4f}, var_dev={var_dev:.4f})"
        )

        all_run_records = []
        summary_records = []
        contracts: List[ExperimentMetricsContract] = []
        ddof = 1 if len(self.seeds) > 1 else 0

        for model_name in models:
            print(f"\n{'=' * 70}\nHUẤN LUYỆN MÔ HÌNH: {model_name.upper()}\n{'=' * 70}")
            model_runs = []
            best_overall_val_score = float("inf")
            best_overall_ckpt_src: Optional[Path] = None

            for seed in self.seeds:
                rec = self.train_single_seed(
                    model_name=model_name,
                    seed=seed,
                    loaders=loaders,
                    in_features=in_features,
                    feature_cols=feature_cols,
                    target_mean=target_mean,
                    target_scale=target_scale,
                    var_train_dev=var_train_dev,
                    var_dev=var_dev,
                )
                model_runs.append(rec)
                all_run_records.append(rec)

                if float(rec["best_val_score"]) < best_overall_val_score:
                    best_overall_val_score = float(rec["best_val_score"])
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
                        f"WP03 Pre-training v2 ({self.mode}, D={in_features}, L={self.lookback}, H={self.horizon}, "
                        f"loss={self.loss_fn}, val={self.val_selection_mode}) | RMSE={rmse_mean:.6f}±{rmse_std:.6f}"
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
                "range_ratio_mean": round(
                    float(df_m["test_range_ratio"].to_numpy().mean()), 4
                ),
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
            if "test_spike_mae_ugm3" in df_m.columns:
                summary_row["spike_mae_ugm3_mean"] = round(
                    float(df_m["test_spike_mae_ugm3"].to_numpy().mean()), 2
                )
                summary_row["spike_rmse_ugm3_mean"] = round(
                    float(df_m["test_spike_rmse_ugm3"].to_numpy().mean()), 2
                )
            if "test_peak_capture_ratio" in df_m.columns:
                summary_row["peak_capture_ratio_mean"] = round(
                    float(df_m["test_peak_capture_ratio"].to_numpy().mean()), 4
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
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--train-dev-stride", type=int, default=None)
    parser.add_argument("--dev-stride", type=int, default=None)
    parser.add_argument("--test-stride", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dlinear-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--loss-fn", type=str, default="PeakShapeLoss")
    parser.add_argument("--huber-delta", type=float, default=2.5)
    parser.add_argument("--peak-threshold-z", type=float, default=0.84)
    parser.add_argument("--peak-alpha", type=float, default=0.15)
    parser.add_argument("--diff-weight", type=float, default=0.05)
    parser.add_argument("--peak-max-weight", type=float, default=3.0)
    parser.add_argument(
        "--use-residual-anchor",
        action="store_true",
        default=False,
        help="Bật ResidualAnchorWrapper (chỉ dùng cho ablation so sánh bản cũ; mặc định Tắt).",
    )
    parser.add_argument(
        "--val-selection-mode",
        type=str,
        default="balanced_dev",
        choices=["balanced_dev", "dev", "train_dev"],
        help="Tiêu chí chọn checkpoint / Early Stopping (mặc định balanced_dev cân bằng Đông & Hè).",
    )
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
    model_configs = load_model_configs(args.config)
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
        peak_threshold_z=args.peak_threshold_z,
        peak_alpha=args.peak_alpha,
        diff_weight=args.diff_weight,
        peak_max_weight=args.peak_max_weight,
        use_residual_anchor=args.use_residual_anchor,
        val_selection_mode=args.val_selection_mode,
        model_configs=model_configs,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        max_grad_norm=args.max_grad_norm,
        seeds=args.seeds,
    )
    pipeline.run(
        models=args.models,
        include_wp02_baseline=args.include_wp02_baseline,
    )
