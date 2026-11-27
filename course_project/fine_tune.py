"""
T0304 — Fine-Tuning mô hình Pre-trained cho từng trạm quan trắc PM2.5 Bắc Kinh:
- Tích hợp đọc cả checkpoint của WP03 (`outputs/checkpoints/global/`) lẫn
  checkpoint Baseline của WP02 (`train_baseline.py` -> `checkpoints/`).
- Sửa dứt điểm lỗi khớp tên lớp đầu ra (`linear_projection` của LSTM, `projection_pm25`/
  `linear_*` của DLinear, `decoder_layers`/`horizon_queries`/`projection` của Informer,
  `head_linear`/`head_mlp`/`exo_fusion`/`revin` của PatchTST).
- Hỗ trợ Discriminative Learning Rate (`backbone_lr_ratio`) khi mở khóa backbone và
  Decoupled AdamW Weight Decay.
- Chọn checkpoint và Early Stopping cân bằng mùa (`balanced_dev`: 50% `train_dev` Đông-Xuân
  + 50% `dev` Hè) để khắc phục tình trạng fine-tune trên Hè làm giảm chất lượng trên Đông.
- Đánh giá chuẩn hóa thông qua `MetricsCalculator` từ `utils_metrics.py` kèm chẩn đoán đỉnh.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data_pipeline.dataset import make_sliding_windows
from src.data_pipeline.preprocess import TARGET_COL
from training_loop import (
    ALL_SPLITS,
    ALL_SUPPORTED_MODELS,
    DEFAULT_MODELS,
    apply_yaml_config,
    build_criterion,
    build_model,
    compute_loader_target_variance,
    compute_selection_score,
    evaluate,
    load_and_prepare_wp01_splits,
    load_model_configs,
    set_seed,
    train_one_epoch,
)
from utils_metrics import MetricsCalculator


def load_and_scale_with_wp01(
    data_dir: str | Path,
    mode: str,
    scaler_path: Path,
) -> Tuple[Dict[str, pd.DataFrame], StandardScaler, List[str]]:
    """Nạp 4 tập từ WP01 và đảm bảo không chuẩn hóa 2 lần."""
    return load_and_prepare_wp01_splits(
        data_dir=data_dir,
        mode=mode,
        scaler_save_path=scaler_path,
    )


def build_single_station_loaders(
    scaled_splits: Dict[str, pd.DataFrame],
    station: str,
    feature_cols: List[str],
    lookback: int = 96,
    horizon: int = 24,
    stride: int = 2,
    batch_size: int = 256,
    use_pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """Tạo đầy đủ 4 DataLoader (train, train_dev, dev, test) cho riêng 1 trạm từ dữ liệu WP01."""
    num_workers = 4 if use_pin_memory else 0
    loaders: Dict[str, DataLoader] = {}

    for split_name in ALL_SPLITS:
        st_df = scaled_splits[split_name]
        st_df = st_df[st_df["station"] == station].reset_index(drop=True)
        current_stride = stride if split_name == "train" else 1

        X, y = make_sliding_windows(
            st_df,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            lookback=lookback,
            horizon=horizon,
            stride=current_stride,
        )
        if len(X) == 0:
            raise ValueError(
                f"Trạm '{station}' không đủ dữ liệu tạo cửa sổ cho tập '{split_name}' (L={lookback}, H={horizon})."
            )
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            drop_last=False,
            pin_memory=use_pin_memory,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )
    return loaders


def _get_head_keywords(model_name: str) -> Tuple[str, ...]:
    """Trả về tập từ khóa khớp chính xác các lớp đầu ra / adapter thích nghi theo trạm."""
    return {
        "lstm": (
            "linear_projection",
            "fc",
            "head",
            "output_projection",
            "weight_ih_l1",
            "weight_hh_l1",
            "bias_ih_l1",
            "bias_hh_l1",
            "affine_weight",
            "affine_bias",
            "res_scale",
            "station_bias",
        ),
        "dlinear": (
            "linear_seasonal",
            "linear_trend",
            "projection_pm25",
            "res_scale",
            "station_bias",
        ),
        "informer": (
            "projection",
            "decoder_layers",
            "decoder_norm",
            "horizon_queries",
            "affine_weight",
            "affine_bias",
            "head",
            "res_scale",
            "station_bias",
        ),
        "informer_full": (
            "projection",
            "decoder_layers",
            "decoder_norm",
            "horizon_queries",
            "affine_weight",
            "affine_bias",
            "head",
            "res_scale",
            "station_bias",
        ),
        "patchtst": (
            "head_linear",
            "head_mlp",
            "head_dropout",
            "temporal_head",
            "exo_fusion",
            "channel_mixer",
            "revin",
            "res_scale",
            "station_bias",
        ),
    }.get(
        model_name.strip().lower(),
        ("head", "fc", "projection", "linear", "revin", "res_scale", "station_bias"),
    )


def configure_trainable_layers(
    model: nn.Module,
    model_name: str,
    freeze_backbone: bool = True,
    use_wrappers: bool = True,
) -> int:
    """
    Cấu hình các lớp được phép cập nhật khi Fine-tuning:
    - Khớp chính xác tên lớp thực tế của cả 4 kiến trúc (`LSTM`, `DLinear`, `Informer`, `PatchTST`).
    - Trả về tổng số tham số scalar có `requires_grad=True`.
    """
    del use_wrappers
    for param in model.parameters():
        param.requires_grad = True

    if not freeze_backbone:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    for param in model.parameters():
        param.requires_grad = False

    head_keywords = _get_head_keywords(model_name)
    unfrozen_tensors = 0
    for param_name, param in model.named_parameters():
        if any(key in param_name.lower() for key in head_keywords):
            param.requires_grad = True
            unfrozen_tensors += 1

    if unfrozen_tensors == 0:
        for param in model.parameters():
            param.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_finetune_param_groups(
    model: nn.Module,
    model_name: str,
    lr: float,
    weight_decay: float,
    freeze_backbone: bool = True,
    backbone_lr_ratio: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Tạo nhóm tham số AdamW cho Fine-tuning với Decoupled Weight Decay và
    Discriminative LR (head học tốc độ `lr`, backbone học `lr * backbone_lr_ratio` khi mở khóa).
    """
    head_keywords = _get_head_keywords(model_name)
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
    head_decay: List[nn.Parameter] = []
    head_no_decay: List[nn.Parameter] = []
    bb_decay: List[nn.Parameter] = []
    bb_no_decay: List[nn.Parameter] = []

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = param_name.lower()
        is_no_decay = param.ndim < 2 or any(tok in lower for tok in no_decay_tokens)
        is_head = freeze_backbone or any(key in lower for key in head_keywords)
        if is_head:
            (head_no_decay if is_no_decay else head_decay).append(param)
        else:
            (bb_no_decay if is_no_decay else bb_decay).append(param)

    groups: List[Dict[str, Any]] = []
    if head_decay:
        groups.append({"params": head_decay, "lr": lr, "weight_decay": weight_decay})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": lr, "weight_decay": 0.0})
    bb_lr = lr * max(0.01, float(backbone_lr_ratio))
    if bb_decay:
        groups.append({"params": bb_decay, "lr": bb_lr, "weight_decay": weight_decay})
    if bb_no_decay:
        groups.append({"params": bb_no_decay, "lr": bb_lr, "weight_decay": 0.0})
    return groups


def extract_state_dict(
    ckpt_path: Path, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object], bool, bool]:
    """
    Trích xuất an toàn `(model_state_dict, metadata, use_wrappers, use_residual_anchor)`.
    Tự động nhận diện cả checkpoint mới v2 (`use_residual_anchor=False`), checkpoint WP03 cũ
    (có `res_scale`) lẫn checkpoint WP02 Baseline (`use_wrappers=False`).
    """
    raw_obj = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(raw_obj, dict) and "model_state_dict" in raw_obj:
        meta = {k: v for k, v in raw_obj.items() if k != "model_state_dict"}
        state_dict = raw_obj["model_state_dict"]
        use_wrappers = bool(meta.get("use_wrappers", True))
        if "use_residual_anchor" in meta:
            use_residual_anchor = bool(meta["use_residual_anchor"])
        else:
            use_residual_anchor = any(
                k in {"res_scale", "station_bias"} for k in state_dict.keys()
            )
        return state_dict, meta, use_wrappers, use_residual_anchor

    # WP02 BaselineTrainer lưu trực tiếp model.state_dict()
    state_dict = raw_obj
    has_wrapper_keys = any(
        k.startswith(("model.", "backbone.", "res_scale", "station_bias"))
        for k in state_dict.keys()
    )
    has_res_anchor = any(k in {"res_scale", "station_bias"} for k in state_dict.keys())
    return state_dict, {}, has_wrapper_keys, has_res_anchor


def resolve_pretrained_checkpoint(
    model_name: str,
    mode: str,
    horizon: int,
    seed: int,
    ckpt_source: Literal["wp03", "wp02"],
    global_ckpt_dir: Path,
    wp02_ckpt_dir: Path,
) -> Path:
    """Tìm đường dẫn checkpoint nguồn từ WP03 hoặc WP02 Baseline."""
    if ckpt_source == "wp03":
        return global_ckpt_dir / f"{model_name}_{mode}_H{horizon}_pretrained.pt"

    candidates = [
        wp02_ckpt_dir / f"{model_name.lower()}_h{horizon}_seed{seed}_best.pth",
        Path("checkpoints") / f"{model_name.lower()}_h{horizon}_seed{seed}_best.pth",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def finetune_single_station(
    model_name: str,
    station: str,
    station_loaders: Dict[str, DataLoader],
    pretrained_ckpt: str | Path,
    in_features: int,
    feature_cols: Optional[List[str]] = None,
    lookback: int = 96,
    horizon: int = 24,
    mode: str = "multivariate",
    epochs: int = 15,
    lr: float = 5e-5,
    backbone_lr_ratio: float = 0.1,
    weight_decay: float = 1e-2,
    patience: int = 5,
    loss_fn: str = "PeakShapeLoss",
    huber_delta: float = 2.5,
    peak_threshold_z: float = 0.84,
    peak_alpha: float = 0.25,
    diff_weight: float = 0.10,
    peak_max_weight: float = 3.5,
    val_selection_mode: str = "balanced_dev",
    freeze_backbone: bool = True,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    target_mean: Optional[float] = None,
    target_scale: Optional[float] = None,
    station_ckpt_dir: str | Path = "outputs/checkpoints/stations",
    log_dir: str | Path = "outputs/logs",
    seed: int = 42,
    device: Optional[torch.device] = None,
    metrics_calc: Optional[MetricsCalculator] = None,
) -> Dict[str, object]:
    """
    Nạp trọng số nền (từ WP03 hoặc WP02), đo Zero-shot trên cả 4 tập,
    Fine-tune trên 1 trạm với chọn checkpoint cân bằng mùa và đo lại cả 4 tập qua `MetricsCalculator`.
    """
    set_seed(seed)
    calc = metrics_calc or MetricsCalculator()
    pretrained_ckpt = Path(pretrained_ckpt)
    if not pretrained_ckpt.exists():
        raise FileNotFoundError(
            f"Không tìm thấy trọng số Pre-trained tại: {pretrained_ckpt}. "
            "Hãy chạy training_loop.py (WP03) hoặc train_baseline.py (WP02) trước."
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, ckpt_meta, use_wrappers, use_residual_anchor = extract_state_dict(
        pretrained_ckpt, device
    )
    source_tag = "wp03" if use_wrappers else "wp02"

    out_dir = Path(station_ckpt_dir) / station
    out_dir.mkdir(parents=True, exist_ok=True)
    ft_ckpt_path = out_dir / f"{model_name}_{source_tag}_{mode}_H{horizon}_finetuned.pt"

    ft_log_dir = Path(log_dir)
    ft_log_dir.mkdir(parents=True, exist_ok=True)
    ft_log_csv = (
        ft_log_dir
        / f"finetune_{model_name}_{source_tag}_{station}_{mode}_H{horizon}.csv"
    )

    ckpt_in_features = int(ckpt_meta.get("in_features", in_features))
    ckpt_dropout = float(ckpt_meta.get("dropout", 0.20))
    if ckpt_in_features != in_features:
        raise ValueError(
            f"Lệch số đặc trưng đầu vào giữa dữ liệu hiện tại ({in_features}) và "
            f"Checkpoint {pretrained_ckpt.name} ({ckpt_in_features})."
        )
    if feature_cols is not None and "feature_cols" in ckpt_meta:
        ckpt_cols = list(ckpt_meta["feature_cols"])
        if ckpt_cols != list(feature_cols):
            raise ValueError(
                f"Lệch thứ tự cột đặc trưng giữa dữ liệu hiện tại và Checkpoint {pretrained_ckpt.name}."
            )

    ckpt_model_kwargs = (
        dict(ckpt_meta["model_kwargs"])
        if isinstance(ckpt_meta.get("model_kwargs"), dict)
        else (dict(model_configs.get(model_name.lower(), {})) if model_configs else {})
    )
    # Tự động nhận diện checkpoint cũ/mới có bọc RevINModelWrapper (tiền tố backbone.) hay PatchTST RevIN nội tại
    has_revin_wrapper = any(
        k.startswith(("backbone.", "model.backbone.")) for k in state_dict.keys()
    )
    if "use_revin_wrapper" in ckpt_meta:
        ckpt_model_kwargs["use_revin_wrapper"] = bool(ckpt_meta["use_revin_wrapper"])
    elif has_revin_wrapper:
        ckpt_model_kwargs["use_revin_wrapper"] = True
    else:
        ckpt_model_kwargs.setdefault("use_revin_wrapper", False)

    if model_name.strip().lower() == "patchtst":
        has_internal_revin = any(
            k.startswith(("revin.", "model.revin.")) for k in state_dict.keys()
        )
        ckpt_model_kwargs["use_revin"] = has_internal_revin

    model = build_model(
        model_name=model_name,
        lookback=lookback,
        horizon=horizon,
        in_features=in_features,
        device=device,
        dropout=ckpt_dropout,
        use_wrappers=use_wrappers,
        use_residual_anchor=use_residual_anchor,
        use_revin_wrapper=bool(ckpt_model_kwargs["use_revin_wrapper"]),
        model_kwargs=ckpt_model_kwargs,
    )
    model.load_state_dict(state_dict)

    train_criterion = build_criterion(
        loss_fn=loss_fn,
        huber_delta=huber_delta,
        peak_threshold_z=peak_threshold_z,
        peak_alpha=peak_alpha,
        diff_weight=diff_weight,
        peak_max_weight=peak_max_weight,
    )
    eval_criterion = nn.MSELoss()

    var_train_dev = compute_loader_target_variance(station_loaders["train_dev"])
    var_dev = compute_loader_target_variance(station_loaders["dev"])

    # Đo Zero-shot (Trước fine-tune) trên đầy đủ cả 4 tập
    before_train = evaluate(
        model, station_loaders["train"], eval_criterion, device, calc
    )
    before_train_dev = evaluate(
        model, station_loaders["train_dev"], eval_criterion, device, calc
    )
    before_dev = evaluate(model, station_loaders["dev"], eval_criterion, device, calc)
    before_test = evaluate(
        model,
        station_loaders["test"],
        eval_criterion,
        device,
        calc,
        target_mean=target_mean,
        target_scale=target_scale,
    )

    best_val_score = compute_selection_score(
        train_dev_mse=before_train_dev["mse"],
        dev_mse=before_dev["mse"],
        var_train_dev=var_train_dev,
        var_dev=var_dev,
        mode=val_selection_mode,
    )
    best_dev_loss = before_dev["loss"]
    best_epoch = 0

    def _save_station_ckpt(
        epoch_idx: int, dev_loss_val: float, val_score_val: float
    ) -> None:
        payload: Dict[str, object] = {
            "ckpt_source": source_tag,
            "use_wrappers": use_wrappers,
            "use_residual_anchor": use_residual_anchor,
            "use_revin_wrapper": bool(ckpt_model_kwargs.get("use_revin_wrapper", False)),
            "val_selection_mode": val_selection_mode,
            "loss_fn": loss_fn,
            "model_name": model_name,
            "model_kwargs": ckpt_model_kwargs,
            "station": station,
            "mode": mode,
            "lookback": lookback,
            "horizon": horizon,
            "in_features": in_features,
            "dropout": ckpt_dropout,
            "seed": seed,
            "best_epoch": epoch_idx,
            "best_dev_loss": dev_loss_val,
            "best_val_score": val_score_val,
            "model_state_dict": model.state_dict(),
        }
        if feature_cols is not None:
            payload["feature_cols"] = list(feature_cols)
        elif "feature_cols" in ckpt_meta:
            payload["feature_cols"] = list(ckpt_meta["feature_cols"])
        torch.save(payload, ft_ckpt_path)

    _save_station_ckpt(
        epoch_idx=0, dev_loss_val=best_dev_loss, val_score_val=best_val_score
    )

    unfrozen_param_count = configure_trainable_layers(
        model,
        model_name=model_name,
        freeze_backbone=freeze_backbone,
        use_wrappers=use_wrappers,
    )
    param_groups = build_finetune_param_groups(
        model=model,
        model_name=model_name,
        lr=lr,
        weight_decay=weight_decay,
        freeze_backbone=freeze_backbone,
        backbone_lr_ratio=backbone_lr_ratio,
    )
    optimizer = torch.optim.AdamW(param_groups)

    warmup_steps = max(1, len(station_loaders["train"]))
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)),
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()

    history_rows = [
        {
            "epoch": 0,
            "train_loss": None,
            "train_mse": round(before_train["mse"], 6),
            "train_rmse": round(before_train["rmse"], 6),
            "train_dev_mse": round(before_train_dev["mse"], 6),
            "train_dev_rmse": round(before_train_dev["rmse"], 6),
            "dev_mse": round(before_dev["mse"], 6),
            "dev_rmse": round(before_dev["rmse"], 6),
            "val_score": round(best_val_score, 6),
            "lr": lr,
        }
    ]

    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        step_sched = warmup_scheduler if epoch == 1 else None
        train_loss = train_one_epoch(
            model=model,
            dataloader=station_loaders["train"],
            criterion=train_criterion,
            optimizer=optimizer,
            device=device,
            scaler=grad_scaler,
            scheduler=step_sched,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        train_dev_metrics = evaluate(
            model, station_loaders["train_dev"], eval_criterion, device, calc
        )
        dev_metrics = evaluate(
            model, station_loaders["dev"], eval_criterion, device, calc
        )
        val_score = compute_selection_score(
            train_dev_mse=train_dev_metrics["mse"],
            dev_mse=dev_metrics["mse"],
            var_train_dev=var_train_dev,
            var_dev=var_dev,
            mode=val_selection_mode,
        )
        if epoch > 1:
            plateau_scheduler.step(val_score)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_mse": None,
                "train_rmse": None,
                "train_dev_mse": round(train_dev_metrics["mse"], 6),
                "train_dev_rmse": round(train_dev_metrics["rmse"], 6),
                "dev_mse": round(dev_metrics["mse"], 6),
                "dev_rmse": round(dev_metrics["rmse"], 6),
                "val_score": round(val_score, 6),
                "lr": current_lr,
            }
        )

        if val_score < best_val_score:
            best_val_score = val_score
            best_dev_loss = dev_metrics["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            _save_station_ckpt(
                epoch_idx=best_epoch,
                dev_loss_val=best_dev_loss,
                val_score_val=best_val_score,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    else:
        peak_vram_mb = 0.0
    finetune_time_sec = time.perf_counter() - t_start

    pd.DataFrame(history_rows).to_csv(ft_log_csv, index=False)

    # Nạp lại checkpoint tốt nhất và đánh giá đầy đủ trên cả 4 tập
    best_state_dict, _, _, _ = extract_state_dict(ft_ckpt_path, device)
    model.load_state_dict(best_state_dict)

    after_train = evaluate(
        model, station_loaders["train"], eval_criterion, device, calc
    )
    after_train_dev = evaluate(
        model, station_loaders["train_dev"], eval_criterion, device, calc
    )
    after_dev = evaluate(model, station_loaders["dev"], eval_criterion, device, calc)
    after_test = evaluate(
        model,
        station_loaders["test"],
        eval_criterion,
        device,
        calc,
        target_mean=target_mean,
        target_scale=target_scale,
    )

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
        f"  [{source_tag.upper()} | {model_name:<13} | {station:<14}] "
        f"Test MSE: {before_test['mse']:.4f} -> {after_test['mse']:.4f} ({mse_gain_pct:+.2f}%) | "
        f"RMSE: {before_test['rmse']:.4f} -> {after_test['rmse']:.4f} ({rmse_gain_pct:+.2f}%) | "
        f"MAE: {before_test['mae']:.4f} -> {after_test['mae']:.4f} ({mae_gain_pct:+.2f}%) "
        f"[unfrozen={unfrozen_param_count:,}, best_ep={best_epoch}, {finetune_time_sec:.2f}s]"
    )

    del model, optimizer, warmup_scheduler, plateau_scheduler, grad_scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rec: Dict[str, object] = {
        "ckpt_source": source_tag,
        "model": model_name,
        "station": station,
        "mode": mode,
        "horizon": horizon,
        "freeze_backbone": freeze_backbone,
        "unfrozen_params": unfrozen_param_count,
        "best_epoch": best_epoch,
        "finetune_time_sec": round(finetune_time_sec, 3),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "before_train_mse": round(before_train["mse"], 6),
        "after_train_mse": round(after_train["mse"], 6),
        "before_train_rmse": round(before_train["rmse"], 6),
        "after_train_rmse": round(after_train["rmse"], 6),
        "before_train_dev_mse": round(before_train_dev["mse"], 6),
        "after_train_dev_mse": round(after_train_dev["mse"], 6),
        "before_train_dev_rmse": round(before_train_dev["rmse"], 6),
        "after_train_dev_rmse": round(after_train_dev["rmse"], 6),
        "before_dev_mse": round(before_dev["mse"], 6),
        "after_dev_mse": round(after_dev["mse"], 6),
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
        "before_range_ratio": round(before_test["range_ratio"], 4),
        "after_range_ratio": round(after_test["range_ratio"], 4),
        "improved": bool(after_test["mse"] < before_test["mse"]),
        "checkpoint": str(ft_ckpt_path),
    }
    if "peak_capture_ratio" in after_test and not math.isnan(
        after_test["peak_capture_ratio"]
    ):
        rec["before_peak_capture_ratio"] = round(before_test["peak_capture_ratio"], 4)
        rec["after_peak_capture_ratio"] = round(after_test["peak_capture_ratio"], 4)
    return rec


def run_fine_tuning_pipeline(
    data_dir: str | Path = "data/preprocess",
    models: Sequence[str] = DEFAULT_MODELS,
    stations: Optional[Sequence[str]] = None,
    mode: Literal["multivariate", "univariate"] = "multivariate",
    lookback: int = 96,
    horizon: int = 24,
    stride: int = 2,
    batch_size: int = 256,
    epochs: int = 15,
    lr: float = 5e-5,
    backbone_lr_ratio: float = 0.1,
    weight_decay: float = 1e-2,
    patience: int = 5,
    loss_fn: str = "PeakShapeLoss",
    huber_delta: float = 2.5,
    peak_threshold_z: float = 0.84,
    peak_alpha: float = 0.25,
    diff_weight: float = 0.10,
    peak_max_weight: float = 3.5,
    val_selection_mode: str = "balanced_dev",
    freeze_backbone: bool = True,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ckpt_source: Literal["wp03", "wp02"] = "wp03",
    wp02_ckpt_dir: str | Path = "outputs/checkpoints/wp02_baseline",
    output_root: str | Path = "outputs",
    seed: int = 42,
) -> Dict[str, object]:
    """Chạy Fine-tuning toàn bộ các mô hình trên từng trạm (hỗ trợ cả nguồn WP03 và WP02)."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    metrics_calc = MetricsCalculator()

    output_root = Path(output_root)
    global_ckpt_dir = output_root / "checkpoints" / "global"
    station_ckpt_dir = output_root / "checkpoints" / "stations"
    log_dir = output_root / "logs"
    report_dir = output_root / "reports"
    station_ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = global_ckpt_dir / f"scaler_12stations_{mode}.joblib"
    scaled_splits, scaler, feature_cols = load_and_scale_with_wp01(
        data_dir=data_dir, mode=mode, scaler_path=scaler_path
    )
    target_mean = (
        float(scaler.mean_[0])
        if hasattr(scaler, "mean_") and len(scaler.mean_) > 0
        else None
    )
    target_scale = (
        float(scaler.scale_[0])
        if hasattr(scaler, "scale_") and len(scaler.scale_) > 0
        else None
    )
    in_features = len(feature_cols)

    all_stations = sorted(scaled_splits["train"]["station"].dropna().unique().tolist())
    target_stations = list(stations) if stations is not None else all_stations

    per_station_loaders: Dict[str, Dict[str, DataLoader]] = {
        st: build_single_station_loaders(
            scaled_splits=scaled_splits,
            station=st,
            feature_cols=feature_cols,
            lookback=lookback,
            horizon=horizon,
            stride=stride,
            batch_size=batch_size,
            use_pin_memory=use_pin_memory,
        )
        for st in target_stations
    }

    all_station_records: List[Dict[str, object]] = []
    model_summary_records: List[Dict[str, object]] = []

    for model_name in models:
        pretrained_ckpt = resolve_pretrained_checkpoint(
            model_name=model_name,
            mode=mode,
            horizon=horizon,
            seed=seed,
            ckpt_source=ckpt_source,
            global_ckpt_dir=global_ckpt_dir,
            wp02_ckpt_dir=Path(wp02_ckpt_dir),
        )
        print("\n" + "=" * 88)
        print(
            f"[FINE-TUNE TỪNG TRẠM] Source={ckpt_source.upper()} | Model={model_name.upper()} | "
            f"Mode={mode} (D={in_features}) | H={horizon} | ValSel={val_selection_mode} | Số trạm={len(target_stations)}"
        )
        print("=" * 88)

        model_records: List[Dict[str, object]] = []
        for st in target_stations:
            rec = finetune_single_station(
                model_name=model_name,
                station=st,
                station_loaders=per_station_loaders[st],
                pretrained_ckpt=pretrained_ckpt,
                in_features=in_features,
                feature_cols=feature_cols,
                lookback=lookback,
                horizon=horizon,
                mode=mode,
                epochs=epochs,
                lr=lr,
                backbone_lr_ratio=backbone_lr_ratio,
                weight_decay=weight_decay,
                patience=patience,
                loss_fn=loss_fn,
                huber_delta=huber_delta,
                peak_threshold_z=peak_threshold_z,
                peak_alpha=peak_alpha,
                diff_weight=diff_weight,
                peak_max_weight=peak_max_weight,
                val_selection_mode=val_selection_mode,
                freeze_backbone=freeze_backbone,
                model_configs=model_configs,
                target_mean=target_mean,
                target_scale=target_scale,
                station_ckpt_dir=station_ckpt_dir,
                log_dir=log_dir,
                seed=seed,
                device=device,
                metrics_calc=metrics_calc,
            )
            model_records.append(rec)
            all_station_records.append(rec)

        df_m = pd.DataFrame(model_records)
        win_count = int(df_m["improved"].to_numpy().sum())
        model_summary_records.append(
            {
                "ckpt_source": ckpt_source,
                "model": model_name,
                "mode": mode,
                "horizon": horizon,
                "freeze_backbone": freeze_backbone,
                "mean_before_train_mse": round(
                    float(df_m["before_train_mse"].to_numpy().mean()), 6
                ),
                "mean_after_train_mse": round(
                    float(df_m["after_train_mse"].to_numpy().mean()), 6
                ),
                "mean_before_train_rmse": round(
                    float(df_m["before_train_rmse"].to_numpy().mean()), 6
                ),
                "mean_after_train_rmse": round(
                    float(df_m["after_train_rmse"].to_numpy().mean()), 6
                ),
                "mean_before_train_dev_mse": round(
                    float(df_m["before_train_dev_mse"].to_numpy().mean()), 6
                ),
                "mean_after_train_dev_mse": round(
                    float(df_m["after_train_dev_mse"].to_numpy().mean()), 6
                ),
                "mean_before_train_dev_rmse": round(
                    float(df_m["before_train_dev_rmse"].to_numpy().mean()), 6
                ),
                "mean_after_train_dev_rmse": round(
                    float(df_m["after_train_dev_rmse"].to_numpy().mean()), 6
                ),
                "mean_before_dev_mse": round(
                    float(df_m["before_dev_mse"].to_numpy().mean()), 6
                ),
                "mean_after_dev_mse": round(
                    float(df_m["after_dev_mse"].to_numpy().mean()), 6
                ),
                "mean_before_dev_rmse": round(
                    float(df_m["before_dev_rmse"].to_numpy().mean()), 6
                ),
                "mean_after_dev_rmse": round(
                    float(df_m["after_dev_rmse"].to_numpy().mean()), 6
                ),
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
                "mean_finetune_time_sec": round(
                    float(df_m["finetune_time_sec"].to_numpy().mean()), 3
                ),
                "total_finetune_time_sec": round(
                    float(df_m["finetune_time_sec"].to_numpy().sum()), 3
                ),
                "peak_vram_max_mb": round(
                    float(df_m["peak_vram_mb"].to_numpy().max()), 2
                ),
                "stations_improved": f"{win_count}/{len(target_stations)}",
            }
        )
        gc.collect()

    suffix = f"_{ckpt_source}" if ckpt_source == "wp02" else ""
    detail_csv = (
        report_dir / f"before_vs_after_by_station{suffix}_{mode}_H{horizon}.csv"
    )
    summary_csv = report_dir / f"before_vs_after_by_model{suffix}_{mode}_H{horizon}.csv"
    summary_json = report_dir / f"before_vs_after_full{suffix}_{mode}_H{horizon}.json"

    pd.DataFrame(all_station_records).to_csv(
        detail_csv, index=False, encoding="utf-8-sig"
    )
    summary_df = pd.DataFrame(model_summary_records)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "ckpt_source": ckpt_source,
                    "mode": mode,
                    "in_features": in_features,
                    "lookback": lookback,
                    "horizon": horizon,
                    "stride": stride,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "backbone_lr_ratio": backbone_lr_ratio,
                    "weight_decay": weight_decay,
                    "loss_fn": loss_fn,
                    "val_selection_mode": val_selection_mode,
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

    print("\n" + "=" * 96)
    print(
        f"TỔNG HỢP TRƯỚC VS. SAU FINE-TUNING (SOURCE={ckpt_source.upper()}, MODE={mode}, H={horizon})"
    )
    print("=" * 96)
    print(summary_df.to_string(index=False))
    print(
        f"\n[SAVED] Chi tiết từng trạm ({len(all_station_records)} dòng): {detail_csv}"
    )
    print(
        f"[SAVED] Tổng hợp theo mô hình ({len(model_summary_records)} dòng): {summary_csv}"
    )

    return {
        "model_summary": model_summary_records,
        "station_details": all_station_records,
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T0304: Fine-tuning mô hình theo từng trạm từ WP03 hoặc WP02 Baseline."
    )
    parser.add_argument("--data-dir", type=str, default="data/preprocess")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=ALL_SUPPORTED_MODELS,
    )
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
    parser.add_argument(
        "--stride", "--train-stride", dest="stride", type=int, default=4
    )
    parser.add_argument("--train-dev-stride", type=int, default=None)
    parser.add_argument("--dev-stride", type=int, default=None)
    parser.add_argument("--test-stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--backbone-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--loss-fn", type=str, default="PeakShapeLoss")
    parser.add_argument("--huber-delta", type=float, default=2.5)
    parser.add_argument("--peak-threshold-z", type=float, default=0.84)
    parser.add_argument("--peak-alpha", type=float, default=0.15)
    parser.add_argument("--diff-weight", type=float, default=0.05)
    parser.add_argument("--peak-max-weight", type=float, default=3.0)
    parser.add_argument(
        "--val-selection-mode",
        type=str,
        default="balanced_dev",
        choices=["balanced_dev", "dev", "train_dev"],
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Mở khóa toàn bộ backbone với Discriminative LR (mặc định mở khóa head + adapter + RevIN).",
    )
    parser.add_argument(
        "--ckpt-source",
        type=str,
        default="wp03",
        choices=["wp03", "wp02"],
        help="Chọn nguồn trọng số khởi tạo: 'wp03' (Global Pretrained) hoặc 'wp02' (BaselineTrainer).",
    )
    parser.add_argument(
        "--wp02-ckpt-dir",
        type=str,
        default="outputs/checkpoints/wp02_baseline",
        help="Thư mục chứa checkpoint của WP02 BaselineTrainer.",
    )
    parser.add_argument(
        "--output-dir",
        "--output-root",
        dest="output_dir",
        type=str,
        default="outputs",
    )
    parser.add_argument("--seed", type=int, default=42)
    apply_yaml_config(parser, ["paths", "data", "finetune"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_configs = load_model_configs(args.config)
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
        backbone_lr_ratio=args.backbone_lr_ratio,
        weight_decay=args.weight_decay,
        patience=args.patience,
        loss_fn=args.loss_fn,
        huber_delta=args.huber_delta,
        peak_threshold_z=args.peak_threshold_z,
        peak_alpha=args.peak_alpha,
        diff_weight=args.diff_weight,
        peak_max_weight=args.peak_max_weight,
        val_selection_mode=args.val_selection_mode,
        freeze_backbone=(not args.unfreeze_backbone),
        model_configs=model_configs,
        ckpt_source=args.ckpt_source,  # type: ignore[arg-type]
        wp02_ckpt_dir=args.wp02_ckpt_dir,
        output_root=args.output_dir,
        seed=args.seed,
    )
