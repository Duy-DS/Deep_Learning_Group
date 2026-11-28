"""
T0305 — Tệp chạy độc lập toàn bộ ma trận thực nghiệm Ablation Studies (RQ1 & RQ2),
tích hợp đối chứng trực tiếp với WP02 BaselineTrainer (`train_baseline.py`) và
xuất hợp đồng dữ liệu hợp nhất `ExperimentMetricsContract` (`utils_metrics.py`).

Bản nâng cấp:
- Nạp đầy đủ mục `models:` và `finetune:` từ `configs/default.yaml` (không còn hardcode
  `freeze_backbone=True`, `patience=4` hay bỏ qua cấu hình mô hình).
- Truyền đồng bộ `PeakShapeLoss`, `val_selection_mode` (`balanced_dev`) và
  `use_residual_anchor` xuống `TrainingPipeline` và `run_fine_tuning_pipeline`.
- Xuất thêm các cột chẩn đoán đỉnh & độ phẳng (`range_ratio_mean`, `spike_mae_ugm3_mean`,
  `peak_capture_ratio_mean`) song song với chuẩn `ExperimentMetricsContract`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml

from fine_tune import run_fine_tuning_pipeline
from training_loop import (
    DEFAULT_MODELS,
    TrainingPipeline,
    apply_yaml_config,
    load_model_configs,
    run_wp02_baseline_pipeline,
)
from utils_metrics import ExperimentMetricsContract

ALL_HORIZONS: List[int] = [24, 48, 96, 192]
LONG_HORIZONS: List[int] = [96, 192]


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Chuyển DataFrame sang bảng Markdown, tự động fallback nếu thiếu gói tabulate."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = [str(c) for c in df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in df.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)


def ensure_pretrain_results(
    data_dir: Path,
    output_dir: Path,
    mode: str,
    models: Sequence[str],
    lookback: int,
    batch_size: int,
    train_stride: int,
    lr: float,
    dlinear_lr: float,
    weight_decay: float,
    dropout: float,
    epochs: int,
    patience: int,
    loss_fn: str,
    huber_delta: float,
    peak_threshold_z: float,
    peak_alpha: float,
    diff_weight: float,
    peak_max_weight: float,
    use_residual_anchor: bool,
    val_selection_mode: str,
    model_configs: Dict[str, Dict[str, Any]],
    seeds: List[int],
) -> pd.DataFrame:
    """
    Đảm bảo đã có đủ kết quả Pre-training cho danh sách `models` ở cấu hình `(mode, horizon)`.
    """
    report_dir = output_dir / "reports"
    summary_csv = report_dir / f"global_pretrain_summary_{mode}_H{horizon}.csv"
    runs_csv = report_dir / f"global_pretrain_runs_{mode}_H{horizon}.csv"

    existing_summary = (
        pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
    )
    existing_runs = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame()

    if (
        skip_existing
        and not existing_summary.empty
        and "model" in existing_summary.columns
    ):
        done_models = set(existing_summary["model"].astype(str).tolist())
        missing_models = [m for m in models if m not in done_models]
    else:
        missing_models = list(models)

    if missing_models:
        print(
            f"\n>>> [ABLATION RUN] Mode={mode} | H={horizon} | "
            f"Huấn luyện các mô hình: {missing_models}"
        )
        pipeline = TrainingPipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            mode=mode,
            lookback=lookback,
            horizon=horizon,
            batch_size=batch_size,
            train_stride=train_stride,
            lr=lr,
            dlinear_lr=dlinear_lr,
            weight_decay=weight_decay,
            dropout=dropout,
            epochs=epochs,
            patience=patience,
            loss_fn=loss_fn,
            huber_delta=huber_delta,
            peak_threshold_z=peak_threshold_z,
            peak_alpha=peak_alpha,
            diff_weight=diff_weight,
            peak_max_weight=peak_max_weight,
            use_residual_anchor=use_residual_anchor,
            val_selection_mode=val_selection_mode,
            model_configs=model_configs,
            seeds=seeds,
        )
        new_runs_df, new_summary_df = pipeline.run(models=missing_models)

        if not existing_summary.empty:
            existing_summary = existing_summary[
                ~existing_summary["model"].isin(missing_models)
            ]
            merged_summary = pd.concat(
                [existing_summary, new_summary_df], ignore_index=True
            )
            merged_summary.to_csv(summary_csv, index=False)
        else:
            merged_summary = new_summary_df

        if not existing_runs.empty:
            existing_runs = existing_runs[~existing_runs["model"].isin(missing_models)]
            merged_runs = pd.concat([existing_runs, new_runs_df], ignore_index=True)
            merged_runs.to_csv(runs_csv, index=False)
    else:
        print(
            f">>> [SKIP EXISTING] Đã có sẵn kết quả Mode={mode} | H={horizon} "
            f"cho {list(models)} tại {summary_csv}"
        )
        merged_summary = existing_summary

    return merged_summary[merged_summary["model"].isin(models)].reset_index(drop=True)


def ensure_wp02_baseline_results(
    data_dir: Path,
    output_dir: Path,
    mode: str,
    horizon: int,
    lookback: int,
    batch_size: int,
    train_stride: int,
    epochs: int,
    seeds: List[int],
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Chạy hoặc nạp kết quả WP02 BaselineTrainer (LSTM, DLinear) trên dữ liệu thật WP01."""
    report_dir = output_dir / "reports"
    base_csv = report_dir / f"wp02_baseline_summary_{mode}_H{horizon}.csv"
    if skip_existing and base_csv.exists():
        print(f">>> [SKIP EXISTING] Nạp kết quả WP02 Baseline tại: {base_csv}")
        return pd.read_csv(base_csv)

    base_df, _ = run_wp02_baseline_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        mode=mode,
        lookback=lookback,
        horizon=horizon,
        batch_size=batch_size,
        train_stride=train_stride,
        epochs=epochs,
        seeds=seeds,
        models=("lstm", "dlinear"),
    )
    return base_df


def ensure_finetune_results(
    data_dir: Path,
    output_dir: Path,
    mode: str,
    horizon: int,
    models: Sequence[str],
    lookback: int,
    stride: int,
    batch_size: int,
    epochs: int,
    lr: float,
    backbone_lr_ratio: float,
    weight_decay: float,
    patience: int,
    loss_fn: str,
    huber_delta: float,
    peak_threshold_z: float,
    peak_alpha: float,
    diff_weight: float,
    peak_max_weight: float,
    val_selection_mode: str,
    freeze_backbone: bool,
    model_configs: Dict[str, Dict[str, Any]],
    seed: int,
    num_stations: Optional[int] = None,
) -> pd.DataFrame:
    """Chạy hoặc nạp kết quả Fine-tuning cho các station IDs trong dữ liệu.

    `num_stations` là số lượng kỳ vọng để kiểm tra; danh sách tên trạm vẫn được
    suy ra động từ split train và xác minh nhất quán trong pipeline Fine-tuning.
    """
    report_dir = output_dir / "reports"
    ft_summary_csv = report_dir / f"before_vs_after_by_model_{mode}_H{horizon}.csv"
    ft_detail_csv = report_dir / f"before_vs_after_by_station_{mode}_H{horizon}.csv"

    existing_ft = (
        pd.read_csv(ft_summary_csv) if ft_summary_csv.exists() else pd.DataFrame()
    )
    existing_detail = (
        pd.read_csv(ft_detail_csv) if ft_detail_csv.exists() else pd.DataFrame()
    )

    if skip_existing and not existing_ft.empty and "model" in existing_ft.columns:
        done_models = set(existing_ft["model"].astype(str).tolist())
        missing_models = [m for m in models if m not in done_models]
    else:
        missing_models = list(models)

    if missing_models:
        res = run_fine_tuning_pipeline(
            data_dir=data_dir,
            models=missing_models,
            num_stations=num_stations,
            mode=mode,  # type: ignore[arg-type]
            lookback=lookback,
            horizon=horizon,
            stride=stride,
            batch_size=batch_size,
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
            output_root=output_dir,
            seed=seed,
        )
        new_ft_df = pd.DataFrame(res["model_summary"])
        new_detail_df = pd.DataFrame(res["station_details"])

        if not existing_ft.empty:
            existing_ft = existing_ft[~existing_ft["model"].isin(missing_models)]
            merged_ft = pd.concat([existing_ft, new_ft_df], ignore_index=True)
            merged_ft.to_csv(ft_summary_csv, index=False, encoding="utf-8-sig")
        else:
            merged_ft = new_ft_df

        if not existing_detail.empty:
            existing_detail = existing_detail[
                ~existing_detail["model"].isin(missing_models)
            ]
            merged_detail = pd.concat(
                [existing_detail, new_detail_df], ignore_index=True
            )
            merged_detail.to_csv(ft_detail_csv, index=False, encoding="utf-8-sig")
    else:
        merged_ft = existing_ft

    return merged_ft[merged_ft["model"].isin(models)].reset_index(drop=True)


def build_wp02_vs_wp03_table(
    wp02_df: pd.DataFrame, wp03_df: pd.DataFrame, horizon: int = 24
) -> pd.DataFrame:
    """
    Đối chứng trực tiếp WP02 (`BaselineTrainer`: Adam lr=1e-3)
    với WP03 (`TrainingPipeline` v2: Selective RevIN + PeakShapeLoss + Balanced Val + Stepwise WarmupCosine).
    """
    rows: List[Dict[str, object]] = []
    for model_name in ("lstm", "dlinear"):
        b_sub = wp02_df[wp02_df["model"] == model_name]
        w_sub = wp03_df[wp03_df["model"] == model_name]
        if b_sub.empty or w_sub.empty:
            continue
        b_row = b_sub.iloc[0]
        w_row = w_sub.iloc[0]

        mse_gain = (
            (float(b_row["mse_mean"]) - float(w_row["mse_mean"]))
            / float(b_row["mse_mean"])
        ) * 100.0
        rmse_gain = (
            (float(b_row["rmse_mean"]) - float(w_row["rmse_mean"]))
            / float(b_row["rmse_mean"])
        ) * 100.0
        mae_gain = (
            (float(b_row["mae_mean"]) - float(w_row["mae_mean"]))
            / float(b_row["mae_mean"])
        ) * 100.0

        rows.append(
            {
                "model": model_name,
                "horizon": horizon,
                "wp02_baseline_mse": b_row["mse_mean_std"],
                "wp03_pretrain_mse": w_row["mse_mean_std"],
                "mse_improvement_pct": round(mse_gain, 2),
                "wp02_baseline_rmse": b_row["rmse_mean_std"],
                "wp03_pretrain_rmse": w_row["rmse_mean_std"],
                "rmse_improvement_pct": round(rmse_gain, 2),
                "wp02_baseline_mae": b_row["mae_mean_std"],
                "wp03_pretrain_mae": w_row["mae_mean_std"],
                "mae_improvement_pct": round(mae_gain, 2),
            }
        )
    return pd.DataFrame(rows)


def build_ablation1_table(
    multi_df: pd.DataFrame, uni_df: pd.DataFrame, horizon: int = 24
) -> pd.DataFrame:
    """
    Ablation 1 (RQ1/RQ3): So sánh Multivariate vs. Univariate
    và tính tỷ lệ giảm sai số (%) nhờ bổ sung biến khí tượng.
    """
    rows: List[Dict[str, object]] = []
    for model_name in DEFAULT_MODELS:
        m_row = multi_df[multi_df["model"] == model_name].iloc[0]
        u_row = uni_df[uni_df["model"] == model_name].iloc[0]

        mse_gain = (
            (float(u_row["mse_mean"]) - float(m_row["mse_mean"]))
            / float(u_row["mse_mean"])
        ) * 100.0
        rmse_gain = (
            (float(u_row["rmse_mean"]) - float(m_row["rmse_mean"]))
            / float(u_row["rmse_mean"])
        ) * 100.0
        mae_gain = (
            (float(u_row["mae_mean"]) - float(m_row["mae_mean"]))
            / float(u_row["mae_mean"])
        ) * 100.0

        rows.append(
            {
                "model": model_name,
                "horizon": horizon,
                "univariate_mse": u_row["mse_mean_std"],
                "multivariate_mse": m_row["mse_mean_std"],
                "mse_reduction_pct": round(mse_gain, 2),
                "univariate_rmse": u_row["rmse_mean_std"],
                "multivariate_rmse": m_row["rmse_mean_std"],
                "rmse_reduction_pct": round(rmse_gain, 2),
                "univariate_mae": u_row["mae_mean_std"],
                "multivariate_mae": m_row["mae_mean_std"],
                "mae_reduction_pct": round(mae_gain, 2),
            }
        )
    return pd.DataFrame(rows)


def build_ablation2_table(horizon_dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Ablation 2 (RQ1/RQ2): Tổng hợp 4 mô hình trên đầy đủ 4 khung dự báo H in {24, 48, 96, 192}.
    """
    frames: List[pd.DataFrame] = []
    for h in sorted(horizon_dfs.keys()):
        df_h = horizon_dfs[h].copy()
        cols = [
            "horizon",
            "model",
            "trainable_params",
            "mse_mean",
            "mse_std",
            "mse_mean_std",
            "rmse_mean",
            "rmse_std",
            "rmse_mean_std",
            "mae_mean",
            "mae_std",
            "mae_mean_std",
            "range_ratio_mean",
            "spike_mae_ugm3_mean",
            "peak_capture_ratio_mean",
            "train_time_mean_sec",
            "peak_vram_max_mb",
        ]
        available_cols = [c for c in cols if c in df_h.columns]
        frames.append(df_h[available_cols])

    long_df = pd.concat(frames, ignore_index=True)
    order_map = {m: idx for idx, m in enumerate(DEFAULT_MODELS)}
    long_df["_order"] = long_df["model"].map(order_map).fillna(99)
    long_df = long_df.sort_values(["horizon", "_order"]).drop(columns=["_order"])
    return long_df.reset_index(drop=True)


def build_ablation3_table(long_horizon_dfs: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Ablation 3 (RQ2): So sánh Informer (ProbSparse) vs. Informer_FullAttention O(L^2)
    trên khung dự báo dài (H=96, H=192) về độ chính xác, tốc độ và VRAM đỉnh.
    """
    rows: List[Dict[str, object]] = []
    for h in sorted(long_horizon_dfs.keys()):
        df_h = long_horizon_dfs[h]
        sparse_row = df_h[df_h["model"] == "informer"].iloc[0]
        full_row = df_h[df_h["model"] == "informer_full"].iloc[0]

        t_sparse = float(sparse_row["train_time_mean_sec"])
        t_full = float(full_row["train_time_mean_sec"])
        speedup = (t_full / t_sparse) if t_sparse > 0 else 1.0

        vram_sparse = float(sparse_row["peak_vram_max_mb"])
        vram_full = float(full_row["peak_vram_max_mb"])
        vram_saved_pct = (
            ((vram_full - vram_sparse) / vram_full) * 100.0 if vram_full > 0 else 0.0
        )

        rmse_sparse = float(sparse_row["rmse_mean"])
        rmse_full = float(full_row["rmse_mean"])
        rmse_diff_pct = (
            ((rmse_full - rmse_sparse) / rmse_full) * 100.0 if rmse_full > 0 else 0.0
        )

        rows.append(
            {
                "horizon": h,
                "probsparse_mse": sparse_row["mse_mean_std"],
                "full_attn_mse": full_row["mse_mean_std"],
                "probsparse_rmse": sparse_row["rmse_mean_std"],
                "full_attn_rmse": full_row["rmse_mean_std"],
                "rmse_reduction_pct": round(rmse_diff_pct, 2),
                "probsparse_mae": sparse_row["mae_mean_std"],
                "full_attn_mae": full_row["mae_mean_std"],
                "probsparse_time_sec": round(t_sparse, 2),
                "full_attn_time_sec": round(t_full, 2),
                "speedup_ratio": f"{speedup:.2f}x",
                "probsparse_vram_mb": round(vram_sparse, 2),
                "full_attn_vram_mb": round(vram_full, 2),
                "vram_saved_pct": round(vram_saved_pct, 2),
            }
        )
    return pd.DataFrame(rows)


def aggregate_all_contracts(report_dir: Path, ablation_dir: Path) -> Optional[Path]:
    """
    Gom toàn bộ tệp JSON `ExperimentMetricsContract` của WP02 và WP03 thành một tệp
    hợp nhất `ablation_contracts.json` phục vụ trực tiếp WP05 (Dashboard).
    """
    contract_files = sorted(report_dir.glob("*_contracts_*.json"))
    all_contracts: List[Dict[str, object]] = []
    seen_ids = set()

    for fpath in contract_files:
        items = json.loads(fpath.read_text(encoding="utf-8"))
        for item in items:
            validated = ExperimentMetricsContract.model_validate(item)
            if validated.experiment_id not in seen_ids:
                seen_ids.add(validated.experiment_id)
                all_contracts.append(validated.model_dump())

    if not all_contracts:
        return None

    out_json = ablation_dir / "ablation_contracts.json"
    out_json.write_text(
        json.dumps(all_contracts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_json


def export_markdown_report(
    ablation_dir: Path,
    wp02_vs_wp03_df: Optional[pd.DataFrame],
    ab1_df: Optional[pd.DataFrame],
    ab2_df: Optional[pd.DataFrame],
    ab3_df: Optional[pd.DataFrame],
    ft_ablation_df: Optional[pd.DataFrame] = None,
    multi_dim: int = 31,
) -> Path:
    """Xuất toàn bộ các bảng đối chứng WP02-WP03 và Ablation sang Markdown."""
    md_path = ablation_dir / "ablation_report_summary.md"
    sections: List[str] = [
        "# TỔNG HỢP KẾT QUẢ THỰC NGHIỆM & ABLATION STUDIES — DỰ BÁO PM2.5 BẮC KINH\n"
    ]

    if wp02_vs_wp03_df is not None and not wp02_vs_wp03_df.empty:
        sections.append(
            "## 0. Đối chứng WP02 (`BaselineTrainer`) vs. WP03 (`TrainingPipeline` v2) tại $H=24$\n"
        )
        sections.append(_df_to_markdown(wp02_vs_wp03_df) + "\n")

    if ab1_df is not None and not ab1_df.empty:
        sections.append(
            f"## 1. Ablation 1 (RQ1): Multivariate ($D={multi_dim}$) vs. Univariate ($D=1$) tại $H=24$\n"
        )
        sections.append(_df_to_markdown(ab1_df) + "\n")

    if ab2_df is not None and not ab2_df.empty:
        sections.append(
            "## 2. Ablation 2 (RQ2): Đánh giá 4 mô hình trên 4 khung dự báo $H \\in \\{24, 48, 96, 192\\}$\n"
        )
        preferred_cols = [
            "horizon",
            "model",
            "mse_mean_std",
            "rmse_mean_std",
            "mae_mean_std",
            "range_ratio_mean",
            "spike_mae_ugm3_mean",
            "peak_capture_ratio_mean",
            "train_time_mean_sec",
            "peak_vram_max_mb",
        ]
        display_cols = [c for c in preferred_cols if c in ab2_df.columns]
        sections.append(_df_to_markdown(ab2_df[display_cols]) + "\n")

    if ab3_df is not None and not ab3_df.empty:
        sections.append(
            "## 3. Ablation 3 (RQ2): Informer (`ProbSparse`) vs. `Informer_FullAttention` ($O(L^2)$)\n"
        )
        sections.append(_df_to_markdown(ab3_df) + "\n")

    if ft_ablation_df is not None and not ft_ablation_df.empty:
        sections.append(
            "## 4. Tổng hợp Fine-Tuning (Trước vs. Sau) qua các cấu hình Ablation\n"
        )
        sections.append(_df_to_markdown(ft_ablation_df) + "\n")

    md_path.write_text("\n".join(sections), encoding="utf-8")
    return md_path


def run_ablation_suite(args: argparse.Namespace) -> None:
    """Điều phối toàn bộ quy trình Ablation Studies và đối chứng WP02."""
    ablation_dir = args.output_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.output_dir / "reports"
    model_configs = load_model_configs(args.config)

    run_all = "all" in args.ablations
    do_ab1 = run_all or ("ablation1" in args.ablations)
    do_ab2 = run_all or ("ablation2" in args.ablations)
    do_ab3 = run_all or ("ablation3" in args.ablations)

    wp02_vs_wp03_df: Optional[pd.DataFrame] = None
    ab1_df: Optional[pd.DataFrame] = None
    ab2_df: Optional[pd.DataFrame] = None
    ab3_df: Optional[pd.DataFrame] = None
    ft_records: List[pd.DataFrame] = []
    detected_multi_dim = 31

    def _get_pretrain(mode: str, horizon: int, models: Sequence[str]) -> pd.DataFrame:
        nonlocal detected_multi_dim
        df_res = ensure_pretrain_results(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            mode=mode,
            horizon=horizon,
            models=models,
            lookback=args.lookback,
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
            seeds=args.seeds,
            skip_existing=args.skip_existing,
        )
        if (
            mode == "multivariate"
            and "in_features" in df_res.columns
            and not df_res.empty
        ):
            detected_multi_dim = int(df_res["in_features"].iloc[0])

        if args.with_finetune:
            ft_df = ensure_finetune_results(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                mode=mode,
                horizon=horizon,
                models=models,
                num_stations=args.num_stations,
                lookback=args.lookback,
                stride=args.ft_stride,
                batch_size=args.ft_batch_size,
                epochs=args.ft_epochs,
                lr=args.ft_lr,
                backbone_lr_ratio=args.ft_backbone_lr_ratio,
                weight_decay=args.weight_decay,
                patience=args.ft_patience,
                loss_fn=args.loss_fn,
                huber_delta=args.huber_delta,
                peak_threshold_z=args.peak_threshold_z,
                peak_alpha=args.peak_alpha,
                diff_weight=args.diff_weight,
                peak_max_weight=args.peak_max_weight,
                val_selection_mode=args.val_selection_mode,
                freeze_backbone=args.ft_freeze_backbone,
                model_configs=model_configs,
                seed=args.seeds[0],
                skip_existing=args.skip_existing,
            )
            ft_records.append(ft_df)
        return df_res

    multi_h24: Optional[pd.DataFrame] = None
    if do_ab1:
        print("\n" + "=" * 84)
        print("BẮT ĐẦU ABLATION 1 (RQ1): MULTIVARIATE VS. UNIVARIATE (D=1) @ H=24")
        print("=" * 84)
        multi_h24 = _get_pretrain("multivariate", 24, DEFAULT_MODELS)
        uni_h24 = _get_pretrain("univariate", 24, DEFAULT_MODELS)

        ab1_df = build_ablation1_table(multi_h24, uni_h24, horizon=24)
        ab1_csv = ablation_dir / "ablation1_multivariate_vs_univariate_H24.csv"
        ab1_df.to_csv(ab1_csv, index=False, encoding="utf-8-sig")
        print(f"\n[SAVED] Ablation 1: {ab1_csv}")
        print(ab1_df.to_string(index=False))

    if do_ab2:
        print("\n" + "=" * 84)
        print("BẮT ĐẦU ABLATION 2 (RQ2): ĐA KHUNG DỰ BÁO H IN {24, 48, 96, 192}")
        print("=" * 84)
        horizon_dfs: Dict[int, pd.DataFrame] = {}
        for h in ALL_HORIZONS:
            horizon_dfs[h] = _get_pretrain("multivariate", h, DEFAULT_MODELS)
        if multi_h24 is None and 24 in horizon_dfs:
            multi_h24 = horizon_dfs[24]

        ab2_df = build_ablation2_table(horizon_dfs)
        ab2_csv = ablation_dir / "ablation2_multi_horizon_H24_48_96_192.csv"
        ab2_df.to_csv(ab2_csv, index=False, encoding="utf-8-sig")
        print(f"\n[SAVED] Ablation 2: {ab2_csv}")
        print(ab2_df.to_string(index=False))

    if do_ab3:
        print("\n" + "=" * 84)
        print(
            "BẮT ĐẦU ABLATION 3 (RQ2): INFORMER (PROBSPARSE) VS. INFORMER_FULL @ H={96, 192}"
        )
        print("=" * 84)
        long_dfs: Dict[int, pd.DataFrame] = {}
        for h in args.ab3_horizons:
            long_dfs[h] = _get_pretrain(
                "multivariate", h, ["informer", "informer_full"]
            )

        ab3_df = build_ablation3_table(long_dfs)
        ab3_csv = ablation_dir / "ablation3_informer_probsparse_vs_full.csv"
        ab3_df.to_csv(ab3_csv, index=False, encoding="utf-8-sig")
        print(f"\n[SAVED] Ablation 3: {ab3_csv}")
        print(ab3_df.to_string(index=False))

    # Tích hợp WP02 BaselineTrainer (chạy mới nếu bật --with-wp02-baseline hoặc nạp file sẵn có)
    existing_wp02_csv = report_dir / "wp02_baseline_summary_multivariate_H24.csv"
    if args.with_wp02_baseline or existing_wp02_csv.exists():
        print("\n" + "=" * 84)
        print("ĐỐI CHỨNG WP02 (BaselineTrainer) VS. WP03 (TrainingPipeline v2) @ H=24")
        print("=" * 84)
        wp02_h24 = ensure_wp02_baseline_results(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            mode="multivariate",
            horizon=24,
            lookback=args.lookback,
            batch_size=args.batch_size,
            train_stride=args.train_stride,
            epochs=args.epochs,
            seeds=args.seeds,
            skip_existing=args.skip_existing,
        )
        if multi_h24 is None:
            multi_h24 = _get_pretrain("multivariate", 24, DEFAULT_MODELS)
        wp02_vs_wp03_df = build_wp02_vs_wp03_table(wp02_h24, multi_h24, horizon=24)
        if not wp02_vs_wp03_df.empty:
            cmp_csv = ablation_dir / "wp02_baseline_vs_wp03_H24.csv"
            wp02_vs_wp03_df.to_csv(cmp_csv, index=False, encoding="utf-8-sig")
            print(f"[SAVED] Bảng đối chứng WP02 vs WP03: {cmp_csv}")
            print(wp02_vs_wp03_df.to_string(index=False))

    ft_all_df = pd.concat(ft_records, ignore_index=True) if ft_records else None
    if ft_all_df is not None:
        ft_ab_csv = ablation_dir / "ablation_finetune_summary.csv"
        ft_all_df.to_csv(ft_ab_csv, index=False, encoding="utf-8-sig")

    contracts_path = aggregate_all_contracts(
        report_dir=report_dir, ablation_dir=ablation_dir
    )
    if contracts_path is not None:
        print(
            f"[SAVED] Hợp đồng tổng hợp WP02 + WP03 (ExperimentMetricsContract): {contracts_path}"
        )

    md_path = export_markdown_report(
        ablation_dir=ablation_dir,
        wp02_vs_wp03_df=wp02_vs_wp03_df,
        ab1_df=ab1_df,
        ab2_df=ab2_df,
        ab3_df=ab3_df,
        ft_ablation_df=ft_all_df,
        multi_dim=detected_multi_dim,
    )
    print(f"\n[HOÀN TẤT] Báo cáo tổng hợp Markdown đã lưu tại: {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T0305: Chạy toàn bộ ma trận Ablation Studies & đối chứng WP02."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/preprocess"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["all"],
        choices=["all", "ablation1", "ablation2", "ablation3"],
        help="Chọn bộ thí nghiệm cần chạy (mặc định: all).",
    )
    parser.add_argument(
        "--ab3-horizons",
        nargs="+",
        type=int,
        default=LONG_HORIZONS,
        choices=[24, 48, 96, 192],
        help="Khung dự báo cho Ablation 3 (mặc định: 96 192).",
    )
    parser.add_argument("--num-stations", type=int, default=None)
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
    parser.add_argument("--use-residual-anchor", action="store_true", default=False)
    parser.add_argument(
        "--val-selection-mode",
        type=str,
        default="balanced_dev",
        choices=["balanced_dev", "dev", "train_dev"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 1024, 2026],
        help="Danh sách hạt giống (mặc định 3 seeds: 42 1024 2026).",
    )
    parser.add_argument(
        "--with-wp02-baseline",
        action="store_true",
        help="Chạy kèm WP02 BaselineTrainer (LSTM, DLinear) trên dữ liệu thật WP01 để so sánh trực tiếp.",
    )
    parser.add_argument(
        "--with-finetune",
        action="store_true",
        help="Chạy kèm Fine-tuning 12 trạm cho từng cấu hình Ablation.",
    )
    parser.add_argument("--ft-batch-size", type=int, default=256)
    parser.add_argument("--ft-epochs", type=int, default=15)
    parser.add_argument("--ft-lr", type=float, default=5e-5)
    parser.add_argument("--ft-backbone-lr-ratio", type=float, default=0.1)
    parser.add_argument("--ft-stride", type=int, default=2)
    parser.add_argument("--ft-patience", type=int, default=5)
    parser.add_argument("--ft-freeze-backbone", action="store_true", default=True)
    parser.add_argument(
        "--ft-unfreeze-backbone",
        dest="ft_freeze_backbone",
        action="store_false",
        help="Mở khóa toàn bộ backbone với Discriminative LR khi fine-tune trong ablation.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Bắt buộc chạy lại từ đầu kể cả khi đã có file CSV kết quả.",
    )
    parser.set_defaults(skip_existing=True)
    apply_yaml_config(parser, ["paths", "data", "pretrain", "ablation"])
    if parser.get_default("num_stations") is None:
        parser.set_defaults(num_stations=12)
    pre_args, _ = parser.parse_known_args()
    if Path(pre_args.config).exists():
        with open(pre_args.config, "r", encoding="utf-8") as f:
            ft_cfg = (yaml.safe_load(f) or {}).get("finetune", {})
        if isinstance(ft_cfg, dict):
            parser.set_defaults(
                ft_batch_size=ft_cfg.get("batch_size", 256),
                ft_epochs=ft_cfg.get("epochs", 15),
                ft_lr=ft_cfg.get("lr", 5e-5),
                ft_backbone_lr_ratio=ft_cfg.get("backbone_lr_ratio", 0.1),
                ft_stride=ft_cfg.get("stride", 2),
                ft_freeze_backbone=bool(
                    ft_cfg.get(
                        "freeze_backbone",
                        not bool(ft_cfg.get("unfreeze_backbone", False)),
                    )
                ),
            )

    args = parser.parse_args()
    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)
    return args


if __name__ == "__main__":
    run_ablation_suite(parse_args())
