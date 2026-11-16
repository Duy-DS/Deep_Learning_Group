"""
T0305 — Tệp chạy độc lập toàn bộ ma trận thực nghiệm Ablation Studies:
- Ablation 1 (RQ1): Multivariate (D=15) vs. Univariate (D=1) tại H=24.
- Ablation 2 (RQ2): Đánh giá 4 mô hình qua 4 khung dự báo H in {24, 48, 96, 192}.
- Ablation 3 (RQ2): Informer (ProbSparse) vs. Informer_FullAttention O(L^2) tại H in {96, 192}.

Tính năng:
- Gọi trực tiếp TrainingPipeline (training_loop.py) và tùy chọn run_fine_tuning_pipeline (fine_tune.py).
- Tự động tái sử dụng kết quả đã chạy trước đó (--skip-existing) theo từng mô hình để tránh chạy trùng.
- Tự động tổng hợp 3 bảng CSV chuẩn và xuất báo cáo Markdown (outputs/ablation/ablation_report_summary.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from fine_tune import run_fine_tuning_pipeline
from training_loop import DEFAULT_MODELS, TrainingPipeline

ALL_HORIZONS: List[int] = [24, 48, 96, 192]
LONG_HORIZONS: List[int] = [96, 192]


def ensure_pretrain_results(
    data_dir: Path,
    output_dir: Path,
    mode: str,
    horizon: int,
    models: Sequence[str],
    lookback: int,
    batch_size: int,
    train_stride: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    epochs: int,
    patience: int,
    seeds: List[int],
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Đảm bảo đã có đủ kết quả Pre-training cho danh sách `models` ở cấu hình `(mode, horizon)`.
    Nếu bật `skip_existing`, chỉ huấn luyện các mô hình còn thiếu và gộp vào file báo cáo hiện có.
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
            weight_decay=weight_decay,
            dropout=dropout,
            epochs=epochs,
            patience=patience,
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
    weight_decay: float,
    patience: int,
    freeze_backbone: bool,
    seed: int,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Chạy hoặc nạp kết quả Fine-tuning 12 trạm cho cấu hình `(mode, horizon)`."""
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
            mode=mode,  # type: ignore[arg-type]
            lookback=lookback,
            horizon=horizon,
            stride=stride,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            freeze_backbone=freeze_backbone,
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


def build_ablation1_table(
    multi_df: pd.DataFrame, uni_df: pd.DataFrame, horizon: int = 24
) -> pd.DataFrame:
    """
    Ablation 1 (RQ1): So sánh Multivariate (D=15) vs. Univariate (D=1)
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
    Ablation 2 (RQ2): Tổng hợp 4 mô hình trên đầy đủ 4 khung dự báo H in {24, 48, 96, 192}.
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
            "train_time_mean_sec",
            "peak_vram_max_mb",
        ]
        available_cols = [c for c in cols if c in df_h.columns]
        frames.append(df_h[available_cols])

    long_df = pd.concat(frames, ignore_index=True)
    # Sắp xếp theo thứ tự mô hình chuẩn trong từng horizon
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

        rows.append(
            {
                "horizon": h,
                "probsparse_mse": sparse_row["mse_mean_std"],
                "full_attn_mse": full_row["mse_mean_std"],
                "probsparse_rmse": sparse_row["rmse_mean_std"],
                "full_attn_rmse": full_row["rmse_mean_std"],
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


def export_markdown_report(
    ablation_dir: Path,
    ab1_df: Optional[pd.DataFrame],
    ab2_df: Optional[pd.DataFrame],
    ab3_df: Optional[pd.DataFrame],
    ft_ablation_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Xuất toàn bộ các bảng Ablation sang định dạng Markdown sẵn sàng đưa vào báo cáo."""
    md_path = ablation_dir / "ablation_report_summary.md"
    sections: List[str] = [
        "# TỔNG HỢP KẾT QUẢ ABLATION STUDIES — DỰ BÁO PM2.5 BẮC KINH\n"
    ]

    if ab1_df is not None and not ab1_df.empty:
        sections.append(
            "## 1. Ablation 1 (RQ1): Multivariate ($D=15$) vs. Univariate ($D=1$) tại $H=24$\n"
        )
        sections.append(ab1_df.to_markdown(index=False) + "\n")

    if ab2_df is not None and not ab2_df.empty:
        sections.append(
            "## 2. Ablation 2 (RQ2): Đánh giá 4 mô hình trên 4 khung dự báo $H \\in \\{24, 48, 96, 192\\}$\n"
        )
        display_cols = [
            "horizon",
            "model",
            "mse_mean_std",
            "rmse_mean_std",
            "mae_mean_std",
            "train_time_mean_sec",
            "peak_vram_max_mb",
        ]
        sections.append(ab2_df[display_cols].to_markdown(index=False) + "\n")

    if ab3_df is not None and not ab3_df.empty:
        sections.append(
            "## 3. Ablation 3 (RQ2): Informer (`ProbSparse`) vs. `Informer_FullAttention` ($O(L^2)$)\n"
        )
        sections.append(ab3_df.to_markdown(index=False) + "\n")

    if ft_ablation_df is not None and not ft_ablation_df.empty:
        sections.append(
            "## 4. Tổng hợp Fine-Tuning (Trước vs. Sau) qua các cấu hình Ablation\n"
        )
        sections.append(ft_ablation_df.to_markdown(index=False) + "\n")

    md_path.write_text("\n".join(sections), encoding="utf-8")
    return md_path


def run_ablation_suite(args: argparse.Namespace) -> None:
    """Điều phối toàn bộ quy trình Ablation Studies theo lựa chọn người dùng."""
    ablation_dir = args.output_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    run_all = "all" in args.ablations
    do_ab1 = run_all or ("ablation1" in args.ablations)
    do_ab2 = run_all or ("ablation2" in args.ablations)
    do_ab3 = run_all or ("ablation3" in args.ablations)

    ab1_df: Optional[pd.DataFrame] = None
    ab2_df: Optional[pd.DataFrame] = None
    ab3_df: Optional[pd.DataFrame] = None
    ft_records: List[pd.DataFrame] = []

    pretrain_cache: Dict[tuple[str, int], pd.DataFrame] = {}

    def _get_pretrain(mode: str, horizon: int, models: Sequence[str]) -> pd.DataFrame:
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
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            epochs=args.epochs,
            patience=args.patience,
            seeds=args.seeds,
            skip_existing=args.skip_existing,
        )
        pretrain_cache[(mode, horizon)] = df_res
        if args.with_finetune:
            ft_df = ensure_finetune_results(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                mode=mode,
                horizon=horizon,
                models=models,
                lookback=args.lookback,
                stride=args.train_stride,
                batch_size=args.ft_batch_size,
                epochs=args.ft_epochs,
                lr=args.ft_lr,
                weight_decay=args.weight_decay,
                patience=4,
                freeze_backbone=True,
                seed=args.seeds[0],
                skip_existing=args.skip_existing,
            )
            ft_records.append(ft_df)
        return df_res

    # ------------------------------------------------------------------
    # ABLATION 1: Multivariate vs. Univariate (H = 24)
    # ------------------------------------------------------------------
    if do_ab1:
        print("\n" + "=" * 84)
        print(
            "BẮT ĐẦU ABLATION 1 (RQ1): MULTIVARIATE (D=15) VS. UNIVARIATE (D=1) @ H=24"
        )
        print("=" * 84)
        multi_h24 = _get_pretrain("multivariate", 24, DEFAULT_MODELS)
        uni_h24 = _get_pretrain("univariate", 24, DEFAULT_MODELS)

        ab1_df = build_ablation1_table(multi_h24, uni_h24, horizon=24)
        ab1_csv = ablation_dir / "ablation1_multivariate_vs_univariate_H24.csv"
        ab1_df.to_csv(ab1_csv, index=False, encoding="utf-8-sig")
        print(f"\n[SAVED] Ablation 1: {ab1_csv}")
        print(ab1_df.to_string(index=False))

    # ------------------------------------------------------------------
    # ABLATION 2: 4 mô hình x 4 khung dự báo H in {24, 48, 96, 192}
    # ------------------------------------------------------------------
    if do_ab2:
        print("\n" + "=" * 84)
        print("BẮT ĐẦU ABLATION 2 (RQ2): ĐA KHUNG DỰ BÁO H IN {24, 48, 96, 192}")
        print("=" * 84)
        horizon_dfs: Dict[int, pd.DataFrame] = {}
        for h in ALL_HORIZONS:
            horizon_dfs[h] = _get_pretrain("multivariate", h, DEFAULT_MODELS)

        ab2_df = build_ablation2_table(horizon_dfs)
        ab2_csv = ablation_dir / "ablation2_multi_horizon_H24_48_96_192.csv"
        ab2_df.to_csv(ab2_csv, index=False, encoding="utf-8-sig")
        print(f"\n[SAVED] Ablation 2: {ab2_csv}")
        print(ab2_df.to_string(index=False))

    # ------------------------------------------------------------------
    # ABLATION 3: Informer (ProbSparse) vs. Informer_FullAttention O(L^2)
    # ------------------------------------------------------------------
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

    ft_all_df = pd.concat(ft_records, ignore_index=True) if ft_records else None
    if ft_all_df is not None:
        ft_ab_csv = ablation_dir / "ablation_finetune_summary.csv"
        ft_all_df.to_csv(ft_ab_csv, index=False, encoding="utf-8-sig")

    md_path = export_markdown_report(
        ablation_dir=ablation_dir,
        ab1_df=ab1_df,
        ab2_df=ab2_df,
        ab3_df=ab3_df,
        ft_ablation_df=ft_all_df,
    )
    print(f"\n[HOÀN TẤT] Báo cáo tổng hợp Markdown đã lưu tại: {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T0305: Chạy toàn bộ ma trận Ablation Studies (RQ1 & RQ2)."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
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
    parser.add_argument("--lookback", type=int, default=96)
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
        help="Danh sách hạt giống (mặc định 3 seeds: 42 1024 2026).",
    )
    parser.add_argument(
        "--with-finetune",
        action="store_true",
        help="Chạy kèm Fine-tuning 12 trạm cho từng cấu hình Ablation.",
    )
    parser.add_argument("--ft-batch-size", type=int, default=256)
    parser.add_argument("--ft-epochs", type=int, default=12)
    parser.add_argument("--ft-lr", type=float, default=3e-5)
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Bắt buộc chạy lại từ đầu kể cả khi đã có file CSV kết quả.",
    )
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation_suite(parse_args())
