from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COL = "pm2.5"

CONTINUOUS_COLS: List[str] = [
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
]

WIND_DIRECTIONS: List[str] = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]

CANONICAL_COLS: List[str] = (
    CONTINUOUS_COLS
    + [
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]
    + [f"wd_{d}" for d in WIND_DIRECTIONS]
)


def build_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" not in out.columns:
        out["datetime"] = pd.to_datetime(out[["year", "month", "day", "hour"]])
    else:
        out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values(["station", "datetime"]).reset_index(drop=True)


def split_chronological_per_station(
    df: pd.DataFrame,
    train_end: str = "2015-12-31 23:00:00",
    train_dev_end: str = "2016-04-30 23:00:00",
    dev_end: str = "2016-08-31 23:00:00",
) -> Dict[str, pd.DataFrame]:
    """Chia 4 tập theo trục thời gian TRƯỚC khi thực hiện bất kỳ thao tác điền khuyết nào."""
    df = build_datetime_column(df)
    t_train = pd.Timestamp(train_end)
    t_train_dev = pd.Timestamp(train_dev_end)
    t_dev = pd.Timestamp(dev_end)

    return {
        "train": df[df["datetime"] <= t_train].copy().reset_index(drop=True),
        "train_dev": df[(df["datetime"] > t_train) & (df["datetime"] <= t_train_dev)]
        .copy()
        .reset_index(drop=True),
        "dev": df[(df["datetime"] > t_train_dev) & (df["datetime"] <= t_dev)]
        .copy()
        .reset_index(drop=True),
        "test": df[df["datetime"] > t_dev].copy().reset_index(drop=True),
    }


def causal_impute_splits(splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Điền khuyết nhân quả tuyệt đối (không nhìn trước):
    1. Chỉ dùng ffill() theo từng trạm bên trong từng phân vùng.
    2. Các ô NaN đầu chuỗi chưa có lịch sử chỉ dùng median của tập TRAIN.
    3. Cột hướng gió 'wd' điền ffill() và fallback bằng mode của tập TRAIN.
    """
    train_df = splits["train"].copy()
    station_medians = train_df.groupby("station")[CONTINUOUS_COLS].median()
    global_medians = train_df[CONTINUOUS_COLS].median()

    wd_col = "wd" if "wd" in train_df.columns else "cbwd"
    station_wd_mode = (
        train_df.groupby("station")[wd_col]
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "NW")
        .to_dict()
    )

    imputed_splits: Dict[str, pd.DataFrame] = {}
    for split_name, part in splits.items():
        df_s = part.sort_values(["station", "datetime"]).copy()
        df_s[CONTINUOUS_COLS] = df_s.groupby("station")[CONTINUOUS_COLS].ffill()

        for st, idx in df_s.groupby("station").groups.items():
            fill_vals = (
                station_medians.loc[st].fillna(global_medians)
                if st in station_medians.index
                else global_medians
            )
            df_s.loc[idx, CONTINUOUS_COLS] = df_s.loc[idx, CONTINUOUS_COLS].fillna(
                fill_vals
            )

            df_s.loc[idx, wd_col] = (
                df_s.loc[idx, wd_col].ffill().fillna(station_wd_mode.get(st, "NW"))
            )

        imputed_splits[split_name] = df_s.reset_index(drop=True)
    return imputed_splits


def encode_features(splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Mã hóa chu kỳ thời gian và cố định đúng 16 cột One-Hot hướng gió wd_* (tổng D=31)."""
    encoded: Dict[str, pd.DataFrame] = {}
    for split_name, part in splits.items():
        df_e = part.copy()
        dt = pd.to_datetime(df_e["datetime"])
        df_e["hour_sin"] = np.sin(2.0 * np.pi * dt.dt.hour / 24.0)
        df_e["hour_cos"] = np.cos(2.0 * np.pi * dt.dt.hour / 24.0)
        df_e["month_sin"] = np.sin(2.0 * np.pi * dt.dt.month / 12.0)
        df_e["month_cos"] = np.cos(2.0 * np.pi * dt.dt.month / 12.0)

        wd_col = "wd" if "wd" in df_e.columns else "cbwd"
        wd_series = df_e[wd_col].astype(str).str.strip().str.upper()
        for d in WIND_DIRECTIONS:
            df_e[f"wd_{d}"] = (wd_series == d).astype(np.float32)

        keep_cols = ["station", "datetime"] + CANONICAL_COLS
        encoded[split_name] = df_e[keep_cols].copy()
    return encoded


def scale_splits(
    splits: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    scaler_path: Optional[Path] = None,
) -> Tuple[Dict[str, pd.DataFrame], StandardScaler]:
    """Khớp StandardScaler duy nhất trên tập train và biến đổi cả 4 tập."""
    scaler = StandardScaler()
    scaler.fit(splits["train"][feature_cols])

    scaled: Dict[str, pd.DataFrame] = {}
    for split_name, part in splits.items():
        df_sc = part.copy()
        df_sc[feature_cols] = scaler.transform(df_sc[feature_cols])
        scaled[split_name] = df_sc

    if scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
    return scaled, scaler


def verify_causal_preprocessing(
    raw_df: pd.DataFrame,
    processed_splits: Dict[str, pd.DataFrame],
) -> None:
    """Kiểm toán tự động 4 tiêu chí chống rò rỉ dữ liệu trước khi huấn luyện."""
    # 1. Ranh giới thời gian không chồng lấn
    assert (
        processed_splits["train"]["datetime"].max()
        < processed_splits["train_dev"]["datetime"].min()
    )
    assert (
        processed_splits["train_dev"]["datetime"].max()
        < processed_splits["dev"]["datetime"].min()
    )
    assert (
        processed_splits["dev"]["datetime"].max()
        < processed_splits["test"]["datetime"].min()
    )

    # 2. Đúng hợp đồng đặc trưng D=31 & không NaN/Inf
    for name, df_p in processed_splits.items():
        assert list(df_p.columns) == ["station", "datetime"] + CANONICAL_COLS, (
            f"Sai cột ở {name}"
        )
        vals = df_p[CANONICAL_COLS].to_numpy(dtype=np.float64)
        assert np.isfinite(vals).all(), f"Tồn tại NaN/Inf trong tập {name}"

    # 3. Kiểm tra nhiễu loạn tương lai trên Test (Perturbation Test)
    perturbed_raw = raw_df.copy()
    perturbed_raw = build_datetime_column(perturbed_raw)
    test_mask = perturbed_raw["datetime"] > processed_splits["dev"]["datetime"].max()
    last_idx = perturbed_raw[test_mask].index[-50:]
    perturbed_raw.loc[last_idx, "pm2.5"] = 9999.0

    perturbed_splits = encode_features(
        causal_impute_splits(split_chronological_per_station(perturbed_raw))
    )
    orig_head = processed_splits["test"][CANONICAL_COLS].iloc[:-100].to_numpy()
    pert_head = perturbed_splits["test"][CANONICAL_COLS].iloc[:-100].to_numpy()
    assert np.allclose(orig_head, pert_head), (
        "RÒ RỈ TƯƠNG LAI: Thay đổi cuối Test làm đổi đầu Test!"
    )
    print(
        "[VERIFIED] Preprocessing đạt chuẩn nhân quả 100%, D=31, không rò rỉ ranh giới hay tương lai."
    )


def run_preprocess(raw_dir: Path, out_dir: Path) -> None:
    csv_files = sorted(raw_dir.glob("PRSA_Data_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu thô PRSA_Data_*.csv tại {raw_dir}"
        )
    raw_df = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)

    splits = split_chronological_per_station(raw_df)
    imputed = causal_impute_splits(splits)
    encoded = encode_features(imputed)
    verify_causal_preprocessing(raw_df, encoded)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df_out in encoded.items():
        df_out.to_csv(out_dir / f"{split_name}.csv", index=False)
    print(f"[SAVED] Đã xuất 4 tập sạch nhân quả (D=31) tại: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/preprocess"))
    args = parser.parse_args()
    run_preprocess(args.raw_dir, args.out_dir)

