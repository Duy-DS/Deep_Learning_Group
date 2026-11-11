import numpy as np
import pandas as pd


def make_sliding_windows(
    df,
    feature_cols,
    target_col="pm2.5",
    lookback=96,
    horizon=24,
    stride=1,
):
    """Tạo X (samples, lookback, features) và y (samples, horizon).

    Nếu dữ liệu có nhiều trạm, tạo cửa sổ riêng từng trạm để không nối
    dữ liệu của hai trạm khác nhau vào cùng một mẫu.
    """
    if lookback <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("lookback, horizon và stride phải > 0.")

    required = set(feature_cols) | {target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame thiếu các cột: {sorted(missing)}")

    groups = (
        [group for _, group in df.groupby("station", sort=False)]
        if "station" in df.columns
        else [df]
    )

    all_X, all_y = [], []
    for group in groups:
        if "datetime" in group.columns:
            group = group.sort_values("datetime")

        data_x = group[feature_cols].to_numpy(dtype=np.float32)
        data_y = group[target_col].to_numpy(dtype=np.float32)
        last_start = len(group) - lookback - horizon

        for start in range(0, last_start + 1, stride):
            x_end = start + lookback
            y_end = x_end + horizon
            all_X.append(data_x[start:x_end])
            all_y.append(data_y[x_end:y_end])

    if not all_X:
        return (
            np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )

    return np.stack(all_X), np.stack(all_y)


def build_windows_for_all_horizons(
    splits,
    feature_cols,
    target_col="pm2.5",
    lookback=96,
    horizons=(24, 48, 96, 192),
    stride=1,
):
    """Tạo X, y cho từng split và từng forecast horizon."""
    result = {}
    for split_name, split_df in splits.items():
        result[split_name] = {}
        for horizon in horizons:
            X, y = make_sliding_windows(
                split_df,
                feature_cols=feature_cols,
                target_col=target_col,
                lookback=lookback,
                horizon=horizon,
                stride=stride,
            )
            result[split_name][horizon] = {"X": X, "y": y}
    return result
