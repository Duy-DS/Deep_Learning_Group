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
    """
    Tạo X: (samples, lookback, features)
    Tạo y: (samples, horizon) dự báo target trong tương lai.
    Mỗi cửa sổ được tạo trong từng split riêng để tránh vượt ranh giới split.
    """
    if lookback <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("lookback, horizon và stride phải > 0.")

    data_x = df[feature_cols].to_numpy(dtype=np.float32)
    data_y = df[target_col].to_numpy(dtype=np.float32)

    X, y = [], []
    last_start = len(df) - lookback - horizon

    for start in range(0, last_start + 1, stride):
        x_end = start + lookback
        y_end = x_end + horizon

        X.append(data_x[start:x_end])
        y.append(data_y[x_end:y_end])

    if not X:
        return (
            np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )

    return np.stack(X), np.stack(y)


def build_windows_for_all_horizons(
    splits,
    feature_cols,
    target_col="pm2.5",
    lookback=96,
    horizons=(24, 48, 96, 192),
):
    """Tạo bộ X, y cho mỗi split và từng forecast horizon."""
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
            )
            result[split_name][horizon] = {"X": X, "y": y}
    return result
