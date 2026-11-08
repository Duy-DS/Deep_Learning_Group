from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


TIME_COLS = ["year", "month", "day", "hour"]
TARGET_COL = "pm2.5"


def load_pm25(path):
    """Đọc dữ liệu PM2.5 và tạo cột thời gian."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Chuẩn hóa tên cột thường gặp trong bộ dữ liệu UCI Beijing PM2.5
    if "datetime" not in df.columns:
        required = {"year", "month", "day", "hour"}
        if not required.issubset(df.columns):
            raise ValueError(f"Thiếu cột thời gian: {required - set(df.columns)}")
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]], errors="coerce"
        )

    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df


def missing_report(df, output_dir=None):
    """Thống kê missing values; hỗ trợ phân tích theo tháng và trạm."""
    report = df.isna().sum().to_frame("missing_count")
    report["missing_rate"] = df.isna().mean()
    report = report.sort_values("missing_rate", ascending=False)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_dir / "missing_overall.csv")

        if "datetime" in df.columns:
            by_month = df.set_index("datetime").isna().resample("MS").mean()
            by_month.to_csv(output_dir / "missing_by_month.csv")

        station_col = next((c for c in ["station", "station_id"] if c in df.columns), None)
        if station_col:
            by_station = df.groupby(station_col).apply(
                lambda x: x.isna().mean(), include_groups=False
            )
            by_station.to_csv(output_dir / "missing_by_station.csv")

    return report


def clean_missing_values(df, target_col=TARGET_COL):
    """
    Làm sạch theo thời gian:
    - Nội suy tuyến tính cho các cột số.
    - Forward Fill cho các giá trị còn thiếu ở đầu/cuối chuỗi.
    Không nội suy cột mục tiêu nếu cần giữ nguyên missing target để đánh giá.
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit_direction="forward"
    )
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Biến phân loại: dùng giá trị gần nhất, sau đó mới dùng mode nếu đầu chuỗi còn thiếu
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        df[col] = df[col].ffill().bfill()
        if df[col].isna().any() and not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    if target_col in df.columns:
        df = df.dropna(subset=[target_col])

    return df


def encode_features(df, wind_col="cbwd"):
    """Mã hóa hướng gió One-Hot và thời gian theo chu kỳ."""
    df = df.copy()

    # Chu kỳ giờ: 24 giờ; tháng: 12 tháng
    if "hour" not in df.columns and "datetime" in df.columns:
        df["hour"] = df["datetime"].dt.hour
    if "month" not in df.columns and "datetime" in df.columns:
        df["month"] = df["datetime"].dt.month

    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # UCI Beijing PM2.5 thường có cbwd: NW, SE, cv, NE.
    # Nếu dữ liệu có 5 nhóm hướng gió, thêm nhóm thứ 5 sẽ được tự động mã hóa.
    if wind_col in df.columns:
        df[wind_col] = df[wind_col].fillna("Unknown").astype(str)
        dummies = pd.get_dummies(df[wind_col], prefix="wind", dtype=np.float32)
        df = pd.concat([df.drop(columns=[wind_col]), dummies], axis=1)

    return df


def chronological_split(df, ratios=(0.60, 0.10, 0.10, 0.20)):
    """
    Chia tuần tự theo thời gian thành:
    train, train_dev, dev, test.
    Tỷ lệ tương ứng 60/10/10/20.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Tổng tỷ lệ split phải bằng 1.")

    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    n_train = int(n * ratios[0])
    n_train_dev = int(n * ratios[1])
    n_dev = int(n * ratios[2])

    i1 = n_train
    i2 = i1 + n_train_dev
    i3 = i2 + n_dev

    return {
        "train": df.iloc[:i1].copy(),
        "train_dev": df.iloc[i1:i2].copy(),
        "dev": df.iloc[i2:i3].copy(),
        "test": df.iloc[i3:].copy(),
    }


def scale_splits(splits, feature_cols, scaler_path=None):
    """
    Fit StandardScaler chỉ trên train; các tập còn lại chỉ transform.
    Trả về các DataFrame đã chuẩn hóa và scaler.
    """
    scaler = StandardScaler()
    train = splits["train"].copy()

    train[feature_cols] = scaler.fit_transform(train[feature_cols])

    scaled = {"train": train}
    for name in ["train_dev", "dev", "test"]:
        part = splits[name].copy()
        part[feature_cols] = scaler.transform(part[feature_cols])
        scaled[name] = part

    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return scaled, scaler
