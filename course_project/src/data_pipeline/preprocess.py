from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import argparse

from src.configuration import ProjectConfiguration

TIME_COLS = ["year", "month", "day", "hour"]
TARGET_COL = "pm2.5"
WIND_CATEGORIES = ["NW", "SE", "cv", "NE", "Unknown"]


def load_pm25(path):
    """Đọc một file CSV hoặc toàn bộ CSV trong thư mục dữ liệu PRSA."""
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy đường dẫn dữ liệu: {path}")

    if path.is_file():
        csv_files = [path] if path.suffix.lower() == ".csv" else []
    else:
        csv_files = sorted(path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV trong: {path}")

    frames = []
    for file in csv_files:
        station_df = pd.read_csv(file)
        station_df.columns = station_df.columns.str.strip().str.lower()

        if "station" not in station_df.columns:
            station_df["station"] = file.stem.replace("PRSA_Data_", "")

        if "datetime" not in station_df.columns:
            required = {"year", "month", "day", "hour"}
            missing = required - set(station_df.columns)
            if missing:
                raise ValueError(f"File {file.name} thiếu cột thời gian: {missing}")
            station_df["datetime"] = pd.to_datetime(
                station_df[["year", "month", "day", "hour"]], errors="coerce"
            )

        frames.append(station_df)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["datetime"])
    df = df.sort_values(["station", "datetime"])
    df = df.drop_duplicates(["station", "datetime"]).reset_index(drop=True)

    print(f"Số file/trạm: {len(csv_files)}")
    print(f"Kích thước dữ liệu: {df.shape}")
    return df


def missing_report(df, output_dir=None):
    """Thống kê missing values tổng thể, theo tháng và theo trạm."""
    report = df.isna().sum().to_frame("missing_count")
    report["missing_rate"] = df.isna().mean()
    report = report.sort_values("missing_rate", ascending=False)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_dir / "missing_overall.csv")

        if "datetime" in df.columns:
            by_month = (
                df.set_index("datetime").groupby("station").resample("MS").mean(numeric_only=True)
                if "station" in df.columns
                else df.set_index("datetime").resample("MS").mean(numeric_only=True)
            )
            by_month.to_csv(output_dir / "missing_by_month.csv")

        if "station" in df.columns:
            df.groupby("station").apply(
                lambda group: group.isna().mean(), include_groups=False
            ).to_csv(output_dir / "missing_by_station.csv")

    return report


def _clean_one_series(group, target_col):
    group = group.copy().sort_values("datetime")
    numeric_cols = group.select_dtypes(include=np.number).columns.tolist()
    group[numeric_cols] = group[numeric_cols].interpolate(
        method="linear", limit_direction="both"
    ).ffill().bfill()

    categorical_cols = group.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        group[col] = group[col].ffill().bfill()
        if group[col].isna().any() and not group[col].mode().empty:
            group[col] = group[col].fillna(group[col].mode().iloc[0])

    if target_col in group.columns:
        group = group.dropna(subset=[target_col])
    return group


def clean_missing_values(df, target_col=TARGET_COL):
    """Làm sạch theo thời gian riêng từng trạm, tránh nội suy chéo giữa các trạm."""
    df = df.copy()
    if "station" in df.columns:
        parts = [
            _clean_one_series(group, target_col)
            for _, group in df.groupby("station", sort=False)
        ]
        return pd.concat(parts, ignore_index=True).sort_values(
            ["station", "datetime"]
        ).reset_index(drop=True)
    return _clean_one_series(df, target_col).reset_index(drop=True)


def encode_features(df, wind_col="cbwd"):
    """Mã hóa hướng gió One-Hot và thời gian theo chu kỳ."""
    df = df.copy()

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

    if wind_col in df.columns:
        wind = df[wind_col].fillna("Unknown").astype(str).str.strip()
        wind = wind.where(wind.isin(WIND_CATEGORIES[:-1]), "Unknown")
        df[wind_col] = pd.Categorical(wind, categories=WIND_CATEGORIES)
        dummies = pd.get_dummies(df[wind_col], prefix="wind", dtype=np.float32)
        df = pd.concat([df.drop(columns=[wind_col]), dummies], axis=1)

    return df


def chronological_split(df, ratios=(0.60, 0.10, 0.10, 0.20)):
    """Chia theo thời gian; nếu có nhiều trạm thì chia riêng từng trạm."""
    if len(ratios) != 4 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Cần 4 tỷ lệ train/train_dev/dev/test có tổng bằng 1.")

    names = ["train", "train_dev", "dev", "test"]

    def split_one(group):
        group = group.sort_values("datetime").reset_index(drop=True)
        n = len(group)
        i1 = int(n * ratios[0])
        i2 = i1 + int(n * ratios[1])
        i3 = i2 + int(n * ratios[2])
        return {
            "train": group.iloc[:i1].copy(),
            "train_dev": group.iloc[i1:i2].copy(),
            "dev": group.iloc[i2:i3].copy(),
            "test": group.iloc[i3:].copy(),
        }

    groups = [group for _, group in df.groupby("station", sort=False)] if "station" in df.columns else [df]
    results = {name: [] for name in names}
    for group in groups:
        parts = split_one(group)
        for name in names:
            results[name].append(parts[name])

    return {
        name: pd.concat(parts, ignore_index=True).sort_values(
            ["station", "datetime"] if "station" in df.columns else ["datetime"]
        ).reset_index(drop=True)
        for name, parts in results.items()
    }


def scale_splits(splits, feature_cols, scaler_path=None):
    """Fit scaler trên train; chỉ transform các split còn lại."""
    scaler = StandardScaler()
    scaled = {}
    train = splits["train"].copy()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    scaled["train"] = train

    for name in ["train_dev", "dev", "test"]:
        part = splits[name].copy()
        part[feature_cols] = scaler.transform(part[feature_cols])
        scaled[name] = part

    if scaler_path:
        scaler_path = Path(scaler_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return scaled, scaler

def save_splits_csv(splits, output_dir):
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        output_path = output_dir / f"{name}.csv"
        data.to_csv(output_path, index=False)
        print(f"Đã lưu: {output_path}")


class PreprocessingPipeline:
    """Điều phối tiền xử lý, chia tập và chuẩn hóa theo cấu hình dự án."""

    def __init__(self, configuration: ProjectConfiguration, overwrite: bool = False) -> None:
        """Khởi tạo pipeline từ cấu hình đã nạp.

        Args:
            configuration: Đối tượng ``ProjectConfiguration``.
            overwrite: Cho phép ghi đè CSV và scaler đã tồn tại.
        """
        self.configuration = configuration
        self.overwrite = overwrite

    def run(self) -> dict[str, pd.DataFrame]:
        """Chạy pipeline dữ liệu và lưu các split đã chuẩn hóa.

        Returns:
            Từ điển các DataFrame đã chia và chuẩn hóa.

        Raises:
            KeyError: Nếu thiếu tham số đường dẫn hoặc nhóm dữ liệu cần thiết.
            ValueError: Nếu danh sách đặc trưng cấu hình không có trong dữ liệu.
            FileExistsError: Nếu đầu ra đã tồn tại và chưa cho phép ghi đè.
        """
        paths = self.configuration.section("paths")
        data_config = self.configuration.section("data")
        raw_path = Path(paths["data_dir"]).expanduser()
        nested_raw_path = raw_path / "PRSA_Data_20130301-20170228"
        if raw_path.is_dir() and not any(raw_path.glob("*.csv")) and nested_raw_path.is_dir():
            raw_path = nested_raw_path
        raw_data = load_pm25(raw_path)
        ratios = data_config.get("split_ratios", [0.60, 0.10, 0.10, 0.20])
        raw_splits = chronological_split(raw_data, ratios=tuple(ratios))

        # Kiểm tra artifact trước khi xử lý để mặc định không ghi đè dữ liệu người dùng.
        output_dir = Path(paths["processed_dir"])
        scaler_path = paths.get("scaler_template")
        if scaler_path:
            scaler_path = scaler_path.format(mode=data_config.get("mode", "multivariate"))
        existing_outputs = [
            output_dir / f"{split_name}.csv" for split_name in raw_splits
            if (output_dir / f"{split_name}.csv").exists()
        ]
        existing_scaler = bool(scaler_path and Path(scaler_path).exists())
        if not self.overwrite and (existing_outputs or existing_scaler):
            existing_paths = existing_outputs + (
                [Path(scaler_path)] if existing_scaler else []
            )
            raise FileExistsError(
                "Từ chối ghi đè artifact đã tồn tại: "
                + ", ".join(map(str, existing_paths))
                + ". Chọn đường dẫn mới hoặc truyền --overwrite."
            )

        encoded_splits = {}
        for split_name, split_data in raw_splits.items():
            # Tiền xử lý riêng sau khi chia để thống kê/điền khuyết không đi qua ranh giới tập.
            cleaned_data = clean_missing_values(
                split_data, target_col=data_config.get("target_col", TARGET_COL)
            )
            encoded_splits[split_name] = encode_features(cleaned_data)
        if data_config.get("mode", "multivariate") == "univariate":
            feature_names = [data_config.get("target_col", TARGET_COL)]
        else:
            feature_names = data_config["canonical_multivariate_cols"]
        available_columns = set.union(
            *(set(frame.columns) for frame in encoded_splits.values())
        )
        missing_features = sorted(set(feature_names) - available_columns)
        if missing_features:
            raise ValueError(f"Thiếu đặc trưng cấu hình trong dữ liệu: {missing_features}")

        for split_name, split_data in encoded_splits.items():
            missing_split_features = sorted(set(feature_names) - set(split_data.columns))
            if missing_split_features:
                raise ValueError(
                    f"Split '{split_name}' thiếu đặc trưng: {missing_split_features}"
                )
        scaled_splits, _ = scale_splits(
            encoded_splits, feature_cols=feature_names, scaler_path=scaler_path
        )
        save_splits_csv(scaled_splits, output_dir)
        return scaled_splits


def parse_args() -> argparse.Namespace:
    """Đọc tùy chọn dòng lệnh của pipeline tiền xử lý.

    Returns:
        Đối tượng chứa đường dẫn tệp cấu hình.
    """
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu PRSA theo cấu hình YAML.")
    parser.add_argument(
        "--config", type=Path, default=ProjectConfiguration.DEFAULT_PATH,
        help="Đường dẫn YAML (mặc định: configs/default.yaml).",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--scaler-path", type=Path, default=None,
        help="Đường dẫn scaler đầu ra; ghi đè đường dẫn trong YAML.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Cho phép ghi đè split CSV và scaler đã tồn tại.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = ProjectConfiguration(arguments.config)
    if arguments.data_dir is not None:
        configuration.section("paths")["data_dir"] = str(arguments.data_dir)
    if arguments.output_dir is not None:
        configuration.section("paths")["processed_dir"] = str(arguments.output_dir)
    if arguments.scaler_path is not None:
        configuration.section("paths")["scaler_template"] = str(arguments.scaler_path)
    PreprocessingPipeline(configuration, overwrite=arguments.overwrite).run()