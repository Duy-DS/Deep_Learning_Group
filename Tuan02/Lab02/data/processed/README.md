# Data preprocessing handoff

Thư mục này là đầu ra bàn giao giữa nhóm EDA/preprocessing và người train model.
Các CSV ở đây đã được xử lý bởi `notebooks/02_data_preprocessing.ipynb`.
Người train model không cần impute, encode hoặc scale lại các file này.

## Nguồn quyết định

- `notebooks/01_eda.ipynb`
- `memory-bank/eda-findings.md`
- `memory-bank/systemPatterns.md`
- `memory-bank/project-management.md`

Repo hiện không có file PDF kiến trúc; luồng được đối chiếu theo các markdown trên.

```text
raw CSV
  -> semantic missing preparation
  -> train/validation split
  -> imputer + missing indicators
  -> categorical encoding
  -> numerical scaling
  -> model-ready matrices
```

## Các quyết định từ EDA

### Missing values

Raw CSV được đọc bằng `keep_default_na=False` để giữ literal `NA` và `None`.

- Basement absent khi `TotalBsmtSF == 0`.
- Garage absent khi `GarageCars == 0` và `GarageArea == 0`.
- Fireplace absent khi `Fireplaces == 0`.
- Pool absent khi `PoolArea == 0`.
- `Alley`, `Fence`, `MiscFeature` dùng `NA` trực tiếp cho trạng thái không có.

Structural absence categorical được chuyển thành category `None`.
Unknown missing categorical được đưa thành `Unknown`.
Unknown missing numeric được điền bằng median học từ training split và có missing indicator.

### Categorical encoding

`MSSubClass` và `MoSold` là numeric codes nhưng được xem là categorical.
Ordinal features được one-hot encode ở baseline để không áp đặt khoảng cách giả giữa các mức.

Encoder dùng `min_frequency=5` vì EDA phát hiện 54 level hiếm dưới 5 mẫu.
`handle_unknown="infrequent_if_exist"` xử lý level mới ở test, ví dụ `MSSubClass=150`.

### Numerical scaling

Numeric branch:

```text
median imputation -> missing indicators -> StandardScaler
```

`Id` không được đưa vào model. Thứ tự xử lý phù hợp cho Ridge và MLP.

### Target và validation

`SalePrice` được train ở log scale:

```python
y_log = np.log1p(SalePrice)
prediction = np.expm1(prediction_log)
```

Validation chiếm 20%, `random_state=42`. Imputer, encoder và scaler chỉ fit trên training split.

## Files

### `train_processed.csv`

Toàn bộ 1.460 dòng train sau preprocessing.

- `Id`: identifier để trace record.
- `SalePrice`: target gốc.
- `SalePrice_log`: target dùng để train.
- `split`: `train` hoặc `validation`.
- 314 cột còn lại: features đã impute, one-hot encode và scale.

Ví dụ dùng cho người train model:

```python
import pandas as pd

df = pd.read_csv("data/processed/train_processed.csv")
feature_columns = [
    c for c in df.columns
    if c not in {"Id", "SalePrice", "SalePrice_log", "split"}
]
train_df = df[df["split"] == "train"]
valid_df = df[df["split"] == "validation"]
X_train = train_df[feature_columns]
y_train = train_df["SalePrice_log"]
X_valid = valid_df[feature_columns]
y_valid = valid_df["SalePrice_log"]
```

### `test_processed.csv`

1.459 dòng test sau preprocessing, gồm `Id` và 314 features; không có target.

```python
test_df = pd.read_csv("data/processed/test_processed.csv")
X_test = test_df[feature_columns]
```

### `feature_names.csv`

Danh sách và thứ tự 314 output features. Thứ tự này phải được giữ nguyên khi tạo tensor/DataLoader cho MLP.

## Quy tắc sử dụng

- Không impute, encode hoặc scale lại các CSV processed.
- Không dùng `SalePrice` hoặc `SalePrice_log` làm input feature.
- Không trộn validation vào training khi đánh giá.
- Nếu đổi preprocessing, hãy chạy lại notebook để cập nhật toàn bộ artifacts.
- Khi inference từ raw data mới, ưu tiên dùng `outputs/preprocessing/ames_preprocessing_ridge_pipeline.pkl` để giữ đúng preprocessing object.

## Artifacts liên quan

Trong `outputs/preprocessing/`:

- `ames_preprocessing_ridge_pipeline.pkl`: fitted preprocessing + Ridge baseline.
- `transformed_feature_names.csv`: danh sách output features.
- `validation_predictions.csv`: prediction trên validation split.

Notebook đã ghi lại các rule EDA và tự tạo lại dữ liệu processed khi chạy từ đầu đến cuối.
