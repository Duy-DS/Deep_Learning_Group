# Preprocessing Handoff

## Trạng thái

Preprocessing đã hoàn thành và notebook đã chạy top-to-bottom không lỗi:

- Notebook: `notebooks/Data Preprocessing.ipynb`
- Pipeline: semantic missing preparation -> train/validation split -> imputer -> encoder -> scaler -> Ridge baseline
- Split: 80/20, `random_state=42`
- Train rows: 1,168
- Validation rows: 292
- Output features: 314
- Validation log-RMSE: 0.1437
- Validation log-MAE: 0.0963

## Quyết định được chốt

1. Đọc raw CSV với `keep_default_na=False` để phân biệt literal `NA` và `None`.
2. Structural absence được xác định theo feature đồng hành, không map `NA -> None` theo toàn bộ cột.
3. Numeric unknown missing dùng median và `add_indicator=True`.
4. Categorical unknown missing dùng `Unknown`; structural absence giữ category `None`.
5. `MSSubClass` và `MoSold` được xem là categorical code.
6. Categorical/ordinal baseline dùng `OneHotEncoder` với `min_frequency=5` và `handle_unknown="infrequent_if_exist"`.
7. Numeric dùng `StandardScaler` sau imputation.
8. Target dùng `log1p(SalePrice)`; prediction về scale gốc bằng `expm1`.
9. `Id` không phải feature model.
10. Không xóa outlier `GrLivArea > 4000` ở preprocessing baseline; giữ để đánh giá bằng experiment.

## Data handoff

Thư mục `data_processed/` chứa dữ liệu model-ready:

- `train_processed.csv`: 1,460 dòng, target gốc, target log, split label và 314 features.
- `test_processed.csv`: 1,459 dòng và 314 features.
- `feature_names.csv`: thứ tự 314 output features.
- `README.md`: hướng dẫn cho người train model.

Artifacts tại `outputs/preprocessing/`:

- `ames_preprocessing_ridge_pipeline.pkl`
- `transformed_feature_names.csv`
- `validation_predictions.csv`

## Quy tắc tái lập

Không fit lại imputer, encoder hoặc scaler trên validation/test. Khi thay đổi rule preprocessing, phải chạy lại notebook để cập nhật đồng bộ CSV, feature names, pipeline và validation artifacts.

## Việc tiếp theo

- Dùng `train_processed.csv` để xây PyTorch Dataset/DataLoader.
- So sánh Ridge với ElasticNet và MLP.
- Chạy experiment có kiểm soát cho rare-category grouping và outlier scenario.
- Giữ validation split và metric log-RMSE nhất quán giữa các model.
