# EDA and Data Understanding Findings

## Trạng thái xác minh

- Notebook: `notebooks/01_eda.ipynb`
- Ngày chạy: 2026-09-19
- Trạng thái: đã chạy top-to-bottom không lỗi
- Phạm vi: data understanding và EDA; chưa impute, xóa outlier hoặc train model
- Feature audit: đủ 79 explanatory features, không tính `Id` và `SalePrice`

## 1. Data integrity

| Kiểm tra | Kết quả |
| --- | ---: |
| Train shape | 1.460 x 81 |
| Test shape | 1.459 x 80 |
| Duplicate train rows | 0 |
| Duplicate train `Id` | 0 |
| Missing `SalePrice` | 0 |
| `SalePrice <= 0` | 0 |

Train và test có cùng feature schema; train chỉ có thêm target `SalePrice`.

## 2. Missing-value semantics

### Observation

`NA` trong CSV không luôn có nghĩa là dữ liệu chưa biết. Ý nghĩa phải được xác định theo từng dòng: basement, garage, fireplace và pool chỉ được gán structural absence khi feature đồng hành xác nhận tiện ích không tồn tại. `Alley`, `Fence` và `MiscFeature` dùng `NA` trực tiếp cho trạng thái không có.

`MasVnrType` có 864 giá trị literal `None` hợp lệ và 8 giá trị `NA`. Nếu dùng `pd.read_csv()` mặc định, cả hai token bị chuyển thành `NaN`, làm số missing bị báo sai thành 872.

### Evidence

- Raw train chứa 6.965 token `NA` trên 19 cột. Parser mặc định báo 7.829 missing vì gộp thêm 864 literal `None` của `MasVnrType`.
- Row-level rules nhận diện 6.687 structural-absence flags trên 15 features của train. Con số này là tổng theo feature, không phải số dòng duy nhất.
- Train còn 278 unknown missing trên 6 cột: `LotFrontage` 259, `MasVnrType` 8, `MasVnrArea` 8, `Electrical` 1, `BsmtExposure` 1 và `BsmtFinType2` 1.
- `GarageYrBlt` có 81 storage `NaN` trong train nhưng cả 81 đều đi cùng trạng thái không có garage; vì vậy chúng là structural absence, không phải unknown missing.
- Các anomaly không bị đổi nhầm thành `None`: train `Id=949` thiếu `BsmtExposure`, `Id=333` thiếu `BsmtFinType2`; test có missing trên basement dù basement tồn tại, garage tại `Id=2127`, và `PoolQC` tại `Id=2421`, `2504`, `2600` dù `PoolArea > 0`.
- Test còn 303 unknown missing trên 28 cột sau khi loại structural absence.

### Implication

Không được map `NA -> None` theo toàn bộ cột hoặc dùng một chiến lược imputation chung. Pipeline phải tái sử dụng cùng row-level rules, xử lý được missing chỉ xuất hiện ở test và không làm mất category literal `None` của `MasVnrType`.

### Recommended action

- Đọc dữ liệu với quy tắc NA có kiểm soát hoặc phục hồi semantic categories trước preprocessing.
- Map categorical structural absence sang `None` theo feature đồng hành, không theo tên cột đơn thuần.
- Với `GarageYrBlt`, giữ representation numeric có thể là `NaN` nhưng tạo structural-absence indicator từ trạng thái garage; không coi 81 dòng train này là unknown missing.
- Fit imputer trên training split rồi transform validation/test.

### Missing indicators và target

`missing_target_effect.csv` so sánh median `log1p(SalePrice)` giữa nhóm có/không có indicator. Không có garage và không có basement lần lượt có median log-target thấp hơn khoảng `0,516` và `0,483`; không có fireplace thấp hơn khoảng `0,347`. Đây là quan hệ mô tả, không phải bằng chứng nhân quả. Các indicator chỉ có 1-8 dòng, đặc biệt unknown missing, không đủ ổn định để tự quyết định feature engineering.

## 3. Target distribution

| Statistic | `SalePrice` | `log1p(SalePrice)` |
| --- | ---: | ---: |
| Mean | 180.921,20 | 12,024 |
| Median | 163.000,00 | 12,002 |
| Minimum | 34.900 | 10,460 |
| Maximum | 755.000 | 13,534 |
| Skewness | 1,883 | 0,121 |

### Finding

`SalePrice` lệch phải mạnh; `log1p` làm phân phối gần đối xứng và phù hợp với metric log-RMSE của Kaggle.

### Recommended action

Thử `log1p(SalePrice)` trong baseline và MLP. Đây vẫn là `Proposed`; quyết định chính thức cần dựa trên validation.

## 4. Feature-target relationships

Các quan hệ mạnh nhất với `log1p(SalePrice)`:

| Feature | Pearson | Spearman |
| --- | ---: | ---: |
| `OverallQual` | 0,817 | 0,810 |
| `GrLivArea` | 0,701 | 0,731 |
| `GarageCars` | 0,681 | 0,691 |
| `YearBuilt` | 0,587 | 0,653 |
| `GarageArea` | 0,651 | 0,649 |
| `FullBath` | 0,595 | 0,636 |
| `TotalBsmtSF` | 0,612 | 0,603 |
| `1stFlrSF` | 0,597 | 0,575 |

`Neighborhood` và `OverallQual` cho thấy mức giá trung vị khác biệt đáng kể giữa các nhóm. Khi trình bày categorical effects cần kèm sample size vì một số neighborhood có ít quan sát.

## 5. Redundancy

Các cặp numeric tương quan cao nhất:

| Feature pair | Correlation |
| --- | ---: |
| `GarageCars` - `GarageArea` | 0,882 |
| `YearBuilt` - `GarageYrBlt` | 0,826 |
| `GrLivArea` - `TotRmsAbvGrd` | 0,825 |
| `TotalBsmtSF` - `1stFlrSF` | 0,820 |

Không loại feature chỉ dựa trên pairwise correlation. Baseline cần theo dõi multicollinearity hoặc dùng regularization; MLP cần kiểm soát capacity và overfitting.

## 6. Outliers

Bốn căn có `GrLivArea > 4.000`:

| Id | GrLivArea | SalePrice |
| ---: | ---: | ---: |
| 524 | 4.676 | 184.750 |
| 692 | 4.316 | 755.000 |
| 1183 | 4.476 | 745.000 |
| 1299 | 5.642 | 160.000 |

`Id=524` và `Id=1299` là hai điểm có diện tích rất lớn nhưng giá thấp bất thường. EDA không xóa các dòng này.

Recommended action: chạy ít nhất hai thí nghiệm giữ toàn bộ và loại hai điểm rõ nhất; chỉ chọn quy tắc dựa trên validation, không dựa trên trực giác hoặc Kaggle leaderboard đơn lẻ.

## 7. Train-test comparison

- Numeric SMD cao nhất: `TotRmsAbvGrd = 0,082`.
- Categorical TVD cao nhất: `Neighborhood = 0,072`.
- Các score còn lại đều dưới các giá trị trên trong audit hiện tại.

Không phát hiện dịch chuyển đơn biến lớn theo hai chỉ báo này. Kết quả không loại trừ shift đa biến và không cho phép dùng test để fit preprocessing.

### Category-level audit

- Audit ghi nhận 331 mức trên categorical/ordinal features: 297 shared, 19 train-only và 15 test-only.
- Có 54 mức xuất hiện dưới 5 lần trong train; đây là danh sách cần xem xét khi quyết định gộp rare categories, chưa phải quyết định gộp tự động.
- Mức hợp lệ chỉ có ở test là `MSSubClass=150` với 1 dòng. Các test-only còn lại trong audit là `<MISSING>` ở 14 features, phản ánh unknown missing chứ không phải category mới.
- Encoder phải hỗ trợ unseen level; audit không được dùng để fit category vocabulary trên test.

## 8. Handoff cho preprocessing

1. Bảo toàn literal `None` của `MasVnrType` khi đọc CSV.
2. Tách structural absence khỏi unknown missing.
3. Xử lý `MSSubClass` và `MoSold` như categorical codes.
4. Dùng `category_level_audit.csv` để review rare/test-only levels, nhưng chỉ fit encoder vocabulary trên training split.
5. Fit imputer, encoder và scaler chỉ trên training split.
6. Dùng `handle_unknown="ignore"` hoặc cơ chế tương đương cho test categories.
7. Thử log target và outlier scenarios như hai quyết định validation riêng biệt.
8. Ghi lại số chiều sau encoding vì tập train chỉ có 1.460 dòng.

## 9. Artifacts

- `outputs/metrics/feature_audit.csv`: semantic type, cardinality và missing train/test.
- `outputs/metrics/missing_audit.csv`: raw `NA`, structural absence và unknown missing.
- `outputs/metrics/missing_target_effect.csv`: chênh lệch target theo missing/absence indicator.
- `outputs/metrics/category_level_audit.csv`: mức category, train/test coverage và rare flag.
- `outputs/metrics/target_correlations.csv`: Pearson/Spearman với log target.
- `outputs/metrics/train_test_drift.csv`: SMD và TVD.
- `outputs/figures/eda_*.png`: chín figures phục vụ report.

## 10. Quyết định còn mở

- `DEC-005`: target transform và preprocessing chính thức vẫn là `Proposed`.
- Chưa quyết định loại outlier.
- Chưa quyết định encoding ordinal theo thứ tự hay one-hot.
- Chưa quyết định gộp rare categories.
