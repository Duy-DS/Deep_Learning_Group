# EDA and Data Understanding Plan

## Mục tiêu

Hoàn thành `T03` và `T04` bằng một quy trình EDA có thể chạy lại, cung cấp bằng chứng cho preprocessing, baseline và báo cáo CRISP-DM.

## Câu hỏi cần trả lời

1. Schema và dữ liệu train/test có nhất quán không?
2. Mỗi feature thuộc semantic type nào?
3. Missing nào là thiếu dữ liệu thật, missing nào biểu diễn "không có"?
4. `SalePrice` phân phối thế nào và `log1p` thay đổi phân phối ra sao?
5. Feature nào liên hệ mạnh với target?
6. Outlier nào cần đánh dấu để kiểm tra độ nhạy?
7. Feature nào dư thừa hoặc có tương quan cao?
8. Train/test có dấu hiệu distribution shift không?
9. EDA dẫn đến đề xuất preprocessing nào?

## Đầu ra

| Artifact | Vai trò |
| --- | --- |
| `notebooks/01_eda.ipynb` | Phân tích có thể chạy lại từ đầu đến cuối. |
| `outputs/figures/eda_*.png` | Hình dùng cho report và thảo luận nhóm. |
| `outputs/metrics/feature_audit.csv` | Schema, semantic type, cardinality và missing của 79 predictors. |
| `memory-bank/eda-findings.md` | Insight, rủi ro và đề xuất bàn giao cho preprocessing. |

## Trình tự thực hiện

### 1. Load và kiểm tra dữ liệu

- Đọc một bản raw với `keep_default_na=False` để giữ nguyên token `NA` và `None`.
- Phân biệt structural absence với missing thật dựa trên `data_description.txt`.
- Xác nhận shape, schema, duplicate row, duplicate `Id` và target missing.
- Không chỉnh sửa file trong `data/`.

### 2. Feature audit

- Phân loại identifier, categorical, ordinal, temporal, numeric continuous và numeric discrete.
- Ghi dtype, số mức, missing count, missing rate và số lượng zero.
- Xuất bảng audit để TV3 sử dụng khi xây preprocessing.

### 3. Target analysis

- So sánh `SalePrice` và `log1p(SalePrice)` bằng thống kê và biểu đồ.
- Ghi mean, median, quartiles, skewness và kurtosis.
- Liên hệ kết quả với metric log-RMSE của Kaggle.

### 4. Missing-value analysis

- Xếp hạng missing rate theo cột.
- Kiểm tra co-missing trong nhóm basement và garage.
- So sánh train/test và quan hệ giữa missing indicator với target.
- Chỉ đề xuất chiến lược; không impute trong EDA.

### 5. Univariate và bivariate analysis

- Phân tích phân phối numeric, zero-inflated features, cardinality và rare categories.
- Tính Pearson và Spearman correlation với target.
- Phân tích các feature quan trọng bằng scatterplot, boxplot và grouped median.

### 6. Redundancy và outlier analysis

- Kiểm tra các cặp feature tương quan cao.
- Phân tích `GrLivArea > 4000` và gắn nhãn các quan sát liên quan.
- Không xóa outlier; bàn giao hai kịch bản giữ/loại cho thí nghiệm sau.

### 7. Train-test comparison

- So sánh range, missing rate và category set.
- Dùng standardized mean difference cho numeric và total variation distance cho categorical.
- Không dùng test set để fit hoặc chọn preprocessing parameter.

### 8. Tổng hợp và bàn giao

Mỗi kết luận dùng cấu trúc `Observation -> Evidence -> Implication -> Action`. Findings phải chỉ rõ quyết định nào đã có bằng chứng và vấn đề nào còn cần thí nghiệm.

## Definition of Done

- Notebook chạy top-to-bottom không lỗi.
- Audit đủ 79 predictors.
- Phân biệt đúng token `NA`, `None` và missing thật.
- Phân tích target ở original và log scale.
- Có missing, feature-target, redundancy, outlier và train-test shift analysis.
- Figures có title, axis labels, đơn vị và độ phân giải phù hợp.
- Raw data không bị thay đổi.
- `eda-findings.md` đủ để TV3 thiết kế preprocessing mà không phải đọc lại toàn notebook.
