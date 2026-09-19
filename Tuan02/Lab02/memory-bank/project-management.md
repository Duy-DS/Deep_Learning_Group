# Project Management Document

## House Prices Prediction Using PyTorch

**Lab 02 - Deep Learning | Kaggle House Prices: Advanced Regression Techniques**

| Thông tin | Nội dung |
| --- | --- |
| Môn học | Deep Learning |
| Nhóm | Nhóm 5 thành viên |
| Nhóm trưởng / Thành viên 1 | Nguyễn Tấn Duy - 3123580008 |
| Phiên bản tài liệu | v1.0 - 17/09/2026 |
| Trạng thái | Data Understanding / EDA Review |

Tài liệu dùng để điều hành, theo dõi và lưu lại quyết định của dự án.

## 1. About Project

### 1.1 Project Name

House Prices Prediction Using PyTorch.

### 1.2 Topic

Deep Learning for Tabular Regression - dự đoán giá bán nhà từ dữ liệu bất động sản Ames Housing.

### 1.3 Project Description

Dự án xây dựng một quy trình học máy hoàn chỉnh cho bài toán dự đoán giá nhà trong cuộc thi Kaggle "House Prices: Advanced Regression Techniques". Nhóm sẽ nghiên cứu bộ dữ liệu Ames Housing, thực hiện khám phá và tiền xử lý dữ liệu, xây dựng baseline, sau đó phát triển mô hình Multilayer Perceptron (MLP) bằng PyTorch để dự đoán `SalePrice`. Kết quả được đánh giá trên tập validation nội bộ và qua điểm số Kaggle; toàn bộ quá trình được trình bày theo CRISP-DM.

### 1.4 Motivation

- Thực hành quy trình giải quyết một bài toán regression thực tế với dữ liệu tabular có nhiều kiểu biến và missing values.
- Áp dụng kiến thức Deep Learning bằng cách tự xây dựng MLP trong PyTorch thay vì chỉ sử dụng thư viện mô hình mức cao.
- Rèn luyện kỹ năng EDA, data preprocessing, đánh giá mô hình, kiểm soát overfitting và tái lập thí nghiệm.
- Tham gia Kaggle để kiểm tra khả năng tổng quát hóa của mô hình trên dữ liệu ẩn.

## 2. Problem Statement

Từ các đặc trưng mô tả một căn nhà tại Ames, Iowa, có thể xây dựng mô hình MLP bằng PyTorch để dự đoán giá bán (`SalePrice`) với sai số thấp và khả năng tổng quát hóa tốt hay không?

| Thuộc tính | Nội dung |
| --- | --- |
| Loại bài toán | Supervised Learning - Regression |
| Input | Đặc trưng số và phân loại về căn nhà, lô đất, chất lượng, vị trí, tiện ích, garage, basement, năm xây dựng, v.v. |
| Target | `SalePrice` |
| Thách thức | Dữ liệu numeric/categorical hỗn hợp, missing values, nhiều mức categorical sau one-hot encoding, target lệch phải và nguy cơ overfitting trên tập train nhỏ. |

## 3. Project Objectives

### 3.1 General Objective

Xây dựng và đánh giá một pipeline dự đoán giá nhà sử dụng MLP trong PyTorch, tuân theo CRISP-DM và đủ khả năng tạo submission hợp lệ cho Kaggle House Prices.

### 3.2 Specific Objectives

- Đọc và tóm tắt paper Ames Housing để hiểu nguồn gốc dữ liệu, kiểu biến và các vấn đề mô hình hóa.
- Thực hiện EDA để nhận diện phân phối `SalePrice`, missing values, outliers và các feature có quan hệ mạnh với target.
- Xây dựng pipeline preprocessing có thể tái sử dụng cho train, validation và test.
- Xây dựng tối thiểu một baseline model để có mốc so sánh.
- Thiết kế, huấn luyện và lưu mô hình MLP bằng PyTorch.
- Đánh giá bằng RMSE, MAE, R2 trên validation và Kaggle leaderboard score trên test.
- Ghi lại thí nghiệm, quyết định, rủi ro và kết quả để đảm bảo khả năng tái lập.
- Hoàn thiện source code, báo cáo CRISP-DM và gói nộp theo yêu cầu giảng viên.

## 4. Scope

### 4.1 In Scope

- Dataset Kaggle House Prices / Ames Housing.
- EDA, xử lý missing values, encoding categorical và scaling numerical features.
- Train/validation split và DataLoader cho PyTorch.
- Baseline regression và MLP PyTorch.
- Hyperparameter experiments ở mức phù hợp với Lab 02.
- Đánh giá nội bộ, tạo `submission.csv` cho Kaggle và báo cáo theo CRISP-DM.

### 4.2 Out of Scope

- Web/mobile/production API hoàn chỉnh.
- Hệ thống định giá bất động sản thương mại ngoài dữ liệu Ames.
- Thu thập dữ liệu mới hoặc scraping dữ liệu bất động sản.
- Computer Vision, xử lý ảnh, distributed training, MLOps production và real-time inference.
- Stacking/ensemble competition-grade, trừ khi còn thời gian.

## 5. Expected Outputs

| Deliverable | Mô tả | Bắt buộc |
| --- | --- | --- |
| Source code | Notebook/scripts cho EDA, preprocessing, training, evaluation, inference | Có |
| PyTorch MLP | Định nghĩa model và checkpoint tốt nhất (`.pth`) | Có |
| Evaluation results | Metrics, learning curves, actual-vs-predicted/residual plots | Có |
| Kaggle submission | `submission.csv` hợp lệ và ảnh/ghi chú điểm số | Có |
| Project report | Trang bìa, phân công, CRISP-DM, kết quả và thảo luận | Có |
| README + requirements | Hướng dẫn chạy và phiên bản thư viện | Khuyến nghị |
| Project management document | Task, milestone, decision log, progress log, meeting notes | Nội bộ |

## 6. Functional Requirements

| ID | Yêu cầu chức năng |
| --- | --- |
| FR-01 | Đọc được `train.csv`, `test.csv` và `sample_submission.csv`. |
| FR-02 | Phân tích schema, dtype, missing values và thống kê mô tả. |
| FR-03 | Tiền xử lý numeric/categorical nhất quán giữa train, validation và test. |
| FR-04 | Tạo train/validation split có random state để tái lập. |
| FR-05 | Chuyển dữ liệu sang PyTorch Tensor/Dataset/DataLoader. |
| FR-06 | Huấn luyện MLP và lưu train/validation loss theo epoch. |
| FR-07 | Lưu checkpoint mô hình tốt nhất dựa trên validation metric. |
| FR-08 | Tính RMSE, MAE và R2 trên validation. |
| FR-09 | Sinh dự đoán cho `test.csv` và tạo `submission.csv` đúng định dạng Kaggle. |
| FR-10 | Chạy lại quy trình từ raw data đến kết quả cuối bằng hướng dẫn trong README. |

## 7. Non-Functional Requirements

| Nhóm yêu cầu | Mô tả |
| --- | --- |
| Reproducibility | Cố định random seed; lưu requirements; ghi preprocessing, hyperparameters và split. |
| Maintainability | Tách preprocessing, model, training và evaluation thành module/notebook section rõ ràng. |
| Readability | Tên biến/hàm rõ nghĩa; docstring/comment tại logic quan trọng; tránh cell/code thừa. |
| Correctness | Không data leakage; fit scaler/encoder trên train rồi apply cho validation/test. |
| Traceability | Mỗi thí nghiệm có ID hoặc bảng ghi hyperparameters, metric và kết luận. |
| Usability | Có README hoặc cell hướng dẫn chạy theo thứ tự; dùng đường dẫn dữ liệu tương đối. |
| Efficiency | Mô hình chạy được trên CPU/GPU phổ biến trong thời gian phù hợp cho lab. |

## 8. Proposed Solution

Giải pháp gồm ba lớp: (1) data understanding và preprocessing; (2) baseline và MLP modeling; (3) evaluation và Kaggle submission.

- Ưu tiên thử `log1p(SalePrice)` để giảm độ lệch; dùng `expm1` khi tạo submission.
- Numerical: impute bằng median/giá trị phù hợp, sau đó `StandardScaler`.
- Categorical: impute bằng mode hoặc nhãn `None` khi missing mang nghĩa không có; one-hot encoding.
- Baseline: Linear Regression, Ridge hoặc mô hình cây.
- MLP: fully-connected layers với ReLU; thử Dropout và/hoặc BatchNorm khi cần.
- Optimization: Adam/AdamW; theo dõi validation loss và checkpoint tốt nhất.
- Evaluation: RMSE, MAE, R2 nội bộ; leaderboard score dùng để kiểm tra test ẩn.

Các lựa chọn trên là kế hoạch ban đầu. Mọi thay đổi quan trọng sau EDA và thí nghiệm phải ghi vào Decision Log.

## 9. System Pipeline

```text
Raw Kaggle Data
  -> Data Understanding / EDA
  -> Preprocessing
  -> Train / Validation Split
  -> Baseline
  -> PyTorch MLP
  -> Evaluation
  -> Final Training / Inference
  -> submission.csv
  -> Kaggle
  -> CRISP-DM Report
```

1. Load và kiểm tra dữ liệu.
2. EDA target, missing values, outliers, correlation và categorical distributions.
3. Fit preprocessing pipeline trên training split.
4. Chia train/validation, chuyển sang tensor và chạy baseline.
5. Train các cấu hình MLP có kiểm soát, so sánh metrics, chọn checkpoint tốt nhất.
6. Predict test, tạo submission và tổng hợp kết quả vào báo cáo.

## 10. Technical Components

| Component | Trách nhiệm |
| --- | --- |
| `data_loader` | Đọc file dữ liệu và kiểm tra schema. |
| `eda` | Missing values, distribution, correlation, outliers và feature-target relationships. |
| `preprocessing` | Imputation, encoding, scaling, target transform; tránh leakage. |
| `dataset` | PyTorch Dataset/DataLoader hoặc TensorDataset. |
| `model` | Định nghĩa `HousePriceMLP`. |
| `trainer` | Training loop, validation loop, early stopping/checkpoint, logging. |
| `evaluation` | RMSE/MAE/R2, plots và error analysis. |
| `inference` | Load checkpoint, predict test, inverse-transform target. |
| `submission` | Tạo file `Id,SalePrice` đúng format. |
| `reporting` | Bảng thí nghiệm, hình ảnh, kết quả Kaggle và CRISP-DM report. |

## 11. Dataset

Nguồn chính: Kaggle "House Prices: Advanced Regression Techniques", dựa trên Ames Housing dataset của Dean De Cock.

| Tập dữ liệu | Kích thước theo code mẫu | Vai trò |
| --- | --- | --- |
| `train.csv` | 1460 dòng x 81 cột | Huấn luyện; chứa `SalePrice`. |
| `test.csv` | 1459 dòng x 80 cột | Dự đoán; không chứa target. |
| `sample_submission.csv` | Theo Kaggle | Mẫu định dạng submission. |

Paper gốc mô tả dataset Ames Housing sau làm sạch gồm 2.930 quan sát và 80 biến, bao gồm 23 nominal, 23 ordinal, 14 discrete và 20 continuous variables. Phiên bản Kaggle dùng train/test split khác với dataset gốc.

### 11.1 Data Issues to Track

- Missing values có thể mang nghĩa "không có" với basement, garage, pool và các feature liên quan.
- Categorical features có nhiều mức; one-hot encoding làm tăng số chiều.
- `SalePrice` lệch phải; cần kiểm tra log transform.
- Các căn có `GrLivArea` rất lớn có thể là outlier; kiểm tra trước khi xử lý.
- Không fit preprocessing trên toàn bộ train trước khi tách validation nếu điều đó gây leakage.

## 12. Technology Stack

| Category | Technology |
| --- | --- |
| Programming Language | Python 3.x |
| Data Processing | pandas, numpy, scikit-learn |
| Deep Learning | PyTorch |
| Baseline / Comparison | scikit-learn; XGBoost nếu nhóm sử dụng |
| Visualization | matplotlib, seaborn |
| Notebook | Jupyter Notebook / Google Colab |
| Version Control | Git + GitHub |
| Project Management | Google Docs + task table / Notion hoặc GitHub Projects nếu nhóm dùng |
| Competition Platform | Kaggle |

## 13. Project Structure

```text
Lab02/
├── data/          # train.csv, test.csv, sample_submission.csv
├── docs/          # tài liệu tham khảo và báo cáo
├── notebooks/     # EDA và experiments
├── outputs/       # figures, metrics, submission.csv
├── src/           # preprocessing, dataset, model, train, evaluate, predict
├── memory-bank/   # project management và project context
└── README.md
```

Các thư mục `models/`, `outputs/figures/`, `outputs/metrics/` và các module trong `src/` sẽ chỉ được tạo khi bắt đầu phần việc tương ứng.

## 14. Task Breakdown

| ID | Task | Module | Assignee | Status | Priority | Deadline | Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T01 | Đọc paper Ames Housing, tóm tắt bài toán/dataset | Understanding | TV1 | Done | High | TBD | [Research notes](ames-housing-paper-summary.md) |
| T02 | Chuẩn hóa tài liệu điều hành, scope/objectives | Management | TV1 | Doing | High | TBD | Project doc |
| T03 | EDA schema, target, missing values | EDA | TV2 | Review | High | TBD | [`01_eda.ipynb`](../notebooks/01_eda.ipynb) |
| T04 | EDA correlation/outlier/feature relationships | EDA | TV2 | Review | Medium | TBD | [EDA findings](eda-findings.md) |
| T05 | Thiết kế preprocessing numeric/categorical | Preprocess | TV3 | To Do | High | TBD | Pipeline/module |
| T06 | Train/validation split và leakage check | Preprocess | TV3 | To Do | High | TBD | Prepared data |
| T07 | Xây baseline model | Baseline | TV5 | To Do | Medium | TBD | Baseline metrics |
| T08 | Xây PyTorch Dataset/DataLoader | Modeling | TV4 | To Do | High | TBD | Dataset code |
| T09 | Xây MLP PyTorch | Modeling | TV4 | To Do | High | TBD | `model.py` |
| T10 | Training loop, checkpoint, curves | Training | TV4 | To Do | High | TBD | Best model + logs |
| T11 | Experiment/tuning có kiểm soát | Experiment | TV4 + TV5 | To Do | Medium | TBD | Experiment table |
| T12 | Evaluation RMSE/MAE/R2 và error analysis | Evaluation | TV5 | To Do | High | TBD | Metrics + figures |
| T13 | Tạo submission và submit Kaggle | Deployment | TV5 | To Do | High | TBD | Submission + score |
| T14 | Viết báo cáo CRISP-DM | Report | TV1 + All | To Do | High | TBD | Report draft |
| T15 | Review code, README, requirements, đóng gói | QA | All | To Do | High | TBD | Final ZIP |

Quy ước trạng thái: `To Do` -> `Doing` -> `Review` -> `Done`. Deadline hiện là `TBD`; nhóm trưởng cập nhật sau khi chốt lịch môn học.

## 15. Team Assignment

| Member | Main Responsibility | Supporting Responsibility |
| --- | --- | --- |
| TV1 - Nhóm trưởng / Nguyễn Tấn Duy | Business & Data Understanding; quản lý tiến độ; tổng hợp report | Review tính thống nhất code-report; quản lý decision/progress log |
| TV2 - Nguyễn Công Trí | EDA & Data Understanding | Chuẩn bị figures và insight cho report |
| TV3 - Trần Chí Vỹ | Data Preprocessing | Reproducibility, leakage checks, bàn giao processed data |
| TV4 - Nguyễn Thành Đạt | PyTorch MLP & Training | Experiment logging, checkpoint, training curves |
| TV5 - Đinh Thái Toàn | Baseline, Evaluation & Kaggle | Hyperparameter comparison, submission, result table |

## 16. Milestones

| Milestone | Target / Definition of Done | Status |
| --- | --- | --- |
| M1 - Project Setup | Tài liệu điều hành, repo/folder, phân công, dataset sẵn sàng | Done |
| M2 - Understanding & EDA | Paper notes, EDA notebook và issues list | Review |
| M3 - Preprocessing Ready | Pipeline fit/transform nhất quán train/val/test | To Do |
| M4 - Baseline Ready | Có baseline metrics để so sánh | To Do |
| M5 - MLP v1 | MLP PyTorch train được và có validation curve | To Do |
| M6 - Best Experiment | Chọn cấu hình tốt nhất dựa trên validation | To Do |
| M7 - Kaggle Submission | `submission.csv` hợp lệ và ghi nhận leaderboard score | To Do |
| M8 - Final Delivery | Report, source, model/output, README, ZIP đúng format | To Do |

## 17. Experiment & Evaluation Plan

### 17.1 Metrics

- RMSE là metric chính trên validation. Nếu model dùng `log1p(SalePrice)`, phải ghi rõ metric tính ở log scale hay original scale.
- MAE bổ sung để diễn giải sai số tuyệt đối trung bình.
- R2 đo mức độ giải thích biến thiên trên validation.
- Kaggle leaderboard score chỉ kiểm chứng trên test ẩn, không thay thế validation nội bộ.

### 17.2 Experiment Matrix

| Experiment | Model/Change | Mục đích | Kết quả cần lưu |
| --- | --- | --- | --- |
| EXP-00 | Baseline | Tạo mốc so sánh | RMSE/MAE/R2 |
| EXP-01 | MLP cơ bản | Xác nhận pipeline PyTorch hoạt động | Train/val loss + metrics |
| EXP-02 | Target `log1p` | Kiểm tra tác động target transform | Metrics + curves |
| EXP-03 | Dropout/BatchNorm | Giảm overfitting | Metrics + gap train/val |
| EXP-04 | Depth/width khác nhau | Tìm capacity phù hợp | Metrics + training time |
| EXP-05 | Learning rate / optimizer | Ổn định convergence | Best epoch + metrics |

### 17.3 Model Selection Rule

Chọn model trước hết dựa trên validation performance và tính ổn định. Kaggle score dùng để ghi nhận kết quả cuối, không phải lý do duy nhất để chọn cấu hình.

## 18. Risks & Issues

| Risk / Issue | Impact | Mitigation |
| --- | --- | --- |
| Data leakage khi preprocessing | Validation score ảo, Kaggle giảm | Fit imputer/scaler/encoder trên train split; review pipeline. |
| Categorical encoding không khớp train/test | Lỗi shape hoặc inference | Fit encoder một lần; `handle_unknown="ignore"` hoặc concat logic có kiểm soát. |
| Overfitting do mẫu nhỏ, feature nhiều | Validation loss tăng | Dropout, weight decay, early stopping, đơn giản hóa MLP. |
| Target skew / outliers | Loss bị chi phối bởi giá cao | Kiểm tra `log1p` target và outlier analysis. |
| Mỗi thành viên dùng preprocessing khác nhau | Kết quả không ghép được | TV3 cung cấp pipeline chuẩn; mọi experiment dùng cùng interface. |
| Code chỉ chạy trên máy tác giả | Khó nộp/kiểm tra | Requirements, seed, relative paths, README. |
| Tuning quá nhiều, thiếu thời gian report | Trễ deadline | Giới hạn experiment matrix; ưu tiên deliverables bắt buộc. |
| Kaggle submission sai format | Không có score | Dùng sample submission và validate cột `Id`, `SalePrice`. |

## 19. Decision Log

Chỉ ghi các quyết định đã được nhóm hoặc giảng viên chốt. Đề xuất chưa chốt phải để trạng thái `Proposed`.

| Date | ID | Decision | Reason | Status |
| --- | --- | --- | --- | --- |
| 17/09/2026 | DEC-001 | Dùng Kaggle House Prices làm dữ liệu/competition chính | Yêu cầu Lab 02 của giảng viên | Fixed |
| 17/09/2026 | DEC-002 | Mô hình Deep Learning chính là MLP bằng PyTorch | Yêu cầu Lab 02 của giảng viên | Fixed |
| 17/09/2026 | DEC-003 | Báo cáo tổ chức theo CRISP-DM | Yêu cầu Lab 02 của giảng viên | Fixed |
| 17/09/2026 | DEC-004 | Dùng tài liệu này làm nguồn điều hành nhóm | Giảm quên nhiệm vụ/quyết định; hỗ trợ truy vết | Active |
| TBD | DEC-005 | Target transform / preprocessing chính thức | Chốt sau EDA và baseline | Proposed |

### Decision Template

```text
Decision ID:
Date:
Problem:
Options considered:
Decision:
Reason:
Impact:
Can this decision be revisited? Yes / No
```

## 20. Progress Log

### 2026-09-17

| Category | Nội dung |
| --- | --- |
| Completed | Đã nhận yêu cầu Lab 02; xác định dự án Kaggle House Prices; thống nhất nhóm 5 người; có paper và code tham khảo. |
| In Progress | TV1 hoàn thiện Project Management Document; đọc paper và xác định problem/objectives/scope. |
| Next | Chốt thông tin thành viên và deadline; tạo repo/folder; TV2 bắt đầu EDA; TV3 thiết kế preprocessing interface. |
| Issues | Chưa có deadline chính thức trong tài liệu hiện tại; tên/MSSV thành viên chưa được điền. |

### 2026-09-19

| Category | Nội dung |
| --- | --- |
| Completed | Hoàn thành T01; xây dựng và chạy `01_eda.ipynb`; audit đủ 79 features; xuất 9 figures và 7 metric tables; sửa missing semantics theo row-level rules; bổ sung category-level và missing-target audits. |
| In Progress | T03/T04 ở trạng thái `Review`; đồng bộ findings và chuẩn bị bàn giao artifacts cho TV3. |
| Next | Nhóm review EDA; sau khi chấp nhận chuyển T03/T04 và M2 sang `Done`; TV3 bắt đầu preprocessing pipeline và leakage checks. |
| Issues | Deadline chính thức vẫn là `TBD`; chưa chốt target transform, ordinal encoding, rare-category handling và outlier rule. |

### Progress Log Template

| Date | Completed | In Progress | Next | Issues |
| --- | --- | --- | --- | --- |
| YYYY-MM-DD | ... | ... | ... | ... |

## 21. Meeting Notes

```text
Meeting - YYYY-MM-DD
Participants:
Objectives:
Discussion:
Decisions:
Completed since previous meeting:
Current blockers:
Next meeting:
```

| Task | Owner | Deadline |
| --- | --- | --- |
| ... | ... | ... |

## 22. Open Questions

| ID | Question | Owner | Status |
| --- | --- | --- | --- |
| Q-01 | Deadline chính thức của Lab 02 là ngày nào? | TV1 | Open |
| Q-02 | Nhóm có cần bắt buộc baseline ngoài MLP hay chỉ khuyến nghị? | TV1 | Open |
| Q-03 | Giảng viên mong muốn metric nào trong report ngoài Kaggle score? | TV1 | Open |
| Q-04 | Có yêu cầu cụ thể về số lượng experiment/hyperparameter tuning không? | TV4/TV5 | Open |
| Q-05 | Nhóm xử lý missing categorical mang nghĩa "không có" theo từng cột hay mode chung? | TV3 | Open |

## 23. Future Improvements

- So sánh MLP với Gradient Boosting/XGBoost để phân tích giới hạn của MLP.
- Thử embedding cho categorical features thay vì one-hot encoding.
- Dùng K-fold cross-validation thay cho một train/validation split.
- Feature engineering có cơ sở domain: total square footage, total bathrooms, house age.
- Ensemble/stacking nếu mở rộng mục tiêu Kaggle leaderboard.
- Dùng MLflow/W&B nếu dự án phát triển lớn hơn.

## 24. References

### Dataset / Competition

- Kaggle. [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

### Paper

- De Cock, D. (2011). [Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project](https://jse.amstat.org/v19n3/decock.pdf). *Journal of Statistics Education*, 19(3).

### Provided Reference Code

House Prices: Advanced Regression Techniques reference EDA/preprocessing/modeling code provided with the lab materials. The example includes XGBoost, Decision Tree and a Keras ANN; the required neural-network implementation will be rewritten in PyTorch.

## 2026-09-19 Preprocessing status update

T05 (thiết kế preprocessing numeric/categorical) và T06 (train/validation split, leakage check) đã hoàn thành ở mức baseline. `Data Preprocessing.ipynb` đã chạy thành công và tạo pipeline có semantic missing rules, median imputation, missing indicators, one-hot encoding, `StandardScaler`, split 80/20 với seed 42 và target `log1p(SalePrice)`.

Kết quả baseline: 314 output features, validation log-RMSE `0.1437`, validation log-MAE `0.0963`. Dữ liệu bàn giao nằm trong `data_processed/`; fitted pipeline, feature names và validation predictions nằm trong `outputs/preprocessing/`. Chi tiết contract được ghi trong `memory-bank/preprocessing-handoff.md`.

T03/T04 đã có notebook, findings và artifacts; việc review EDA vẫn cần nhóm xác nhận chính thức. Bước tiếp theo của project là Dataset/DataLoader, MLP và baseline comparison.

### Project Requirements

- Tham gia Kaggle competition.
- Đọc paper để xác định bài toán.
- Dựa vào code đính kèm để xây dựng mô hình.
- Xây dựng MLP trong PyTorch.
- Viết báo cáo theo CRISP-DM và nộp source code quá trình chạy + kiểm tra.
- File nộp: `lab02_house_price_hoten_masv.zip` theo yêu cầu giảng viên.

## Appendix A - Ames Housing Paper Summary

Bản tóm tắt paper và các insight phục vụ EDA, preprocessing, modeling và evaluation được lưu riêng tại [Ames Housing Paper Summary](ames-housing-paper-summary.md) để tài liệu điều hành tập trung vào quản lý dự án.
