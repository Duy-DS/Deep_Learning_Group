# BÁO CÁO KHOA HỌC: DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN AMES BẰNG MẠNG NƠ-RON MULTILAYER PERCEPTRON (PYTORCH) THEO TIẾN TRÌNH CRISP-DM

---

## THÔNG TIN CHUNG

* **Cơ quan đào tạo:** [Tên Trường Đại học] – [Tên Khoa / Bộ môn]
* **Học phần:** Deep Learning (Học sâu)
* **Mã lớp học phần:** [Điền Mã Lớp Học Phần]
* **Giảng viên hướng dẫn:** [Họ và tên Giảng viên]
* **Đề tài:** House Prices: Advanced Regression Techniques (Kaggle Competition)
* **Thời gian thực hiện:** Học kỳ [Điền Học Kỳ] – Năm học 2025–2026

---

## BẢNG PHÂN CÔNG NHIỆM VỤ VÀ ĐÓNG GÓP CỦA THÀNH VIÊN

| STT | Họ và tên | Mã số sinh viên (MSSV) | Vai trò chính | Nhiệm vụ chi tiết phụ trách | Mức độ đóng góp |
| :---: | :--- | :---: | :--- | :--- | :---: |
| 1 | **Nguyễn Tấn Duy** | **3123580008** | **Trưởng nhóm** | Quản lý dự án, Business Understanding, thiết kế kiến trúc hệ thống, tổng hợp và chuẩn hóa báo cáo CRISP-DM | 100% |
| 2 | [Thành viên 2 - Họ tên] | [MSSV 2] | Thành viên | Data Understanding, thực hiện phân tích dữ liệu khám phá (EDA), trích xuất biểu đồ phân phối và tương quan | 100% |
| 3 | [Thành viên 3 - Họ tên] | [MSSV 3] | Thành viên | Data Preparation, thiết kế pipeline tiền xử lý, kiểm soát rò rỉ dữ liệu (Data Leakage), mã hóa One-Hot và chuẩn hóa | 100% |
| 4 | [Thành viên 4 - Họ tên] | [MSSV 4] | Thành viên | PyTorch Modeling, thiết kế kiến trúc mạng MLPRegressor, xây dựng DataLoader, huấn luyện với Early Stopping và Dropout | 100% |
| 5 | [Thành viên 5 - Họ tên] | [MSSV 5] | Thành viên | Baseline & Evaluation, huấn luyện mô hình đối chuẩn (Linear, Ridge, RF), đánh giá định lượng, sinh file submission Kaggle | 100% |

---

## TÓM TẮT NỘI DUNG (ABSTRACT)

Báo cáo trình bày toàn diện quá trình nghiên cứu, mô hình hóa và thực nghiệm giải quyết bài toán dự đoán giá bán bất động sản tại Ames, Iowa (dựa trên cuộc thi *Kaggle House Prices: Advanced Regression Techniques* và công trình của Dean De Cock, 2011). Nghiên cứu được tổ chức tuần tự theo chuẩn mực công nghiệp **CRISP-DM (Cross-Industry Standard Process for Data Mining)** gồm 6 giai đoạn:
1. **Business Understanding:** Thiết lập bài toán hồi quy có giám sát với hàm mục tiêu tối thiểu hóa Root-Mean-Squared-Error trên log-scale ($\text{log-RMSE}$).
2. **Data Understanding:** Thăm dò 79 đặc trưng trên 1.460 mẫu huấn luyện, phát hiện độ lệch phải nghiêm trọng của `SalePrice` ($Skewness \approx 1.88$), nhận diện cơ chế khuyết thiếu theo ngữ cảnh (Missing Not at Random - MNAR) và cô lập các điểm ngoại lai (outliers).
3. **Data Preparation:** Xây dựng pipeline tiền xử lý tự động hóa, thực hiện biến đổi $\ln(1 + \text{SalePrice})$, xử lý thiếu dữ liệu ngữ cảnh ('None' cho danh mục, 0 cho kích thước), mã hóa One-Hot mở rộng không gian lên 318 chiều, và chuẩn hóa Z-score với quy trình chống rò rỉ thông tin tuyệt đối.
4. **Modeling:** Thiết kế mạng nơ-ron truyền thẳng sâu **Multilayer Perceptron (MLP)** trong PyTorch tích hợp phi tuyến ReLU, Dropout, Weight Decay ($L_2$) và Early Stopping; song song phát triển các mô hình đối chuẩn (Ordinary Least Squares Linear Regression, Ridge Regression, Random Forest).
5. **Evaluation:** Đánh giá định lượng đa chiều. Mô hình Linear Regression đạt $\text{log-RMSE} = 0.1332$ ($R^2 = 0.9049$), Ridge Regression đạt $\text{log-RMSE} = 0.1363$ ($R^2 = 0.9004$), trong khi PyTorch MLP đạt $\text{log-RMSE} = 0.1447$ ($R^2 = 0.8878$) và Random Forest đạt $0.1517$. Báo cáo cung cấp luận chứng khoa học sâu sắc lý giải nguyên nhân các mô hình tuyến tính chính quy hóa vượt trội hơn mạng nơ-ron trên tập dữ liệu dạng bảng quy mô nhỏ.
6. **Deployment:** Đóng gói toàn bộ mã nguồn thành kiến trúc module Python chuyên nghiệp (`src/`), lưu trữ checkpoint mô hình tối ưu, xuất tệp dự báo `submission_mlp.csv` chuẩn định dạng Kaggle và cung cấp hướng dẫn tái lập thực nghiệm.

---

# GIAI ĐOẠN 1: BUSINESS UNDERSTANDING (HIỂU BÀI TOÁN NGHIỆP VỤ)

## 1.1. Bối cảnh nghiên cứu và Nguồn gốc dữ liệu
Thị trường bất động sản là một trong những lĩnh vực kinh tế có mức độ phức tạp cao, trong đó giá bán một căn nhà phụ thuộc đồng thời vào rất nhiều yếu tố: vị trí địa lý, diện tích sử dụng, chất lượng xây dựng tổng thể, năm hoàn thành, tiện ích đi kèm (tầng hầm, nhà để xe, hồ bơi), và các điều kiện kinh tế vi mô. Trong nhiều thập kỷ, bộ dữ liệu *Boston Housing Data* (Harrison & Rubinfeld, 1978) đã được sử dụng rộng rãi như bộ dữ liệu quy chuẩn để nghiên cứu các thuật toán hồi quy. Tuy nhiên, như giáo sư Dean De Cock (2011) đã phân tích trong bài báo *"Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project"*, bộ dữ liệu Boston đã trở nên lạc hậu, kích thước mẫu nhỏ (506 quan sát), biến số thô sơ và chứa một số yếu tố nhân khẩu học nhạy cảm.

Bộ dữ liệu bất động sản Ames Housing được Dean De Cock thu thập trực tiếp từ Văn phòng Thẩm định Giá Thành phố Ames (Ames City Assessor's Office) trong giai đoạn 2006 – 2010. Dữ liệu ghi nhận thông tin thực tế của 2.930 giao dịch nhà ở dân cư cá nhân với 79 biến mô tả chi tiết, phản ánh toàn diện các thuộc tính vật lý và vị trí của một căn nhà. Cuộc thi **Kaggle House Prices: Advanced Regression Techniques** chia tập dữ liệu này thành hai phần:
* Tập huấn luyện công khai (`train.csv`): gồm 1.460 dòng quan sát kèm giá bán thực tế (`SalePrice`).
* Tập kiểm thử độc lập (`test.csv`): gồm 1.459 dòng quan sát bị ẩn giá bán nhằm đánh giá khả năng tổng quát hóa trên bảng xếp hạng (Leaderboard).

## 1.2. Mục tiêu nghiên cứu và Định nghĩa bài toán
* **Mục tiêu kỹ thuật (Machine Learning Goal):** Xây dựng mô hình học có giám sát (Supervised Learning - Regression) có khả năng ánh xạ véc-tơ đặc trưng thuộc tính ngôi nhà $\mathbf{x} \in \mathbb{R}^D$ thành giá trị dự đoán $\hat{y} = f(\mathbf{x}; \boldsymbol{\theta})$ xấp xỉ giá bán thực tế $y = \text{SalePrice}$.
* **Mục tiêu thực hành Deep Learning:** Tự thiết kế, lập trình và tinh chỉnh một mạng nơ-ron Multilayer Perceptron (MLP) thuần PyTorch, hiểu rõ cấu trúc tensor, cơ chế lan truyền ngược (backpropagation), hàm kích hoạt, tối ưu hóa và chống hiện tượng quá khớp (overfitting), thay vì chỉ dựa vào các thư viện mô hình đóng gói cấp cao.
* **Mục tiêu quy chuẩn MLOps:** Tổ chức dự án khoa học, khả năng tái lập thực nghiệm (reproducibility), cấu trúc module hóa, không rò rỉ dữ liệu và tuân thủ chặt chẽ tiến trình CRISP-DM.

## 1.3. Tiêu chí đánh giá chất lượng mô hình (Evaluation Metric)
Theo chuẩn mực của cuộc thi Kaggle, thước đo chính thức để xếp hạng mô hình là **Root-Mean-Squared-Error (RMSE) tính trên logarit của giá bán**:

$$\text{log-RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( \ln(\hat{y}_i) - \ln(y_i) \right)^2}$$

Trong đó:
* $N$: Số lượng quan sát trong tập đánh giá.
* $y_i$: Giá bán thực tế của ngôi nhà thứ $i$ (`SalePrice`).
* $\hat{y}_i$: Giá bán dự đoán của mô hình đối với ngôi nhà thứ $i$.

**Ý nghĩa khoa học của log-RMSE trong định giá tài sản:**
Trong bài toán bất động sản, việc sử dụng sai số tuyệt đối hoặc sai số bình phương trên giá gốc ($\text{RMSE} = \sqrt{\frac{1}{N}\sum (\hat{y}_i - y_i)^2}$) sẽ gây ra độ lệch lớn khi dự đoán các căn nhà giá trị cao (ví dụ: căn nhà 700.000 USD sai lệch 50.000 USD sẽ bị phạt nặng gấp nhiều lần so với căn nhà 100.000 USD sai lệch 20.000 USD, dù xét về tỷ lệ phần trăm thì sai số căn thứ hai nghiêm trọng hơn). Việc lấy logarit giúp chuyển đổi sai số tuyệt đối thành **sai số tỷ lệ phần trăm tương đối (relative percentage error)**:

$$\ln(\hat{y}_i) - \ln(y_i) = \ln\left(\frac{\hat{y}_i}{y_i}\right) \approx \frac{\hat{y}_i - y_i}{y_i}$$

Do đó, một sai lệch 10% trên ngôi nhà giá rẻ và ngôi nhà đắt tiền sẽ bị phạt tương đương nhau, đảm bảo tính công bằng và ổn định về mặt kinh tế lượng.

---

# GIAI ĐOẠN 2: DATA UNDERSTANDING (KHÁM PHÁ & HIỂU DỮ LIỆU - EDA)

## 2.1. Phân loại cấu trúc 79 biến đặc trưng
Bộ dữ liệu gồm 79 biến đầu vào mô tả 1.460 quan sát trong tập `train.csv`. Các đặc trưng được chia thành 4 nhóm hình thức dữ liệu chính:
1. **Biến số thực liên tục (Continuous Numerical - 20 biến):** Thể hiện kích thước diện tích (`LotFrontage`, `LotArea`, `MasVnrArea`, `BsmtFinSF1`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `GrLivArea`, `GarageArea`, `WoodDeckSF`, `OpenPorchSF`, v.v.).
2. **Biến số nguyên rời rạc (Discrete Numerical - 16 biến):** Thể hiện số lượng phòng và mốc thời gian (`YearBuilt`, `YearRemodAdd`, `BsmtFullBath`, `FullBath`, `HalfBath`, `BedroomAbvGr`, `TotRmsAbvGrd`, `Fireplaces`, `GarageYrBlt`, `GarageCars`, `MoSold`, `YrSold`, v.v.).
3. **Biến định danh/Phân loại không thứ tự (Nominal Categorical - 23 biến):** Loại hình bất động sản, vị trí và vật liệu (`MSSubClass`, `MSZoning`, `Neighborhood`, `Condition1`, `BldgType`, `HouseStyle`, `RoofStyle`, `Foundation`, `Exterior1st`, `SaleType`, `SaleCondition`, v.v.).
4. **Biến thứ bậc/Phân loại có thứ tự (Ordinal Categorical - 20 biến):** Đánh giá chất lượng và tiện nghi (`OverallQual`, `OverallCond`, `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, v.v.).

## 2.2. Phân tích phân phối của biến mục tiêu `SalePrice`
* Phân tích thống kê mô tả cho biến `SalePrice` trên 1.460 căn nhà tập train:
  * Giá trị nhỏ nhất (Min): 34.900 USD.
  * Phân vị 25% (Q1): 129.975 USD.
  * Trung vị (Median - Q2): 163.000 USD.
  * Trung bình (Mean): 180.921 USD.
  * Phân vị 75% (Q3): 214.000 USD.
  * Giá trị lớn nhất (Max): 755.000 USD.
  * Độ lệch (Skewness): **+1.8828** (Lệch phải mạnh - Positively Skewed).
  * Độ nhọn (Kurtosis): **6.5362** (Đuôi nặng - Leptokurtic).

Do giá bán có xu hướng phân phối lệch phải với đuôi kéo dài về phía giá cao, nếu huấn luyện trực tiếp trên thang đo gốc, các thuật toán hồi quy và mạng nơ-ron sẽ bị chi phối bởi các mẫu giá cao hiếm hoi, dẫn đến gradient bùng nổ và hiệu năng dự đoán kém ở vùng giá trung bình và thấp. 
* **Giải pháp biến đổi hàm log:** Áp dụng phép biến đổi:

$$y_{\text{trans}} = \ln(1 + \text{SalePrice})$$

Sau phép biến đổi, hệ số Skewness giảm xuống còn **+0.1213**, phân phối trở về dạng phân phối chuẩn (Gaussian-like distribution), hoàn toàn thỏa mãn các giả định hồi quy tuyến tính và giúp quá trình lan truyền gradient trong mạng MLP ổn định.

*(Tham khảo biểu đồ: `outputs/figures/eda_target_distribution.png`)*

## 2.3. Phân tích cơ chế dữ liệu khuyết thiếu (Missing Data Mechanisms)
Kiểm toán dữ liệu khuyết thiếu (`outputs/metrics/missing_audit.csv`) chỉ ra rằng có 19 cột chứa giá trị thiếu trên tập huấn luyện:
* Các biến khuyết thiếu nghiêm trọng nhất: `PoolQC` (99.52%), `MiscFeature` (96.30%), `Alley` (93.77%), `Fence` (80.75%), `FireplaceQu` (47.26%).
* Các nhóm đặc trưng liên quan đến Tầng hầm (Basement) và Nhà để xe (Garage) khuyết thiếu đồng thời từ 2.5% đến 5.5% (`GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`).
* Biến số liên tục khuyết thiếu: `LotFrontage` (17.74%), `MasVnrArea` (0.55%), `GarageYrBlt` (5.55%).

**Nhận định ngữ cảnh (Domain Insight từ Dean De Cock):**
Phần lớn các trường hợp khuyết thiếu trong bộ dữ liệu này **không phải là dữ liệu thiếu ngẫu nhiên (Missing Completely at Random - MCAR)** mà là **dữ liệu thiếu có chủ đích mang ý nghĩa ngữ cảnh (Missing Not at Random - MNAR)**. Cụ thể: giá trị `NA` trong `PoolQC` có nghĩa là "Ngôi nhà không có hồ bơi"; `NA` trong `Alley` có nghĩa là "Không có lối đi phụ"; `NA` trong `GarageType` có nghĩa là "Nhà không có garage". Nếu áp dụng phương pháp thông thường là xóa bỏ các hàng/cột này hoặc thay bằng giá trị xuất hiện nhiều nhất (mode), mô hình sẽ mất đi thông tin cực kỳ quý giá về sự thiếu vắng của tiện nghi. Do đó, bắt buộc phải có quy tắc xử lý ngữ cảnh chuyên biệt.

*(Tham khảo biểu đồ: `outputs/figures/eda_missing_rates.png`)*

## 2.4. Phân tích tương quan và Mối quan hệ với Biến mục tiêu
Phân tích ma trận tương quan Pearson giữa các biến định lượng với `log(SalePrice)` (`outputs/metrics/target_correlations.csv`) xác định những đặc trưng có ảnh hưởng mạnh nhất:
1. `OverallQual` ($r = 0.8179$): Chất lượng vật liệu và độ hoàn thiện tổng thể của ngôi nhà là nhân tố chi phối mạnh nhất.
2. `GrLivArea` ($r = 0.7009$): Diện tích sinh hoạt trên mặt đất.
3. `GarageCars` ($r = 0.6806$) và `GarageArea` ($r = 0.6509$): Sức chứa và diện tích garage.
4. `TotalBsmtSF` ($r = 0.6121$) và `1stFlrSF` ($r = 0.5970$): Diện tích tầng hầm và tầng 1.
5. `FullBath` ($r = 0.5945$) và `YearBuilt` ($r = 0.5866$): Số phòng tắm đầy đủ và năm xây dựng.

*(Tham khảo biểu đồ: `outputs/figures/eda_target_correlations.png`)*

## 2.5. Nhận diện các điểm dị biệt (Outliers)
Trong bài báo gốc của Dean De Cock (2011), tác giả đã chỉ rõ sự xuất hiện của 5 điểm dị biệt lớn khi đối chiếu `GrLivArea` với `SalePrice`. Trong đó, có hai trường hợp bất thường là những căn nhà có diện tích sinh hoạt cực lớn ($> 4.000\text{ sq ft}$) nhưng giá bán lại rất thấp ($< 300.000\text{ USD}$), nguyên nhân là do giao dịch nội bộ hoặc đất nông nghiệp chưa hoàn thiện. Việc nhận diện chính xác các điểm này cho phép nhà phân tích kiểm soát độ nhạy cảm của các hàm mục tiêu sai số bình phương.

*(Tham khảo biểu đồ: `outputs/figures/eda_grlivarea_outliers.png`)*

---

# GIAI ĐOẠN 3: DATA PREPARATION (CHUẨN BỊ & TIỀN XỬ LÝ DỮ LIỆU)

Để đảm bảo mô hình học tập hiệu quả và có khả năng áp dụng thực tế trên dữ liệu kiểm thử, quy trình tiền xử lý được thiết kế khép kín theo dạng pipeline với nguyên tắc **Zero Data Leakage** (mọi tham số thống kê chỉ được học trên tập Train, sau đó áp dụng biến đổi cho tập Validation và Test).

## 3.1. Chiến lược Điền giá trị khuyết thiếu theo Ngữ cảnh (Domain-Specific Imputation)
* **Nhóm phân loại tiện nghi (Categorical Amenities):** Các cột `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` được điền giá trị khuyết thiếu bằng nhãn chuỗi `'None'` (đại diện cho trạng thái không sở hữu tiện nghi).
* **Nhóm kích thước tiện nghi liên quan (Numeric Amenities):** Khi một căn nhà không có garage hoặc tầng hầm, các biến số đo lường kích thước tương ứng như `GarageArea`, `GarageCars`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath`, `MasVnrArea` được điền giá trị $0$.
* **Biến khoảng cách tiếp giáp đường (`LotFrontage`):** Điền bằng giá trị trung vị (`median`) của `LotFrontage` theo từng khu vực lân cận (`Neighborhood`), phản ánh thực tế rằng các lô đất trong cùng một khu quy hoạch thường có mặt tiền tương đương nhau.
* **Các biến số còn lại:** Sử dụng `SimpleImputer(strategy='median')` để hạn chế ảnh hưởng của giá trị ngoại lai.
* **Các biến phân loại còn lại:** Sử dụng `SimpleImputer(strategy='most_frequent')` để thay thế bằng giá trị mode phổ biến nhất.

## 3.2. Mã hóa biến phân loại (Categorical Encoding)
* Áp dụng kỹ thuật **One-Hot Encoding (OHE)** (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`).
* Tùy chọn `handle_unknown='ignore'` đóng vai trò sống còn trong môi trường sản xuất: khi tập Test xuất hiện những giá trị hạng mục mới chưa từng có trong tập Train, encoder sẽ tự động gán toàn bộ các cột one-hot tương ứng bằng $0$ mà không làm phát sinh lỗi kích thước ma trận (shape mismatch).
* Sau khi thực hiện One-Hot Encoding trên toàn bộ các biến định tính, không gian đầu vào mở rộng từ 79 biến ban đầu lên **318 đặc trưng số học**.

## 3.3. Chuẩn hóa thang đo đặc trưng (Feature Scaling)
Mạng nơ-ron học sâu sử dụng thuật toán tối ưu hóa dựa trên gradient (Gradient Descent) và các hàm kích hoạt phi tuyến. Nếu các đặc trưng đầu vào có thang đo chênh lệch quá lớn (ví dụ `GrLivArea` có giá trị hàng ngàn trong khi `FullBath` chỉ từ 1 đến 3), bề mặt hàm lỗi sẽ bị kéo dài thành dạng elip hẹp (ill-conditioned surface), dẫn đến hiện tượng dao động gradient và tốc độ hội tụ cực kỳ chậm.
* Toàn bộ 318 đặc trưng sau khi xử lý được chuẩn hóa theo phương pháp **Z-score Standardization**:

$$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

Trong đó $\mu_{\text{train}}$ và $\sigma_{\text{train}}$ là giá trị trung bình và độ lệch chuẩn được tính toán duy nhất trên tập huấn luyện.

## 3.4. Chiến lược phân chia dữ liệu và Cơ chế đóng gói (Artifact Serialization)
* Dữ liệu tập huấn luyện (1.460 dòng) được phân tách thành hai tập độc lập:
  * **Train set:** 80% (1.168 mẫu) dùng để huấn luyện mô hình.
  * **Validation set:** 20% (292 mẫu) dùng để đánh giá độc lập, tinh chỉnh siêu tham số và kích hoạt cơ chế Early Stopping.
  * Cố định ngẫu nhiên: `random_state = 42` để đảm bảo khả năng tái lập 100% kết quả thực nghiệm.
* Toàn bộ pipeline tiền xử lý sau khi fit được đóng gói thành tệp nhị phân (`outputs/preprocessing/preprocessing_pipeline.pkl`), bảo đảm tính nhất quán khi suy diễn trên tập dữ liệu kiểm thử `test.csv`.

---

# GIAI ĐOẠN 4: MODELING (XÂY DỰNG MÔ HÌNH HỌC SÂU & ĐỐI CHUẨN)

## 4.1. Cơ sở lý thuyết Mạng Nơ-ron Multilayer Perceptron (MLP)

Mạng nơ-ron truyền thẳng đa tầng (Multilayer Perceptron - MLP) là một trong những kiến trúc nền tảng của học sâu (Deep Learning), hoạt động như một bộ xấp xỉ hàm vạn năng (Universal Function Approximator - định lý Hornik, 1989).

```
Input (318 features) 
       │
       ▼
[Linear Layer 1: 318 -> 128] ──> [ReLU] ──> [Dropout (p=0.2)]
       │
       ▼
[Linear Layer 2: 128 -> 64]  ──> [ReLU] ──> [Dropout (p=0.2)]
       │
       ▼
[Linear Layer 3: 64 -> 1]    ──> Output: ln(SalePrice)
```

### A. Quá trình Lan truyền tiến (Forward Propagation)
Giả sử mạng có $L$ tầng ẩn. Đối với mỗi tầng $l \in \{1, 2, \dots, L\}$:
1. **Phép biến đổi affine tuyến tính:**

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

Trong đó:
* $\mathbf{a}^{[l-1]}$: Véc-tơ kích hoạt đầu ra của tầng trước đó (với $\mathbf{a}^{[0]} = \mathbf{x} \in \mathbb{R}^{318}$ là véc-tơ đặc trưng đầu vào).
* $\mathbf{W}^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$: Ma trận trọng số kết nối giữa tầng $l-1$ và tầng $l$.
* $\mathbf{b}^{[l]} \in \mathbb{R}^{n_l}$: Véc-tơ độ lệch (bias).

2. **Hàm kích hoạt phi tuyến (Nonlinear Activation Function):**

$$\mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$$

Nghiên cứu sử dụng hàm **Rectified Linear Unit (ReLU)**:

$$g(z) = \max(0, z)$$

**Ưu điểm của ReLU so với Sigmoid/Tanh:** Hàm Sigmoid ($\sigma(z) = \frac{1}{1 + e^{-z}}$) và Tanh có đạo hàm bão hòa khi $|z|$ lớn, dẫn đến hiện tượng triệt tiêu gradient (Vanishing Gradient Problem) khi truyền ngược qua nhiều tầng. Ngược lại, đạo hàm của ReLU bằng hằng số $1$ với mọi $z > 0$, giúp dòng gradient truyền qua các tầng mạng mà không bị suy giảm, đồng thời tạo ra tính thưa (sparsity) cho các nơ-ron kích hoạt, nâng cao năng lực biểu diễn phi tuyến và tốc độ tính toán.

3. **Tầng đầu ra (Output Layer):**
Do đây là bài toán hồi quy (Regression), tầng đầu ra $L$ gồm đúng 1 nơ-ron tuyến tính không qua hàm kích hoạt phi tuyến ($g^{[L]}(z) = z$):

$$\hat{y}_{\text{log}} = z^{[L]} = \mathbf{w}^{[L]\top} \mathbf{a}^{[L-1]} + b^{[L]}$$

### B. Hàm mất mát (Loss Function)
Mô hình được huấn luyện bằng hàm mất mát sai số toàn phương trung bình (Mean Squared Error - MSE) trên không gian logarit:

$$\mathcal{L}_{\text{MSE}}(\boldsymbol{\theta}) = \frac{1}{B} \sum_{i=1}^{B} \left( \hat{y}_{\text{log}, i} - y_{\text{log}, i} \right)^2$$

với $B$ là kích thước mini-batch và $y_{\text{log}, i} = \ln(1 + \text{SalePrice}_i)$.

### C. Thuật toán tối ưu hóa Adam (Adaptive Moment Estimation)
Để cập nhật tập trọng số $\boldsymbol{\theta} = \{\mathbf{W}^{[l]}, \mathbf{b}^{[l]}\}$, nghiên cứu áp dụng thuật toán tối ưu hóa **Adam** (Kingma & Ba, 2014) kết hợp sức mạnh của phương pháp quán tính (Momentum) và chia tỷ lệ gradient theo từng tọa độ (RMSProp):
1. Tính gradient của hàm mất mát tại bước $t$: $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}_t(\boldsymbol{\theta})$.
2. Cập nhật trung bình trượt bậc 1 (ước lượng kỳ vọng gradient - Momentum):

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$

3. Cập nhật trung bình trượt bậc 2 (ước lượng phương sai không căn chỉnh của gradient - RMSProp):

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$

4. Hiệu chỉnh độ lệch khởi tạo (Bias correction):

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

5. Cập nhật tham số:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

Trong đó: tốc độ học $\eta = 10^{-3}$, các hệ số phân rã $\beta_1 = 0.9$, $\beta_2 = 0.999$, và hằng số làm mịn $\epsilon = 10^{-8}$.

### D. Các kỹ thuật kiểm soát Quá khớp (Regularization Techniques)
Do tập dữ liệu có quy mô khiêm tốn trong khi số lượng đặc trưng lớn ($D = 318$), nguy cơ quá khớp (overfitting) của mạng nơ-ron là rất cao. Ba kỹ thuật điều hòa trọng số được tích hợp chặt chẽ:
1. **Dropout (Srivastava et al., 2014):** Đặt xác suất $p = 0.2$ ngắt ngẫu nhiên các nơ-ron ở mỗi lượt truyền tiến trong quá trình train. Kỹ thuật này ngăn chặn các nơ-ron đồng thích nghi (co-adaptation), buộc mạng phải học các biểu diễn phân tán và bền vững hơn.
2. **Weight Decay ($L_2$ Regularization):** Bổ sung số hạng phạt $\frac{\lambda}{2} \|\mathbf{W}\|_2^2$ vào hàm mục tiêu với $\lambda = 10^{-4}$ để hạn chế các trọng số tăng trưởng quá lớn.
3. **Early Stopping:** Theo dõi liên tục hàm mất mát trên tập kiểm định $\mathcal{L}_{\text{val}}$. Nếu sau 20 epochs liên tiếp (patience = 20) mà $\mathcal{L}_{\text{val}}$ không cải thiện, quá trình huấn luyện sẽ dừng lại ngay lập tức và khôi phục lại trạng thái trọng số tốt nhất (`mlp_model_best.pth`).

## 4.2. Xây dựng Mô hình Đối chuẩn (Baselines)
Nhằm tạo cơ sở khoa học để đánh giá thực chất năng lực của mạng nơ-ron, ba mô hình học máy truyền thống được thiết lập làm đường cơ sở (Baseline):
1. **Ordinary Least Squares (OLS) Linear Regression:** Mô hình hồi quy tuyến tính cổ điển không có thành phần phạt, tìm nghiệm giải tích tối ưu: $\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$.
2. **Ridge Regression ($L_2$ Regularized Linear Model):** Bổ sung chuẩn $L_2$ vào hàm mất mát tuyến tính với siêu tham số $\alpha = 1.0$:

$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$

Nghiệm giải tích của Ridge: $\mathbf{w}_{\text{ridge}} = (\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$ giúp giải quyết triệt để hiện tượng đa cộng tuyến (multicollinearity) khi số chiều cao.
3. **Random Forest Regressor:** Mô hình học kết hợp dạng đóng bao (bagging ensemble) gồm 100 cây quyết định (Decision Trees) có khả năng chụp các mối tương tác phi tuyến phức tạp mà không đòi hỏi dữ liệu phải có phân phối chuẩn.

---

# GIAI ĐOẠN 5: EVALUATION (ĐÁNH GIÁ & SO SÁNH CHUYÊN SÂU)

## 5.1. Bảng Tổng hợp Kết quả Đánh giá Thực nghiệm
Toàn bộ các mô hình được huấn luyện và đánh giá trên cùng một tập dữ liệu phân chia thống nhất (Train 80% / Validation 20%, seed 42). Các chỉ số đánh giá gồm: $\text{log-RMSE}$ (tiêu chí chính), $\text{log-MAE}$ (Mean Absolute Error) và Hệ số xác định $R^2$.

| STT | Mô hình (Model) | Validation log-RMSE | Validation log-MAE | Validation $R^2$ | Đánh giá tổng quan |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **Linear Regression (OLS)** | **0.1332** | **0.0955** | **0.9049** | Hiệu năng giải thích phương sai cao nhất ($90.49\%$) |
| 2 | **Ridge Regression ($\alpha=1.0$)** | **0.1363** | **0.0973** | **0.9004** | Cực kỳ ổn định, kiểm soát đa cộng tuyến xuất sắc |
| 3 | **PyTorch MLP (Deep Learning)** | **0.1447** | **0.1005** | **0.8878** | Dự đoán tốt, hội tụ mượt mà nhưng chịu tác động tham số |
| 4 | **Random Forest Baseline** | **0.1517** | **0.0992** | **0.8767** | Kém nhất trong nhóm do hạn chế phân vùng trực giao |

*(Số liệu trích xuất chính thức từ tệp `outputs/metrics/Comparison.csv`)*

*(Tham khảo biểu đồ so sánh phân tán dự đoán thực tế: `outputs/figures/actual_vs_predicted_mlp.png`)*

## 5.2. Luận cứ Khoa học Chuyên sâu: Vì sao Linear/Ridge lại vượt trội hơn Mạng Nơ-ron MLP trên tập dữ liệu này?

Một kết quả thực nghiệm nổi bật và mang tính học thuật cao trong Lab 02 là: **Mô hình hồi quy tuyến tính cổ điển (Linear Regression) và Ridge Regression lại cho kết quả vượt trội hơn so với mạng nơ-ron học sâu Multilayer Perceptron (MLP)** (log-RMSE $0.1332$ và $0.1363$ so với $0.1447$). Hiện tượng này hoàn toàn nhất quán với các công trình nghiên cứu hiện đại về học sâu cho dữ liệu dạng bảng (Tabular Data) như Kadra et al. (NeurIPS 2021) hay Grinsztajn et al. (NeurIPS 2022). Nguyên nhân khoa học được lý giải qua 4 luận điểm then chốt:

### Luận điểm 1: Lời nguyền số chiều và Sự khan hiếm dữ liệu (Curse of Dimensionality & Small Sample Size)
* Tập dữ liệu huấn luyện sau khi chia chỉ có **1.168 mẫu**, nhưng không gian đặc trưng sau khi One-Hot Encoding đã phình to lên đến **318 chiều**.
* Số lượng tham số tự do (parameters) của mô hình:
  * **Ridge / Linear Regression:** Chỉ có đúng $318 \text{ weights} + 1 \text{ bias} = \mathbf{319}$ tham số cần tối ưu. Số bậc tự do hoàn toàn tương thích với quy mô 1.168 mẫu ($N > D$).
  * **PyTorch MLP (Kiến trúc 318 - 128 - 64 - 1):**
    * Tầng 1: $318 \times 128 + 128 = 40.832$ tham số.
    * Tầng 2: $128 \times 64 + 64 = 8.256$ tham số.
    * Tầng 3: $64 \times 1 + 1 = 65$ tham số.
    * Tổng cộng: $\mathbf{49.153}$ tham số tự do!
* Tỷ lệ dữ liệu trên tham số của MLP là $\frac{1.168}{49.153} \approx \mathbf{0.0238}$ (trung bình chỉ có khoảng $0.02$ điểm dữ liệu để ước lượng một tham số nơ-ron!). Với năng lực biểu diễn quá lớn (high capacity) trên một tập dữ liệu quá nhỏ, dù có áp dụng Dropout và Early Stopping, mạng nơ-ron vẫn có xu hướng ghi nhớ nhiễu cục bộ của tập huấn luyện hơn là học được quy luật tổng quát.

### Luận điểm 2: Thiên kiến quy nạp (Inductive Bias) và Bản chất cộng dồn của Giá bất động sản
* Trong kinh tế học lượng giác và định giá tài sản, giá trị của một ngôi nhà có bản chất **cộng dồn tuyến tính (Additive Nature)** rất mạnh:
  $$\text{Giá nhà} \approx (\text{Đơn giá đất} \times \text{Diện tích}) + (\text{Chất lượng} \times \text{Hệ số vật liệu}) + \text{Giá trị tiện ích thêm} + \dots$$
* Mô hình tuyến tính sở hữu **thiên kiến quy nạp (Inductive Bias)** hoàn hảo cho giả thiết này: nó giả định mối quan hệ giữa biến đầu vào và đầu ra là siêu phẳng tuyến tính.
* Ngược lại, mạng MLP không có bất kỳ giả thiết tiên nghiệm nào về cấu trúc cộng dồn này. MLP phải tự mình xấp xỉ lại các phép cộng đơn giản thông qua sự kết hợp phức tạp của hàng chục nghìn nơ-ron và hàm kích hoạt phi tuyến ReLU. Điều này gây lãng phí năng lực tính toán và dễ tạo ra các sai số xấp xỉ không đáng có.

### Luận điểm 3: Bản chất toán học của Không gian Tối ưu hóa (Optimization Landscape)
* **Đối với Ridge Regression:** Hàm mục tiêu là một hàm lồi ngặt (Strictly Convex Optimization Problem). Bài toán luôn tồn tại một nghiệm cực tiểu toàn cục duy nhất dạng đóng (Closed-form Analytical Solution) thông qua phép nghịch đảo ma trận $(\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1}$. Quá trình tìm nghiệm không phụ thuộc vào giá trị khởi tạo và không bị kẹt ở các điểm cực tiểu cục bộ.
* **Đối với Mạng Nơ-ron MLP:** Bề mặt hàm mất mát là phi lồi (Non-convex), chứa vô số điểm yên ngựa (saddle points) và cực tiểu địa phương (local minima). Các thuật toán tối ưu hóa ngẫu nhiên (như Adam/SGD) cùng với kích thước batch nhỏ ($B=32$) tạo ra các bước nhảy gradient ngẫu nhiên lớn, khiến mạng khó có thể hội tụ chính xác đến điểm tối ưu tuyệt đối trên không gian mẫu hạn hẹp.

### Luận điểm 4: Đặc thù của Dữ liệu dạng bảng (Tabular Data Heterogeneity) so với Cây quyết định
* Dữ liệu dạng bảng không có cấu trúc không gian (như ảnh - CNN) hay cấu trúc chuỗi thời gian (như văn bản - RNN/Transformer). Các cột dữ liệu hoàn toàn không đồng nhất: có cột liên tục thang đo lớn, có cột nhị phân thưa ($0/1$ sau OHE).
* Mô hình **Random Forest** tuy mạnh nhưng trong bài toán này lại đạt kết quả kém nhất ($\text{log-RMSE} = 0.1517$). Lý do là các cây quyết định phân chia không gian bằng các siêu phẳng trực giao với các trục tọa độ (axis-aligned splits), tạo ra các hàm dự đoán dạng bậc thang (step functions). Cấu trúc bậc thang này rất khó xấp xỉ mượt mà một hàm hồi quy tuyến tính liên tục như giá nhà, đồng thời cây quyết định bị suy giảm hiệu năng nghiêm trọng khi đối mặt với không gian thưa 318 chiều.

---

# GIAI ĐOẠN 6: DEPLOYMENT (TRIỂN KHAI & TÁI LẬP KẾT QUẢ)

## 6.1. Kiến trúc Đóng gói Mã nguồn Chuyên nghiệp (`src/`)
Toàn bộ dự án đã được tái cấu trúc từ các notebook nghiên cứu ban đầu thành một gói thư viện Python hoàn chỉnh, chuẩn mực theo kiến trúc MLOps:
* `src/config.py`: Tự động định vị đường dẫn thư mục gốc `LAB02_ROOT` bằng `pathlib.Path`, quản lý tập trung toàn bộ siêu tham số thực nghiệm (learning rate, batch size, epochs, random seed) mà không bị phụ thuộc vào hệ điều hành.
* `src/data/dataset.py`: Định nghĩa lớp `HousePriceDataset` chuẩn PyTorch kế thừa từ `torch.utils.data.Dataset`, hỗ trợ nhận dữ liệu từ DataFrame hoặc Tensor, chuyển đổi dữ liệu thành các mini-batch phục vụ `DataLoader`.
* `src/models/mlp.py`: Khai báo lớp mô hình `MLPRegressor` kế thừa `nn.Module`, cho phép cấu hình linh hoạt số tầng ẩn, số nơ-ron và xác suất dropout.
* `src/utils/metrics.py`: Module tính toán độc lập các chỉ số đánh giá (RMSE, MAE, $R^2$) trên cả hai thang đo (log-scale và thang đo giá gốc đô la USD).

## 6.2. Quy trình Dự báo và Chuẩn bị Nộp bài Kaggle
* Toàn bộ pipeline tiền xử lý và trọng số mô hình tốt nhất (`models/mlp_model_best.pth`) được nạp để thực hiện suy diễn (Inference) trên tập kiểm thử ẩn `data/raw/test.csv` (1.459 dòng).
* Do mô hình được huấn luyện trên biến mục tiêu biến đổi $\ln(1 + \text{SalePrice})$, đầu ra dự đoán của mạng $\hat{y}_{\text{log}}$ được khôi phục về thang đo giá trị tiền tệ thực tế thông qua hàm số mũ:

$$\hat{y}_{\text{dollar}} = \exp(\hat{y}_{\text{log}}) - 1$$

* Tệp kết quả được xuất ra tại `outputs/submissions/submission_mlp.csv` với đúng 1.459 dòng dự đoán và 2 cột chuẩn xác theo yêu cầu cuộc thi: `Id` và `SalePrice`.
* Kiểm tra tính hợp lệ: tệp không chứa giá trị `NaN`, không có giá trị âm hoặc vô cùng, định dạng hoàn toàn khớp với `sample_submission.csv` và sẵn sàng ghi nhận điểm xếp hạng công khai (Public Leaderboard Score).

## 6.3. Khả năng Tái lập (Reproducibility) và Hướng phát triển trong tương lai
* **Khả năng tái lập:** Toàn bộ quá trình từ nạp dữ liệu thô, tiền xử lý, huấn luyện baseline đến huấn luyện PyTorch MLP được cố định bằng một seed duy nhất (`SEED = 42`). Bất kỳ nhà nghiên cứu nào khi tải kho mã nguồn đều có thể chạy lại từ đầu và tái lập 100% các con số sai số trong báo cáo này.
* **Định hướng phát triển tiếp theo:**
  1. *Feature Engineering chuyên sâu:* Tích hợp các biến kết hợp mang tri thức nghiệp vụ sâu như: Tổng diện tích sinh hoạt thực tế (`TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`), Tuổi thọ căn nhà lúc bán (`Age = YrSold - YearBuilt`), Tổng số phòng tắm (`TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`).
  2. *Kiến trúc Deep Learning tiên tiến cho dữ liệu Tabular:* Thử nghiệm mô hình **TabNet** (Arik & Pfister, Google Cloud AI) sử dụng cơ chế Sequential Attention để chọn lọc đặc trưng động hoặc **FT-Transformer** (Gorishniy et al., 2021).
  3. *Mô hình Ensembling:* Kết hợp sức mạnh của mô hình tuyến tính chính quy hóa (Ridge/Lasso) cùng với Gradient Boosting (XGBoost, LightGBM, CatBoost) và PyTorch MLP thông qua kỹ thuật Stacking hoặc Blending để đạt thứ hạng tối ưu trên bảng xếp hạng Kaggle toàn cầu.

---

# TÀI LIỆU THAM KHẢO (REFERENCES)

1. **De Cock, D. (2011).** *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*. Journal of Statistics Education, 19(3), 1–14.
2. **Kaggle Inc.** *House Prices: Advanced Regression Techniques*. [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
3. **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).** *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems (NeurIPS 2019), 32.
4. **Kingma, D. P., & Ba, J. (2014).** *Adam: A Method for Stochastic Optimization*. arXiv preprint arXiv:1412.6980.
5. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. The Journal of Machine Learning Research, 15(1), 1929–1958.
6. **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).** *Why do tree-based models still outperform deep learning on typical tabular data?*. Advances in Neural Information Processing Systems (NeurIPS 2022), 35, 507–520.
7. **Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000).** *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS inc, 9–13.
