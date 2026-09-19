# Ames Housing Paper Summary

## 1. Thông tin tài liệu

| Thuộc tính | Nội dung |
| --- | --- |
| Tên bài báo | *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project* |
| Tác giả | Dean De Cock |
| Tạp chí | *Journal of Statistics Education* |
| Năm xuất bản | 2011 |
| Volume | 19, Number 3 |

Bài báo giới thiệu bộ dữ liệu Ames Housing, được xây dựng từ dữ liệu giao dịch bất động sản tại thành phố Ames, bang Iowa, Hoa Kỳ. Mục tiêu chính của tác giả là cung cấp một bộ dữ liệu thực tế, hiện đại hơn và giàu thông tin hơn Boston Housing Dataset để phục vụ giảng dạy và thực hành các phương pháp hồi quy.

## 2. Bối cảnh và động lực xây dựng dữ liệu

Dean De Cock từng sử dụng Boston Housing Dataset trong các môn học hồi quy. Tuy nhiên, bộ dữ liệu này có số lượng quan sát và biến tương đối nhỏ; dữ liệu có nguồn gốc từ những năm 1970 nên giá nhà không còn phản ánh thị trường hiện đại.

Trong quá trình tìm kiếm dữ liệu thay thế, tác giả tiếp cận dữ liệu từ Ames City Assessor's Office. Dữ liệu ban đầu gồm các giao dịch bất động sản tại Ames trong giai đoạn 2006-2010:

- 3.970 giao dịch bất động sản;
- 113 biến mô tả tài sản.

Sau khi loại bỏ các biến đòi hỏi kiến thức chuyên môn hoặc liên quan đến hệ số tính toán nội bộ của cơ quan thẩm định, tác giả giữ lại 80 biến liên quan trực tiếp đến đặc điểm và giao dịch của bất động sản.

## 3. Quá trình xây dựng Ames Housing Dataset

Không phải tất cả 3.970 giao dịch ban đầu đều được giữ lại. Tác giả loại bỏ:

- các giao dịch không phải nhà ở dân dụng;
- các tài sản như garage độc lập, kho hoặc một số loại tài sản đặc biệt;
- các lần bán lặp lại của cùng một bất động sản trong giai đoạn nghiên cứu.

Đối với bất động sản được giao dịch nhiều lần, tác giả chỉ giữ lần bán gần nhất để tránh một số tài sản xuất hiện lặp lại và ảnh hưởng quá lớn đến dữ liệu. Bộ dữ liệu cuối cùng còn **2.930 quan sát**, mỗi quan sát tương ứng với một giao dịch nhà ở dân dụng.

## 4. Cấu trúc bộ dữ liệu

Theo paper, 80 biến được chia thành:

| Loại biến | Số lượng |
| --- | ---: |
| Nominal | 23 |
| Ordinal | 23 |
| Discrete | 14 |
| Continuous | 20 |

### 4.1 Continuous variables

Các biến liên tục chủ yếu mô tả kích thước và diện tích của bất động sản, chẳng hạn diện tích đất, tầng hầm, không gian sinh hoạt, hiên nhà và các khu vực khác. Chúng giúp mô hình phản ánh quy mô vật lý của bất động sản.

### 4.2 Discrete variables

Các biến rời rạc chủ yếu biểu diễn số lượng thành phần trong căn nhà, ví dụ số phòng ngủ, phòng tắm, bếp, sức chứa garage và một số biến liên quan đến năm xây dựng hoặc cải tạo.

### 4.3 Nominal variables

Các biến nominal biểu diễn nhóm không có thứ tự tự nhiên, chẳng hạn loại nhà, loại garage, vật liệu, điều kiện môi trường và neighborhood.

### 4.4 Ordinal variables

Các biến ordinal chủ yếu đánh giá chất lượng hoặc tình trạng của các thành phần trong nhà. Các mức có thứ tự từ thấp đến cao nhưng cần được mã hóa cẩn thận khi đưa vào mô hình.

## 5. Một số biến có ý nghĩa đặc biệt

### 5.1 Neighborhood

`Neighborhood` biểu diễn khu vực căn nhà tọa lạc. Tác giả lưu ý đây không chỉ là một biến phân loại mà còn phản ánh ảnh hưởng của vị trí đến giá nhà. Trong các phân tích của tác giả, `Neighborhood` kết hợp với tổng diện tích nhà có thể giải thích một phần lớn biến động của giá bán.

### 5.2 PID

`PID` là mã định danh của bất động sản trong hệ thống Ames Assessor's Office. Biến này có thể dùng để truy xuất thông tin chi tiết nhưng không phải đặc trưng có ý nghĩa trực tiếp để dự đoán giá nhà.

## 6. Outlier và các quan sát bất thường

Mặc dù lỗi dữ liệu đã được kiểm tra và sửa chữa, một số quan sát bất thường vẫn được giữ nguyên. Tác giả chỉ ra năm trường hợp đáng chú ý khi xét quan hệ giữa `SalePrice` và `Gr Liv Area`:

- ba trường hợp là outlier thực sự;
- hai trường hợp là nhà rất lớn nhưng có mức giá tương đối hợp lý.

Trong một số bài toán hồi quy, có thể cân nhắc loại các căn nhà có diện tích sinh hoạt trên khoảng **4.000 square feet** để giảm ảnh hưởng của quan sát bất thường. Đối với dự án này, nhóm cần thực hiện EDA trước khi quyết định giữ hay loại outlier, thay vì áp dụng ngưỡng tự động.

## 7. Xử lý biến categorical và ordinal

Về lý thuyết, mức chất lượng cao hơn của biến ordinal được kỳ vọng tác động tích cực hơn đến giá nhà. Trong thực tế, hệ số ước lượng không phải lúc nào cũng tăng theo thứ tự các mức chất lượng. Nguyên nhân có thể gồm:

- quan hệ và tương tác giữa các đặc trưng;
- số lượng quan sát không đồng đều giữa các nhóm;
- cách mã hóa biến.

Tác giả gợi ý một số hướng xử lý:

- gộp các mức của biến ordinal;
- chuyển một số biến sang dạng indicator/dummy;
- xử lý một số biến discrete như covariate.

Những nhận xét này cho thấy preprocessing có ảnh hưởng đáng kể đến chất lượng mô hình.

## 8. Modeling và validation

Tác giả khuyến khích chia dữ liệu thành `training set` và `validation set`. Training set được dùng để xây dựng mô hình; validation set đánh giá khả năng dự đoán trên dữ liệu chưa được sử dụng trong quá trình huấn luyện. Việc chia dữ liệu có thể thực hiện bằng random sampling hoặc systematic sampling.

Đây là nguyên tắc quan trọng đối với dự án hiện tại vì MLP không nên chỉ được đánh giá dựa trên training loss.

## 9. Tiêu chí đánh giá mô hình trong paper

| Chỉ số | Công thức | Ý nghĩa |
| --- | --- | --- |
| Bias | `Average(y_hat - y)` | Đo xu hướng định giá cao hoặc thấp hơn thực tế. |
| Maximum Deviation | `max(abs(y - y_hat))` | Xác định sai số lớn nhất trên tập dữ liệu. |
| Mean Absolute Deviation | `Average(abs(y - y_hat))` | Thể hiện chênh lệch tuyệt đối trung bình giữa dự đoán và thực tế. |
| Mean Square Error | `Average((y - y_hat)^2)` | Phạt các lỗi dự đoán lớn mạnh hơn do bình phương sai số. |

## 10. Insight về các yếu tố ảnh hưởng đến giá nhà

Paper cho thấy nhiều biến có quan hệ rõ ràng với giá nhà. Một mô hình hồi quy minh họa sử dụng:

- `Lot Area`;
- `Total Bsmt SF`;
- `Gr Liv Area`;
- `Garage Cars`;
- biến chỉ báo về fireplace.

Mô hình này đạt **R² = 71,4%**. Ngoài ra:

- `Neighborhood` kết hợp `Total Square Footage` có thể giải thích khoảng **80%** biến thiên của giá bán;
- mô hình phức tạp hơn với 36 biến có thể giải thích khoảng **92%** biến thiên, nhưng khó diễn giải hơn.

Giá nhà chịu ảnh hưởng đồng thời từ vị trí, diện tích, chất lượng, tiện ích, cấu trúc và điều kiện của căn nhà.

## 11. Ý nghĩa đối với dự án

Paper là nền tảng để nhóm hiểu bản chất bài toán trước khi xây dựng mô hình. Các kết luận áp dụng trực tiếp cho dự án gồm:

- đây là bài toán regression;
- giá nhà phụ thuộc vào nhiều biến continuous, discrete, categorical và ordinal;
- dữ liệu cần được preprocessing cẩn thận;
- categorical và ordinal features cần chiến lược encoding phù hợp;
- diện tích và vị trí có ảnh hưởng mạnh đến giá bán;
- cần kiểm tra outlier trước khi huấn luyện;
- cần validation độc lập để đánh giá khả năng tổng quát hóa;
- mô hình phức tạp hơn chưa chắc tốt hơn nếu khó giải thích hoặc bị overfitting.

Các insight này là cơ sở cho EDA, preprocessing, baseline model và MLP bằng PyTorch.

## 12. Kết luận

Ames Housing phù hợp với bài toán dự đoán giá nhà vì có nhiều quan sát, biến đa dạng và phản ánh nhiều khía cạnh thực tế của bất động sản. Paper không chỉ cung cấp nguồn gốc dữ liệu mà còn chỉ ra các vấn đề quan trọng về categorical variables, ordinal variables, outlier, lựa chọn đặc trưng và validation.

Đối với dự án *House Prices Prediction Using PyTorch*, paper cung cấp cơ sở để thiết kế quy trình CRISP-DM:

```text
Data Understanding -> Data Preparation -> Modeling -> Evaluation
```

## Tài liệu tham khảo

De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*. Journal of Statistics Education, 19(3). [Bản PDF lưu trong dự án](../docs/reports/decock.pdf).
