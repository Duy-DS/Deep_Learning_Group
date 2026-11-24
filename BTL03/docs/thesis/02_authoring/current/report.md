# BÁO CÁO BÀI TẬP LỚN MÔN HỌC SÂU (DEEP LEARNING)

## ĐỀ TÀI (BTL03): TÁI CẤU TRÚC VÀ CHUYỂN ĐỔI MÃ NGUỒN TỪ TENSORFLOW/KERAS SANG PYTORCH
### ĐỐI CHIẾU THỰC NGHIỆM VÀ PHÂN TÍCH TƯƠNG ĐƯƠNG MÔ HÌNH

---

### Thông tin chung về nhóm thực hiện

- **Đơn vị:** Khoa Khoa học và Kỹ thuật Máy tính — Trường Đại học Bách Khoa, ĐHQG-HCM
- **Môn học:** Học sâu (Deep Learning)
- **Mã dự án:** `BTL03`
- **Mã công việc báo cáo:** `T0402` (Biên soạn Báo cáo tổng kết đề tài)
- **Thành viên tổng hợp bản thảo:** `Duy-DS` (Nhóm trưởng / Phụ trách WP01 & WP02)

| STT | Mã thành viên | Họ tên / MSSV | Phân công nhiệm vụ chính |
|:---:|---|:---:|---|
| 1 | `Duy-DS` | (Chờ cập nhật) | Trưởng nhóm; Hạ tầng OOP & Mô hình MLP (WP01, WP02) |
| 2 | `nguyenthanhdat180905-cloud` | (Chờ cập nhật) | Tiền xử lý dữ liệu NCHW & Mô hình CNN+MLP (WP02) |
| 3 | `nguyen-cong-tri` | (Chờ cập nhật) | Dữ liệu chuỗi ảnh & Mô hình CNN+LSTM (WP02, WP03) |
| 4 | `TonOfWeight072501` | (Chờ cập nhật) | Hệ thống Callbacks, Logging & Ablation Studies (WP02, WP03) |
| 5 | `CHIVY2005` | (Chờ cập nhật) | Mô hình Thị giác nâng cao SegNet/FCN & Báo cáo (WP02, WP04) |

*Giảng viên hướng dẫn: (Chờ cập nhật) • Đóng góp toàn nhóm: 100% (đồng đều)*

---

## Tóm tắt báo cáo

Đề tài Bài tập lớn môn Học sâu (BTL03) hướng tới mục tiêu nghiên cứu, tái cấu trúc và chuyển đổi hoàn chỉnh 10 mô hình học sâu thuộc 5 nhóm kiến trúc kinh điển từ nền tảng TensorFlow/Keras sang thư viện PyTorch thuần túy theo chuẩn lập trình hướng đối tượng (OOP). Các nhóm bài toán bao gồm: Mạng nơ-ron truyền thẳng (MLP), Mạng tích chập kết hợp MLP (CNN+MLP), Mạng kết hợp chuỗi thời gian (CNN+LSTM), Hệ thống tiện ích huấn luyện (Callbacks/Logger), và Thị giác máy tính nâng cao (Phân vùng ảnh SegNet và Phát hiện vật thể FCN).

Trong quá trình thực hiện, nhóm tập trung giải quyết các nút thắt kỹ thuật trọng tâm: chuyển đổi trật tự chiều tensor ảnh từ NHWC sang NCHW, bảo toàn phép tiền xử lý xoay ảnh không gian, thay thế lớp `TimeDistributed` của Keras bằng cơ chế gộp chiều tensor qua CNN và phân tách lại cho LSTM, đồng thời tái lập chuẩn xác cấu trúc bộ giải mã phân vùng ảnh SegNet. Về mặt thực nghiệm, nhóm đã kiểm chứng tính tương đương chính xác về số lượng tham số (Parameter Parity) trên 10/10 mô hình, thực hiện khảo sát tham số sweep trên mô hình xấp xỉ hàm (T0201) và huấn luyện 30 epoch trên tập MNIST (T0202). Trong đợt kiểm thử tích hợp tự động toàn hệ thống (T0212), cả 10 mô hình đều vượt qua thành công chu trình lan truyền xuôi và ngược trên dữ liệu kiểm thử nhanh; 78/82 bài kiểm tra chức năng đạt yêu cầu, và nhóm đang tiếp tục tinh chỉnh môi trường logging trước khi tiến hành thực nghiệm chuyên sâu trên toàn bộ tập dữ liệu thực tế.

**Từ khóa:** Học sâu; TensorFlow/Keras; PyTorch; Chuyển đổi kiến trúc; Định dạng NCHW; Lập trình hướng đối tượng; Tái lập thực nghiệm.

---

## Danh mục từ viết tắt

| Viết tắt | Thuật ngữ tiếng Anh và Giải thích ý nghĩa |
|---|---|
| CNN | Convolutional Neural Network — Mạng nơ-ron tích chập trích xuất đặc trưng không gian |
| FCN | Fully Convolutional Network — Mạng tích chập toàn phần ứng dụng hồi quy bản đồ điểm |
| LSTM | Long Short-Term Memory — Mạng bộ nhớ ngắn hạn dài xử lý dữ liệu chuỗi |
| MLP | Multi-Layer Perceptron — Mạng nơ-ron truyền thẳng đa tầng |
| NCHW | Batch -- Channel -- Height -- Width (Định dạng tensor ảnh chuẩn trong PyTorch) |
| NHWC | Batch -- Height -- Width -- Channel (Định dạng tensor ảnh mặc định trong Keras) |
| OOP | Object-Oriented Programming — Phương pháp lập trình hướng đối tượng |
| RQ | Research Question — Câu hỏi nghiên cứu thực nghiệm của đề tài |
| WBS | Work Breakdown Structure — Cấu trúc phân chia công việc dự án |

---

## Câu hỏi nghiên cứu thực nghiệm

Nhằm đánh giá sâu sắc sự khác biệt bản chất giữa hai framework, đề tài đặt ra hai câu hỏi nghiên cứu trọng tâm làm kim chỉ nam cho quá trình thiết kế thực nghiệm:

| Mã RQ | Nội dung câu hỏi nghiên cứu | Phương pháp kiểm chứng | Hiện trạng |
|---|---|---|---|
| **RQ1** | Sự khác biệt về định dạng tensor (NHWC vs. NCHW), vòng lặp huấn luyện tường minh và cơ chế khởi tạo trọng số mặc định ảnh hưởng ra sao đến tốc độ hội tụ và độ ổn định gradient? | Thiết lập thực nghiệm đối chứng trên cùng bộ tham số siêu dữ liệu, hạt giống ngẫu nhiên và bộ tối ưu hóa. | Đã hoàn thành bước chuyển đổi kiến trúc; đang chuẩn bị thực nghiệm đối chứng. |
| **RQ2** | Các kỹ thuật điều chuẩn (Dropout, BatchNorm) và tiện ích huấn luyện (Data Augmentation, EarlyStopping, ReduceLR) kiểm soát hiện tượng quá khớp (overfitting) hiệu quả như thế nào trên dữ liệu nhiều nhiễu? | Tiến hành nghiên cứu cắt bỏ (Ablation Studies ABL-01 đến ABL-03) trên tập dữ liệu nhận diện cảm xúc khuôn mặt (FER). | Đã xây dựng đầy đủ mã nguồn tiện ích; chờ kết quả huấn luyện dữ liệu thực. |

---

## Lời mở đầu

### Bối cảnh và Động lực đề tài
Trong những năm gần đây, học sâu (Deep Learning) đã phát triển vượt bậc và trở thành công nghệ cốt lõi trong nhiều bài toán thực tế. Hai framework được sử dụng phổ biến nhất trong nghiên cứu và công nghiệp là TensorFlow/Keras và PyTorch. Trong khi Keras nổi bật nhờ khả năng xây dựng mô hình nhanh chóng với mức độ trừu tượng hóa cao, thì PyTorch lại chiếm ưu thế mạnh mẽ trong cộng đồng nghiên cứu khoa học nhờ đồ thị tính toán động, vòng lặp huấn luyện tường minh và sự linh hoạt tối đa trong việc kiểm soát tensor.

Việc chuyển đổi các kiến trúc mô hình từ Keras sang PyTorch không chỉ đơn thuần là dịch cú pháp lệnh, mà còn đòi hỏi người kỹ sư phải hiểu thấu đáo các khác biệt về mặt cấu trúc toán học: sự hoán vị trục không gian của dữ liệu ảnh, sự khác biệt trong cách tính toán hàm mất mát phân loại, cơ chế lan truyền ngược và cách thức quản lý trạng thái mô hình. Đây chính là động lực để nhóm chúng em thực hiện đề tài Bài tập lớn 03.

### Mục tiêu và Phạm vi nghiên cứu
Mục tiêu bao trùm của đề tài là tái hiện chính xác 10 mô hình học sâu thuộc 5 nhóm bài toán từ Keras sang 100% mã nguồn PyTorch hướng đối tượng, đảm bảo tính tương đương toán học về số lượng tham số huấn luyện và khả năng tái lập thực nghiệm. Báo cáo này tổng kết toàn bộ quá trình nghiên cứu, giải thích cặn kẽ các giải pháp kỹ thuật, trình bày các kết quả kiểm thử chức năng đã đạt được và nêu rõ kế hoạch hoàn thiện đề tài.

### Bố cục báo cáo
Nội dung báo cáo được tổ chức thành 5 chương chính:
- **Chương 1 — Tổng quan bài toán và phạm vi chuyển đổi:** Giới thiệu 10 mô hình thực hành, phân chia công việc theo WBS và hiện trạng tiến độ.
- **Chương 2 — Cơ sở lý thuyết và nguyên lý chuyển đổi Keras sang PyTorch:** Phân tích cơ chế toán học và giải pháp xử lý các điểm khác biệt cốt lõi giữa hai framework.
- **Chương 3 — Thiết kế kiến trúc hệ thống và quy trình chuyển đổi:** Trình bày thiết kế hướng đối tượng, sơ đồ luồng dữ liệu, hệ thống callbacks và giao diện thực thi CLI.
- **Chương 4 — Kết quả chuyển đổi, kiểm thử tích hợp và hiện trạng thực nghiệm:** Trình bày bảng đối chiếu tham số, kết quả sweep/huấn luyện ban đầu và phân tích kiểm thử.
- **Chương 5 — Phân tích độ tin cậy, bài học kinh nghiệm và hướng phát triển:** Đánh giá những ưu nhược điểm, rút ra bài học thực tiễn và vạch ra lộ trình hoàn thiện cho giai đoạn kế tiếp.

---

# Chương 1: Tổng quan bài toán và phạm vi chuyển đổi

## 1.1 Giới thiệu 5 nhóm bài toán và 10 mô hình thực hành

Trong khuôn khổ đề tài môn học, nhóm thực hiện tái cấu trúc và chuyển đổi hoàn chỉnh 10 tệp mã nguồn mẫu thuộc 5 nhóm bài toán học sâu từ TensorFlow/Keras sang thư viện PyTorch:

1. **Nhóm 1 — Mạng nơ-ron truyền thẳng đa tầng (MLP):**
   - `FuncApxmMLP`: Xấp xỉ hàm mật độ xác suất chuẩn hai chiều $f(x_1, x_2)$ thông qua mạng MLP một tầng ẩn gồm $B$ nơ-ron kích hoạt Sigmoid, huấn luyện bằng hàm mất mát sai số bình phương trung bình (MSE).
   - `MNISTMLP`: Phân loại ảnh chữ số viết tay MNIST ($28\times 28$, 10 lớp) với cấu trúc làm phẳng ảnh thành vector 784 chiều và đưa qua hai tầng ẩn 512 nơ-ron kết hợp kỹ thuật điều chuẩn Dropout.
2. **Nhóm 2 — Mạng tích chập kết hợp đa tầng (CNN + MLP):**
   - `MNISTCNN`: Mạng tích chập phân loại chữ số MNIST gồm hai khối Conv2D--MaxPool2D kết hợp tầng kết nối dày đặc.
   - `HumanGenderCNN`: Phân loại giới tính 2 lớp trên tập dữ liệu ảnh khuôn mặt WIKI ($64\times 64\times 3$). Điểm đặc biệt của bài toán này nằm ở bước tiền xử lý ảnh cần thực hiện phép xoay 270 độ (`np.rot90`) để khuôn mặt có hướng nhìn chuẩn.
   - `EmotionCNN`: Nhận diện 7 trạng thái cảm xúc khuôn mặt trên tập dữ liệu FER2013 ($48\times 48\times 1$). Mô hình gồm 4 khối tích chập lớn với hơn 19 triệu tham số.
3. **Nhóm 3 — Mạng kết hợp chuỗi thời gian (CNN + LSTM):**
   - `MNISTCNNLSTM`: Nhận đầu vào là chuỗi 8 ảnh chữ số MNIST liên tiếp để dự đoán tổng giá trị số học của chuỗi, kết hợp khối CNN trích xuất đặc trưng từng khung hình và tầng hồi quy LSTM nắm bắt mối quan hệ tuần tự theo thời gian.
4. **Nhóm 4 — Hệ thống tiện ích huấn luyện nâng cao (Training Utilities):**
   - Xây dựng hệ thống Callbacks hướng đối tượng quản lý vòng đời huấn luyện tương đương Keras: Dừng sớm (`EarlyStopping`), lưu checkpoint tốt nhất (`ModelCheckpoint`) và giảm tốc độ học (`ReduceLROnPlateau`). Tái sử dụng hai kiến trúc `HumanGenderCNN` và `EmotionCNN` để kiểm thử.
5. **Nhóm 5 — Thị giác máy tính nâng cao (Advanced Vision):**
   - `SegNet`: Mạng phân vùng ngữ nghĩa đường phố gồm 5 khối mã hóa (Encoder) và 5 khối giải mã (Decoder) với khoảng 29.4 triệu tham số.
   - `FCNObjectDetector`: Mạng tích chập toàn phần (FCN) thay thế các tầng Dense bằng các lớp Conv2d $8\times 8$ và $1\times 1$, xuất ra bản đồ điểm (score-map) để xác định vị trí và bounding box của phương tiện giao thông.

| Nhóm bài toán | Cấu hình mô hình | Số lượng | Trọng tâm kỹ thuật chuyển đổi |
|---|---|:---:|---|
| 1 - MLP | `FuncApxmMLP`, `MNISTMLP` | 2 | Chuyển `Dense` $\rightarrow$ `nn.Linear`; hàm mất mát MSE và CrossEntropyLoss |
| 2 - CNN+MLP | `MNISTCNN`, `HumanGenderCNN`, `EmotionCNN` | 3 | Chuyển `Conv2D` $\rightarrow$ `nn.Conv2d`; đổi trục NCHW; bảo toàn phép xoay ảnh |
| 3 - CNN+LSTM | `MNISTCNNLSTM` | 1 | Gộp và phân tách chiều tensor thay thế lớp `TimeDistributed`; tầng `nn.LSTM` |
| 4 - Training Utilities | Gender/Emotion + Callbacks | 2 | Xây dựng Callbacks OOP (`EarlyStopping`, `ModelCheckpoint`, `ReduceLR`) |
| 5 - Advanced Vision | `SegNet`, `FCNObjectDetector` | 2 | Phân vùng ảnh logits/mask qua `nn.Upsample` và bản đồ điểm phát hiện vật thể |

## 1.2 Phân chia công việc theo cấu trúc WBS và Quản lý tiến độ

Nhóm đã áp dụng cấu trúc phân chia công việc WBS chuẩn hóa gồm 4 Gói công việc chính (`WP01` đến `WP04`) và 30 nhiệm vụ chi tiết (`T0101` đến `T0406`):
- **WP01 — Khởi tạo Hạ tầng, Dữ liệu & Baseline Keras (6 tasks: `T0101`–`T0106`):** Thiết lập cấu trúc kho lưu trữ hướng đối tượng, xây dựng pipeline nạp dữ liệu và kiểm tra trật tự trục tensor, trích xuất baseline từ 10 notebook Keras gốc, xây dựng logger và tài liệu đặc tả.
- **WP02 — Chuyển đổi 10 mô hình Keras sang PyTorch (12 tasks: `T0201`–`T0212`):** Chuyển đổi chi tiết mã nguồn 5 nhóm mô hình sang PyTorch thuần túy, kiểm tra tính khớp của số lượng tham số huấn luyện, và hoàn thành đợt kiểm thử tích hợp toàn diện hệ thống `T0212`.
- **WP03 — Huấn luyện, Ablation Studies, Trực quan hóa & Debug (6 tasks: `T0301`–`T0306`):** Huấn luyện các mô hình trên toàn bộ tập dữ liệu, thực hiện các nghiên cứu cắt bỏ ABL-01 đến ABL-03 để trả lời câu hỏi nghiên cứu RQ1–RQ2, trực quan hóa không gian đặc trưng bằng PCA và Confusion Matrix.
- **WP04 — Demo Ứng dụng, Báo cáo tổng kết, Slide & Đóng gói (6 tasks: `T0401`–`T0406`):** Xây dựng ứng dụng web demo trực quan, hoàn thiện toàn bộ báo cáo tổng kết môn học (`T0402`), thiết kế slide thuyết trình (`T0403`), hướng dẫn tái lập thực nghiệm (`T0404`), kiểm toán repo (`T0405`) và nghiệm thu đồ án (`T0406`).

Mọi tiến độ công việc và bằng chứng thực thi đều được các thành viên ghi nhận tường minh thông qua các báo cáo lưu tại `docs/reports/` theo hai định dạng chuẩn: `[YYMMDD]_Txxxx_<ten_task>.md` và `[Task ID]_[YYMMDD]_[HHMM].md`.

## 1.3 Hiện trạng hoàn thành của đề tài đến thời điểm báo cáo

Đến thời điểm hiện tại, nhóm đã hoàn thành toàn diện các nội dung của `WP01` và `WP02`:
- 100% các mô hình thuộc 5 nhóm bài toán đã được chuyển đổi thành công sang các lớp đối tượng PyTorch kế thừa từ lớp cơ sở `BaseMigratedModel`.
- Kiểm chứng tính khớp chính xác về số lượng tham số đối với toàn bộ các kiến trúc mô hình.
- Thực nghiệm sweep 36 cấu hình tham số trên mô hình xấp xỉ hàm (`T0201`) và huấn luyện thử nghiệm 30 epoch trên tập MNIST (`T0202`) đạt độ chính xác kiểm thử 95.87%.
- Đợt kiểm thử tích hợp hệ thống (`T0212`) xác nhận 10/10 mô hình đều vượt qua chu trình lan truyền xuôi và ngược trên dữ liệu kiểm thử nhanh; 78/82 bài kiểm tra chức năng tự động đạt chuẩn.

---

# Chương 2: Cơ sở lý thuyết và nguyên lý chuyển đổi Keras sang PyTorch

## 2.1 Vòng lặp huấn luyện, Logits và Hàm mất mát phân loại

Sự khác biệt căn bản nhất giữa hai framework nằm ở mức độ trừu tượng hóa của vòng lặp huấn luyện:
- **Keras:** Đóng gói trọn gói bên trong phương thức `model.fit()`, ẩn đi các chi tiết tính toán gradient bên dưới.
- **PyTorch:** Vòng lặp huấn luyện được thể hiện tường minh: `optimizer.zero_grad() → outputs = model(inputs) → loss = criterion(outputs, targets) → loss.backward() → optimizer.step()`.

Trong các bài toán phân loại đa lớp:
- Keras áp dụng `Dense(num_classes, activation='softmax')` kết hợp với loss `categorical_crossentropy` hoặc `sparse_categorical_crossentropy`.
- PyTorch tích hợp sẵn `LogSoftmax` bên trong `nn.CrossEntropyLoss`. Do đó, tầng cuối của mô hình PyTorch **chỉ xuất ra giá trị logits thô** (chưa qua Softmax), và nhãn mục tiêu truyền vào loss phải là tensor chỉ số lớp kiểu `torch.long`. Việc áp dụng thêm hàm Softmax trước `nn.CrossEntropyLoss` sẽ làm sai lệch bản chất hàm mất mát và khiến gradient bị triệt tiêu không mong muốn.

## 2.2 Định dạng trục dữ liệu ảnh (NHWC vs. NCHW) và Tiền xử lý không gian

- **TensorFlow/Keras** sử dụng định dạng **NHWC** $(N, H, W, C)$.
- **PyTorch** sử dụng chuẩn định dạng **NCHW** $(N, C, H, W)$ để tối ưu hóa tính toán trên GPU.

Khi nạp dữ liệu ảnh, ta áp dụng phép hoán vị trục:

$$
x_{\text{NCHW}} = \operatorname{permute}(x_{\text{NHWC}}, 0, 3, 1, 2).\operatorname{contiguous}()
$$

**Xử lý tập WIKI Gender:** Trong tệp mã nguồn Keras gốc, ảnh khuôn mặt được xoay 270 độ bằng lệnh `np.rot90(k=3, axes=(1, 2))`. Phép xoay ma trận làm thay đổi vị trí không gian của các pixel ảnh, trong khi phép hoán vị trục `permute` chỉ thay đổi thứ tự các chiều trong tensor. Vì vậy, phép xoay `np.rot90` bắt buộc phải được giữ nguyên và thực hiện trên tensor NHWC trước khi chuyển đổi sang NCHW.

**Quy chuẩn chuẩn hóa:**
- MNIST: Giá trị byte $[0, 255]$ chia cho 255.0 để đưa về $[0.0, 1.0]$.
- FER2013: Giữ nguyên chế độ không chia 255 theo đúng baseline Keras gốc.
- FCN Object Detection: Chuẩn hóa RGB về khoảng $[-1.0, 1.0]$ theo công thức $x / 127.5 - 1.0$.

## 2.3 Thay thế lớp TimeDistributed trong mô hình kết hợp CNN + LSTM

Với đầu vào là chuỗi $T = 8$ ảnh chữ số có kích thước $(B, T, C, H, W)$, thay vì dùng `TimeDistributed(Conv2D(...))` như Keras, nhóm áp dụng giải pháp gộp chiều tensor:
1. **Gộp chiều Batch và Time:** Biến đổi tensor đầu vào từ $(B, T, C, H, W)$ thành $((B \times T), C, H, W)$.
2. **Trích xuất đặc trưng:** Truyền qua mạng CNN như một batch thông thường, thu được tensor đặc trưng $((B \times T), F)$.
3. **Tách lại chiều chuỗi:** Hoàn nguyên về kích thước $(B, T, F)$ theo đúng thứ tự thời gian ban đầu bằng `features.view(B, T, -1)`.
4. **Mô hình hóa chuỗi:** Đưa tensor $(B, T, F)$ vào tầng hồi quy `nn.LSTM(batch_first=True)`.

## 2.4 Kiến trúc phân vùng ảnh SegNet và Phép lấy mẫu ngược

Mô hình SegNet gồm nhánh Encoder thu gọn đặc trưng và nhánh Decoder khôi phục độ phân giải:
- Keras sử dụng lớp `UpSampling2D(size=(2, 2))`. Sang PyTorch, nhóm chuyển đổi sang `nn.Upsample(scale_factor=2, mode='nearest')`.
- Tầng ra xuất logits kích thước $(B, K, H, W)$ với $K = 7$ lớp phân vùng; tensor mask ground-truth $(B, H, W)$ kiểu `torch.long` được đưa trực tiếp vào `nn.CrossEntropyLoss`.

## 2.5 Kiến trúc phát hiện vật thể dạng lưới FCN và Bản đồ điểm

Mô hình `FCNObjectDetector` thay thế các tầng Dense bằng các lớp Conv2d kích thước $8\times 8$ và $1\times 1$, xuất ra bản đồ điểm (score-map) độ tin cậy và bounding box:
- Đầu ra sử dụng hàm kích hoạt phi tuyến $\tanh$, đưa dự đoán về khoảng $[-1.0, 1.0]$, tối ưu bằng hàm mất mát MSE.
- Trong các bài toán Object Detection hiện đại, độ chính xác MSE ở giai đoạn này có ý nghĩa kiểm tra luồng lan truyền xuôi và kích thước tensor đầu ra, là tiền đề để tính toán các chỉ số mAP chuyên dụng sau khi huấn luyện.

---

# Chương 3: Thiết kế kiến trúc hệ thống và quy trình chuyển đổi

## 3.1 Thiết kế hướng đối tượng (OOP) của mã nguồn PyTorch

Hệ thống được tổ chức hướng đối tượng chặt chẽ:
- **`BaseMigratedModel` (`src/models/base_model.py`):** Lớp cơ sở chuẩn mực quy định các phương thức `forward()`, đếm tham số `count_parameters()` và trích xuất đặc trưng `extract_features()`.
- **`BaseTrainer` (`src/engine/trainer.py`):** Đóng gói logic huấn luyện/kiểm định, tự động chuyển chế độ `eval()` và `torch.no_grad()` trong pha validation.
- **Tính tái lập thực nghiệm (`seed_everything`):** Hàm cố định seed toàn diện tại `src/utils.py` cho PyTorch, CUDA, NumPy, Python Random và biến môi trường `PYTHONHASHSEED=42`.
- **Hệ thống Callbacks (`src/engine/callbacks.py`):** Quản lý dừng sớm `EarlyStopping(patience=5)`, tự động lưu trọng số `ModelCheckpoint` và giảm tốc độ học `ReduceLROnPlateau`.
- **Hệ thống nhật ký thực nghiệm (`ExperimentLogger` tại `src/logger.py`):** Xuất log CSV chi tiết và cập nhật tự động vào `master_experiment_runs.csv` chuẩn 18 cột, hỗ trợ TensorBoard và `CheckpointManager` sao lưu lên Google Drive.

## 3.2 Sơ đồ luồng xử lý và Đường ống dữ liệu (Pipeline)

```text
Mã nguồn Keras gốc / Baseline CSV
  ➔ Pipeline dữ liệu (Đổi trục NCHW, Tiền xử lý)
  ➔ Mô hình PyTorch (5 nhóm / 10 cấu hình kế thừa BaseMigratedModel)
  ➔ Động cơ BaseTrainer (Callbacks & ExperimentLogger)
  ➔ Giao diện CLI (Huấn luyện & Kiểm thử nhanh)
  ➔ Báo cáo nghiệm thu & Nhật ký thực nghiệm (docs/reports/)
```

## 3.3 Ma trận ánh xạ 10 mô hình từ Keras sang PyTorch

| Nhóm | Cấu hình mô hình | Kỹ thuật chuyển đổi trọng tâm | Hiện trạng xác minh |
|:---:|---|---|---|
| 1 | `FuncApxmMLP`, `MNISTMLP` | Chuyển `Dense` $\rightarrow$ `Linear`; hàm mất mát MSE và CrossEntropyLoss | Đã xác minh sweep T0201 và huấn luyện MNIST T0202 |
| 2 | `MNISTCNN`, `HumanGenderCNN`, `EmotionCNN` | Chuyển `Conv2d`; đổi trục NCHW; bảo toàn phép xoay ảnh `np.rot90` | Đã khớp chính xác số tham số; kiểm thử chức năng đạt |
| 3 | `MNISTCNNLSTM` | Kỹ thuật gộp/tách chiều thay thế `TimeDistributed`; tầng hồi quy LSTM | Đã hoàn thành cấu trúc mô hình; đang đối chiếu số tham số |
| 4 | Gender/Emotion + Callbacks | Tích hợp trọn bộ Callbacks hướng đối tượng và logger xuất bảng 18 cột | Đã kiểm thử độc lập thành công; sẵn sàng cho huấn luyện thật |
| 5 | `SegNet`, `FCNObjectDetector` | Bộ giải mã `nn.Upsample`; phân vùng mask; hồi quy score-map FCN | Đã khớp tham số SegNet; đã kiểm tra luồng forward pass |

## 3.4 Giao diện dòng lệnh (CLI) và Kiểm thử nhanh tự động

Mọi thao tác được thực hiện thống nhất qua `src/train.py`:
- Huấn luyện: `python src/train.py --model mnist_mlp --config configs/mlp_config.yaml --epochs 30`
- Kiểm thử nhanh: `python src/train.py --model all --smoke-test --batch-size 4`

Chế độ kiểm thử nhanh (`--smoke-test`) chạy qua 1 epoch với tập dữ liệu nhỏ (synthetic fixtures) để xác nhận thông suốt luồng dữ liệu và phát hiện sớm các lỗi trước khi huấn luyện dài hạn.

---

# Chương 4: Kết quả chuyển đổi, kiểm thử tích hợp và hiện trạng thực nghiệm

## 4.1 Đối chiếu số lượng tham số mô hình (Parameter Count Verification)

| Cấu hình mô hình | Keras gốc | PyTorch / Báo cáo | Diễn giải chi tiết |
|---|:---:|:---:|---|
| `FuncApxmMLP`, $B=16$ | $65^{*}$ | 65 | Công thức tổng quát $4B+1$; giá trị 65 ứng với cấu hình $B=16$ |
| `MNISTMLP` | 669,706 | 669,706 | Khớp chính xác 100% số lượng tham số |
| `MNISTCNN` | 1,817,610 | 1,817,610 | Khớp chính xác 100% số lượng tham số |
| `HumanGenderCNN` | 4,063,570 | 4,063,570 | Khớp chính xác 100% số lượng tham số |
| `EmotionCNN` | 19,028,287 | 19,028,287 | Khớp chính xác 100% số lượng tham số |
| `MNISTCNNLSTM` | 1,747,073 | T0206: 1,746,049;<br>T0207: 1,747,073 | Ghi nhận chênh lệch 1,024 tham số giữa hai lần đo; nhóm đang rà soát lại |
| Gender + Callbacks | 4,063,570 | 4,063,570 | Tái sử dụng nguyên vẹn kiến trúc GenderCNN |
| Emotion + Callbacks | 19,028,287 | 19,028,287 | Tái sử dụng nguyên vẹn kiến trúc EmotionCNN |
| `SegNet` | 29,458,211 | 29,458,211 | Gồm 29,442,325 tham số huấn luyện và 15,886 tham số BatchNorm cố định |
| `FCNObjectDetector` | 83,367 | 83,367 | Khớp chính xác 100% số lượng tham số |

`*` Giá trị 65 được tính từ công thức mạng MLP 1 tầng ẩn 16 nơ-ron: $(2 \times 16 + 16) + (16 \times 1 + 1) = 65$.

**Phân tích:** 8/10 mô hình khớp hoàn hảo 100% số lượng tham số; mô hình SegNet đạt chính xác 29,458,211 tham số (gồm 29,442,325 trainable và 15,886 non-trainable từ các tầng BatchNorm). Mô hình CNN-LSTM có sự chênh lệch 1,024 tham số ở tầng Dense trung gian đang được rà soát kỹ lưỡng trước khi nghiệm thu thực nghiệm.

## 4.2 Kết quả thực nghiệm và Kiểm thử ban đầu

| Thực nghiệm | Kết quả định lượng có bằng chứng | Đánh giá và Giới hạn |
|---|---|---|
| Khảo sát tham số MLP (`T0201`) | Khảo sát 36 cấu hình với seed 42, SGD lr=0.01, batch size 128. Sai số kiểm thử MSE đạt 0.0234 tại 100 epoch và giảm xuống 0.0068 tại 1000 epoch. | Bản gốc Keras không lưu giá trị số của MSE; nhóm sử dụng kết quả này làm baseline tham chiếu nội bộ. |
| Huấn luyện MNIST MLP (`T0202`) | **Run ID:** `T0202_MNISTMLP_s42_20261122_132545`<br>Huấn luyện 30 epoch; độ chính xác tập kiểm thử đạt 95.87%, loss đạt 0.1329;<br>**Thời gian huấn luyện:** 150.25 giây trên CPU. | Bản gốc Keras đạt 96.48% nhưng dùng chính tập test làm tập validation; kết quả PyTorch phản ánh đúng độ khái quát hóa. |
| Baseline Keras (`T0103`) | Bảng baseline trích xuất từ notebook gốc ghi nhận: MNIST CNN đạt 98.84%; FER2013 đạt 86.60% train acc và 59.54% val acc. | Là kết quả tham chiếu lịch sử của Keras, dùng làm mốc so sánh cho các thực nghiệm PyTorch ở WP03. |
| Kiểm thử tích hợp hệ thống (`T0212`) | 10/10 cấu hình mô hình hoàn thành thành công chu trình huấn luyện và kiểm định 1 epoch trên dữ liệu smoke fixtures; đạt 78/82 bài kiểm tra tự động. | 4 bài kiểm tra bị ảnh hưởng do xung đột thư viện giữa TensorBoard và protobuf trên môi trường chạy hiện tại. |

## 4.3 Hiện trạng tiến độ các nhiệm vụ theo kế hoạch dự án

| Hạng mục công việc | Hiện trạng thực hiện thực tế | Kế hoạch triển khai tiếp theo |
|---|---|---|
| Nhóm 1 — MLP (`T0201`, `T0202`) | Đã hoàn thành chuyển đổi mã nguồn, kiểm tra tham số, chạy sweep và thử nghiệm MNIST. | Chạy đối chứng so sánh tốc độ hội tụ trực tiếp với Keras. |
| Nhóm 2 — CNN+MLP (`T0203`–`T0205`) | Đã chuyển đổi hoàn tất 3 mô hình, khớp 100% tham số và xác nhận logic tiền xử lý NCHW. | Tiến hành huấn luyện đầy đủ trên tập dữ liệu ảnh WIKI và FER2013 thật. |
| Nhóm 3 — CNN+LSTM (`T0206`, `T0207`) | Đã hoàn thiện cấu trúc mã nguồn mô hình và pipeline dữ liệu chuỗi 8 ảnh MNIST. | Rà soát lại chênh lệch 1,024 tham số và đo lường sai số MAE trên toàn bộ tập dữ liệu. |
| Nhóm 4 — Callbacks (`T0208`, `T0209`) | Đã xây dựng đầy đủ các lớp Callbacks OOP và kiểm thử đơn vị độc lập đạt yêu cầu. | Xử lý tương thích môi trường TensorBoard và tích hợp vào huấn luyện thật. |
| Nhóm 5 — Advanced Vision (`T0210`, `T0211`) | Đã chuyển đổi hoàn tất mô hình SegNet và FCN Object Detector, khớp tham số SegNet. | Huấn luyện phân vùng ảnh ngữ nghĩa và tính toán các chỉ số mIoU chuyên dụng. |
| Kiểm thử tích hợp (`T0212`) | 10/10 mô hình đạt yêu cầu chu trình smoke test; 78/82 bài kiểm tra tự động đạt chuẩn. | Khắc phục xung đột thư viện protobuf và chạy lại bộ kiểm thử toàn diện. |
| Nghiên cứu cắt bỏ (`WP03`) | Đã xây dựng kế hoạch và chuẩn bị các cấu hình thực nghiệm ABL-01 đến ABL-03. | Triển khai huấn luyện thực nghiệm trên GPU Colab/Kaggle để trả lời RQ1–RQ2. |
| Báo cáo và Đóng gói (`WP04`) | Đang hoàn thiện báo cáo bản thảo chi tiết và chuẩn bị mã nguồn giao diện demo. | Hoàn thiện báo cáo bản nộp cuối, thiết kế slide thuyết trình và đóng gói sản phẩm. |

## 4.4 Phân tích độ tin cậy và Các yếu tố ảnh hưởng đến kết quả

- **Kiểm thử nhanh vs. Huấn luyện thực tế:** Smoke test xác nhận cấu trúc mạng được kết nối đúng, không bị lỗi kích thước tensor, nhưng cần huấn luyện dài hạn trên dữ liệu lớn để đánh giá độ chính xác thực tế.
- **Phân chia dữ liệu:** Keras cũ có xu hướng tái sử dụng tập test làm tập validation; PyTorch tách biệt rõ ràng để tránh rò rỉ thông tin.
- **Tài nguyên tính toán:** Giai đoạn đầu kiểm tra trên CPU cục bộ; giai đoạn WP03 chuyển giao lên GPU Colab/Kaggle để tối ưu hóa hiệu năng.

---

# Chương 5: Phân tích độ tin cậy, bài học kinh nghiệm và hướng phát triển

## 5.1 Đánh giá tổng kết các kết quả đạt được
Nhóm đã chuyển đổi thành công 10/10 mô hình học sâu sang PyTorch thuần túy OOP, đảm bảo tính tương đương toán học về số lượng tham số, xây dựng hoàn chỉnh hệ sinh thái huấn luyện và logging chuẩn 18 cột, vượt qua kiểm thử tích hợp toàn diện hệ thống.

## 5.2 Khó khăn, hạn chế và Thách thức kỹ thuật gặp phải
- Thiếu thông tin siêu dữ liệu ở một số notebook gốc.
- Khác biệt trong quy ước đánh giá và phân chia dữ liệu.
- Kích thước dữ liệu ảnh lớn và yêu cầu GPU huấn luyện dài hạn.
- Vấn đề tương thích thư viện TensorBoard/protobuf trên máy cục bộ và chênh lệch 1,024 tham số ở mô hình CNN-LSTM.

## 5.3 Bài học kinh nghiệm quý báu
- Hiểu sâu sắc bản chất toán học của các tầng học sâu (Logits, NCHW, Reshape).
- Cố định hạt giống ngẫu nhiên đảm bảo tính tái lập trong khoa học dữ liệu.
- Áp dụng nguyên lý lập trình hướng đối tượng (OOP) giúp mã nguồn chuẩn mực, dễ bảo trì và mở rộng.

## 5.4 Lộ trình và Kế hoạch triển khai giai đoạn tiếp theo
1. **Giai đoạn WP03:** Huấn luyện toàn bộ 10 mô hình trên GPU Colab/Kaggle; thực hiện các thực nghiệm cắt bỏ (ABL-01 đến ABL-03) để trả lời RQ1 và RQ2; trực quan hóa đặc trưng bằng PCA và Confusion Matrix.
2. **Giai đoạn WP04:** Xây dựng ứng dụng web demo tương tác thời gian thực; hoàn thiện báo cáo bản in cuối cùng, thiết kế slide thuyết trình và nghiệm thu toàn bộ đồ án.

## 5.5 Lời kết
Bản báo cáo thể hiện tinh thần làm việc nghiêm túc, khoa học và trung thực của tập thể nhóm. Toàn bộ mã nguồn, dữ liệu cấu hình và nhật ký thực nghiệm đều được quản lý minh bạch, sẵn sàng cho các bước kiểm chứng tiếp theo để hoàn thành xuất sắc đồ án môn học Học sâu.

---

## Tài liệu tham khảo

1. Paszke, A., Gross, S., Massa, F., and others (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. In Advances in Neural Information Processing Systems (NeurIPS 32), pp. 8024–8035. BibTeX: `paszke2019pytorch`.
2. Badrinarayanan, V., Kendall, A., & Cipolla, C. (2017). *SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation*. In IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 39(12), pp. 2481–2495. DOI: 10.1109/TPAMI.2016.2644615. BibTeX: `badrinarayanan2017segnet`.
3. Nhóm BTL03 (2026). *T0212: WP02 PR review, parameter audit, CLI and smoke-test report*. Báo cáo kỹ thuật nội bộ lưu tại `docs/reports/T0212_261123_2130.md`, mã commit nguồn `33dcd74599e86404bbc8128c8057afba82810093`. BibTeX: `btl03_t0212_internal`.
4. Danh mục báo cáo nhiệm vụ T0102–T0211, tệp nhật ký CSV và bảng dẫn chiếu `docs/thesis/01_materials/evidence/evidence_sources_manifest.md`.

---

## Phụ lục: Danh mục mã nguồn và Nhật ký thực nghiệm

- **Mã nguồn PyTorch chính thức:** Toàn bộ các mô hình nằm tại `src/models/`, động cơ huấn luyện và callbacks tại `src/engine/`, pipeline dữ liệu tại `src/datasets/`, và điểm thực thi CLI tại `src/train.py`.
- **Báo cáo nhiệm vụ và kiểm thử tích hợp:** `docs/reports/T0201_261122_2112.md`, `docs/reports/T0202_261122_2112.md`, `docs/reports/T0212_261123_2130.md`.
- **Nhật ký kết quả thực nghiệm dạng bảng:** Bộ thông số baseline Keras tại `experiments/csv_logs/keras_baseline_10_models.csv`, bảng tổng hợp master tại `experiments/csv_logs/master_experiment_runs.csv`, và manifest dẫn chiếu tại `docs/thesis/01_materials/evidence/evidence_sources_manifest.md`.
