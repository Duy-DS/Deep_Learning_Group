# Keras legacy baseline — BTL03

Các notebook dưới đây là bản sao để đối chiếu, không phải mã PyTorch đã chuyển đổi. Nguồn gốc nằm trong `docs/project_charter/examples/` và được giữ nguyên. Thông tin được trích từ nội dung notebook; mục `không nêu rõ` có nghĩa là không xác định được một cách chắc chắn trong notebook.

| Nhóm | Tệp trong legacy_keras | Input shape | Output / số lớp | Loss | Optimizer | Ghi chú |
|---|---|---|---|---|---|---|
| 01 MLP | `01_mlp/Func_apxm_MLP.ipynb` | `(2,)` | 1 đầu ra liên tục (xấp xỉ mật độ Gaussian) | `mean_squared_error` | SGD (`lr=0.01`) | MLP xấp xỉ hàm Gaussian; không phải bài toán phân lớp. |
| 01 MLP | `01_mlp/MNIST_MLP.ipynb` | `(28, 28)` | 10 chữ số MNIST | `categorical_crossentropy` | SGD (`lr` lấy từ biến trong notebook) | Nhãn one-hot; Flatten/Dense. |
| 02 CNN + MLP | `02_cnn_mlp/1_MNIST_CNN.ipynb` | `(28, 28, 1)` | 10 chữ số MNIST | `categorical_crossentropy` | SGD (`lr` lấy từ biến trong notebook) | CNN cho ảnh MNIST, đầu ra phân lớp softmax. |
| 02 CNN + MLP | `02_cnn_mlp/2_Human_Gender_CNN.ipynb` | `(64, 64, 1)` | 2 (male/female) | `categorical_crossentropy` | Adam (`lr` lấy từ biến trong notebook) | Tăng cường dữ liệu dùng `np.rot90`; không có nhóm callbacks của Training Utilities. |
| 02 CNN + MLP | `02_cnn_mlp/3_Emotion_Reg_CNN.ipynb` | `(48, 48, 1)` | 7 cảm xúc | `categorical_crossentropy` | Adam (`lr` lấy từ biến trong notebook) | CNN phân lớp cảm xúc; bản cơ bản, không khai báo callbacks huấn luyện ở nhóm 4. |
| 03 CNN + LSTM | `03_cnn_lstm/MNIST_CNN_LSTM.ipynb` | `(maxToAdd, 28, 28, 1)` cho chuỗi frame | 1 đầu ra cho mỗi chuỗi (hồi quy) | `mean_absolute_error` | SGD (`lr` lấy từ biến trong notebook) | Dùng `TimeDistributed` bọc CNN trước LSTM; `maxToAdd` được cấu hình trong notebook. |
| 04 Training Utilities | `04_training_utilities/2_Human_Gender_CNN.ipynb` | `(64, 64, 1)` | 2 (male/female) | `categorical_crossentropy` | Adam (`lr` lấy từ biến trong notebook) | Bản có `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, `TensorBoard`; cũng có `np.rot90`. |
| 04 Training Utilities | `04_training_utilities/3_Emotion_Reg_CNN.ipynb` | `(48, 48, 1)` | 7 cảm xúc | `categorical_crossentropy` | Adam (`lr` lấy từ biến trong notebook) | Bản có `EarlyStopping` và `ImageDataGenerator`. |
| 05 Others | `05_others/1_segnet.ipynb` | `(96, 96, 1)` | 7 lớp segmentation cho từng pixel | `categorical_crossentropy` | Adam (`lr` lấy từ biến trong notebook) | Dùng `UpSampling2D`; dữ liệu nhãn được đổi sang one-hot/pixel-class. |
| 05 Others | `05_others/2_obj_detection.ipynb` | `(64, 64, 3)` | Bản đồ điểm/heatmap phát hiện xe, không phải vector lớp softmax | `mse` | Adam (`'adam'`) | Chuẩn hóa `x / 127.5 - 1`; notebook gọi đây là vehicle detection. |

## Tài nguyên hình minh họa

- `04_training_utilities/checkpoints.PNG`
- `04_training_utilities/tfboard.PNG`

## Tài liệu và tài nguyên nguồn khác

Các ví dụ hướng dẫn chung, tệp C++, cấu hình editor và dữ liệu ZIP/ảnh minh họa không thuộc 10 notebook baseline nên không được nhân bản vào thư mục này. Chúng vẫn được giữ nguyên tại `docs/project_charter/examples/`.