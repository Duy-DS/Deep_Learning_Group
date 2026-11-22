# Hướng dẫn giải thích code — Task T0102

## 1. Luồng dữ liệu

`scripts/download_data.py` quản lý việc tải MNIST IDX, kiểm kê file WIKI/FER, tải tài nguyên chỉ khi cấu hình URL/Google Drive ID thật và sinh dữ liệu giả deterministic cho smoke-test. Dữ liệu lớn ở `data/raw/` và cache trong `data/processed/` bị `.gitignore` bảo vệ.

`src/datasets/vision_datasets.py` chuẩn hóa dữ liệu ảnh và nhãn:

1. MNIST nhận `NHW` 28×28, chuyển sang `NHWC` một kênh và chia pixel byte cho 255 như notebook gốc.
2. WIKI nhận `NHW`/`NHWC`; mặc định xoay ảnh NHWC theo đúng `np.rot90(x, k=3, axes=(1,2))`, rồi mới đổi trục.
3. FER nhận mảng hoặc CSV legacy gồm 2304 pixel và 7 nhãn one-hot; kiểm tra schema trước khi chuyển ảnh sang tensor.
4. Ảnh đầu ra có layout `(N,C,H,W)` và dtype `torch.float32`; nhãn phân loại có dtype `torch.long` để dùng được với loss PyTorch nhận class index.

## 2. Phép xoay WIKI và đổi layout

Với ảnh vuông kích thước `H=W=64`, `np.rot90(A, k=3)` xoay 270° ngược chiều kim đồng hồ, tương đương 90° theo chiều kim đồng hồ. Tọa độ pixel `(y,x)` chuyển thành `(x, W-1-y)`. Vì `np.rot90` có thể tạo view không liên tục/readonly, Dataset tạo bản sao liên tục trước khi tạo Tensor.

Sau khi xoay, phép đổi layout là:

```text
(N, 64, 64, 1) --xoay trên axes=(1,2)--> (N, 64, 64, 1)
             --permute(0,3,1,2)--------> (N, 1, 64, 64)
```

Phép xoay làm đổi tọa độ không gian; phép `permute` chỉ đổi thứ tự trục. Hai thao tác không thể thay thế cho nhau. Khi dữ liệu đầu vào đã được xoay ở upstream, có thể đặt `apply_legacy_rotation=False` để không xoay hai lần.

## 3. Chuẩn hóa pixel và nhãn

- MNIST notebook gốc gọi `x /= 255`; Dataset giữ cùng quy ước `[0,1]`.
- WIKI `normalization="auto"`: giữ nguyên nếu max ≤ 1; chia 255 nếu dữ liệu pixel byte có max > 1.
- FER notebook được kiểm tra không có phép `/255` tường minh. Vì CSV thực tế chưa được cung cấp, loader `auto` bảo toàn giá trị unit-range và chuẩn hóa dữ liệu byte nếu cần; khi parity với CSV cụ thể, cần ghi nhận range nguồn trước khi kết luận.
- Nhãn one-hot `(N,K)` được đổi về class index bằng `argmax(axis=1)`; nhãn trả ra `torch.long`, nằm trong `[0,K-1]`.

## 4. Câu hỏi thường gặp

**Vì sao `Conv2d` cần NCHW?** PyTorch convolution nhận batch, channels, height, width. Tensor Keras NHWC cần `permute(0,3,1,2)` trước khi đưa vào model.

**Tại sao không hardcode Google Drive URL?** Workspace không có Drive ID/URL thật. Cấu hình để `null` nhằm tránh bịa nguồn hoặc tải nhầm dữ liệu; chỉ điền giá trị đã được nhóm xác nhận.

**Dummy data có dùng để huấn luyện/báo cáo accuracy không?** Không. Chỉ dùng xác minh shape, dtype, range, phép xoay và wiring của DataLoader.

**Ảnh PNG smoke có chứng minh dữ liệu WIKI thật đúng chiều không?** Không. PNG được tạo từ ảnh giả để kiểm tra phép biến đổi. Cần xem ảnh nguồn WIKI thật sau khi nhóm cung cấp dữ liệu.

**Vì sao CSV FER không tự coi mọi giá trị là byte?** Notebook gốc không khai báo chuẩn hóa. `auto` phân biệt unit-range và byte-range để tránh chia 255 hai lần nếu nguồn đã được chuẩn hóa.

---

## 5. Phụ lục T0103 — kiến trúc và baseline Keras gốc

### Phạm vi và provenance

Mười notebook trong `legacy_keras/` được giữ nguyên. `scripts/extract_keras_baselines.py` chỉ đọc JSON, source cells và execution outputs đã lưu; không import TensorFlow, không chạy notebook. Bảng chuẩn hóa nằm ở `experiments/csv_logs/keras_baseline_10_models.csv`; `NOT_RECORDED` nghĩa là notebook không lưu bằng chứng tương ứng. Số params dưới đây là output `model.summary()`, ngoại trừ Func_apxm_MLP chưa có summary (chỉ có công thức giải tích theo B).

### Mười kiến trúc theo nhóm

| EXP / mô hình | Shape đầu vào Keras → đầu ra | Tham số legacy | Luồng kiến trúc và chuyển đổi trọng tâm |
|---|---|---:|---|
| EXP-01 `Func_apxm_MLP` | `(2,) → (1,)` | Summary không lưu; Dense(2→B)+Dense(B→1) có `4B+1` params (65 khi B=16) | Xấp xỉ hàm Gaussian, sigmoid ở hidden/output, MSE + SGD. Notebook sweep A/B/C nên không tồn tại một B duy nhất cho mọi run; output metric số không lưu. |
| EXP-02 `MNIST_MLP` | `(28,28) → (10,)` | 669,706 (trainable 669,706) | Flatten 784 → Dense 512 → Dropout → Dense 512 → Dropout → softmax 10; SGD 0.01, categorical cross-entropy; ảnh chia 255. PyTorch cần thêm channel khi chuyển sang Dataset CNN dùng chung. |
| EXP-03 `1_MNIST_CNN` | `(28,28,1) → (10,)` | 1,817,610 | Conv 64/128/256 + MaxPool → Flatten → hai Dense 512 + Dropout → softmax 10; SGD 0.01. Soát padding/pooling và kích thước flatten trước khi đối chiếu params. |
| EXP-04 `2_Human_Gender_CNN` basic | `(64,64,1) → (2,)` | 4,063,570 | 5 block CNN tăng channels 16→256, Flatten, 2 Dense 1024; xoay `np.rot90(k=3, axes=(1,2))` trước model. Adam 1e-4; không callbacks. |
| EXP-05 `3_Emotion_Reg_CNN` basic | `(48,48,1) → (7,)` | 19,028,287 | CNN valid/same conv, Dense 1024, 7-way softmax; Adam 1e-4. CSV 2304 pixel + 7 one-hot; public-test file cũng dùng trong validation/evaluate. |
| EXP-06 `MNIST_CNN_LSTM` | `(8,28,28,1) → (1,)` | 1,747,073 | TimeDistributed Conv/Pool/Dropout/Flatten → LSTM 128 hai tầng → Dense 1; MAE + SGD 0.01. PyTorch thay wrapper bằng reshape batch×time qua CNN rồi dựng `(B,T,F)` cho LSTM. |
| EXP-07 `2_Human_Gender_CNN` callbacks | `(64,64,1) → (2,)` | 4,063,570 | Cùng CNN Gender; thêm ModelCheckpoint, EarlyStopping patience 5, ReduceLROnPlateau patience 5, TensorBoard. Max epoch 100000 nhưng execution dừng epoch 32. |
| EXP-08 `3_Emotion_Reg_CNN` callbacks | `(48,48,1) → (7,)` | 19,028,287 | Cùng CNN Emotion; ImageDataGenerator + EarlyStopping patience 5. Max 100, execution dừng epoch 57. |
| EXP-09 `1_segnet` | `(96,96,1) → (9216,7)` | 29,458,211; trainable 29,442,325; non-trainable 15,886 | Encoder–decoder convolution/BatchNorm, 5 lần `UpSampling2D`, softmax theo 7 lớp pixel. Keras summary output là sequence pixel-class; PyTorch thường biểu diễn logits NCHW và cần khớp target/loss trước parity. |
| EXP-10 `2_obj_detection` | `(64,64,3) → (1,)` sau pooling/Flatten | 83,367 | Chuẩn hóa Lambda `x/127.5-1`, Conv 3×3, MaxPool 8×8, Conv 8×8 và 1×1 `tanh`, MSE. Notebook gọi `accuracy`, nhưng đây không phải IoU/mAP; PyTorch port cần định nghĩa metric detection phù hợp riêng. |

### Cách đọc kết quả và các giới hạn

- Train/Val metrics trong CSV là số cuối cùng trên progress-bar output đã lưu. Thời gian là chuỗi notebook in rồi đổi về giây; phần cứng và phiên bản thư viện không có nên không so sánh CPU với T4.
- Ô `notebook_printed_eval_*` phản ánh chính xác lệnh evaluate trong notebook, không đảm bảo là test holdout độc lập. Đặc biệt WIKI basic, FER basic và hai callback notebooks đánh giá trên split validation/public-test đã dùng trong `validation_data`.
- CNN-LSTM chỉ có train/validation loss trong output; không có evaluate/test MAE đã lưu. Không suy diễn test score từ validation loss.
- SegNet metric được giữ tên legacy là `accuracy`; không tự coi là mean IoU. ObjDet accuracy cũng không phải mAP/IoU.
- Colab helper `experiments/colab/run_keras_baseline_t4.py` chỉ xác nhận thiết bị/hướng dẫn lưu bản sao. Muốn có baseline mới, chạy notebook copy trong runtime Colab T4, lưu log đầy đủ và không ghi đè file legacy.

### Checklist chuyển đổi sang PyTorch

- [ ] Input layout NHWC→NCHW; bảo toàn thứ tự kênh và normalization.
- [ ] WIKI xoay `k=3`, `axes=(1,2)` trước `permute`; xác minh ảnh thật và nhãn.
- [ ] Dense/Conv padding, pooling, Dropout mode, BatchNorm epsilon/momentum và initializer khớp Keras.
- [ ] Keras softmax + categorical cross-entropy được đối chiếu với logits PyTorch + `CrossEntropyLoss` (không softmax trước loss).
- [ ] CNN-LSTM gộp/tách Batch×Time đúng thứ tự sequence; đối chiếu output target scalar/MAE.
- [ ] SegNet khớp 7-class pixel target, thứ tự trục, upsampling và class-index/one-hot loss contract.
- [ ] ObjDet không dùng `accuracy` legacy làm IoU/mAP; xác định metric và cách tạo score-map rõ trước khi so sánh.
- [ ] Params parity chỉ chốt sau khi tái tạo summary bằng code PyTorch; không lấy giá trị suy luận làm kết quả kiểm chứng.