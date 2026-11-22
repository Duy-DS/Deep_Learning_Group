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