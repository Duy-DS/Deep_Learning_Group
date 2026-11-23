# BÁO CÁO TỔNG KẾT BTL03 — [MỐC SNAPSHOT]

> Mẫu READ-ONLY. Sao chép sang `md_drafts/` trước khi điền. Chỉ xóa placeholder khi có nguồn xác nhận.

## Thông tin đề tài và nhóm

- Đề tài: Chuyển đổi mã nguồn Deep Learning từ Keras sang PyTorch và phân tích thực nghiệm
- Mã dự án: BTL03
- Snapshot/task/commit: [ĐIỀN]
- Ngày chốt bằng chứng: [YYYY-MM-DD]
- Trường/lớp/giảng viên: [CẦN XÁC NHẬN]
- Thành viên: [Alias đã xác nhận; họ tên và MSSV theo nguồn chính thức]
- Tỷ lệ đóng góp: [CẦN NHÓM XÁC NHẬN; tổng 100%]
- Người tổng hợp: [THÀNH VIÊN THẬT ĐƯỢC CHỈ ĐỊNH]

## Tóm tắt

[Bài toán, phạm vi snapshot, phương pháp, kết quả đã đo và hạn chế. Không điền metric thiếu nguồn.]

## 1. Tổng quan và phạm vi

### 1.1 Bối cảnh, mục tiêu, câu hỏi nghiên cứu
### 1.2 Năm nhóm và mười cấu hình/bài chuyển đổi
### 1.3 Phạm vi snapshot và tiêu chí hoàn thành
### 1.4 Hạng mục ngoài phạm vi/chưa hoàn tất

## 2. Cơ sở lý thuyết và khác biệt framework

### 2.1 Keras/PyTorch, vòng lặp huấn luyện, logits và loss
### 2.2 NHWC/NCHW; WIKI `np.rot90` trước `permute`
### 2.3 `TimeDistributed` và reshape CNN-LSTM
### 2.4 `UpSampling2D`, SegNet, masks và loss
### 2.5 FCN score-map, chuẩn hóa và metric
### 2.6 Chuẩn hóa dữ liệu, split và leakage

## 3. Thiết kế PyTorch OOP và các mô hình

### 3.1 `BaseMigratedModel`, parameter counters, feature hooks
### 3.2 `BaseTrainer`, seed, logger/checkpoint và callbacks
### 3.3 YAML, CLI, runners, smoke-test
### 3.4 Ma trận chuyển đổi
| Nhóm | Mô hình/cấu hình | Thành phần PyTorch | Bằng chứng và hạn chế |
|---|---|---|---|
| 1–5 | [Điền từ evidence] | | |

## 4. Kiểm toán T0212 và trạng thái thực nghiệm

### 4.1 Snapshot, phương pháp và nguồn
### 4.2 Kiểm toán PR/nhánh (chỉ ghi history đã xem)
### 4.3 Parameter parity
| Cấu hình | Keras total/trainable/non-trainable | PyTorch total/trainable/non-trainable | Sai lệch | Provenance/caveat |
|---|---:|---:|---:|---|
| | | | | |

> Nếu hai nguồn cùng trường mâu thuẫn, giữ cả hai giá trị/snapshot và ghi `CẦN ĐỐI CHIẾU`; không tuyên bố parity trước khi xác minh source và quy ước đếm.
### 4.4 Smoke và kiểm thử phần mềm; tách synthetic khỏi metric khoa học
### 4.5 Tình trạng task, WP03 và WP04
| Hạng mục | Trạng thái | Bằng chứng | Việc tiếp theo |
|---|---|---|---|
| | | | |
### 4.6 Full training (chỉ điền sau run thật)
| Mô hình/dữ liệu/split | Seed/môi trường/config | Metric/đơn vị | Run/log/checkpoint | Protocol/caveat |
|---|---|---|---|---|
| [CHƯA THỰC HIỆN - DỰ KIẾN Ở GIAI ĐOẠN TIẾP THEO] | | | | |

## 5. Đánh giá, hạn chế, bài học và hướng phát triển

### 5.1 Kết luận được phép từ evidence
### 5.2 Giới hạn, blocker, validity threats, tái lập
### 5.3 Full training, ablation, visualization, RQ
### 5.4 Demo, nghiệm thu, đóng gói
### 5.5 Kết luận đúng phạm vi snapshot

## Tài liệu tham khảo và phụ lục evidence

| Evidence ID | Tệp/commit/run | Phần báo cáo | Điều chứng minh | Không chứng minh |
|---|---|---|---|---|
| | | | | |