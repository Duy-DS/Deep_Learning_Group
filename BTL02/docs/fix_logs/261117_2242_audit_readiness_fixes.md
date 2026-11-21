# 🐛 Báo Cáo Sửa Lỗi: Hoàn thiện các điểm nghẽn Audit & Readiness BTL02

## 1. 🚨 Mô Tả Hiện Tượng (The Issue)
* **Triệu chứng lỗi:** Manifest dependency rỗng; CSV MLflow không giới hạn schema; cờ deterministic chỉ bật khi CUDA khả dụng; CNN chưa cấu hình được activation/channel; Trainer chưa áp dụng scheduler và hardcode một số tham số; pipeline RGB chưa có chuẩn ImageNet.
* **Nguyên nhân cốt lõi:** Các deliverable ban đầu ưu tiên chạy baseline trên môi trường Colab, chưa hoàn thiện các cổng chất lượng và khả năng ablation theo `AGENTS.md`.
* **Mức độ nghiêm trọng:** Cao — ảnh hưởng tính tái lập, độ tin cậy của dữ liệu tổng hợp và khả năng chạy các thí nghiệm tiếp theo.

## 2. 🔧 Phương Án Sửa Lỗi (Resolution)
* **Vùng mã nguồn xử lý:** `BTL02/requirements.txt`, `BTL02/.gitignore`, `BTL02/src/train/utils.py`, `BTL02/src/train/mlflow_logger.py`, `BTL02/src/models/custom_cnn.py`, `BTL02/src/train/trainer.py`, `BTL02/src/data/dataset.py`, `BTL02/src/data/transforms.py`, YAML trong `BTL02/experiments/baseline/` và `BTL02/tests/`.
* **Mã code trước và sau khi sửa (Diff summary):**
  - *Trước:* Thiếu danh sách dependency; không có quy tắc CSV tổng quát; deterministic cuDNN phụ thuộc CUDA; CSV MLflow xuất mọi cột; activation ReLU và channel widths cố định; scheduler không được khởi tạo/cập nhật; `num_workers` và `num_classes` hardcode; RGB chỉ dùng thống kê FER2013.
  - *Sau:* Khóa phiên bản dependency và thêm ignore CSV có ngoại lệ docs; đặt cờ deterministic vô điều kiện; ép CSV về schema cố định 24 cột; thêm ActivationFactory và cấu hình activation/channels/dropout; tích hợp `ReduceLROnPlateau`, `CosineAnnealingLR`, `None`; lấy worker/class count từ config; bổ sung ImageNet normalization tùy chọn cho RGB; thêm unit tests tương ứng.
  - *Bảo toàn tương thích:* Giữ thứ tự/tên layer mặc định của các CNN; số tham số mặc định được kiểm chứng không đổi.

## 3. 🧪 Xác Minh Sau Khi Sửa (Verification)
* **Kết quả kiểm thử:** Kiểm tra cú pháp thành công cho 17 file Python; ba YAML được parse thành công; unit tests chạy với stub tiến trình cho `torchvision`/`mlflow` (không ghi artifact) đạt `12 passed, 6 skipped`. Sáu test Dataset cần dữ liệu thật được skip vì không có FER2013 CSV. Đã xác minh forward pass của ba model, bốn activation, schema CSV và scheduler. Môi trường hiện tại không có `torchvision`/`mlflow`; không cài thêm dependency.
* **Phạm vi an toàn:** Không chạm hoặc stage `course_project/`, CSV dữ liệu, checkpoint hay `mlruns/`. Không chạy huấn luyện và không tạo checkpoint/CSV trong repo.