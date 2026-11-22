# 🐛 Báo Cáo Sửa Lỗi: MLflow FileStore thiếu thư mục `.trash` trên Colab Drive

## 1. 🚨 Mô Tả Hiện Tượng (The Issue)
* **Triệu chứng lỗi:** EXP-04 hoàn tất epoch 1 nhưng dừng khi ghi metric epoch tiếp theo với `Exception: Invalid parent directory '/content/drive/MyDrive/BTL02/T0304_RQ2/mlruns/.trash'`.
* **Nguyên nhân cốt lõi:** MLflow 2.18 FileStore duyệt thư mục experiments đã xóa trong mỗi lần `log_batch`. Thư mục `.trash` có thể bị thiếu hoặc biến mất trên Google Drive mount sau khi logger được khởi tạo, khiến lần ghi metric kế tiếp thất bại.
* **Mức độ nghiêm trọng:** Cao — tiến trình huấn luyện thoát giữa chừng và các cấu hình tiếp theo không chạy.

## 2. 🔧 Phương Án Sửa Lỗi (Resolution)
* **Vùng mã nguồn xử lý:** `src/train/mlflow_logger.py`, `notebooks/run_t0304_colab.ipynb`.
* **Mã code trước và sau khi sửa (Diff summary):**
  - *Trước:* Logger chỉ chọn tracking URI và khởi tạo experiment; không bảo đảm `<tracking-root>/.trash` còn tồn tại trước lần ghi metric.
  - *Sau:* Với URI local/file, logger tạo root/`.trash` khi khởi tạo và ngay trước mỗi lần `log_metrics`; notebook dùng MLflow và checkpoint trên SSD local, resume/sync với Drive và xuất CSV qua bản 24 cột chuẩn. URI tracking từ xa không bị sửa.
  - Thêm regression test xóa `.trash` sau khi mở run, sau đó xác minh logger tự tạo lại thư mục và metric được đọc lại từ FileStore.
  - Giữ launcher notebook truyền stdout/stderr của Trainer trực tiếp theo thời gian thực để traceback nội bộ hiển thị đầy đủ.

## 3. 🧪 Xác Minh Sau Khi Sửa (Verification)
* **Kết quả kiểm thử:** Full suite cuối `94 passed in 313.27s`; notebook nbformat hợp lệ, đúng 8 cells; kiểm tra sanitizer với `np.int64`, `np.float64`, `np.bool_`, ndarray và DataFrame thành công. Chưa thực thi huấn luyện đầy đủ trên runtime Colab.
