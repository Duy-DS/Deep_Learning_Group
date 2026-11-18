# 🐛 Báo Cáo Sửa Lỗi: Audit Readiness và Downloader kagglehub

## 1. 🚨 Mô Tả Hiện Tượng (The Issue)
* **Triệu chứng lỗi:** Dependency trước đây chưa đầy đủ; cờ cuDNN phụ thuộc CUDA; CSV MLflow có số cột động; CNN chỉ dùng ReLU; scheduler chưa được áp dụng; downloader Kaggle CLI chưa hỗ trợ token mới; tests phụ thuộc dữ liệu thật và môi trường khác dự án.
* **Nguyên nhân cốt lõi:** Mã ban đầu tập trung baseline/Colab, thiếu hợp đồng xuất log, kiểm thử offline và cấu hình ablation dùng chung.
* **Mức độ nghiêm trọng:** Trung bình — chặn kiểm thử và chuyển giao kỹ thuật; không thay đổi kết quả thực nghiệm lịch sử.

## 2. 🔧 Phương Án Sửa Lỗi (Resolution)
* **Vùng mã nguồn xử lý:** `src/train/utils.py`, `src/train/mlflow_logger.py`, `src/train/trainer.py`, `src/models/baseline.py`, `src/models/custom_cnn.py`, `src/data/downloader.py`, `src/data/transforms.py`, `src/data/dataset.py`, `tests/`, `experiments/baseline/`, `requirements.txt`, `.gitignore`, tài liệu hướng dẫn.
* **Mã code trước và sau khi sửa (Diff summary):**
  - *Trước:* Schema MLflow tự do, ReLU/channel hardcode, scheduler chỉ được log nhưng không dùng; tải bằng Kaggle CLI và chỉ xét credentials cũ.
  - *Sau:* Schema cố định đúng 24 cột; seed CPU/CUDA và cuDNN nhất quán; activation factory bốn lựa chọn, channels/dropout tùy biến; cấu hình worker/class count/scheduler qua YAML; ImageNet normalization RGB tùy chọn; kagglehub 1.0.3 với `competition_download(..., force_download=force)`; nhận diện token môi trường → access_token → legacy; cache dùng chung chỉ đọc, candidate được kiểm chứng trước atomic replace, thư mục tạm tự dọn; từ chối CSV sai split/nhãn/pixel và archive mơ hồ; lỗi network được che nội dung có thể chứa secrets.

## 3. 🧪 Xác Minh Sau Khi Sửa (Verification)
* **Kết quả kiểm thử:** `BTL02/.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider BTL02/tests`: **29 passed in 252.40s**, không stub dependency, không skip; `pip check`: **No broken requirements found**. PyTorch 2.5.1, torchvision 0.20.1, pytest 8.3.3, MLflow 2.18.0, kagglehub 1.0.3. CSV/archive test chỉ được tạo trong tmp_path ngoài repo; Kaggle network được mock. Không tải FER2013 thật, không đọc token thật, không đụng course_project, data/checkpoint thật, không stage/commit/push. Kết quả này xác nhận các cổng kỹ thuật được kiểm thử, không chứng minh DoD nghiên cứu đã hoàn tất 100%.