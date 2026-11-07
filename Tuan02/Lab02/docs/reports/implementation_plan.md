# Kế Hoạch Tái Cấu Trúc Bố Cục Thư Mục Lab02 (Non-Breaking Refactoring)

## Mục tiêu
Tái cấu trúc bố cục dự án `Tuan02/Lab02` từ trạng thái lộn xộn/phân mảnh thành một cấu trúc chuẩn mực chuyên nghiệp (Production-Grade Data Science & MLOps), đồng thời **đảm bảo 100% không làm gãy bất kỳ đường dẫn (paths) hay mã nguồn nào** trong các notebook và script hiện có.

## Các giải pháp kỹ thuật cốt lõi
1. **Kiến trúc dữ liệu tập trung & Cầu nối tương thích (Backward Compatibility Junction):**
   - Di chuyển dữ liệu đã xử lý vào `data/processed/`.
   - Tạo NTFS Directory Junction từ `data_processed` trỏ về `data/processed`, giúp code cũ gọi `../data_processed/...` và code mới gọi `../data/processed/...` đều phân giải thành công.
2. **Kích hoạt tầng mã nguồn tái sử dụng `src/`:**
   - Xây dựng `src/config.py` để định vị đường dẫn động linh hoạt.
   - Module hóa `HousePriceDataset` vào `src/data/dataset.py`.
   - Module hóa `MLPRegressor` vào `src/models/mlp.py`.
   - Viết các helper metrics và validation vào `src/utils/metrics.py`.
3. **Chuẩn hóa quy ước đặt tên Notebook:**
   - Đổi `Data Preprocessing.ipynb` thành `02_data_preprocessing.ipynb` (loại bỏ khoảng trắng nguy hiểm).
   - Đổi `baseline.ipynb` thành `03_baseline.ipynb`.
   - Đổi `train_mlp.ipynb` thành `04_train_mlp.ipynb`.
   - Đưa `mlp_model.ipynb` vào thư mục `notebooks/archive/` (hoặc giữ stub) để tránh rác.
4. **Bảo vệ kho mã nguồn với `.gitignore` chuyên nghiệp:**
   - Chặn các file weights/pickles lớn vô tình bị commit tiếp.
5. **Cập nhật tài liệu `README.md`:**
   - Mô tả cây thư mục mới và hướng dẫn quy trình chạy pipeline `01` -> `02` -> `03` -> `04`.

## Kế hoạch kiểm chứng (Verification Plan)
- Chạy test script Python từ thư mục `notebooks/` để kiểm tra việc đọc file từ cả đường dẫn cũ (`../data_processed/train_processed.csv`) lẫn đường dẫn mới.
- Kiểm tra việc import các module từ `src/` (`src.config`, `src.data.dataset`, `src.models.mlp`, `src.utils.metrics`).
- Đảm bảo toàn bộ test pass 100%.
