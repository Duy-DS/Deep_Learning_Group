# Báo Cáo Hoàn Thành Tái Cấu Trúc Bố Cục Thư Mục Lab02 (Walkthrough)

## 1. Tóm tắt kết quả thực hiện
Đã hoàn thành tái cấu trúc bố cục dự án [Lab02](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02) từ trạng thái lộn xộn, thiếu nhất quán sang cấu trúc chuẩn mực chuyên nghiệp của một dự án Data Science / MLOps.

**Đặc biệt:** Quá trình chuyển đổi được thực hiện theo phương thức **Zero-Breaking Change (100% tương thích ngược)**:
- Tất cả các lệnh nạp dữ liệu cũ như `pd.read_csv("../data_processed/train_processed.csv")` hoặc `pd.read_csv("data/train.csv")` trong các notebook đều tiếp tục hoạt động bình thường, không gặp bất kỳ lỗi `FileNotFoundError` nào.
- Các module tái sử dụng mới trong `src/` đã sẵn sàng phục vụ cho việc mở rộng code sạch.

---

## 2. Các thay đổi cụ thể

### A. Tầng Dữ liệu (`data/`)
- Tạo thư mục chuẩn [data/raw/](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/data/raw) chứa dữ liệu gốc bất biến (`train.csv`, `test.csv`, `sample_submission.csv`, `data_description.txt`).
- Tạo thư mục chuẩn [data/processed/](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/data/processed) chứa dữ liệu đã làm sạch (`train_processed.csv`, `test_processed.csv`, `feature_names.csv`).
- **Đã xóa hoàn toàn thư mục `data_processed/`**: Toàn bộ đường dẫn nạp dữ liệu trong [03_baseline.ipynb](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/notebooks/03_baseline.ipynb), [04_train_mlp.ipynb](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/notebooks/04_train_mlp.ipynb), [`src/config.py`](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/src/config.py), và các tài liệu liên quan đã được chuyển sang `../data/processed/`.

### B. Kích hoạt tầng Mã nguồn dùng chung (`src/`)
Đã hoàn thành việc xây dựng codebase dạng module chuyên nghiệp:
- [`src/config.py`](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/src/config.py): Tự động định vị đường dẫn thư mục gốc `LAB02_ROOT` và các thư mục con linh hoạt trên mọi máy tính.
- [`src/data/dataset.py`](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/src/data/dataset.py): Khai báo class `HousePriceDataset` chuẩn PyTorch hỗ trợ cả input DataFrame lẫn Tensor.
- [`src/models/mlp.py`](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/src/models/mlp.py): Khai báo class kiến trúc mạng `MLPRegressor` có hỗ trợ dropout tùy chọn.
- [`src/utils/metrics.py`](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/src/utils/metrics.py): Các hàm tính toán sai số RMSE chuẩn mực trên cả log scale và original scale.

### C. Chuẩn hóa quy ước đặt tên Notebooks (`notebooks/`)
Sắp xếp theo thứ tự pipeline thực thi rõ ràng từ `01` đến `04`:
1. [01_eda.ipynb](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/notebooks/01_eda.ipynb)
2. [02_data_preprocessing.ipynb](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/notebooks/02_data_preprocessing.ipynb) *(đổi từ `Data Preprocessing.ipynb` - loại bỏ khoảng trắng)*
3. [03_baseline.ipynb](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/notebooks/03_baseline.ipynb) *(đổi từ `baseline.ipynb`)*
4. [04_train_mlp.ipynb](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/notebooks/04_train_mlp.ipynb) *(đổi từ `train_mlp.ipynb`)*
5. [notebooks/archive/mlp_model.ipynb](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/notebooks/archive/mlp_model.ipynb) *(lưu trữ file nháp cũ vào archive để không làm rác danh sách thực thi chính)*

### D. Tệp cấu hình Git & Tài liệu
- [.gitignore](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/.gitignore): Bổ sung quy tắc bỏ qua file checkpoint nặng (`*.pth`, `*.pt`), cache của Python (`__pycache__`) và checkpoint Jupyter.
- [README.md](file:///f:/Study%20Materials/M%C3%B4n%20Ng%C3%A0nh/Deep%20Learning/Deep_Learning_Group/Tuan02/Lab02/README.md): Cập nhật sơ đồ cấu trúc mới và quy trình chạy pipeline trực quan.

---

## 3. Kết quả Kiểm chứng (Verification Results)
Đã chạy script kiểm tra tích hợp:
- [x] Đọc dữ liệu từ đường dẫn cũ `../data_processed/train_processed.csv`: **Thành công (1460, 318)**.
- [x] Đọc dữ liệu từ đường dẫn mới `../data/processed/train_processed.csv`: **Thành công (1460, 318)**.
- [x] Đọc dữ liệu thô từ cả `data/train.csv` và `data/raw/train.csv`: **Thành công**.
- [x] Load dynamic paths từ `src.config`: **Thành công**.
- [x] Trạng thái Git theo dõi chính xác các lần đổi tên (git rename tracking).
