# Dữ liệu BTL03

Không commit dữ liệu thô/đã xử lý, khóa Google Drive hoặc tệp `.npy`/CSV dataset. `.gitignore` của dự án bỏ qua nội dung `data/raw/` và `data/processed/`, chỉ giữ `.gitkeep` cùng README hướng dẫn.

## Cấu trúc cục bộ

```text
data/raw/
├── mnist/                 # IDX gzip nguồn và các mảng .npy giải mã
├── wiki_gender/           # train_x/train_y/validation_x/validation_y .npy
└── fer2013/               # CSV: 2304 pixel + 7 nhãn one-hot
```

## Tải / kiểm tra

Từ thư mục `BTL03/`:

```powershell
python scripts/download_data.py download-mnist
python scripts/download_data.py check
```

MNIST được tải từ endpoint IDX công khai đã khai báo trong `scripts/download_data.py`, xác thực magic/header và giải mã thành `data/raw/mnist/{train_images,train_labels,test_images,test_labels}.npy`.

Tệp WIKI Gender và FER2013 dung lượng lớn được quản lý ngoài Git. Không có Google Drive URL/ID xác thực trong workspace hiện tại. Sau khi nhóm điền `download_urls` hoặc `google_drive_file_ids` vào `configs/wiki_gender.yaml` / `configs/fer_emotion.yaml`, có thể tải như sau:

```powershell
python scripts/download_data.py fetch-configured configs/wiki_gender.yaml
python scripts/download_data.py fetch-configured configs/fer_emotion.yaml
```

Google Drive ID cần thư viện tùy chọn `gdown`; chỉ cài khi thực sự dùng: `python -m pip install gdown`. Không điền ID giả hoặc chia sẻ token trong YAML. Có thể tải file thủ công rồi đặt đúng tên đã liệt kê trong cấu hình. `python scripts/download_data.py check` kiểm kê file.

## Dữ liệu giả lập kiểm thử

```powershell
python scripts/download_data.py dummy --samples 8 --seed 42
python -m scripts.smoke_test_datasets
```

Dummy data có schema đúng để kiểm tra loader, nhưng không đại diện phân phối hay chất lượng dữ liệu thật và không dùng huấn luyện/báo cáo kết quả mô hình.

## Hợp đồng dữ liệu

- **MNIST:** ảnh thô `uint8` NHW `28×28`, 10 lớp; legacy chia pixel cho 255. Dataset xuất `(N,1,28,28)` `float32` trong `[0,1]`, nhãn `long`.
- **WIKI Gender:** legacy lưu ảnh NHWC `(N,64,64,1)`, nhãn two-class one-hot; trước đổi trục, áp dụng `np.rot90(x, k=3, axes=(1,2))` đúng cả train và validation. Dataset sau đó xuất NCHW; pixel `[0,255]` được chia 255 còn `[0,1]` được giữ nguyên.
- **FER2013:** notebook đọc CSV có 2304 pixel trước 7 nhãn one-hot theo thứ tự `angry, disgust, fear, happy, neutral, sad, surprise`; không thấy phép chia 255 trong legacy notebook. Loader không chia dữ liệu đã trong `[0,1]`, nhưng tự chuẩn hóa dữ liệu byte `[0,255]` khi dùng `normalization="auto"`. Đầu ra ảnh `(N,1,48,48)`, nhãn `long`.

Dataset classes: `src/datasets/vision_datasets.py`. Không áp augmentation ở lớp validation/test; augmentation sẽ thuộc pipeline task sau.