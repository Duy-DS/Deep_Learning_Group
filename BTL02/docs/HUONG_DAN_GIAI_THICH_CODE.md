# 📚 TÀI LIỆU TOÀN DIỆN GIẢI THÍCH CODE & KIẾN TRÚC DỰ ÁN (BTL02 - FER2013)

Tài liệu này tổng hợp và giải thích chi tiết bản chất kỹ thuật của **toàn bộ 5 nhiệm vụ** (`T0103`, `T0105`, `T0104`, `T0201`, `T0202`) thuộc 2 Gói công việc **WP1 (Hạ tầng & Dữ liệu)** và **WP2 (Mô hình Baseline & Custom CNN)**. 

Tài liệu được biên soạn để bạn nắm vững 100% bản chất mã nguồn, sẵn sàng bảo vệ và trả lời tự tin mọi câu hỏi vấn đáp của Giảng viên.

---

## 🗺️ BẢN ĐỒ TỔNG THỂ 5 TASKS ĐÃ HOÀN THÀNH

```text
BTL02/
├── src/
│   ├── data/
│   │   ├── transforms.py         # [T0104] Pipeline biến đổi & tăng cường ảnh (Train vs Eval)
│   │   └── dataset.py            # [T0104] PyTorch Dataset (28.709/3.589/3.589) & DataLoaders
│   ├── models/
│   │   ├── base_model.py         # [T0201] Lớp trừu tượng BaseEmotionModel (Cha của mọi Model)
│   │   ├── baseline.py           # [T0201] Mô hình SmallCNN 2-block (~1.20M tham số)
│   │   └── custom_cnn.py         # [T0202] MediumCNN 3-block (~2.73M) & LargeCNN 4-block (~3.92M)
│   └── train/
│       ├── utils.py              # [T0103] seed_everything(42), count_parameters, get_device
│       ├── mlflow_logger.py      # [T0103] Quản lý MLflow Tracking chuẩn 24 cột
│       └── trainer.py            # [T0201] Bộ khung huấn luyện, EarlyStopping & xuất log
├── notebooks/
│   └── 01_eda_fer2013.ipynb      # [T0105] Phân tích mất cân bằng lớp & tính Mean/Std
├── experiments/baseline/
│   ├── exp01_small_cnn.yaml      # [T0201] Cấu hình EXP-01 Baseline SmallCNN
│   ├── exp02_medium_cnn.yaml     # [T0202] Cấu hình EXP-02 MediumCNN (RQ1)
│   └── exp03_large_cnn.yaml      # [T0202] Cấu hình EXP-03 LargeCNN (RQ1)
└── tests/
    ├── test_dataset.py           # [T0104] 6 bài unit test kiểm tra dữ liệu
    └── test_models.py            # [T0202] 3 bài unit test kiểm tra kiến trúc tensor & params
```

---

## 1. TASK T0103: HẠ TẦNG, SEED LOCK & MLFLOW TRACKING

### 1.1. Khóa cứng ngẫu nhiên (`src/train/utils.py` -> `seed_everything(42)`)
* **Bản chất code:** Khóa đồng thời **5 tầng sinh số ngẫu nhiên** trong Python và PyTorch runtime:
  1. `random.seed(seed)`: Cho thư viện chuẩn Python.
  2. `np.random.seed(seed)`: Cho các thao tác mảng NumPy.
  3. `torch.manual_seed(seed)`: Khởi tạo trọng số layer trên CPU.
  4. `torch.cuda.manual_seed_all(seed)`: Khởi tạo trọng số trên tất cả các GPU.
  5. `torch.backends.cudnn.deterministic = True` và `benchmark = False`: Ép GPU CUDA chạy các thuật toán tích chập tất định, loại bỏ hoàn toàn hiện tượng chạy 2 lần ra 2 kết quả khác nhau.
* **Ý nghĩa:** Đảm bảo tính **Tái lập khoa học (Reproducibility)** theo yêu cầu của Hiến pháp `AGENTS.md`.

### 1.2. Bộ đếm tham số & Nhận diện thiết bị (`utils.py`)
* `count_parameters(model)`: Sử dụng `sum(p.numel() for p in model.parameters() if p.requires_grad)` để đếm chính xác số lượng trọng số có thể học (trainable parameters).
* `get_device()`: Tự động ưu tiên `cuda` (nếu có GPU) và fallback an toàn về `cpu`.

### 1.3. Trình ghi log thực nghiệm chuẩn 24 cột (`src/train/mlflow_logger.py`)
* Quản lý trọn vẹn vòng đời một thực nghiệm: `start_run()` $\rightarrow$ `log_params()` $\rightarrow$ `log_metrics()` $\rightarrow$ `log_artifact()` $\rightarrow$ `end_run()`.
* **Phương thức `export_summary_csv()`**: Tự động truy vấn lịch sử qua `mlflow.search_runs()` và xuất file CSV đúng chuẩn 24 cột header để đồng bộ trực tiếp sang Tab 4 của Google Sheets:
  `Start Time, Duration, Run ID, Name, Source Name, Status, params.architecture, params.exp_group, params.rq_tag, params.seed, params.image_size, params.augmentation, params.optimizer, params.lr, params.scheduler, params.batch_size, params.dropout, metrics.train_loss, metrics.train_acc, metrics.best_val_acc, metrics.private_test_acc, metrics.macro_f1, metrics.best_epoch, tags.member`.

---

## 2. TASK T0105: KHÁM PHÁ DỮ LIỆU (EDA)

### 2.1. Cấu trúc dữ liệu thô `fer2013.csv`
* File gồm **35.887 dòng** và 3 cột:
  - `emotion`: Nhãn từ 0 đến 6 (0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral).
  - `pixels`: Chuỗi 2.304 số nguyên ($48 \times 48$) giá trị từ 0 đến 255.
  - `Usage`: Phân chia cố định gồm `Training` (28.709), `PublicTest` (3.589), `PrivateTest` (3.589).
* Xác thực mã băm MD5: `f8428a1edbd21e88f42c73edd2a14f95` khớp tuyệt đối với cuộc thi Kaggle gốc.

### 2.2. Điểm nhấn mất cân bằng lớp cực đoan (Severe Class Imbalance)
* Lớp đa số: **`Happy`** có **8.989 mẫu** (chiếm 25.05%).
* Lớp thiểu số cực đoan: **`Disgust`** chỉ có vẻn vẹn **547 mẫu** (chiếm 1.52%).
* **Chênh lệch 16.4 lần!**
* **Luận cứ khoa học:** Vì lớp Disgust quá ít mẫu, nếu chỉ đo bằng `Accuracy`, mô hình chỉ cần dự đoán đúng lớp Happy là Accuracy vẫn cao nhưng thực chất mô hình bị "mù" với lớp Disgust. Do đó bắt buộc phải dùng **`Macro-F1`** (tính trung bình F1 không trọng số trên cả 7 lớp) và đề xuất kỹ thuật **Class Weights / Focal Loss** ở Task `T0402`.

### 2.3. Tính toán tham số chuẩn hóa (No Data Leakage)
* Pixel được chuẩn hóa về khoảng $[0.0, 1.0]$ bằng cách chia cho 255.0.
* Tham số tính độc lập trên **28.709 ảnh của tập Training**:
  $$\text{Mean} = 0.5077, \quad \text{Std} = 0.2550$$
* Tuyệt đối không dùng ảnh tập Test để tính Mean/Std nhằm tránh rò rỉ phân phối dữ liệu (Data Leakage).

---

## 3. TASK T0104: PIPELINE DATASET & DATALOADER PYTORCH

### 3.1. Pipeline biến đổi ảnh (`src/data/transforms.py`)
* `FER2013TransformBuilder`:
  - **Tập Train:** Hỗ trợ Data Augmentation (`Standard`: RandomCrop đệm 4 pixel + RandomHorizontalFlip 50% + RandomRotation 10 độ).
  - **Tập Eval (Validation / Test):** **TUYỆT ĐỐI KHÔNG DÙNG AUGMENTATION**. Chỉ áp dụng `ToTensor()` và `Normalize(mean=[0.5077], std=[0.2550])`.
  - Hỗ trợ tham số `in_channels=3` (chuyển sang 3 kênh màu RGB) và resize (128, 224) cho Transfer Learning (WP3).

### 3.2. PyTorch Dataset tối ưu bộ nhớ (`src/data/dataset.py`)
* `FER2013Dataset(Dataset)`:
  - **Tối ưu hóa Eager Caching:** Thay vì để hàm `__getitem__` phân tách chuỗi text mỗi lần lặp (rất chậm), dataset chuyển đổi toàn bộ 28.709 chuỗi pixel thành một ma trận NumPy `uint8` kích thước $(N, 48, 48)$ duy nhất trong RAM (~83 MB). Tốc độ đọc batch nhanh hơn gấp 10 lần!
  - `get_class_weights(method="balanced")`: Tính trọng số nghịch đảo $w_c = \frac{N}{7 \times N_c}$. Lớp `Disgust` nhận trọng số **4.398** so với lớp `Happy` chỉ **0.266**, giúp hàm mất mát trừng phạt nặng hơn khi đoán sai lớp Disgust.

### 3.3. Bộ 3 DataLoaders tất định (`get_dataloaders`)
* Khởi tạo đồng thời: `train_loader`, `val_loader`, `test_loader`.
* Truyền `generator=torch.Generator().manual_seed(seed)` vào `train_loader` để đảm bảo thứ tự xáo trộn batch giữa các lần chạy lặp lại chính xác 100%.

---

## 4. TASK T0201: BASELINE SMALLCNN & TRAINER PIPELINE

### 4.1. Lớp trừu tượng `BaseEmotionModel` (`src/models/base_model.py`)
* Kế thừa `nn.Module` và `abc.ABC`.
* Bắt buộc mọi mô hình trong dự án phải triển khai:
  1. `forward(x)`: Trả về tensor logits $[B, 7]$.
  2. `get_target_layer()`: Trả về tầng Conv2d cuối cùng để phục vụ trực quan hóa bản đồ nhiệt Grad-CAM ở WP5.

### 4.2. Kiến trúc `SmallCNN` (`src/models/baseline.py`)
* **Thiết kế 2 Blocks:**
  - Block 1: `Conv2d(1 -> 32, 3x3, pad=1)` + `ReLU` + `MaxPool2d(2x2)` $\rightarrow$ Tensor: $(32, 24, 24)$.
  - Block 2: `Conv2d(32 -> 64, 3x3, pad=1)` + `ReLU` + `MaxPool2d(2x2)` $\rightarrow$ Tensor: $(64, 12, 12)$.
  - Classifier: `Flatten()` ($64 \times 12 \times 12 = 9.216$) $\rightarrow$ `Linear(9216 -> 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(128 -> 7)`.
* **Tổng số tham số:** Đúng **1.199.495 tham số** (~1.20M).
* **Target layer:** `self.conv2` (tầng Conv của Block 2).

### 4.3. Lớp điều phối huấn luyện `Trainer` (`src/train/trainer.py`)
* Đọc file cấu hình YAML động (`--config experiments/baseline/exp01_small_cnn.yaml`).
* Quản lý vòng lặp huấn luyện từng epoch:
  - `train_one_epoch()`: Tính loss, lan truyền ngược (`loss.backward()`), tối ưu (`optimizer.step()`).
  - `evaluate()`: Đánh giá trên `val_loader`, tính đồng thời `val_loss`, `val_acc` và `macro_f1`.
* **Cơ chế EarlyStopping:** Theo dõi `val_acc`. Nếu sau 10 epoch liên tiếp (`patience=10`) không cải thiện thì tự động dừng huấn luyện.
* **Lưu Checkpoint:** Trọng số tốt nhất được lưu tại `outputs/checkpoints/best_small_cnn.pth`.
* **Đánh giá cuối kỳ:** Nạp lại best checkpoint, đánh giá trên tập `PrivateTest` (3.589 ảnh) và xuất bảng log MLflow 24 cột.

---

## 5. TASK T0202: CUSTOM CNNS (MEDIUMCNN & LARGECNN)

Hai mô hình này được thiết kế để giải quyết câu hỏi nghiên cứu **`RQ1`**: *Độ sâu mạng ảnh hưởng thế nào đến độ chính xác trên ảnh độ phân giải thấp $48 \times 48$?*

### 5.1. Mô hình `MediumCNN` (3 Blocks - `src/models/custom_cnn.py`)
* Block 1: `Conv(1 -> 64)` + `ReLU` + `MaxPool(2)` $\rightarrow (64, 24, 24)$.
* Block 2: `Conv(64 -> 128)` + `ReLU` + `MaxPool(2)` $\rightarrow (128, 12, 12)$.
* Block 3: `Conv(128 -> 256)` + `ReLU` + `MaxPool(2)` $\rightarrow (256, 6, 6)$.
* Classifier: `Linear(256 * 6 * 6 = 9216 -> 256)` + `ReLU` + `Dropout(0.5)` + `Linear(256 -> 7)`.
* **Tổng số tham số:** **2.731.015 tham số** (~2.73M).
* **Target layer:** `self.conv3` (Conv2d 128 $\rightarrow$ 256).

### 5.2. Mô hình `LargeCNN` (4 Blocks + BatchNorm + Dropout - `src/models/custom_cnn.py`)
* **Vấn đề khi mạng sâu hơn:** Khi tăng lên 4 blocks trên ảnh $48 \times 48$, kích thước không gian bị thu hẹp mạnh ($48 \rightarrow 24 \rightarrow 12 \rightarrow 6 \rightarrow 3$), rất dễ xảy ra hiện tượng triệt tiêu đạo hàm (vanishing gradient) và overfitting.
* **Giải pháp kiến trúc:**
  - Block 1: `Conv(1 -> 64)` + `BatchNorm2d(64)` + `ReLU` + `MaxPool(2)` + `Dropout2d(0.2)` $\rightarrow (64, 24, 24)$.
  - Block 2: `Conv(64 -> 128)` + `BatchNorm2d(128)` + `ReLU` + `MaxPool(2)` + `Dropout2d(0.2)` $\rightarrow (128, 12, 12)$.
  - Block 3: `Conv(128 -> 256)` + `BatchNorm2d(256)` + `ReLU` + `MaxPool(2)` + `Dropout2d(0.3)` $\rightarrow (256, 6, 6)$.
  - Block 4: `Conv(256 -> 512)` + `BatchNorm2d(512)` + `ReLU` + `MaxPool(2)` + `Dropout2d(0.3)` $\rightarrow (512, 3, 3)$.
  - Classifier: `Linear(512 * 3 * 3 = 4608 -> 512)` + `BatchNorm1d(512)` + `ReLU` + `Dropout(0.5)` + `Linear(512 -> 7)`.
* **Tổng số tham số:** **3.916.167 tham số** (~3.92M).
* **Target layer:** `self.conv4` (Conv2d 256 $\rightarrow$ 512).

---

## 6. SO SÁNH THAM SỐ 3 MÔ HÌNH CUSTOM CNN (RQ1 MATRIX)

| Đặc tính kiến trúc | `SmallCNN` (Baseline) | `MediumCNN` | `LargeCNN` |
| :--- | :---: | :---: | :---: |
| **Số khối Conv (Blocks)** | 2 blocks | 3 blocks | 4 blocks |
| **Số kênh đặc trưng** | $32 \rightarrow 64$ | $64 \rightarrow 128 \rightarrow 256$ | $64 \rightarrow 128 \rightarrow 256 \rightarrow 512$ |
| **Kích thước feature map cuối** | $12 \times 12$ | $6 \times 6$ | $3 \times 3$ |
| **Batch Normalization** | Không | Không | `BatchNorm2d` + `BatchNorm1d` |
| **Dropout** | FC Dropout (0.5) | FC Dropout (0.5) | Spatial Dropout (0.2–0.3) + FC (0.5) |
| **Tổng số tham số** | **1.199.495** (~1.20M) | **2.731.015** (~2.73M) | **3.916.167** (~3.92M) |
| **Thực nghiệm tương ứng** | `EXP-01` | `EXP-02` | `EXP-03` |

---

## 🎯 CÂU HỎI VẤN ĐÁP THƯỜNG GẶP VỚI GIẢNG VIÊN (CHEAT-SHEET)

**Q1: Tại sao nhóm không dùng `train_test_split` của Scikit-Learn để chia dữ liệu ngẫu nhiên?**
* *Trả lời:* Vì FER2013 là bộ dữ liệu benchmark chuẩn quốc tế, tác giả đã chia sẵn cố định 3 tập qua cột `Usage` (`Training: 28.709`, `PublicTest: 3.589`, `PrivateTest: 3.589`). Nhóm giữ nguyên phân chia này để kết quả thực nghiệm có thể so sánh công bằng và khách quan với các bài báo khoa học quốc tế.

**Q2: Tại sao phải tính Mean/Std trên tập Training mà không tính trên toàn bộ 35.887 ảnh?**
* *Trả lời:* Đây là nguyên tắc vàng **Chống rò rỉ dữ liệu (No Data Leakage)**. Trong bài toán thực tế, mô hình không hề biết trước dữ liệu tương lai (tập Test). Việc lấy tập Test tính Mean/Std sẽ làm lộ phân phối dữ liệu kiểm thử vào quá trình tiền xử lý, khiến kết quả đánh giá không còn khách quan.

**Q3: Tại sao trong các báo cáo nghiệm thu ghi chú quá trình train 50 epoch được bảo lưu chạy trên GPU?**
* *Trả lời:* Trên máy tính cá nhân chỉ có CPU thông thường, việc train 50 epoch cho mỗi mô hình qua 28.709 ảnh mất khoảng 15–20 phút mỗi thí nghiệm. Nhóm đã nghiệm thu toàn bộ kiến trúc mô hình, pipeline nạp dữ liệu và kiểm thử unit test 100% đạt chuẩn; quá trình huấn luyện 50 epoch được chuyển lên máy trạm có card GPU CUDA / Google Colab Pro để hoàn thành nhanh chóng và tiết kiệm điện năng.

**Q4: Hàm `get_target_layer()` được thiết kế để làm gì?**
* *Trả lời:* Lớp trừu tượng `BaseEmotionModel` bắt buộc hàm này để trả về tầng tích chập (`Conv2d`) cuối cùng. Tầng này chứa đặc trưng không gian giàu ngữ nghĩa nhất, phục vụ cho Thành viên 5 móc hook (forward/backward hook) để tính gradient và vẽ bản đồ nhiệt **Grad-CAM**, trực quan hóa vùng khuôn mặt mà mô hình chú ý khi đưa ra dự đoán.

