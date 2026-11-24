# 📚 TÀI LIỆU TOÀN DIỆN GIẢI THÍCH CODE & KIẾN TRÚC DỰ ÁN (BTL02 - FER2013)

## WP5 T0501: Metrics, Confusion Matrix và Classification Report

`src/evaluate/plot_metrics.py` đóng gói việc đọc epoch-history, suy luận và trực quan hóa trong `MetricsVisualizer`. Input `DataLoader` phải được tạo từ pipeline evaluation hiện hữu: ảnh FER2013 grayscale 48×48 qua transform deterministic; ảnh SOTA RGB/resize dùng eval transform ImageNet. Module không tạo augmentation và `infer()` gọi `model.eval()`/`torch.no_grad()`. Tensor logits `[B,7]` được argmax để lấy dự đoán; `sklearn.metrics` tính confusion matrix cố định thứ tự bảy lớp và precision/recall/F1/support, macro/weighted averages. Ma trận chuẩn hóa chia tổng mỗi hàng, hiển thị phần trăm; ma trận đếm giữ số nguyên. Hình lưu 300 dpi, report xuất JSON/CSV/Markdown.

Learning curves cần CSV **epoch-level** có epoch, train_loss, train_acc, lr (val_loss/val_acc tùy chọn). CSV tổng hợp MLflow 24 cột chỉ giữ metric cuối run, không phải history; module chủ động báo lỗi thay vì nội suy/giả lập đường cong. `--mock` tạo dữ liệu tổng hợp chỉ để thử luồng vẽ, không phải kết quả thí nghiệm FER2013.

**Hỏi: Vì sao không thể vẽ đường cong từ CSV 24 cột hiện có?** Trả lời: CSV T0403 có một dòng/run và `metrics.train_loss` là số tổng kết; không có chuỗi epoch, do đó không chứa trajectory Loss/Accuracy/LR.

**Hỏi: PublicTest có phải holdout độc lập không?** Trả lời: Không trong pipeline hiện tại; Trainer dùng PublicTest làm validation/model selection. Chỉ diễn giải như validation; PrivateTest là split đánh giá cuối.

---

## WP5 T0502: Conv Filters, Feature Maps và Grad-CAM (RQ4)

`src/evaluate/gradcam_vis.py` cung cấp `GradCAMVisualizer` làm việc với mọi `nn.Module` có `nn.Conv2d`, bao gồm CNN dự án và backbone. Nếu không chỉ định tầng, feature-map extraction chọn convolution đầu/cuối; Grad-CAM ưu tiên `get_target_layer()` của model rồi fallback Conv cuối. Forward hook lưu activation $A\in\mathbb{R}^{1\times K\times h\times w}$; backward hook lấy gradient. Trọng số kênh $\alpha_k=\frac{1}{hw}\sum_{ij}\frac{\partial y^c}{\partial A^k_{ij}}$; heatmap $L^c=\operatorname{ReLU}(\sum_k\alpha_kA^k)$ được nội suy bilinear về $H\times W$ và min-max chuẩn hóa về [0,1]. Hook được gỡ trong `finally`, model trở về train/eval mode ban đầu và gradient được xóa.

Filter visualization hiển thị kernel `Conv2d.weight`; feature-map hook chỉ giữ activation detach trên CPU. Overlay chuẩn hóa tensor đầu vào đã nhận từ evaluation pipeline để hiển thị; đây là minh họa chứ không phục hồi pixel ảnh gốc sau ImageNet normalization. RQ4 xem vùng kích hoạt là tín hiệu giải thích, không phải bằng chứng nhân quả hay khẳng định model luôn tập trung đúng mắt/miệng.

**Hỏi: Vì sao ResNet50 chọn convolution cuối của layer4?** Trả lời: Model cung cấp `get_target_layer()` để chọn feature map ngữ nghĩa sâu, nhưng vẫn giữ độ phân giải không gian để nội suy CAM. Có thể truyền tầng khác khi cần kiểm tra độ phân giải.

**Hỏi: Tại sao ảnh overlay được chuẩn hóa min-max?** Trả lời: Input evaluation có thể là grayscale [0,1] hoặc RGB chuẩn hóa ImageNet, nên chuẩn hóa chỉ nhằm hiển thị. Phân tích định lượng không dùng ảnh hiển thị này và input model không bị thay đổi.

---

## WP5 T0503: Phân tích True/Fail Cases (RQ4)

`src/evaluate/error_analysis.py` cung cấp `ErrorAnalyzer`. Với batch ảnh `[B,C,H,W]`, model ở `eval()` và `torch.no_grad()` trả logits `[B,7]`; softmax cho xác suất mỗi lớp, `topk(3)` lưu ba lớp/xác suất, argmax là nhãn dự đoán. Bảng mỗi mẫu lưu chỉ số, nhãn thật/dự đoán, confidence và top-3; nhóm đúng chia confidence cao/thấp theo ngưỡng cấu hình, sai giữ nguyên để phân tích. Cặp lỗi là confusion có hướng `true → predicted`, sắp theo số lượng giảm dần. Lưới hình dùng đúng tensor evaluation của DataLoader, không có augmentation mới.

Phần nguyên nhân trong Markdown là danh sách giả thuyết cần xác minh ảnh thật: che khuất, góc/ánh sáng, nhãn mơ hồ/nhiễu và mất cân bằng lớp thiểu số Disgust. Chỉ thống kê lớp không chứng minh nguyên nhân cụ thể; cần xem ảnh gốc/người gán nhãn và đối chiếu support/per-class recall. Kết quả synthetic unit test chỉ xác nhận logic phần mềm.

**Hỏi: Vì sao cặp lỗi có hướng?** Trả lời: Fear→Sad và Sad→Fear là lỗi khác nhau và có thể bất đối xứng; giữ nhãn thật và nhãn dự đoán riêng giúp chẩn đoán lớp nguồn đích.

**Hỏi: Top-3 xác suất là độ tin cậy đã hiệu chuẩn chưa?** Trả lời: Không. Đây là softmax score, hữu ích để xem thứ tự tương đối; không được diễn giải như xác suất calibrated nếu chưa chạy calibration đánh giá riêng.

**Hỏi: Có thể kết luận Disgust bị nhầm do mất cân bằng chỉ từ confusion matrix không?** Trả lời: Không; mất cân bằng là giả thuyết cần kiểm tra support, recall, thí nghiệm class-weight/focal và xem ảnh/nhãn.

---

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

## 6. SO SÁNH THAM SỐ & KẾT QUẢ THỰC NGHIỆM 3 MÔ HÌNH CUSTOM CNN (RQ1 MATRIX)

### 6.1. Bảng Đối Sánh Kiến Trúc & Kết Quả Thực Nghiệm Trên Google Colab (GPU T4)

| Đặc tính kiến trúc & Kết quả | `SmallCNN` (Baseline) | `MediumCNN` | `LargeCNN` (Đề xuất tối ưu) |
| :--- | :---: | :---: | :---: |
| **Mã thực nghiệm (Experiment ID)** | `EXP-01` | `EXP-02` | `EXP-03` |
| **Số khối Conv (Blocks)** | 2 blocks | 3 blocks | **4 blocks** |
| **Số kênh đặc trưng** | $32 \rightarrow 64$ | $64 \rightarrow 128 \rightarrow 256$ | $64 \rightarrow 128 \rightarrow 256 \rightarrow 512$ |
| **Kích thước feature map cuối** | $12 \times 12$ | $6 \times 6$ | $3 \times 3$ |
| **Batch Normalization** | Không | Không | **`BatchNorm2d` + `BatchNorm1d`** |
| **Dropout Regularization** | FC Dropout (0.5) | FC Dropout (0.5) | **Spatial Dropout (0.2–0.3) + FC (0.5)** |
| **Tổng số tham số** | **1.199.495** (~1.20M) | **2.731.015** (~2.73M) | **3.916.167** (~3.92M) |
| **Best Epoch (Điểm hội tụ tối ưu)** | **Epoch 8** | **Epoch 20** | **Epoch 32** |
| **Best Validation Accuracy** | **55.36%** | **59.60%** (+4.24%) | **63.83%** (+8.47%) |
| **Private Test Accuracy (Mù)** | **55.17%** | **58.09%** (+2.92%) | **64.95%** (**+9.78%**) |
| **Private Test Macro-F1** | **0.4936** | **0.5708** (+15.64%) | **0.6356** (**+28.77%**) |
| **Train Accuracy** | 71.28% | 91.95% | **92.40%** |
| **Val Loss tại Best Epoch** | 1.5215 | 2.8824 (Overfitting) | **1.6093 (Kiểm soát tốt)** |
| **Thời gian train (T4 GPU)** | 3m 13s (18 epochs) | 5m 55s (30 epochs) | 10m 19s (42 epochs) |
| **Trọng số lưu trữ (.pth)** | `best_small_cnn.pth` (4.8MB) | `best_medium_cnn.pth` (10.9MB) | `best_large_cnn.pth` (15.7MB) |
| **MLflow Run ID** | `49137a591d164169...` | `1f46b0e87c324a63...` | `5a71e5b71f034b21...` |

### 6.2. Kết Luận Khoa Học Cho Câu Hỏi Nghiên Cứu RQ1
1. **Mở rộng độ sâu giúp tăng hiệu năng đơn điệu tuyệt đối:** Tăng từ 2 lên 4 khối Conv giúp Test Acc tăng từ **$55.17\%$ lên $64.95\%$** ($+9.78\%$) và Macro-F1 tăng từ **$0.4936$ lên $0.6356$** ($+14.20\%$).
2. **Vai trò sống còn của BatchNorm:** `MediumCNN` dù có dung lượng học lớn (Train Acc $91.95\%$) nhưng do thiếu BatchNorm nên bị overfit nghiêm trọng (Val Loss bùng nổ lên $2.88$). Sang `LargeCNN`, việc tích hợp `BatchNorm2d` sau mỗi tầng tích chập đã triệt tiêu hiện tượng trôi dạt hiệp biến nội tại (internal covariate shift), giữ Val Loss ở mức $1.60$ và kéo dài đà học ổn định tới tận Epoch 32.

---

## 7. 🎯 CÂU HỎI VẤN ĐÁP THƯỜNG GẶP VỚI GIẢNG VIÊN (CHEAT-SHEET)

**Q1: Tại sao nhóm không dùng `train_test_split` của Scikit-Learn để chia dữ liệu ngẫu nhiên?**
* *Trả lời:* Vì FER2013 là bộ dữ liệu benchmark chuẩn quốc tế, tác giả đã chia sẵn cố định 3 tập qua cột `Usage` (`Training: 28.709`, `PublicTest: 3.589`, `PrivateTest: 3.589`). Nhóm giữ nguyên phân chia này để kết quả thực nghiệm có thể so sánh công bằng và khách quan với các bài báo khoa học quốc tế.

**Q2: Tại sao phải tính Mean/Std trên tập Training mà không tính trên toàn bộ 35.887 ảnh?**
* *Trả lời:* Đây là nguyên tắc vàng **Chống rò rỉ dữ liệu (No Data Leakage)**. Trong bài toán thực tế, mô hình không hề biết trước dữ liệu tương lai (tập Test). Việc lấy tập Test tính Mean/Std sẽ làm lộ phân phối dữ liệu kiểm thử vào quá trình tiền xử lý, khiến kết quả đánh giá không còn khách quan.

**Q3: Nhóm đã thực hiện quá trình huấn luyện thực nghiệm ở đâu và mất bao lâu?**
* *Trả lời:* Nhóm xây dựng notebook chuẩn hóa `notebooks/colab/02_colab_training.ipynb` và thực hiện huấn luyện tự động toàn bộ 3 mô hình trên phần cứng GPU NVIDIA Tesla T4 của Google Colab:
  - `SmallCNN`: 18 epochs, mất 3 phút 13 giây.
  - `MediumCNN`: 30 epochs, mất 5 phút 55 giây.
  - `LargeCNN`: 42 epochs, mất 10 phút 19 giây.
  Tổng cộng toàn bộ 3 thực nghiệm chỉ mất chưa đầy 20 phút trên GPU, mọi thông số metric và trọng số `.pth` đều được tự động lưu vết qua MLflow và đồng bộ vào file CSV 24 cột.

**Q4: Hàm `get_target_layer()` được thiết kế để làm gì?**
* *Trả lời:* Lớp trừu tượng `BaseEmotionModel` bắt buộc hàm này để trả về tầng tích chập (`Conv2d`) cuối cùng (`conv2` ở SmallCNN, `conv3` ở MediumCNN, `conv4` ở LargeCNN). Tầng này chứa đặc trưng không gian giàu ngữ nghĩa nhất, phục vụ cho Thành viên 5 móc hook (forward/backward hook) để tính gradient và vẽ bản đồ nhiệt **Grad-CAM**, trực quan hóa vùng khuôn mặt mà mô hình chú ý khi đưa ra dự đoán.

**Q5: Tại sao SmallCNN kích hoạt EarlyStopping rất sớm ở Epoch 8 trong khi LargeCNN học bền bỉ đến tận Epoch 32?**
* *Trả lời:* `SmallCNN` chỉ có 2 khối tích chập, receptive field chỉ đạt $10 \times 10$ pixels và thiếu tầng chuẩn hóa `BatchNorm2d`. Mô hình nhanh chóng khai thác hết các đặc trưng biên cạnh đơn giản và bị bão hòa. Ngược lại, `LargeCNN` có 4 khối tích chập kết hợp `BatchNorm2d`, giúp duy trì dòng gradient ổn định sâu xuống tầng dưới cùng và liên tục kết hợp các đặc trưng bậc thấp thành đặc trưng trừu tượng cao hơn (mắt, miệng, nếp nhăn) trong suốt 32 epochs.

**Q6: Tại sao ở mô hình LargeCNN, độ chính xác trên Private Test (64.95%) lại cao hơn Validation Acc (63.83%)?**
* *Trả lời:* Trong quá trình huấn luyện, mô hình được áp dụng cơ chế điều hòa rất mạnh: `Dropout2d (0.2–0.3)` ở các tầng Conv và `Dropout (0.5)` ở tầng Fully Connected. Khi bước sang pha đánh giá (`model.eval()`), toàn bộ Dropout được tắt đi, 100% neuron hoạt động đồng thời. Tập Private Test chứa nhiều khuôn mặt có biểu cảm rõ nét, giúp mô hình phát huy tối đa năng lực khái quát hóa mà không bị bias bởi tập Validation.
## 8. TASK T0401 — CONTROLLED ABLATION DATA AUGMENTATION (RQ3)

### 8.1. Luồng cấu hình và pipeline biến đổi

- `experiments/ablation/exp09_aug_none.yaml`, `exp09_aug_basic.yaml` và `exp10_aug_advanced.yaml` cố định EfficientNet-B0 pretrained, RGB 128×128, ImageNet normalization, seed 42, split chuẩn, AdamW, LR/weight decay, ReduceLROnPlateau, batch size và epochs. Chỉ `augmentation.mode` khác nhau giữa `none`, `basic`, `advanced`; EXP-09 chứa hai đối chứng, EXP-10 là mức RandomErasing.
- `Trainer` đọc mode từ YAML, truyền mode và `aug_config` qua `get_dataloaders()` tới `SOTATransformBuilder`. `none` không thêm phép ngẫu nhiên; `basic` áp dụng HorizontalFlip(p=0.5), Rotation(10°), Affine translation 8%; `advanced` thêm `RandomErasing(p=0.25, scale=(0.02,0.15), ratio=(0.3,3.3), value="random")` sau `ToTensor`/Normalize.
- `build_eval_transform()` là pipeline độc lập và chỉ gồm chuyển RGB, resize, `ToTensor`, Normalize. Do dataset gán transform này cho cả PublicTest và PrivateTest, không có augmentation ngẫu nhiên trên hai split đánh giá.
- Run name SOTA có hậu tố `aug-<mode>` để Auto-Skip phân biệt ba recipe; param `augmentation` lưu mode, còn `scheduler` và `batch_size` được ghi vào MLflow để xuất qua đúng các cột CSV 24-column hiện có.

### 8.2. Runner đa nền tảng và trạng thái thực nghiệm

- `notebooks/colab/run_t0401_colab.ipynb` dùng local-first MLflow/checkpoint trên SSD, đồng bộ Drive bằng thay thế file tạm atomic, giữ `.trash` cho MLflow FileStore, bỏ qua cài lại Torch/Torchvision và cập nhật pin scikit-learn khi Python 3.13.
- `notebooks/kaggle/run_t0401_kaggle.ipynb` giữ nguyên CUDA image, đọc GitHub/Kaggle Secrets, ưu tiên competition archive, fallback downloader, có thể merge backup `mlruns` từ Kaggle Dataset và xuất artifacts vào `/kaggle/working/BTL02/T0401/`.
- Auto-Skip chỉ áp dụng cho run MLflow `FINISHED`; không phải resume giữa epoch. Khi chuyển file store, vẫn cần xác minh artifact URI và checkpoint thực tế.
- **Trạng thái định lượng (cập nhật 2026-11-23):** CSV thực tế `outputs/logs/t0401_ablation_augmentation_24col.csv` có đúng 24 cột và 3 run `FINISHED`, đều seed 42. `none`: Run ID `7498d3c0b68743a1805ca06826e5c1ea`, Best Epoch 16, Train Loss 0.205217, Train Acc 93.252987%, Public Acc 66.759543%, Private Acc 68.152689%, Macro-F1 0.665690, Train−Private gap 25.100298 pp. `basic`: Run ID `622a723ec334441898003282a5d8d725`, Epoch 16, Loss 0.455350, Train Acc 83.520848%, Public 69.657286%, Private 70.130956%, Macro-F1 0.700074, gap 13.389891 pp. `advanced`: Run ID `35093d88d1a44e67b5117d8076f715ec`, Epoch 20, Loss 0.478602, Train Acc 82.326100%, Public 69.267205%, Private 70.186682%, Macro-F1 0.698806, gap 12.139418 pp. Gap là Train Acc trừ Private Test Acc; PublicTest được Trainer dùng làm validation/model selection.
- **Diễn giải RQ3 có điều kiện:** từ none sang basic, Private Acc tăng 1.9783 điểm phần trăm, Macro-F1 tăng 0.034384 và gap giảm 11.7104 điểm phần trăm. Từ basic sang advanced, Private Acc chỉ tăng 0.0557 điểm, Macro-F1 giảm 0.001269, gap giảm thêm 1.2505 điểm. Đây là mô tả của một seed, không có mean±std/kiểm định nên không khẳng định ý nghĩa thống kê hoặc recipe tối ưu phổ quát.
- **Giới hạn per-class:** CSV 24 cột không có classification report từng lớp, confusion matrix, epoch history hoặc ví dụ ảnh bị erase. Không thể khẳng định EXP-10 làm giảm điểm một lớp cảm xúc cụ thể hay đã che mắt/miệng/lông mày; đó là cơ chế có thể xảy ra cần kiểm chứng bằng per-class metrics và trực quan hóa các mẫu biến đổi.

**Hỏi: Vì sao run name phải kèm augmentation mode?**
Trả lời: Nếu chỉ dựa vào EXP ID/architecture/optimizer, hai recipe EXP-09 none và basic có thể trùng identity; Auto-Skip có nguy cơ bỏ qua nhầm một biến thể. Hậu tố mode làm mỗi recipe có run name riêng.

**Hỏi: RandomErasing có thể áp dụng lên PublicTest/PrivateTest không?**
Trả lời: Không. RandomErasing chỉ có trong train transform của mode advanced. Eval transform riêng không chứa random operator, nên PublicTest và PrivateTest luôn deterministic.

**Hỏi: Có thể kết luận mức augmentation tốt nhất khi mới có smoke test không?**
Trả lời: Smoke test chỉ kiểm tra forward/backward và hợp đồng pipeline. Hiện đã có ba run GPU `FINISHED` trong CSV 24 cột nên có thể so sánh mô tả các metric; tuy nhiên chỉ một seed và không có per-class report, vì vậy chưa đủ để kết luận thống kê hoặc xác định augmentation tối ưu tổng quát.

**Hỏi: Basic augmentation đã thay đổi kết quả thế nào so với không augmentation trong T0401?**
Trả lời: Ở seed 42, Private Acc tăng từ 68.1527% lên 70.1310%, Macro-F1 từ 0.665690 lên 0.700074, trong khi Train−Private gap giảm từ 25.1003 xuống 13.3899 điểm phần trăm. Đây là bằng chứng mô tả của một lần chạy, không thay thế đánh giá nhiều seed.

**Hỏi: RandomErasing có làm mô hình tổng quát hóa tốt hơn basic không?**
Trả lời: Advanced có gap nhỏ hơn 1.2505 điểm phần trăm và Private Acc nhỉnh hơn 0.0557 điểm, nhưng Macro-F1 thấp hơn 0.001269. Chênh lệch nhỏ, chỉ một seed; kết quả chưa cho thấy cải thiện nhất quán. Cần nhiều seed và metrics theo lớp trước khi kết luận.

**Hỏi: Có thể nói RandomErasing làm mất chi tiết mắt/miệng/lông mày và gây giảm điểm lớp nào không?**
Trả lời: Chưa. RandomErasing có thể che ngẫu nhiên một vùng ảnh, nhưng CSV tổng kết không lưu tọa độ vùng che hoặc F1 từng lớp. Muốn kết luận cần lưu/kiểm tra ảnh augmented, confusion matrix và classification report theo 7 lớp; không được suy ra lớp bị ảnh hưởng chỉ từ Macro-F1 tổng.

**Hỏi: Chép thư mục MLflow có tiếp tục một epoch đang chạy dở không?**
Trả lời: Không. MLflow chứa tracking và trạng thái run, không mặc nhiên chứa đủ trạng thái optimizer/model để tiếp tục epoch. Auto-Skip chỉ áp dụng cho run `FINISHED`; resume cần checkpoint được pipeline hỗ trợ.

## 9. TASK T0402 — CLASS IMBALANCE LOSSES (EXP-11, RQ3)

### 9.1. Luồng cấu hình và ranh giới chống rò rỉ

- Ba YAML `experiments/ablation/exp11_loss_ce.yaml`, `exp11_loss_weighted_ce.yaml`, `exp11_loss_focal.yaml` giữ nguyên EfficientNet-B0 pretrained, ảnh RGB 128×128, seed 42, split 28.709/3.589/3.589, augmentation `basic`, AdamW, LR/weight decay, scheduler, batch size và epoch budget. Biến độc lập duy nhất là loss: `cross_entropy`, `weighted_ce`, `focal` (gamma=2).
- `Trainer` khởi tạo ba DataLoader theo hợp đồng hiện có; class weights chỉ nhận `train_loader.dataset.labels`, vốn được tạo sau khi `FER2013Dataset` lọc `Usage == "Training"`. Nhãn của `val_loader` và `test_loader` không được đưa vào `LossFactory`. Nếu weighted mode không có nhãn Training hoặc thiếu bất kỳ lớp nào thì Factory dừng bằng lỗi rõ ràng, không ước lượng từ tập đánh giá.
- Công thức `inverse_freq`: $w_c=N/(K\,N_c)$, sau đó chuẩn hóa vector để trung bình trọng số bằng 1. `effective_num` là tùy chọn thay thế trong Factory. Với FER2013, nhãn có shape `[N]`, class weights `[7]`, logits của EfficientNet-B0 `[B,7]`, targets `[B]`.
- Chỉ vòng lặp train sử dụng `self.criterion` do YAML chọn. Validation và PrivateTest luôn dùng `self.eval_criterion = CrossEntropyLoss()` không trọng số để chọn checkpoint/đánh giá trên cùng thước đo giữa ba ablation; augmentation của các split đánh giá tiếp tục deterministic và không nhận augmentation train.

### 9.2. CrossEntropy, Weighted CE và Focal Loss

- CrossEntropy chuẩn lấy negative log-probability của lớp đúng. Weighted CE thay đổi đóng góp từng mẫu bằng `weight[target]`, nhờ đó lỗi của lớp ít mẫu như Disgust có ảnh hưởng lớn hơn tới gradient. Chuẩn hóa trọng số trung bình 1 giúp hạn chế thay đổi thang loss tổng.
- `FocalLoss(nn.Module)` tính `ce = F.cross_entropy(logits, targets, reduction="none")`, `pt = exp(-ce)`, rồi `((1-pt)**gamma) * ce`. Với gamma=2, mẫu phân loại dễ (pt gần 1) bị giảm mạnh loss; mẫu khó giữ ảnh hưởng tương đối lớn. `alpha` có thể là hệ số vô hướng hoặc vector lớp, được đăng ký thành buffer để chuyển cùng criterion sang CPU/CUDA. Khi gamma=0 và alpha=None, focal loss bằng CrossEntropy chuẩn.
- `LossFactory.create()` đọc khối YAML, sinh criterion và chuyển buffer/weight sang device của Trainer. YAML cũ không khai báo `loss` mặc định dùng `cross_entropy`; tên run chỉ thêm hậu tố loss khi loss được khai báo, giúp không đổi identity các run cũ và ngăn Auto-Skip lẫn ba EXP-11.
- Trainer tạo `classification_report` trên PrivateTest với danh sách nhãn cố định 0–6 và `zero_division=0`, rồi lưu JSON thành MLflow artifact. Báo cáo gồm precision/recall/F1 từng lớp và macro/weighted averages; file này bổ sung chứ không sửa schema CSV MLflow 24 cột.

### 9.3. Trạng thái kiểm chứng T0402

- Đã triển khai module loss, Trainer integration, ba recipe EXP-11, unit tests và runner notebook Colab/Kaggle. Kiểm thử T0402 được chạy lại trong phiên nghiệm thu này; trạng thái cuối được ghi trong execution report.
- CSV thật `outputs/logs/t0402_ablation_imbalance_24col.csv` có đúng 24 cột và 3 run `FINISHED`, cùng seed 42. Số liệu đối sánh và diễn giải thực tế được ghi tại mục 9.4 bên dưới; CSV không chứa per-class report.
- Không suy luận rằng Weighted CE/Focal cải thiện riêng Disgust/Fear từ Macro-F1 tổng. Muốn khẳng định cần đối chiếu artifact classification report/confusion matrix của đúng các Run ID.
- Colab runner dùng `/content/mlruns_t0402`, checkpoint/log local SSD rồi atomic-sync Drive tại `MyDrive/BTL02/T0402_Imbalance`; lọc torch/torchvision khi cài requirements. Kaggle runner giữ nguyên framework trong image, đọc Secrets, hỗ trợ competition archive/Kaggle API token và khôi phục chỉ run MLflow `FINISHED` từ dataset backup mà không ghi đè run local. Hai runner xuất schema `MLflowLogger.CSV_COLUMNS` 24 cột; Auto-Skip chỉ áp dụng run FINISHED.

**Hỏi: Vì sao class weights chỉ được tính từ Training?**
Trả lời: Tần suất lớp là thống kê được dùng để thay đổi quá trình tối ưu. Dùng nhãn PublicTest/PrivateTest sẽ làm lộ phân phối đánh giá vào huấn luyện, gây data leakage và làm so sánh không còn độc lập. Trainer chỉ chuyển nhãn từ `train_loader.dataset.labels` vào LossFactory.

**Hỏi: Weighted CE khác Focal Loss ở điểm nào?**
Trả lời: Weighted CE tăng/giảm loss theo lớp thật của mẫu; trọng số nghịch đảo tần suất nhắm tới cân bằng đóng góp lớp. Focal Loss điều tiết theo độ khó từng mẫu bằng `(1-pt)^gamma`, giảm ảnh hưởng mẫu dễ bất kể lớp. Chúng có thể kết hợp qua alpha/class weights nhưng EXP-11 focal hiện để alpha=None để cô lập focal modulation.

**Hỏi: Vì sao validation/test không dùng criterion có class weights?**
Trả lời: Validation dùng chọn checkpoint và cần cùng một thước đo cho cả ba cấu hình; PrivateTest cần báo cáo loss/accuracy theo đánh giá không trọng số. Chỉ loss của các batch train thay đổi theo ablation. Accuracy và Macro-F1 được tính từ logits/nhãn, không từ class-weighted loss.

**Hỏi: Focal Loss gamma=0 có gì đặc biệt?**
Trả lời: Với alpha=None, `(1-pt)^0=1`, nên loss trở về CrossEntropy. Đây là kiểm tra quan trọng để xác minh implementation và cho thấy gamma điều khiển mức tập trung vào mẫu khó.

**Hỏi: Nếu Macro-F1 tăng thì có chắc Recall của Disgust và Fear tăng không?**
Trả lời: Không. Macro-F1 là trung bình F1 của các lớp, nhưng có thể tăng do lớp khác. Cần đọc precision/recall/F1 từng lớp từ JSON classification report và confusion matrix; đồng thời so sánh PrivateTest Acc để lượng hóa trade-off.

**Hỏi: Auto-Skip có resume run bị ngắt giữa epoch không?**
Trả lời: Không. Runner chỉ bỏ qua run MLflow `FINISHED`. Run dở không được coi là hoàn thành; resume giữa epoch cần checkpoint chứa đủ model, optimizer, scheduler và epoch state, hiện không được runner giả định.

## 10. TASK T0403 — ABLATION LEARNING-RATE SCHEDULER (EXP-12, RQ3)

### 10.1. Controlled recipe và cách đặt tên run

- Ba YAML `experiments/ablation/exp12_sched_step_or_plateau.yaml`, `exp12_sched_cosine.yaml`, `exp12_sched_warmup_cosine.yaml` kế thừa recipe EXP-11 trên EfficientNet-B0 pretrained, ảnh RGB 128×128, seed 42, split chuẩn FER2013, augmentation basic chỉ trên Training, AdamW `lr=1e-4`, `weight_decay=1e-4`, batch 32, tối đa 30 epochs, patience 7 và CrossEntropyLoss. Scheduler là biến thí nghiệm.
- Biến thể đối chứng mang tên file `step_or_plateau` dùng `ReduceLROnPlateau(mode=min)` trên `val_loss`, factor 0.1, patience 3; cosine dùng `CosineAnnealingLR(T_max=30, eta_min=1e-6)`; warmup-cosine dùng `LinearLR` 3 epochs từ 0.2× LR nền tới LR nền rồi `CosineAnnealingLR` 27 epochs.
- `Trainer._build_scheduler()` chuẩn hóa alias để chấp nhận `step`/`StepLR`, `plateau`/`reduce_on_plateau`/`ReduceLROnPlateau`, `cosine`/`cosine_annealing`/`CosineAnnealingLR` và `WarmupCosine`. YAML cũ vẫn mặc định `None`, còn tên ReduceLROnPlateau/CosineAnnealingLR cũ hoạt động như trước.
- Scheduler được cập nhật một lần mỗi epoch sau khi tính validation metrics: Plateau nhận `val_loss` nếu `mode=min`, hoặc `val_acc` nếu `mode=max`; scheduler theo lịch gọi `.step()` không đối số. `SequentialLR` ghép warmup/cosine theo milestone epoch. Optimizer update theo batch vẫn diễn ra trước scheduler step của epoch.
- T0403 thêm hậu tố scheduler vào run name cho ba recipe mới để Auto-Skip không thể nhầm các scheduler cùng EXP/architecture/optimizer. Hậu tố chỉ xuất hiện khi YAML bật `scheduler_run_suffix`; run name của cấu hình cũ không thay đổi.
- Notebook Colab đặt tracking/checkpoint/log trên SSD trước, atomic-sync artifact ổn định sang Drive sau mỗi run; Kaggle hỗ trợ competition CSV/archive hoặc Secret `KAGGLE_API_TOKEN`, GitHub PAT dạng `x-access-token:...`, và phục hồi các MLflow run `FINISHED` từ dataset backup. Cả hai xuất cùng `MLflowLogger.CSV_COLUMNS` 24 cột.

### 10.2. Giải thích thuật toán và ý nghĩa trong RQ3

- **StepLR:** sau mỗi `step_size` epoch, nhân learning rate hiện tại với `gamma`. Đây là lịch giảm theo mốc cố định, đơn giản và không phản ứng trực tiếp với validation.
- **ReduceLROnPlateau:** theo dõi metric validation đã chọn; khi metric không cải thiện đủ số epoch theo `patience`, LR nhân `factor` nhưng không thấp hơn `min_lr`. Với `mode=min`, metric là validation loss; với `mode=max`, Trainer dùng validation accuracy.
- **CosineAnnealingLR:** LR giảm theo nửa chu kỳ cosine từ LR ban đầu tới `eta_min` trong `T_max` epoch. Lịch mượt, không cần metric validation để quyết định thời điểm giảm.
- **Warmup + Cosine:** `LinearLR` khởi động ở một tỷ lệ LR nhỏ để giảm cập nhật quá lớn lúc đầu; sau số warmup epoch định trước, `SequentialLR` chuyển sang cosine annealing. `warmup_epochs` phải lớn hơn 0 và nhỏ hơn tổng số epoch.
- Với EfficientNet-B0, logits cho một batch là `[B,7]`; scheduler không thay đổi shape tensor hoặc số tham số mô hình, chỉ cập nhật `optimizer.param_groups[*]['lr']`. PublicTest là validation/model-selection split; PrivateTest chỉ đánh giá checkpoint sau huấn luyện. Eval transforms xác định và không áp dụng augmentation.
- CSV 24 cột lưu LR tổng kết qua params nhưng không lưu toàn bộ chuỗi learning rate theo epoch dưới một cột riêng. MLflow history ghi `metrics.lr`, `train_loss`, `val_loss`, `train_acc`, `val_acc` tại từng epoch; cần truy vấn history để định lượng độ mượt/quỹ đạo hội tụ. Best Epoch và CSV tổng kết một mình không chứng minh dao động hay kẹt cực tiểu.

### 10.3. Kiểm thử và trạng thái thực nghiệm

- `tests/test_ablation_scheduler.py` kiểm tra lớp scheduler PyTorch, alias cũ/mới, thay đổi LR, warmup rồi cosine giảm, controlled invariants, forward/backward một batch CPU và tính duy nhất của tên run. Chạy targeted suite: **11 passed**.
- Hai notebook T0403 có JSON hợp lệ, code cell parse được; chưa thực thi trên Colab/Kaggle trong phiên nghiệm thu. Máy phát triển dùng PyTorch CPU-only, `torch.cuda.is_available() == False`.
- CSV thực tế `outputs/logs/t0403_ablation_scheduler_24col.csv` có đúng 24 cột, ba run `FINISHED`, seed 42. Bảng metric và diễn giải bên dưới là quan sát của một run mỗi scheduler; không có per-epoch history kèm theo CSV.
- Full test suite đã hoàn tất: **152 passed**, 2 warning, 298.84s; targeted scheduler tests **11 passed**.

### 10.4. Kết quả thực nghiệm EXP-12 thực tế

| Scheduler | MLflow Run ID | Best Epoch | Train Loss | Train Acc | Best Val / Public Acc | Private Test Acc | Macro-F1 | Train−Private Gap |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ReduceLROnPlateau (đối chứng) | `f97118d157a04325842acaa091362a03` | 16 | 0.455350 | 83.520847% | 69.657286% | **70.130956%** | **0.700074** | **13.389891 pp** |
| CosineAnnealingLR | `7348e1b10b22408fb71dc61920b1ef5c` | 28 | 0.247717 | 91.187433% | 69.490109% | 69.963778% | 0.694598 | 21.223654 pp |
| Linear Warmup + Cosine | `a312721f1e884bb7891f464f984e6551` | 22 | 0.264893 | 90.682364% | 69.517972% | 69.935915% | 0.689152 | 20.746449 pp |

- Trong seed 42, Plateau cao hơn cosine **0.167177 điểm phần trăm Private Acc** và **0.005476 Macro-F1**; cao hơn warmup-cosine lần lượt **0.195040 điểm** và **0.010922 Macro-F1**. Gap của cosine và warmup-cosine lần lượt lớn hơn Plateau **7.833763 pp** và **7.356558 pp**.
- Cosine đạt Train Loss nhỏ nhất (0.247717) và Train Acc cao nhất (91.187433%), nhưng Private Acc/Macro-F1 thấp hơn Plateau. Điều này minh họa rằng metric train hoặc loss cuối không tự động đồng nghĩa tổng quát hóa tốt hơn.
- Best Epoch lần lượt 16/28/22; Duration CSV lần lượt 1,988.973/2,561.110/2,608.358 giây (~33.15/42.69/43.47 phút; Plateau/Cosine/WarmupCosine). Best Epoch là epoch đạt validation accuracy cao nhất; duration run đơn không cô lập chi phí scheduler vì throughput có thể biến thiên.
- **Giới hạn trajectory:** CSV 24 cột không lưu per-epoch history và không có VRAM. Best Epoch/Duration không đủ để khẳng định scheduler nào mượt, ít dao động hoặc tránh kẹt cực tiểu tốt hơn. Cần MLflow history của `metrics.lr`, `train_loss`, `val_loss` và nhiều seed.
- **Kết luận có điều kiện:** Plateau là tốt nhất về Private Acc, Macro-F1 và gap trong ba lần chạy được cung cấp; với một seed, không tuyên bố đây là scheduler tối ưu phổ quát hoặc kết quả có ý nghĩa thống kê.

**Hỏi: Scheduler nào thắng EXP-12 theo số liệu hiện có?**
Trả lời: ReduceLROnPlateau đạt Private Acc 70.130956%, Macro-F1 0.700074 và gap 13.389891 pp, tốt nhất trong ba run seed 42. Cosine và warmup-cosine có Private Acc/Macro-F1 thấp hơn; kết luận chỉ áp dụng cho các run này.

**Hỏi: Vì sao cosine có Train Loss thấp nhất nhưng không thắng trên PrivateTest?**
Trả lời: Train Loss phản ánh tối ưu trên tập train, còn PrivateTest đo khả năng tổng quát hóa. Cosine có Train Acc 91.187433% nhưng gap 21.223654 pp và Private Acc 69.963778%, thấp hơn Plateau; chỉ nhìn train loss sẽ bỏ qua khoảng cách tổng quát hóa.

**Hỏi: Best Epoch 28 của cosine chứng minh hội tụ chậm hoặc mượt hơn không?**
Trả lời: Không. Best Epoch chỉ cho biết epoch có validation accuracy tốt nhất, không mô tả các epoch trước/sau. Muốn đánh giá tốc độ và độ mượt cần chuỗi `lr`, train/validation loss/accuracy theo epoch.

**Hỏi: Đã chứng minh Plateau là scheduler tốt nhất cho FER2013 chưa?**
Trả lời: Chưa. Mỗi scheduler chỉ có một seed; chưa có mean±std hoặc kiểm định. Có thể nói Plateau tốt nhất theo ba run được ghi nhận, nhưng cần lặp nhiều seed để kết luận ổn định.

**Hỏi: Vì sao cần hậu tố scheduler trong tên MLflow run?**
Trả lời: Nếu identity chỉ gồm EXP-12, kiến trúc và optimizer, ba lịch LR có thể trùng tên; Auto-Skip sẽ bỏ qua sai cấu hình. Hậu tố được bật cho riêng EXP-12 để phân biệt run mà không thay đổi tên các run lịch sử.

**Hỏi: Vì sao gọi ReduceLROnPlateau với validation metric còn CosineAnnealingLR không có đối số?**
Trả lời: Plateau cần metric để xác định có chững lại hay không; `mode=min` truyền validation loss, `mode=max` truyền validation accuracy. Cosine và StepLR tiến theo lịch epoch định trước nên `.step()` không cần metric.

**Hỏi: Warmup có làm tăng số tham số hoặc đổi kiến trúc không?**
Trả lời: Không. Warmup chỉ thay learning rate của optimizer qua các epoch; kiến trúc, tensor logits `[B,7]` và số tham số giữ nguyên.

**Hỏi: Có thể nhìn Best Epoch/CSV tổng kết để kết luận scheduler nào hội tụ mượt nhất không?**
Trả lời: Không. CSV cuối run chỉ lưu điểm tổng kết và Best Epoch, không giữ chuỗi loss/LR. Cần truy vấn per-epoch metrics từ MLflow và vẽ `train_loss`, `val_loss`, `lr`; so sánh nhiều seed nếu muốn đánh giá ổn định.

**Hỏi: Có thể chọn scheduler có PublicTest accuracy cao nhất làm scheduler tốt nhất không?**
Trả lời: PublicTest được dùng làm validation/model-selection nên không phải holdout độc lập. Cần báo cáo PrivateTest Accuracy/Macro-F1 cùng gap, đồng thời nhiều seed; không suy luận độ mượt chỉ từ accuracy cuối.

**Hỏi: T0403 đã có kết luận thực nghiệm chưa?**
Trả lời: Đã có CSV gồm ba run `FINISHED`; trong seed 42, Plateau dẫn đầu Private Acc/Macro-F1 và có gap thấp nhất. Tuy nhiên chưa có epoch history và chỉ có một seed, nên chưa thể đánh giá độ mượt hoặc kết luận scheduler tối ưu phổ quát.

### 9.4. Kết quả định lượng EXP-11 thực tế (CSV 24 cột)

CSV `outputs/logs/t0402_ablation_imbalance_24col.csv` xác nhận ba cấu hình cùng EfficientNet-B0 pretrained, ảnh RGB 128×128, seed 42, augmentation basic, AdamW, LR 0.0001, batch 32 và scheduler ReduceLROnPlateau; chỉ loss thay đổi. `Best Val Acc` là PublicTest được dùng làm validation/model selection trong pipeline. Generalization gap được tính bằng Train Acc trừ Private Test Acc, đơn vị điểm phần trăm (pp).

| Chiến lược loss | MLflow Run ID | Best Epoch | Train Loss | Train Acc | Best Val / Public Test Acc | Private Test Acc | Macro-F1 (Private) | Generalization Gap |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CrossEntropyLoss (đối chứng) | `382d2ca2a18f4a219c877c7c5d4354c6` | 16 | 0.455350 | 83.520847% | 69.657286% | 70.130956% | 0.700074 | 13.389891 pp |
| Weighted CrossEntropyLoss | `a82a528fa1154003a203021ba8f71c45` | 16 | 0.464777 | 81.629454% | 68.124826% | 69.434383% | 0.684786 | 12.195071 pp |
| Focal Loss (γ=2.0, α=None) | `2f3cd08e2bc145baa0636a9317af3c58` | 28 | 0.155761 | 85.899892% | 70.075230% | 69.685149% | 0.695393 | 16.214743 pp |

- So với đối chứng, Weighted CE giảm Macro-F1 **0.015288** (xấp xỉ 2.18% tương đối) và Private Acc **0.696573 pp**; Public/validation Acc giảm **1.532460 pp**. Gap giảm **1.194820 pp**, nhưng đây không bù được suy giảm ở cả Macro-F1 lẫn Private Acc.
- Focal Loss giảm Macro-F1 **0.004682** (xấp xỉ 0.67% tương đối) và Private Acc **0.445807 pp**, dù Public/validation Acc tăng **0.417944 pp**. Gap tăng **2.824852 pp**. Với cấu hình/seed quan sát được, focal chưa thể hiện ưu thế tổng quát hóa trên PrivateTest.
- Cả hai loss được đề xuất đều cho Macro-F1 thấp hơn đối chứng; do đó thực nghiệm này **không ủng hộ giả thuyết** rằng chúng cải thiện Macro-F1 tổng trong recipe hiện tại. Weighted CE có thể tăng trọng số lớp hiếm về mặt mục tiêu tối ưu, còn Focal Loss tập trung vào mẫu khó, nhưng cơ chế đó không đảm bảo metric kiểm thử sẽ tăng.
- Macro-F1 là trung bình F1 của 7 lớp, không cho biết riêng F1/recall của `Disgust` hay `Fear`. CSV tổng kết không mang precision/recall/F1 từng lớp; cần artifact `classification_report`/confusion matrix gắn đúng Run ID để kết luận nhận diện lớp thiểu số. Kết quả hiện tại chỉ có một seed, nên đây là so sánh mô tả, không phải kết luận ý nghĩa thống kê.

**Hỏi: Weighted CE có cải thiện Macro-F1 và Private Accuracy trong T0402 không?**
Trả lời: Không trong run seed 42 này. So với CE, Macro-F1 giảm từ 0.700074 xuống 0.684786 và Private Acc giảm 0.696573 điểm phần trăm. Gap nhỏ hơn khoảng 1.194820 điểm, nhưng không làm thay đổi kết luận về hai metric đánh giá chính.

**Hỏi: Focal Loss có thể coi là tốt hơn vì Best Val Acc cao nhất không?**
Trả lời: Không. Focal đạt Public/validation Acc 70.075230%, cao hơn CE 0.417944 điểm, nhưng Private Acc thấp hơn 0.445807 điểm và Macro-F1 thấp hơn 0.004682. Hơn nữa, PublicTest tham gia chọn checkpoint nên không thay thế kết quả PrivateTest.

**Hỏi: Vì sao Focal Loss có Train Loss thấp nhưng Macro-F1 không cao nhất?**
Trả lời: Các trị loss của các hàm mất mát khác nhau không cùng thang đo để so sánh trực tiếp. Focal nhân CE từng mẫu với $(1-p_t)^\gamma$, làm thay đổi mục tiêu tối ưu; Train Loss thấp hơn không đồng nghĩa với accuracy/Macro-F1 kiểm thử cao hơn.

**Hỏi: Có thể nói Weighted CE/Focal nhận diện Disgust hoặc Fear tốt hơn không?**
Trả lời: Chưa thể. Bảng CSV chỉ ghi Macro-F1 tổng và accuracy, không có điểm từng lớp. Cần mở classification report hoặc confusion matrix đúng Run ID, kiểm tra riêng precision/recall/F1 của Disgust, Fear và đối chứng; không suy ra per-class từ Macro-F1 tổng.

**Hỏi: Kết quả này đã chứng minh CE luôn tốt nhất chưa?**
Trả lời: Không. Chỉ có một seed (42), một recipe EfficientNet-B0/augmentation basic và một lần chạy cho mỗi loss. Kết quả chỉ cho thấy CE dẫn đầu trên Private Acc và Macro-F1 ở lần chạy này; cần nhiều seed và per-class report để đánh giá độ ổn định và khả năng cải thiện lớp hiếm.

## Bổ sung kỹ thuật ngày 2026-11-18: Audit và kagglehub

### Nạp .env an toàn (phiên 22:33)

`FER2013Downloader.__init__` gọi `load_environment`: project .env ưu tiên trước CWD, chỉ nạp một tệp, không dò thư mục cha khác. `override=False` giữ quyền ưu tiên của shell; `interpolate=False` giữ nguyên ký tự token. Tham số `env_file` cho phép tests chỉ đọc cấu hình giả trong tmp_path. Không tự tạo .env và không in giá trị biến. Năm test mới nâng toàn suite lên **34 passed** với dependency thật.

**Hỏi: Vì sao copy .env.example chưa đủ để tải?** Phải thay placeholder bằng token mới riêng tư và Accept Rules; token mẫu bị từ chối. Không đưa token vào chat/Git. SHA-256 chỉ dùng đối chiếu dữ liệu, không chứng minh provenance bằng chữ ký Kaggle.

- `FER2013Downloader` gọi `kagglehub.competition_download` sau kiểm tra nguồn chứng thực. Ưu tiên token môi trường, access_token, rồi legacy env/json; không log token. Cache kagglehub chỉ đọc. ZIP/TAR.GZ được sao chép theo khối vào vùng tạm, không extractall; CSV đúng header, nhãn, pixel và split mới được atomic replace. SHA-256 là checksum đối chiếu byte, không phải chữ ký của Kaggle.
- `ActivationFactory` tạo module mới cho từng tầng; mặc định giữ nguyên số tham số: Small 1.199.495, Medium 2.731.015, Large 3.916.167. Với input `[2,1,48,48]`, Medium sau pooling có `[2,256,6,6]`, Large `[2,512,3,3]`; cả hai xuất logits `[2,7]`. Channels đổi sẽ làm đổi shape trọng số; chỉ cấu hình mặc định tương thích state_dict mặc định cũ.
- `Trainer` truyền activation/channels/dropout, worker/class count và normalization từ YAML. Plateau dùng validation loss hoặc accuracy theo mode; Cosine cập nhật sau epoch. SmallCNN cũng nhận activation, tránh YAML bị bỏ qua âm thầm. Builder RGB hỗ trợ 128/224 và ImageNet normalization, không augment tập đánh giá.
- `MLflowLogger.CSV_COLUMNS` cố định 24 cột; metadata ánh xạ từ MLflow, Duration tính theo giây, trường thiếu điền None. Cột activation/channel vẫn log vào MLflow nhưng không thêm vào schema Sheets đã khóa.
- Test mới dùng dependencies thật và dữ liệu tổng hợp ngoài repo: **29 passed**, không skip. Không xác minh download/auth thật hoặc huấn luyện lại. Seed hỗ trợ tái lập nhưng không bảo đảm bitwise giữa các phiên bản/phần cứng.

**Hỏi: Kết quả ba CNN đã chứng minh tăng độ sâu gây tăng accuracy chưa?**
Trả lời: Chưa. Width, số tham số, BatchNorm và Dropout cùng thay đổi; T0203 cần đối chứng, không được suy diễn nhân quả chỉ từ bảng kết quả cũ.

**Hỏi: Private Test cao hơn Public Test có phải do tắt Dropout không?**
Trả lời: Không thể kết luận như vậy: cả Public và Private đều phải dùng eval(), nên Dropout đều tắt. Chênh lệch tập mẫu cần được phân tích, không giả định PrivateTest dễ hơn nếu chưa có bằng chứng.

**Q7: Tại sao MediumCNN lại bị overfit nặng (Val Loss tăng vọt lên 2.88) trong khi LargeCNN dù nhiều tham số hơn (3.92M so với 2.73M) lại kiểm soát loss tốt hơn nhiều (Val Loss 1.60)?**
* *Trả lời:* Đây là minh chứng kinh điển trong Deep Learning: **Dung lượng mạng (Capacity) phải đi đôi với Kỹ thuật điều hòa (Regularization)**. 
  - `MediumCNN` có 2.73M tham số nhưng không có `BatchNorm2d` và chỉ có 1 lớp Dropout ở cuối, dẫn đến các bộ lọc ở khối 3 bị đồng kích hoạt (co-adaptation).
  - `LargeCNN` dù sâu hơn và có 3.92M tham số, nhưng được trang bị `BatchNorm2d` sau mỗi tầng tích chập (giúp phân phối kích hoạt luôn chuẩn hóa) và `Dropout2d` (ngẫu nhiên ngắt toàn bộ kênh đặc trưng trong quá trình học), buộc các bộ lọc phải học các đặc trưng độc lập, chống phụ thuộc lẫn nhau, từ đó ghìm chặt Val Loss ở mức thấp.

## Bổ sung kỹ thuật T0203: chạy Colab và định tuyến MLflow lên Google Drive

- `notebooks/run_t0203_colab.ipynb` tách quy trình thành sáu code cell: mount Drive; clone đúng branch; cài các dependency không thay PyTorch/Torchvision GPU có sẵn của Colab; nạp Kaggle Secret và xác thực ba split; liên kết thư mục checkpoint và chạy ba YAML tuần tự; xuất CSV tổng hợp.
- Token GitHub/Kaggle được lấy qua `google.colab.userdata`, không ghi literal vào notebook. Với clone private, Git AskPass lấy token từ môi trường tiến trình thay vì đặt trong URL hoặc tham số command line; sau clone, remote URL không chứa credential. Người dùng cần thu hồi token nếu từng commit/paste token vào file khác.
- Cell cài dependency lọc `torch`/`torchvision` để giữ nguyên bản CUDA của Colab. Nếu runtime là Python 3.13, notebook thay pin scikit-learn 1.5.2 bằng 1.6.1; các phiên bản Python khác vẫn cài theo `requirements.txt`.
- `MLflowLogger` chọn URI theo thứ tự: đối số `tracking_uri`, biến `MLFLOW_TRACKING_URI`, mặc định `mlruns`. Vì vậy notebook có thể ghi tracking file-store trực tiếp vào Drive. CSV xuất qua `CSV_COLUMNS` cố định, đúng 24 cột.
- Symlink `outputs/checkpoints` cho phép YAML hiện tại giữ nguyên đường dẫn tương đối nhưng checkpoint được ghi vào Drive. Notebook chỉ thay symlink/thư mục rỗng có `.gitkeep`; nếu phát hiện dữ liệu khác thì dừng, không xóa/ghi đè.
- MLflow FileStore duyệt cả experiment đã xóa khi ghi metric và yêu cầu thư mục `<tracking-root>/.trash`. Google Drive mount có thể làm thiếu thư mục này sau khi logger đã khởi tạo; `MLflowLogger` vì thế bảo đảm root và `.trash` tồn tại lúc khởi tạo cũng như trước mỗi lần ghi metric. Notebook T0304 dùng FileStore trên SSD local và đồng bộ Drive theo chu kỳ/sau mỗi cấu hình; checkpoint được đặt local để tránh FUSE trong vòng lặp huấn luyện.
- **Hỏi: Vì sao không chỉ đặt `MLFLOW_TRACKING_URI` trong notebook?** Trước thay đổi T0203, `MLflowLogger.__init__` luôn gọi URI mặc định `mlruns`, bỏ qua biến môi trường; notebook sẽ ghi sai nơi. Regression test bảo đảm URI Drive được sử dụng.
- **Hỏi: Vì sao không cài lại torch trong cell requirements?** Colab cung cấp PyTorch/Torchvision tương thích CUDA với runtime của họ; cài pin CPU/version khác có thể thay bản GPU. Notebook lọc hai package này khỏi requirements.
- **Giới hạn thí nghiệm:** Cấu hình mạng tăng đồng thời channels, độ sâu và regularization; bảng kết quả chỉ so sánh các gói kiến trúc, không cô lập nhân quả riêng số block. Chưa có kết quả T4 thì không được suy luận bão hòa/overfitting.

## Chuẩn bị T0204: controlled ablation các activation trên LargeCNN

- Bốn YAML `experiments/ablation/exp07_act_*.yaml` dùng LargeCNN channels `(64, 128, 256, 512)`, seed 42, ảnh grayscale 48×48, split chuẩn, augmentation `None`, Adam `lr=0.001`, weight decay 0, batch 64, tối đa 50 epochs và patience 10. Chỉ `activation` thay đổi (`relu`, `leaky_relu`, `gelu`, `elu`).
- `ActivationFactory` được dùng tại từng block tích chập và tầng fully-connected; cùng một activation được áp dụng xuyên feature extractor/classifier. BatchNorm, Dropout và số kênh giữ nguyên giữa các run.
- Trainer định dạng `{activation}` trong đường dẫn checkpoint và summary CSV; tên MLflow run chứa activation. Logger ưu tiên URI đối số, `MLFLOW_TRACKING_URI`, rồi mặc định `mlruns`, để notebook có thể ghi vào Drive.
- Notebook `run_t0204_colab.ipynb` mount Drive, clone branch qua Colab Secret, giữ PyTorch/Torchvision CUDA sẵn có, tải và xác thực split FER2013, chạy tuần tự bốn YAML, xuất summary CSV 24 cột và epoch-metrics CSV (train/validation loss/accuracy) phục vụ đường cong hội tụ.
- **Hỏi: Vì sao cần epoch-metrics CSV ngoài schema 24 cột?** CSV chuẩn chỉ chứa metrics tổng kết của run, không lưu chuỗi từng epoch; muốn phân tích tốc độ hội tụ phải lấy history từ MLflow.
- **Hỏi: Vì sao thay activation cả ở convolution và FC?** Thiết kế T0204 kiểm tra chính sách activation thống nhất toàn LargeCNN. Muốn tách riêng vị trí activation cần một nghiên cứu factorial khác.
- **Trạng thái:** Hạ tầng và forward tests đã chuẩn bị; chưa chạy bốn train runs và chưa kết luận activation tối ưu.

## T0302: VGG16 fine-tuning ImageNet cho FER2013 (RQ2)

- `src/models/pretrained_vgg.py` bọc `torchvision.models.vgg16` trong `BaseEmotionModel`; đầu vào cần RGB `[B,3,H,W]`. `AdaptiveAvgPool2d((7,7))` của torchvision giữ tensor đặc trưng cuối thành `[B,512,7,7]` cho cả ảnh 128 và 224. Flatten tạo 25.088 đặc trưng rồi head mới gồm `Linear(25088,512) -> ReLU -> Dropout -> Linear(512,7)` trả logits `[B,7]`.
- `pretrained=True` chọn `VGG16_Weights.IMAGENET1K_V1`; `False` khởi tạo backbone ngẫu nhiên, dùng cho kiểm thử offline và đối chứng scratch. Trọng số head được khởi tạo Kaiming; bias bằng 0.
- `freeze_backbone()` chỉ khóa `backbone.features`, giữ classifier trainable để warm-up. `unfreeze_backbone()` mở toàn bộ; `unfreeze_backbone(from_block=1..5)` chỉ mở các tầng thuộc block chỉ định trở về cuối. Bộ đếm summary phân biệt tổng, trainable và frozen parameters.
- `experiments/sota/exp04_vgg16_finetune.yaml` cấu hình EXP-04, RQ2, ảnh RGB 128, chuẩn hóa ImageNet, AdamW, ReduceLROnPlateau và augmentation chỉ ở train. `get_dataloaders(transform_mode="sota")` tái sử dụng transform eval xác định cho cả PublicTest và PrivateTest, không augmentation.
- Trainer chấp nhận tên `VGG16` hoặc `PretrainedVGG16`; CNN hiện hữu vẫn dùng transform `custom` mặc định. Optimizer chỉ nhận tham số `requires_grad=True`.

**Hỏi: Vì sao VGG16 vẫn nhận được ảnh 128×128 dù cấu hình gốc thường dùng 224×224?**
Trả lời: Feature extractor dùng các phép pooling tích lũy và AdaptiveAvgPool2d `(7,7)` chuẩn hóa kích thước không gian trước classifier; cả hai đầu vào cho kích thước feature map đủ lớn để pooling về 7×7.

**Hỏi: Freeze backbone có nghĩa là tổng số tham số giảm không?**
Trả lời: Không. Tổng số tham số không đổi; chỉ số tham số cập nhật gradient giảm. `get_parameter_summary()` báo riêng ba con số này.

**Hỏi: Chạy unit test có tải pretrained weights không?**
Trả lời: Không. Các test kiến trúc chạy `pretrained=False`; kiểm tra chế độ pretrained xác nhận enum và đối số weights qua mock, tránh tải mạng. Khi huấn luyện EXP-04 thật, torchvision cần tải/có sẵn weights ImageNet V1.

## T0303: ResNet50 và EfficientNet-B0 ImageNet cho FER2013 (RQ2)

- `src/models/pretrained_resnet.py` bọc `torchvision.models.resnet50`, dùng weights ImageNet V2 khi pretrained. Tensor RGB đi qua stem và residual stages `layer1`–`layer4`, pooling toàn cục trong ResNet tạo vector 2048 chiều. Head mặc định `Dropout -> Linear(2048,7)`; có thể đặt `hidden_dim` để dùng head hai lớp.
- `src/models/pretrained_efficientnet.py` bọc `efficientnet_b0`, dùng ImageNet V1. Backbone tạo vector 1280 chiều; head mặc định `Dropout(inplace=True) -> Linear(1280,7)`, có thể bổ sung hidden layer. Hai mô hình đều nhận ảnh 128×128 hoặc 224×224 và trả logits `[B,7]`.
- Cả hai kế thừa `BaseEmotionModel`, cung cấp tầng Conv cuối cho Grad-CAM, `get_parameter_summary()` và điều khiển freeze. ResNet có thể unfreeze từ residual stage 1–4; EfficientNet từ feature block index 0–8. Optimizer Trainer giữ tham chiếu toàn bộ tham số, nên fine-tuning có thể mở khóa thêm các layer sau khi khởi tạo.
- `exp05_resnet50_finetune.yaml` và `exp06_efficientnet_b0_finetune.yaml` dùng seed 42, ảnh RGB 128, ImageNet normalization và `transform_mode: sota`. `get_dataloaders` gán augmentation chỉ cho transform train, dùng eval transform xác định cho PublicTest/PrivateTest.
- Trainer nhận các alias `ResNet50`/`PretrainedResNet50` và `EfficientNetB0`/`PretrainedEfficientNetB0`; các kiến trúc cũ và VGG16 vẫn được giữ.

**Hỏi: Freeze backbone có làm giảm tổng số tham số của mô hình không?**
Trả lời: Không, tổng tham số kiến trúc không đổi; chỉ lượng tham số được cập nhật gradient giảm. Summary phân biệt total, trainable, frozen.

**Hỏi: Vì sao ResNet50 có thể mở từ `layer4` mà không mở stem?**
Trả lời: Các residual stage có thứ tự tuần tự. Mở layer4 chỉ tinh chỉnh tầng đặc trưng cuối và classifier; các stage trước giữ trọng số ImageNet, giảm số tham số cần cập nhật.

**Hỏi: Vì sao EfficientNet-B0 phù hợp để so sánh với ResNet50 trong RQ2?**
Trả lời: Đây là đối chứng kiến trúc hiệu quả tham số; phải so sánh trên cùng split, seed, transform, batch/recipe và đánh giá riêng tài nguyên tính toán, không kết luận chỉ từ một chỉ số accuracy.

## T0304: Thiết kế đối chứng Transfer Learning và kích thước ảnh (RQ2)

- Chín YAML trong `experiments/sota/` tạo ma trận 3 kiến trúc × 3 điều kiện: ImageNet pretrained 128, scratch 128, ImageNet pretrained 48. Seed, split counts, transform, augmentation train-only, optimizer, LR, scheduler, batch, epochs và patience được kiểm tra bằng unit test để giữ recipe cố định.
- `SOTATransformBuilder` cho phép cạnh ảnh từ 48 trở lên; ảnh FER grayscale được đổi thành 3 kênh và chuẩn hóa ImageNet. Kiểm tra forward 48×48 cho thấy output logits của cả ba model là `[B,7]`; feature cuối thực đo: VGG16 `[B,512,1,1]`, ResNet50 `[B,2048,2,2]`, EfficientNet-B0 `[B,1280,2,2]`. Vì vậy không được giả định mọi backbone có cùng năm lần giảm kích thước.
- Trainer đặt tên run SOTA duy nhất theo EXP ID, kiến trúc, pretrained/scratch, image size và optimizer. Thời gian epoch có synchronize CUDA trước/sau vùng đo; tổng tham số và Public Macro-F1 được ghi vào MLflow metrics bổ sung.
- CSV 24 cột là schema cố định và không có exp_id, pretrained, Macro-F1 Public hoặc total params. Notebook giữ nguyên schema này, nối mỗi run với manifest YAML qua tên run, đồng thời xuất metrics chi tiết bổ sung riêng. PublicTest là validation dùng early stopping/model selection theo pipeline hiện hữu; không diễn giải nó như holdout độc lập. PrivateTest được đánh giá sau khi chọn checkpoint.
- `run_t0304_colab.ipynb` chạy chín cấu hình tuần tự, gắn Drive cho MLflow/checkpoint/log/figure, kiểm tra CSV FER2013 có đúng split counts và tiếp tục bỏ qua các run đã FINISHED. Colab Secret được dùng khi tải dữ liệu và token được gỡ khỏi môi trường sau tiến trình tải.

**Hỏi: Vì sao chưa thể kết luận Pretrained tốt hơn Scratch từ smoke test?**
Trả lời: Smoke test chỉ xác minh pipeline forward/backward; không đo hội tụ hay accuracy. Chỉ CSV/MLflow của đủ run thật mới cho phép tính ΔAcc, macro-F1 và best epoch.

**Hỏi: Vì sao 48×48 không cho cùng feature map ở ba model?**
Trả lời: ResNet50 có stem stride/pooling và adaptive average pooling; EfficientNet có stem cùng stride stages; VGG16 dùng chuỗi max-pooling và adaptive pool 7×7 ở classifier. Hình dạng không gian do kiến trúc cụ thể quyết định, nên phải đo từng backbone.

**Hỏi: PublicTest có phải test set độc lập trong Trainer hiện tại không?**
Trả lời: Không. Trainer dùng PublicTest làm validation để chọn checkpoint/EarlyStopping. Do đó chỉ báo cáo là validation/model-selection metric; không tuyên bố nó là đánh giá holdout không thiên lệch. PrivateTest là split đánh giá cuối theo cấu hình hiện có.

