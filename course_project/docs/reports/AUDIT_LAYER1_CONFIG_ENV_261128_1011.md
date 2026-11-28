# [REPORT] BPM25_T0305_LAYER1_AUDIT: Kiểm toán Tầng 1 — Quản trị Cấu hình, Môi trường & Tính Tái lập

- **Mã công việc (Task ID):** `BPM25_T0305_LAYER1_AUDIT`
- **Tên file lưu vết:** `docs/reports/AUDIT_LAYER1_CONFIG_ENV_261128_1011.md`
- **Thời gian nghiệm thu:** 2026-11-28 10:11
- **Người thực hiện:** Đinh Thái Toàn
- **Work Package liên quan:** WP03 / WP06
- **Nhánh Git (Branch):** `fix/BPM25_T0305_wp03-training-audit`
- **Commit SHA hiện tại:** `9605ade`
- **Trạng thái nghiệm thu:** Ready for Review (HALT FOR REVIEW — Chờ duyệt phương án khắc phục)

---

## 1. Mục tiêu & Phạm vi công việc (Scope & Objectives)
- **Mục tiêu kỹ thuật:** Thực hiện kiểm toán toàn diện Tầng 1 đối với dự án `DL-PM25-BJ` (`course_project/`) trước khi xử lý dữ liệu và tinh chỉnh mô hình, bao gồm 4 hạng mục:
  1. Quản trị cấu hình tập trung (`Config Decoupling` & `configs/config.yaml` / `configs/default.yaml`).
  2. Cơ chế cố định ngẫu nhiên & tính tái lập (`Multi-Seed Reproducibility`).
  3. Môi trường thực thi & thư viện (`Environment & Dependencies`).
  4. Phân tách tài nguyên & tình trạng Git (`Git & Artifacts Separation`).
- **Ranh giới thực thi (In-Scope / Out-of-Scope):**
  - *Đã thực hiện (In-Scope):* Đọc, đối chiếu chéo mã nguồn, kiểm tra môi trường Python/PyTorch, kiểm tra trạng thái Git và xuất báo cáo kiểm toán kèm nhật ký phiên.
  - *Không bao gồm (Out-of-Scope):* Chưa sửa đổi mã nguồn, chưa chỉnh sửa `.gitignore`, chưa chạy huấn luyện hay commit thay đổi.

---

## 2. Đối chiếu tiêu chí hoàn thành (Definition of Done - DoD Verification)

| STT | Hạng mục kiểm toán Tầng 1 | Kết quả | Tóm tắt phát hiện chính |
| :---: | :--- | :---: | :--- |
| 1 | Quản trị Cấu hình Tập trung (`Config Decoupling`) | [!] Phát hiện lệch | Thiếu `configs/config.yaml` (đang dùng `configs/default.yaml`), thiếu `scripts/evaluate.sh` (đang dùng `scripts/ablation.sh`); còn nhiều tham số hardcode và key YAML chưa được code đọc. |
| 2 | Cố định Ngẫu nhiên & Tái lập (`Multi-Seed`) | [!] Phát hiện lệch | `training_loop.py:287` thiếu `cudnn.deterministic=True` & `cudnn.benchmark=False`; `DataLoader` không gắn `Generator`; `fine_tune.py` bỏ qua `seed` khi đọc/lưu checkpoint WP03. |
| 3 | Môi trường Thực thi & Thư viện | [!] Phát hiện thiếu | `requirements.txt` trống (0 bytes); PyTorch là bản `2.5.1+cpu` (không có CUDA); thiếu `captum`, `streamlit`, `onnxruntime`. |
| 4 | Phân tách Tài nguyên & Tình trạng Git | [!] Cảnh báo nghiêm trọng | Git đang track nhầm nhiều file `.pt`, `.csv` (`data/preprocess/`, `outputs/`), `.pkl` và `__pycache__/*.pyc`; `data/preprocess/train.csv` chưa tồn tại trên đĩa. |

---

## 3. Chi tiết kết quả kiểm toán 4 hạng mục

### 3.1. Bước 1: Quản trị Cấu hình Tập trung (`Config Decoupling`)

1. **Tên file cấu hình & Script:**
   - `configs/config.yaml` **không tồn tại**; hệ thống đang sử dụng `configs/default.yaml` (`training_loop.py:306,321`, `scripts/train.sh:16`, `scripts/ablation.sh:15`, `src/xai/ig_explainer.py:478`).
   - `scripts/evaluate.sh` **không tồn tại**; hiện có `scripts/train.sh`, `scripts/ablation.sh`, `scripts/download_data.py`.

2. **Đồng bộ tham số cơ bản:**
   - `lookback = 96`: Đồng bộ giữa `configs/default.yaml:32`, `training_loop.py:1756`, `fine_tune.py:1076`, `src/data_pipeline/dataset.py:9,60`.
   - `horizons = [24, 48, 96, 192]`: Trong `configs/default.yaml:33`, `data.horizon` chỉ đặt mặc định `24`; danh sách 4 mốc nằm ở `ablation.all_horizons` (`configs/default.yaml:200`) và viết cứng tại `src/data_pipeline/dataset.py:61`, `run_ablation.py:35`, `utils_metrics.py:39`.
   - `split_ratios [0.6, 0.1, 0.1, 0.2]`: **Không có** trong `configs/default.yaml`. `src/data_pipeline/preprocess.py:68-89` dùng mốc ngày cố định (`2015-12-31 23:00:00`, `2016-04-30 23:00:00`, `2016-08-31 23:00:00`); `train_baseline.py:327` dùng chia 60/20/20 cho dữ liệu synthetic.
   - `num_stations`: Khớp `12` trạm UCI PRSA trong `configs/default.yaml:29,54-66` (không dùng 36 trạm).
   - `num_features`: `configs/default.yaml:28,36-53` và `src/data_pipeline/preprocess.py:14-56` định nghĩa `D=31` cho `multivariate` (11 biến liên tục + 4 biến chu kỳ `sin/cos` + 16 cột One-Hot hướng gió `wd_*`) và `D=1` cho `univariate`.
   - `target_col`: Khớp `"pm2.5"` (`configs/default.yaml:30`, `src/data_pipeline/preprocess.py:12`).

3. **Các điểm Hardcode phát hiện trong mã nguồn (`file:line`):**
   - `src/data_pipeline/preprocess.py:12-56`: Hardcode `TARGET_COL`, `CONTINUOUS_COLS`, `WIND_DIRECTIONS` (16 hướng).
   - `src/data_pipeline/preprocess.py:68-73`: Hardcode `train_end`, `train_dev_end`, `dev_end`.
   - `src/data_pipeline/preprocess.py:243-244`: Hardcode `--raw-dir data/raw` và `--out-dir data/preprocess` (không hỗ trợ `--config`).
   - `src/data_pipeline/dataset.py:8-10,59-61`: Hardcode `target_col="pm2.5"`, `lookback=96`, `horizon=24`, `horizons=(24, 48, 96, 192)`.
   - `training_loop.py:51-91`: Hardcode `ALL_SPLITS`, `DEFAULT_MODELS`, `ALL_SUPPORTED_MODELS`, `CANONICAL_MULTIVARIATE_COLS`.
   - `training_loop.py:141,213` & `src/models/patchtst.py:88`: Hardcode ngưỡng `11 if in_features >= 15 else in_features`.
   - `train_baseline.py:71,134-137,432`: Hardcode `learning_rate=1e-3`, `checkpoint_dir="checkpoints"`, `log_dir="outputs/logs"`, không nạp `configs/default.yaml`.
   - `run_ablation.py:35-36`: Hardcode `ALL_HORIZONS = [24, 48, 96, 192]`, `LONG_HORIZONS = [96, 192]`.
   - `src/xai/shap_explainer.py:496` & `src/xai/ig_explainer.py:488`: Hardcode đường dẫn fallback `"outputs/checkpoints/global/patchtst_multivariate_H24_seed42_pretrained.pt"`.
   - `scripts/train.sh:14-16` & `scripts/ablation.sh:14-15`: Hardcode `MODE="multivariate"`, `HORIZON="24"`, `CONFIG_PATH="configs/default.yaml"`.

4. **Độ lệch Key YAML:**
   - **Key có trong `configs/default.yaml` nhưng bị code bỏ qua (hoặc không có CLI dest tương ứng trong `apply_yaml_config`):**
     - `paths.global_ckpt_dir`, `paths.station_ckpt_dir`, `paths.wp02_ckpt_dir`, `paths.wp02_log_dir`, `paths.log_dir`, `paths.report_dir`, `paths.ablation_dir`, `paths.scaler_template` (`configs/default.yaml:9-16`) — `training_loop.py` và `fine_tune.py:804-812` tự ghép đường dẫn từ `output_dir` thay vì đọc các key này.
     - `data.stations`, `data.canonical_multivariate_cols`, `data.eval_stride`, `data.num_continuous_features`, `data.include_wind_onehot`, `data.target_col`, `data.target_idx` (`configs/default.yaml:30-66`).
     - `pretrain.all_supported_models`, `pretrain.eval_metric`, `pretrain.optimizer`, `pretrain.scheduler`, `pretrain.use_amp` (`configs/default.yaml:86-114`).
     - `finetune.freeze_backbone`, `finetune.scheduler`, `finetune.scheduler_factor`, `finetune.scheduler_patience` (`configs/default.yaml:126,136-138`).
     - Toàn bộ block `wp02_baseline.*` (`configs/default.yaml:68-78`) không được `train_baseline.py` đọc.
   - **Key code đang định nghĩa nhưng thiếu trong `configs/default.yaml`:**
     - `split_ratios` / mốc thời gian chia tập cho `preprocess.py`.
     - `train_dev_stride`, `dev_stride`, `test_stride` (`training_loop.py:1760-1762`, `fine_tune.py:1081-1083`).
     - `finetune.models` (khi `fine_tune.py` nạp section `finetune`).

---

### 3.2. Bước 2: Cơ chế Cố định Ngẫu nhiên & Tính Tái lập (`Multi-Seed Reproducibility`)

1. **Cấu hình Seeds:**
   - `configs/default.yaml:72,92` khai báo `seeds: [42, 1024, 2026]` cho `wp02_baseline` và `pretrain`.
   - `configs/default.yaml:129` (`finetune.seed: 42`) và `configs/default.yaml:24` (`xai.seed: 42`) chỉ chạy đơn seed.
   - `scripts/train.sh:85-98` truyền toàn bộ `--seeds` cho `training_loop.py`, nhưng chỉ lấy `first_seed` truyền `--seed` cho `fine_tune.py`.

2. **Hàm cố định ngẫu nhiên (`set_seed`):**
   - `training_loop.py:287-294` (được `fine_tune.py:45` tái sử dụng): Đã khóa 4 nguồn `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed(seed)` và `torch.cuda.manual_seed_all(seed)`. **Thiếu** `torch.backends.cudnn.deterministic = True` và `torch.backends.cudnn.benchmark = False`.
   - `train_baseline.py:37-51`: Đã khóa đủ cả 4 nguồn và thiết lập `cudnn.deterministic = True`, `cudnn.benchmark = False`.

3. **DataLoader & Đặt tên Checkpoint giữa Pre-train và Fine-tune:**
   - `DataLoader` tại `training_loop.py:596-604` và `fine_tune.py:97-105` đặt `num_workers=0`, `shuffle=(split_name == "train")`, nhưng không truyền `generator=torch.Generator().manual_seed(seed)`.
   - **Bất nhất tên Checkpoint:**
     - Pre-train (`training_loop.py:1288-1290`) lưu từng seed dạng `{model}_{mode}_H{horizon}_seed{seed}_pretrained.pt` và sao chép bản tốt nhất sang `{model}_{mode}_H{horizon}_pretrained.pt` (`training_loop.py:1578-1583`, không gắn `seed`).
     - Fine-tune (`fine_tune.py:356-357`) khi `ckpt_source == "wp03"` chỉ nạp `{model}_{mode}_H{horizon}_pretrained.pt` (bỏ qua `seed`), và lưu tại `fine_tune.py:422` dưới tên `{model}_{source_tag}_{mode}_H{horizon}_finetuned.pt` (**không gắn `seed`**).

---

### 3.3. Bước 3: Môi trường Thực thi & Thư viện (`Environment & Dependencies`)

1. **Thông số runtime hiện tại:**
   - **Python:** `3.11.1` (64-bit Windows)
   - **PyTorch:** `2.5.1+cpu` (`CUDA available: False`, `Device count: 0`)
   - **`requirements.txt`:** Tệp hiện **rỗng (0 bytes)**.

2. **Đối chiếu các gói đang cài đặt:**
   - **Đã cài đặt:** `pyyaml==6.0.2`, `pydantic==2.14.1`, `scikit-learn==1.5.2`, `pandas==2.2.3`, `numpy==2.1.3`, `torch==2.5.1+cpu`, `shap==0.51.0`, `plotly==7.1.0`, `joblib==1.5.3`, `tabulate==0.10.0`.
   - **Thiếu (`ModuleNotFoundError`):**
     - `captum` (cần cho `src/xai/ig_explainer.py`).
     - `streamlit` (cần cho `src/app/main.py`).
     - `onnxruntime` (cần cho kiểm thử/suy luận ONNX ở WP05).

---

### 3.4. Bước 4: Phân tách Tài nguyên & Tình trạng Git (`Git & Artifacts Separation`)

1. **Kiểm tra `.gitignore` (chỉ đọc):**
   - Đã chặn `__pycache__/`, `*.py[cod]`, `*.pth`, `*.pt`, `*.onnx`, `checkpoints/`, `data/raw/`, `data/preprocess/train.csv`, `!docs/`.
   - **Thiếu rule chặn:** `data/preprocess/dev.csv`, `data/preprocess/test.csv`, `data/preprocess/train_dev.csv`, `data/preprocess/standard_scaler.pkl`, `data/processed/*`, `outputs/`, `*.csv`, `*.log`.

2. **Trạng thái Git (`git status`, `git branch`, `git ls-files`):**
   - Nhánh hiện tại: `fix/BPM25_T0305_wp03-training-audit` (HEAD `9605ade`).
   - **Tệp bị track nhầm trong Git index (`git ls-files`):**
     - Dữ liệu & Scaler: `data/preprocess/dev.csv`, `data/preprocess/test.csv`, `data/preprocess/train_dev.csv`, `data/preprocess/standard_scaler.pkl`. (Ngoài ra 24 file trong `data/raw/` đang hiển thị trạng thái `D` trong `git status`).
     - Trọng số mô hình: Nhiều file `outputs/checkpoints/global/*.pt` và `outputs/checkpoints/stations/*/*.pt`.
     - Logs & Reports CSV: `outputs/logs/*.csv`, `outputs/reports/*.csv`, `outputs/ablation/*.csv`, `outputs/eda/*.csv`.
     - Cache Python: `__pycache__/*.pyc` và `src/**/__pycache__/*.pyc`.

3. **Tồn tại thư mục & quyền ghi:**
   - `data/raw/`: Tồn tại (có 12 file CSV và thư mục con `PRSA_Data_20130301-20170228`).
   - `data/processed/`: Tồn tại (chỉ có `.gitkeep`).
   - `data/preprocess/`: Tồn tại nhưng **thiếu `train.csv`** (chỉ có `dev.csv`, `test.csv`, `train_dev.csv`, `standard_scaler.pkl`).
   - `outputs/checkpoints/` và `outputs/logs/`: Tồn tại và có quyền ghi.

---

## 4. Bằng chứng thực thi kiểm toán (Verification Logs)

```text
Python: 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)]
PyTorch: 2.5.1+cpu CUDA available: False Device count: 0
yaml: 6.0.2 pydantic: 2.14.1 sklearn: 1.5.2 pandas: 2.2.3 numpy: 2.1.3
shap 0.51.0 | plotly 7.1.0 | joblib 1.5.3 | tabulate 0.10.0
captum MISSING: No module named 'captum'
streamlit MISSING: No module named 'streamlit'
onnxruntime MISSING: No module named 'onnxruntime'

Git branch: fix/BPM25_T0305_wp03-training-audit (HEAD 9605ade)
Test-Path data/preprocess/train.csv -> False
```

---

## 5. Đề xuất phương án khắc phục (Chờ duyệt trước khi thực thi)
1. **Cấu hình:** Chuẩn hóa tên tệp cấu hình (`configs/config.yaml` hoặc đồng bộ tài liệu sang `configs/default.yaml`), đưa `split_ratios`/mốc ngày vào YAML để `preprocess.py` đọc trực tiếp, và nối các key `paths.*`, `data.*` còn thiếu vào `training_loop.py` / `fine_tune.py`.
2. **Reproducibility:** Bổ sung `torch.backends.cudnn.deterministic = True` và `torch.backends.cudnn.benchmark = False` vào `training_loop.py:set_seed()`, gắn `seed` vào tên checkpoint fine-tune khi cần chạy multi-seed.
3. **Môi trường:** Xin phép cập nhật `requirements.txt` đầy đủ các gói chuẩn của dự án và cài đặt `captum`, `streamlit`, `onnxruntime` (cùng bản PyTorch CUDA nếu chạy trên GPU).
4. **Git Hygiene:** Sau khi được phép, gỡ bỏ các file `.pt`, `.csv`, `.pkl`, `__pycache__` khỏi Git index (`git rm --cached`) và bổ sung rule chặn `data/preprocess/`, `outputs/` vào `.gitignore`.

