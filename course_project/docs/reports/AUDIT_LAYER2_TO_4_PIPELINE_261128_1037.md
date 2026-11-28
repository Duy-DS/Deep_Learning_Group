# Báo cáo Kiểm toán & Đồng bộ Tầng 2 – Tầng 4 (`BPM25_T0305_DATA_PIPELINE_AUDIT`)

- **Ngày thực hiện:** 2026-11-28
- **Người thực hiện:** Đinh Thái Toàn
- **Nhánh Git:** `fix/BPM25_T0305_wp03-training-audit`
- **Trạng thái:** Đã hoàn tất kiểm toán & vá lỗi Tầng 2–4; **ĐANG CHỜ DUYỆT** trước khi chạy huấn luyện hàng loạt.

---

## 1. Tái tạo Dữ liệu Tiền xử lý & Xác minh Chống Data Leakage (Tầng 2 & Tầng 3)

### 1.1. Vá lỗi chuẩn hóa tên cột & đồng bộ `standard_scaler.pkl` trong `src/data_pipeline/preprocess.py`
- **Vấn đề phát hiện:**
  1. Các tệp CSV thô chuẩn UCI PRSA (`data/raw/PRSA_Data_*.csv`) dùng tên cột viết hoa (`PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`, `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`), gây `KeyError` khi `causal_impute_splits()` truy xuất `CONTINUOUS_COLS` viết thường.
  2. Tệp `data/preprocess/standard_scaler.pkl` cũ trên ổ đĩa là artifact lệch phiên bản (`sklearn 1.9.1`, `n_features_in_ = 30`, `mean_[0] = 108.3531` không khớp tập `train.csv` `D=31` có `mean = 82.1942`), và `train.csv` trước đó bị thiếu.
- **Khắc phục:**
  - Chuẩn hóa tên cột về chữ thường trong `run_preprocess()` (giữ nguyên `station`, `wd`, `cbwd`, `datetime`).
  - Xuất kèm `data/preprocess/standard_scaler.pkl` khớp duy nhất trên `encoded["train"][CANONICAL_COLS]` (`D=31`), đồng thời giữ nguyên 4 tệp CSV chưa scale để `load_and_prepare_wp01_splits()` phục vụ cả `multivariate` (`D=31`) lẫn `univariate` (`D=1`) mà không bị Double-Scaling.

### 1.2. Kết quả kiểm chứng trên `data/preprocess/`
Chạy `python -m src.data_pipeline.preprocess --config configs/default.yaml`:
- **Số trạm:** Đủ **12 trạm** chuẩn UCI PRSA cho cả 4 phân vùng.
- **Kích thước & Ranh giới thời gian (tuyến tính, không chồng lấn):**
  - `train.csv`: `(298368, 33)` — `2013-03-01 00:00:00` đến `2015-12-31 23:00:00` (`<= 2015-12-31 23:00:00`)
  - `train_dev.csv`: `(34848, 33)` — `2016-01-01 00:00:00` đến `2016-04-30 23:00:00` (`<= 2016-04-30 23:00:00`)
  - `dev.csv`: `(35424, 33)` — `2016-05-01 00:00:00` đến `2016-08-31 23:00:00` (`<= 2016-08-31 23:00:00`)
  - `test.csv`: `(52128, 33)` — `2016-09-01 00:00:00` đến `2017-02-28 23:00:00` (`> 2016-08-31 23:00:00`)
- **Hợp đồng đặc trưng:** Đúng `D = 31` cột (`15` cột cơ sở `canonical_multivariate_cols` trong `configs/default.yaml` + `16` cột `wd_*` khi `include_wind_onehot: true`), cột mục tiêu `pm2.5` nằm tại `target_idx = 0`, không tồn tại `NaN`/`Inf`.
- **Kiểm tra `standard_scaler.pkl`:** `n_features_in_ == 31`, `scaler.mean_[0] = 82.1942` khớp tuyệt đối với `train['pm2.5'].mean() = 82.1942` (sai số `< 1e-5`).
- **Perturbation Test (trong `verify_causal_preprocessing`):** Đạt chuẩn nhân quả 100%, không rò rỉ tương lai.

---

## 2. Kiểm toán Hợp đồng Tensor, Không gian Độ đo & Stride (Tầng 3 & Tầng 4)

### 2.1. Hợp đồng Tensor `[B, L, D] -> [B, H, 1]`
- `src/data_pipeline/dataset.py::make_sliding_windows()` tạo `X` shape `[N, 96, D]` và `y` shape `[N, H]`.
- Toàn bộ 5 biến thể mô hình (`lstm`, `dlinear`, `informer`, `informer_full`, `patchtst`) đều xuất `[B, H, 1]`.
- **SOTA (WP03 — `training_loop.py::evaluate`):** Ép `preds = model(batch_x).squeeze(-1)` để tính loss `[B, H]`, sau đó khôi phục `[N, H, 1]` trước khi gọi `MetricsCalculator.compute(all_preds, all_targets)`.
- **Baseline (WP02 — `training_loop.py::_to_3d_target_loader` & `train_baseline.py::BaselineTrainer.evaluate_test`):** Chuyển `y` sang `[N, H, 1]` và truyền `[N, H, 1]` trực tiếp vào `MetricsCalculator.compute()`.

### 2.2. Tính nhất quán Không gian Độ đo (Metric Space)
- `utils_metrics.py::MetricsCalculator.compute()` tính `MSE`, `RMSE`, `MAE` trực tiếp trên tensor đầu vào, **không** thực hiện inverse transform.
- Cả WP02 (`BaselineTrainer`) và WP03 (`evaluate()` trong `training_loop.py` và `fine_tune.py`) đều dùng dữ liệu từ `load_and_prepare_wp01_splits()` đã chuẩn hóa bằng `StandardScaler` khớp trên `train`.
- Do đó, **cả WP02 lẫn WP03 đều tính `MSE`/`MAE` chính thức trên không gian Z-score**, đảm bảo so sánh công bằng 1-1. Ngoài ra, `evaluate()` của WP03 có tính thêm chỉ số chẩn đoán đỉnh theo đơn vị vật lý (`spike_mae_ugm3`, `spike_rmse_ugm3`, `peak_capture_ratio`) khi truyền `target_mean` và `target_scale`.

### 2.3. Đồng bộ `eval_stride` và `drop_last`
- `training_loop.py::prepare_global_12stations_data()` dùng `current_stride = train_stride if split_name == "train" else 1` (`eval_stride = 1` cho `train_dev`, `dev`, `test`) và `drop_last = False`.
- `fine_tune.py::build_single_station_loaders()` dùng `current_stride = stride if split_name == "train" else 1` và `drop_last = False`.
- `training_loop.py::_to_3d_target_loader()` cho WP02 Baseline tái sử dụng trực tiếp `loaders["dev"]`, `loaders["test"]` với `drop_last = False`.
- Kết luận: **100% đồng nhất** về cửa sổ đánh giá giữa WP02 và WP03.

---

## 3. Vá lỗi Đóng băng/Mở khóa Tầng `Informer` trong `fine_tune.py` (Tầng 4)

- **Nguyên nhân gốc:** Trong `fine_tune.py::_get_head_keywords()`, mục `"informer"` và `"informer_full"` chứa từ khóa `"projection"`. Do `configure_trainable_layers()` kiểm tra theo kiểu chuỗi con (`key in param_name.lower()`), từ khóa `"projection"` vô tình mở khóa toàn bộ `query_projection`, `key_projection`, `value_projection` và `out_projection` bên trong các lớp Attention của Encoder/Decoder, dẫn đến nguy cơ *catastrophic forgetting* khi fine-tune trên dữ liệu 1 trạm.
- **Bản vá (`fine_tune.py`):** Thay `"projection"` bằng `"projection.0"` và `"projection.3"` (khớp chính xác 2 lớp `nn.Linear` của `self.projection = nn.Sequential(...)` ở đầu ra của `InformerModel`).
- **Kiểm thử đơn vị bổ sung (`tests/test_wp03_sota_fixes.py`):** Thêm `test_informer_attention_projections_frozen_during_finetune` (tham số hóa cho cả `informer` và `informer_full`) xác minh `query_projection`, `key_projection`, `value_projection` giữ `requires_grad=False`, trong khi `projection.0`, `projection.3`, `horizon_queries` được mở khóa (`requires_grad=True`).

---

## 4. Kết quả Chạy Kiểm thử (`pytest`)

Lệnh thực thi: `python -m pytest tests/test_wp03_sota_fixes.py -v --tb=short`
- **Kết quả:** Toàn bộ **57/57 test cases PASSED** (bao gồm 55 test gốc và 2 test mới cho `informer` & `informer_full`).

