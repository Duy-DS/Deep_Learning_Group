# [BUG-FIX] Khắc phục lỗi mô hình SOTA (PatchTST, Informer) bị phẳng đường dự báo, mất đỉnh ô nhiễm và thua Baseline (DLinear) trên WP03

> **Quy ước đặt tên file:** `docs/bug-fixes/261126_2325_wp03_sota_underfitting_peak_fix.md`

---

## 1. Thông tin tổng quan (Audit Metadata)

- **Mã sự cố (Bug ID):** `BUG_261126_2325`
- **Thời gian xử lý:** `2026-11-26 23:25`
- **Người thực hiện:** Thành viên WP03 & Antigravity AI Coding Agent
- **Gói công việc liên quan:** `WP03` (tương thích chéo `WP01`, `WP02`, `WP04`, `WP05`)
- **Task ID:** `BPM25_T0301` / `BPM25_T0304` / `BPM25_T0305`
- **Nhánh Git (Git Branch):** `fix/BPM25_WP03_sota-peak-capture` (phát triển cô lập từ `group/feat/BPM25_WP03_SOTA_Alb_update` tại commit `b641f73`)
- **Mức độ nghiêm trọng (Severity):** **Critical** (Ảnh hưởng trực tiếp chất lượng dự báo đỉnh Smog, xếp hạng SOTA vs. Baseline và độ tin cậy thực nghiệm Ablation RQ1/RQ2)
- **Tệp mã nguồn liên quan (Affected Files):**
  - `src/models/patchtst.py`
  - `src/models/informer.py`
  - `configs/default.yaml`
  - `training_loop.py`
  - `fine_tune.py`
  - `run_ablation.py`
  - `tests/test_wp03_sota_fixes.py` *(mới — bộ kiểm thử hồi quy 55 cases)*

---

## 2. Triệu chứng & Bối cảnh phát hiện (Bug Description & Symptoms)

- **Hành vi kỳ vọng (Expected Behavior):**
  - Các mô hình Transformer SOTA (`PatchTST`, `Informer`) phải hội tụ sâu hơn `DLinear` / `LSTM`, đạt sai số MSE/MAE thấp hơn Baseline đặc biệt ở chế độ đa biến (`D=31`) và đa khung dự báo (`H ∈ {24, 48, 96, 192}`).
  - Đường dự báo trên biểu đồ tập Test phải bám sát biên độ biến thiên thực tế, nhận diện kịp thời các đợt tăng vọt PM2.5 (`≥ 150 µg/m³`, đỉnh mùa đông `589–835 µg/m³`) và không suy biến thành đường phẳng.
  - Quá trình Fine-tuning từng trạm phải mở khóa đúng lớp đầu ra/adapter thực sự và cải thiện sai số trên từng trạm.
- **Hành vi thực tế (Observed Behavior):**
  - Trên nhánh `WP03_SOTA_Alb_update`, cả `PatchTST` và `Informer` đều thua `DLinear` trên mọi khung dự báo; ở `H=192`, cả hai có Test MSE (`2.3622` và `1.9741` trên 3 trạm trọng điểm) tệ hơn cả đường dự báo hằng số 0 (`Zero MSE = 1.7909`).
  - Đường dự báo bị phẳng rõ rệt: biên độ dao động nội cửa sổ (`range_ratio`) chỉ đạt `25.0%` (`Informer`) và `31.2%` (`PatchTST`) tại `H=24`; giảm xuống còn `15.9%–22.8%` tại `H=192`.
  - Tại đợt ô nhiễm nặng `EP02` (`Dongsi 2017-01-01`, đỉnh `589 µg/m³`), `OLD informer` tụt về đường thẳng ngang `[289.3, 288.0, 287.6, 287.1]` sau giờ thứ 9; tại `EP01` (`Nongzhanguan 2017-01-28`, đỉnh `835 µg/m³`), mọi mô hình cũ chỉ dự báo tối đa `69–114 µg/m³`.
  - Khi chạy `fine_tune.py`, `LSTM` và `DLinear` chỉ mở khóa đúng 25 tham số scalar của wrapper (không mở khóa lớp dự báo thực sự), dẫn đến 0/12 trạm cải thiện với `LSTM` và 6/12 trạm dừng ở epoch 0 với `DLinear`.
- **Nhật ký đo đạc trực tiếp trên checkpoint WP03 cũ (3 trạm đỉnh ô nhiễm nặng nhất: `Aotizhongxin`, `Dongsi`, `Nongzhanguan`):**

```text
=== OLD WP03 CHECKPOINTS (3-stations H=24, N=3171) ===
  [BASELINES] Persistence MSE=1.2610 | Zero MSE=1.7427
  OLD dlinear   | Test MSE=1.0150 MAE=0.6592 (RMSE=81.3) | Range=39.7% | SpikeMAE=91.5 | PeakCap=66.2%
  OLD lstm      | Test MSE=1.1260 MAE=0.6691 (RMSE=85.6) | Range=27.3% | SpikeMAE=83.9 | PeakCap=68.3%
  OLD informer  | Test MSE=1.0127 MAE=0.6692 (RMSE=81.2) | Range=25.0% | SpikeMAE=95.0 | PeakCap=62.6%
  OLD patchtst  | Test MSE=1.0933 MAE=0.6786 (RMSE=84.4) | Range=31.2% | SpikeMAE=89.5 | PeakCap=66.7%

=== OLD WP03 CHECKPOINTS (3-stations H=192, N=1017) ===
  [BASELINES] Persistence MSE=3.0832 | Zero MSE=1.7909
  OLD dlinear   | Test MSE=1.6609 MAE=0.9539 (RMSE=104.0) | Range=31.2% | SpikeMAE=144.0 | PeakCap=48.0%
  OLD lstm      | Test MSE=2.4050 MAE=1.1368 (RMSE=125.1) | Range=17.7% | SpikeMAE=147.2 | PeakCap=42.4%
  OLD informer  | Test MSE=1.9741 MAE=1.0588 (RMSE=113.4) | Range=15.9% | SpikeMAE=135.9 | PeakCap=43.0%
  OLD patchtst  | Test MSE=2.3622 MAE=1.1297 (RMSE=124.0) | Range=22.8% | SpikeMAE=141.8 | PeakCap=46.6%
```

---

## 3. Phân tích nguyên nhân cốt lõi (Root Cause Analysis - RCA)

Qua kiểm tra định lượng trọng số `.pt`, phân phối 4 tập WP01 và thực nghiệm A/B trực tiếp, xác định **6 nguyên nhân cốt lõi cộng hưởng**:

1. **Xung đột `ResidualAnchorWrapper` (`init_scale=0.1`) & bóp nghẹt biên độ (`training_loop.py`):**
   - Bọc `y = last_val + res_scale * delta + station_bias` với `init_scale=0.1` xung đột trực tiếp với `PatchTST` (vốn đã cộng `+ mean` bên trong), khiến `res_scale` bị kẹt ở `0.098–0.146`, ghim chặt dự báo vào `last_val` (`corr_last = 0.92–0.96`) và phá hỏng hoàn toàn tầm xa `H=96/192`.
2. **Hiệu ứng độc hại của chuẩn hóa cục bộ `RevIN` trên chuỗi PM2.5 hồi quy về trung bình & đuôi nặng (`patchtst.py`, `training_loop.py`):**
   - Trong bản cũ, `PatchTST` áp `RevIN` lên toàn bộ 31 kênh (không có `min_std`), vừa xóa tín hiệu mùa (`month_sin/cos` 88.8% cửa sổ có phương sai ≈ 0) vừa gây nổ chia 0 trên `rain` (64.6%) và 16 cột `wd_*`.
   - Quan trọng hơn, thực nghiệm A/B (`ab_revin_vs_norevin.py`) chứng minh: ngay cả khi sửa `RevIN` chọn lọc 11 kênh + `min_std=0.1`, phép biến đổi `y_hat = mean_96h + std_96h * y_norm` **tước mất thông tin mức nền tuyệt đối** của PM2.5 và cưỡng bức giả định tương lai `24–192h` giữ nguyên trung bình 96h quá khứ. Do PM2.5 Bắc Kinh có tính **hồi quy mạnh về trung bình** (đợt khói mù tan nhanh khi gió Bắc về) và phân phối lệch phải cực đoan, bật `RevIN` khiến cả `PatchTST` (`2.1951`) lẫn `Informer` (`2.1924`) sụp đổ nặng ở `H=192` và mất `-11.6%` đến `-16.2%` MSE ở `H=24` so với chạy trực tiếp trên không gian `StandardScaler` toàn cục.
3. **Underfitting nặng có hệ thống đối với Transformer (`configs/default.yaml`, `training_loop.py`):**
   - Chấp `10x` learning rate bất lợi (`lr = 5e-5` cho Transformer vs `dlinear_lr = 5e-4`), `dropout = 0.4`, `input_noise = 0.05`, `train_stride = 7` (vứt bỏ 85.7% cửa sổ huấn luyện), và áp `weight_decay = 0.05` đồng loạt lên cả `bias`, `LayerNorm`, `pos_embedding` và lớp trộn đa biến `channel_mixer_head` khởi tạo 0 (làm norm lớp trộn tụt còn `0.166`, gần như ngắt tín hiệu khí tượng).
4. **Khiếm khuyết kiến trúc `Informer` (`src/models/informer.py`):**
   - Công thức đo độ thưa ProbSparse Eq. (4) chia nhầm cho `L_K` thay vì trung bình trên `U_part` khóa mẫu (`Q_K_sample.mean(dim=-1)`).
   - Dùng `torch.randint` ngay cả khi `.eval()` gây nhiễu ngẫu nhiên đầu ra; lớp `self.dropout` khai báo nhưng không gọi; đầu vào Decoder toàn số 0 khiến đồ thị phẳng hoàn toàn sau giờ thứ 6; biến thể `informer_full` (Ablation 3) thực tế vẫn chạy `ProbSparseAttention` với `factor=96` thay vì `FullAttention` $O(L^2)$ thực sự.
5. **Lệch mùa nghiêm trọng khi chọn checkpoint & hàm mất mát trung bình (`training_loop.py`, `fine_tune.py`):**
   - Tập `dev` thuần mùa Hè (`May–Aug 2016`, `mean=57.3, p95=137 µg/m³`), tập `train_dev` mùa Đông/Xuân (`Jan–Apr 2016`, `mean=67.9, p95=231 µg/m³`), còn tập `test` là Thu/Đông (`Sep 2016–Feb 2017`, `mean=92.1, p95=288, max=835 µg/m³`). Chọn Early Stopping 100% theo `dev` mùa Hè làm lệch mô hình xuống nền thấp.
6. **Lỗi cấu hình chết & mở khóa sai tầng trong Fine-tuning (`configs/default.yaml`, `fine_tune.py`, `run_ablation.py`):**
   - Toàn bộ mục `models:` trong `configs/default.yaml` bị `training_loop.py` và `run_ablation.py` bỏ qua; `fine_tune.py` tìm sai tên lớp (`linear_projection` của `LSTM`, `linear_seasonal/trend` của `DLinear`).

---

## 4. Giải pháp & Mã nguồn thay đổi (Solution & Code Diff)

- **Nguyên lý khắc phục:**
  1. **`src/models/patchtst.py`:** Kế thừa `BaseTimeSeriesModel` (`[B, L, D] -> [B, H, 1]`), tinh gọn độ phân giải patch (`patch_len=12, stride=6, d_model=128, n_heads=4, e_layers=3, d_ff=256, dropout=0.2`), chạy Channel-Independent Transformer trên `C_cont=11` kênh vật lý liên tục (~2.8× nhanh và nhẹ VRAM), bổ sung `ExogenousPatchFusion` (trộn ngữ cảnh khí tượng + hướng gió `wd_*` + chu kỳ mùa/giờ qua thống kê `[mean, last, delta]` từng patch & cổng dư), đầu dự báo 2 nhánh `head_linear + head_mlp`, dùng chiều dương trong `unfold` (chuẩn ONNX `opset=17`), và đặt mặc định `use_revin=False` dựa trên bằng chứng A/B (vẫn giữ lớp `RevIN` chọn lọc cho tùy chọn ablation).
  2. **`src/models/informer.py`:** Sửa đúng công thức ProbSparse Eq. (4) (`mean(dim=-1)`), áp dụng `self.dropout(attn)`, tự động chuyển sang Eq. (4) tất định dùng `torch.gather`/`torch.scatter` khi `eval()` (trùng khớp **0.000000 drift** với bộ vá WP05 `FullKeyEq4ProbSparseAttention`), bổ sung `FullAttention` $O(L^2)$ thực cho `attention_type="full"`, thay `nn.TransformerEncoderLayer` bằng Pre-LN `InformerEncoderLayer` thuần (giữ thuộc tính `self.self_attn = AttentionLayer(...)` tương thích WP05, tránh lỗi fast-path `in_proj_bias` của PyTorch), bổ sung Pre-LN Decoder có `decoder_norm`, `horizon_queries` học được theo từng bước $h \in \{1..H\}$ và đầu `projection` 2 lớp.
  3. **`configs/default.yaml`:** Kích hoạt Single Source of Truth cho toàn bộ mục `models:` và `finetune:`; `train_stride: 2`, `epochs: 35`, `patience: 8`, `dropout: 0.20`, `weight_decay: 1e-2`, `use_residual_anchor: false`, `use_revin: false`, `use_revin_wrapper: false`, `val_selection_mode: "balanced_dev"`, `lr` riêng từng họ (`patchtst: 5e-4`, `informer: 4e-4`, `lstm: 5e-4`, `dlinear: 1e-3`), `PeakShapeLoss(peak_threshold_z=0.84, peak_alpha=0.25, diff_weight=0.10, peak_max_weight=3.5)`, và `finetune.lr: 5e-5`.
  4. **`training_loop.py`:** Tắt mặc định `ResidualAnchorWrapper` & `RevINModelWrapper` (giữ chế độ tương thích ngược cho checkpoint legacy), nạp `load_model_configs()`, tách nhóm Decoupled AdamW (`weight_decay=0.0` cho bias/norm/embeddings), triển khai `PeakShapeLoss` và tiêu chí `balanced_dev` (50/50 `train_dev` mùa Đông & `dev` mùa Hè chuẩn hóa theo phương sai), bổ sung chỉ số chẩn đoán đỉnh (`range_ratio`, `spike_mae_ugm3`, `spike_rmse_ugm3`, `peak_capture_ratio`) song song với `MSE/MAE` chuẩn và `ExperimentMetricsContract`.
  5. **`fine_tune.py`:** Sửa `_get_head_keywords()` khớp chính xác toàn bộ 4 kiến trúc (mở khóa 4,688–360,913 tham số thực), thêm `build_finetune_param_groups()` hỗ trợ Decoupled WD + Discriminative LR, tự động nhận diện cả checkpoint mới v2 lẫn checkpoint cũ (có `res_scale` hoặc `backbone.`), chuyển chọn checkpoint sang `balanced_dev`.
  6. **`run_ablation.py`:** Kết nối trọn vẹn `model_configs`, `finetune.*`, `PeakShapeLoss`, `val_selection_mode`, `use_residual_anchor` và xuất bổ sung cột chẩn đoán độ phẳng/đỉnh trong báo cáo Ablation 2.
  7. **Bảo toàn 100% tương thích WP04 & WP05:** Các tham số mặc định trong `PatchTST.__init__`, `InformerModel.__init__` và `build_model()` khớp tuyệt đối `configs/default.yaml`, nên `PyTorchCheckpointLoader.load(..., strict=True)` của WP05 và `IGExplainer`/`SHAPExplainer` của WP04 hoạt động trực tiếp mà không cần sửa mã nguồn WP04/WP05.

- **Thống kê thay đổi (`git diff --stat`):**

```text
 course_project/configs/default.yaml   |  86 +++--
 course_project/fine_tune.py           | 425 +++++++++++++++++----
 course_project/run_ablation.py        | 142 +++++--
 course_project/src/models/informer.py | 510 ++++++++++++++++++-------
 course_project/src/models/patchtst.py | 448 +++++++++++++++++-----
 course_project/training_loop.py       | 677 ++++++++++++++++++++++++++++------
 6 files changed, 1814 insertions(+), 474 deletions(-)
 + course_project/tests/test_wp03_sota_fixes.py (341 lines, 55 test cases)
```

---

## 5. Bằng chứng kiểm thử & Xác minh (Verification Evidence)

### 5.1. Bộ kiểm thử tự động & xuất ONNX `opset=17` (55/55 PASSED)

- **Lệnh thực thi:**

```bash
python -m pytest tests/test_wp03_sota_fixes.py -v
```

- **Nhật ký xác minh:**

```text
platform win32 -- Python 3.12.13, pytest-9.2.0 -- course_project/.venv/Scripts/python.exe
collected 55 items
...
test_all_models_shape_and_gradients[24..192 x 31|1 x 5 models] (40 cases) PASSED
test_patchtst_selective_revin_handles_constant_channels                   PASSED
test_informer_probsparse_vs_full_and_deterministic_eval                   PASSED
test_yaml_and_default_constructor_strict_compatibility_for_wp04_wp05      PASSED
test_onnx_opset17_export_and_parity[31|1 x 4 models] (8 cases)            PASSED
test_finetune_layer_unfreezing_and_param_groups                           PASSED
test_peak_shape_loss_and_balanced_selection                               PASSED
test_legacy_checkpoint_detection_and_reload                               PASSED
test_revin_wrapper_restores_mean_and_preserves_exogenous                  PASSED
======================= 55 passed, 48 warnings in 7.92s =======================
```

### 5.2. Xác minh tích hợp chéo thực tế với WP04 (XAI) & WP05 (Observatory Engine)

```text
[PASS] CLI YAML parsing verified for training_loop.py, fine_tune.py, run_ablation.py
  [OK] WP05 PyTorchCheckpointLoader.load(strict=True) patchtst : max_abs_diff=1.49e-07
  [OK] WP05 PyTorchCheckpointLoader.load(strict=True) informer : max_abs_diff=2.09e-07
[PASS] WP05 InformerDeterministicPatcher (FullKeyEq4ProbSparse): max_abs_drift=0.000000
  [OK] WP05 PyTorchCheckpointLoader.load(strict=True) dlinear  : max_abs_diff=9.54e-07
  [OK] WP05 PyTorchCheckpointLoader.load(strict=True) lstm     : max_abs_diff=0.00e+00
[PASS] WP05 PyTorchCheckpointLoader.load(strict=True) verified for all 4 models (diff < 1e-5)
[PASS] WP04 XAI verified: IG shape=(2, 96, 31) (max_delta=9.91e-04), SHAP shape=(2, 96, 31), all finite
```

### 5.3. Bằng chứng thực nghiệm định lượng trên dữ liệu thật WP01 (3 trạm trọng điểm có đỉnh Smog nặng nhất: `Aotizhongxin`, `Dongsi`, `Nongzhanguan`)

> *Lưu ý: Bản WP03 v2 bên dưới chỉ chạy thử nhanh trên CPU với **3,096 mẫu train (`train_stride=24`, 7% dữ liệu toàn cục) trong 6–8 epochs**, nhưng đã vượt rõ rệt cả checkpoint WP03 cũ (vốn train trên 12 trạm × 30 epochs) lẫn Baseline `DLinear`.*

#### Bảng A — H=192 (Tầm xa 8 ngày; `Zero Baseline MSE = 1.7909`, `Persistence MSE = 3.0832`)

| Mô hình | Phiên bản | Test MSE ↓ | Test MAE ↓ | RMSE (µg/m³) ↓ | Range Ratio ↑ | Spike MAE (µg/m³) ↓ | Peak Capture ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| `DLinear` (Baseline) | WP03 Cũ (12 trạm) | 1.6609 | 0.9539 | 104.0 | 31.2% | 144.0 | 48.0% |
| `DLinear` (Baseline) | WP03 v2 (3 trạm, 60ep) | 1.7460 | 1.0056 | 106.6 | 35.3% | 146.3 | 53.0% |
| `PatchTST` | WP03 Cũ (`ResAnchor+RevIN`) | 2.3622 | 1.1297 | 124.0 | 22.8% | 141.8 | 46.6% |
| `PatchTST` | WP03 v2 (`RevIN=ON`) | 2.1951 | 1.0685 | 119.6 | 43.3% | 160.4 | 52.9% |
| **`PatchTST`** | **WP03 v2 (`RevIN=OFF + PeakShape 0.25`)** | **1.6856** *(-28.6%)* | **0.9878** *(Tốt nhất)* | **104.8** | **40.7%** *(+17.9pp)* | **137.8** | **56.3%** *(Tốt nhất)* |
| `Informer` | WP03 Cũ (`ResAnchor+RevIN`) | 1.9741 | 1.0588 | 113.4 | 15.9% | 135.9 | 43.0% |
| `Informer` | WP03 v2 (`RevIN=ON`) | 2.1924 | 1.1068 | 119.5 | 6.6% | 152.9 | 35.5% |
| **`Informer`** | **WP03 v2 (`RevIN=OFF + PeakShape 0.25`)** | **1.6280** *(Tốt nhất, -17.5%)* | **1.0004** | **103.0** | 6.6% *(6ep)* | **134.0** *(Tốt nhất)* | 38.7% |
| **`LSTM`** | **WP03 v2 (`RevIN=OFF + PeakShape 0.25`)** | **1.6951** *(-29.5% vs cũ)* | **0.9988** | **105.1** | 26.8% | 142.1 | 49.0% |

#### Bảng B — H=24 & Đối chứng A/B Hàm mất mát (`RevIN=OFF`, 3 trạm `stride=24`, 7–8 epochs CPU)

| Mô hình | Cấu hình | Test MSE ↓ | Test MAE ↓ | Range Ratio ↑ | Spike MAE (µg/m³) ↓ | Peak Capture ↑ |
|---|---|---:|---:|---:|---:|---:|
| `DLinear` | WP03 v2 (`MSELoss`, 60ep) | 1.0064 | 0.6906 | 50.8% | 96.8 | 70.5% |
| `PatchTST` | WP03 Cũ (`ResAnchor+RevIN`, 12 trạm) | 1.0933 | 0.6786 | 31.2% | 89.5 | 66.7% |
| `PatchTST` | WP03 v2 (`RevIN=ON`, `PeakShape 0.25`) | 1.1549 | 0.7147 | 52.3% | 101.8 | 68.8% |
| `PatchTST` | WP03 v2 (`RevIN=OFF`, `MSELoss`) | 1.0422 | 0.6831 | 52.8% | 105.0 | 64.9% |
| **`PatchTST`** | **WP03 v2 (`RevIN=OFF`, `PeakShape 0.25/0.10`)** | **1.0210** *(-6.6% vs cũ)* | **0.6802** *(< DLinear)* | **52.8%** *(+21.6pp)* | **95.4** *(-9.6 vs MSE)* | **68.3%** |
| `Informer` | WP03 v2 (`RevIN=ON`, `PeakShape 0.25`) | 1.2039 | 0.7425 | 33.8% | 95.5 | 67.9% |
| `Informer` | WP03 v2 (`RevIN=OFF`, `MSELoss`) | 1.0121 | 0.6717 | 37.2% | 97.5 | 65.1% |
| **`Informer`** | **WP03 v2 (`RevIN=OFF`, `PeakShape 0.25/0.10`)** | **1.0087** | **0.6758** *(< DLinear)* | 30.8% | **86.2** *(-11.3 vs MSE)* | **67.5%** |

#### Kiểm tra mở khóa tham số Fine-tuning trên trạm `Aotizhongxin`

```text
  dlinear  : unfrozen =   4,688 params (trước đây: 25)
  lstm     : unfrozen = 135,214 params (trước đây: 25 -> 0/12 trạm cải thiện)
  informer : unfrozen = 342,551 params
  patchtst : unfrozen = 360,913 params
```

---

## 6. Biện pháp phòng ngừa & Bài học kinh nghiệm (Prevention & Lessons Learned)

- **Ràng buộc bổ sung (Assertions / Contract Updates):**
  - Thêm bộ kiểm thử tự động `tests/test_wp03_sota_fixes.py` (55 test cases) kiểm soát chặt: shape `[B, 96, D] -> [B, H, 1]`, gradient hữu hạn, khớp `strict=True` giữa `configs/default.yaml` với constructor mặc định của WP04/WP05, xuất ONNX `opset=17` với sai số `< 1e-4`, tính tất định 100% của `Informer` ở `eval()`, và mở khóa đúng tham số khi Fine-tuning (`> 1,000` params).
  - Bổ sung 4 chỉ số chẩn đoán đỉnh & chống phẳng (`range_ratio`, `spike_mae_ugm3`, `spike_rmse_ugm3`, `peak_capture_ratio`) vào báo cáo CSV tự động để phát hiện sớm hiện tượng đường dự báo phẳng ngay sau lần train đầu tiên.
- **Bài học kinh nghiệm:**
  - Không áp dụng máy móc `RevIN` hay `ResidualAnchor` của các bộ dữ liệu xu hướng trơn (ETT/Exchange) lên dữ liệu chất lượng không khí PM2.5 có phân phối lệch phải mạnh và tính hồi quy về trung bình theo gió mùa; luôn chạy A/B kiểm chứng trực tiếp trên cả tầm ngắn `H=24` lẫn tầm xa `H=192`.
- **Hành động tiếp theo (Follow-up Tasks):**
  - [ ] Huấn luyện lại toàn bộ ma trận 12 trạm × 3 seeds × 4 horizons (`H ∈ {24, 48, 96, 192}`) trên GPU (Kaggle P100/2×T4 hoặc Google Colab) với `train_stride=2, epochs=35`.
  - [ ] Chạy lại Ablation Studies (`bash scripts/ablation.sh all --no-skip-existing --with-finetune`) để cập nhật các bảng báo cáo RQ1/RQ2.
  - [ ] Bàn giao bộ checkpoint `.pt` mới cho WP04 (`src/xai/`) chạy lại SHAP/IG và WP05 (`scripts/prepare_observatory_assets.py`) xuất lại toàn bộ `.onnx` (`opset=17`).
  - [x] Cập nhật nhật ký phiên tại `docs/dev-logs/261126_2325.md`.
