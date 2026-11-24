# 📄 Post-Execution Report: Task T0103

- **Date**: 2026-11-22
- **Target Module**: `legacy_keras/`, `scripts/extract_keras_baselines.py`, `experiments/csv_logs/`
- **Task Goal**: Kiểm kê 10 notebook Keras legacy và trích xuất baseline kiến trúc/hyperparameter/metrics đã lưu để đồng bộ bảng đối chiếu.
- **Author / Executor**: `nguyen-cong-tri` (giá trị `MEMBER_03_NAME` trong `.env`; email không ghi trong report)
- **Execution Engine**: Python 3.11.9, JSON notebook parser, pytest 9.1.1; môi trường hiện tại không có TensorFlow/Keras hay GPU.
- **Overall Result**: PASSED WITH LIMITATIONS

---

## 1. 📊 EXECUTIVE SUMMARY & ACCEPTANCE MATRIX

Đã xác nhận đủ 10 notebook trong 5 nhóm và tạo CSV 10 dòng từ `model.summary()`/execution outputs đã lưu bên trong notebook. Không thực thi lại Keras: TensorFlow/Keras không cài trong môi trường này và không có GPU. Các trường không xuất hiện trong notebook output được ghi `NOT_RECORDED`; không thay thế bằng số liệu suy đoán.

| Deliverable / Requirement | Component File | Status | Tests Executed | Passed / Failed | Details |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Kiểm kê 10 bài gốc | `legacy_keras/{01..05}_*/` | PASSED | Notebook inventory | 10 / 0 | 2 MLP, 3 CNN+MLP, 1 CNN+LSTM, 2 Training Utilities, 2 Others. Không sửa nội dung legacy. |
| Trích xuất params/metrics/runtime | `scripts/extract_keras_baselines.py` | PASSED WITH LIMITATIONS | Parser test + đối chiếu cell outputs | 2 / 0 | Lấy summary/history/evaluate/runtime output đã lưu; mỗi hàng có notebook path và cell index provenance. |
| CSV cho Google Sheets | `experiments/csv_logs/keras_baseline_10_models.csv` | PASSED | Parse CSV / 10-row assertion | 2 / 0 | UTF-8 BOM, header chuẩn hóa; cột metric/split tách riêng. CSV được whitelist khỏi ignore `*.csv`. |
| Hướng dẫn kiến trúc baseline | `docs/HUONG_DAN_GIAI_THICH_CODE.md` | PASSED | Diff/content review | 1 / 0 | Thêm mục T0103 về 5 nhóm kiến trúc và cảnh báo đọc kết quả. |
| Hướng dẫn rerun Colab | `experiments/colab/run_keras_baseline_t4.py` | PASSED WITH LIMITATIONS | Compile/source review | 1 / 0 | Helper chỉ kiểm tra runtime và hướng dẫn; không tự chạy/sửa notebook gốc. |
| Chạy lại 10 notebook | Keras runtime | NOT RUN | Environment probe | 0 / 1 not applicable | TensorFlow/Keras không cài, GPU device không có; đây không phải thất bại kiểm thử, nhưng DoD rerun chưa hoàn tất. |

### Ma trận baseline trích xuất

`Train/Val` dưới đây là dòng cuối của training log khi notebook đã lưu. Cột “Evaluate” giữ nguyên số được cell cuối in dưới nhãn `Test`; split thật có ghi trong CSV. Thời gian là chuỗi runtime notebook đã in, không gắn với GPU T4 vì metadata thiết bị không có.

| ID / Nhóm | Input NHWC → Output | Params (Total / Trainable / Non-trainable) | Optimizer · LR · Loss · Batch · Epoch | Train cuối (loss / metric) | Val cuối (loss / metric) | Evaluate đã lưu (loss / metric) | Runtime đã in |
|---|---|---:|---|---|---|---|---|
| EXP-01 · MLP `Func_apxm_MLP` | `(2,) → (None,1)` | `NOT_RECORDED` (công thức `4B+1`; 65 chỉ khi B=16) | SGD · 0.01 · MSE · 128 · sweep C | `NOT_RECORDED` | `NOT_RECORDED` | `NOT_RECORDED` | `NOT_RECORDED` |
| EXP-02 · MLP `MNIST_MLP` | `(28,28) → (None,10)` | 669,706 / 669,706 / 0 | SGD · 0.01 · categorical CE · 128 · 30 | 0.1332 / 0.9612 acc | 0.1143 / 0.9648 acc | 0.1142885 / 0.9648000 acc | 67.872 s |
| EXP-03 · CNN `1_MNIST_CNN` | `(28,28,1) → (None,10)` | 1,817,610 / 1,817,610 / 0 | SGD · 0.01 · categorical CE · 128 · 30 | 0.0627 / 0.9803 acc | 0.0339 / 0.9884 acc | 0.0339137 / 0.9884000 acc | 203.202 s |
| EXP-04 · WIKI Gender basic | `(64,64,1) → (None,2)` | 4,063,570 / 4,063,570 / 0 | Adam · 0.0001 · categorical CE · 128 · 30 | 0.2590 / 0.8977 acc | 0.4602 / 0.8264 acc | 0.4602156 / 0.8264075 acc | 176.112 s |
| EXP-05 · FER basic | `(48,48,1) → (None,7)` | 19,028,287 / 19,028,287 / 0 | Adam · 0.0001 · categorical CE · 128 · 15 | 0.3848 / 0.8660 acc | 1.4070 / 0.5954 acc | 1.4070247 / 0.5954305 acc | 366.720 s |
| EXP-06 · CNN+LSTM | `(8,28,28,1) → (None,1)` | 1,747,073 / 1,747,073 / 0 | SGD · 0.01 · MAE · 128 · 30 | 0.9357 / `NOT_RECORDED` | 0.6502 / `NOT_RECORDED` | `NOT_RECORDED` | 1,597.602 s |
| EXP-07 · WIKI callbacks | `(64,64,1) → (None,2)` | 4,063,570 / 4,063,570 / 0 | Adam · 0.001 · categorical CE · 128 · max 100000 | 0.2580 / 0.9063 acc | 0.3725 / 0.8760 acc | 0.3237280 / 0.8773458 acc | 191.274 s |
| EXP-08 · FER callbacks | `(48,48,1) → (None,7)` | 19,028,287 / 19,028,287 / 0 | Adam · 0.0001 · categorical CE · 128 · max 100 | 1.0828 / 0.5898 acc | 1.0158 / 0.6163 acc | 1.0158261 / 0.6163277 acc | 1,364.934 s |
| EXP-09 · SegNet | `(96,96,1) → (None,9216,7)` | 29,458,211 / 29,442,325 / 15,886 | Adam · 0.0001 · categorical CE · 8 · 15 | 0.3496 / 0.9014 acc | 0.6860 / 0.7732 acc | 0.6860425 / 0.7732083 acc | 750.025 s |
| EXP-10 · FCN Object Detection | `(64,64,3) → (None,1)` | 83,367 / 83,367 / 0 | adam · `NOT_RECORDED` · MSE · 128 · 5 | 0.0793 / 0.9391 acc | 0.0642 / 0.9523 acc | 0.0641823 / 0.9523273 acc | `NOT_RECORDED` |

Các số chính xác, split, params provenance, cell index và ghi chú metric nằm trong CSV. EXP-10 `accuracy` là metric Keras áp lên score/regression target; **không phải IoU/mAP**. EXP-09 `accuracy` là metric pixel classification của Keras; được giữ nguyên tên thay vì tự đổi thành metric khác.

---

## 2. 🛡️ GUARDRAILS & INVARIANT VALIDATION

- [x] Đủ 10 notebook: số file và thứ tự được đối chiếu với năm thư mục legacy.
- [x] Không chạy hoặc sửa notebook gốc; script chỉ parse JSON/source/output cells.
- [x] Không ghi metric thiếu thành 0; trường không có bằng chứng là `NOT_RECORDED`.
- [x] Ghi provenance: notebook path, cell `model.summary`, epoch output, evaluate output, compile/fit cell.
- [x] CSV được allowlist trong `.gitignore`; không stage dataset, checkpoint, `.env` hoặc thư mục `outputs/`.
- [x] Định danh báo cáo lấy từ `.env` `MEMBER_03_NAME`; email không được chép vào báo cáo. Commit phải dùng cặp Member_03 theo `.env`.
- [ ] Chạy lại Keras baseline trên GPU Colab T4: chưa thực hiện trong môi trường hiện tại.
- [ ] Tính so sánh CPU vs T4: chưa thể thực hiện; notebook outputs không ghi nhận hardware model.
- [ ] Charter markdown theo tên `Project_Charter_BTL03.md`: không có file này. Đã dùng `AGENTS.md`, CSV phân công và notebook legacy. PDF Project Charter tồn tại nhưng không trích phần WBS trong bước này.
- [ ] Phân công WBS: CSV ghi Member_03 phụ trách CNN+LSTM/visualization, trong khi prompt giao T0103 cho Member_03 để tổng hợp cả 10 baseline. Report làm theo chỉ định trực tiếp, ghi rõ khác biệt này.

---

## 3. 🧪 DETAILED ACCEPTANCE FINDINGS

### A. Functional Implementation

- `scripts/extract_keras_baselines.py` định nghĩa mười mục baseline, đọc output notebook mà không import TensorFlow/Keras hoặc chạy cell, ghi CSV bằng UTF-8 BOM và giữ provenance cell.
- `experiments/csv_logs/keras_baseline_10_models.csv` có 10 dòng, tổng hợp shape, params, hyperparameters, train/val cuối epoch, evaluate output, runtime và split ghi nhận.
- `experiments/colab/run_keras_baseline_t4.py` là helper kiểm tra runtime/hướng dẫn; không tự cài dependencies, không tải dữ liệu và không thực thi notebook. Bản sao notebook cần được chạy riêng trên Colab T4 sau khi thiết lập dataset.
- `docs/HUONG_DAN_GIAI_THICH_CODE.md` đã thêm phụ lục mô tả MLP, CNN, CNN+LSTM, utilities, SegNet và FCN detection.

**Các điểm kỹ thuật cần giữ khi chuyển đổi sang PyTorch (WP02 và các WP tiếp theo):**

1. Keras ảnh dùng NHWC; PyTorch `Conv2d` dùng NCHW. WIKI phải `np.rot90(k=3, axes=(1,2))` trước permute, theo đúng notebook.
2. MNIST_MLP đầu vào `(28,28)`; MNIST CNN `(28,28,1)`. Kiểm tra flatten order, initializer, padding, pooling và Dropout.
3. WIKI basic và callback là hai baseline khác nhau. Callback variant thêm ModelCheckpoint/EarlyStopping/ReduceLROnPlateau/TensorBoard; không gộp hai kết quả.
4. FER basic dùng 15 epoch; callback variant cấu hình max 100 nhưng output dừng epoch 57. Public-test CSV được dùng làm validation trong notebook, do đó metric cuối không đại diện independent test.
5. CNN-LSTM có sequence length 8, `TimeDistributed` bọc Conv/Pool/Dropout/Flatten và 2 LSTM 128. PyTorch thay bằng gộp `(B*T,C,H,W)` qua CNN rồi tách `(B,T,F)`; notebook chỉ lưu `val_loss`, không lưu test MAE cuối.
6. SegNet dùng grayscale 96×96, 7 nhãn one-hot/pixel, 5 lần pooling/upsampling. Output `accuracy` là metric pixel classification legacy; lớp BatchNorm có 15,886 non-trainable params.
7. Object Detection notebook huấn luyện score map với `x/127.5-1`, Conv2D 8×8 và 1×1, `tanh`, MSE. `accuracy` in trong notebook không phải IoU/mAP; LR không được ghi rõ.

### B. Testing & Quality Gates

- `python scripts/extract_keras_baselines.py` — tạo 10 dòng; không thực thi Keras.
- `python -m pytest tests/test_baseline_extraction.py -q` — **2 passed**.
- Kiểm tra thủ công các output `model.summary`, cuối log epoch, test cell và thời gian trong 9 notebook có output; EXP-01 không có numeric output.
- Cần chạy `python scripts/extract_keras_baselines.py` từ thư mục `BTL03/` sau khi cập nhật notebook outputs để tái tạo CSV.
- Không tuyên bố đã chạy lint toàn repository; chưa có linter được cài/khai báo cho task này.

---

## 4. ⏱️ PERFORMANCE & EXECUTION METRICS

- **Số liệu baseline**: 9 notebook có bảng `model.summary`; EXP-01 không có output params, nên CSV để `NOT_RECORDED` và ghi công thức `4B+1` (sweep B thay đổi; 65 chỉ tại B=16).
- **Runtime legacy đã in**: 8 bài có con số runtime trong notebook (khoảng 67.872–1,597.602 giây); EXP-01 và Object Detection không in runtime.
- **Hardware**: không có device metadata trong notebook outputs; không gán các thời gian này cho CPU hay GPU T4 và không suy luận so sánh thiết bị.
- **Môi trường phân tích hiện tại**: Python 3.11; TensorFlow/Keras không cài; GPU devices không có.

---

## 5. 🚩 OUTSTANDING ISSUES & RECOMMENDATIONS

- **Current limitations**:
  - Đây là inventory của execution outputs đã lưu trong notebook, không phải lượt chạy mới trên môi trường hiện tại. Một số output cũ không ghi package version, hardware, seed hoặc checksum dữ liệu.
  - `Func_apxm_MLP` không lưu summary, train/test error số cụ thể hay runtime; `B` biến thiên nên không có một parameter count đơn nhất.
  - CNN-LSTM không có evaluate/test output; không ghi test MAE.
  - MNIST_MLP và MNIST_CNN truyền `x_test/y_test` vào `validation_data`, rồi evaluate lại trên cùng split; các con số được giữ vì có output, nhưng không phải test holdout độc lập theo protocol hiện đại.
  - WIKI/FER callback và basic dùng split public/validation cho validation và cell evaluate; không gọi đó là independent holdout.
  - Không kết luận overfitting chỉ từ train–val gap khác thời điểm/metric; tuy nhiên FER basic có train accuracy cao hơn val trong log cuối, cần kiểm tra lại với split và best checkpoint khi rerun.
  - Chưa có bằng chứng runtime Colab T4; hướng dẫn T4 chỉ là chuẩn bị rerun.
- **Recommendations for next task**: Chạy từng notebook/bản sao trên Google Colab T4 với dataset được cấp phép; lưu runtime type, Python/TensorFlow/Keras version, GPU model, data checksum, seed, full stdout/history và evaluate split. Đối với CNN-LSTM, thêm evaluate có kiểm soát sau khi xác nhận test protocol; không sửa notebook legacy gốc. Chuẩn hóa lại WBS để phân biệt “người tổng hợp baseline 10 model” và owner chuyên trách từng model group.