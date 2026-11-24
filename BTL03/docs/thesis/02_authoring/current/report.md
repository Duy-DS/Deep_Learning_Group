# BÁO CÁO TỔNG KẾT BTL03 — SNAPSHOT T0402

> **Bản thảo review theo style v2, chưa phải bản nộp cuối.** Snapshot repository: base `main` tại `b2b88db3fc7444f7c37391e7211416ea3843a31e`, ngày 2026-11-24. Report thực nghiệm tổng thể mới nhất trong repository là T0212 (2026-11-23). WBS workbook giao T0402 cho Member 05; nhóm trưởng chỉ định trực tiếp Member 01 tổng hợp bản này. Đây là mâu thuẫn owner cần xác nhận trước nghiệm thu chính thức. Nội dung chỉ tổng hợp evidence đã lưu, không phải lần chạy mới.

## Thông tin đề tài và nhóm

- **Đề tài:** Bài tập lớn 03 — Chuyển đổi mã nguồn Deep Learning từ Keras sang PyTorch và phân tích thực nghiệm.
- **Mã dự án:** `BTL03`.
- **Người tổng hợp snapshot này:** `Duy-DS` (Member 01; xác nhận trong `.env` và task evidence).
- **Thành viên/alias trong task reports:** `Duy-DS`, `nguyenthanhdat180905-cloud`, `nguyen-cong-tri`, `TonOfWeight072501`, `CHIVY2005`.
- **Họ tên pháp lý, MSSV, trường/khoa/lớp, giảng viên, tỷ lệ đóng góp cuối kỳ:** **[NHÓM CẦN XÁC NHẬN; không suy đoán từ alias hoặc mẫu]**.
- **Phân công T0402:** workbook `Quan_Ly_Tasks` ghi Member 05; chỉ định trực tiếp hiện tại giao Member 01. Bản thảo làm theo chỉ định trực tiếp; không sửa WBS.

## Tóm tắt

BTL03 chuyển mười cấu hình thuộc năm nhóm bài toán từ notebook Keras sang PyTorch OOP. Báo cáo trình bày theo mạch framework/data contract → kiến trúc/source → kiểm toán → kết quả có Run ID → discussion/validity. Evidence hiện có gồm baseline Keras trích từ notebook, 36 cấu hình sweep PyTorch T0201, một MNIST MLP run T0202 và kiểm toán T0212. T0212 ghi 10/10 synthetic one-epoch train/validation cases PASS, nhưng logger bị BLOCKED bởi protobuf/TensorBoard và full test suite theo shards có 78 passed, 4 failed trên 82; trạng thái tổng thể **BLOCKED**. Điều này không chứng minh mười mô hình đã full-train trên dữ liệu thật. Parameter count CNN-LSTM còn bất đồng giữa T0206 và T0207/T0212.

**Từ khóa:** Keras; PyTorch; NCHW; reproducibility; evidence-first; parameter audit.

## Danh mục viết tắt

| Viết tắt | Ý nghĩa |
|---|---|
| CNN | Convolutional Neural Network — mạng tích chập |
| FCN | Score-map regression theo legacy; không mặc nhiên là detector có IoU/mAP |
| LSTM | Long Short-Term Memory |
| NCHW / NHWC | Batch–channel–height–width / batch–height–width–channel |
| OOP | Lập trình hướng đối tượng |
| RQ / WBS | Câu hỏi nghiên cứu / cấu trúc phân rã công việc |

## Câu hỏi nghiên cứu

README BTL03 nêu RQ1/RQ2 nhưng `Project_Charter_BTL03.md` đúng tên chưa có trong repository. Bảng này là tóm lược theo README, chưa xác nhận câu chữ nguyên văn charter; không tự tạo RQ3/RQ4.

| ID | Câu hỏi tóm lược | Evidence cần | Trạng thái |
|---|---|---|---|
| RQ1* | NHWC/NCHW, vòng lặp và khởi tạo ảnh ảnh hưởng thế nào đến hội tụ/gradient? | Đối chứng cùng split, seed, initializer, optimizer và runtime | Chưa kết luận tổng quát |
| RQ2* | Regularization, augmentation và callbacks ảnh hưởng ra sao đến overfitting trên dữ liệu thật? | Ablation đối chứng có dữ liệu, split, Run ID và log | Chưa kết luận hiệu quả PyTorch |
| RQ3/RQ4 | Chưa xác minh câu chữ charter | Charter chính thức | Không tự dựng câu hỏi |

\* Cần nhóm xác minh với charter chính thức.

## Mở đầu

### Bối cảnh và mục tiêu

Mục tiêu kỹ thuật là chuyển các notebook Keras sang các thành phần PyTorch có dataset, model, trainer, callbacks, cấu hình và runner tách biệt, đồng thời lưu provenance để kiểm toán. Mục tiêu thiết kế không đồng nghĩa kết quả đã đạt.

### Đối tượng, phạm vi và phương pháp

Đối tượng gồm mười cấu hình trong năm nhóm. Snapshot tài liệu lấy từ base commit `b2b88db3fc7444f7c37391e7211416ea3843a31e`; reports T0101–T0212, baseline/epoch/master CSV, configs và source được đối chiếu. Không chạy lại mô hình, không nội suy metric thiếu, không xem synthetic smoke là thực nghiệm dữ liệu thật.

### Đóng góp đã xác minh và giới hạn

Repository có code chuyển đổi, baseline inventory, task reports, T0201 sweep, T0202 run và T0212 smoke/audit. T0212 overall vẫn `BLOCKED`; một số nhóm thiếu dữ liệu/full training. Họ tên pháp lý, MSSV, tỷ lệ đóng góp và RQ nguyên văn cần nhóm xác nhận.

### Cấu trúc báo cáo

Chương 1 nêu phạm vi; Chương 2 giải thích hợp đồng Keras–PyTorch; Chương 3 trình bày pipeline và OOP; Chương 4 tổng hợp parameter/results/status; Chương 5 bàn về validity, hạn chế và hướng tiếp theo; phụ lục dẫn provenance.

# 1. Tổng quan dự án và phạm vi

## 1.1 Năm nhóm, mười cấu hình

| Nhóm | Cấu hình | Số cấu hình | Trọng tâm chuyển đổi |
|---|---|---:|---|
| 1 — MLP | `FuncApxmMLP`, `MNISTMLP` | 2 | Dense→Linear; MSE và logits classification |
| 2 — CNN+MLP | `MNISTCNN`, `HumanGenderCNN`, `EmotionCNN` | 3 | Conv2D→Conv2d; layout/preprocessing/pooling |
| 3 — CNN+LSTM | `MNISTCNNLSTM` | 1 | Reshape thay `TimeDistributed` |
| 4 — Training Utilities | Gender + callbacks; Emotion + callbacks | 2 | Callbacks/checkpoint/logger; dùng lại model nhóm 2 |
| 5 — Advanced Vision | `SegNet`, `FCNObjectDetector` | 2 | Segmentation logits/mask và score-map |

Hai cấu hình utilities dùng lại GenderCNN/EmotionCNN; mười cấu hình không có nghĩa mười kiến trúc độc lập.

## 1.2 Trạng thái theo snapshot

- T0101 xây dựng hệ thống Markdown/LaTeX; PDF Overleaf cuối chưa có evidence được duyệt.
- T0102–T0106 có reports về pipeline, baseline, logger và core OOP; từng report nêu giới hạn runtime/data.
- T0201 có 36 PyTorch sweep configs với CSV/Run ID; Keras numeric MSE cho EXP-01 `NOT_RECORDED`.
- T0202 có một MNIST MLP run 30 epoch; không đại diện chín cấu hình còn lại.
- T0203–T0211 có implementation/tests/parity evidence; nhiều full-data runs chưa được xác nhận hoặc bị blocker.
- T0212 có 10/10 synthetic one-epoch smoke cases PASS, logger BLOCKED và 78/4 test result theo shards trên 82; overall `BLOCKED`.
- Trong reports đã rà soát chưa có bằng chứng nghiệm thu full training mọi model, ablation hoàn chỉnh, demo cuối hoặc PDF. Ghi trạng thái là **chưa có evidence trong repository đã kiểm tra**, không suy diễn hoạt động ngoài repository.

# 2. Cơ sở lý thuyết và hợp đồng chuyển đổi

## 2.1 Vòng lặp, logits và loss

Keras thường dùng `model.fit`; PyTorch thể hiện `zero_grad → forward → loss → backward → optimizer.step`. Validation dùng `eval()` và inference/no-gradient mode. Classification đưa logits thô cùng nhãn class-index `torch.long` vào `CrossEntropyLoss`, không áp softmax trước loss. Parameter count khớp không chứng minh output/gradient/metric parity.

## 2.2 NHWC/NCHW, WIKI rotation và normalization

Keras ảnh thường có shape `(N,H,W,C)`, PyTorch convolution nhận `(N,C,H,W)`:

\[
x_{NCHW}=\operatorname{permute}(x_{NHWC},0,3,1,2).\operatorname{contiguous}().
\]

WIKI áp `np.rot90(k=3, axes=(1,2))` trên NHWC trước khi đổi trục. Rotation đổi vị trí pixel; permute chỉ đổi thứ tự chiều. MNIST legacy chia pixel byte cho 255. T0102/T0205 ghi FER legacy không chia `/255` tường minh; giữ `preserve` theo source. FCN detection chuẩn hóa RGB bằng `x/127.5 - 1.0`.

## 2.3 CNN-LSTM thay TimeDistributed

Với tensor `(B,T,C,H,W)`, gộp `(B,T)` thành `(BT,C,H,W)` để CNN xử lý từng frame; vector `(BT,F)` được khôi phục về `(B,T,F)` đúng thứ tự trước LSTM. Đây là hợp đồng shape tương ứng `TimeDistributed(CNN)`, không tự chứng minh weight hay metric parity.

## 2.4 SegNet, upsampling và mask loss

Nearest `UpSampling2D` được ánh xạ sang `nn.Upsample(scale_factor=2, mode="nearest")`. SegNet trả logits `(B,K,H,W)`; mask class-index `(B,H,W)` kiểu `long` dùng với cross-entropy. Pixel accuracy khác mIoU; giữ đúng tên metric legacy.

## 2.5 FCN score-map

Notebook detector dùng score `tanh`, target ±1 và MSE. Accuracy legacy không phải IoU/mAP. Heatmap từ model chưa huấn luyện chỉ kiểm tra inference/shape, không đo chất lượng detection.

# 3. Thiết kế hệ thống và phương pháp chuyển đổi

## 3.1 Kiến trúc OOP

- `BaseMigratedModel`: forward contract, NCHW validation, parameter counters và feature hooks.
- `BaseTrainer`: train/validation, callbacks, logger/checkpoint; validation chạy inference mode.
- `seed_everything(42)` và worker/generator helpers hỗ trợ reproducibility; `PYTHONHASHSEED` cần đặt trước khi khởi chạy process nếu cần cố định hash seed của process hiện tại.
- `src/engine/callbacks.py`: lifecycle, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
- `ExperimentLogger`: epoch CSV, master CSV 18 cột, TensorBoard; `CheckpointManager`: state/metadata và mirror Drive khi mount root có thật.
- YAML/runners/`src/train.py` dispatch CLI. `--task all --smoke-test` là synthetic harness, không phải full training.

T0104 report ghi logger smoke độc lập PASS; T0212 ghi SummaryWriter BLOCKED bởi protobuf/TensorBoard. Hai trạng thái thuộc hai mốc/runtime khác nhau, không triệt tiêu nhau.

## 3.2 Pipeline được kiểm toán

```text
Keras notebooks / baseline
  → Dataset & preprocessing (NHWC, normalization, labels)
  → PyTorch models (5 nhóm / 10 cấu hình)
  → BaseTrainer / callbacks / logger / checkpoint
  → YAML / CLI / tests / reports
```

Đây là sơ đồ quy trình khái quát, không khẳng định mọi cấu hình đã chạy end-to-end trên dữ liệu thật.

## 3.3 Ma trận model/component

| Nhóm/cấu hình | Kỹ thuật | Evidence | Giới hạn |
|---|---|---|---|
| `FuncApxmMLP` | `Linear(2,B)→Sigmoid→Linear(B,1)→Sigmoid`; `4B+1` | T0201, 36 sweep | Keras numeric MSE thiếu |
| `MNISTMLP` | Flatten 784, Dense 512×2, 10 logits | T0202, 669,706 params, 30 epochs | Một run; split khác legacy |
| `MNISTCNN` | Conv/pool, Flatten 2304, Dense 512×2 | 1,817,610 params | Full PyTorch metric chưa xác nhận |
| `HumanGenderCNN` | 5 block; WIKI rotation trước NCHW | 4,063,570 params | WIKI thật/PyTorch full metric chưa có report |
| `EmotionCNN` | FER 48×48, 7 logits, legacy preprocessing | 19,028,287 params | FER thật/PyTorch full metric chưa có report |
| `MNISTCNNLSTM` | Gộp/tách batch-time; 2 LSTM | T0206/T0207 count lệch 1,024 | Full MNIST MAE chưa xác nhận |
| Gender/Emotion callbacks | Callback lifecycle/checkpoint/logging | Code/tests reports T0208–T0209 | Full run/logger blocked hoặc thiếu data |
| `SegNet` | 5-stage encoder/decoder; nearest upsample | 29,458,211 tổng | Chưa full train data thật |
| `FCNObjectDetector` | `tanh` score-map, Conv 8×8/1×1 | 83,367 params | Chưa có metric vehicle data thật |

# 4. Thực nghiệm, kiểm toán và thảo luận

## 4.1 Parameter-count audit

| Cấu hình | Keras params | PyTorch/report | Diễn giải |
|---|---:|---:|---|
| `FuncApxmMLP`, B=16 | 65* | 65 | Keras summary số không lưu; 65 suy từ `4B+1`, chỉ tại B=16 |
| `MNISTMLP` | 669,706 | 669,706 | Count khớp |
| `MNISTCNN` | 1,817,610 | 1,817,610 | Count khớp |
| `HumanGenderCNN` | 4,063,570 | 4,063,570 | Count khớp |
| `EmotionCNN` | 19,028,287 | 19,028,287 | Count khớp |
| `MNISTCNNLSTM` | 1,747,073 | T0206: 1,746,049; T0207/T0212: 1,747,073 | **CẦN ĐỐI CHIẾU**, lệch 1,024 |
| Gender + callbacks | 4,063,570 | 4,063,570 | Model nhóm 2 dùng lại |
| Emotion + callbacks | 19,028,287 | 19,028,287 | Model nhóm 2 dùng lại |
| `SegNet` | 29,458,211 | 29,458,211 | Report ghi 29,442,325 trainable + 15,886 non-trainable |
| `FCNObjectDetector` | 83,367 | 83,367 | Count khớp |

`*` Parameter-count không chứng minh output/gradient/metric parity. CNN-LSTM bất đồng nguồn phải được giải quyết bằng source snapshot và counting semantics, không chọn số theo đa số.

## 4.2 Evidence results

| Nguồn | Quan sát có provenance | Caveat |
|---|---|---|
| T0201 sweep | 36 configs A/B/C; seed 42, SGD 0.01, batch 128. Test-grid MSE 0.02343764 tại A=100; 0.00683120 tại A=1000. | Keras numeric MSE `NOT_RECORDED`; không tuyên bố parity định lượng. |
| T0202 `T0202_MNISTMLP_s42_20261122_132545` | 30 epochs; accuracy 0.95870, loss 0.13292970, 150.24952 giây CPU theo report. | Keras 0.9648 dùng test split cho validation/evaluate; protocol khác. |
| Keras EXP-03/EXP-05 | CSV baseline lưu MNIST CNN accuracy 0.9884; FER train 0.8660, val 0.5954, val loss cuối 1.4070. | Legacy outputs, không phải PyTorch run; validation/public-test reuse. |
| T0212 smoke | 10/10 synthetic one-epoch train/validation cases PASS; parameter audit report khớp 10/10. | Chỉ xác nhận software wiring/shape, không phải kết quả data thật. |
| T0212 logger/suite | Logger 10/10 BLOCKED; 78 pass, 4 fail trên 82 test theo shards. | T0212 overall `BLOCKED`, không PASS toàn phần. |

## 4.3 Task status tại evidence snapshot

| Hạng mục | Trạng thái trong reports | Điều kiện/bước tiếp |
|---|---|---|
| T0201 | 36 sweep PyTorch có CSV/Run ID; Keras MSE thiếu | Chạy Keras đối chứng cùng protocol nếu cần parity |
| T0202 | Một MNIST MLP run 30 epoch | Giữ caveat split; không suy rộng |
| T0203–T0205 | Code/count/shape evidence; full-data metrics chưa xác nhận | Data thật, split/config/runtime, log/report |
| T0206–T0207 | Dataset/model tests; full MAE chưa xác nhận, count discrepancy | Audit source/counting semantics rồi chạy đúng protocol |
| T0208–T0209 | Callback code/tests; logger/data blockers | SummaryWriter tương thích và WIKI/FER thật |
| T0210–T0211 | Architecture/shape/count; thiếu full-data metrics | Data thật và metric đúng nghĩa |
| T0212 | **BLOCKED**; logger blocked, 4 suite failures theo shards | Sửa dependency trong môi trường được duyệt, rerun gates |
| Full training, ablation, RQ, visual analytics | **[CHƯA CÓ BẰNG CHỨNG ĐẦY ĐỦ TRONG REPOSITORY ĐÃ KIỂM TRA]** | Run thật, Run ID, split/seed/config/log/checkpoint/report |
| Demo, PDF và final acceptance | `NOT RECORDED / NOT RUN` trong reports rà soát | WBS/member confirmation và Overleaf review |
| T0402 ownership | Workbook: Member 05; chỉ định trực tiếp hiện tại: Member 01 | Nhóm xác nhận owner trước nghiệm thu chính thức |

## 4.4 Validity matrix

| Evidence class | Hỗ trợ kết luận | Không hỗ trợ kết luận |
|---|---|---|
| Source/config | Implementation/cấu hình được khai báo | Đã chạy hoặc đạt mục tiêu |
| Parameter count | Count theo snapshot/counting rule | Weight/output/gradient/metric parity |
| Synthetic smoke | Shape/wiring/một epoch fixture | Generalization/convergence/data thật |
| Legacy notebook output | Keras output được lưu | PyTorch performance hoặc independent holdout khi split reuse |
| T0201/T0202 | Kết quả ở Run ID/config cụ thể | Benchmark đồng nhất mười cấu hình |
| T0212 test/logger | 78/4 theo shards; logger blocked | Overall PASS/acceptance |

## 5. Kết luận, validity threats và kiến nghị

### 5.1 Kết luận trong phạm vi snapshot

Repository có code chuyển đổi năm nhóm, task reports, baseline inventory, T0201 sweep, T0202 run và T0212 audit/smoke. Evidence này cho phép mô tả kiến trúc, parameter audit và một phần wiring; không chứng minh cả mười cấu hình full-train hoặc đạt baseline. T0212 vẫn `BLOCKED`; full training, ablation/RQ, demo cuối và PDF Overleaf chưa có evidence hoàn tất trong reports được rà soát.

### 5.2 Hạn chế và validity threats

1. Notebook legacy thiếu hardware/seed/checksum hoặc metric số; `NOT_RECORDED` không được thay bằng suy đoán.
2. Một số validation/public-test split được tái sử dụng; không gọi là independent holdout.
3. Reports T0203–T0211 ghi thiếu dữ liệu thật cho một số nhóm; CPU-only và dependency blocker giới hạn run.
4. Parameter count không chứng minh parity đầu ra; CNN-LSTM reports lệch 1,024 params.
5. Synthetic smoke không đo metric khoa học; T0212 logger BLOCKED và suite có 4 failures theo shards.
6. Workbook giao T0402 cho Member 05, còn chỉ định trực tiếp giao Member 01 soạn bản này; nhóm cần xác nhận trước bản cuối.
7. Họ tên/MSSV/giảng viên/tỷ lệ và RQ nguyên văn charter chưa được xác nhận.

### 5.3 Bài học và khuyến nghị

- Khóa split, seed, preprocessing, loss, optimizer và runtime trước khi so sánh framework.
- Gắn mỗi metric với task, source, Run ID, split và caveat; phân biệt legacy output/PyTorch run.
- Xử lý protobuf/TensorBoard như blocker môi trường; không giả lập SummaryWriter.
- Audit source/count semantics CNN-LSTM trước khi chốt parity.
- Trả lời RQ/ablation bằng run đối chứng thật và metric đúng domain.
- Nhóm xác nhận task owner và thông tin pháp lý, rồi review bundle và PDF trên Overleaf.

## Tài liệu tham khảo

1. Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 32. BibTeX: `paszke2019pytorch`.
2. Badrinarayanan, V., Kendall, A., & Cipolla, C. (2017). *SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation*. IEEE TPAMI, 39(12), 2481–2495. DOI: 10.1109/TPAMI.2016.2644615. BibTeX: `badrinarayanan2017segnet`.
3. Nhóm BTL03 (2026). *T0212: WP02 PR review, parameter audit, CLI and smoke-test report*. `docs/reports/T0212_261123_2130.md`, source commit `33dcd74599e86404bbc8128c8057afba82810093`. BibTeX: `btl03_t0212_internal`.
4. Task reports T0102–T0211, baseline/sweep/master/epoch CSV và `docs/thesis/01_materials/evidence/evidence_sources_manifest.md`.

## Phụ lục: evidence register

- WBS/rules: `AGENTS.md`, workbook trong `docs/project_charter/`.
- Baseline: `docs/reports/261127_T0103_keras_baseline_log.md`, `experiments/csv_logs/keras_baseline_10_models.csv`.
- T0201: `docs/reports/T0201_261122_2112.md`, `experiments/csv_logs/t0201_func_apxm_abc_sweep.csv`, epoch traces theo Run ID.
- T0202: `docs/reports/T0202_261122_2112.md`, Run ID `T0202_MNISTMLP_s42_20261122_132545`.
- T0212: `docs/reports/T0212_261123_2130.md`.
- Source contract: `src/models/`, `src/engine/`, `src/logger.py`, `src/datasets/`, `configs/`.