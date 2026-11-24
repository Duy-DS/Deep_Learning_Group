# BÁO CÁO TỔNG KẾT BTL03 — SNAPSHOT T0402

> **Bản thảo review, chưa phải bản nộp cuối.** Mốc kiểm kê: nhánh `docs/BTL03_T0402_thesis-report-writing`, base `main` tại `b2b88db3fc7444f7c37391e7211416ea3843a31e`, ngày 2026-11-24. Bằng chứng thực nghiệm mới nhất có report trong repository là T0212 (2026-11-23); T0402 ở đây là task viết tài liệu theo chỉ định trực tiếp, không phải bằng chứng rằng WP4 đã hoàn tất. Không chạy test, training hay LaTeX build trong task này.

## Thông tin đề tài và nhóm

- **Đề tài:** Bài tập lớn 03 — Chuyển đổi mã nguồn Deep Learning từ Keras sang PyTorch và phân tích thực nghiệm.
- **Mã dự án:** `BTL03`.
- **Snapshot báo cáo:** T0402; commit base `b2b88db3fc7444f7c37391e7211416ea3843a31e`.
- **Tác giả/người tổng hợp bản thảo:** `Duy-DS` (Member 01; xác nhận bởi `BTL03/.env` và các task report). Đây là alias Git, không thay thế họ tên pháp lý.
- **Thành viên theo alias trong task reports:** `Duy-DS`, `nguyenthanhdat180905-cloud`, `nguyen-cong-tri`, `TonOfWeight072501`, `CHIVY2005`.
- **Họ tên pháp lý, MSSV, trường/khoa/lớp, giảng viên, người chủ trì cuối kỳ, vai trò cuối kỳ và tỷ lệ đóng góp:** **[NHÓM CẦN XÁC NHẬN; không suy đoán]**.
- **Mâu thuẫn WBS:** prompt giao T0402 viết báo cáo; `AGENTS.md` xếp T0401–T0404 cho WP4 Advanced Vision, còn báo cáo tổng kết thuộc WP6/T0601–T0603. Bản thảo làm theo chỉ định trực tiếp về T0402 nhưng không sửa WBS hoặc khẳng định đã được nhóm phê chuẩn phân công chính thức.

## Tóm tắt

BTL03 tổ chức việc chuyển mười cấu hình thuộc năm nhóm từ notebook Keras sang PyTorch hướng đối tượng. Repository có dataset/model/trainer/callback/logger/CLI, baseline CSV, task reports và các kiểm tra kiến trúc. Các bằng chứng phải được đọc theo loại: output Keras lưu trong notebook là **legacy baseline**; T0201 và T0202 là những run PyTorch có report riêng; còn T0212 chỉ chạy mười ca train/validation một epoch trên dữ liệu tổng hợp đúng shape. T0212 báo logger bị chặn do không tương thích protobuf/TensorBoard, 78 test pass và 4 test fail theo các shard; trạng thái tổng thể là **BLOCKED**, không phải nghiệm thu toàn phần. Bất đồng parameter count CNN-LSTM giữa T0206 và T0207/T0212 vẫn cần đối chiếu. Không có cơ sở để kết luận cả mười mô hình đã được huấn luyện đầy đủ hoặc đạt metric Keras.

**Từ khóa:** Keras; PyTorch; chuyển đổi mô hình; NCHW; khả năng tái lập; provenance; giới hạn bằng chứng.

## Danh mục viết tắt

| Viết tắt | Ý nghĩa |
|---|---|
| CNN | Mạng tích chập (Convolutional Neural Network) |
| FCN | Mạng tích chập toàn phần theo score-map của notebook legacy; không mặc nhiên là detector có IoU/mAP |
| LSTM | Long Short-Term Memory |
| NCHW / NHWC | Batch–channel–height–width / batch–height–width–channel |
| OOP | Lập trình hướng đối tượng |
| RQ | Câu hỏi nghiên cứu |
| WBS | Cấu trúc phân rã công việc |

## Câu hỏi nghiên cứu và phạm vi kết luận

README BTL03 lưu hai câu hỏi RQ1 và RQ2 dưới đây, nhưng file Project Charter Markdown đúng tên được nhắc trong một số báo cáo không có trong repository. Vì vậy các câu hỏi này được ghi là **câu hỏi theo README, cần nhóm xác minh với charter chính thức**, không coi là trích nguyên văn charter. Không tự tạo RQ3/RQ4.

| ID | Câu hỏi được ghi trong README | Bằng chứng hiện có | Trạng thái kết luận |
|---|---|---|---|
| RQ1 (theo README; chờ xác minh charter) | Khác biệt NHWC/NCHW, vòng lặp huấn luyện và khởi tạo ảnh hưởng thế nào đến hội tụ/ổn định gradient? | Có source, parity-count, T0201 sweep và một số run; không có đối chứng đồng nhất đủ cho mọi kiến trúc | **Chưa kết luận tổng quát** |
| RQ2 (theo README; chờ xác minh charter) | Dropout, BatchNorm, augmentation và callbacks kiểm soát overfitting ra sao trên dữ liệu thực? | Có baseline Keras lưu; callbacks/code/tests; chưa có đối chứng PyTorch full-data đầy đủ | **Chưa kết luận hiệu quả PyTorch** |
| RQ3/RQ4 | Chưa xác minh được câu chữ chính thức trong charter Markdown | Không tự suy diễn từ mục tiêu chung hoặc README | **Không ghi câu hỏi giả định** |

### Mở đầu: bối cảnh, mục tiêu, phương pháp

Mục tiêu kỹ thuật là chuyển các notebook thực hành Keras sang các module PyTorch có dataset, model, engine huấn luyện, cấu hình và runner tách biệt; đồng thời lưu provenance và đối chiếu công bằng. Phạm vi báo cáo này là tổng hợp tài liệu có trong repository tại base nêu trên, không phải một lần chạy mới. Phương pháp là đối chiếu `AGENTS.md`, hướng dẫn, task reports, baseline CSV/epoch CSV, config và source code đã có. Không diễn giải cấu hình mục tiêu thành kết quả, không xem synthetic smoke là thực nghiệm trên dữ liệu thật, và không sửa model/legacy/data.

## 1. Tổng quan dự án và phạm vi

### 1.1 Năm nhóm và mười cấu hình

| Nhóm | Cấu hình/bài toán | Số cấu hình | Trọng tâm chuyển đổi |
|---|---|---:|---|
| 1 — MLP | `FuncApxmMLP`, `MNISTMLP` | 2 | `Dense`→`nn.Linear`; hồi quy MSE và phân loại logits |
| 2 — CNN+MLP | `MNISTCNN`, `HumanGenderCNN`, `EmotionCNN` | 3 | `Conv2D`→`nn.Conv2d`; layout, preprocessing, pooling/flatten |
| 3 — CNN+LSTM | `MNISTCNNLSTM` | 1 | Thay `TimeDistributed` bằng gộp/tách batch-time |
| 4 — Training Utilities | Gender + callbacks, Emotion + callbacks | 2 | Callback lifecycle, checkpoint, early stopping, scheduler và logging; dùng lại hai kiến trúc nhóm 2 |
| 5 — Advanced Vision | `SegNet`, `FCNObjectDetector` | 2 | Segmentation logits/mask và detector score-map hồi quy |

“Mười cấu hình” không đồng nghĩa mười kiến trúc độc lập: hai cấu hình nhóm 4 dùng lại GenderCNN/EmotionCNN với tiện ích huấn luyện.

### 1.2 Trạng thái bằng chứng tại snapshot

- T0101 tạo framework soạn thảo Markdown/LaTeX và quy tắc evidence; PDF cuối trên Overleaf **chưa có**.
- T0102–T0106 có report về pipeline dữ liệu, baseline inventory, logger/core; dữ liệu thật cho một số nhóm chưa có và các trạng thái runtime phải đọc theo từng report.
- T0201 có 36 cấu hình sweep PyTorch với CSV/Run_ID; baseline Keras số MSE không được lưu.
- T0202 có một run MNIST MLP 30 epoch; không đại diện cho chín cấu hình còn lại.
- T0203–T0211 có triển khai, báo cáo task và một số unit/shape/parameter checks; nhiều full-data run chưa thực hiện hoặc bị chặn.
- T0212 ghi 10/10 synthetic one-epoch train/validation pass, logger blocked và full-suite shard 78 pass/4 fail trên 82; kết luận tổng thể `BLOCKED`.
- Trong bộ báo cáo đã rà soát không có report nghiệm thu sau T0212 chứng minh full training, ablation, WP3/WP4 hoàn tất, demo cuối hoặc PDF Overleaf. Trạng thái mới hơn do đó là **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY ĐƯỢC KIỂM TRA**, không suy diễn rằng không có hoạt động ngoài repository.

## 2. Cơ sở lý thuyết và khác biệt Keras–PyTorch

### 2.1 Vòng lặp, logits và loss

Keras thường đóng gói vòng lặp qua `model.fit`; PyTorch thể hiện rõ `optimizer.zero_grad()` → forward → loss → `backward()` → `optimizer.step()`. Validation dùng trạng thái `eval()` và inference/no-gradient mode. Phân loại PyTorch thường đưa logits thô vào `CrossEntropyLoss` cùng nhãn class-index `torch.long`; không áp softmax trước loss. Softmax chỉ cần khi cần xác suất dự đoán. Số tham số bằng nhau là kiểm tra cấu trúc, không chứng minh weights, outputs, gradients hay metric bằng nhau.

### 2.2 NHWC, NCHW, rotation và chuẩn hóa

Keras ảnh thường dùng `(N,H,W,C)`, trong khi `nn.Conv2d` nhận `(N,C,H,W)`:

\[
x_{NCHW}=\operatorname{permute}(x_{NHWC},0,3,1,2).\operatorname{contiguous}().
\]

WIKI áp dụng `np.rot90(k=3, axes=(1,2))` trên hai trục không gian NHWC trước khi permute; rotation làm thay đổi tọa độ ảnh, còn permute chỉ đổi thứ tự trục. MNIST legacy chia pixel byte cho 255. T0102/T0205 ghi FER legacy không có phép chia `/255` tường minh và loader có chế độ `preserve`; cần dựa vào range dữ liệu nguồn thực tế trước khi kết luận parity. FCN detection dùng chuẩn hóa `x/127.5 - 1.0`.

### 2.3 CNN-LSTM và `TimeDistributed`

Với tensor `(B,T,C,H,W)`, gộp `(B,T)` thành `(B×T,C,H,W)` để CNN xử lý từng frame cùng trọng số; sau khi flatten đặc trưng thành `(B×T,F)`, khôi phục `(B,T,F)` theo đúng thứ tự thời gian rồi đưa vào LSTM. Đây là tương đương về tổ chức shape với `TimeDistributed(CNN)`, không tự đảm bảo parity weights/metric.

### 2.4 SegNet, upsampling và mask loss

`UpSampling2D` nearest được ánh xạ sang `nn.Upsample(scale_factor=2, mode="nearest")`, không thay bằng phép chuyển vị tích chập có trọng số. SegNet trả logits `(B,K,H,W)`; mask phân lớp là class-index `(B,H,W)` kiểu `long` cho cross-entropy. Pixel accuracy khác mIoU; không đổi tên metric legacy.

### 2.5 FCN score-map

Notebook detection dùng đầu ra score `tanh`, target ±1 và MSE; head là convolution 8×8 và 1×1. Trường `accuracy` legacy không phải IoU/mAP. Heatmap/box từ model chưa huấn luyện chỉ chứng minh đường đi shape/inference, không chứng minh chất lượng phát hiện.

## 3. Kiến trúc hệ thống và chuyển đổi

### 3.1 Các thành phần OOP

- `src/models/base_model.py`: `BaseMigratedModel` cung cấp hợp đồng `forward`, kiểm tra ảnh NCHW, bộ đếm tham số total/trainable/non-trainable và feature hooks.
- `src/engine/trainer.py`: `BaseTrainer` quản lý train/validation, callback, logger và checkpoint; loss tổng hợp theo số mẫu, validation dùng inference mode.
- `src/utils.py`: `seed_everything(42)`, worker seed và generator DataLoader. T0106 lưu ý `PYTHONHASHSEED` cần được đặt trước khi khởi chạy tiến trình nếu muốn cố định hash seed của chính tiến trình.
- `src/engine/callbacks.py`: callback lifecycle, `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`.
- `src/logger.py`: `ExperimentLogger` có schema master CSV 18 cột, epoch CSV và TensorBoard `SummaryWriter`; `CheckpointManager` lưu state/metadata cục bộ và chỉ mirror Drive nếu mount root đã tồn tại.
- `src/train.py`, `configs/`, `scripts/`: dispatch runner theo YAML/CLI. T0212 bổ sung `--task all --smoke-test`; chế độ này dùng harness tổng hợp, không phải full training.

Sự tồn tại của API/code không có nghĩa luồng logger hoạt động ở mọi môi trường: T0104 report một lượt logger độc lập PASS, còn T0212 report logger BLOCKED vì protobuf/TensorBoard. Hai mốc không được gộp thành một kết luận duy nhất.

### 3.2 Ma trận chuyển đổi và bằng chứng

| Nhóm/cấu hình | Kỹ thuật/mô-đun | Bằng chứng và giới hạn |
|---|---|---|
| `FuncApxmMLP` | `Linear(2,B)→Sigmoid→Linear(B,1)→Sigmoid`; số tham số `4B+1` | T0201 sweep 36 cấu hình; Keras numeric MSE `NOT_RECORDED` |
| `MNISTMLP` | Flatten 784→512→512→10; logits + CE | T0202 một run 30 epoch, 669,706 params; split khác Keras |
| `MNISTCNN` | Conv/pool, Flatten 2304, Dense 512×2 | 1,817,610 params; full training PyTorch chưa được report xác nhận |
| `HumanGenderCNN` | 5 block; xoay WIKI trước NCHW | 4,063,570 params; thiếu WIKI thật và PyTorch full metric |
| `EmotionCNN` | CNN 48×48, 7 logits, FER preprocessing theo legacy | 19,028,287 params; thiếu FER thật và PyTorch full metric |
| `MNISTCNNLSTM` | CNN frame encoder + hai LSTM; reshape thay `TimeDistributed` | T0206 ghi 1,746,049; T0207/T0212 ghi 1,747,073; **CẦN ĐỐI CHIẾU** |
| Gender/Emotion + callbacks | Callbacks áp lên kiến trúc nhóm 2 | Code/test callback có; full run/log PyTorch bị thiếu hoặc blocker |
| `SegNet` | Encoder-decoder 5 tầng; nearest upsample; logits/mask class-index | 29,458,211 tổng (29,442,325 trainable; 15,886 non-trainable theo report T0210); chưa full train trên data thật |
| `FCNObjectDetector` | Score-map tanh, conv 8×8/1×1, RGB normalization | 83,367 params; chưa có metric trên ảnh xe thật |

## 4. Kiểm toán bằng chứng, kết quả và trạng thái

### 4.1 Bảng parameter-count audit

Các số là giá trị báo cáo/CSV baseline và T0212 ghi; không phải xác nhận weight/output/accuracy parity.

| Cấu hình | Keras params | PyTorch params/report | Diễn giải |
|---|---:|---:|---|
| `FuncApxmMLP`, B=16 | 65* | 65 | `*` Keras không lưu `model.summary`; 65 suy ra từ công thức `4B+1` và chỉ đúng khi B=16 |
| `MNISTMLP` | 669,706 | 669,706 | Count khớp; chỉ có một run MNIST PyTorch được báo cáo |
| `MNISTCNN` | 1,817,610 | 1,817,610 | Count khớp |
| `HumanGenderCNN` | 4,063,570 | 4,063,570 | Count khớp |
| `EmotionCNN` | 19,028,287 | 19,028,287 | Count khớp |
| `MNISTCNNLSTM` | 1,747,073 (baseline) | T0206: 1,746,049; T0207/T0212: 1,747,073 | **CẦN ĐỐI CHIẾU**, lệch report 1,024 tham số |
| Gender + callbacks | 4,063,570 | 4,063,570 | Dùng lại GenderCNN |
| Emotion + callbacks | 19,028,287 | 19,028,287 | Dùng lại EmotionCNN |
| `SegNet` | 29,458,211 | 29,458,211 | Gồm 15,886 BN non-trainable theo quy ước report |
| `FCNObjectDetector` | 83,367 | 83,367 | Count khớp |

### 4.2 Kết quả thực nghiệm có nguồn

| Nguồn | Quan sát được | Caveat bắt buộc |
|---|---|---|
| T0201 — `t0201_func_apxm_abc_sweep.csv` | 36 cấu hình A/B/C, seed 42, SGD 0.01, batch 128; PyTorch có MSE và Run_ID. Ví dụ sweep A: Test-grid MSE 0.02343764 tại A=100 và 0.00683120 tại A=1000. | Grid test là 100 điểm theo report; không có numeric Keras MSE đối chứng nên không tuyên bố parity định lượng. |
| T0202 — Run_ID `T0202_MNISTMLP_s42_20261122_132545` | 30 epoch; PyTorch test accuracy 0.95870, test loss 0.13292970, thời gian 150.24952 giây CPU theo report; 669,706 params. | Keras ghi 0.9648 nhưng dùng test split làm validation/evaluate; protocol không đồng nhất, không quy chênh lệch chỉ do framework. |
| Baseline Keras EXP-03 MNIST CNN | CSV T0103 lưu accuracy 0.9884 và 1,817,610 params. | Đây là legacy notebook output; report T0203 nói PyTorch full accuracy chưa chạy. |
| Baseline Keras EXP-05 FER | CSV T0103 lưu train accuracy 0.8660, val accuracy 0.5954 và val loss cuối 1.4070. | Legacy metric; validation/public-test được tái sử dụng; không phải run PyTorch mới. |
| T0212 smoke | 10/10 synthetic one-epoch train + validation pass; parameter audit được báo cáo khớp cho 10/10 cấu hình. | Không phải training khoa học, không phải accuracy trên dữ liệu thật. |
| T0212 logger/test suite | Logger 10/10 `BLOCKED`; full suite shard 78 passed, 4 failed / 82 tests; protobuf/TensorBoard mismatch. | T0212 overall `BLOCKED`; không gọi PASS toàn phần. |

### 4.3 Trạng thái công việc

| Hạng mục | Trạng thái evidence được rà soát | Hướng xác minh tiếp |
|---|---|---|
| T0201 | 36 sweep PyTorch có CSV/Run_ID; Keras numeric MSE `NOT_RECORDED` | Nếu cần parity, chạy baseline đối chứng có seed/protocol và lưu history |
| T0202 | Một MNIST MLP PyTorch 30 epoch được báo cáo | Giữ caveat split; không suy rộng ra toàn bộ model |
| T0203–T0205 | Code/count/shape tests có; full-data PyTorch runs không được xác nhận trong reports | Dữ liệu thật, split/config/runtime và log có provenance |
| T0206–T0207 | Dataset/model evidence có; full MNIST train/MAE chưa xác nhận; count mâu thuẫn | Đối chiếu source/commit/counting semantics rồi chạy protocol thống nhất |
| T0208–T0209 | Callback code/tests; logger/data blocker, chưa có kết quả PyTorch convergence | Xác minh runtime logger và WIKI/FER thật trước khi so sánh |
| T0210–T0211 | Kiến trúc, shape/count evidence; chưa có PyTorch full-data quality metric | Dữ liệu thật; pixel accuracy/mIoU hoặc metric detection được định nghĩa rõ |
| T0212 | **BLOCKED**; logger blocked, 4 lỗi suite theo shard | Tương thích dependency trong môi trường được duyệt rồi chạy lại quality gates |
| Full training 10 cấu hình | **[CHƯA CÓ BẰNG CHỨNG ĐẦY ĐỦ TRONG REPOSITORY ĐƯỢC KIỂM TRA]** | Lưu split, seed, config, runtime, Run_ID, epoch CSV, checkpoint và report từng model |
| Ablation, trả lời RQ, trực quan hóa/prediction có provenance | **[CHƯA CÓ BẰNG CHỨNG ĐỦ ĐỂ KẾT LUẬN]** | Thiết kế đối chứng; dùng dữ liệu thật và lưu nguồn hình, metric, split |
| Demo/đóng gói cuối và PDF Overleaf | **NOT RECORDED / NOT RUN trong reports hiện có** | Xác nhận WBS, thông tin nhóm; upload LaTeX bundle và biên dịch/duyệt trên Overleaf |
| T0402 theo prompt hiện tại | Đang soạn tài liệu; mã task xung đột WBS `AGENTS.md` như đã nêu | Nhóm xác nhận mapping task chính thức trước khi nghiệm thu cuối |

### 4.4 Ma trận evidence và giới hạn suy luận

| Loại bằng chứng | Cho phép kết luận | Không cho phép kết luận |
|---|---|---|
| Source/config | Có implementation/cấu hình được khai báo | Cấu hình đã chạy hoặc mục tiêu đã đạt |
| Parameter count | Số lượng tham số theo snapshot/quy ước báo cáo | Weight/output/gradient/metric parity |
| Synthetic smoke | Shape/wiring và một lượt train/validation giả theo report | Generalization, accuracy hay convergence trên dữ liệu thật |
| Legacy notebook output | Metric đã in trong notebook Keras | Metric PyTorch hoặc metric độc lập nếu split bị reuse |
| T0201/T0202 run | Kết quả ở đúng cấu hình/Run_ID được report | Kết quả đại diện mười cấu hình hoặc benchmark công bằng toàn hệ thống |
| T0212 suite | 78 pass/4 fail theo shard; logger blocked | Overall PASS hoặc nghiệm thu T0212 |

## 5. Thảo luận, hạn chế và kết luận

### 5.1 Hạn chế và nguy cơ sai lệch

1. Nhiều notebook legacy thiếu metadata phần cứng, seed/checksum hoặc metric số; `NOT_RECORDED` không được thay bằng suy đoán.
2. Một số validation/public-test split bị tái sử dụng. Phải giữ tên split và không gọi chúng là independent holdout.
3. Dữ liệu WIKI/FER/SegNet/vehicle thiếu trong các reports T0203–T0211; CPU-only và dependency blocker hạn chế run PyTorch.
4. Parameter-count parity không chứng minh parity toán học đầy đủ; CNN-LSTM có bất đồng 1,024 tham số giữa hai report.
5. T0212 synthetic smoke xác nhận wiring, không xác nhận metric. Logger bị block và bốn test lỗi khiến trạng thái tích hợp là BLOCKED.
6. README tổng quan có các con số/diễn đạt không khớp task-level evidence; báo cáo ưu tiên task reports, baseline CSV và source provenance, đồng thời nêu rõ caveat.
7. WBS có mâu thuẫn giữa T0402 của prompt và `AGENTS.md`; danh tính pháp lý/MSSV/tỷ lệ cuối kỳ chưa được xác nhận.

### 5.2 Bài học và kiến nghị

- Khóa split, seed, preprocessing, loss reduction, optimizer/initializer và runtime trước khi so sánh framework.
- Gắn từng metric với task, source path, split, Run_ID và trạng thái; tách legacy baseline khỏi kết quả PyTorch.
- Xử lý blocker TensorBoard/protobuf như task môi trường có kiểm soát; không dùng writer giả để tuyên bố logger hoạt động.
- Đối chiếu CNN-LSTM parameter semantics từ source snapshot trước khi chốt parity.
- Chỉ báo cáo ablation/RQ sau khi có đối chứng thực nghiệm; segmentation dùng metric đúng định nghĩa, detection không dùng legacy `accuracy` thay IoU/mAP.
- Nhóm cần xác nhận WBS, tên/MSSV/giảng viên, đóng góp tổng 100%, sau đó duyệt nội dung và PDF trên Overleaf.

### 5.3 Kết luận trong phạm vi snapshot

Repository lưu được mã chuyển đổi cho năm nhóm, các báo cáo task, baseline inventory, sweep T0201, run T0202, parameter audits và smoke tích hợp T0212. Các evidence này hỗ trợ mô tả kiến trúc và một phần khả năng vận hành, nhưng **không chứng minh mười cấu hình đều đã được huấn luyện đầy đủ hoặc đạt baseline**. T0212 vẫn `BLOCKED`; full-training, ablation, trả lời RQ, demo cuối và PDF Overleaf chưa có bằng chứng hoàn tất trong reports được rà soát. Bản thảo này là tổng hợp có giới hạn, không phải xác nhận nghiệm thu cuối kỳ.

## Tài liệu tham khảo

1. Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 32. BibTeX key: `paszke2019pytorch`.
2. Badrinarayanan, V., Kendall, A., & Cipolla, C. (2017). *SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation*. IEEE TPAMI, 39(12), 2481–2495. DOI: 10.1109/TPAMI.2016.2644615. BibTeX key: `badrinarayanan2017segnet`.
3. Nhóm BTL03 (2026). *T0212: WP02 PR review, parameter audit, CLI and smoke-test report*. `docs/reports/T0212_261123_2130.md`; snapshot nguồn `33dcd74599e86404bbc8128c8057afba82810093`. BibTeX key: `btl03_t0212_internal`.
4. Báo cáo và nguồn nội bộ T0102–T0211 được dẫn theo đường dẫn/task trong repository; baseline chi tiết: `experiments/csv_logs/keras_baseline_10_models.csv`; manifest: `docs/thesis/source_materials/evidence_sources_manifest.md`.

## Phụ lục: provenance chính

- Quy tắc và WBS: `AGENTS.md`.
- Baseline inventory: `docs/reports/261127_T0103_keras_baseline_log.md`, `experiments/csv_logs/keras_baseline_10_models.csv`.
- Run T0201: `docs/reports/T0201_261122_2112.md`, `experiments/csv_logs/t0201_func_apxm_abc_sweep.csv` và các epoch CSV có Run_ID.
- Run T0202: `docs/reports/T0202_261122_2112.md`, Run_ID `T0202_MNISTMLP_s42_20261122_132545`.
- T0212: `docs/reports/T0212_261123_2130.md`.
- Quy tắc tổng hợp: `docs/thesis/AGENT_THESIS_RULES.md` và `docs/thesis/source_materials/evidence_sources_manifest.md`.