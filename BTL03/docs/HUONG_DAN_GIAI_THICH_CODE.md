# Hướng dẫn giải thích code — Task T0102

## 1. Luồng dữ liệu

`scripts/download_data.py` quản lý việc tải MNIST IDX, kiểm kê file WIKI/FER, tải tài nguyên chỉ khi cấu hình URL/Google Drive ID thật và sinh dữ liệu giả deterministic cho smoke-test. Dữ liệu lớn ở `data/raw/` và cache trong `data/processed/` bị `.gitignore` bảo vệ.

`src/datasets/vision_datasets.py` chuẩn hóa dữ liệu ảnh và nhãn:

1. MNIST nhận `NHW` 28×28, chuyển sang `NHWC` một kênh và chia pixel byte cho 255 như notebook gốc.
2. WIKI nhận `NHW`/`NHWC`; mặc định xoay ảnh NHWC theo đúng `np.rot90(x, k=3, axes=(1,2))`, rồi mới đổi trục.
3. FER nhận mảng hoặc CSV legacy gồm 2304 pixel và 7 nhãn one-hot; kiểm tra schema trước khi chuyển ảnh sang tensor.
4. Ảnh đầu ra có layout `(N,C,H,W)` và dtype `torch.float32`; nhãn phân loại có dtype `torch.long` để dùng được với loss PyTorch nhận class index.

## 2. Phép xoay WIKI và đổi layout

Với ảnh vuông kích thước `H=W=64`, `np.rot90(A, k=3)` xoay 270° ngược chiều kim đồng hồ, tương đương 90° theo chiều kim đồng hồ. Tọa độ pixel `(y,x)` chuyển thành `(x, W-1-y)`. Vì `np.rot90` có thể tạo view không liên tục/readonly, Dataset tạo bản sao liên tục trước khi tạo Tensor.

Sau khi xoay, phép đổi layout là:

```text
(N, 64, 64, 1) --xoay trên axes=(1,2)--> (N, 64, 64, 1)
             --permute(0,3,1,2)--------> (N, 1, 64, 64)
```

Phép xoay làm đổi tọa độ không gian; phép `permute` chỉ đổi thứ tự trục. Hai thao tác không thể thay thế cho nhau. Khi dữ liệu đầu vào đã được xoay ở upstream, có thể đặt `apply_legacy_rotation=False` để không xoay hai lần.

## 3. Chuẩn hóa pixel và nhãn

- MNIST notebook gốc gọi `x /= 255`; Dataset giữ cùng quy ước `[0,1]`.
- WIKI `normalization="auto"`: giữ nguyên nếu max ≤ 1; chia 255 nếu dữ liệu pixel byte có max > 1.
- FER notebook được kiểm tra không có phép `/255` tường minh. Vì CSV thực tế chưa được cung cấp, loader `auto` bảo toàn giá trị unit-range và chuẩn hóa dữ liệu byte nếu cần; khi parity với CSV cụ thể, cần ghi nhận range nguồn trước khi kết luận.
- Nhãn one-hot `(N,K)` được đổi về class index bằng `argmax(axis=1)`; nhãn trả ra `torch.long`, nằm trong `[0,K-1]`.

## 4. Câu hỏi thường gặp

**Vì sao `Conv2d` cần NCHW?** PyTorch convolution nhận batch, channels, height, width. Tensor Keras NHWC cần `permute(0,3,1,2)` trước khi đưa vào model.

**Tại sao không hardcode Google Drive URL?** Workspace không có Drive ID/URL thật. Cấu hình để `null` nhằm tránh bịa nguồn hoặc tải nhầm dữ liệu; chỉ điền giá trị đã được nhóm xác nhận.

**Dummy data có dùng để huấn luyện/báo cáo accuracy không?** Không. Chỉ dùng xác minh shape, dtype, range, phép xoay và wiring của DataLoader.

**Ảnh PNG smoke có chứng minh dữ liệu WIKI thật đúng chiều không?** Không. PNG được tạo từ ảnh giả để kiểm tra phép biến đổi. Cần xem ảnh nguồn WIKI thật sau khi nhóm cung cấp dữ liệu.

**Vì sao CSV FER không tự coi mọi giá trị là byte?** Notebook gốc không khai báo chuẩn hóa. `auto` phân biệt unit-range và byte-range để tránh chia 255 hai lần nếu nguồn đã được chuẩn hóa.

---

## 5. Phụ lục T0103 — kiến trúc và baseline Keras gốc

### Phạm vi và provenance

Mười notebook trong `legacy_keras/` được giữ nguyên. `scripts/extract_keras_baselines.py` chỉ đọc JSON, source cells và execution outputs đã lưu; không import TensorFlow, không chạy notebook. Bảng chuẩn hóa nằm ở `experiments/csv_logs/keras_baseline_10_models.csv`; `NOT_RECORDED` nghĩa là notebook không lưu bằng chứng tương ứng. Số params dưới đây là output `model.summary()`, ngoại trừ Func_apxm_MLP chưa có summary (chỉ có công thức giải tích theo B).

### Mười kiến trúc theo nhóm

| EXP / mô hình | Shape đầu vào Keras → đầu ra | Tham số legacy | Luồng kiến trúc và chuyển đổi trọng tâm |
|---|---|---:|---|
| EXP-01 `Func_apxm_MLP` | `(2,) → (1,)` | Summary không lưu; Dense(2→B)+Dense(B→1) có `4B+1` params (65 khi B=16) | Xấp xỉ hàm Gaussian, sigmoid ở hidden/output, MSE + SGD. Notebook sweep A/B/C nên không tồn tại một B duy nhất cho mọi run; output metric số không lưu. |
| EXP-02 `MNIST_MLP` | `(28,28) → (10,)` | 669,706 (trainable 669,706) | Flatten 784 → Dense 512 → Dropout → Dense 512 → Dropout → softmax 10; SGD 0.01, categorical cross-entropy; ảnh chia 255. PyTorch cần thêm channel khi chuyển sang Dataset CNN dùng chung. |
| EXP-03 `1_MNIST_CNN` | `(28,28,1) → (10,)` | 1,817,610 | Conv 64/128/256 + MaxPool → Flatten → hai Dense 512 + Dropout → softmax 10; SGD 0.01. Soát padding/pooling và kích thước flatten trước khi đối chiếu params. |
| EXP-04 `2_Human_Gender_CNN` basic | `(64,64,1) → (2,)` | 4,063,570 | 5 block CNN tăng channels 16→256, Flatten, 2 Dense 1024; xoay `np.rot90(k=3, axes=(1,2))` trước model. Adam 1e-4; không callbacks. |
| EXP-05 `3_Emotion_Reg_CNN` basic | `(48,48,1) → (7,)` | 19,028,287 | CNN valid/same conv, Dense 1024, 7-way softmax; Adam 1e-4. CSV 2304 pixel + 7 one-hot; public-test file cũng dùng trong validation/evaluate. |
| EXP-06 `MNIST_CNN_LSTM` | `(8,28,28,1) → (1,)` | 1,747,073 | TimeDistributed Conv/Pool/Dropout/Flatten → LSTM 128 hai tầng → Dense 1; MAE + SGD 0.01. PyTorch thay wrapper bằng reshape batch×time qua CNN rồi dựng `(B,T,F)` cho LSTM. |
| EXP-07 `2_Human_Gender_CNN` callbacks | `(64,64,1) → (2,)` | 4,063,570 | Cùng CNN Gender; thêm ModelCheckpoint, EarlyStopping patience 5, ReduceLROnPlateau patience 5, TensorBoard. Max epoch 100000 nhưng execution dừng epoch 32. |
| EXP-08 `3_Emotion_Reg_CNN` callbacks | `(48,48,1) → (7,)` | 19,028,287 | Cùng CNN Emotion; ImageDataGenerator + EarlyStopping patience 5. Max 100, execution dừng epoch 57. |
| EXP-09 `1_segnet` | `(96,96,1) → (9216,7)` | 29,458,211; trainable 29,442,325; non-trainable 15,886 | Encoder–decoder convolution/BatchNorm, 5 lần `UpSampling2D`, softmax theo 7 lớp pixel. Keras summary output là sequence pixel-class; PyTorch thường biểu diễn logits NCHW và cần khớp target/loss trước parity. |
| EXP-10 `2_obj_detection` | `(64,64,3) → (1,)` sau pooling/Flatten | 83,367 | Chuẩn hóa Lambda `x/127.5-1`, Conv 3×3, MaxPool 8×8, Conv 8×8 và 1×1 `tanh`, MSE. Notebook gọi `accuracy`, nhưng đây không phải IoU/mAP; PyTorch port cần định nghĩa metric detection phù hợp riêng. |

### Cách đọc kết quả và các giới hạn

- Train/Val metrics trong CSV là số cuối cùng trên progress-bar output đã lưu. Thời gian là chuỗi notebook in rồi đổi về giây; phần cứng và phiên bản thư viện không có nên không so sánh CPU với T4.
- Ô `notebook_printed_eval_*` phản ánh chính xác lệnh evaluate trong notebook, không đảm bảo là test holdout độc lập. Đặc biệt WIKI basic, FER basic và hai callback notebooks đánh giá trên split validation/public-test đã dùng trong `validation_data`.
- CNN-LSTM chỉ có train/validation loss trong output; không có evaluate/test MAE đã lưu. Không suy diễn test score từ validation loss.
- SegNet metric được giữ tên legacy là `accuracy`; không tự coi là mean IoU. ObjDet accuracy cũng không phải mAP/IoU.
- Colab helper `experiments/colab/run_keras_baseline_t4.py` chỉ xác nhận thiết bị/hướng dẫn lưu bản sao. Muốn có baseline mới, chạy notebook copy trong runtime Colab T4, lưu log đầy đủ và không ghi đè file legacy.

### Checklist chuyển đổi sang PyTorch

- [ ] Input layout NHWC→NCHW; bảo toàn thứ tự kênh và normalization.
- [ ] WIKI xoay `k=3`, `axes=(1,2)` trước `permute`; xác minh ảnh thật và nhãn.
- [ ] Dense/Conv padding, pooling, Dropout mode, BatchNorm epsilon/momentum và initializer khớp Keras.
- [ ] Keras softmax + categorical cross-entropy được đối chiếu với logits PyTorch + `CrossEntropyLoss` (không softmax trước loss).
- [ ] CNN-LSTM gộp/tách Batch×Time đúng thứ tự sequence; đối chiếu output target scalar/MAE.
- [ ] SegNet khớp 7-class pixel target, thứ tự trục, upsampling và class-index/one-hot loss contract.
- [ ] ObjDet không dùng `accuracy` legacy làm IoU/mAP; xác định metric và cách tạo score-map rõ trước khi so sánh.
- [ ] Params parity chỉ chốt sau khi tái tạo summary bằng code PyTorch; không lấy giá trị suy luận làm kết quả kiểm chứng.

---

## 6. Kiến trúc MLOps — `ExperimentLogger` và `CheckpointManager` (T0104)

### Trách nhiệm OOP

- `ExperimentLogger` sở hữu metadata của một run, TensorBoard `SummaryWriter`, file CSV epoch và append tổng kết vào master CSV. Mỗi run có `Run_ID` riêng, tạo ra đường dẫn:
  `experiments/tensorboard/<Model_Group>/<Model_Name>/<Exp_Type>_s<Seed>_<YYYYMMDD_HHMMSS>/`.
- Mỗi epoch ghi `Loss/Train`, `Loss/Validation`, `Metric/Train/<Tên metric>`, `Metric/Validation/<Tên metric>`, `Hyperparameters/Learning_Rate` và thời gian epoch. `metric_name` có thể là `Accuracy`, `MAE` hoặc `Pixel_Acc`.
- Khi `finish_run()` được gọi, logger ghi HParams cùng các metric tóm tắt và append đúng một dòng vào `master_experiment_runs.csv`. Lược đồ 18 cột được cố định trong `ExperimentLogger.MASTER_COLUMNS`; thứ tự không được đổi.
- `CheckpointManager` lưu `epoch`, `model_state_dict`, `optimizer_state_dict`, `best_metric`, `seed`, `config`. Lưu cục bộ atomically trước; Drive root được resolve theo thứ tự tham số `drive_root` → `BTL03_GOOGLE_DRIVE_ROOT` → `project.drive_mount_root` trong YAML. Chỉ copy mirror nếu đường dẫn đã tồn tại; không mount Drive, gọi API mạng hay chứa credential trong lớp.

### Vòng đời run đề xuất

1. Gọi `seed_everything(42)`, khởi tạo model/optimizer và truyền alias thành viên thật vào `ExperimentLogger`.
2. Mỗi epoch gọi `log_epoch(...)` với train/validation loss, metric, learning rate và elapsed seconds.
3. Khi validation metric cải thiện, gọi `CheckpointManager.save(...)` với config/seed/optimizer state.
4. Kết thúc bằng `finish_run(...)`, sau đó đóng logger (khuyến nghị dùng context manager); master CSV chỉ thêm một row cho run.
5. Smoke test dùng `Exp_Type="SmokeTest"` và sandbox tạm; không dùng smoke metric làm kết quả khoa học.

### Vì sao TensorBoard tạo event binary?

`SummaryWriter` serialize scalar/HParams thành event files `events.out.tfevents*` dưới run directory. TensorBoard đọc các event này để dựng đồ thị; file binary là artifact tái sinh được, không phải tài liệu nguồn, vì vậy bị `.gitignore` chặn. CSV là bản trao đổi gọn để review/đồng bộ Sheets.

### Hỏi đáp nhanh

**CSV có đúng schema Sheets không?** `ExperimentLogger.MASTER_COLUMNS` định nghĩa đúng 18 tên và thứ tự; khởi tạo sẽ từ chối master CSV đã tồn tại nếu header lệch.

**`Test_Metric` có bắt buộc không?** Không. Nếu chưa có test evaluation độc lập, để trống; không chép `Val_Metric` sang cột test.

**Logger có tự dùng tên người chạy không?** Không. Phải truyền `member_alias`, đặt `MEMBER_ALIAS`, hoặc dùng `member_id` để đọc `MEMBER_XX_NAME` từ `.env`; thiếu alias thì khởi tạo báo lỗi.

**Checkpoint có tự ghi lên Google Drive không?** Chỉ mirror khi thư mục mount được truyền/cấu hình và tồn tại. Không có mount thì checkpoint vẫn nằm ở `outputs/checkpoints/`.

**Đa tiến trình có ghi master CSV an toàn tuyệt đối không?** Logger dùng khóa trong process và kiểm tra Run_ID trùng trước append; nếu nhiều process/host đồng ghi một file, cần một writer trung tâm hoặc backend giao dịch thay vì chia sẻ CSV qua mạng.

---

## 7. T0106 — Core Engine OOP & Reproducibility

### `BaseMigratedModel`: hợp đồng chung của model

`src/models/base_model.py` cung cấp abstract class `BaseMigratedModel(nn.Module, ABC)`. Mỗi model con cài `forward()`; khi input là ảnh 4D, `validate_input_shape()` xác thực `(Batch, Channels, Height, Width)` và báo lỗi dễ hiểu nếu nhận tensor NHWC thường gặp từ Keras. `count_parameters()` trả ba bộ đếm độc lập:

```text
total_params = trainable_params + non_trainable_params
```

`summary(input_size)` đăng ký hook tạm trên leaf modules, chạy một forward giả trong `eval()`/`no_grad()`, ghi output shape và params từng layer, rồi luôn gỡ hook và khôi phục trạng thái train/eval ban đầu. Đây là bảng fallback nội bộ, không tuyên bố tương đương mọi tính năng của `torchinfo`.

### Feature extraction bằng forward hooks

1. Gọi `register_feature_hooks(["features.0", ...])` với tên chính xác trong `model.named_modules()`.
2. Gọi `output, features = model.extract_features(x)`. Output hook được `detach()` để không giữ computation graph cho PCA/visualization.
3. Gọi `clear_hooks()` sau khi trích xuất; `extract_features()` không tự gỡ hooks để có thể lấy nhiều batch.

Hook chỉ giữ activation của layer được yêu cầu; không nên đăng ký cả mạng cho dataset lớn vì sẽ tăng RAM/VRAM.

### `BaseTrainer`: vòng lặp và ranh giới gradient

`src/engine/trainer.py` đóng gói model, optimizer, criterion, metric và device. Model chuyển sang CUDA nếu khả dụng, nếu không dùng CPU; có thể truyền device tường minh. API nhận `nn.Module` tổng quát để WP02 tích hợp dần, nhưng `BaseMigratedModel` được validator NCHW gọi tự động cho input ảnh 4D.

```text
train: model.train → zero_grad → forward → loss → backward → optimizer.step
valid: model.eval  → torch.inference_mode → forward → loss/metric
```

Loss được nhân với batch size rồi chia tổng số mẫu, tránh batch cuối nhỏ bị tính trọng số ngang batch đầy đủ. `fit()` ghi epoch qua `ExperimentLogger`, lưu checkpoint khi `val_loss` cải thiện và gọi callback `on_epoch_end(trainer, epoch, logs)`; callback trả `True` để yêu cầu dừng. `Best_Epoch` phải tường minh, không tự suy ra nếu logger yêu cầu summary.

### Tái lập seed trên CPU/CUDA

`seed_everything(42)` seed Python `random`, NumPy, Torch CPU và CUDA, đồng thời bật `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`. `get_torch_generator(42)` kiểm soát shuffle của DataLoader; `seed_worker` đồng bộ Python/NumPy/Torch trong mỗi worker.

`PYTHONHASHSEED` được gán để tiến trình con thừa hưởng cấu hình. Tuy nhiên Python đã chọn hash seed khi interpreter khởi chạy; đổi biến trong runtime không thể sửa hash seed của tiến trình hiện tại. Muốn hash iteration cố định ngay từ đầu, đặt `PYTHONHASHSEED=42` trước khi launch Python.

Strict deterministic mode có thể raise nếu một toán tử CUDA không có implementation tất định. Kết quả bitwise cũng không được bảo đảm xuyên phiên bản thư viện/driver/phần cứng; cần ghi runtime cùng kết quả thực nghiệm. Môi trường T0106 này chỉ có PyTorch CPU, vì vậy CUDA behavior chưa được chạy kiểm chứng.

### Hỏi đáp core engine

**Vì sao `BaseTrainer` nhận `nn.Module` chứ không bắt buộc `BaseMigratedModel`?** Để không chặn quá trình chuyển đổi từng model ở WP02. Model kế thừa base class nhận thêm validation NCHW/hook/summary thống nhất; model cũ vẫn có thể chạy trainer trong giai đoạn chuyển tiếp.

**Validation có cập nhật gradient hoặc BatchNorm statistics không?** Trainer đặt `eval()` và dùng `inference_mode()`, nên không tạo graph gradient; các module có train/eval behavior dùng trạng thái eval.

**Hai lần chạy cùng seed có luôn giống bitwise trên mọi GPU không?** Không. Test CPU hiện chứng minh cùng môi trường cho cùng initial weights, history và final weights. CUDA strict mode từ chối một số phép toán không deterministic; driver/GPU/PyTorch khác nhau vẫn cần được ghi nhận.

**Vì sao seed DataLoader cần cả generator lẫn worker init?** Generator kiểm soát thứ tự lấy mẫu/shuffle; `seed_worker` seed các RNG Python/NumPy/Torch bên trong từng worker process.

**Tại sao `PYTHONHASHSEED` trong hàm không đủ?** Hash seed được interpreter khởi tạo lúc process start. Hàm đặt biến để child process dùng; process hiện tại cần được launch với biến môi trường có sẵn nếu cần hash iteration cố định.

---

## 7. Kiến trúc lõi huấn luyện — T0106

### `BaseMigratedModel`

`src/models/base_model.py` định nghĩa abstract class kế thừa `nn.Module` và `ABC`. Mô hình con cài `forward()`; với ảnh 4D, có thể gọi `validate_input_shape()` để kiểm tra NCHW và phát hiện tensor NHWC thường gặp (channel cuối bằng 1/3/4 nhưng chiều channel thứ hai là chiều không gian). `count_parameters()` trả riêng total/trainable/non-trainable để đối chiếu Keras.

`summary(input_size)` đăng ký hook tạm trên leaf modules, chạy một forward giả ở `eval()` + `no_grad()`, ghi output shapes/params, sau đó luôn gỡ hook và khôi phục training mode. Đây là summary gọn nội bộ, không thay thế đầy đủ mọi tính năng của `torchinfo`.

Feature hooks được đăng ký bằng tên chính xác từ `named_modules()`. `extract_features(x)` trả output chính và activation `detach()` theo tên layer; gọi `clear_hooks()` sau khi dùng để tránh giữ tham chiếu tensor. `init_weights_keras_style()` áp Glorot/Xavier Uniform cho Linear/Conv2d và zero bias; các lớp khác không bị thay đổi.

### `BaseTrainer`

`src/engine/trainer.py` nhận bất kỳ `nn.Module` tương thích, optimizer, criterion, metric tùy chọn và thiết bị. Chấp nhận rộng `nn.Module` để các model WP02 có thể chuyển dần; nếu model là `BaseMigratedModel` thì validator NCHW được gọi tự động với ảnh 4D.

- Train: `train()` → `zero_grad(set_to_none=True)` → forward → criterion → `backward()` → `step()`.
- Validation: `eval()` và `torch.no_grad()`; không cập nhật optimizer/gradient.
- Loss/metric được trung bình theo sample count, kể cả batch cuối nhỏ hơn.
- `fit()` đo thời gian epoch, gọi logger T0104, lưu checkpoint khi `val_loss` cải thiện và gọi callback `on_epoch_end(trainer, epoch, logs)`; callback trả `True` để dừng sớm.
- Khi gắn logger, phải truyền logger đã khởi tạo bằng alias thành viên thật; trainer từ chối logger không có alias. Checkpoint manager tiếp tục nhận seed/config theo schema T0104.

### Reproducibility, CPU và CUDA

`seed_everything(seed=42, deterministic=True)` đặt `PYTHONHASHSEED`, Python `random`, NumPy, CPU/CUDA Torch và bật `torch.use_deterministic_algorithms(True)` cùng `cudnn.deterministic=True`, `cudnn.benchmark=False`. `seed_worker` đồng bộ seed cho Python/NumPy/Torch trong DataLoader worker; `get_torch_generator` cấp generator riêng cho shuffle.

Tính tất định phụ thuộc operator, backend, driver và phần cứng. Nếu CUDA operator không có implementation deterministic, strict mode có thể báo lỗi thay vì âm thầm cho kết quả không xác định. `PYTHONHASHSEED` đặt trong tiến trình hiện tại không thể đảo ngược hash seed đã khởi tạo; muốn cố định hash iteration phải đặt biến trước khi khởi động Python. CPU reproducibility được kiểm thử trong môi trường này; CUDA được kiểm thử có điều kiện khi GPU khả dụng.

### Câu hỏi vấn đáp

**Vì sao validation dùng `no_grad()`?** Để không tạo graph/backward gradient, giảm bộ nhớ và đảm bảo validation không cập nhật trọng số.

**Tại sao average loss phải nhân batch size?** Nếu batch cuối nhỏ hơn, trung bình đơn giản các batch sẽ cho mỗi batch trọng số bằng nhau; cộng `loss × số mẫu` rồi chia tổng sample count cho trung bình theo mẫu.

**`cudnn.deterministic=True` đảm bảo tuyệt đối mọi máy cho cùng kết quả không?** Không. Nó buộc chọn thuật toán cuDNN deterministic khi có, còn deterministic algorithms strict có thể từ chối operator không hỗ trợ; khác biệt phần cứng/driver/phiên bản vẫn cần được ghi nhận.

**Forward hook ảnh hưởng gradient không?** Activation được lưu bằng `detach()` để chỉ phục vụ trực quan/feature analysis, không giữ graph. Hook phải được gỡ sau tác vụ.

**Vì sao BaseTrainer nhận `nn.Module`, không chỉ `BaseMigratedModel`?** Để chuyển đổi 10 kiến trúc theo từng task mà vẫn dùng trainer chung; validator NCHW được gọi tự động khi model có method đó.