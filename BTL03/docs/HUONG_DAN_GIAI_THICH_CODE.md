# Hướng dẫn giải thích code — Task T0102

## Phụ lục T0101 — Hệ thống viết báo cáo tổng kết T0212

### Luồng tạo và xuất bản

- `docs/thesis/templates/` chứa cấu trúc gốc Markdown/LaTeX; sửa nội dung báo cáo trong `md_drafts/` và `tex_source/`, không sửa template để lấp số liệu.
- `source_materials/evidence_sources_manifest.md` ánh xạ evidence ID đến report, CSV, source/config và giới hạn diễn giải.
- Bản Markdown là bản review dễ diff; bộ XeLaTeX có năm chapter và BibLaTeX là nguồn upload Overleaf. Đồng bộ số liệu, bảng, trạng thái, caveat và citation trước mỗi snapshot.
- Từ gốc `BTL03/`, build local bằng `cd docs/thesis/tex_source; latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`. Build local cần XeLaTeX/polyglossia/Biber; không tuyên bố build thành công nếu chưa xem exit code và log.
- Overleaf workflow: nén *nội dung* `tex_source/` để `main.tex` ở root ZIP → upload/chọn XeLaTeX/compile → kiểm tra và lưu PDF đã duyệt dưới `docs/thesis/`. Không commit `.aux/.log/.fls/.fdb_latexmk` hoặc PDF build tạm.

### Cách đọc snapshot T0212

- T0212 ghi trạng thái tổng thể `BLOCKED`, dù 10/10 synthetic one-epoch smoke cases và parameter count audit PASS.
- Logger 10/10 bị chặn bởi protobuf/TensorBoard; full suite được chia shard ghi 78 pass, 4 fail trên 82 test.
- Không diễn giải parameter parity thành weight/output/accuracy parity; synthetic smoke không thay thế dataset thật.
- T0201 có 36 PyTorch sweep runs nhưng Keras numeric MSE không lưu; T0202 có một MNIST MLP run 30 epoch, test accuracy 0.95870, với protocol validation Keras/PyTorch khác nhau.
- WP03/WP04 giữ trạng thái chưa nghiệm thu tại snapshot theo workbook/report; trước một phiên bản báo cáo sau phải kiểm tra evidence mới, không sao chép status lịch sử.
- Họ tên pháp lý/MSSV/tỷ lệ đóng góp chưa đủ bằng chứng: dùng placeholder, hỏi nhóm; alias được trích từ hồ sơ thành viên.

### Câu hỏi chuyển giao

**Smoke 10/10 có nghĩa là mười model huấn luyện xong không?** Không. Mỗi case chỉ train/validate một epoch trên synthetic fixtures; logger bị block, không đại diện convergence/full data.

**Parameter parity có nghĩa hai framework cho cùng accuracy không?** Không. Nó chỉ so sánh số lượng tham số theo cách đếm đã ghi; còn phụ thuộc dữ liệu, layout, initializer, padding, loss, optimizer, weights và split.

**Có thể ghi T0301–T0306 hoàn toàn chưa triển khai ở snapshot không?** Chỉ ghi deliverable/workbook state có nguồn; giữ đúng phân biệt Todo, code đã có, blocked và chưa có evidence. Kiểm tra lại workbook/report sau T0212 trước khi phát hành snapshot mới.

**Vì sao chưa có PDF?** Chưa có xác nhận thông tin pháp lý/đóng góp và build đầy đủ chưa được xác nhận; không tạo PDF giả hoặc gọi bản local tạm là Overleaf deliverable.

---

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

---

## 8. T0210 — SegNet: phân vùng ngữ nghĩa

### Dataset và hợp đồng tensor

Notebook `legacy_keras/05_others/1_segnet.ipynb` đọc ảnh depth grayscale, resize nearest-neighbor về `96×96`, và mask palette `{255,43,85,128,170,212,0}` tương ứng lớp `{0,1,2,3,4,5,6}`. `SemanticSegmentationDataset` chuyển ảnh `(N,H,W,1)` thành float32 `(N,1,H,W)`, đồng thời chuyển mask one-hot/palette sang class-index `(N,H,W)` kiểu `torch.long`. Vì thế `DataLoader` trả logits `(B,7,H,W)` và target `(B,H,W)` đúng hợp đồng `nn.CrossEntropyLoss()`; không thêm chiều kênh hoặc softmax trước loss.

### Kiến trúc và parameter parity

Encoder gồm 5 khối VGG-style `(64×2, 128×2, 256×3, 512×3, 512×3)`, mỗi khối kết thúc bằng max-pooling 2×2. Decoder thực hiện đúng 5 lần `nn.Upsample(scale_factor=2, mode="nearest")`, lần lượt với số convolution `(512×3, 512×2+256, 256×2+128, 128+64, 64)`, rồi head `Conv2d(64,7,1)` và BatchNorm cuối như legacy. Trong Keras, thứ tự mỗi khối là Conv→BatchNormalization→ReLU. Nearest upsampling lặp giá trị lân cận để nhân đôi H/W mà không học thêm trọng số; `ConvTranspose2d` thay vào đó học kernel upsampling và có số tham số/đáp ứng khác, nên không thay thế tương đương chính xác.

| Thành phần | Trainable params | Non-trainable BN state | Tổng đối chiếu |
|---|---:|---:|---:|
| Encoder khối 1 | 37,824 | 256 | 38,080 |
| Encoder khối 2 | 221,952 | 512 | 222,464 |
| Encoder khối 3 | 1,476,864 | 1,536 | 1,478,400 |
| Encoder khối 4 | 5,902,848 | 3,072 | 5,905,920 |
| Encoder khối 5 | 7,082,496 | 3,072 | 7,085,568 |
| Decoder khối 1 | 7,082,496 | 3,072 | 7,085,568 |
| Decoder khối 2 | 5,902,080 | 2,560 | 5,904,640 |
| Decoder khối 3 | 1,476,480 | 1,280 | 1,477,760 |
| Decoder khối 4 | 221,760 | 384 | 222,144 |
| Decoder khối 5 | 37,056 | 128 | 37,184 |
| Head `1×1` + BatchNorm | 469 | 14 | 483 |
| **SegNet tổng** | **29,442,325** | **15,886** | **29,458,211** |

PyTorch BatchNorm lưu running mean/variance dưới dạng buffers, không phải `Parameter`; `count_parameters()` cộng hai buffer đó khi lập tổng parity với `model.summary()` Keras và loại `num_batches_tracked` vốn không có trong Keras. Baseline Keras đã lưu là validation pixel accuracy `0.7732` ở 15 epoch; con số này không phải kết quả chạy PyTorch hiện tại. Để báo cáo pixel accuracy PyTorch, cần train bằng dữ liệu depth/mask thật.

### Độ đo

Pixel accuracy là số pixel có class argmax đúng chia tổng pixel. mIoU tính từng lớp `intersection / union`, sau đó lấy trung bình macro trên các lớp có union khác 0. Hai số này khác nhau: pixel accuracy có thể cao khi nền chiếm đa số, trong khi mIoU phạt các lớp hiếm bị bỏ sót.

## 9. T0211 — FCN score-map object detection

### Dense trên cửa sổ và convolution trượt

Notebook detection thực chất đã biểu diễn các tầng head bằng Conv2D có tên `dense1`/`dense2`: backbone Conv(3×3, 10 channels) hai lần + MaxPool(8×8), rồi Conv(8×8, 10→128), ReLU, Dropout, Conv(1×1,128→1), tanh. Trên đặc trưng `10×8×8`, một Dense `10·8·8 → 128` có kernel `(128,640)` và bias 128, tức 82,048 tham số. Conv2d 8×8 giữ kernel tương đương dưới reshape `(128,10,8,8)`; mỗi vị trí đặc trưng ảnh lớn áp dụng cùng phép nhân ma trận cho một patch trượt. Tầng `1×1` tương đương Dense cuối `(128→1)` tại từng vị trí, có `128+1=129` tham số. Tổng mạng là 83,367 params.

Ảnh được chuyển NHWC→NCHW và chuẩn hóa chính xác `x / 127.5 - 1.0`. Target legacy là xe `+1`, nền `-1`; đầu ra tanh và loss MSE. Vì đây là score regression chứ không phải softmax classifier, confidence heatmap ánh xạ `(score+1)/2` về `[0,1]`; metric sign accuracy tương ứng accuracy legacy, không gọi là IoU/mAP. Kích thước ảnh lớn tạo heatmap `(B,1,Hout,Wout)` nhờ convolution trượt, không cần cắt từng patch rồi gọi model lặp.

Một điểm đáng chú ý trong notebook là MaxPool 8×8 có stride mặc định 8. Vì vậy tọa độ ô `(r,c)` trên heatmap ánh xạ về góc trên trái ảnh `(c×8,r×8)`; box kết thúc ở `(min(c×8+64,W), min(r×8+64,H))`. NMS cục bộ giữ cực đại có score cao, sau đó loại các cửa sổ 64×64 giao nhau.

### Giới hạn kiểm chứng

Baseline Keras ghi 83,367 params, MSE validation 0.0642 và `accuracy` 0.9523; accuracy này là metric Keras trên score regression, không phải mAP/IoU. Notebook không ghi learning rate tường minh (Keras Adam mặc định); cấu hình PyTorch khai báo 0.001 để minh bạch, nhưng chưa thể khẳng định trajectory huấn luyện đồng nhất nếu khác phiên bản optimizer/initializer. Không có ảnh/data local thì heatmap synthetic chỉ chứng minh shape và vẽ hình, không chứng minh phát hiện xe đúng.

### Câu hỏi vấn đáp

**Vì sao dùng Upsample thay ConvTranspose2d trong SegNet?** Để khớp `UpSampling2D` nearest của Keras, tránh học trọng số bổ sung và tránh thay đổi parameter count.

**Vì sao CrossEntropyLoss không nhận mask one-hot?** `nn.CrossEntropyLoss` yêu cầu logits `(B,C,H,W)` và class index `(B,H,W)` kiểu long; class dimension được loss xử lý nội bộ.

**Conv 8×8 có thực sự tương đương Dense?** Có trên một patch 8×8 phẳng với cùng thứ tự flatten/weights; trên ảnh lớn convolution chia sẻ đúng ma trận đó cho từng patch. Để weight-surgery số học khớp, cần reshape kernel theo layout Keras/PyTorch đúng và kiểm tra phép flatten—không thể chỉ chép bytes không đổi.

**Có thể dùng softmax cho score detector không?** Không nên: legacy học scalar tanh với MSE và target ±1, không có logits nhiều lớp. Ánh xạ tuyến tính sang confidence giữ nguyên ý nghĩa score hơn softmax.

**Smoke synthetic chứng minh đạt 77% pixel accuracy không?** Không. Mọi metric trên synthetic chỉ xác minh loop và shape; nghiệm thu accuracy cần split thật, seed/config đã khóa và ghi rõ trong báo cáo.

---

## 10. T0212 — Kiểm toán WP02, parity và smoke test thống nhất

### Nhánh tích hợp và phạm vi history

Nhánh `test/BTL03_T0212_wp02-review-smoke` được tạo từ nhánh Group 5 vì Group 5 đã chứa tip của Group 1–4 trong history. Kiểm tra `merge-base --is-ancestor` xác nhận cả năm tip WP02 nằm trong lịch sử nền tích hợp; merge lại sẽ là thao tác thừa. `main` vẫn là nhánh độc lập, không checkout/merge. Thư mục `docs/project_charter/` không có markdown mang đúng tên `Project_Charter_BTL03.md`; baseline dùng `docs/reports/261127_T0103_keras_baseline_log.md` làm nguồn số học, đồng thời ghi thiếu charter markdown trong báo cáo.

Git author audit cho thấy author là người thật, nhưng lịch sử có nhiều alias/email cho một số thành viên (ví dụ `Duy-DS`, `DY-DS` và UID noreply; `TonOfWeight072501`, `Thái Toàn`, `ToanDinh`). Không có rewrite history để chuẩn hóa: report lưu nguyên provenance. Quét tree không thấy `.env`, data thật, checkpoint/weights/TensorBoard events được track; `legacy_keras/` không có diff so với main. Quét AST xác nhận source runtime không import TensorFlow/Keras.

### Parity và smoke train

`scripts/wp02_smoke.py` là factory/dataset/criterion chung cho mười cấu hình. Mỗi case gọi `BaseTrainer.fit(..., epochs=1)` trên CPU với seed 42 và Tensor giả đúng shape; logger, epoch CSV, checkpoints của callbacks được sandbox trong thư mục tạm. Group 4 khởi tạo `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`; hooks MNISTMLP `relu2`, MNISTCNN `pool3`, và heatmap FCN đều được kiểm tra.

| Case | Baseline tham chiếu | Ghi chú |
|---|---:|---|
| FuncApxmMLP | 65 | EXP-01 không có Keras summary; config B=16, công thức `4B+1`. |
| MNISTMLP | 669,706 | Exact. |
| MNISTCNN | 1,817,610 | Exact. |
| HumanGenderCNN | 4,063,570 | Exact; WIKI rotation `k=3` kiểm thử riêng. |
| EmotionCNN | 19,028,287 | Exact. |
| MNISTCNNLSTM | 1,747,073 | Exact; input `(B,8,1,28,28)`, LSTM `batch_first=True`. Recurrent bias PyTorch zero cố định là buffer không trainable, tránh truyền None vào fused LSTM mà vẫn giữ số parameter Keras. |
| Gender + Callbacks | 4,063,570 | Cùng kiến trúc; callbacks được chạy. |
| Emotion + Callbacks | 19,028,287 | Cùng kiến trúc; callbacks được chạy. |
| SegNet | 29,458,211 | Exact, gồm 15,886 BN state non-trainable. |
| FCNObjectDetector | 83,367 | Exact; score heatmap được tạo trên đầu vào lớn. |

### CLI chuẩn

```bash
python -m src.train --config configs/cnn_mnist.yaml --seed 42 --epochs 1 --smoke-test
python -m src.train --task all --smoke-test
```

`--task all` chỉ cho phép synthetic smoke, không chạy 10 bộ train chính thức ngoài ý muốn. Training cases dùng dữ liệu tổng hợp NCHW/sequence đúng schema và `BaseTrainer`; logs/checkpoints được tạo trong temporary directory. CLI không thay đổi `master_experiment_runs.csv` thật. Task `configured` không bật smoke sẽ dispatch runner nhóm tương ứng, do đó yêu cầu dữ liệu thật và logger hoạt động.

### Trạng thái môi trường

PyTorch train/validation smoke chạy được trên CPU cho cả 10 case. Tuy nhiên, `ExperimentLogger` không thể khởi tạo TensorBoard do TensorBoard 2.21.0 protobuf gencode 6.31.1 không tương thích protobuf runtime 5.29.6; vì vậy logger status bị đánh dấu `BLOCKED`, và CLI all-smoke trả mã 2 (không báo PASS toàn phần). Không nâng/cài package khi chưa được duyệt; cần sửa dependency lock/runtime trong tác vụ môi trường riêng rồi chạy lại logger/full suite. Smoke data không chứng minh accuracy baseline Keras.

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

---

## 8. T0201/T0202 — Nhóm 1 MLP PyTorch

### `FuncApxmMLP`: xấp xỉ mật độ Gaussian hai biến

`src/datasets/mlp_datasets.py` tạo chính xác lưới legacy `np.mgrid[-3.0:3.0:32j, -3.0:3.0:32j]` (1.024 điểm) và lưới test `10j` (100 điểm). Nhãn dùng mật độ chuẩn với `mu=[0,0]`, `Sigma=diag([1,1])`:

```text
f(x1,x2) = exp(-0.5 * (x1²+x2²)) / (2π)
```

Chỉ số train dùng `RandomState(seed).choice(range(1,1001), A, replace=False)`, như notebook gốc; do đó A tối đa 1.000 và cùng seed tái tạo cùng mẫu. Dataset giữ riêng `full_inputs/full_targets` để tính lỗi trên toàn lưới 1.024 điểm, không nhầm với subset dùng cập nhật gradient.

`FuncApxmMLP` giữ tên tầng `fc` và `gauss_apxm`: `(N,2) → Linear(2,B) → Sigmoid → Linear(B,1) → Sigmoid → (N,1)`. Cả hai activation sigmoid đều có trong notebook legacy. Số tham số là `(2B+B)+(B+1)=4B+1`, tức 65 khi B=16. SGD dùng lr 0.01, batch 128, `MSELoss`; các sweep giữ đúng các mốc A=100..1000 (B=16,C=100), B=8..512 theo lũy thừa 2 (A=800,C=100), và C=10..190 (A=800,B=128) như notebook gốc.

Notebook Keras cũ không lưu đường cong hội tụ hay MSE số của EXP-01. Vì thế log PyTorch có epoch curve và bảng sweep có thể tái lập; đối chiếu định lượng Keras chỉ có thể bổ sung sau khi chạy lại notebook legacy trên cùng dữ liệu/seed/runtime. Không được trình bày metric Keras bị thiếu như thể đã quan sát.

### `MNISTMLP`: phân loại chữ số và parity tham số

Luồng tương đương Keras là:

```text
(N,28,28) hoặc (N,1,28,28)
→ Flatten (N,784)
→ fc1: Linear(784,512) → ReLU → Dropout(0.2)
→ fc2: Linear(512,512) → ReLU → Dropout(0.2)
→ fc3_10ways_softmax: Linear(512,10) logits
```

Tổng tham số: `(784×512+512)+(512×512+512)+(512×10+10)=669,706`, khớp Keras. Tầng cuối PyTorch trả logits thô: `nn.CrossEntropyLoss` nội bộ tính log-softmax và negative log likelihood, nên **không** thêm softmax trước loss. Keras `categorical_crossentropy` nhận vector one-hot cùng xác suất softmax; khi đổi sang PyTorch, nhãn one-hot được đổi về chỉ số lớp `torch.long` (0..9). `predict_proba()` chỉ áp dụng `softmax(dim=1)` khi thật sự cần xác suất suy luận.

Keras dùng `validation_data=(x_test,y_test)` rồi đánh giá lại đúng split đó; baseline test accuracy 0.9648 vì vậy không phải holdout độc lập với validation. Runner PyTorch giữ test chuẩn 10.000 ảnh ngoài vòng chọn checkpoint: validation được tách từ 60.000 train, checkpoint chọn theo validation loss, còn test chỉ đánh giá sau khi chọn checkpoint.

### Forward hook đặc trưng `fc2`

`fc2` là phép tuyến tính; activation cần cho PCA là đầu ra ReLU ngay sau đó. Vì vậy model đặt `self.fc2` và `self.relu2` thành module riêng; đăng ký `register_feature_hooks(["relu2"])` trong `eval()` bắt đúng activation sau ReLU, trước dropout. `extract_features()` trả tensor đã `detach()`; runner ghép batch theo thứ tự test, gỡ hook bằng `clear_hooks()`, rồi lưu `features: (10000,512)` và `labels: (10000,)` vào NPZ. Dropout ở eval không làm rơi feature.

### Bảng đối chiếu layer

| Keras | PyTorch | Shape/đầu ra | Tham số |
|---|---|---:|---:|
| `Flatten` | `nn.Flatten(start_dim=1)` | `(N,28,28)→(N,784)` | 0 |
| `Dense(512,relu)` | `fc1` + `relu1` | `(N,784)→(N,512)` | 401,920 |
| `Dropout(0.2)` | `dropout1` | `(N,512)` | 0 |
| `Dense(512,relu)` | `fc2` + `relu2` | `(N,512)→(N,512)` | 262,656 |
| `Dropout(0.2)` | `dropout2` | `(N,512)` | 0 |
| `Dense(10,softmax)` | `fc3_10ways_softmax` logits | `(N,512)→(N,10)` | 5,130 |
| **Tổng** |  |  | **669,706** |

### Câu hỏi vấn đáp

**Vì sao không softmax logits trước `CrossEntropyLoss`?** PyTorch loss đã kết hợp log-softmax ổn định số học với NLL; softmax trước đó vừa thừa vừa làm sai thang đầu vào.

**Vì sao Keras one-hot được đổi thành `torch.long`?** `CrossEntropyLoss` mặc định nhận một class index nguyên cho mỗi ảnh, không phải vector one-hot; class index cũng giảm bộ nhớ nhãn.

**Hook đặt ở `fc2` hay `relu2`?** Tên `fc2` giữ phép `Linear` đúng kiến trúc. Đặc trưng sau kích hoạt ReLU phải hook `relu2`; hook `fc2` sẽ giữ cả giá trị âm trước ReLU.

**Dropout có làm thay đổi PCA giữa các lượt trích xuất?** Không khi gọi `model.eval()`: Dropout bị vô hiệu, các activation và feature cố định theo weights/input.

**Test accuracy có dùng chọn checkpoint không?** Không. Chỉ validation split từ train chọn checkpoint; test được mở một lần sau khi nạp checkpoint tốt nhất.

**Vì sao kết quả có thể lệch baseline Keras 96.48%?** Khác biệt thứ tự batch/shuffle, initializer và backend SGD có thể tạo quỹ đạo tối ưu khác. Cần cố định split/seed, initializer, batch/epochs và so sánh trong cùng runtime; không diễn giải sai số một lần chạy thành lỗi kiến trúc.

---

## 9. T0203–T0205 — Nhóm 2 CNN + MLP chuyển sang PyTorch

### Hợp đồng tensor và tiền xử lý

`MNISTDataset`, `WikiGenderDataset` và `FEREmotionDataset` trả tensor ảnh `(N,C,H,W)` kiểu `torch.float32`; nhãn được đổi từ one-hot sang `torch.long` class index. MNIST byte chia 255 như notebook gốc. WIKI thực hiện `np.rot90(images, k=3, axes=(1,2))` trên NHWC trước khi `permute(0,3,1,2)`. FER basic legacy lấy 2304 cột pixel trực tiếp và không chia 255; runner dùng `normalization: preserve` để không tự đổi thang dữ liệu.

Với ảnh vuông 64×64, `k=3` là xoay 270° ngược chiều kim đồng hồ (90° theo chiều kim đồng hồ). Tọa độ pixel `(y,x)` ánh xạ thành `(x,63-y)`. Phép xoay thay vị trí pixel; phép `permute` sau đó chỉ đổi layout `(N,H,W,C)→(N,C,H,W)`. Ảnh xác minh runner tạo là `outputs/figures/t0204_wiki_rot90_verify.png`; khi chỉ smoke-test bằng dữ liệu tổng hợp, hình đó không chứng minh orientation của WIKI thật.

### MNISTCNN và hook `pool3`

Notebook `1_MNIST_CNN.ipynb` dùng Conv2D 64/128/256 kênh, mỗi convolution `padding='same'`, theo sau bởi MaxPool2D và Dropout 0.2. Kích thước đi qua pooling là `28→14→7→3`; `pool3` là MaxPooling2D cuối, không phải GlobalAveragePooling2D. Flatten map `(256,3,3)` thành 2304, rồi Dense 512 hai lần và Dense 10.

PyTorch `padding=1` với kernel 3×3 cho kích thước `H_out = floor((H+2P−K)/S)+1 = H` khi stride 1, tương đương Keras `same`. MaxPool2d kernel/stride 2 dùng floor: 28→14→7→3. Hook đăng ký `pool3`, nhận `(B,256,3,3)`; runner làm phẳng mỗi activation thành `(B,2304)` và ghép nhãn test theo đúng thứ tự DataLoader. Keras summary và PyTorch đều có **1,817,610 tham số**. Không thay Flatten bằng GAP vì khi đó Dense đầu vào chỉ 256, làm thay đổi số params và mô hình.

### HumanGenderCNN

WIKI CNN có 5 block với channel `16,32,64,128,256`; số convolution mỗi block lần lượt `2,3,3,3,3`, tổng 14? Cụ thể block đầu 2 và bốn block sau 3, tổng **14 convolution**. Mỗi Conv3×3 dùng same padding; sau mỗi block MaxPool2d giảm đôi, Dropout 0.2. Sau pool thứ năm, 64→32→16→8→4→2; Flatten `256×2×2=1024`; hai Dense 1024 và Dense output 2. Tổng **4,063,570 params**.

`padding='same'` của Keras giữ kích thước khi stride 1. Với kernel lẻ 3, PyTorch `padding=1` cho kết quả này. Nếu kernel chẵn hoặc stride khác 1, Keras có thể phân bổ padding bất đối xứng; chỉ dùng đối chiếu công thức trước khi thay bằng padding đối xứng PyTorch.

Baseline WIKI basic lưu validation accuracy **82.6408%**. Mức **87.7346%** trong baseline CSV là notebook khác có callbacks (T0209/nhóm training utilities); không dùng nó để tuyên bố đã tái lập baseline basic.

### EmotionCNN và bằng chứng overfitting legacy

FER2013 sử dụng Conv 64 same → Conv 64 valid → pool; Conv 128 same → Conv 128 valid → Conv 218 same → Conv 218 valid → pool. Công thức convolution stride 1, dilation 1: same giữ H/W; valid cho `H_out=H−K+1`, do đó `48→48→46→23→23→21→21→19→9`. Flatten `218×9×9=17,658`, Dense 1024 rồi Dense 7. Tổng **19,028,287 params**, khớp Keras. Đây là bản basic 15 epoch, Adam `1e-4`, batch 128; không thêm callbacks hoặc augmentation.

Log Keras legacy cho epoch 15: Train accuracy 0.8660, validation accuracy 0.5954, chênh lệch 27.06 điểm phần trăm. Validation loss đạt thấp nhất đã lưu 1.1372 ở epoch 9, rồi tăng tới 1.4070 ở epoch 15. Đây là bằng chứng overfitting trong lịch sử notebook. Tập validation dùng public-test CSV theo legacy nên không phải holdout độc lập; các số trên không phải kết quả chạy PyTorch mới. Mạng có nhiều tham số dense so với biểu diễn ảnh grayscale 48×48 là một yếu tố cần khảo sát; T0209 mới là nơi đối chiếu callbacks/augmentation có kiểm soát.

### Công thức kích thước và tham số

- Conv2d: `H_out=floor((H+2P−D(K−1)−1)/S+1)` (tương tự W). Số tham số có bias: `C_out×(C_in×K_h×K_w+1)`.
- MaxPool2d: dùng cùng công thức hình học, nhưng không có tham số học.
- Linear: `N_params = in_features×out_features + out_features` nếu bật bias.
- GlobalAveragePooling2D Keras tương ứng `AdaptiveAvgPool2d((1,1))` khi kiến trúc thật có GAP. Ba CNN legacy Nhóm 2 đang xét không dùng GAP; `pool3` MNIST là MaxPool2D.
- PyTorch CrossEntropyLoss nhận logits và nhãn class index; không gọi softmax trước criterion. Khi cần xác suất dự đoán để hiển thị, mới dùng `softmax(dim=1)` sau forward.

### Câu hỏi vấn đáp

**Vì sao phải xoay WIKI trước permute?** Notebook gốc xoay hai trục không gian NHWC `(1,2)`. Sau permute, các chỉ số trục đã đổi; dùng nhầm axes có thể xoay channel hoặc batch thay vì ảnh.

**Tại sao `pool3` không phải GAP?** Tên `pool3` trong Keras thuộc MaxPooling2D cuối. Đổi sang GAP sẽ mất không gian 3×3 và đổi Dense input/parameter count, không còn parity.

**`padding='same'` có luôn tương đương `padding=1` không?** Với kernel 3×3, stride 1 thì có; kernel chẵn/stride khác có thể cần padding bất đối xứng và phải tính riêng.

**Vì sao EmotionCNN có khoảng 19M tham số dù input 48×48?** Sau convolutions còn 17,658 activations được nối đầy đủ tới 1024 units; riêng Dense này có 18,082,816 weights+biases. Đây là phần chi phối parameter count.

**Có thể ghi WIKI 87.7% là kết quả T0204 không?** Không cho baseline basic: số đó thuộc biến thể callbacks. Baseline basic CSV ghi 82.64%; split và biến thể phải được nêu rõ.

**Các smoke-test có thể xác nhận accuracy mục tiêu không?** Không. Chúng chỉ kiểm tra shape, preprocessing, parameter parity, hook và wiring bằng dữ liệu giả. Accuracy cần train/evaluate trên dữ liệu thật.

---

## 10. T0206/T0207 — Chuỗi MNIST và CNN-LSTM

### Sinh chuỗi zero-padded có thể tái lập

`SequentialMNISTDataset` nhận ảnh/nhãn MNIST qua pipeline `MNISTDataset`, nên mỗi frame được đổi từ byte sang float32 `[0,1]` rồi layout `(1,28,28)`. Với mỗi mẫu, bộ sinh chọn `numToAdd = ceil(U×8)` (1..8), lấy chỉ số ảnh có hoàn lại, đặt ảnh ở đầu chuỗi và giữ các frame còn lại bằng 0. Target là tổng digit của đúng các frame đã chọn. Mỗi item trả `(8,1,28,28)` cùng target float32 `(1,)`; DataLoader ghép thành `(B,8,1,28,28)` và `(B,1)`.

Kế hoạch số lượng frame và chỉ số nguồn được sinh một lần bằng `np.random.default_rng(42)` cùng stream tách riêng cho train/test; dataset lưu index/kế hoạch chứ không nhân đôi cả tập thành mảng 5D trong RAM. Cùng seed và cùng split tái tạo giống nhau; train/test không dùng chung luồng RNG. Notebook gốc đặt seed NumPy 2018 cho toàn bộ vòng lặp; bản chuyển đổi tuân thủ seed toàn cục dự án 42 nhưng giữ thuật toán phân phối, zero-padding và lấy mẫu có hoàn lại. Hình minh họa được tạo bằng `--dataset-only` ở `outputs/figures/t0206_seq_mnist_samples.png`; nếu chạy smoke thì đó là dữ liệu giả, không phải ảnh MNIST thật.

### CNN thay `TimeDistributed`

Đầu vào `(B,T,C,H,W)=(B,8,1,28,28)` được reshape thành `(B×8,1,28,28)` để Conv2d xử lý các frame độc lập cùng trọng số. Ba block Conv 3×3 padding 1 + ReLU + MaxPool2d + Dropout cho kích thước `28→14→7→3`; frame cuối có `(B×8,256,3,3)`. Flatten thành `(B×8,2304)`, sau đó reshape trở lại `(B,8,2304)` theo đúng thứ tự batch/time ban đầu.

LSTM thứ nhất có `batch_first=True`, nhận `(B,8,2304)` và giữ output toàn chuỗi `(B,8,128)`, tương đương `return_sequences=True`. LSTM thứ hai nhận toàn chuỗi, cho `(B,8,128)`; chỉ lấy `[:, -1, :]` thành `(B,128)`, tương đương `return_sequences=False`. Dropout 0.22 và Linear 128→1 tạo dự đoán tổng `(B,1)`. Đầu ra là số hồi quy, không có softmax.

### Parameter parity và LSTM bias

| Layer | PyTorch shape | Params |
|---|---|---:|
| Conv1 1→64, 3×3 | 64×28×28 → pool 64×14×14 | 640 |
| Conv2 64→128, 3×3 | 128×14×14 → pool 128×7×7 | 73,856 |
| Conv3 128→256, 3×3 | 256×7×7 → pool 256×3×3 | 295,168 |
| LSTM1 2304→128 | input/recurrent kernels và bias | 1,245,696 |
| LSTM2 128→128 | input/recurrent kernels và bias | 131,584 |
| Linear 128→1 | hồi quy tổng | 129 |
| **Tổng** |  | **1,747,073** |

Keras LSTM lưu một vector bias kích thước `4H`, trong khi PyTorch `nn.LSTM` mặc định có `bias_ih` và `bias_hh` (tổng `8H`). Để parity tuyệt đối, model đăng ký `bias_hh_l0=None`, giữ `bias_ih_l0` học được; tổng số tham số của mỗi LSTM khi đó đúng công thức Keras `4H×input + 4H×H + 4H`. Initializer tùy chỉnh dùng Glorot cho input kernels, orthogonal cho recurrent kernels và bias forget bằng 1 theo mặc định Keras.

### Cấu hình, metric và giới hạn baseline

YAML giữ `seq_len=8`, seed dự án 42, SGD `0.01`, MAE, batch 128, 30 epoch, Dropout CNN 0.2 và LSTM 0.22. Runner đánh giá `mean(abs(prediction-target))`, lưu epoch history, chọn checkpoint theo validation MAE và ghi Test_Metric sau đánh giá checkpoint. Keras legacy đã dùng đúng MNIST test split làm `validation_data`, và baseline cuối ghi `val_loss=0.6502`; không có test evaluate độc lập lưu trong notebook. Vì vậy kết quả rerun trên cùng split phải ghi là validation/public-test theo protocol legacy, không gọi là test holdout độc lập. Ngưỡng MAE `0.04` là mục tiêu riêng cần kiểm chứng; không thể suy ra đã đạt từ baseline 0.6502 hay từ parameter parity.

### Câu hỏi vấn đáp

**Vì sao gộp B và T trước CNN?** Mỗi frame đi qua cùng một CNN và không tương tác với frame khác ở tầng không gian; gộp thành batch lớn hơn tương đương `TimeDistributed` và tận dụng convolution batch hiệu quả.

**Vì sao reshape lại `(B,T,F)` trước LSTM?** LSTM cần giữ trục thời gian và thứ tự frame; nếu không phục hồi đúng B/T, recurrent state sẽ trộn chuỗi khác nhau.

**LSTM thứ nhất và thứ hai khác gì?** Tầng đầu phát output cho mọi timestep để tầng hai xử lý chuỗi; tầng hai chỉ lấy output timestep cuối làm embedding toàn chuỗi.

**Vì sao bỏ `bias_hh_l0`?** PyTorch mặc định tách bias input/recurrent, còn Keras dùng một bias. Bỏ một vector bias PyTorch cho mỗi gate/layer giúp khớp 1,747,073 tham số mà vẫn giữ một bias hiệu dụng có thể học.

**Zero padding có làm LSTM dừng ở frame zero đầu tiên không?** Không; LSTM vẫn xử lý đủ 8 timestep và không có masking. Đây cũng là hành vi legacy; các frame zero cuối vẫn ảnh hưởng trạng thái recurrent qua bias/hidden dynamics.

**Có thể tuyên bố MAE ≤0.04 chỉ từ smoke test không?** Không. Smoke chỉ xác nhận shape, tính tái lập, parameter parity, forward/backward wiring và artifacts. MAE cần đánh giá mô hình đã huấn luyện trên split MNIST được ghi rõ.

---

## 11. T0208/T0209 — Callbacks PyTorch và TensorBoard cho WP Utilities

### Vòng đời OOP và tương thích `BaseTrainer`

`BaseCallback` có bốn hook vòng đời: `on_train_begin`, `on_epoch_begin`, `on_epoch_end`, `on_train_end`. `CallbackList` gắn `trainer/model/optimizer`, reset state theo run, dispatch các hook theo thứ tự cấu hình và hợp nhất tín hiệu dừng. Với callback cũ dạng `on_epoch_end(trainer, epoch, logs)`, bộ điều phối giữ adapter tương thích để các test/task trước không phải sửa API.

`BaseTrainer.fit()` tạo `logs` mỗi epoch với `train_loss`, `val_loss`, `train_metric`, `val_metric`, alias `train_acc/val_acc`, `lr/learning_rate`, `epoch_time_sec`, cùng tham chiếu `model` và `optimizer`. Trainer lưu best validation checkpoint nền trước khi callbacks xử lý; ghi epoch metrics vào ExperimentLogger; chạy callback scheduler/early stop; và truyền `stop_training=True` khi callback yêu cầu. LR được log là LR thực đã dùng để train epoch đó; nếu scheduler đổi LR cuối epoch, bậc mới xuất hiện từ epoch kế tiếp.

### EarlyStopping

Callback theo dõi `monitor` theo `mode=min/max`, với cải thiện chỉ được tính khi vượt `min_delta`. Khi cải thiện, lưu `copy.deepcopy(model.state_dict())`, best epoch và logs; nếu monitor không cải thiện đủ patience, đặt `stopped_epoch`, bật `trainer.stop_training`, khôi phục state tốt nhất khi được cấu hình. Checkpoint callback có hook train-end đồng bộ best model với file nghiệm thu ổn định. Một lưu ý: optimizer/LR scheduler là state riêng; khôi phục model weights không đồng nghĩa rollback toàn bộ lịch optimizer về best epoch.

### ModelCheckpoint và `CheckpointManager`

`ModelCheckpoint` chỉ ghi file `.pt/.pth` đúng khi validation monitor cải thiện (`save_best_only=True`), tạo thư mục cha và gọi `CheckpointManager` để lưu atomic metadata: epoch, model/optimizer state, best metric, seed, config, monitor/best score. Chỉ mirror lên Drive khi truyền/configure mount root đang tồn tại; lớp callback không tự mount Drive. `save_weights_only=True` giữ định dạng checkpoint tương thích nhưng bỏ optimizer state.

### ReduceLROnPlateau

Scheduler đếm các epoch không cải thiện vượt `min_delta`. Khi đạt patience, cập nhật tất cả `optimizer.param_groups`: `lr_new=max(lr_old×factor,min_lr)`, reset bộ đếm và bật cooldown. Nếu LR đã ở `min_lr`, không ghi nhận một lần giảm giả. `lr_changes` lưu `(epoch, lr_cũ, lr_mới)` để báo cáo/đánh dấu biểu đồ.

### TensorBoard và schema CSV

`ExperimentLogger` dùng `torch.utils.tensorboard.SummaryWriter` ghi event binary cho `Loss/Train`, `Loss/Validation`, `Metric/Train/Accuracy`, `Metric/Validation/Accuracy`, `Hyperparameters/Learning_Rate`; CSV epoch giữ loss, metric, LR và thời gian. Master CSV giữ schema 18 cột T0104; không thêm cột vào master để tránh phá Google Sheets. `T0209` bổ sung CSV history riêng và biểu đồ Loss/Accuracy/LR, có marker best epoch, LR reduction và EarlyStopping.

### Baseline gốc và phạm vi cải tiến

- WIKI callback legacy: Adam `0.001`, batch 128, tối đa 100.000 epoch; ModelCheckpoint theo `val_loss`, EarlyStopping patience 5, ReduceLROnPlateau factor 0.1/patience 5/min_lr 0.0001, TensorBoard. Vì min_lr bằng LR ban đầu, cấu hình này không có khoảng giảm LR hữu hiệu dưới mức khởi đầu. Baseline output dừng tại epoch 32; printed accuracy `0.8773458` dùng validation WIKI, không phải holdout độc lập.
- FER callback legacy: Adam `0.0001`, batch 128, tối đa 100 epoch, EarlyStopping patience 5 và ImageDataGenerator (xoay ±45°, shift ngang/dọc 0.2, lật ngang); output dừng epoch 57. Legacy **không** khai báo ModelCheckpoint hoặc ReduceLROnPlateau. Bản T0209 thêm hai tiện ích đó như một extension có kiểm soát; không được gán extension là hành vi gốc.
- Validation/public-test WIKI/FER được tái sử dụng như notebook legacy; metric printed không được gọi independent final test.
- Workspace hiện thiếu dữ liệu WIKI/FER, CUDA và có lỗi tương thích TensorBoard/Protobuf. Vì vậy chỉ được báo cáo unit test/parity; không bịa số liệu cải thiện hội tụ mới. So sánh trước/sau chỉ hoàn tất sau khi chạy dữ liệu thật cùng protocol và ghi history đầy đủ.

### Câu hỏi vấn đáp

**`min_delta` khác `patience` thế nào?** `min_delta` định nghĩa mức thay đổi tối thiểu để gọi là cải thiện; `patience` là số epoch liên tiếp không có cải thiện trước khi hành động.

**Vì sao checkpoint phải được lưu trước EarlyStopping restore?** Trước callback, model đang ở state epoch vừa đo. Checkpoint callback lưu state nếu đó là best; EarlyStopping cuối callback chain có thể khôi phục weights tốt nhất. Hook train-end đồng bộ file checkpoint ổn định với model đã restore.

**Scheduler giảm LR ở epoch nào được log?** Epoch đang kết thúc dùng LR cũ; scheduler cập nhật ở cuối epoch nên LR mới được logger ghi từ epoch kế tiếp, đúng với LR đã tham gia forward/backward.

**WIKI `min_lr=0.0001` có giảm LR khi bắt đầu ở 0.001 không?** Có thể giảm một lần từ 0.001 xuống 0.0001; sau đó chạm floor. Baseline không thể giảm nhiều bậc thấp hơn `1e-4`.

**FER T0209 có phải bản callback Keras nguyên xi không?** Không hoàn toàn: legacy chỉ có EarlyStopping + augmentation; ModelCheckpoint/ReduceLROnPlateau là phần mở rộng T0209 và phải ghi rõ trong YAML/báo cáo.

**Một test smoke có chứng minh TensorBoard ghi event không?** Chỉ khi dùng SummaryWriter thật và xác nhận event file/scalars. Nếu môi trường SummaryWriter import lỗi, smoke cần dừng với blocker rõ ràng; không thay bằng writer giả rồi tuyên bố TensorBoard pass.