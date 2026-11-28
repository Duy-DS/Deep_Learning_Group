# [REPORT] BPM25_T0305: Kiểm toán huấn luyện WP03 và chẩn đoán SOTA

- **Mã công việc (Task ID):** `BPM25_T0305` (`BPM25_WP03_AUDIT`)
- **Tên file lưu vết:** `docs/reports/AUDIT_WP03_TRAINING_261128_0815.md`
- **Thời gian kiểm toán:** 2026-11-28 08:15 (+07:00)
- **Người thực hiện:** Nguyễn Tấn Duy
- **Work Package liên quan:** WP01–WP06, trọng tâm WP03
- **Nhánh Git:** `fix/BPM25_T0305_wp03-training-audit`
- **Base:** `feat/BPM25_WP04_xai_integration`, `9605ade8da03c08d921ec41197bd9d17dbec8f80`
- **Commit SHA kiểm toán:** `9605ade` (chưa commit các thay đổi trong phiên)
- **Trạng thái nghiệm thu:** Ready for Review — dừng chờ chủ dự án duyệt

> **Phạm vi tích hợp:** Đây là **selective integration**, không phải merge hoặc cherry-pick nguyên nhánh. Chỉ các đường dẫn mã/cấu hình WP03 và bộ test hồi quy từ `origin/fix/BPM25_WP03_sota-peak-capture` được đưa vào worktree để kiểm tra. Tài liệu, notebook, output và source XAI WP04 không bị lấy từ tip upstream. Đây là tích hợp theo file (không phải chọn từng hunk); do đó cần review diff toàn file, nhất là các module WP03 giao tiếp với WP04/WP05. Các sửa đổi chưa commit để tiện review/đảo lại.

---

## 1. Mục tiêu, phạm vi và phương pháp

- Đối chiếu luồng dữ liệu → cửa sổ → model → loss/evaluation → fine-tune theo bảy nhóm nghi vấn.
- Kiểm tra contract tensor, scaling, cấu hình hiệu lực, chế độ train/eval, decoder/attention, fine-tune/checkpoint.
- Xem xét nhánh và bằng chứng lịch sử WP03; không coi số liệu trong báo cáo upstream là kết quả vừa tái chạy.
- Không chạy huấn luyện, không chạy preprocess ghi artifact và không sửa logic ngoài selective integration đã được yêu cầu.

### 1.1 Trạng thái nhánh BPM25 sau `git fetch --all --prune`

| Ref | SHA tại thời điểm kiểm toán | Phạm vi / ghi chú |
|---|---|---|
| `origin/feat/BPM25_T0101_T0102_T0103_T0104_T0105_T0106` | `974c960affea9b9fc5b459717964b5cc3a90b1db` | WP01, pipeline dữ liệu |
| `origin/feat/BPM25_T0201_T0202_T0203_T0204_baseline-models` | `eba3f38e105a4e144bbea104ab61a0542f8ca9e5` | WP02, baseline |
| `feat/BPM25_WP03_SOTA_Alb` / `origin/feat/BPM25_WP03_SOTA_Alb` | `7fdab17d0f707f10fca9667a1d70c708064b5dc0` | WP03 SOTA; local và remote cùng SHA |
| `feat/BPM25_WP03_SOTA_Alb_update` / `origin/feat/BPM25_WP03_SOTA_Alb_update` | `9cf9b89a3be290eabf2f48cb6632db06a9bbdb33` | WP03 update; local và remote cùng SHA |
| `origin/feat/BPM25_T0401_shap-explainer` | `aee82852b14554a13c7d1d9265039a0a735328fe` | WP04 SHAP/XAI |
| `feat/BPM25_WP04_xai_integration` / `origin/feat/BPM25_WP04_xai_integration` | `9605ade8da03c08d921ec41197bd9d17dbec8f80` | Base audit; giữ nguyên WP04 |
| `origin/fix/BPM25_WP03_sota-peak-capture` | `3aed7193c427249b775ab190db6d05e4dcb77253` | WP03 fix/peak capture; selective integration |

Trong refs đã fetch không thấy nhánh BPM25 riêng WP05/WP06. Các nhánh công việc độc lập không có common ancestor với base khi kiểm tra `git merge-base`; vì vậy không merge nguyên nhánh và không phát sinh merge conflict. SHA trong bảng là snapshot sau fetch, không hàm ý mọi nhánh đã được hợp nhất.

### 1.2 Nội dung selective integration

Từ ba commit liên tiếp của `origin/fix/BPM25_WP03_sota-peak-capture`:

- `f7baf63`: cập nhật training loop, cấu hình, Informer/PatchTST, fine-tune, ablation và thêm test.
- `27d884d`: cập nhật stride/LR/dropout/PeakShape cho ma trận 12 trạm.
- `3aed719`: ép `num_workers=0` cho `TensorDataset` trong global training và fine-tuning để hạn chế nhân bản RAM/OOM.

Đã lấy chọn lọc các đường dẫn `course_project/training_loop.py`, `fine_tune.py`, `run_ablation.py`, `configs/default.yaml`, `src/models/informer.py`, `src/models/patchtst.py`, `tests/test_wp03_sota_fixes.py`. Không lấy các thay đổi xóa tài liệu, notebook hoặc output WP04 tại tip upstream. Tài liệu `docs/bug-fixes/261126_2325_wp03_sota_underfitting_peak_fix.md` chỉ được đọc như bằng chứng lịch sử, không được coi là kết quả kiểm toán độc lập hay tái lập trong phiên này.

---

## 2. Phát hiện và RCA theo bảy nhóm

| Mức độ | Phát hiện / bằng chứng | Nhận định |
|---|---|---|
| **Critical — blocker xác minh** | `course_project/training_loop.py:450–478` yêu cầu bốn split gồm `train.csv`. Tại kiểm tra, `course_project/data/preprocess/train.csv` **không tồn tại**; `train_dev.csv`, `dev.csv`, `test.csv` tồn tại và header cho thấy `pm2.5` cùng 30 feature. | Không đủ điều kiện tái lập pipeline end-to-end hoặc xác minh kết quả/checkpoint trong lượt này. Không tự chạy preprocess vì nó ghi artifact. |
| **Critical — blocker xác minh dữ liệu** | `course_project/training_loop.py:450–478` yêu cầu bốn split, nhưng `course_project/data/preprocess/train.csv` không có trong worktree. `train_dev.csv`, `dev.csv`, `test.csv` hiện có header D31 và giá trị PM2.5 ở thang gốc. | Không thể dựng DataLoader/train/evaluate end-to-end hoặc xác minh scaler trên split Train. Không tự chạy preprocess vì lệnh sẽ ghi artifact. |
| **High — lineage/contract scaler cần xác minh** | `src/data_pipeline/preprocess.py:154–172` có hàm scaler fit train rồi transform, nhưng `run_preprocess()` tại `:222–238` chỉ split/impute/encode/verify rồi ghi CSV, không gọi `scale_splits()` hoặc lưu scaler. Có scaler `data/preprocess/standard_scaler.pkl` với 30 feature, không gồm `pm2.5`; đồng thời có scaler runtime local `outputs/checkpoints/global/scaler_12stations_multivariate.joblib` với 31 feature gồm `pm2.5`, đúng thứ tự D31. Scaler runtime là output local/ignored và không được Git track; split Train tương ứng không còn để xác minh lineage. | Không nên kết luận scaler đang được dùng sai: đường nạp hiện hành tìm `outputs/checkpoints/global/scaler_12stations_{mode}.joblib` (`training_loop.py:481–535`), và scaler local đa biến có shape/feature contract phù hợp. Nhưng không thể xác minh nó được fit từ đúng train split hiện tại hoặc tương ứng checkpoint nào. Cần khôi phục/đối chiếu train split và provenance scaler trước khi train/reproduce. |
| **High — checkpoint fine-tune có thể chọn sai/không tồn tại** | WP03 lưu checkpoint seed-specific tại `training_loop.py:1288–1290`; `fine_tune.py:291–312` nhận `seed` nhưng resolver WP03 tạo tên không có seed. | Mismatch naming rõ trong code; cần sửa resolver hoặc thống nhất naming trước khi chạy fine-tune. |
| **High — nguy cơ unfreeze rộng hơn head** | `fine_tune.py:133–143` dùng token tổng quát `projection` cho Informer; `configure_trainable_layers()` tại `:190–204` unfreeze mọi parameter name chứa token. | Có thể mở cả query/key/value/out projection của attention, không chỉ output head. Không nhất thiết làm hỏng pretrained weights, nhưng tăng nguy cơ catastrophic forgetting và cần kiểm tra danh sách tensor thực tế. |
| **High — RevIN làm thay đổi target mapping** | `training_loop.py:154–192` chuẩn hóa channel liên tục rồi có thể inverse theo `mean/std` local; PatchTST có RevIN nội tại và inverse tại `src/models/patchtst.py:443–487`. Config tích hợp tắt cả hai (`default.yaml:150–168`). | Không thấy double normalization trên cấu hình hiện hành sau integration. Rủi ro chủ yếu là checkpoint cũ/config override và khác biệt between train/eval. Chưa thể kết luận RevIN là nguyên nhân duy nhất từ code; upstream ghi nhận A/B nhưng chưa tái chạy. |
| **Medium — scheduler/effective LR cần đối chiếu theo model** | `TrainingPipeline._resolve_model_hyperparams()` lấy LR/dropout riêng trong YAML (`training_loop.py:1222–1233`); config đặt SOTA 2e-4, LSTM 3e-4, DLinear 1e-3 (`default.yaml:139–195`). Có warmup/cosine, clipping và 25 epochs/patience 5 ở `default.yaml:91–113`. | Không còn dấu hiệu dùng một LR lớn chung cho mọi model; nhưng baseline WP02 (`train_baseline.py`) dùng trainer/optimizer khác. So sánh WP02 vs WP03 không cô lập kiến trúc/optimizer/loss. Không có cơ sở kết luận chỉ cần giảm LR SOTA. |
| **Medium — objective khác giữa SOTA pipeline và baseline** | WP03 train dùng `PeakShapeLoss` theo config (`default.yaml:101–106`; `training_loop.py:1265–1273`), còn evaluation dùng `nn.MSELoss()` (`training_loop.py:1273`, `evaluate():885–977`). WP02 BaselineTrainer sử dụng objective/optimizer riêng. | Train loss không trực tiếp so sánh được với baseline. Test MSE vẫn được tính bằng objective chuẩn sau khi model trả output; cần xác nhận mọi model cùng target scaling/feature set và cùng split trước khi diễn giải xếp hạng. |
| **Medium — Informer train-time ProbSparse ngẫu nhiên** | `src/models/informer.py:184–202`: training lấy mẫu `torch.randint`, eval deterministic nếu `deterministic_eval=True`; decoder nhận causal mask ở `:460–508`, đầu vào horizon tương lai là zeros cộng learned queries tại `:493–504`. | Có stochasticity train là đặc tính sampling, không phải bằng chứng leakage. Eval config deterministic. Không thấy future labels được đưa vào decoder; causal mask hiện diện. Cần regression test/ablation khi thay kiến trúc, nhưng không ghi nhận lỗi leakage đã chứng minh. |
| **Medium — early stopping và selection mùa** | WP03 chọn `balanced_dev` qua `train_dev` + `dev`, 25 epochs, patience 5 (`training_loop.py:812–833`, `:1310–1398`; config `:91–113`). Test chỉ đánh giá sau khi nạp checkpoint tốt nhất (`:1410–1430`). | Logic giúp tránh chọn chỉ theo dev mùa hè và không cho thấy test leakage trong code. Patience 5 có thể dừng sớm tùy convergence; chưa thể kết luận underfitting nếu không có learning curves/log theo run. |
| **Low / chưa xác nhận** | Contract output sau integration: Informer trả `[B,H,1]` (`informer.py:475–512`); PatchTST lấy `target_idx=0`, patch/unfold và trả `out.unsqueeze(-1)` (`patchtst.py:442–487`). | Không thấy squeeze nhầm batch/time trục trong forward path được đọc. Bộ test mới có shape/gradient assertions. Cần test trên đúng feature artifact và tất cả H; test upstream phần lớn đã chạy, ONNX chưa chạy ở máy này. |

### 2.1 Đối chiếu scaling, metric và Data Leakage

- **Thiết kế fit đúng:** `src/data_pipeline/preprocess.py:160–167` gọi `scaler.fit(splits["train"][feature_cols])`, rồi chỉ `transform()` từng split. `training_loop.py:515–532` cũng fit lại qua `scale_splits()` khi không tìm thấy scaler tương thích.
- **Điểm yếu cần xử lý:** heuristic `already_scaled` tại `training_loop.py:494–513` chỉ nhìn mean/std PM2.5. Nếu scaler artifact không khớp feature set/order, nhánh này có thể bỏ qua scaling hoặc chọn scaler sai; nhánh scaler đã có nhưng không mang `feature_names_in_` cũng bị transform trực tiếp tại `:515–529` mà không kiểm tra số lượng/tên/target index.
- **Bằng chứng môi trường:** scaler WP01 `standard_scaler.pkl` (30 feature) phát `InconsistentVersionWarning`: artifact được ghi bởi sklearn 1.9.1, runtime là 1.5.2. Scaler runtime trong `outputs/checkpoints/global/` có 31 feature đa biến và 1 feature đơn biến, tương ứng target contract; hai artifact có vai trò/lineage khác nhau và chưa được đối chiếu với split Train/checkpoint gốc.
- **Metrics:** `utils_metrics.py:90–118` tính MSE/MAE trực tiếp trên tensor nhận được, không inverse transform. Vì vậy metrics là scaled-space nếu input/output scaled. `training_loop.evaluate()` đổi sang µg/m³ chỉ cho metric chẩn đoán peak khi có `target_mean/target_scale`; phải xác minh mean/scale đến từ đúng scaler và target index.
- **Split/window:** `dataset.py:26–53` group theo station và tạo target bắt đầu sau lookback; không ghép cửa sổ giữa trạm. Cần giữ `shuffle=True` chỉ ở loader train, không shuffle split thời gian.

### 2.2 Fine-tuning / checkpoint

- `fine_tune.py:291–312` là mismatch seed-path nêu trên; `extract_state_dict()` và metadata có thể nhận diện format checkpoint nhưng không sửa được path mismatch.
- `fine_tune.py:172–204` đóng băng backbone trước, sau đó bật parameter nếu tên khớp token. Token `projection` cần được thay bằng tên module chính xác/whitelist sau khi duyệt.
- Upstream đã bổ sung kiểm thử `tests/test_wp03_sota_fixes.py:198–233` để kiểm tra số parameter mở khóa và param groups. 47 test đã pass tại máy này (xem Mục 4), nhưng test chỉ xác nhận điều kiện được viết; không chứng minh fine-tune cải thiện held-out test.

---

## 3. Đề xuất khắc phục (chưa áp dụng)

1. **P0 — chốt data/scaler contract với WP01:** xác định artifact train split đúng; lưu scaler fit chỉ trên train; kiểm tra scaler feature names/order/target index và số chiều trước khi transform; bỏ heuristic dựa riêng mean/std hoặc bổ sung validation contract. Không train cho tới khi `train.csv` và scaler đúng contract có mặt.
2. **P1 — sửa checkpoint resolver:** truyền đúng seed-specific checkpoint vào fine-tune hoặc chuẩn hóa tên global checkpoint; kiểm tra metadata `seed`, `mode`, `horizon`, `in_features`, `feature_cols` trước khi load.
3. **P1 — giới hạn unfreeze:** dùng danh sách module head/adapter rõ ràng theo từng architecture; test tên parameter được mở khóa, số lượng, gradient và giữ nguyên encoder khi freeze.
4. **P1 — so sánh công bằng:** chạy lại trên cùng splits/scaler/target/horizon/seeds; dùng metric chuẩn giống nhau cho LSTM, DLinear, Informer, PatchTST; báo MSE/MAE trên scaled space và inverse-transformed µg/m³ rõ ràng. Tách ảnh hưởng optimizer/loss khỏi ảnh hưởng kiến trúc bằng ablation.
5. **P2 — xác nhận hội tụ:** lưu learning curves theo seed/model/horizon; đánh giá patience/epochs bằng train, train_dev và dev, không dùng test để chọn checkpoint. Không áp dụng giảm LR/cosine khác trước khi có run log chứng minh.
6. **P2 — regression:** giữ test shape `[B,96,D] -> [B,H,1]`, finite gradients, eval determinism, decoder no-future-label, scaler train-only, strict checkpoint compatibility và fine-tune freeze/unfreeze. ONNX parity cần chạy trong môi trường có `onnxruntime`.

Không sửa các logic/đề xuất trên và không chạy lại ma trận train trong phiên này.

---

## 4. Bằng chứng thực thi và giới hạn

### Đã chạy

```text
git fetch --all --prune
git switch -c fix/BPM25_T0305_wp03-training-audit
python -B -m py_compile course_project/training_loop.py course_project/fine_tune.py course_project/run_ablation.py course_project/src/models/informer.py course_project/src/models/patchtst.py
47 passed, 8 deselected in 13.05s
```

Để bỏ qua dependency chỉ phục vụ ONNX trong lượt test CPU, test runner inject stub module `onnxruntime` và chọn `-k "not onnx_opset17_export_and_parity"`; 8 ca ONNX bị deselect, **không** được tính là pass.

```text
[PASS] StandardScaler fit trên synthetic train; dev/test transform bằng mean=4, scale=sqrt(5).
[PASS] Sliding windows không trộn station; X=(6,4,2), y=(6,2).
[PASS] training_loop.py --help (exit 0)
[PASS] fine_tune.py --help (exit 0)
[PASS] run_ablation.py --help (exit 0)
```

`git diff --check` pass; không có conflict marker trong nội dung `HEAD`. **Không chạy train, fine-tune, preprocess trên dữ liệu thật, benchmark hoặc đánh giá checkpoint.**

### Chưa xác minh được

- `onnxruntime` không cài trong môi trường; 8 ONNX export/parity cases chưa chạy.
- Thiếu `data/preprocess/train.csv`; scaler WP01 30-feature và scaler runtime WP03 31-feature là artifact khác nhau, scaler runtime không được track cùng source train trong Git. Vì vậy không xác minh được provenance end-to-end hoặc tái lập MSE/MAE SOTA.
- Các con số A/B và benchmark trong báo cáo upstream là bằng chứng do upstream ghi nhận, không phải kết quả thực nghiệm phiên này.

### Git hygiene trong snapshot base

Base `9605ade` hiện đang track 166 CSV, 136 `.pt/.pth`, 28 file `course_project/data/` không phải placeholder và 12 bytecode/cache. `.gitignore` có ignore checkpoint và một đường dẫn train CSV, nhưng ignore không tự bỏ theo dõi file đã tracked. **Không sửa `.gitignore`, không `git rm --cached`, không xóa dữ liệu** trong phiên này vì `.gitignore` protected và cần giữ artifact để chủ dự án review. Rủi ro này còn tồn tại ở snapshot audit; phải xử lý bằng task hygiene riêng sau khi thống nhất chính sách bảo toàn bằng chứng.

---

## 5. Deliverables và bàn giao

- **Selective integration chưa commit:** `configs/default.yaml`, `fine_tune.py`, `run_ablation.py`, `src/models/informer.py`, `src/models/patchtst.py`, `training_loop.py`, `tests/test_wp03_sota_fixes.py`.
- **Báo cáo:** `docs/reports/AUDIT_WP03_TRAINING_261128_0815.md`.
- **Nhật ký:** `docs/dev-logs/261128_0815.md`.
- **Tài liệu upstream không nhập:** báo cáo WP03 upstream không được chép đè vào WP04; nội dung/số liệu chỉ được trích dẫn như provenance.
- **Tiếp theo:** xin WP01 xác nhận split/scaler; duyệt mismatch checkpoint và whitelist fine-tune; sau đó mới quyết định thử nghiệm có kiểm soát. Phiên hiện tại dừng ở trạng thái Ready for Review.
