# 📄 Post-Execution Report: Task T0104

- **Date**: 2026-11-23
- **Target Module**: `src/logger.py`, `src/utils.py`, `configs/paths.yaml`, `configs/kaggle_paths.yaml`
- **Task Goal**: Thiết lập Google Drive checkpoint mirror, CSV logger schema 18 cột và TensorBoard SummaryWriter.
- **Author / Executor**: `TonOfWeight072501` (giá trị `MEMBER_04_NAME` trong `.env`; email không ghi trong báo cáo)
- **Execution Engine**: Python 3.11, PyTorch 2.13.0+cpu, TensorBoard 2.21.0, pytest
- **Overall Result**: PASSED

---

## 1. 📊 EXECUTIVE SUMMARY & ACCEPTANCE MATRIX

| Deliverable / Requirement | Component File | Status | Tests Executed | Passed / Failed | Details |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Logger OOP hợp nhất CSV/TensorBoard | `src/logger.py` | PASSED | Smoke + unit tests | 4 / 0 | Ghi epoch CSV 7 cột, master CSV 18 cột, scalar groups và HParams. |
| Schema Google Sheets | `ExperimentLogger.MASTER_COLUMNS` | PASSED | Header/order assertions | 1 / 0 | Đúng thứ tự Timestamp → Member_Alias; từ chối header đã có không khớp. |
| Checkpoint local metadata | `CheckpointManager` | PASSED | Save/load round-trip | 1 / 0 | Lưu epoch, model/optimizer state, best_metric, seed=42, config; file ghi atomic. |
| Drive mirror tùy chọn | `CheckpointManager._mirror_to_drive` | PASSED | Explicit + YAML mounted directory | 2 / 0 | Ưu tiên root truyền vào/env; nếu không có thì đọc `project.drive_mount_root`; chỉ mirror khi thư mục mount tồn tại, không tự mount hoặc gọi mạng. |
| Config đường dẫn | `configs/paths.yaml`, `configs/kaggle_paths.yaml` | PASSED | YAML parse | 2 / 0 | Chứa Drive folder URL, layout checkpoint 5 nhóm, local/Kaggle path. |
| Git protection | `.gitignore` | PASSED | `git check-ignore` | 4 / 0 | TensorBoard events, `.pt/.pth`, outputs bị ignore; CSV logs nhẹ vẫn được track. |

---

## 2. 🛡️ GUARDRAILS & INVARIANT VALIDATION

- [x] **CSV schema integrity**: 18 cột master đúng thứ tự yêu cầu; epoch CSV đúng 7 cột.
- [x] **Reproducibility metadata**: Run có Seed; checkpoint có seed/config.
- [x] **Human alias**: Alias bắt buộc truyền hoặc lấy từ biến môi trường/`.env`; không có tên AI/default giả.
- [x] **Drive safety**: Không mount Drive tự động; chỉ mirror tới thư mục đã mount tồn tại.
- [x] **Config decoupling**: Drive/Kaggle/local paths nằm trong YAML hoặc tham số/env, không hardcode trong logger.
- [x] **Git protection**: `.env`, TensorBoard event, weights, outputs bị ignore; `.gitkeep` được giữ.
- [x] **Smoke isolation**: Test chạy trong thư mục tạm, không để CSV/event/checkpoint giả ở repo.

---

## 3. 🧪 DETAILED ACCEPTANCE FINDINGS

### A. Functional Implementation

- `ExperimentLogger` dùng `SummaryWriter` ghi `Loss/Train`, `Loss/Validation`, `Metric/Train/<metric>`, `Metric/Validation/<metric>`, `Hyperparameters/Learning_Rate` và `Timing/Epoch_Time_Sec`.
- `finish_run()` ghi `add_hparams(...)`, flush event writer và append đúng một dòng vào `experiments/csv_logs/master_experiment_runs.csv`.
- CSV epoch được đặt `<Run_ID>_epochs.csv`; master dùng 18 cột: `Timestamp, Run_ID, Model_Group, Model_Name, Exp_Type, Seed, Optimizer, LR, Batch_Size, Epochs, Best_Epoch, Train_Loss, Val_Loss, Train_Metric, Val_Metric, Test_Metric, Train_Time_Sec, Member_Alias`.
- `CheckpointManager` lưu local vào `outputs/checkpoints/<Model_Group>/`; metadata chứa `epoch`, `model_state_dict`, `optimizer_state_dict`, `best_metric`, `seed`, `config`. Nếu Drive root mounted tồn tại, file được mirror vào `checkpoints/<Model_Group>/`; root được lấy theo thứ tự tham số tường minh → env → YAML.
- Drive project URL được đặt trong paths YAML và README. Không lưu token/credential.
- Google Sheets là nơi copy/paste đích; không có API Sheets credential hay kết nối tự động được cấu hình.

### B. Testing & Quality Gates

- `python -m pytest tests/test_mlops_logger.py -q` — **5 passed, 0 failed** trong 7.74 giây (lần chạy logger riêng).
- Regression suite kết hợp logger, baseline extractor và vision dataset — **13 passed, 0 failed** (xác minh cuối task).
- `python scripts/smoke_test_logger.py` — **PASS**: ba epoch, đúng 18 master columns, đúng epoch rows, event file tồn tại, checkpoint load/round-trip và mirror trên Drive directory giả lập.
- YAML parse: `paths.yaml`, `kaggle_paths.yaml` — PASS.
- `.gitignore`: xác nhận `.env`, event, checkpoint/output ignore; master CSV được phép track.
- Smoke artifacts được tạo trong `TemporaryDirectory` và tự dọn khi kết thúc; không có binary runtime được thêm vào Git.

---

## 4. ⏱️ PERFORMANCE & EXECUTION METRICS

- Regression suite kết hợp: 13 test pass; thời gian dùng để kiểm thử phần mềm, không dùng làm benchmark mô hình.
- Smoke run: 3 epoch trên model `nn.Linear` giả; không đại diện hiệu năng mô hình BTL03.
- TensorBoard event được tạo trong sandbox tạm; không đưa event vào repository.

---

## 5. 🚩 OUTSTANDING ISSUES & RECOMMENDATIONS

- **Current Limitations**:
  - Không kết nối Google Sheets API; nhóm copy/paste CSV theo `Raw_TensorBoard_CSV_Logs`.
- Drive mount phải được notebook Colab thực hiện riêng; logger chỉ nhận biết mount qua `drive_root`, `BTL03_GOOGLE_DRIVE_ROOT` hoặc `project.drive_mount_root` đã tồn tại trong YAML.
  - CSV dùng lock trong process; nhiều host không nên đồng ghi chung file master nếu không có writer trung tâm.
  - `SummaryWriter.add_hparams` tạo event phụ trong TensorBoard; các scalar chính vẫn ở run directory chuẩn.
- **Recommendations for Next Task**: Tích hợp logger vào vòng train chuẩn và checkpoint selection; xác minh CSV trên Sheets; Colab adapter mount Drive và truyền mount root; giữ Kaggle artifacts trong `/kaggle/working/` rồi đóng gói `.zip` theo notebook adapter.