# Manifest nguồn bằng chứng

**Snapshot gốc:** `BTL03` commit `33dcd74599e86404bbc8128c8057afba82810093` (T0212, 2026-11-23). Path dưới đây tương đối với thư mục `BTL03/`. Đọc lại trạng thái Git và source trước mọi bản cập nhật mới.

| Evidence ID | Nguồn thực tế | Nội dung có thể chứng minh | Giới hạn cần ghi |
|---|---|---|---|
| E-CHARTER | `AGENTS.md`; `docs/project_charter/README_BTL03.md`, `QUY_DINH_SAN_PHAM.txt`, `Yêu cầu.txt`; workbook `.xlsx`; các PDF | Mục tiêu, chính sách, WBS và giao nộp khi trích cụ thể | `Project_Charter_BTL03.md` không tồn tại dưới tên đó; PDF chưa xác minh trích xuất thành công; WBS nguồn có mâu thuẫn |
| E-T0102 | `docs/reports/261126_T0102_data_pipeline_report.md` | Dataset loader, NCHW/rotation tests, thiếu dữ liệu thật | Người thực hiện ghi Member_02; chưa khẳng định tải MNIST/WIKI/FER thật |
| E-T0103 | `docs/reports/261127_T0103_keras_baseline_log.md`; `scripts/extract_keras_baselines.py` | Provenance baseline được parse từ notebook outputs | Không thực thi lại Keras; hardware/metadata thiếu ở một số notebook |
| E-BASELINE | `experiments/csv_logs/keras_baseline_10_models.csv` | Params, output/history đã lưu, split, cell provenance, `NOT_RECORDED` | Giá trị là notebook output legacy, không phải PyTorch result; không đổi tên metric |
| E-T0104 | `docs/reports/261127_T0104_mlops_logging_setup.md`; `src/logger.py` | Logger/schema/checkpoint tests trong run cũ | PASS lịch sử task không phủ nhận blocker protobuf ở runtime T0212 |
| E-T0106 | `docs/reports/261128_T0106_core_architecture.md`; `src/models/base_model.py`, `src/engine/trainer.py`, `src/train.py`, `src/utils.py` | Contract/core implementation và test khi T0106 chạy | Không chứng minh full model training hoặc CUDA verification |
| E-T0201 | `docs/reports/T0201_261122_2112.md`; `experiments/csv_logs/t0201_func_apxm_abc_sweep.csv`; các `T0201_*_epochs.csv` | 36 PyTorch sweep configs, logged MSE/Run_ID | Legacy numeric MSE không lưu; không kết luận Keras parity metric |
| E-T0202 | `docs/reports/T0202_261122_2112.md`; `experiments/csv_logs/master_experiment_runs.csv`; `T0202_MNISTMLP_s42_20261122_132545_epochs.csv` | Một run MNIST MLP 30 epoch, seed 42, alias, test metric/reported time | Keras/PyTorch validation protocol khác; run đó không đại diện mọi model |
| E-T0203--05 | `docs/reports/T0203_261123_1752.md` đến `T0205_261123_1752.md`; `tests/test_cnn_models.py`, configs | Kiến trúc, parity counts, preprocessing tests | Full-data training/accuracy PyTorch chưa chạy theo report |
| E-T0206--07 | `docs/reports/T0206_261123_1812.md`, `T0207_261123_1812.md`; `src/datasets/seq_mnist_dataset.py`, `src/models/cnn_lstm.py` | Dataset/model tests, shape và các parameter count đã ghi | Hai report bất đồng 1,024 params (1,746,049 so với 1,747,073); kiểm tra source/commit/counting semantics. Synthetic fixture; full MNIST training/MAE chưa xác nhận |
| E-T0208--09 | `docs/reports/T0208_261123_1858.md`, `T0209_261123_1858.md`; `src/engine/callbacks.py` | Callback code/tests, runner setup | Full WIKI/FER runs/logs bị blocker hoặc thiếu data |
| E-T0210--11 | `docs/reports/T0210_261123_1924.md`, `T0211_261123_1924.md`; `src/models/segnet.py`, `obj_detection.py` | Parity/shape/tests và metric caveats | Không có full training trên dữ liệu thật; previews không phải quality evidence |
| E-T0212 | `docs/reports/T0212_261123_2130.md`; `tests/test_wp02_smoke_all_models.py`; `scripts/wp02_smoke.py` | PR/history audit, 10 synthetic one-epoch cases, parameter audit, blocker, test counts | Overall `BLOCKED`; không dùng synthetic để báo cáo benchmark. CNN-LSTM count parity trong summary cần reconciliation với hai report T0206/T0207 |
| E-WBS | Workbook trong `docs/project_charter/`, sheet quản lý tasks | T0301--T0306/T0401--T0406 được đánh dấu Todo trong snapshot trích đọc | Workbook có thể thay đổi theo nhánh; xác nhận lại khi chốt snapshot sau |
| E-SOURCE | `src/models/`, `src/engine/`, `src/datasets/`, `src/logger.py`, `src/train.py`, `configs/` | Cấu trúc/code hiện có và cấu hình mục tiêu | Code/config không chứng minh run thực tế hoặc metric |
| E-TESTS | `tests/` và report task/T0212 | Test assertions/lần chạy được báo cáo | Đọc test source không đồng nghĩa đã chạy; report T0212 ghi 78 pass/4 fail theo shards |
| E-FIGURES | `outputs/figures/` | Hiện chỉ `.gitkeep` được Git track ở snapshot khảo sát | Không có hình metric/prediction đã track; không tự tạo thay thế |
| E-IDENTITY | `BTL03/.env` (chỉ đọc cục bộ, không commit) và report task | Alias/email thành viên được cấu hình; dùng alias public trong tài liệu | Không trích email riêng tư/credential; họ tên/MSSV chưa xác nhận |

## Quy tắc trích dẫn evidence

Gắn evidence ID ở manifest này với đường dẫn tương đối trong report hoặc bản thảo. Số Keras giữ nhãn “legacy notebook output”; số PyTorch giữ Run_ID/task report; không trộn nguồn trong một ô mà thiếu chú thích. Đọc `AGENT_THESIS_RULES.md` trước khi diễn giải.
