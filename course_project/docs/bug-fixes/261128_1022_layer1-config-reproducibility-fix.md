# [BUG-FIX] Khắc phục Tầng 1: Quản trị Cấu hình, Môi trường, Tính Tái lập & Dọn dẹp Git Index

- **Thời gian:** 2026-11-28 10:22
- **Người thực hiện:** Đinh Thái Toàn
- **Nhánh Git:** `fix/BPM25_T0305_wp03-training-audit`
- **Task ID liên quan:** `BPM25_T0305_LAYER1_AUDIT` -> Khắc phục Tầng 1

---

## 1. Triệu chứng lỗi & Vấn đề từ Kiểm toán Tầng 1
1. **Git Index Pollution:** Hàng loạt file dữ liệu (`data/preprocess/*.csv`, `standard_scaler.pkl`), trọng số mô hình (`outputs/checkpoints/**/*.pt`), logs (`outputs/logs/*.csv`) và cache (`__pycache__/*.pyc`) bị Git track nhầm; `.gitignore` chưa chặn `data/preprocess/`, `outputs/`, `*.pkl`, `*.log`.
2. **Cấu hình phân tán & Hardcode:** `AGENTS.md` và `README.md` vẫn còn tham chiếu `scripts/evaluate.sh`; `configs/default.yaml` thiếu mốc ngày chia tập (`train_end`, `train_dev_end`, `dev_end`), `train_dev_stride`/`dev_stride`/`test_stride`, và `finetune.models`; `src/data_pipeline/preprocess.py` và `train_baseline.py` chưa đọc YAML; `training_loop.py` và `fine_tune.py` bỏ qua nhiều key `paths.*` và `data.*`.
3. **Lỗ hổng Reproducibility:** `training_loop.py:set_seed()` thiếu `torch.backends.cudnn.deterministic = True` và `torch.backends.cudnn.benchmark = False`; `DataLoader` không truyền `generator`; `fine_tune.py` bỏ qua `seed` khi đọc checkpoint WP03 và không gắn `_seed{seed}` vào tên file checkpoint/log fine-tune.
4. **Môi trường & Dependencies:** `requirements.txt` trống (0 bytes); thiếu `captum`, `streamlit`, `onnxruntime`, `onnx` khiến 8 test ONNX trong `tests/test_wp03_sota_fixes.py` bị lỗi `ModuleNotFoundError`.

---

## 2. Nguyên nhân gốc rễ (Root Cause)
- Các file artifact từng được commit trước khi hoàn thiện `.gitignore`.
- Quá trình nâng cấp WP03 chuyển sang `configs/default.yaml` nhưng chưa đồng bộ đầy đủ với WP01 (`preprocess.py`), WP02 (`train_baseline.py`), `fine_tune.py` và tài liệu gốc.
- `torch.onnx.export` cần cả `onnx` lẫn `onnxruntime`.

---

## 3. Các thay đổi đã thực hiện
- **`.gitignore` & Git Index:**
  - Cập nhật `.gitignore` bổ sung `data/preprocess/`, `data/processed/*` (`!data/processed/.gitkeep`), `outputs/`, `*.pkl`, `*.log` (giữ nguyên `!docs/`).
  - Chạy `git rm -r --cached` gỡ toàn bộ `data/preprocess/`, `outputs/`, `__pycache__/`, `src/**/__pycache__/` khỏi Git index (giữ nguyên file vật lý trên đĩa).
- **Tài liệu & Cấu hình (`AGENTS.md`, `README.md`, `configs/default.yaml`):**
  - Cập nhật `scripts/evaluate.sh` -> `scripts/ablation.sh` trong `AGENTS.md` và `README.md`.
  - Bổ sung `train_end`, `train_dev_end`, `dev_end`, `train_dev_stride`, `dev_stride`, `test_stride`, `finetune.models` vào `configs/default.yaml`.
- **Mã nguồn (`preprocess.py`, `training_loop.py`, `fine_tune.py`, `train_baseline.py`):**
  - `src/data_pipeline/preprocess.py`: Thêm `--config` (mặc định `configs/default.yaml`) và `load_preprocess_config()` để đọc `train_end`, `train_dev_end`, `dev_end`, `stations`, `paths.data_dir`.
  - `training_loop.py`: Bổ sung `cudnn.deterministic = True` và `cudnn.benchmark = False` vào `set_seed()`; truyền `generator=torch.Generator().manual_seed(seed)` vào `DataLoader` train; bổ sung CLI args cho `paths.*` và `data.*`.
  - `fine_tune.py`: Truyền `generator` vào `DataLoader` train; ưu tiên nạp `{model}_{mode}_H{horizon}_seed{seed}_pretrained.pt` (fallback bản legacy nếu chưa có) và lưu `{model}_{source_tag}_{mode}_H{horizon}_seed{seed}_finetuned.pt` kèm log gắn `_seed{seed}`.
  - `train_baseline.py`: Thêm `--config` và `load_baseline_yaml_config()` đọc `wp02_baseline.*`, `data.*`, `paths.*`.
- **Môi trường & `requirements.txt`:**
  - Cài đặt `captum==0.9.0`, `streamlit==1.67.0`, `onnxruntime==1.32.0`, `onnx==1.23.2`.
  - Điền đầy đủ danh sách gói chuẩn vào `requirements.txt`.

---

## 4. Bằng chứng kiểm thử (Verification Evidence)
```text
$ python -m py_compile course_project\src\data_pipeline\preprocess.py course_project\training_loop.py course_project\fine_tune.py course_project\train_baseline.py
(exit code 0)

$ python -m pytest course_project\tests\test_wp03_sota_fixes.py -v
collected 55 items
...
====================== 55 passed, 32 warnings in 21.52s =======================
```
- Tất cả 55/55 unit tests trong `tests/test_wp03_sota_fixes.py` (bao gồm 8/8 test `test_onnx_opset17_export_and_parity`) đều **PASSED**.

