# 📄 Post-Execution Report: Task T0106

- **Date**: 2026-11-23
- **Target Module**: `src/utils.py`, `src/models/base_model.py`, `src/engine/trainer.py`, `src/train.py`
- **Task Goal**: Xây dựng core OOP model/trainer, seed utilities và kiểm thử reproducibility CPU/CUDA có điều kiện.
- **Author / Executor**: `Duy-DS` (giá trị `MEMBER_01_NAME` trong `.env`; email không ghi trong report)
- **Execution Engine**: Python 3.11.1, PyTorch 2.13.0+cpu, NumPy 2.4.6, pytest 9.1.1
- **Overall Result**: PASSED WITH LIMITATIONS

---

## 1. 📊 EXECUTIVE SUMMARY & ACCEPTANCE MATRIX

| Deliverable / Requirement | Component File | Status | Tests Executed | Passed / Failed | Details |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Reproducibility helpers | `src/utils.py` | PASSED | Seed Python/NumPy/Torch/DataLoader | 1 / 0 | `seed_everything`, `seed_worker`, `get_torch_generator`; giữ nguyên `ExperimentLogger`, `CheckpointManager`, `save_checkpoint`. |
| Base model | `src/models/base_model.py` | PASSED | Params, NHWC/NCHW, summary, hooks | 1 / 0 | `BaseMigratedModel` abstract, đếm total/trainable/non-trainable, summary fallback và hook cleanup. |
| Base trainer | `src/engine/trainer.py` | PASSED | Train/validation, callback, logger/checkpoint | 1 / 0 | Vòng lặp tường minh, loss trung bình theo số mẫu, validation inference mode, checkpoint theo val loss. |
| CLI/export | `src/train.py`, `src/models/__init__.py`, `src/engine/__init__.py` | PASSED | `python -m src.train --help`, `--help-core` | 2 / 0 | Export core API; không giả lập pipeline model chưa được nối. |
| Reproducibility suite | `tests/test_core_engine.py` | PASSED CPU | 5 core tests | 5 / 0 | Hai lượt cùng seed cho weights/history/final state giống hệt; NCHW, counts, hooks, no-gradient validation và callback. |
| Tương thích T0104 | `ExperimentLogger`, `CheckpointManager` | PASSED | Full `tests/` suite | 18 / 0 | Trainer dùng API hiện có; test suite T0102–T0104 vẫn pass. |

---

## 2. 🛡️ GUARDRAILS & INVARIANT VALIDATION

- [x] **Seed=42**: Python, NumPy, PyTorch CPU/CUDA; DataLoader worker và generator có helper riêng.
- [x] **Deterministic settings**: bật strict deterministic algorithms, `cudnn.deterministic=True`, `cudnn.benchmark=False` khi `deterministic=True`.
- [x] **NCHW validation**: `BaseMigratedModel` phát hiện NHWC điển hình và báo cách permute.
- [x] **Parameter parity counters**: total/trainable/non-trainable đếm riêng.
- [x] **Feature hook lifecycle**: đăng ký theo module name; activation detach; có `clear_hooks()`.
- [x] **Train/validation separation**: train gọi zero_grad/backward/step; validation chạy `eval()` + `torch.inference_mode()` và không tạo gradient.
- [x] **T0104 integration**: logger phải có alias thành viên thật; checkpoint manager nhận model/optimizer state, seed/config.
- [x] **Git protection**: `.env`, data, outputs, weights/TensorBoard events bị ignore; `legacy_keras/` không đổi.
- [ ] **CUDA verification**: chưa chạy vì interpreter hiện tại là `torch 2.13.0+cpu`; CUDA parity sẽ tự bật trên máy có GPU.
- [ ] **Project Charter Markdown**: `docs/project_charter/Project_Charter_BTL03.md` không tồn tại; WBS `AGENTS.md` liệt kê WP1 đến T0105, trong khi yêu cầu trực tiếp chỉ định T0106 cho Member_01. Làm theo chỉ định task; không tự sửa WBS.

---

## 3. 🧪 DETAILED ACCEPTANCE FINDINGS

### A. Functional Implementation

- `seed_everything(seed=42, deterministic=True)` đặt `PYTHONHASHSEED`, `random`, NumPy, Torch CPU/CUDA và cuDNN. Ghi chú trong docstring: đặt `PYTHONHASHSEED` sau khi interpreter đã khởi chạy không thể đổi hash seed của tiến trình hiện tại; cần set trước khi launch để bảo đảm hash iteration.
- `seed_worker` lấy seed worker do PyTorch cấp rồi seed Python/NumPy/Torch; `get_torch_generator` trả generator riêng cho shuffle.
- `BaseMigratedModel(nn.Module, ABC)` định nghĩa forward trừu tượng, validate NCHW, `count_parameters()`, summary fallback không phụ thuộc `torchinfo`, hook-based feature extraction, cleanup và Glorot/Xavier Uniform initialization cho Linear/Conv2d.
- `BaseTrainer` nhận `nn.Module` thay vì ép mọi model phải kế thừa base class ngay lập tức; điều này giữ tương thích khi WP02 chuyển đổi từng kiến trúc. Nếu model có `validate_input_shape()` và batch ảnh 4D, trainer gọi validator trước forward.
- Epoch train dùng `zero_grad(set_to_none=True) → forward → criterion → backward → optimizer.step`; train/val loss và metric trung bình theo sample count.
- Validation gọi `model.eval()` trong `torch.inference_mode()`, không tính gradient. Best checkpoint được chọn theo validation loss và nhận epoch, state dicts, best metric, seed, config.
- Callback dùng giao diện `on_epoch_end(trainer, epoch, logs)`; trả `True` để yêu cầu dừng.
- `fit()` tích hợp logger T0104 (`log_epoch`, `finish_run`) và checkpoint manager theo duck-typing; alias thiếu thì lỗi rõ ràng.
- CLI `python -m src.train --help` và `--help-core` hoạt động; config model cụ thể báo lỗi rõ vì pipeline chưa nối.

### B. Testing & Quality Gates

- `python -m pytest tests/test_core_engine.py -q` — **5 passed**, gồm reproducibility 3 epoch, helpers, params/NHWC/hooks, validation no-grad/callback và alias guard.
- `python -m pytest tests/ -q` — **18 passed, 0 failed** (9.26 giây sau thay đổi inference mode).
- `python -m py_compile ...` — PASS.
- `python -m src.train --help` và `--help-core` — PASS.
- Determinism đã xác minh chính xác trên CPU: initial state, lịch sử train loss và final weights bằng nhau giữa hai lượt cùng seed.
- CUDA không khả dụng nên không tuyên bố đã kiểm chứng CUDA. Strict mode sẽ raise nếu toán tử CUDA không có implementation tất định.
- Không tạo checkpoint/output lâu dài; test dùng `tmp_path`. `.env` không hiển thị và không stage.

---

## 4. ⏱️ PERFORMANCE & EXECUTION METRICS

- Full suite: 18 tests, khoảng 9.26 giây trên CPU trong lần chạy cuối.
- Core reproducibility test train model giả 3 epoch; đây là kiểm tra tất định, không phải benchmark hiệu năng.
- CUDA/GPU time: chưa đo; không có CUDA device trong môi trường.
- Tạo/đọc checkpoint test trong thư mục tạm; không để lại `.pt/.pth` trong workspace.

---

## 5. 🚩 OUTSTANDING ISSUES & RECOMMENDATIONS

- **Current Limitations**:
  - CUDA determinism chưa được chạy thực tế; các operator/backend không hỗ trợ deterministic có thể raise trong strict mode.
  - Determinism không đảm bảo bitwise giống nhau xuyên phiên bản PyTorch, CUDA/cuDNN và phần cứng khác nhau.
  - `src/train.py` cung cấp API/CLI trợ giúp nhưng chưa phải runner huấn luyện một trong 10 model.
  - Summary fallback báo shape/params của leaf layers, không thay thế mọi tính năng trình bày của torchinfo.
  - `BaseTrainer` chấp nhận mọi `nn.Module` để giữ tương thích chuyển đổi từng bước; WP02 nên kế thừa `BaseMigratedModel` để có validation/hook thống nhất.
- **Recommendations for Next Task**: WP02 kế thừa `BaseMigratedModel`, gọi `seed_everything(42)` trước khi tạo model/dataloader, dùng `get_torch_generator` + `seed_worker`, truyền `ExperimentLogger` với alias thật, chọn checkpoint theo validation và chạy cùng suite trên CUDA khi GPU được cấp.