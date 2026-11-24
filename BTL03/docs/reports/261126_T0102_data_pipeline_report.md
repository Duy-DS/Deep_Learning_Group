# 📄 Post-Execution Report: Task T0102

- **Date**: 2026-11-22 17:22 (Asia/Ho_Chi_Minh)
- **Target Module**: `scripts/download_data.py`, `src/datasets/vision_datasets.py`
- **Task Goal**: Khởi tạo tải/kiểm tra dữ liệu MNIST, WIKI Gender, FER2013; chuẩn hóa Dataset PyTorch NCHW và xác minh phép xoay WIKI theo notebook Keras gốc.
- **Author / Executor**: Member_02 (được người dùng xác nhận phụ trách task)
- **Execution Engine**: Python 3.11.9, PyTorch 2.13.0+cpu, NumPy 2.4.6, pytest 9.1.1
- **Git Branch**: `feat/BTL03_T0102`
- **Overall Result**: PASSED WITH LIMITATIONS

---

## 1. 📊 EXECUTIVE SUMMARY & ACCEPTANCE MATRIX

Đã tạo downloader MNIST IDX không phụ thuộc `torchvision`, lớp Dataset cho ba bài toán, cấu hình YAML, dữ liệu giả deterministic, smoke-test offline và tài liệu tổ chức dữ liệu. Không tải dữ liệu MNIST/WIKI/FER thật: WIKI/FER chưa có file hoặc Google Drive URL/ID được xác thực trong workspace. Không đưa file dữ liệu hoặc ảnh kết quả vào Git.

| Deliverable / Requirement | Component File | Status | Tests Executed | Passed / Failed | Details |
| :--- | :--- | :---: | :---: | :---: | :--- |
| MNIST downloader và IDX validator | `scripts/download_data.py`, `configs/mnist.yaml` | PASSED (code) | Parser IDX fixture; CLI parse | 1 / 0 | Tải bốn endpoint IDX đã khai báo, xác thực magic/header/kích thước rồi lưu các mảng `.npy`. Chưa thực hiện tải qua mạng trong phiên này. |
| Cấu hình/kiểm tra tài nguyên Google Drive | `configs/wiki_gender.yaml`, `configs/fer_emotion.yaml`, `scripts/download_data.py` | PASSED WITH LIMITATIONS | Parse YAML; CLI `check` | 1 / 0 | Có thể điền URL hoặc Drive file ID; các giá trị hiện `null` vì không có ID nguồn xác minh. Nhánh Drive cần `gdown`, chưa cài/chạy. |
| Dataset MNIST/WIKI/FER chuẩn NCHW | `src/datasets/vision_datasets.py` | PASSED | `tests/test_vision_datasets.py` | 6 / 0 | DataLoader cho ra ảnh `float32` dạng `(N,C,H,W)` và nhãn `torch.long`; hỗ trợ nhãn số/one-hot. |
| WIKI `np.rot90(k=3, axes=(1,2))` | `src/datasets/vision_datasets.py` | PASSED | Kiểm tra sentinel + smoke | 2 / 0 | Xoay trên NHWC trước khi hoán vị NCHW; hướng xoay khớp notebook legacy. |
| FER 7 lớp và CSV legacy | `src/datasets/vision_datasets.py`, `configs/fer_emotion.yaml` | PASSED | CSV fixture one-hot + smoke | 2 / 0 | Xác thực 2311 cột: 2304 pixel và 7 nhãn; thứ tự lớp theo notebook. Không chia pixel nếu nguồn đã `[0,1]`; tự chia 255 nếu nguồn byte `[0,255]`. |
| Dữ liệu giả và ảnh WIKI trực quan | `scripts/smoke_test_datasets.py` | PASSED | Smoke DataLoader, lưu PNG | 1 / 0 | Tám mẫu/bộ, seed 42; ảnh lưu cục bộ ở `outputs/figures/wiki_gender_rot90_smoke.png` và bị ignore. Đây là dữ liệu giả, không phải bằng chứng về hướng của dữ liệu WIKI thật. |
| Bảo vệ Git / legacy | `.gitignore`, `legacy_keras/` | PASSED | `git check-ignore`, `git diff -- legacy_keras` | 2 / 0 | Dữ liệu raw/processed và outputs bị ignore; không sửa file legacy. |

---

## 2. 🛡️ GUARDRAILS & INVARIANT VALIDATION

- [x] **NCHW**: Chuyển `(N,H,W,C)` sang `(N,C,H,W)` bằng `permute(0,3,1,2)` sau xử lý WIKI.
- [x] **WIKI legacy rotation**: `np.rot90(..., k=3, axes=(1,2))` trước khi chuyển trục.
- [x] **Label contract**: Nhãn lớp được trả về `torch.long`; one-hot được đổi bằng `argmax`.
- [x] **Config decoupling**: Cấu hình shape/lớp/đường dẫn nằm trong YAML; không có đường dẫn máy cá nhân.
- [x] **Data integrity**: Loader kiểm tra shape, pixel range, số lớp, nhãn và số cột FER.
- [x] **Git protection**: Đã kiểm tra Git ignore cho MNIST/WIKI/FER `.npy`/CSV, `data/processed/` và PNG đầu ra. Không stage dữ liệu.
- [x] **Legacy read-only**: `git diff -- BTL03/legacy_keras` rỗng.
- [ ] **Dữ liệu nguồn thật**: Chưa tải/xác minh vì MNIST network download không được chạy; WIKI/FER và Google Drive IDs không có trong workspace.
- [ ] **Thông tin phân công**: Người dùng xác nhận Member_02 phụ trách T0102; tuy nhiên `AGENTS.md` ghi T0101–T0105 cho Thành viên 1, trong khi CSV phân công ghi Member_02 sở hữu pipeline dữ liệu. Mâu thuẫn WBS cần nhóm làm rõ; báo cáo này ghi theo xác nhận trực tiếp của người dùng.
- [ ] **Project Charter**: `docs/project_charter/Project_Charter_BTL03.md` không tồn tại. Đã đọc `AGENTS.md`, `README_BTL03.md`, `QUY_DINH_SAN_PHAM.txt`, `Yêu cầu.txt`, CSV phân công, `legacy_keras/README.md` và execution-report template hiện có.

---

## 3. 🧪 DETAILED ACCEPTANCE FINDINGS

### A. Functional Implementation

- `MNISTDataset`: nhận ảnh `uint8` 28×28; thực hiện phép chia `/255` tương đương notebook MLP Keras; xuất `(N,1,28,28)`, 10 lớp.
- `WikiGenderDataset`: nhận NHW/NHWC NumPy hoặc `.npy`, nhãn số/one-hot; mặc định thực hiện phép xoay legacy rồi mới đổi trục. Có thể tắt xoay khi upstream đã áp dụng.
- `FEREmotionDataset`: nhận mảng/NPY hoặc CSV legacy; CSV có 2304 pixel đầu và 7 nhãn one-hot cuối. Thứ tự lớp: angry, disgust, fear, happy, neutral, sad, surprise.
- `DataBootstrapper`: tải MNIST IDX bằng `urllib`, xác thực gzip/IDX và lưu mảng `.npy`; kiểm kê dữ liệu WIKI/FER; hỗ trợ URL cấu hình hoặc Drive ID tùy chọn; tạo dummy data nhỏ deterministic.
- **Hình học WIKI**: `k=3` tương đương xoay 270° ngược chiều kim đồng hồ (hay 90° theo chiều kim đồng hồ). Với tọa độ ảnh `(y,x)`, tọa độ sau xoay là `(x, W-1-y)`. Trên sentinel ở `(2,5)` với ảnh 64×64, pixel chuyển tới `(5,61)`. Sau đó NHWC → NCHW chỉ đổi thứ tự trục, không đổi vị trí pixel.
- **Chuẩn hóa FER**: notebook gốc đọc 2304 pixel từ CSV và reshape; không có phép chia 255 rõ ràng trong các cell nguồn đã kiểm tra. Loader `auto` vì vậy giữ dữ liệu đã chuẩn hóa `[0,1]`, và chỉ chia 255 nếu nhận pixel byte `[0,255]`.
- Tệp `configs/` mới: `mnist.yaml`, `wiki_gender.yaml`, `fer_emotion.yaml`.
- **WBS**: CSV hiện tại ghi Member_02 phụ trách pipeline dữ liệu/NHWC→NCHW/WIKI; `AGENTS.md` có bảng khác giao T0102 cho Thành viên 1. Không chỉnh sửa tài liệu quản trị trong phạm vi task.

### B. Testing & Quality Gates

- `python -m pytest tests/test_vision_datasets.py -q` — **6 passed, 0 failed**.
- `python -m scripts.smoke_test_datasets` — **PASS**:
  - MNIST: `(4,1,28,28)`, `torch.float32`, min/max `0.0000/1.0000`; nhãn `torch.int64`.
  - WIKI: `(4,1,64,64)`, `torch.float32`, min/max `0.0000/1.0000`; nhãn `torch.int64`.
  - FER2013: `(4,1,48,48)`, `torch.float32`, min/max `0.0000/1.0000`; nhãn `torch.int64`.
- `python -m py_compile` cho module, downloader, smoke-test và test — PASS.
- Parse cả ba YAML bằng PyYAML — PASS.
- `scripts/download_data.py check` xác nhận các file dummy tồn tại; MNIST thật báo thiếu như dự kiến.
- Ảnh mẫu được tạo: `outputs/figures/wiki_gender_rot90_smoke.png` (local, gitignored). PNG này minh họa ảnh giả đã xoay; không xác nhận orientation của dataset WIKI thật.
- **Cài đặt môi trường kiểm thử**: Python 3.11.9; torch 2.13.0+cpu; numpy 2.4.6; Pillow 12.3.0; PyYAML 6.0.3; pytest 9.1.1. `torchvision` không có trong interpreter hiện tại; loader MNIST IDX không phụ thuộc torchvision.

---

## 4. ⏱️ PERFORMANCE & EXECUTION METRICS

- **Dữ liệu thật tải/xử lý**: 0 bytes; không tải MNIST, WIKI hoặc FER thật trong phiên.
- **Dummy raw data**: 8 mẫu mỗi bộ; bị Git ignore. Dữ liệu nhỏ chỉ để kiểm thử schema.
- **Tổng thời gian pytest**: khoảng 3.1 giây cho 6 test ở môi trường hiện tại.
- **Download latency**: chưa đo; không thực hiện network download.

---

## 5. 🚩 OUTSTANDING ISSUES & RECOMMENDATIONS

- **Current limitations**:
  - Google Drive file IDs/URL chưa có; cần nhóm điền nguồn chính thức trước khi tải dữ liệu WIKI/FER.
  - Script MNIST có endpoint tải và parser IDX nhưng chưa kiểm chứng tải mạng thực tế hoặc SHA-256 upstream.
  - Hợp đồng WIKI mặc định coi file đầu vào là NHWC theo legacy notebook; phải xác nhận file `.npy` thật có cùng channel/layout trước khi dùng.
  - FER notebook cũ dùng `np.loadtxt` nên CSV phải numeric, không có header; loader hiện xác thực đúng 2311 cột.
  - Dữ liệu giả và PNG smoke không dùng làm kết quả mô hình/báo cáo học thuật.
- **Recommendations for next task**: xác nhận WBS/alias thành viên và bổ sung charter còn thiếu; cung cấp Drive IDs/quyền truy cập; tải dữ liệu thật, kiểm tra checksum/shape/range và ảnh WIKI trực quan trước khi triển khai augmentation/training pipeline.