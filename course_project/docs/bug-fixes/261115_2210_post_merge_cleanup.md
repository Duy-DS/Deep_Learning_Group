# [BUG-FIX] Dọn dẹp sau merge: baseline, Informer và cấu hình chuẩn

## 1. Thông tin tổng quan

- **Mã sự cố:** `BUG_261115_2210`
- **Thời gian xử lý:** 2026-11-15 22:10
- **Người thực hiện:** Nguyễn Tấn Duy
- **Gói công việc liên quan:** WP02, WP03, WP04, WP06
- **Task ID:** `BPM25_FIX_POST_AUDIT`
- **Nhánh Git:** `fix/BPM25_T0304_post-merge-cleanup`
- **Mức độ nghiêm trọng:** High
- **Tệp liên quan:**
  - `src/models/lstm.py`
  - `src/models/dlinear.py`
  - `src/models/informer.py`
  - `configs/config.yaml`

## 2. Triệu chứng và bối cảnh

- Hai baseline còn ký hiệu phân định xung đột ở đầu và cuối file, ngăn Python import module.
- `AttentionLayer` trong Informer khai báo `Optional[torch.Tensor]` nhưng thiếu import `Optional`.
- `configs/config.yaml` vắng mặt trong repository, làm thiếu cấu hình chuẩn có thể nạp bằng YAML.

## 3. Phân tích nguyên nhân cốt lõi (RCA)

- Ký hiệu phân định xung đột còn lại sau merge/revert ở `lstm.py` và `dlinear.py` không được loại bỏ trước khi commit; Python coi các ký hiệu này là cú pháp không hợp lệ.
- `informer.py` sử dụng type hint `Optional` mà không nhập tên đó từ `typing`, gây `NameError` khi đánh giá annotation.
- File cấu hình chuẩn không có trong HEAD hoặc lịch sử Git được rà soát.
- **Giới hạn hợp đồng đặc trưng:** theo lựa chọn của người yêu cầu, cấu hình và smoke test giữ D=15. Tuy nhiên, `preprocess.py` sau mã hóa cùng `get_feature_columns()` hiện suy ra 16 đặc trưng cho dữ liệu PRSA (bao gồm `ir` và 5 cột `wind_*`). Vì vậy danh sách 15 tên trong YAML không khớp hoàn toàn pipeline thực tế; không sửa pipeline trong phạm vi task này.
- Các script CLI hiện tại (`preprocess.py`, `training_loop.py`, `fine_tune.py`) chưa nạp YAML hoặc thực thi `--config`; file mới là cấu hình khai báo, chưa tự động điều khiển các script.

## 4. Giải pháp và thay đổi

- Xóa riêng các dòng marker khỏi hai file baseline; giữ nguyên thân class, kiến trúc, tên tham số và phép tính của mô hình. Không thay đổi hay huấn luyện lại trọng số.
- Thêm `from typing import Optional` trong Informer.
- Tạo YAML có các nhóm `project`, `data`, `models`, `training`, `xai`, gồm seed, đường dẫn, split, cửa sổ/horizons, siêu tham số bốn mô hình và cấu hình XAI theo yêu cầu.
- Không stage hoặc sửa file `.pth`/`.csv`/file tạm.

### Diff mã nguồn rút gọn

```diff
src/models/lstm.py:
 # Các dòng phân định xung đột đã được loại bỏ; thân mô hình được giữ nguyên.

src/models/dlinear.py:
 # Các dòng phân định xung đột đã được loại bỏ; thân mô hình được giữ nguyên.

src/models/informer.py:
 import math
+from typing import Optional
```

### Nội dung cấu hình YAML

`configs/config.yaml` định nghĩa:

- `project`: `DL-PM25-BJ`, task `BPM25`, seeds `[42, 1024, 2026]`, thiết bị `cpu`.
- `data`: thư mục raw/processed, 12 trạm, split `[0.6, 0.1, 0.1, 0.2]`, lookback 96, horizons `[24, 48, 96, 192]`, D=15, 15 tên feature khai báo và target `PM2.5`.
- `models`: tham số LSTM, DLinear, Informer và PatchTST.
- `training`: batch size 64, 20 epochs, learning rate 0.001, patience 5 và thư mục checkpoints/logs.
- `xai`: SHAP background/nsamples và Integrated Gradients steps/ngưỡng spike.

## 5. Bằng chứng xác minh

- **Lệnh:** chạy smoke test Python inline, không tạo file test hoặc artifact.
- **Terminal Pass Logs (nguyên văn):**

```text
[PASS] YAML load and D=15 config assertions
[PASS] Imported 4 model classes and SHAPExplainer
[PASS] LSTMModel [4,96,15] -> [4,24,1]
[PASS] DLinearModel [4,96,15] -> [4,24,1]
[PASS] InformerModel [4,96,15] -> [4,24,1]
[PASS] PatchTSTModel [4,96,15] -> [4,24,1]
[SUCCESS] Verification smoke test passed
PYTHON_EXIT_CODE=0
```

- Đã chạy `git diff --check`; kiểm tra marker conflict trong hai baseline không trả kết quả.

## 6. Lưu ý và việc tiếp theo

- Cần quyết định/cập nhật hợp đồng feature để giải quyết sai khác D=15 so với D=16 của pipeline, trước khi dùng YAML cho dữ liệu thật.
- Cần tích hợp loader YAML vào CLI ở task riêng nếu muốn các lệnh huấn luyện/tiền xử lý tiêu thụ cấu hình này.
- Các thay đổi người dùng có trước task được bảo toàn trong stash `pre-BPM25_T0304-post-merge-cleanup-261115_2210`; chưa tự ý áp dụng lại các sửa đổi baseline cũ.
