# [REPORT] BPM25_AUDIT_POST_MERGE: Kiểm toán hậu hợp nhất WP03 vào WP04

- **Mã công việc (Task ID):** `BPM25_AUDIT_POST_MERGE`
- **Tên file lưu vết:** `docs/reports/AUDIT_POST_MERGE_261116_2132.md`
- **Thời gian nghiệm thu:** 2026-11-16 21:32
- **Người thực hiện:** Cline (AI Coding Agent)
- **Work Package liên quan:** WP03, WP04
- **Nhánh Git (Branch):** `feat/BPM25_T0401_shap-explainer`
- **Commit SHA được kiểm toán:** `bad29596b0a7cd408558b41a7fb74574e3d0d50c`
- **Trạng thái nghiệm thu:** Ready for Review

---

## 1. Mục tiêu & Phạm vi công việc

- **Mục tiêu kỹ thuật:** Đồng bộ cập nhật WP03 và các bản vá hợp đồng tensor từ nhánh `fix/BPM25_T0304_post-merge-cleanup` sang nhánh WP04, sau đó xác minh cấu hình, các mô hình chuỗi thời gian và tích hợp SHAP.
- **Phạm vi thực hiện:** Merge nhánh fix, kiểm thử LSTM, DLinear, Informer ProbSparse/FullAttention, PatchTST và `SHAPExplainer` trên checkpoint PatchTST cục bộ.
- **Cách hợp nhất:** Nhánh fix là hậu duệ trực tiếp của `feat/BPM25_T0401_shap-explainer`; thao tác merge hoàn thành bằng fast-forward, không cần tạo merge commit riêng.

### Tóm tắt lịch sử hợp nhất

- `9df832b7` — merge cập nhật WP03 SOTA/Ablation vào nhánh fix.
- `0b9a5734` — chuẩn hóa đầu ra PatchTST và `RevINModelWrapper` về `[B, H, 1]`.
- `bad29596` — lưu bốn tài liệu audit/dev-log đã chuẩn hóa UTF-8.
- Các commit trên hiện có trong lịch sử `feat/BPM25_T0401_shap-explainer`; hai commit mã nguồn `9df832b7` và `0b9a5734` được xác minh là tổ tiên của HEAD.

**Các cập nhật WP03 chính:** `src/models/patchtst.py`, `training_loop.py`, `fine_tune.py`, `run_ablation.py`, `scripts/train.sh`, `scripts/ablation.sh`, `configs/config.yaml`, `.gitignore`, các báo cáo Ablation/T0305. Báo cáo WP03 được chuẩn hóa tên giờ thành `Ablation_261116_1112.md` và `T0305_261116_0111.md`. File CSV, checkpoint và bytecode bị gỡ khỏi Git tracking theo quy tắc ignore; checkpoint dùng cho smoke test bên dưới là artifact local, không được commit.

---

## 2. Đối chiếu tiêu chí hoàn thành (DoD Verification)

| STT | Tiêu chí | Trạng thái | Bằng chứng |
| :---: | :--- | :---: | :--- |
| 1 | Cấu trúc mô hình OOP và Docstring Tiếng Việt theo chuẩn dự án | Đạt trong phạm vi các lớp/mô-đun được tích hợp; chưa khẳng định audit thủ công toàn bộ mọi docstring repository | Bốn kiến trúc được import và chạy forward sau merge; mã nguồn thuộc `src/models/`. |
| 2 | Anti-Data Leakage | Đạt theo thiết kế pipeline được kiểm tra; không chạy lại fit scaler trong smoke test | Pipeline hiện hành chia tuần tự; scaler được fit trên Train rồi transform các split còn lại. Smoke test chỉ dùng tensor tổng hợp cho attribution, không dùng Validation/Test làm background thực nghiệm. |
| 3 | Tensor contract `[B,96,D] -> [B,H,1]` | Đạt | LSTM, DLinear, Informer ProbSparse, Informer FullAttention và PatchTST đều trả `[4,24,1]` từ `[4,96,15]`; tất cả đầu ra hữu hạn. |
| 4 | SHAP/Pydantic contract và độ hữu hạn | Đạt | Nạp checkpoint PatchTST seed 42 bằng `strict=True`; SHAP shape `(2,96,15)`, global importance có 15 key khớp tập feature names, không NaN/Inf. |
| 5 | Bảo vệ `.gitignore` và không track artifact cấm | Đạt | `outputs/checkpoints/` được ignore; checkpoint local được xác nhận bởi `git check-ignore`; quét Git tree không thấy checkpoint/CSV/bytecode/raw data bị track trong `course_project/`. |
| 6 | Tên file báo cáo tương thích Windows và không còn conflict marker mã nguồn | Đạt | Quét các file source/config/script được track không phát hiện marker; tên báo cáo WP03 đã chuẩn hóa, không còn dấu `:`. |

---

## 3. Bằng chứng thực thi & Kiểm thử

- **Lệnh:** chạy Python inline từ thư mục `course_project/`; không tạo file test tạm. Checkpoint `outputs/checkpoints/global/patchtst_multivariate_H24_seed42_pretrained.pt` là file local bị `.gitignore`, được nạp bằng `weights_only=True` và `load_state_dict(strict=True)`.
- **Terminal Pass Logs (nguyên văn):**

```text
[PASS] YAML safe_load: configs/config.yaml; đủ nhóm project/data/models/training/xai; D=15
[PASS] Imports: LSTMModel, DLinearModel, InformerModel, PatchTSTModel, SHAPExplainer
[PASS] LSTMModel: [4,96,15] -> (4, 24, 1); NaN/Inf=0
[PASS] DLinearModel: [4,96,15] -> (4, 24, 1); NaN/Inf=0
[PASS] InformerProbSparse: [4,96,15] -> (4, 24, 1); NaN/Inf=0
[PASS] InformerFullAttention: [4,96,15] -> (4, 24, 1); NaN/Inf=0
[PASS] PatchTSTModel: [4,96,15] -> (4, 24, 1); NaN/Inf=0
[PASS] PatchTST checkpoint: local checkpoint seed=42, strict=True (gitignored)
[PASS] SHAPExplainer: values=(2,96,15); importance=15 matching feature names; NaN/Inf=0
[SUCCESS] Post-merge integration smoke test passed
PYTHON_EXIT_CODE=0
```

- **Kiểm tra bổ sung:** `git diff --check` đạt; kiểm tra conflict markers trong các file Python/YAML/scripts được track đạt.

---

## 4. Quyết định kỹ thuật & Ghi chú kiến trúc

- `InformerFullAttention` được kiểm tra thông qua `build_model('informer_full', ...)`, đồng thời giữ nhánh ProbSparse mặc định.
- SHAP được tính với mẫu tổng hợp tái lập để xác minh giao diện; checkpoint dùng là checkpoint PatchTST thật có sẵn tại máy, không được đưa vào Git. Đây chưa phải phép tính Global SHAP trên toàn bộ dữ liệu Train/12 trạm.
- **Sai khác số chiều cần xử lý trước thực nghiệm đầy đủ:** `configs/config.yaml` và checkpoint đã kiểm tra dùng `D=15`. `preprocess.py`/`get_feature_columns()` trên dữ liệu PRSA có thể sinh `D=16` do thêm `ir`. Smoke test xác nhận hợp đồng checkpoint D=15, không chứng minh pipeline mới tự động tạo đúng D=15. Cần thống nhất/khóa feature list trước khi chạy Global SHAP trên dữ liệu thật.
- Các stash cũ được giữ nguyên; không có `stash pop` hoặc `stash drop`.

---

## 5. Bàn giao & Mức độ sẵn sàng

- **T0401:** Giao diện explainer đã tương thích và checkpoint local tải được, đủ điều kiện bắt đầu chuẩn bị thực nghiệm. Chưa tuyên bố hoàn thành Global SHAP trên toàn bộ Train vì feature contract D=15/D=16 cần được giải quyết trước.
- **T0402:** Có thể bắt đầu phát triển `src/xai/ig_explainer.py` độc lập; cần dùng cùng feature contract đã chốt và giữ nguyên quy tắc chỉ lấy baseline/background từ Train.
- **Trạng thái Git tại thời điểm kiểm toán:** nhánh WP04, HEAD `bad29596`; các báo cáo/dev-log của phiên sẽ được commit tiếp theo trong commit tài liệu riêng.