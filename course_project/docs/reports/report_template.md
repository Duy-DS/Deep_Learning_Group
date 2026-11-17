# [REPORT] <Mã Task>: <Tên tóm tắt nhiệm vụ nghiệm thu>

- **Mã công việc (Task ID):** `BPM25_<Task_ID>` (Ví dụ: `BPM25_T0104`, `BPM25_T0401`)
- **Tên file lưu vết:** `docs/reports/[Task_ID]_[YYMMDD]_[HHMM].md`
- **Thời gian nghiệm thu:** YYYY-MM-DD HH:MM
- **Người thực hiện:** <Họ và tên thành viên phụ trách WP — chỉ ghi tên thành viên nhóm>
- **Work Package liên quan:** [WP01 | WP02 | WP03 | WP04 | WP05 | WP06]
- **Nhánh Git (Branch):** `[feat|fix|exp|docs]/BPM25_<Task_ID>_<ten_cong_viec>`
- **Commit SHA hoàn thành:** `<Git commit hash ngắn gọn, ví dụ: a1b2c3d>`
- **Trạng thái nghiệm thu:** [Completed / Ready for Review / Blocked]

---

## 1. Mục tiêu & Phạm vi công việc (Scope & Objectives)
- **Mục tiêu kỹ thuật:** Tóm tắt ngắn gọn bài toán và yêu cầu cốt lõi cần giải quyết của task.
- **Ranh giới thực thi (In-Scope / Out-of-Scope):**
  - *Đã thực hiện (In-Scope):* Các module, pipeline hoặc lớp kiến trúc được xây dựng trong task này.
  - *Không bao gồm (Out-of-Scope):* Các hạng mục phụ thuộc thuộc về task tiếp theo hoặc gói công việc khác.

---

## 2. Đối chiếu tiêu chí hoàn thành (Definition of Done - DoD Verification)
*Kiểm toán chi tiết đối chiếu với tiêu chí nghiệm thu của Task:*

| STT | Tiêu chí nghiệm thu (DoD Criteria) | Trạng thái | Bằng chứng xác nhận (Evidence / Reference) |
| :---: | :--- | :---: | :--- |
| 1 | Cấu trúc Class OOP & Docstring Tiếng Việt 100% | [x] Đạt | Class `...` tại `src/...` tuân thủ chuẩn OOP và Docstring Tiếng Việt |
| 2 | Chống rò rỉ dữ liệu (No Data Leakage) | [x] Đạt | Scaler/Imputer chỉ `.fit()` trên tập Train (60%), transform Val/Test |
| 3 | Hợp đồng Pydantic Schema / Type Hints tường minh | [x] Đạt | Dữ liệu đầu ra khớp contract tại `AGENTS.md` |
| 4 | Kiểm thử tự động / Script thực thi pass 100% | [x] Đạt | Exit code `0`, log hiển thị chi tiết tại Mục 4 |

---

## 3. Danh mục sản phẩm bàn giao (Deliverables & Artifacts)
- **Mã nguồn chính thức (Source Code):**
  - `[CREATED]` `src/.../file_name.py`: Mô tả class và các phương thức chính vừa phát triển.
  - `[MODIFIED]` `configs/default.yaml`: Mô tả các tham số cấu hình vừa bổ sung.
- **Tài liệu & Notebook kiểm chứng (nếu có):**
  - `notebooks/0X_...ipynb`: Notebook phân tích, EDA hoặc trực quan hóa kết quả.
- **Trọng số & Dữ liệu (Lưu nội bộ, gitignored):**
  - Artifact: `checkpoints/...pth` hoặc `data/processed/...` (Đã xác minh không bị stage vào Git).

---

## 4. Bằng chứng thực thi & Kiểm thử (Verification & Test Evidence)
- **Lệnh thực thi kiểm toán:**
```bash
# Lệnh chạy script hoặc pytest xác minh tính hoạt động của module
python training_loop.py --config configs/default.yaml --models patchtst --horizon 24 --seeds 42
# Hoặc: python -m pytest tests/test_module.py
```

- **Terminal Logs xác nhận (Pass Logs):**
```text
<Trích 15 - 25 dòng terminal output chứng minh lệnh chạy thành công, exit code 0>
[INFO] Khởi tạo module thành công...
[INFO] Kiểm tra Data Leakage: Scaler chỉ fit trên 60% dữ liệu Train -> PASS.
[INFO] Kiểm tra Tensor Shape đầu ra: [Batch_Size, Lookback=96, Features] -> Khớp schema.
[SUCCESS] Quá trình thực thi hoàn tất không phát sinh exception.
```

---

## 5. Quyết định kỹ thuật & Ghi chú kiến trúc (Technical Decisions)
- **Lựa chọn thuật toán / Design Pattern:** Giải thích lý do chọn phương án kỹ thuật hiện tại thay vì giải pháp thay thế.
- **Ràng buộc & Thỏa hiệp (Trade-offs):** Các đánh đổi về bộ nhớ, thời gian tính toán hoặc tài nguyên phần cứng.

---

## 6. Bàn giao hạ tầng & Bước tiếp theo (Downstream Handoff & Next Steps)
- **Khối tiếp nhận (Downstream Dependents):**
  - Mô tả module/task kế tiếp sẽ sử dụng output của task này (ví dụ: WP01 bàn giao dataset cho WP02/WP03; WP04 bàn giao explainer cho WP05 Streamlit).
- **Checklist chuyển giao:**
  - [ ] Đã merge hoặc tạo Pull Request vào nhánh `develop`.
  - [ ] Cập nhật trạng thái task trong bảng theo dõi tiến độ chung.
  - [ ] Ghi chú nhật ký phiên tương ứng tại `docs/dev-logs/[YYMMDD]_[HHMM].md`.