# [DEV-LOG] <Tiêu đề ngắn gọn tóm tắt phiên làm việc>

- **Mã phiên (Session ID):** `LOG_[YYMMDD]_[HHMM]`
- **Thời gian (Timestamp):** YYYY-MM-DD HH:MM
- **Người thực hiện:** <Họ và tên thành viên phụ trách WP — chỉ ghi tên thành viên nhóm>
- **Work Package liên quan:** [WP01 | WP02 | WP03 | WP04 | WP05 | WP06]
- **Task ID đang xử lý:** `BPM25_<Task_ID>` (Ví dụ: `BPM25_T0104`, `BPM25_T0401`)
- **Nhánh Git (Branch):** `[chức năng]/BPM25_<Task_ID>_<tên_công_việc>`
- **Trạng thái phiên làm việc:** [Completed | In Progress | Blocked]

---

## 1. Mục tiêu phiên làm việc (Session Objectives)
- [ ] **Mục tiêu 1:** Mô tả kết quả cụ thể cần đạt trong phiên (ví dụ: Xây dựng class SlidingWindowGenerator trong `src/data_pipeline/dataset.py`).
- [ ] **Mục tiêu 2:** Ràng buộc hoặc kiểm thử cần vượt qua (ví dụ: Đảm bảo kiểm thử shape tensor [B, 96, D] và [B, H, 1] không bị rò rỉ dữ liệu).

---

## 2. Nhật ký công việc & Thay đổi tệp (Work Done & File Inventory)
*Tóm tắt ngắn gọn các bước can thiệp mã nguồn, refactor hoặc cấu hình đã thực hiện:*

- **Chi tiết tệp thay đổi:**
  - `[CREATED]` `src/.../ten_file.py`: Tạo mới class/module phục vụ nhiệm vụ...
  - `[MODIFIED]` `configs/default.yaml`: Cập nhật tham số batch size hoặc siêu tham số...
  - `[DELETED]` `.ignore/...`: Dọn dẹp cache hoặc tệp tạm không cần thiết...

---

## 3. Quyết định kỹ thuật & Ghi chú kiến trúc (Technical Decisions & Design Notes)
- **Cơ sở thuật toán / Lựa chọn giải pháp:**
  - *Ví dụ:* Áp dụng `StandardScaler` chỉ fit trên tập Train (60%) và transform cho Val (10%), Test (20%) để triệt tiêu Temporal Data Leakage.
  - *Ví dụ:* Đóng gói `SlidingWindowGenerator` kế thừa theo đúng chuẩn OOP và viết Docstring Tiếng Việt 100%.
- **Sự cố phát sinh & Hướng xử lý:** Ghi nhận các vấn đề kỹ thuật đã xử lý trong phiên (ví dụ: lỗi broadcast tensor shape, xử lý missing value bằng Linear Interpolation + Forward Fill).

---

## 4. Bằng chứng thực thi & Trạng thái kiểm tra (Execution Evidence & Testing)
*Tuân thủ nguyên tắc "Evidence-First" — đính kèm câu lệnh terminal và log xác nhận mã nguồn chạy thông suốt:*

- **Lệnh thực thi (Execution Command):**
```bash
python training_loop.py --config configs/default.yaml --models patchtst --horizon 24 --seeds 42
# Hoặc lệnh kiểm thử pytest:
# pytest tests/test_dataset.py -v
```

- **Kết quả đầu ra / Log terminal xác nhận (Terminal Logs):**
```text
[INFO] Bắt đầu khởi tạo SlidingWindowGenerator với L=96, H=24...
[INFO] Đã tạo thành công 25,410 mẫu chuỗi thời gian không rò rỉ dữ liệu.
[INFO] Tensor X shape: torch.Size([32, 96, 18]), Tensor Y shape: torch.Size([32, 24, 1])
[SUCCESS] Pipeline kiểm thử hoàn tất thành công 100%.
```

---

## 5. Rào cản & Khó khăn tồn đọng (Blockers & Challenges)
- [ ] **Rào cản 1:** Không có / Hoặc ghi rõ vấn đề đang gặp (ví dụ: Thiếu checkpoint từ WP02/WP03 để chạy tích hợp XAI).
- [ ] **Rào cản 2:** Lỗi phần cứng, thiếu tài nguyên VRAM hoặc dữ liệu trạm quan trắc khuyết thiếu kéo dài.

---

## 6. Kế hoạch & Bàn giao phiên tiếp theo (Next Steps & Handover)
- [ ] Công việc cụ thể cần làm tiếp theo (gắn kèm Task ID tương ứng).
- [ ] Thông tin cần bàn giao cho thành viên phối hợp hoặc người phụ trách phiên tiếp theo.