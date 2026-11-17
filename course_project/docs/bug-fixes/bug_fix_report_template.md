# [BUG-FIX] <Tên ngắn gọn của sự cố / File bị ảnh hưởng>

> **Quy ước đặt tên file:** `docs/bug-fixes/[YYMMDD]_[HHMM]_[ten_file_hoac_van_de].md`  
> **Ví dụ:** `docs/bug-fixes/261107_1430_preprocess_leakage_fix.md`

---

## 1. Thông tin tổng quan (Audit Metadata)

- **Mã sự cố (Bug ID):** `BUG_[YYMMDD]_[HHMM]`
- **Thời gian xử lý:** `YYYY-MM-DD HH:MM`
- **Người thực hiện:** <Họ và tên thành viên phụ trách WP — chỉ ghi tên thành viên nhóm>
- **Gói công việc liên quan:** [WP01 | WP02 | WP03 | WP04 | WP05 | WP06]
- **Task ID:** `BPM25_T0xxx`
- **Nhánh Git (Git Branch):** `fix/BPM25_[Task_ID]_[ten_cong_viec]`
- **Mức độ nghiêm trọng (Severity):** [Critical (Ảnh hưởng kết quả/Data Leakage) | High | Medium | Low]
- **Tệp mã nguồn liên quan (Affected Files):**
  - `src/...`
  - `configs/...`

---

## 2. Triệu chứng & Bối cảnh phát hiện (Bug Description & Symptoms)

- **Hành vi kỳ vọng (Expected Behavior):**
  - *Mô tả trạng thái hoặc đầu ra mong muốn khi hệ thống vận hành đúng quy chuẩn.*
- **Hành vi thực tế (Observed Behavior):**
  - *Mô tả chi tiết lỗi phát sinh, sai lệch tensor shape, rò rỉ dữ liệu hoặc tiến trình bị crash.*
- **Traceback / Nhật ký lỗi Terminal (Raw Error Log):**

```text
# Dán toàn bộ traceback hoặc thông báo lỗi exception xuất hiện trên terminal tại đây:
Traceback (most recent call last):
  File "training_loop.py", line 45, in <module>
    ...
Exception: <Nội dung chi tiết lỗi>
```

---

## 3. Phân tích nguyên nhân cốt lõi (Root Cause Analysis - RCA)

- **Cơ chế gây lỗi (Underlying Mechanism):**
  - *Giải thích bản chất kỹ thuật dẫn tới lỗi (ví dụ: gọi `.fit()` trên toàn bộ dataset trước khi split khiến rò rỉ phân phối tập Test; hoặc sai lệch chiều batch khi trượt cửa sổ $L=96, H=24$; hoặc tràn bộ nhớ CUDA Attention).*
- **Phạm vi ảnh hưởng (Impact Scope):**
  - *Liệt kê các mô hình (LSTM, DLinear, Informer, PatchTST), checkpoints hoặc báo cáo thực nghiệm bị ảnh hưởng bởi lỗi này (nếu có).*

---

## 4. Giải pháp & Mã nguồn thay đổi (Solution & Code Diff)

- **Nguyên lý khắc phục:**
  - *Trình bày phương án giải quyết dứt điểm nguyên nhân cốt lõi (đảm bảo tuân thủ OOP, có Docstring Tiếng Việt và không phá vỡ hợp đồng dữ liệu Pydantic).*
- **Chi tiết thay đổi mã nguồn (Git Diff):**

```diff
# Đường dẫn: src/...
--- a/src/data_pipeline/preprocess.py
+++ b/src/data_pipeline/preprocess.py
@@ -35,5 +35,7 @@
-        # Code cũ gây lỗi / rò rỉ dữ liệu
-        self.scaler.fit(full_dataframe)
+        # Code mới: Đảm bảo chỉ fit trên tập Train
+        self.scaler.fit(train_dataframe)
+        val_scaled = self.scaler.transform(val_dataframe)
```

---

## 5. Bằng chứng kiểm thử & Xác minh (Verification Evidence)

> **Yêu cầu bắt buộc ("Evidence-First"):** Phải có bằng chứng lệnh thực thi thành công và không xảy ra regression trước khi đóng task.

- **Lệnh thực thi kiểm thử:**

```bash
# Ví dụ lệnh kiểm thử pipeline hoặc pytest
python -m pytest tests/test_data_pipeline.py -v
# Hoặc chạy kiểm tra trực tiếp script
python training_loop.py --config configs/default.yaml --models dlinear --horizon 24 --seeds 42
```

- **Nhật ký xác minh thành công (Verification Output Log):**

```text
# Dán log terminal chứng minh mã nguồn đã chạy thành công, test passed:
[INFO] 2026-11-08 14:35:10 - Kiểm tra tính toàn vẹn: Không phát hiện rò rỉ dữ liệu (No Data Leakage).
[INFO] 2026-11-08 14:35:12 - Epoch 1/50 - Train Loss: 0.0421 - Val Loss: 0.0465 - PASSED
==================== 1 passed, 0 failed in 4.25s ====================
```

---

## 6. Biện pháp phòng ngừa & Bài học kinh nghiệm (Prevention & Lessons Learned)

- **Ràng buộc bổ sung (Assertions / Contract Updates):**
  - *Thêm kiểm tra điều kiện (assertions), type contract hoặc unit test nào để ngăn lỗi tương tự lặp lại?*
- **Hành động tiếp theo (Follow-up Tasks):**
  - [ ] Huấn luyện lại mô hình hoặc cập nhật lại checkpoint trọng số bị sai.
  - [ ] Cập nhật nhật ký ngày tại `docs/dev-logs/[YYMMDD]_[HHMM].md`.
  - [ ] Tạo Pull Request theo quy chuẩn: `[BPM25_T0xxx] fix(<module>): <mô tả>`.