# BÁO CÁO [TÊN DỰ ÁN] — SNAPSHOT [TASK / MỐC]

> Bản thảo [DRAFT/FINAL]. Snapshot commit: `[HASH]`; ngày chốt evidence: `[YYYY-MM-DD]`. Phạm vi evidence và giới hạn được ghi rõ; không suy từ code/config sang kết quả chạy.

## Thông tin đề tài và nhóm

- Đề tài / mã dự án: `[TÊN]` / `[PROJECT CODE]`.
- Tác giả/người tổng hợp: `[TÊN THÀNH VIÊN THẬT, ĐÃ XÁC NHẬN]`.
- Thành viên/alias theo evidence: `[ALIAS CÓ NGUỒN]`.
- Họ tên pháp lý, MSSV, lớp, giảng viên, vai trò/tỷ lệ: `[NHÓM XÁC NHẬN; KHÔNG SUY ĐOÁN]`.

## Tóm tắt và từ khóa

[Vấn đề → mục tiêu → phạm vi/snapshot → phương pháp/evidence → kết quả đã đo → blocker/hạn chế. Tách legacy, run thật và smoke synthetic.]

**Từ khóa:** `[3–6 từ khóa]`

## Danh mục viết tắt (khi cần)

| Viết tắt | Ý nghĩa |
|---|---|
| [ID] | [Định nghĩa] |

## Câu hỏi nghiên cứu

| RQ ID | Câu chữ nguyên văn từ nguồn đã xác minh | Evidence cần | Trạng thái/kết luận |
|---|---|---|---|
| [RQ] | [Không tự tạo; dẫn charter] | [Run/report/test] | [Chưa kết luận nếu thiếu] |

## Mở đầu

### Bối cảnh và mục tiêu
### Đối tượng, phạm vi và snapshot
### Phương pháp kiểm toán evidence
### Đóng góp đã xác minh và giới hạn
### Cấu trúc báo cáo

## 1. Tổng quan dự án và phạm vi

### 1.1 Work packages / nhóm bài toán
### 1.2 Mốc snapshot và nguồn được kiểm tra
### 1.3 Nội dung ngoài phạm vi hoặc chưa nghiệm thu

## 2. Cơ sở lý thuyết và hợp đồng kỹ thuật

### 2.1 Khái niệm/framework
### 2.2 Data/tensor contract và preprocessing
### 2.3 Loss, metric, split và giao thức so sánh
### 2.4 Rủi ro parity và validity

## 3. Thiết kế hệ thống và phương pháp

### 3.1 Kiến trúc/pipeline OOP
### 3.2 Các nhóm model/module
### 3.3 Reproducibility, logging, checkpoint và CLI

| Nhóm/cấu hình | Component/path | Kỹ thuật | Evidence/Run ID; giới hạn |
|---|---|---|---|
| [Nhóm/cấu hình] | [Path có thật] | [Kỹ thuật] | [Evidence/Run ID; caveat] |

## 4. Thực nghiệm, kiểm toán và thảo luận

### 4.1 Snapshot, environment và protocol
### 4.2 Bảng cấu trúc/parameter audit

| Cấu hình | Reference | Port | Delta | Source/count semantics/caveat |
|---|---:|---:|---:|---|
| [Chỉ điền số có provenance] | | | | |

### 4.3 Kết quả run thật

| Task/Run ID | Dataset/split | Seed/config/runtime | Metric/unit | Kết quả và caveat |
|---|---|---|---|---|
| [Nếu chưa có: ghi NOT RUN/NOT_RECORDED] | | | | |

### 4.4 Smoke/unit tests và giới hạn suy luận
### 4.5 Task status / acceptance matrix

| Task | Trạng thái evidence | Nguồn | Việc tiếp theo |
|---|---|---|---|
| [Task] | [DONE/BLOCKED/NOT RUN/CHƯA CÓ BẰNG CHỨNG] | [Path] | [DoD tiếp theo] |

### 4.6 Hình/bảng và diễn giải

[Chỉ chèn asset thật, path tương đối, caption + evidence ID. Nếu chưa có thì ghi placeholder “chưa có evidence”; không vẽ số giả.]

## 5. Kết luận, hạn chế và hướng tiếp theo

### 5.1 Kết luận trong phạm vi snapshot
### 5.2 Hạn chế và validity threats
### 5.3 Bài học kinh nghiệm
### 5.4 Kiến nghị và tiêu chí nghiệm thu tiếp theo

## Tài liệu tham khảo

[Chỉ trích nguồn kiểm tra được; keys `.bib` phải đồng bộ với citations LaTeX.]

## Phụ lục evidence

- Manifest/path/run IDs/config/split/seed.
- Không đưa `.env`, dữ liệu cá nhân, dataset, weights, event logs hoặc artifact build tạm.
