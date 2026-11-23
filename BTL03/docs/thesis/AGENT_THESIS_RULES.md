# Quy tắc biên soạn báo cáo BTL03

Áp dụng cho mọi thay đổi trong `docs/thesis/`. Template trong `templates/` là bản gốc **READ-ONLY theo quy trình**: chỉ đọc/sao chép; viết nội dung vào `md_drafts/` và `tex_source/`.

## 1. Trung thực và bằng chứng

- Chỉ công bố tham số, metric, thời gian, môi trường, test, trạng thái task và kết luận có provenance trong repository. Ưu tiên execution report, notebook output/cell, CSV có run ID, cấu hình và test source. Ghi đường dẫn, commit, task, cell hoặc Run_ID thích hợp.
- Không biến threshold/YAML thành kết quả đạt; không gọi synthetic smoke là training khoa học; không dùng ảnh minh họa giả làm dự đoán đã huấn luyện. Parameter-count parity không chứng minh parity đầu ra hoặc metric.
- Giữ nguyên tên metric và split. Validation/public-test được tái sử dụng không phải holdout độc lập; FCN legacy `accuracy` không phải IoU/mAP; SegNet pixel accuracy không phải mIoU.
- Giá trị không có nguồn ghi `NOT_RECORDED`/`CHƯA CÓ BẰNG CHỨNG`; công thức phải kèm điều kiện. Không nội suy, đặt 0 hay chọn số README mâu thuẫn mà không kiểm tra evidence chi tiết.
- Nếu hai report/CSV khác nhau về cùng metric hoặc parameter count, không xóa bất đồng bằng cách ưu tiên số xuất hiện nhiều hơn. Ghi từng nguồn/snapshot, đánh dấu `CẦN ĐỐI CHIẾU`, rồi kiểm tra code/model state/quy ước đo; chỉ chốt sau khi có provenance.
- Phân biệt `PASSED`, `PASSED WITH LIMITATIONS`, `BLOCKED`, `NOT RUN` và chưa đủ bằng chứng. Với việc thực sự chưa làm, ghi nguyên trạng thái **`[CHƯA THỰC HIỆN - DỰ KIẾN Ở GIAI ĐOẠN TIẾP THEO]`**, phụ thuộc và kế hoạch xác minh. Không dùng nhãn chưa làm thay cho task bị blocker.
- Nêu giới hạn dữ liệu synthetic/thiếu, CPU-only, thiếu GPU, dependency failure, khác split/seed, test fail và thiếu metadata. T0212 không được gọi là nghiệm thu khi report nguồn ghi `BLOCKED`.

## 2. Danh tính con người và đóng góp

- Tác giả/người thực hiện/thành viên phải là người thật đã xác nhận. Dùng alias đã có bằng chứng; họ tên, MSSV, người chủ trì và tỷ lệ chưa xác nhận phải giữ placeholder rõ ràng.
- Cấm dùng tên AI agent làm tác giả, executor, thành viên hoặc Git identity; không thêm `Co-authored-by` cho AI. Không thay trường disclosure nếu quy định môn học yêu cầu khai báo trung thực công cụ hỗ trợ; disclosure không phải attribution tác giả.
- Không suy luận tỷ lệ cuối kỳ từ tỷ lệ mẫu. Nhóm xác nhận tên pháp lý/MSSV và tỷ lệ, tổng phải là 100% trước bản nộp.

## 3. Trích dẫn

- Chỉ đưa BibTeX metadata đã kiểm tra được; không chế DOI, tác giả, năm, trang hoặc URL. Citation trong văn bản phải có key thật trong `.bib`.
- Với nguồn nội bộ, nêu đường dẫn, task, commit/snapshot, bảng/cell hoặc Run_ID. Mỗi nguồn chỉ chứng minh nội dung cụ thể đã lưu.
- Phân biệt đặc tả, mã Keras, mã PyTorch, log, report và suy luận. Gọi suy luận là phân tích, không phải số đo.
- Không gán nội dung cho PDF/XLSX/charter chưa đọc được; nêu giới hạn truy xuất khi liên quan.

## 4. Đồng bộ `.md` ↔ `.tex` và Overleaf

- Mọi thay đổi snapshot cập nhật đồng thời Markdown và LaTeX: số, bảng, trạng thái, tên metric, caveat, tác giả, evidence IDs và citations phải khớp.
- Nguồn LaTeX dùng đường dẫn tương đối; asset phải nằm trong ZIP hoặc được ghi rõ chưa có. Không dùng đường dẫn máy/ổ đĩa.
- Compiler chuẩn là XeLaTeX Unicode, có font tiếng Việt; chú thích tiếng Việt rõ. Tránh shell-escape và dependency không cần thiết.
- `main.tex` ở gốc `tex_source/`, chương tách file, bibliography build được. Không theo dõi build phụ trợ.

## 5. Checklist trước khi hoàn tất

1. Kiểm tra snapshot, working tree và evidence mới.
2. Giữ templates không sửa; nội dung đi vào drafts/source.
3. So khớp Markdown–LaTeX, citation, liên kết, bảng và trạng thái; build XeLaTeX; chạy `git diff --check`.
4. Stage tệp đã xem xét; không stage `.env`, dữ liệu, weights, event logs, output runtime hay build tạm.
5. Ghi report theo template repository, mang tên thành viên thật. Không sửa code mô hình hoặc legacy trong task tài liệu.
