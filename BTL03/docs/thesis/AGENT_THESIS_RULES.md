# Quy tắc biên soạn báo cáo BTL03

Áp dụng cho mọi thay đổi trong `docs/thesis/`. Hệ thống chia ba tầng: `01_materials/` (nguyên liệu), `02_authoring/` (biên soạn/snapshot), `03_releases/<version>/` (thành phẩm đã duyệt). Template trong `01_materials/templates/` là bản gốc **READ-ONLY theo quy trình**: chỉ đọc/sao chép; nội dung hiện hành nằm ở `02_authoring/current/`, snapshot lịch sử ở `02_authoring/snapshots/<SNAPSHOT>/`, còn PDF cuối chỉ ở release version sau khi đã duyệt.

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
- Nguồn LaTeX trong `02_authoring/current/latex/` dùng đường dẫn tương đối; asset phải nằm trong bundle hoặc ghi rõ chưa có. Không dùng đường dẫn máy/ổ đĩa.
- Compiler chuẩn là XeLaTeX Unicode, có font tiếng Việt; chú thích tiếng Việt rõ. Tránh shell-escape và dependency không cần thiết.
- Bundle `02_authoring/current/latex/` phải tự chứa: `main.tex` ở bundle root, chương tách file và bibliography. **Không kiểm thử/biên dịch trên máy local**; người dùng đưa nội dung bundle lên Overleaf để biên dịch.

## 5. Giới hạn thao tác môi trường (bắt buộc)

- Không yêu cầu người dùng cài bất kỳ thư viện/package/TeX/LaTeX nào. Không chạy `pip install`, package manager hoặc installer.
- Không chạy unit test, integration test, smoke test, benchmark, training hoặc inference model trên máy local trong quy trình viết báo cáo.
- Chỉ đọc tài liệu, mã nguồn, config, logs và artifacts đã có trong repository để tổng hợp; không tạo thêm evidence bằng cách chạy code.
- Không chạy XeLaTeX, latexmk, Biber/BibTeX hoặc compiler local. Chỉ chuẩn bị `.md` và bộ `.tex`; người dùng tự upload/biên dịch trên Overleaf.
- Có thể thực hiện kiểm tra văn bản tĩnh không thực thi (Git diff/status, đối chiếu include/citation/source paths, UTF-8, cấu trúc thư mục). Không gọi đó là test/build pass.
- Nếu Overleaf báo lỗi, đọc log người dùng cung cấp rồi sửa source; không yêu cầu cài đặt local để chẩn đoán.

## 6. Hướng dẫn phong cách rút ra từ hai mẫu được cung cấp

Đọc `01_materials/samples/style_analysis.md` khi tạo/cập nhật chương. Chỉ học **quy ước cấu trúc/trình bày**: bìa/front matter, tóm tắt/từ khóa, mở đầu có mục tiêu/phạm vi/phương pháp/đóng góp, chương nền tảng → phương pháp/hệ thống → kiểm toán/kết quả → thảo luận → kết luận/kiến nghị, phụ lục, mục lục/danh mục, nhãn tham chiếu chéo và phân tích pipeline/đối tượng sử dụng khi phù hợp.

- Đây là lựa chọn biên tập tham khảo, không phải quy định hình thức chính thức của trường. Không suy ra lề/font/cỡ chữ bắt buộc từ mẫu.
- Khi dùng bìa/thành viên, lấy họ tên/MSSV/giảng viên từ danh sách nhóm được duyệt. Alias Git chứng minh alias/task provenance, không thay thế danh tính pháp lý. Vai trò/đóng góp phải theo task report/WBS đã xác nhận; không suy ra phần trăm từ số commit hoặc ví dụ trong mẫu.
- Hai file mẫu gốc được lưu nguyên trạng trong `01_materials/samples/`; không nhập nội dung của chúng vào báo cáo BTL03. Không nhập trực tiếp preamble/package chuyên biệt của sample.
- Giữ compiler/package stack riêng của BTL03; lấy cấu trúc front matter và cách giải thích kết quả, không buộc gộp hai mẫu thành một khuôn.
- Hình/sơ đồ/bảng chỉ đưa vào chương khi có asset và provenance; ghi ID/nguồn, caption và tham chiếu. Nếu thiếu, để placeholder có nhãn, không dựng số hoặc hình giả.
- Phân biệt tóm tắt, kết luận, thảo luận, khuyến nghị và phụ lục; mỗi phần có mục tiêu riêng và giữ nguyên caveat T0212.

## 7. Checklist trước khi hoàn tất

1. Kiểm tra snapshot, working tree và evidence mới.
2. Giữ `01_materials/templates/` không sửa; biên soạn trong `02_authoring/current/` và giữ snapshot cũ trong `02_authoring/snapshots/`.
3. So khớp Markdown–LaTeX, citation, liên kết, bảng và trạng thái; chạy kiểm tra văn bản tĩnh/`git diff --check`. Không build/test local.
4. Stage tệp đã xem xét; không stage `.env`, dữ liệu, weights, event logs, output runtime hay build tạm.
5. Ghi report theo template repository, mang tên thành viên thật. Không sửa code mô hình hoặc legacy trong task tài liệu.
