# Hệ thống viết báo cáo tổng kết BTL03

Thư mục này quản lý song song bản thảo Markdown và nguồn LaTeX cho báo cáo chuyển đổi Keras sang PyTorch. Snapshot ban đầu mô tả bằng chứng trong repository tại commit T0212, không phải xác nhận nghiệm thu toàn bộ dự án. Report T0212 ghi trạng thái `BLOCKED` cho logger/full-suite; không được tuyên bố toàn bộ WP02 đã nghiệm thu.

## Cấu trúc

- `AGENT_THESIS_RULES.md`: quy tắc biên soạn, provenance và kiểm tra.
- `templates/`: khung Markdown/LaTeX gốc, bảo vệ theo quy trình; chỉ sao chép để viết.
- `source_materials/`: phạm vi snapshot, manifest nguồn và hướng dẫn lưu mẫu.
- `md_drafts/`: Markdown để đọc nhanh và review trên Git.
- `tex_source/`: nguồn đồng bộ, nén để upload Overleaf.
- `BaoCao_TongKet_BTL03.pdf`: vị trí PDF cuối đã biên dịch/duyệt; không có trong snapshot T0212.

Tên họ và MSSV, người chủ trì báo cáo và tỷ lệ đóng góp cuối kỳ chưa được xác nhận đầy đủ. Dùng alias thành viên có bằng chứng và giữ placeholder cho trường còn thiếu; không suy đoán.

## Đồng bộ Markdown và LaTeX

1. Đọc `AGENT_THESIS_RULES.md`, đề cương và manifest.
2. Kiểm tra nguồn trước khi ghi mỗi số liệu/trạng thái; không sửa mẫu để che thiếu sót.
3. Cập nhật đồng thời `md_drafts/BaoCao_TongKet_BTL03_T0212_draft.md` và năm chương trong `tex_source/chapters/`. Giữ nhất quán bảng, trạng thái, tác giả, caveat, evidence IDs và citation.
4. Kiểm tra Git diff; chạy `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex` từ `tex_source/` hoặc build thư mục tạm. Không commit file LaTeX phụ trợ.
5. Khi có task mới, giữ snapshot T0212 nguyên trạng; tạo snapshot/mốc mới với commit, ngày và evidence riêng. Không hồi tố kết quả WP03/WP04 vào T0212.

## Upload Overleaf và lưu PDF

1. Nén **nội dung bên trong** `tex_source/` để `main.tex` nằm ở cấp gốc ZIP; upload ZIP lên một dự án Overleaf.
2. Chọn `main.tex`, đặt compiler **XeLaTeX** rồi biên dịch; xử lý lỗi theo log, không bỏ citation hoặc caveat để ép build.
3. Kiểm tra PDF, tải về và lưu thành `docs/thesis/BaoCao_TongKet_BTL03.pdf`. Không commit `.aux`, `.log`, `.fls`, `.fdb_latexmk` hoặc PDF build tạm.

Trên Windows, lệnh nén mẫu (chạy từ `BTL03/`) là `Compress-Archive -Path docs/thesis/tex_source/* -DestinationPath $env:TEMP/BTL03_thesis.zip -Force`. Kiểm tra nội dung ZIP trước upload. Build cục bộ không đồng nghĩa đã upload/biên dịch trên Overleaf.

## Cập nhật từ T0212 sang giai đoạn tiếp theo

Tại mốc mới, đối chiếu lại trạng thái logger/test, dữ liệu/split, full training từng model, checkpoint/log, metric đúng nghĩa, parity, ablation, trực quan hóa, phân tích RQ và demo. Ghi `NOT_RECORDED`, `BLOCKED`, `NOT RUN` hoặc `[CHƯA THỰC HIỆN - DỰ KIẾN Ở GIAI ĐOẠN TIẾP THEO]` tùy bằng chứng. Không sao chép số T0212 như kết quả mới.
