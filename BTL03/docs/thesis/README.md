# Hệ thống viết báo cáo tổng kết BTL03

Thư mục này quản lý song song bản thảo Markdown và nguồn LaTeX cho báo cáo chuyển đổi Keras sang PyTorch. Snapshot ban đầu mô tả bằng chứng trong repository tại commit T0212, không phải xác nhận nghiệm thu toàn bộ dự án. Report T0212 ghi trạng thái `BLOCKED` cho logger/full-suite; không được tuyên bố toàn bộ WP02 đã nghiệm thu.

## Cấu trúc

- `AGENT_THESIS_RULES.md`: quy tắc biên soạn, provenance và kiểm tra.
- `templates/`: khung Markdown/LaTeX gốc, bảo vệ theo quy trình; chỉ sao chép để viết.
- `source_materials/`: phạm vi snapshot, manifest nguồn và hướng dẫn lưu mẫu.
- `source_materials/samples/`: hai mẫu `.tex` do người dùng cung cấp (được giữ nguyên trong Git theo yêu cầu) và `style_analysis.md`; chỉ dùng quy ước trình bày, không chép nội dung đề tài mẫu vào BTL03.
- `md_drafts/`: Markdown để đọc nhanh và review trên Git.
- `tex_source/`: nguồn đồng bộ, nén để upload Overleaf.
- `BaoCao_TongKet_BTL03.pdf`: vị trí PDF cuối đã biên dịch/duyệt; không có trong snapshot T0212.

Tên họ và MSSV, người chủ trì báo cáo và tỷ lệ đóng góp cuối kỳ chưa được xác nhận đầy đủ. Dùng alias thành viên có bằng chứng và giữ placeholder cho trường còn thiếu; không suy đoán.

## Đồng bộ Markdown và LaTeX

1. Đọc `AGENT_THESIS_RULES.md`, đề cương và manifest.
2. Kiểm tra nguồn trước khi ghi mỗi số liệu/trạng thái; không sửa mẫu để che thiếu sót.
3. Cập nhật đồng thời `md_drafts/BaoCao_TongKet_BTL03_T0212_draft.md` và năm chương trong `tex_source/chapters/`. Giữ nhất quán bảng, trạng thái, tác giả, caveat, evidence IDs và citation.
4. Kiểm tra Git diff và cấu trúc/include/citation bằng kiểm tra văn bản tĩnh. Không chạy compiler/test/training trên local; Overleaf là nơi biên dịch PDF.
5. Khi có task mới, giữ snapshot T0212 nguyên trạng; tạo snapshot/mốc mới với commit, ngày và evidence riêng. Không hồi tố kết quả WP03/WP04 vào T0212.

### Quy ước trình bày tham khảo

Áp dụng có chọn lọc các pattern trong `source_materials/samples/style_analysis.md`: bìa/front matter, tóm tắt và từ khóa, mở đầu nêu vấn đề–mục tiêu–phạm vi–phương pháp, thân bài theo pipeline, kết quả tách khỏi thảo luận, kết luận/kiến nghị/phụ lục và tham chiếu chéo. Không sao chép nội dung, thông tin cá nhân hoặc format đặc thù trường từ mẫu; hình/bảng phải có evidence thật.

## Upload Overleaf và lưu PDF

1. Nén **nội dung bên trong** `tex_source/` để `main.tex` nằm ở cấp gốc ZIP; upload ZIP lên một dự án Overleaf.
2. Chọn `main.tex`, đặt compiler **XeLaTeX** và biên dịch trực tiếp trên Overleaf. Nếu có lỗi, tải/copy build log và cung cấp để sửa source; không cần cài TeX local.
3. Kiểm tra PDF trên Overleaf, tải về và lưu thành `docs/thesis/BaoCao_TongKet_BTL03.pdf`. Không commit file phụ trợ build (`.aux`, `.log`, `.fls`, `.fdb_latexmk`) hoặc PDF build tạm.

Không chạy lệnh XeLaTeX/latexmk hoặc build cục bộ trong workflow này. Chỉ chuẩn bị nguồn và để Overleaf biên dịch.

Khi cần hình nguồn thật, thêm asset được duyệt vào `tex_source/figures/` (đường dẫn tương đối), ghi evidence/source ID trong caption/manifest, rồi cập nhật `.md` và `.tex` cùng lúc. Hai `sample_*.tex` trong `source_materials/samples/` được lưu nguyên trạng theo yêu cầu để tham khảo; không đưa chúng vào ZIP Overleaf như chương/asset của báo cáo.

## Cập nhật từ T0212 sang giai đoạn tiếp theo

Tại mốc mới, đối chiếu lại trạng thái logger/test, dữ liệu/split, full training từng model, checkpoint/log, metric đúng nghĩa, parity, ablation, trực quan hóa, phân tích RQ và demo. Ghi `NOT_RECORDED`, `BLOCKED`, `NOT RUN` hoặc `[CHƯA THỰC HIỆN - DỰ KIẾN Ở GIAI ĐOẠN TIẾP THEO]` tùy bằng chứng. Không sao chép số T0212 như kết quả mới.
