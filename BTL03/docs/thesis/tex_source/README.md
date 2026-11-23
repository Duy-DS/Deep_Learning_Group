# Nguồn báo cáo T0212 để upload Overleaf

Thư mục này là bộ mã nguồn để upload và biên dịch **trên Overleaf**, gồm front matter, năm chương, bibliography và phụ lục evidence. Nội dung đồng bộ với `../md_drafts/BaoCao_TongKet_BTL03_T0212_draft.md`; quy ước tham khảo hai mẫu được tổng hợp trong `../source_materials/samples/style_analysis.md`. Compiler trên Overleaf: **XeLaTeX**. Template gốc nằm ở `../templates/latex_template/`.

Preamble dùng XeLaTeX với Unicode, `fontspec`/`polyglossia` và Times New Roman (cần font có sẵn trong runtime), có thiết lập front matter, caption, bảng, TikZ và `\evidence{...}` cho nguồn ngay dưới bảng/hình. Chỉ thêm graphic vào ZIP nếu có asset đã duyệt và nguồn; snapshot T0212 không nhúng ảnh kết quả giả.

## Upload và tải PDF — ba bước

1. Nén nội dung `tex_source/` (để `main.tex` ở gốc ZIP), upload lên dự án Overleaf mới và chọn `main.tex`.
2. Trong Menu/Settings, chọn XeLaTeX; bấm Recompile trên Overleaf. Nếu lỗi, lưu build log và sửa source theo log; không cài compiler trên local.
3. Kiểm tra danh tính/placeholder, trạng thái `BLOCKED`, số liệu và citation trong PDF; tải bản đã duyệt về `docs/thesis/BaoCao_TongKet_BTL03.pdf`.

Không chạy unit/smoke test, training, inference hoặc LaTeX compiler trên máy local trong workflow viết báo cáo. Không đưa `.aux`, `.log`, `.fls`, `.fdb_latexmk`, file `.synctex.gz` hoặc PDF chưa duyệt vào Git.
