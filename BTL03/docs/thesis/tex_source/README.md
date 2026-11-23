# Nguồn báo cáo T0212 để upload Overleaf

Thư mục này là bản sao có thể biên dịch, đồng bộ nội dung với `../md_drafts/BaoCao_TongKet_BTL03_T0212_draft.md`. Compiler được chọn: **XeLaTeX**. Các tệp trong `chapters/` là bản nguồn xuất bản; template gốc nằm riêng ở `../templates/latex_template/`.

## Upload và tải PDF — ba bước

1. Nén nội dung `tex_source/` (để `main.tex` ở gốc ZIP), upload lên dự án Overleaf mới và chọn `main.tex`.
2. Trong Menu/Settings, chọn XeLaTeX; bấm Recompile. Nếu cần build lại cục bộ: `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`.
3. Kiểm tra danh tính/placeholder, trạng thái `BLOCKED`, số liệu và citation trong PDF; tải bản đã duyệt về `docs/thesis/BaoCao_TongKet_BTL03.pdf`.

Build cục bộ không chứng minh đã build trên Overleaf. Không đưa `.aux`, `.log`, `.fls`, `.fdb_latexmk`, file `.synctex.gz` hoặc PDF chưa duyệt vào Git.
