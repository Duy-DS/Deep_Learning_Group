# Snapshot T0212 — nguồn LaTeX lưu trữ

Đây là snapshot lưu trữ T0212 gồm front matter, năm chương, bibliography và phụ lục evidence; không phải current authoring bundle. Markdown tương ứng là `../report.md`. Nguồn material hiện hành nằm dưới `../../../01_materials/`; đường dẫn này chỉ phục vụ repository, không được ZIP kèm khi upload.

Hai file mẫu được giữ nguyên tại `../../../../01_materials/samples/`; không đưa vào gói upload như chương hoặc asset của báo cáo.

Preamble dùng XeLaTeX với Unicode, `fontspec`/`polyglossia` và Times New Roman (cần font có sẵn trên Overleaf runtime), có thiết lập front matter, caption, bảng, TikZ và `\evidence{...}` cho nguồn ngay dưới bảng/hình. Chỉ thêm graphic vào ZIP nếu có asset đã duyệt và nguồn; snapshot T0212 không nhúng ảnh kết quả giả.

## Upload và tải PDF — ba bước

1. Đây là bản snapshot lịch sử. Nếu được yêu cầu kiểm tra lại trên Overleaf, nén **nội dung thư mục latex này** để `main.tex` ở gốc ZIP; không thay thế `02_authoring/current/` bằng snapshot này.
2. Trong Menu/Settings, chọn XeLaTeX; bấm Recompile trên Overleaf. Nếu lỗi, lưu build log và sửa source theo log; không cài compiler trên local.
3. Nếu được yêu cầu duyệt lại snapshot lịch sử, lưu PDF vào khu vực review riêng; chỉ release PDF theo quy trình tại `docs/thesis/03_releases/<version>/` sau khi có phê duyệt.

Không chạy unit/smoke test, training, inference hoặc LaTeX compiler trên máy local trong workflow viết báo cáo. Không đưa `.aux`, `.log`, `.fls`, `.fdb_latexmk`, file `.synctex.gz` hoặc PDF chưa duyệt vào Git.
