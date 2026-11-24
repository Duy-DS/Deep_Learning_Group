# 🐛 Báo Cáo Sửa Lỗi: Khắc phục vỡ font tiếng Việt, đè cột bảng longtable và hoàn thiện kết luận báo cáo Overleaf

## 1. 🚨 Mô Tả Hiện Tượng (The Issue)
* **Triệu chứng lỗi:**
  - Tiêu đề bìa và nhãn hộp callout bị vỡ dấu tiếng Việt thành chuỗi dấu hỏi (`TR??NG ??I H?C S?I G?N`, `Ph?t hi?n ch?nh`, `H?nh c?n b? sung`).
  - Khi biên dịch bằng TeXstudio, các nhãn mục lục, danh mục hình, danh mục bảng hiển thị tiếng Anh mặc định (`Contents`, `List of Figures`, `List of Tables`, `Chapter 1`).
  - Mục lục bị ngắt dòng gãy từ số chương (`2 Dữ | liệu`, `4 | quan hóa... Trực`).
  - Các bảng thực nghiệm 3.2, 3.4, 3.5 bị tràn và đè chữ giữa các cột do `Run ID` 32 ký tự và số thập phân hiển thị 15-16 chữ số; bảng thiếu `\caption` và `\label` nên Mục lục bảng trống và xuất hiện lỗi `Bảng ??` ở mục 3.2.
  - Ô bảng thuật ngữ quy ước nhãn dính ký tự `&` thô trong một cột (`$0=$ Angry & Giận dữ`).
  - Công thức (1.3) và (1.5) chưa chuẩn cú pháp `cases`, thiếu phần kết luận sinh viên hoàn chỉnh.
* **Nguyên nhân cốt lõi:**
  - `btl02-style.sty` cấu hình babel với tiếng Anh là tùy chọn cuối cùng (`[vietnamese,english]{babel}`); chuỗi macro bìa bị thay thế bởi chuỗi hỏng UTF-8.
  - Exporter bảng (`latex_exporter.py`) chia đều cột `p{0.1125\textwidth}` không đủ chỗ cho Run ID và số thực chưa làm tròn; không chèn `\caption` và `\label` vào môi trường `longtable`.
  - `build_thesis_bundle.py` escape nhầm `\\&` trong `label_rows`.
* **Mức độ nghiêm trọng:** Khẩn cấp (ảnh hưởng trực tiếp đến chất lượng nộp bài và khả năng biên dịch TeXstudio/Overleaf).

## 2. 🔧 Phương Án Sửa Lỗi (Resolution)
* **Vùng mã nguồn xử lý:**
  - `BTL02/docs/thesis/templates/btl02-style.sty`: Chuẩn hóa `[vietnamese]{babel}` cho pdfLaTeX, bổ sung `\addto\captionsvietnamese`, cấu hình `titletoc` chống gãy từ mục lục; phục hồi 100% tiếng Việt UTF-8 cho bìa, `todofigure`, `calloutbox`.
  - `BTL02/docs/thesis/templates/chap1_theory.tex.j2`: Chuẩn hóa môi trường `\begin{cases}` cho `LeakyReLU` và `ELU`, sửa định danh `EfficientNet-B0`.
  - `BTL02/docs/thesis/templates/chap3_exp.tex.j2`: Xóa bỏ caption thô ngoài bảng, tách rời Bảng 3.1 và Hình 3.1, liên kết tham chiếu `\ref{tab:...}` chuẩn xác.
  - `BTL02/docs/thesis/templates/conclusion.tex.j2`: Hoàn thiện toàn bộ nội dung Kết luận và Hướng phát triển theo đúng 4 câu hỏi nghiên cứu (`RQ1`–`RQ4`) bằng văn phong sinh viên học thuật trung thực, khách quan.
  - `BTL02/src/evaluate/latex_exporter.py`: Rút gọn Run ID 8 ký tự kèm `\code{...}`, làm tròn số học thuật (Acc 2 chữ số, F1 4 chữ số, Best epoch số nguyên, Params/FLOPs phân tách hàng nghìn), thiết kế độ rộng cột tùy chỉnh `col_spec` ($\le 0.94\textwidth$), bổ sung `caption`, `label` và `note` vào `longtable`.
  - `BTL02/scripts/build_thesis_bundle.py`: Sửa `label_rows` tách đúng 2 cột ký hiệu và cảm xúc tiếng Việt.
  - `BTL02/tests/test_latex_report.py`: Bổ sung kiểm thử tự động `test_rendered_tex_contains_no_replacement_or_missing_ref_marks` phát hiện ký tự `\ufffd` hoặc `??`.

## 3. 🧪 Xác Minh Sau Khi Sửa (Verification)
* **Kết quả kiểm thử:**
  - Render thành công 5 bảng TeX và bundle Overleaf:
    `Quality Gate: {'tex_files': 14, 'syntax_errors': 0, 'missing_figures': 0, 'contribution_total': 100.0}`
  - Bộ kiểm thử tự động: `16 passed in 6.00s` (`pytest BTL02/tests/test_latex_report.py -v`).
  - Gói `BTL02/docs/thesis/BTL02_Overleaf_Bundle.zip` đã sẵn sàng tải lên Overleaf / TeXstudio biên dịch không còn lỗi.
