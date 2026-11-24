# 🐛 Báo Cáo Sửa Lỗi: Sửa rụng dấu TOC/LOF, đè chữ TikZ, tỷ lệ biểu đồ cột và hoàn thiện bảng longtable

## 1. 🚨 Mô Tả Hiện Tượng (The Issue)
* **Triệu chứng lỗi:**
  - Mục lục (`.toc`) và danh mục hình (`.lof`) khi biên dịch pdfLaTeX bị rụng dấu ở các từ in hoa/tiêu đề (`CHNG 1`, `D lịu`).
  - Sơ đồ kiến trúc phần mềm TikZ ở Chương 2 (Mục 2.3) có các khối quá hẹp, nhãn `src/evaluate`, `src/inference` bị tràn và đè chữ lên nhau.
  - Các hình so sánh `t0602_rq*_metrics.png` ở Chương 3 bị ép dẹt phần thân cột do nhãn trục X quá dài (60–75 ký tự).
  - Tiêu đề Bảng 3.2 bị lỗi thoát ký tự `math{\times}`, thẻ `RQ/nhóm` chưa có khoảng trắng khiến text không tự xuống dòng và tràn sang cột `Ảnh`.
* **Nguyên nhân cốt lõi:**
  - `hyperref` được nạp trước `caption`/`titletoc` và tự động sinh PDF bookmark từ chuỗi T5 hoa gây mất dấu ký tự `Ư`, `Ơ` trong file phụ trợ.
  - Các node TikZ không có `text width` cố định và khoảng cách `node distance` 1.4cm quá hẹp.
  - Tên run trên trục X biểu đồ Matplotlib lấy nguyên chuỗi định danh thô từ MLflow.
* **Mức độ nghiêm trọng:** Trung bình - Cao (ảnh hưởng trực tiếp đến tính thẩm mỹ và độ rõ ràng của báo cáo tốt nghiệp).

## 2. 🔧 Phương Án Sửa Lỗi (Resolution)
* **Vùng mã nguồn xử lý:**
  - `BTL02/docs/thesis/templates/btl02-style.sty`: Nạp `[hidelinks,unicode,psdextra]{hyperref}` sau `caption` và `titletoc`; tắt bookmark tự động cho pdfLaTeX (`\hypersetup{bookmarks=false}`) để bảo toàn 100% tiếng Việt trong `.toc` và `.lof`.
  - `BTL02/docs/thesis/templates/chap2_system.tex.j2`: Thiết lập `text width=3.4cm`, `minimum height=1.1cm`, tăng khoảng cách node (Col 1: 0, Col 2: 5.0cm, Col 3: 10.0cm; Row 1: 0, Row 2: -2.4cm) giúp các khối kiến trúc thoáng, chữ nằm gọn trong khung.
  - `BTL02/src/evaluate/latex_exporter.py`:
    - Ánh xạ các tag `params.rq_tag` dài thành chuỗi có dấu cách (`RQ2 SOTA (Pretrained)`, `RQ3 Ablation (Loss)`...) cho phép xuống dòng an toàn.
    - Chỉnh bảng sang cỡ `\footnotesize`, `\tabcolsep=4pt`, `\LTcapwidth=\textwidth`, độ rộng cột chuẩn `p{1.5cm}`, `p{2.3cm}`, `p{3.0cm}`, `c`, `c`, `r`, `r`, `r`.
    - Viết hàm `shorten_run_name` rút gọn tên run trên biểu đồ (`EXP-09 (Aug-Basic)`, `EXP-11 (Loss-CE)`...), giảm góc xoay xuống 35 độ để thân cột biểu đồ cao và rõ ràng.
    - Sửa tiêu đề bảng sang dạng toán `$128 \times 128$`.
  - `BTL02/scripts/build_thesis_bundle.py`: Bổ sung loại trừ các file phụ trợ `.lof`, `.lot`, `.pdf`, `.synctex.gz` khỏi gói nén zip.

## 3. 🧪 Xác Minh Sau Khi Sửa (Verification)
* **Kết quả kiểm thử:**
  - Toàn bộ 16 tests tự động đạt: `16 passed in 5.94s` (`pytest BTL02/tests/test_latex_report.py -v`).
  - Đã dọn dẹp sạch sẽ các tệp rác biên dịch trong `overleaf_bundle/`.
  - Gói `BTL02/docs/thesis/BTL02_Overleaf_Bundle.zip` đã được tái xuất hoàn chỉnh.
