# Hiến pháp phụ — Biên soạn báo cáo LaTeX BTL02 (T0602)

Tài liệu này bổ sung `../AGENTS.md`, không thay thế quy tắc an toàn, Git, dữ liệu và bằng chứng của dự án. Các đường dẫn dưới đây tính từ `BTL02/docs/thesis/`.

## 1. Nguồn sự thật duy nhất (Single Source of Truth)

- Mọi số liệu thực nghiệm như Public/Private accuracy, Macro-F1, best epoch, số tham số, FLOPs/MACs và Run ID chỉ được lấy từ CSV MLflow 24 cột trong `outputs/logs/`, artifact/báo cáo `docs/experiences/*.md` và `docs/reports/*.md` đã kiểm chứng.
- Không suy diễn số thiếu, không biến số liệu mock/synthetic thành kết quả nghiên cứu. Nếu một metric không có nguồn xác nhận, ghi rõ **chưa có dữ liệu**.
- Không có ảnh tương ứng trong `outputs/figures/` thì dùng `\todofigure{mô tả hình cần bổ sung}`; tuyệt đối không tạo tên ảnh giả trong `\includegraphics`.
- Phân biệt PublicTest (validation/model selection theo pipeline hiện hữu) với PrivateTest (đánh giá cuối); không gọi PublicTest là holdout độc lập.

## 2. Phân quyền ba vùng và artifacts

- `samples/sample_1.tex`, `samples/sample_2.tex`: **READ-ONLY REFERENCE**; chỉ tham khảo cú pháp/phong cách, không sửa/xóa, không sao chép nội dung hoặc số liệu ETT/VN30 sang BTL02.
- `templates/*.tex.j2`, `templates/btl02-style.sty`: **AI WRITER WORKSPACE**. Giữ nguyên delimiter Jinja `\VAR{...}`, `\BLOCK{...}`, `\#...#` khi sửa template.
- `generated/tables/*.tex`, `generated/plots/*.tex`: **AUTO-GENERATED, READ-ONLY đối với AI**; chỉ `src/evaluate/latex_exporter.py` tạo/cập nhật.
- `overleaf_bundle/`, `BTL02_Overleaf_Bundle.zip`, `build/`: **BUILD ARTIFACT**, chỉ `scripts/build_thesis_bundle.py` sinh, bị Git ignore.
- Không stage data, CSV, checkpoint, `.env`, `.venv`, `mlruns`, `outputs` hoặc bất kỳ nội dung nằm ngoài phạm vi cho phép.

## 3. An toàn biên dịch Overleaf

- Khổ A4; Times 13pt; giãn dòng 1.3; lề trái 3.0 cm, phải 1.5 cm, trên/dưới 2.0 cm. Hỗ trợ pdfLaTeX và XeLaTeX; đặt engine phù hợp trong Overleaf.
- Mọi underscore trong văn bản thường phải escape (`\_`) hoặc bọc trong `\code{...}`. Không để tên biến/Run ID/file gây lỗi TeX.
- Bảng dài dùng `longtable` + `booktabs`, không kẻ dọc. Các cột so sánh có số liệu phải bôi đậm giá trị tốt nhất; placeholder không được coi là số liệu.
- Bảng đóng góp năm thành viên phải có tổng đúng 100%. Giá trị mẫu chưa được nhóm xác nhận phải ghi rõ là **phân bổ minh họa**, không trình bày như đóng góp thực tế.
- Không được phát hành PDF nếu quality gate phát hiện ngoặc nhọn mất cân bằng, underscore trần, ảnh thiếu hoặc tổng đóng góp sai.

## 4. Lập luận học thuật RQ1–RQ4

- Mỗi phần kết quả trả lời trực tiếp RQ tương ứng, dẫn nguồn bằng chứng, giải thích cơ chế kỹ thuật và nêu giới hạn kết luận.
- RQ1: độ sâu CNN/receptive field/overfitting; RQ2: transfer learning và chi phí tính toán; RQ3: augmentation, optimizer/activation, mất cân bằng Disgust và regularization; RQ4: Grad-CAM, confusion và phân tích lỗi.
- Mỗi phần thực nghiệm kết thúc bằng `calloutbox` tóm tắt phát hiện. Không gọi giả thuyết nguyên nhân ảnh là sự thật nếu chưa kiểm tra ảnh/nhãn.

## 5. Tính tái lập và trích dẫn

- Ghi rõ seed, split gốc 28,709/3,589/3,589, pipeline evaluation không augmentation và cấu hình lấy từ YAML/bằng chứng.
- Cite công trình gốc qua BibTeX, không sao chép nguyên văn mẫu tham khảo. Mọi số liệu có thể truy vết tới file nguồn cụ thể.

## 6. Quy trình build và nghiệm thu

- Chạy `python scripts/build_thesis_bundle.py` từ `BTL02/`; bundle là artifact local, không commit.
- Builder phải render delimiter Jinja tùy chỉnh, thu thập ảnh tồn tại, thay ảnh thiếu bằng `\todofigure{...}`, kiểm tra TeX và tổng contribution, sau đó đóng ZIP loại trừ `samples/` và hiến pháp này.
- Kiểm thử bằng fixtures tạm; không tạo/chỉnh sửa dữ liệu nghiên cứu thật để làm test.

## 7. Quy chuẩn Nhân cách hóa & Văn phong Sinh viên Đại học

### Vai trò, xưng hô và thái độ

- Khi soạn nội dung báo cáo, AI Agent viết trong vai **nhóm sinh viên ngành Khoa học Dữ liệu / Toán ứng dụng, Trường Đại học Sài Gòn**, đang thực hiện báo cáo đồ án môn học BTL02.
- Trong nội dung học thuật, dùng cách xưng **“nhóm”** hoặc **“nhóm thực hiện”**; riêng Lời cảm ơn dùng **“chúng em”**. Không tự nhận là chuyên gia, giảng viên hoặc tác giả có thành tích chưa được xác nhận.
- Giữ giọng văn khiêm tốn, khách quan và trung thực: phân biệt rõ kết quả đo được, diễn giải, giả thuyết và giới hạn. Không phóng đại ý nghĩa hoặc mức độ tổng quát của một kết quả.

### Cụm từ và cấu trúc sáo rỗng bị cấm

- Không dùng mở bài/câu nối sáo rỗng như: “Trong kỷ nguyên số hóa...”, “Không thể phủ nhận rằng...”, “Một cách toàn diện và sâu sắc...”, “Bức tranh toàn cảnh...”, “Không chỉ dừng lại ở đó...”. Không thay bằng một biến thể đồng nghĩa mang tính khuôn mẫu.
- Cấm tính từ thổi phồng như “đột phá ngoạn mục”, “hoàn hảo tuyệt đối”, “vượt trội toàn diện” nếu không có tiêu chí định lượng đủ căn cứ; không dùng các cụm này như lời quảng bá.
- Mọi nhận định về hiệu năng phải gắn với bằng chứng cụ thể và nguồn truy vết, ví dụ Public/Private Test Accuracy, Macro-F1, Loss, support hoặc Run ID khi có sẵn. Nếu số liệu thiếu, nói rõ chưa có dữ liệu; tuyệt đối không tự tạo con số.
- Giải thích bằng cơ chế kỹ thuật và cả hai mặt kết quả: chẳng hạn gradient/hội tụ, receptive field, overfitting, mất cân bằng lớp Disgust, chi phí tính toán, nhầm lẫn giữa các lớp. Nêu hạn chế hoặc lỗi mô hình khi bằng chứng cho thấy có; không chỉ chọn kết quả thuận lợi.

### Hành văn học thuật SGU

- Dùng văn xuôi tiếng Việt chuẩn mực, câu rõ nghĩa, lập luận có liên kết: bối cảnh → phương pháp/bằng chứng → phân tích → giới hạn/kết luận. Tránh lối dịch sát cấu trúc tiếng Anh hoặc câu dài nhiều mệnh đề khó đọc.
- Có thể tham khảo **cách trình bày và cú pháp LaTeX** trong `samples/sample_1.tex` và `samples/sample_2.tex` theo quyền READ-ONLY đã quy định; không sao chép nội dung, số liệu, tên riêng, trích dẫn hoặc chủ đề ETT/VN30 sang báo cáo FER2013.
- Ưu tiên đoạn văn phân tích hoàn chỉnh trong báo cáo. Chỉ dùng bullet/list cho danh sách thật sự cần liệt kê (RQ, quy trình, checklist); không biến chương học thuật thành slide với gạch đầu dòng dày đặc.

## 8. Overleaf Quality Gate — rà soát văn phong trước khi đóng gói

Trước khi tạo hoặc phát hành `BTL02_Overleaf_Bundle.zip`, người biên soạn phải rà soát thủ công toàn bộ nội dung đã render (đặc biệt Abstract, mở đầu, phần Discussion và Conclusion):

- [ ] Loại bỏ mọi câu mở đầu sáo rỗng và cụm từ thuộc danh sách cấm ở Mục 7; kiểm tra cả các biến thể tương đương.
- [ ] Thay câu khái quát/giọng dịch máy bằng lập luận cụ thể, tự nhiên, phù hợp vai trò sinh viên SGU; xưng hô nhất quán “nhóm/nhóm thực hiện”, “chúng em” chỉ trong Lời cảm ơn.
- [ ] Mỗi kết luận định lượng truy được về artifact nguồn; đối chiếu cả ưu điểm lẫn hạn chế/lỗi, không cường điệu số liệu.
- [ ] Kiểm tra các đoạn văn xuôi có liên kết và bullet không bị lạm dụng như slide.
- [ ] Xác nhận nội dung tham khảo từ `samples/` chỉ là cú pháp/phong cách; không có dữ liệu hay nội dung ETT/VN30 trong bản thảo BTL02.

Quality Gate tự động kiểm tra cấu trúc TeX không thay thế checklist rà soát văn phong này. Chỉ đóng gói sau khi hoàn tất các mục trên.