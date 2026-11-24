# Phân tích phong cách trình bày của `sample_1.tex` và `sample_2.tex`

**Phạm vi:** đọc toàn bộ hai file do người dùng cung cấp. Inventory hiện tại lần lượt là 1.459 và 998 dòng. Tài liệu này phân tích cấu trúc/biểu đạt, không xác nhận quy định trường. Hai file gốc được giữ nguyên; tuyệt đối không chép nội dung, số liệu, tên/MSSV, hình hoặc citation của đề tài mẫu vào báo cáo BTL03.

## 1. Kiến trúc tài liệu và phân cấp

| Khía cạnh | `sample_1.tex` | `sample_2.tex` | Chọn lọc cho BTL03 |
|---|---|---|---|
| Class | `book`, A4, 12pt, `openany`, `oneside` | `report`, A4, 12pt, `oneside` | Giữ `book` và XeLaTeX hiện hành trong BTL03; không đổi class chỉ để giống hình thức |
| Preamble | Nạp một gói tùy biến `research-format` | Preamble inline, nhánh `iftex`, font/geometry/heading/table/TikZ/callout/hyperref | Giữ preamble modular BTL03; `research-format.sty` không có trong repo, không bê nguyên package stack của sample 2 |
| Front matter | Bìa → danh mục thuật ngữ → tóm tắt → TOC/LoF/LoT → mainmatter | Bìa và bảng thành viên → lời cảm ơn → TOC/LoF/LoT → thuật ngữ → tóm tắt → mở đầu | Một bìa; snapshot/hash/date; thành viên có evidence, placeholders pháp lý; tóm tắt, từ khóa, abbreviations/RQ, mục lục |
| Phân cấp | Chương → section → subsection; có phần không đánh số và phần đánh số | Chương `report`, mở đầu/kết luận dạng không đánh số, mục đánh số thủ công ở front matter | Dùng 5 chương T0402 có cấp bậc nhất quán; tránh số thủ công dễ lệch cross-reference |
| Luồng | Pipeline nghiên cứu: nền tảng → dữ liệu → mô hình → kết quả | Pipeline hệ thống và user stories → analytics → dashboard/audit | Kết hợp: framework/data → architecture → audit/results → discussion/validity |

## 2. Văn phong và mạch diễn giải

### 2.1 Pattern quan sát trong sample 1

- Mở đầu nêu lý do, lịch sử nghiên cứu, mục đích/nhiệm vụ, đối tượng/phạm vi, phương pháp và cấu trúc.
- Chương nối tiếp theo luồng công việc; bảng/hình được giới thiệu trong văn bản, sau đó diễn giải ý nghĩa và chuyển tiếp sang phần kế tiếp.
- Phần lý thuyết định nghĩa ký hiệu trước; công thức có nhãn, giải nghĩa biến và nêu cách dùng trong pipeline.
- Kết quả được thảo luận cùng giả định/giới hạn, không chỉ xếp bảng số liệu; phụ lục gom code/evidence theo bước nghiên cứu.

### 2.2 Pattern quan sát trong sample 2

- Front matter và mở đầu thiên về deliverable: mục tiêu, nhiệm vụ, phạm vi, phương pháp, đóng góp, cấu trúc.
- Chương data contract và pipeline đi trước phân tích; thiết kế trực quan được tổ chức theo người dùng/chức năng.
- Sơ đồ TikZ, bảng kiểm toán và callout giúp giải thích luồng hệ thống/tiêu chí; kết luận tách findings, hạn chế, hướng phát triển.
- Giọng văn nhấn mạnh claim cụ thể đi với artifact/kiểm toán, phù hợp báo cáo hệ thống kỹ thuật.

### 2.3 Quy tắc diễn đạt áp dụng

Áp dụng mạch **claim → evidence/path/Run ID → diễn giải → caveat**. Phân biệt code/config, unit/shape test, synthetic smoke, legacy output và PyTorch full-data run. Không dùng lời khẳng định tuyệt đối (“hoàn toàn tương đương”, “đã đạt”) nếu evidence chỉ là parameter count hoặc smoke giả. Tách kết luận snapshot khỏi kiến nghị tương lai.

## 3. Bảng, công thức, code, hình và trích dẫn

| Thành phần | Pattern trong mẫu | Quy ước BTL03 T0402 |
|---|---|---|
| Bảng | Sample 1 dùng `longtable`, header lặp và cột tự định nghĩa trong package riêng; sample 2 dùng `tabularx`, bảng có khung/stripe, acceptance và audit tables | Dùng `booktabs`, `longtable` cho bảng dài, `tabularx` cho mô tả; caption/label và giới thiệu trong prose; ghi unit/split/source/caveat |
| Công thức | Sample 1 đánh số equation, label/cross-reference và giải thích ký hiệu; sample 2 dùng equation cho công thức thống kê và tiêu chí | Chỉ đưa công thức cần thiết cho tensor/loss/metric; giải thích điều kiện và ký hiệu; Markdown dùng cùng ký hiệu LaTeX |
| Code/thuật toán | Sample 1 chủ yếu có ảnh chụp code trong phụ lục; sample 2 diễn giải thuật toán, có inline code và sơ đồ | BTL03 có `listings`; dùng đoạn code ngắn có trong source, không screenshot giả, không shell-escape/minted |
| Hình/sơ đồ | Cả hai dùng figure, caption, label và tham chiếu; sample 2 dùng TikZ cho pipeline/What–Why–How | Chỉ chèn asset có provenance; path tương đối; TikZ chỉ mô tả source/config thực tế và caption nêu giới hạn |
| Bibliography | Cả hai dùng `thebibliography` và citation key | Bundle BTL03 dùng BibLaTeX/Biber với `.bib`; Markdown giữ danh sách/key tương ứng; chỉ metadata đã xác minh |
| Bố cục | Sample 2 có màu/callout/logo và quy định trường cụ thể; sample 1 gọi custom style | Không suy lề/font/cỡ chữ chính thức từ sample; dùng XeLaTeX, font tiếng Việt và package stack BTL03 hiện có |

## 4. Tương thích Overleaf và đồng bộ Markdown ↔ LaTeX

- BTL03 hiện cấu hình XeLaTeX với `fontspec`, `polyglossia` Vietnamese, `biblatex`/Biber và TikZ. Nguồn phải nén thành bundle tự chứa, `main.tex` ở root ZIP, mọi `\input`/ảnh dùng đường dẫn tương đối.
- Sample 1 gọi `research-format`; package không hiện diện trong repository. Sample 2 có nhánh pdfLaTeX/XeLaTeX, macro, geometry, palette, logo/figures và callout riêng của đề tài đó. Không sao chép trực tiếp các phụ thuộc này.
- Markdown và TeX phải có cùng thứ tự chương, bảng, số, split, trạng thái, caveat, tác giả, RQ và citation. Bảng dài có thể rút gọn khi dễ đọc nhưng mọi giá trị/kết luận phải khớp.
- PDF chỉ được tạo/duyệt trên Overleaf theo quy tắc thesis; không cài TeX, không build local. Không đưa `.aux/.log/.fls/.fdb_latexmk` hoặc PDF chưa duyệt vào Git.

## 5. Ánh xạ phong cách vào BTL03

1. **Front matter:** bìa có tên đề tài, mã, snapshot/hash/date; alias theo evidence; placeholder rõ cho họ tên/MSSV/giảng viên/tỷ lệ chưa xác nhận.
2. **Mở đầu:** bối cảnh → mục tiêu/RQ đã xác minh → phạm vi/snapshot → cách chọn evidence → đóng góp đã chứng minh → giới hạn → cấu trúc.
3. **Chương 1:** WBS và năm nhóm/mười cấu hình; phân biệt cấu hình với kiến trúc độc lập.
4. **Chương 2:** Keras/PyTorch, NHWC/NCHW, rotation, TimeDistributed, logits/loss, upsampling/mask và parity risks.
5. **Chương 3:** pipeline OOP; component matrix theo code hiện hành; sơ đồ có caption nói rõ test/synthetic/logger status.
6. **Chương 4:** parameter audit, legacy-vs-PyTorch results, protocol, task status, evidence matrix; giữ T0212 BLOCKED và discrepancy CNN-LSTM.
7. **Chương 5/phụ lục:** validity, bài học, kiến nghị có điều kiện; manifest/run IDs và tài liệu tham khảo.

## 6. Ranh giới sample và nguồn gốc

- Không sao chép nội dung chuyên ngành VN30/ETT, số liệu, figure, citation, URL, bìa/logo, tên/MSSV hay đánh giá từ sample.
- Không sửa sample gốc. Template gốc BTL03 cũng giữ READ-ONLY; template style v2 là bản mới, độc lập.
- Format mẫu chỉ là tham khảo hình thức, không phải quy định chính thức của trường.
- Hình/bảng chưa có evidence phải ghi “chưa có evidence” hoặc không đưa vào; không sinh nội dung giả để lấp bố cục.