# Phân tích cấu trúc và phong cách từ hai mẫu LaTeX

**Mục đích:** dùng `sample_1.tex` và `sample_2.tex` do người dùng cung cấp làm ví dụ về tổ chức báo cáo. Đây là phân tích quy ước, không phải sao chép nội dung. Không xem format của mẫu là quy định chính thức của trường.

## 1. Điểm chung

- Phân tách front matter và nội dung đánh số: bìa, mục lục/danh mục, tóm tắt hoặc thuật ngữ trước chương chính.
- Mở đầu làm rõ bối cảnh, vấn đề, mục tiêu/nhiệm vụ, phạm vi/đối tượng và phương pháp; mẫu thứ hai còn nêu đóng góp và cấu trúc báo cáo.
- Thân bài có luồng từ nền tảng đến dữ liệu/phương pháp, kết quả và thảo luận; phần cuối tách kết luận/kiến nghị/giới hạn và phụ lục.
- Dùng phân cấp chương–mục, hình/bảng/phương trình có caption/label và cross-reference để người đọc tra cứu.
- Mẫu theo pipeline giải thích artifact trong mạch lập luận; mẫu theo người dùng phân chia kết quả theo persona/use case và kiểm toán tiêu chí hệ thống.

## 2. Đặc điểm và cách chuyển dụng của từng mẫu

### `sample_1.tex` — báo cáo nghiên cứu theo pipeline

Tổ chức: bìa → thuật ngữ/chữ viết tắt → tóm tắt → mở đầu → lý thuyết → dữ liệu/tiền xử lý → phương pháp/biểu diễn → tối ưu hóa → thực nghiệm/kết quả/thảo luận → kết luận/kiến nghị → tài liệu tham khảo/phụ lục. Có cấu trúc nghiên cứu rõ, nhiều bảng/hình/phương trình và mạng tham chiếu chéo.

**Chuyển dụng BTL03:** dùng mạch cơ sở Keras/PyTorch → dữ liệu/baseline → kiến trúc và chuyển đổi → kiểm toán/test → kết quả run thật → thảo luận validity. Không viết chương kết quả theo cách giả định đủ mười full run khi evidence T0212 không có.

### `sample_2.tex` — báo cáo hệ thống/visual analytics theo người dùng

Tổ chức: bìa và bảng thành viên → lời cảm ơn/tóm tắt → mở đầu có mục tiêu/phạm vi/phương pháp/đóng góp → chương lý thuyết → data pipeline/hợp đồng dữ liệu → phân tích khám phá → thiết kế theo use case → kiểm toán kỹ thuật/hiệu năng/tiếp cận → kết luận, giới hạn, tài liệu tham khảo, phụ lục. Sơ đồ pipeline và hình/bảng được gắn với chức năng hệ thống.

**Chuyển dụng BTL03:** mô tả pipeline Dataset → model → Trainer/Callback → logger/checkpoint → CLI/evaluation; dùng ma trận năm nhóm/mười cấu hình. Có thể nhóm theo vai trò người đọc (người tái lập, reviewer, người đánh giá) nhưng không tạo persona/demo chưa được xác nhận WP04.

## 3. Quy ước trình bày được chọn cho BTL03

1. **Bìa:** tên đề tài/mã BTL03, mốc và commit snapshot, ngày chốt evidence, alias thành viên đã xác nhận. Họ tên/MSSV/giảng viên/tỷ lệ giữ placeholder nếu thiếu xác nhận.
2. **Front matter:** tóm tắt, từ khóa, danh mục từ viết tắt khi cần, mục lục và danh mục hình/bảng chỉ khi có nội dung thực.
3. **Mở đầu:** bối cảnh → mục tiêu/câu hỏi → đối tượng/phạm vi → cách kiểm toán evidence → đóng góp đã kiểm chứng → hạn chế → cấu trúc báo cáo.
4. **Thân bài:** nền tảng/chuyển đổi → dữ liệu và baseline/protocol → kiến trúc OOP/pipeline → kết quả code/test/parity → kết quả full training (chỉ sau khi có run thật) → thảo luận và validity.
5. **Phần cuối:** kết luận snapshot, kiến nghị giai đoạn tiếp theo, tài liệu tham khảo; phụ lục cho evidence manifest, acceptance matrix, config/lệnh và bảng chi tiết dài.
6. **Hình/bảng:** caption tự mô tả; giới thiệu và diễn giải trong văn bản; label/cross-reference ổn định; ghi evidence ID, run/split/seed và caveat liên quan. Sơ đồ phải phản ánh source/config thực tế.

## 4. Ranh giới nội dung và kỹ thuật

- Không sao chép bìa, lời cảm ơn, tên/MSSV, dữ liệu, hình, trích dẫn hoặc kết luận chuyên ngành từ các sample.
- Không bê nguyên preamble/package, macro, màu sắc, layout hay các package đặc thù của mẫu. XeLaTeX và package stack của BTL03 vẫn là lựa chọn hiện tại; mọi thay đổi phải build kiểm tra.
- Không suy ra yêu cầu lề/font/cỡ chữ chính thức từ hai mẫu; xin mẫu/quy định học phần nếu cần chuẩn in.
- Bảng/hình chưa có evidence để placeholder có nhãn “chưa có”, không dựng kết quả giả. WP03/WP04, logger và training giữ trạng thái đúng evidence, đặc biệt T0212 `BLOCKED`.
- Kết luận khác với khuyến nghị: kết luận tổng hợp điều evidence chứng minh; khuyến nghị nêu hành động tương lai, có phụ thuộc và cách nghiệm thu.