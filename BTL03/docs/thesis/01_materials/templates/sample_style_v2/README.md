# BTL03 thesis template — style v2

Đây là template phiên bản mới, dựng từ phân tích cấu trúc/trình bày của `01_materials/samples/sample_1.tex` và `sample_2.tex`. Template legacy và hai file sample vẫn được giữ nguyên READ-ONLY.

## Ý tưởng biên tập

- Mượn mạch nghiên cứu của sample 1: vấn đề → nền tảng → dữ liệu/phương pháp → thực nghiệm → thảo luận/kết luận → phụ lục.
- Mượn cách sample 2 tách front matter, mục tiêu/phạm vi/phương pháp, pipeline, acceptance/evidence matrix và góc nhìn hệ thống.
- Không sao chép package, bìa, PII, số liệu, ảnh, bibliography hoặc nội dung từ đề tài sample.
- Bản Markdown và LaTeX có cùng 5 chương, cùng thứ tự bảng/evidence/caveat. Dùng placeholder có nhãn cho dữ liệu nhóm chưa xác nhận.

## Dùng template

1. Sao chép file Markdown vào `docs/thesis/02_authoring/current/report.md` và bộ `latex_template/` vào `docs/thesis/02_authoring/current/latex/`.
2. Điền snapshot, tác giả thật, evidence paths, metric/run IDs đã xác minh; không sửa số thiếu bằng suy đoán.
3. Giữ XeLaTeX/Overleaf của BTL03. Nén **nội dung** `02_authoring/current/latex/` để `main.tex` nằm ở root ZIP.
4. Không sửa hai sample gốc hoặc template gốc `[READ-ONLY]`; không build/test local trong workflow viết luận.
