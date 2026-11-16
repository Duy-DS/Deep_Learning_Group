# [BUG-FIX] Chuẩn hóa tên báo cáo WP03 tương thích Windows

## Thông tin xử lý

- **Task:** `BPM25_T0305`
- **Thời gian:** 2026-11-16 19:59
- **Nhánh:** `feat/BPM25_WP03_SOTA_Alb_update`
- **Phạm vi:** Làm sạch tên file và Git tracking trong `course_project/` để nhánh có thể checkout/merge trên Windows.

## Triệu chứng

Merge/checkout nhánh trên Windows dừng với `invalid path` tại hai báo cáo trong `docs/reports/`. Windows không cho phép dấu `:` trong tên file.

## Nguyên nhân gốc rễ

Hai tên báo cáo dùng dấu `:` để phân cách giờ/phút, trái với quy ước `HHMM` đủ bốn chữ số và mẫu `[Task ID]_[YYMMDD]_[HHMM].md` trong Mục III.8 của `AGENTS.md`.

## Thay đổi và làm sạch

- `docs/reports/Ablation_261116_11:12.md` → `docs/reports/Ablation_261116_1112.md`.
- `docs/reports/T0305_261116_1:11.md` → `docs/reports/T0305_261116_0111.md`.
- Quét toàn bộ `course_project/docs/`; không còn tên file chứa ký tự cấm của Windows (`:`, `*`, `?`, `"`, `<`, `>`, `|`).
- Bổ sung quy tắc `.gitignore` cho bytecode/cache, checkpoint/trọng số, dữ liệu CSV/bảng, môi trường cục bộ và `.ignore/`.
- Bỏ Git tracking các artifact bị cấm trong `course_project/`: 123 CSV, 136 file trọng số, 10 file bytecode và 24 file dữ liệu raw. Không xóa nội dung local; các file local được giữ lại và bị ignore.
- Không stage thay đổi ngoài `course_project/`; các stash đã có trước đó được giữ nguyên.

## Bằng chứng kiểm tra

```text
PASS: Python source compilation — 14 tracked Python files (`python -m py_compile`).
PASS: Quét conflict markers trong src/, configs/, scripts/, training_loop.py và utils_metrics.py — không phát hiện.
PASS: Tên file trong course_project/docs/ — không còn ký tự cấm Windows.
PASS: Git index trong course_project/ — không còn CSV, checkpoint/trọng số, bytecode/cache, data/raw hoặc thư mục .ignore bị track.
PASS: .gitignore có quy tắc bảo vệ artifact và môi trường cục bộ.
```

## Lưu ý

Các staged deletions của CSV/checkpoint/cache chỉ gỡ chúng khỏi repository; không chủ động xóa dữ liệu local. Sau commit, nhánh nguồn sẵn sàng để merge trên Windows.