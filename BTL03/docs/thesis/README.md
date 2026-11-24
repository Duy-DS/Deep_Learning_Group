# Hệ thống báo cáo tổng kết BTL03 — tổ chức 3 tầng

`docs/thesis/` quản lý vòng đời báo cáo theo ba tầng: **Nguyên liệu → Biên soạn → Thành phẩm**. Markdown và LaTeX là hai dạng đồng bộ của cùng snapshot. PDF chỉ được coi là release sau khi biên dịch và được nhóm duyệt trên Overleaf.

## Cây thư mục hiện hành

```text
docs/thesis/
├── AGENT_THESIS_RULES.md
├── README.md
├── 01_materials/                         # Nguyên liệu
│   ├── samples/                          # sample_1.tex, sample_2.tex, analysis
│   ├── templates/                        # legacy và sample_style_v2
│   └── evidence/                         # manifest và outline
├── 02_authoring/                         # Biên soạn
│   ├── current/
│   │   ├── report.md
│   │   └── latex/                        # bundle tự chứa cho Overleaf
│   └── snapshots/
│       └── T0212/                        # snapshot lịch sử, không ghi đè
└── 03_releases/                          # Chỉ PDF đã duyệt + release notes
    └── README.md
```

Không tạo `guidelines/` hoặc `styles/` rỗng. Nếu sau này có tài liệu/quy định hoặc cấu hình phong cách thực tế, có thể thêm thành thư mục con của `01_materials/` và cập nhật README.

## Trách nhiệm từng tầng

1. **`01_materials/` — Nguyên liệu:** chứa sample gốc, phân tích phong cách, template và manifest/outline evidence. Sample gốc và template `[READ-ONLY]` phải giữ nguyên; không nhân bản dataset/checkpoint hoặc PII.
2. **`02_authoring/` — Biên soạn:** `current/report.md` và `current/latex/` là tài liệu đang làm. Mỗi mốc cũ được lưu riêng trong `snapshots/<SNAPSHOT>/`; không ghi đè snapshot T0212. LaTeX là source thực sự, không phải file tạm được sinh tự động từ Markdown.
3. **`03_releases/` — Thành phẩm:** chỉ lưu PDF đã biên dịch trên Overleaf và được duyệt, cùng `release_notes.md` ghi source commit/snapshot, ngày, người duyệt và giới hạn. Hiện chưa có PDF release; không tạo file giả hoặc coi bản nháp là thành phẩm.

## Quy trình cập nhật và xuất bản

1. Đọc `AGENT_THESIS_RULES.md`, material liên quan và evidence mới nhất.
2. Cập nhật Markdown và LaTeX trong `02_authoring/current/` cùng snapshot; đối chiếu số liệu, bảng, citation, trạng thái và caveat.
3. Chỉ kiểm tra văn bản tĩnh/Git diff tại local. Không cài TeX, không chạy test, training hoặc compiler local theo quy tắc thesis.
4. Nén **nội dung bên trong** `02_authoring/current/latex/` để `main.tex` ở gốc ZIP; upload lên Overleaf, dùng XeLaTeX/Biber theo bundle, compile và kiểm tra log/PDF.
5. Sau khi nhóm duyệt, lưu PDF vào `03_releases/<version>/BaoCao_TongKet_BTL03.pdf` kèm release notes. Không lưu `.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.synctex.gz` hoặc PDF build chưa duyệt.

## Snapshot và trạng thái

- `02_authoring/current/` hiện chứa bản T0402; owner trong workbook là Member 05, còn nhóm trưởng chỉ định trực tiếp Member 01 soạn snapshot này. Cần nhóm xác nhận mapping trước nghiệm thu.
- `02_authoring/snapshots/T0212/` giữ Markdown/LaTeX T0212; report nguồn ghi overall `BLOCKED`, không được hồi tố kết quả mới vào đó.
- `03_releases/` hiện chưa có PDF được duyệt.
- Họ tên pháp lý/MSSV/giảng viên/tỷ lệ đóng góp cuối kỳ cần nhóm xác nhận; không suy từ alias.

## Đường dẫn chính

- Current Markdown: `02_authoring/current/report.md`.
- Current Overleaf bundle: `02_authoring/current/latex/main.tex`.
- Snapshot T0212 Markdown/LaTeX: `02_authoring/snapshots/T0212/report.md` và `02_authoring/snapshots/T0212/latex/main.tex`.
- Samples/style analysis: `01_materials/samples/sample_1.tex`, `sample_2.tex`, `style_analysis.md`.
- Templates: `01_materials/templates/TEMPLATE_THESIS.md`, `latex_template/`, `sample_style_v2/`.
- Evidence manifest/outline: `01_materials/evidence/evidence_sources_manifest.md`, `thesis_outline_and_scope.md`.
- Releases: `03_releases/<version>/BaoCao_TongKet_BTL03.pdf` chỉ sau khi duyệt.

Bundle Overleaf phải tự chứa; không để `main.tex` hoặc chương dùng `\input`/asset bên ngoài bundle qua `../`. README và Markdown sibling không cần nén vào LaTeX ZIP.