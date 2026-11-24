# 02 — Authoring (biên soạn)

- `current/`: source Markdown + bundle LaTeX của snapshot đang biên soạn. LaTeX bundle phải tự chứa để upload Overleaf.
- `snapshots/<ID>/`: bản Markdown/LaTeX lịch sử được giữ nguyên để truy nguyên evidence; không ghi đè bằng snapshot mới.
- Không tự động chuyển `.md` thành `.tex` nếu chưa có công cụ/quy tắc chuyển đổi được nhóm chấp thuận; hai bản là hai source song song phải review đồng bộ.
- Chỉ kiểm tra văn bản tĩnh trên local. Upload/compile PDF trên Overleaf theo `AGENT_THESIS_RULES.md`.