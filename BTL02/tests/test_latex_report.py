"""Kiểm thử exporter LaTeX và đóng gói Overleaf bằng dự án tạm."""

import csv
import shutil
import zipfile
from pathlib import Path

import pytest

from scripts.build_thesis_bundle import OverleafBundleBuilder
from src.evaluate.latex_exporter import ThesisArtifactExporter, sanitize_latex_unicode, unsupported_latex_unicode


class ThesisPipelineFixture:
    """Tạo dự án tối thiểu trong tmp_path, không đụng evidence thật."""

    SUMMARY_COLUMNS = ThesisArtifactExporter.SUMMARY_COLUMNS

    def __init__(self, root: Path):
        self.root = root

    def create(self, with_csv: bool = True) -> Path:
        """Dựng layout và sao chép templates read-only vào vùng test tạm."""
        thesis = self.root / "docs" / "thesis"
        templates = thesis / "templates"
        templates.mkdir(parents=True)
        source_templates = Path(__file__).resolve().parents[1] / "docs" / "thesis" / "templates"
        for source in source_templates.iterdir():
            if source.is_file():
                shutil.copy2(source, templates / source.name)
        (self.root / "outputs" / "logs").mkdir(parents=True)
        (self.root / "outputs" / "figures").mkdir(parents=True)
        (self.root / "docs" / "presentation").mkdir(parents=True)
        shutil.copy2(Path(__file__).resolve().parents[1] / "README.md", self.root / "README.md")
        if with_csv:
            self.write_csv(self.root / "outputs" / "logs" / "fixture_24col.csv")
        (thesis / "samples").mkdir()
        (thesis / "samples" / "sample_1.tex").write_text("READ ONLY FIXTURE", encoding="utf-8")
        (thesis / "AGENTS_THESIS.md").write_text("POLICY FIXTURE", encoding="utf-8")
        return thesis

    @classmethod
    def write_csv(cls, path: Path) -> None:
        """Ghi hai dòng synthetic đúng 24 cột dành riêng cho test."""
        path.parent.mkdir(parents=True, exist_ok=True)
        first = {column: "" for column in cls.SUMMARY_COLUMNS}
        first.update({
            "Run ID": "fixture_run_1", "Name": "SmallCNN", "Status": "FINISHED",
            "params.architecture": "SmallCNN", "params.rq_tag": "RQ1", "params.seed": "42",
            "params.image_size": "48", "metrics.best_epoch": "3", "metrics.train_acc": "80",
            "metrics.best_val_acc": "65", "metrics.private_test_acc": "64",
            "metrics.macro_f1": "0.60",
        })
        second = dict(first)
        second.update({"Run ID": "fixture_run_2", "Name": "LargeCNN", "params.architecture": "LargeCNN",
                       "metrics.private_test_acc": "68", "metrics.macro_f1": "0.66"})
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=cls.SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows((first, second))


class TestThesisArtifactExporter:
    """Kiểm tra bảng, placeholder và escaping LaTeX."""

    def test_exports_tables_from_24_column_csv_and_bolds_best(self, tmp_path):
        """Bảng được sinh từ CSV đúng schema và best metrics được bôi đậm."""
        ThesisPipelineFixture(tmp_path).create()
        exporter = ThesisArtifactExporter(tmp_path / "outputs/logs", tmp_path / "docs/thesis/generated")
        result = exporter.export_all()
        all_runs = result["tables"]["all_runs"].read_text(encoding="utf-8")
        assert r"\begin{longtable}" in all_runs
        assert (r"\textbf{68}" in all_runs or r"\textbf{68.00}" in all_runs)
        assert (r"\textbf{0.66}" in all_runs or r"\textbf{0.6600}" in all_runs)
        assert result["plot"].is_file()
        plot = result["plot"].read_text(encoding="utf-8")
        assert "xticklabels=" in plot and "(1,64)" in plot and "(2,68)" in plot
        assert result["tables"]["compute_cost"].is_file()

    def test_missing_csv_emits_warning_placeholder_not_fake_values(self, tmp_path):
        """Không có CSV thì xuất bảng cảnh báo hợp lệ, không tạo biểu đồ giả."""
        ThesisPipelineFixture(tmp_path).create(with_csv=False)
        exporter = ThesisArtifactExporter(tmp_path / "outputs/logs", tmp_path / "generated")
        output = exporter.export_all()
        content = output["tables"]["all_runs"].read_text(encoding="utf-8")
        assert "CSV 24 c\u1ed9t h\u1ee3p l\u1ec7" in content
        assert output["plot"].is_file()
        assert "todofigure" in output["plot"].read_text(encoding="utf-8")

    def test_latex_escape_covers_special_characters(self):
        """Ký tự đặc biệt được escape để tránh lỗi khi compile."""
        escaped = ThesisArtifactExporter.escape_latex("a_b%&$#")
        assert r"a\_b\%\&\$\#" == escaped

    def test_unicode_sanitizer_preserves_vietnamese_and_maps_symbols(self):
        """Sanitizer chuyển ký hiệu mở rộng sang macro ASCII và giữ chữ Việt NFC."""
        source = "Nhóm: tiếng Việt 98% → 100%; x≈2, ≤3, ≥1, ±, °, ², α, Σ, 🙂"
        cleaned = sanitize_latex_unicode(source)
        assert "Nhóm" in cleaned and "tiếng Việt" in cleaned
        assert r"\ensuremath{\rightarrow}" in cleaned
        assert r"\ensuremath{\approx}" in cleaned
        assert r"\ensuremath{\le}" in cleaned
        assert r"\ensuremath{\ge}" in cleaned
        assert r"\ensuremath{\pm}" in cleaned
        assert r"\textdegree{}" in cleaned
        assert r"\textsuperscript{2}" in cleaned
        assert "[U+1F642]" in cleaned
        assert unsupported_latex_unicode(cleaned) == []

    def test_unicode_sanitizer_normalizes_decomposed_vietnamese(self):
        """Dấu tiếng Việt dạng combining được chuẩn hóa NFC trước khi ghi TeX."""
        cleaned = sanitize_latex_unicode("a\u0323")
        assert cleaned == "ạ"
        assert unsupported_latex_unicode(cleaned) == []

    def test_unicode_sanitizer_maps_remaining_math_symbols_and_emoji(self):
        """Đảm bảo emoji/Unicode toán bổ sung không lọt ra TeX cuối."""
        cleaned = sanitize_latex_unicode("μ σ ρ τ ∞ ∈ ∑ ∂ √ “quoted” … 🙂")
        assert all(token in cleaned for token in (
            r"\ensuremath{\mu}", r"\ensuremath{\sigma}", r"\ensuremath{\rho}",
            r"\ensuremath{\tau}", r"\ensuremath{\infty}", r"\ensuremath{\in}",
            r"\ensuremath{\sum}", r"\ensuremath{\partial}", r"\ensuremath{\sqrt{\vphantom{x}}}",
            "``quoted''", r"\ldots{}", "[U+1F642]",
        ))
        assert unsupported_latex_unicode(cleaned) == []

    def test_comparison_figures_are_grouped_by_research_question(self, tmp_path):
        """Hình PNG chỉ dùng CSV run duy nhất và tag RQ có nghĩa tương ứng."""
        ThesisPipelineFixture(tmp_path).create()
        exporter = ThesisArtifactExporter(tmp_path / "outputs/logs", tmp_path / "generated",
                                         tmp_path / "outputs/figures")
        figures = exporter.export_comparison_figures()
        assert set(figures) == {"t0602_rq1_metrics"}
        assert figures["t0602_rq1_metrics"].is_file()

    def test_vector_plot_uses_only_three_rq1_depth_runs(self, tmp_path):
        """PGFPlots RQ1 không gộp các activation/optimizer ablation."""
        ThesisPipelineFixture(tmp_path).create()
        exporter = ThesisArtifactExporter(tmp_path / "outputs/logs", tmp_path / "generated")
        plot = exporter.export_vector_plot().read_text(encoding="utf-8")
        assert "\\n" not in plot
        assert "SmallCNN" in plot
        assert "MediumCNN" not in plot  # Fixture có hai cấu hình, không phải dữ liệu FER2013 đầy đủ.
        assert "LargeCNN" in plot
        assert "LargeCNN_relu" not in plot
        assert "xticklabels=" in plot

    def test_benchmark_rows_are_read_from_approved_experience_when_csv_absent(self, tmp_path):
        """Chỉ fallback sang bảng benchmark Markdown khi CSV benchmark không có."""
        fixture = ThesisPipelineFixture(tmp_path)
        fixture.create(with_csv=False)
        evidence = tmp_path / "docs/experiences"
        evidence.mkdir(parents=True)
        (evidence / "EXP_COST.md").write_text(
            "| VGG16 | 128 | 1000 | 2000 | 1.5 | 1.6 | 2.5 | 4000000 | OK |\n",
            encoding="utf-8",
        )
        exporter = ThesisArtifactExporter(tmp_path / "outputs/logs", tmp_path / "generated", evidence_dir=evidence)
        table = exporter.export_compute_table().read_text(encoding="utf-8")
        assert "VGG16" in table and ("2.5" in table or "2.50" in table) and ("4000000" in table or "4,000,000" in table)

    def test_full_t0305_experience_table_rows_are_parsed(self, tmp_path):
        """Parser chấp nhận đúng schema 9 cột có trạng thái CPU trong T0305."""
        evidence = tmp_path / "docs/experiences"
        evidence.mkdir(parents=True)
        (evidence / "EXP04_EXP05_EXP06_COST.md").write_text(
            "| VGG16 | 128 | 5,023,993,344 | 10,047,986,688 | 110.255 | 110.266 | 87.148 | 8.828 | Đo hoàn tất trên CPU |\n",
            encoding="utf-8",
        )
        exporter = ThesisArtifactExporter(tmp_path / "logs", tmp_path / "generated", evidence_dir=evidence)
        rows = exporter._benchmark_rows_from_experience()
        assert len(rows) == 1
        assert rows[0]["macs"] == "5023993344"
        assert rows[0]["throughput_images_per_sec"] == "8.828"


class TestOverleafBundleBuilder:
    """Kiểm tra render template, quality gate và nội dung ZIP."""

    def test_build_bundle_and_exclude_samples_and_policy(self, tmp_path):
        """Bundle render đủ file, đạt 0 lỗi và không kèm reference/policy."""
        ThesisPipelineFixture(tmp_path).create()
        result = OverleafBundleBuilder(tmp_path).build()
        assert result["quality_gate"] == {
            "tex_files": 14, "syntax_errors": 0, "missing_figures": 0, "contribution_total": 100.0
        }
        assert result["zip"].is_file()
        with zipfile.ZipFile(result["zip"]) as archive:
            names = archive.namelist()
            assert "main.tex" in names
            assert not any("samples/" in name for name in names)
            assert not any("AGENTS_THESIS.md" in name for name in names)
        assert result["quality_gate"]["syntax_errors"] == 0
        assert (tmp_path / "docs/thesis/samples/sample_1.tex").read_text(encoding="utf-8") == "READ ONLY FIXTURE"
        rendered_main = (tmp_path / "docs/thesis/overleaf_bundle/main.tex").read_text(encoding="utf-8")
        assert "\\\\documentclass" not in rendered_main
        assert "\\documentclass" in rendered_main
        rendered_tex = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "docs/thesis/overleaf_bundle").glob("*.tex"))
        cited = {key.strip() for group in __import__("re").findall(r"\\cite[pt]?\{([^}]+)\}", rendered_tex)
                 for key in group.split(",")}
        bib = (tmp_path / "docs/thesis/overleaf_bundle/references.bib").read_text(encoding="utf-8")
        defined = set(__import__("re").findall(r"@\w+\s*\{\s*([^,]+),", bib))
        assert cited <= defined

    def test_quality_gate_replaces_missing_figure_and_checks_underscores(self, tmp_path):
        """Hình thiếu chuyển thành TODO; underscore văn bản và brace lỗi bị phát hiện."""
        ThesisPipelineFixture(tmp_path).create(with_csv=False)
        builder = OverleafBundleBuilder(tmp_path)
        builder.exporter.export_all()
        builder.render_templates()
        main = builder.bundle_dir / "main.tex"
        text = main.read_text(encoding="utf-8").replace("\\input{frontmatter.tex}",
                                                         "\\includegraphics{absent_face.png}\n\\input{frontmatter.tex}")
        main.write_text(text, encoding="utf-8")
        assert builder.quality_gate()["missing_figures"] == 0
        assert "todofigure" in main.read_text(encoding="utf-8")
        assert OverleafBundleBuilder._bare_underscore_errors("plain_name")
        assert OverleafBundleBuilder._brace_errors("\\textbf{broken")

    def test_quality_gate_rejects_unsanitized_unicode_in_rendered_tex(self, tmp_path):
        """TeX Unicode mở rộng bị Quality Gate từ chối nếu bypass renderer."""
        ThesisPipelineFixture(tmp_path).create(with_csv=False)
        builder = OverleafBundleBuilder(tmp_path)
        builder.exporter.export_all()
        builder.render_templates()
        path = builder.bundle_dir / "main.tex"
        path.write_text(path.read_text(encoding="utf-8") + "\nUnicode ngoài bảng: 🙂\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Unicode chưa sanitize"):
            builder.quality_gate()

    def test_contribution_gate_uses_readme_distribution(self, tmp_path):
        """Gate đọc phân công thật trong README, không tự lấy tổng từ kỳ vọng."""
        ThesisPipelineFixture(tmp_path).create(with_csv=False)
        builder = OverleafBundleBuilder(tmp_path)
        readme = tmp_path / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8").replace("**20.0%**", "**19.0%**", 1), encoding="utf-8")
        builder.exporter.export_all()
        builder.render_templates()
        with pytest.raises(RuntimeError, match="99%"):
            builder.quality_gate()

    def test_rendered_report_carries_no_banned_ai_cliches(self, tmp_path):
        """Kiểm tra văn phong chống sáo rỗng trên các chương đã render."""
        ThesisPipelineFixture(tmp_path).create()
        builder = OverleafBundleBuilder(tmp_path)
        builder.build()
        banned = (
            "Trong kỷ nguyên số hóa", "Không thể phủ nhận rằng", "Một cách toàn diện và sâu sắc",
            "Bức tranh toàn cảnh", "Không chỉ dừng lại ở đó", "đột phá ngoạn mục", "hoàn hảo tuyệt đối",
        )
        for path in builder.bundle_dir.glob("chap*.tex"):
            content = path.read_text(encoding="utf-8")
            assert not any(phrase in content for phrase in banned)

    def test_rendered_tex_contains_no_replacement_or_missing_ref_marks(self, tmp_path):
        """Kiểm tra các file .tex đã render không chứa ký tự lỗi \\ufffd hoặc cụm '??'."""
        ThesisPipelineFixture(tmp_path).create()
        builder = OverleafBundleBuilder(tmp_path)
        builder.build()
        for path in builder.bundle_dir.rglob("*.tex"):
            content = path.read_text(encoding="utf-8")
            assert "\ufffd" not in content, f"Phát hiện ký tự lỗi \\ufffd trong {path.name}"
            assert "??" not in content, f"Phát hiện cụm '??' trong {path.name}"