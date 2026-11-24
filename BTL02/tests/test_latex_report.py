"""Kiểm thử exporter LaTeX và đóng gói Overleaf bằng dự án tạm."""

import csv
import shutil
import zipfile
from pathlib import Path

import pytest

from scripts.build_thesis_bundle import OverleafBundleBuilder
from src.evaluate.latex_exporter import ThesisArtifactExporter


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
        assert r"\textbf{68}" in all_runs
        assert r"\textbf{0.66}" in all_runs
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
        assert (tmp_path / "docs/thesis/samples/sample_1.tex").read_text(encoding="utf-8") == "READ ONLY FIXTURE"

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