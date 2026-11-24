"""Kết xuất bản thảo thesis và đóng gói an toàn cho Overleaf."""

import argparse
import copy
import re
import shutil
import subprocess
import zipfile
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError as error:  # pragma: no cover - nhánh phụ thuộc môi trường
    raise ImportError("Cần cài Jinja2 đã được nhóm phê duyệt để dựng thesis bundle.") from error

# Cho phép chạy CLI trực tiếp từ scripts/ hoặc import như module trong pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate.latex_exporter import (
    ThesisArtifactExporter,
    sanitize_latex_unicode,
    unsupported_latex_unicode,
)

_TEX_LITERALS = {
    r"\documentclass", r"\usepackage", r"\begin", r"\end", r"\input", r"\includegraphics",
    r"\chapter", r"\section", r"\subsection", r"\caption", r"\label", r"\ref", r"\cite",
    r"\centering", r"\textbf", r"\textit", r"\texttt", r"\noindent", r"\tableofcontents",
    r"\listoffigures", r"\listoftables", r"\bibliography", r"\bibliographystyle", r"\appendix",
    r"\pagenumbering", r"\setcounter", r"\graphicspath", r"\toprule", r"\midrule", r"\bottomrule",
    r"\endhead", r"\endfirsthead", r"\endfoot", r"\endlastfoot", r"\multicolumn", r"\todofigure",
    r"\calloutbox", r"\linewidth", r"\textwidth", r"\frac", r"\sum", r"\max", r"\operatorname",
    r"\mathbb", r"\mathcal", r"\mathrm", r"\alpha", r"\gamma", r"\times", r"\in", r"\partial",
    r"\left", r"\right", r"\log", r"\Phi", r"\begin{", r"\end{",
}


def _slashify_tex_literals(text: str) -> str:
    """Chuẩn hóa chuỗi TeX bị escape hai lần trong nguồn Python/template.

    Chỉ giảm cặp backslash trước danh sách lệnh TeX đã biết; giữ nguyên escape
    văn bản như ``\\_`` và delimiter Jinja còn sót để Quality Gate có thể phát hiện.
    """
    for literal in sorted(_TEX_LITERALS, key=len, reverse=True):
        text = text.replace("\\\\" + literal[1:], literal)
    return text


class OverleafBundleBuilder:
    """Render template, thu thập hình và kiểm tra gói Overleaf trước khi nén."""

    ALLOWED_UNDERSCORE_COMMANDS = {
        "code", "input", "includegraphics", "label", "ref", "pageref", "cite", "citet", "citep", "url",
        "text", "mathrm", "operatorname", "texttt", "ensuremath", "emph", "mathit", "mathbf",
    }
    TEMPLATE_CONTEXT: Dict[str, object] = {
        "instructor": "[Bổ sung tên giảng viên]",
        "member_cover": "Thành viên 1, Thành viên 2, Thành viên 3, Thành viên 4, Thành viên 5",
        "members": [
            {"name": "Thành viên 1", "role": "Nhóm trưởng; hạ tầng và điều phối", "contribution": 20},
            {"name": "Thành viên 2", "role": "Baseline và Custom CNN (WP2)", "contribution": 20},
            {"name": "Thành viên 3", "role": "Transfer Learning và SOTA (WP3)", "contribution": 20},
            {"name": "Thành viên 4", "role": "Ablation, trực quan hóa và phân tích lỗi (WP4/WP5)", "contribution": 20},
            {"name": "Thành viên 5", "role": "Demo, tài liệu và báo cáo (WP5/WP6)", "contribution": 20},
        ],
        "label_rows": (
            r"$0 = \text{Angry}$ & Giận dữ " + chr(92) * 2 + "\n"
            r"$1 = \text{Disgust}$ & Ghê tởm " + chr(92) * 2 + "\n"
            r"$2 = \text{Fear}$ & Sợ hãi " + chr(92) * 2 + "\n"
            r"$3 = \text{Happy}$ & Vui vẻ " + chr(92) * 2 + "\n"
            r"$4 = \text{Sad}$ & Buồn " + chr(92) * 2 + "\n"
            r"$5 = \text{Surprise}$ & Ngạc nhiên " + chr(92) * 2 + "\n"
            r"$6 = \text{Neutral}$ & Trung tính " + chr(92) * 2 + "\n"
        ),
        "rq1_finding": "MediumCNN cải thiện rõ rệt so với SmallCNN (Private Acc tăng từ 51,55% lên 63,56%); LargeCNN chững lại ở 64,95% và overfitting gap nới rộng.",
        "rq2_finding": "Trọng số tiền huấn luyện ImageNet và ảnh 128x128 giúp VGG16 đạt 71,02% Private Acc; EfficientNet-B0 đạt hiệu quả chi phí cao nhất trên CPU.",
        "rq3_finding": "Augmentation basic tăng Private Acc từ 68,15% lên 70,13%; hàm mất mát CrossEntropy và ReduceLROnPlateau cho kết quả ổn định nhất.",
        "rq4_finding": "Đã xây dựng pipeline Grad-CAM và demo app; các hình trực quan hóa thực tế đánh dấu TODO chờ checkpoint từ cloud GPU.",
    }

    def __init__(self, project_root: Path) -> None:
        """Khởi tạo các thư mục của dự án, không sửa source evidence."""
        self.project_root = Path(project_root).resolve()
        self.thesis_dir = self.project_root / "docs" / "thesis"
        self.template_dir = self.thesis_dir / "templates"
        self.generated_dir = self.thesis_dir / "generated"
        self.bundle_dir = self.thesis_dir / "overleaf_bundle"
        self.zip_path = self.thesis_dir / "BTL02_Overleaf_Bundle.zip"
        self.exporter = ThesisArtifactExporter(
            self.project_root / "outputs" / "logs", self.generated_dir,
            self.project_root / "outputs" / "figures", self.project_root / "docs" / "experiences",
        )
        self.template_context = copy.deepcopy(self.TEMPLATE_CONTEXT)

    def _environment(self) -> Environment:
        """Tạo Jinja environment dùng delimiter tránh va chạm với TeX."""
        return Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            undefined=StrictUndefined,
            variable_start_string=r"\VAR{", variable_end_string="}",
            block_start_string=r"\BLOCK{", block_end_string="}",
            comment_start_string=r"\#", comment_end_string="#",
            autoescape=False, keep_trailing_newline=True,
        )

    def render_templates(self) -> List[Path]:
        """Render toàn bộ `.tex.j2` ra bundle và chép stylesheet/BibTeX."""
        self._contribution_total()
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        environment = self._environment()
        rendered: List[Path] = []
        for template_path in sorted(self.template_dir.glob("*.tex.j2")):
            template = environment.get_template(template_path.name)
            content = _slashify_tex_literals(sanitize_latex_unicode(template.render(**self.template_context)))
            destination = self.bundle_dir / template_path.name[:-3]
            destination.write_text(content, encoding="utf-8")
            rendered.append(destination)
        for filename in ("btl02-style.sty", "references.bib"):
            source = self.template_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"Thiếu thesis asset: {source}")
            destination = self.bundle_dir / filename
            shutil.copy2(source, destination)
            if filename.endswith(".sty"):
                destination.write_text(_slashify_tex_literals(sanitize_latex_unicode(destination.read_text(encoding="utf-8"))),
                                        encoding="utf-8")
            rendered.append(destination)
        # Bảng/đồ thị generated được exporter sở hữu; bundle chỉ nhận bản sao.
        for relative_root in (Path("tables"), Path("plots")):
            source_root = self.generated_dir / relative_root
            if not source_root.is_dir():
                continue
            for source in source_root.glob("*.tex"):
                destination = self.bundle_dir / "generated" / relative_root / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                destination.write_text(_slashify_tex_literals(sanitize_latex_unicode(destination.read_text(encoding="utf-8"))),
                                       encoding="utf-8")
                rendered.append(destination)
        return rendered

    def collect_figures(self) -> List[Path]:
        """Chỉ sao chép PNG hiện hữu từ outputs/figures và docs/presentation."""
        figure_dir = self.bundle_dir / "figures"
        copied: List[Path] = []
        for source_root in (self.project_root / "outputs" / "figures",
                            self.project_root / "docs" / "presentation"):
            if not source_root.is_dir():
                continue
            for source in source_root.rglob("*.png"):
                if not source.is_file():
                    continue
                destination = figure_dir / source.name
                figure_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied.append(destination)
        return copied

    @staticmethod
    def _strip_comments(text: str) -> str:
        r"""Bỏ comment TeX nhưng giữ ký tự escaped `\%`."""
        lines = []
        for line in text.splitlines():
            output = []
            index = 0
            while index < len(line):
                if line[index] == "%":
                    slashes = 0
                    cursor = index - 1
                    while cursor >= 0 and line[cursor] == "\\":
                        slashes += 1
                        cursor -= 1
                    if slashes % 2 == 0:
                        break
                output.append(line[index])
                index += 1
            lines.append("".join(output))
        return "\n".join(lines)

    @staticmethod
    def _brace_errors(text: str) -> List[str]:
        """Phát hiện ngoặc nhọn mất cân bằng, bỏ qua escaped brace/comment."""
        balance = 0
        errors: List[str] = []
        for line_number, line in enumerate(OverleafBundleBuilder._strip_comments(text).splitlines(), start=1):
            index = 0
            while index < len(line):
                char = line[index]
                if char == "\\":
                    index += 2
                    continue
                if char == "{":
                    balance += 1
                elif char == "}":
                    balance -= 1
                    if balance < 0:
                        errors.append(f"Ngoặc đóng thừa tại dòng {line_number}.")
                        balance = 0
                index += 1
        if balance:
            errors.append(f"Còn {balance} ngoặc mở chưa đóng.")
        return errors

    @classmethod
    def _bare_underscore_errors(cls, text: str) -> List[str]:
        """Phát hiện underscore văn xuôi trần; bỏ qua toán và đối số macro TeX."""
        source = cls._strip_comments(text)
        errors: List[str] = []
        math_mode = False
        math_environment = False
        safe_commands = cls.ALLOWED_UNDERSCORE_COMMANDS | {
            "text", "mathrm", "operatorname", "texttt", "ensuremath", "emph", "mathit", "mathbf",
        }

        def skip_braces(line: str, position: int) -> int:
            """Bỏ qua đối số ngoặc nhọn cân bằng của macro."""
            if position >= len(line) or line[position] != "{":
                return position
            depth = 1
            position += 1
            while position < len(line) and depth:
                if line[position] == "\\":
                    position += 2
                    continue
                if line[position] == "{":
                    depth += 1
                elif line[position] == "}":
                    depth -= 1
                position += 1
            return position

        for line_number, line in enumerate(source.splitlines(), start=1):
            if re.search(r"\\begin\{(?:equation|align|gather|multline|tikzpicture|axis)\}", line):
                math_environment = True
            if math_environment:
                if re.search(r"\\end\{(?:equation|align|gather|multline|tikzpicture|axis)\}", line):
                    math_environment = False
                continue
            if "$" in line or r"\(" in line or r"\[" in line:
                continue
            index = 0
            while index < len(line):
                if line[index] != "\\":
                    if line[index] == "_":
                        errors.append(f"Underscore chưa escape tại dòng {line_number}.")
                    index += 1
                    continue
                if index + 1 < len(line) and line[index + 1] in "_%&#{}$":
                    index += 2
                    continue
                command = re.match(r"\\([A-Za-z]+)", line[index:])
                if not command:
                    index += 2
                    continue
                name = command.group(1)
                index += len(command.group(0))
                if name not in safe_commands:
                    continue
                while index < len(line) and line[index].isspace():
                    index += 1
                if name == "includegraphics" and index < len(line) and line[index] == "[":
                    close = line.find("]", index + 1)
                    index = close + 1 if close >= 0 else index
                    while index < len(line) and line[index].isspace():
                        index += 1
                index = skip_braces(line, index)
        return errors

    def _replace_missing_figures(self, tex_path: Path) -> None:
        """Thay includegraphics thiếu bằng todofigure để build không lỗi ảnh."""
        text = tex_path.read_text(encoding="utf-8")
        pattern = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^{}]+)\}")

        def replace(match: re.Match) -> str:
            requested = match.group(1)
            candidates = [self.bundle_dir / requested, self.bundle_dir / "figures" / Path(requested).name]
            if any(path.is_file() for path in candidates):
                return match.group(0)
            return r"\todofigure{" + ThesisArtifactExporter.escape_latex(
                f"Thiếu ảnh: {requested}"
            ) + "}"

        updated = pattern.sub(replace, text)
        if updated != text:
            tex_path.write_text(updated, encoding="utf-8")

    def _contribution_total(self) -> float:
        """Trích tỷ lệ của năm thành viên từ bảng phân công trong README."""
        readme = self.project_root / "README.md"
        if not readme.is_file():
            raise FileNotFoundError("README.md thiếu bảng phân công thành viên.")
        text = readme.read_text(encoding="utf-8")
        values = re.findall(r"\|\s*\*\*([1-5])\*\*\s*\|.*?\|\s*\*\*([0-9]+(?:\.[0-9]+)?)%\*\*\s*\|", text)
        by_member = {int(index): float(value) for index, value in values}
        if set(by_member) != {1, 2, 3, 4, 5}:
            raise ValueError("Không trích xuất đủ tỷ lệ đóng góp 5 thành viên từ README.md.")
        for index, member in enumerate(self.template_context["members"], start=1):
            member["contribution"] = by_member[index]
        total = sum(by_member.values())
        if not math.isfinite(total):
            raise ValueError("Tỷ lệ đóng góp không hữu hạn.")
        self.template_context["contribution_total"] = total
        return total

    def quality_gate(self) -> Dict[str, object]:
        """Kiểm tra cú pháp cơ bản, underscore, ảnh và tổng contribution."""
        tex_files = sorted(self.bundle_dir.rglob("*.tex"))
        if not tex_files:
            raise RuntimeError("Bundle không có file TeX để kiểm tra.")
        for path in tex_files:
            self._replace_missing_figures(path)
        errors: List[str] = []
        missing_figures: List[str] = []
        for path in tex_files:
            text = path.read_text(encoding="utf-8")
            unsupported = unsupported_latex_unicode(text)
            if unsupported:
                preview = ", ".join(f"U+{ord(character):04X}" for _, character in unsupported[:12])
                errors.append(f"{path.relative_to(self.bundle_dir)}: Unicode chưa sanitize ({preview}).")
            errors.extend(f"{path.relative_to(self.bundle_dir)}: {item}" for item in self._brace_errors(text))
            errors.extend(f"{path.relative_to(self.bundle_dir)}: {item}" for item in self._bare_underscore_errors(text))
            for image in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^{}]+)\}", text):
                candidate = self.bundle_dir / image
                if not candidate.is_file():
                    candidate = self.bundle_dir / "figures" / Path(image).name
                if not candidate.is_file():
                    missing_figures.append(f"{path.name}: {image}")
        # Quét cả nguồn template sau đúng phép sanitize mà renderer áp dụng.
        # Nhờ vậy ký hiệu hợp lệ như ×/→ được chuyển sang macro trước kiểm tra;
        # chữ Việt NFC vẫn được phép giữ nguyên cho T5/pdfLaTeX.
        for template_path in sorted(self.template_dir.glob("*.tex.j2")):
            sanitized_source = sanitize_latex_unicode(template_path.read_text(encoding="utf-8"))
            unsupported = unsupported_latex_unicode(sanitized_source)
            if unsupported:
                errors.append(f"{template_path.name}: còn Unicode ngoài ASCII/tiếng Việt sau sanitize.")
        contribution = self._contribution_total()
        if abs(contribution - 100.0) > 1e-8:
            errors.append(f"Tỷ lệ đóng góp bằng {contribution:g}%, yêu cầu 100%.")
        if errors or missing_figures:
            raise RuntimeError("Overleaf Quality Gate thất bại: " + "; ".join(errors + missing_figures))
        return {"tex_files": len(tex_files), "syntax_errors": len(errors),
                "missing_figures": len(missing_figures), "contribution_total": contribution}

    def _archive_paths(self) -> List[Path]:
        """Chọn files bundle, loại trừ samples/chính sách và cache."""
        excluded = {"samples", "AGENTS_THESIS.md"}
        return [path for path in sorted(self.bundle_dir.rglob("*")) if path.is_file()
                and not any(part in excluded for part in path.relative_to(self.bundle_dir).parts)
                and path.suffix.lower() not in {".aux", ".log", ".out", ".toc", ".bbl", ".blg"}]

    def create_zip(self) -> Path:
        """Nén danh sách file đã nghiệm thu, không đưa mẫu tham khảo vào archive."""
        self.zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in self._archive_paths():
                archive.write(path, path.relative_to(self.bundle_dir).as_posix())
        return self.zip_path

    def compile_pdf(self, engine: str = "pdflatex") -> Path:
        """Tùy chọn biên dịch hai lượt; build output chỉ nằm trong thư mục ignored."""
        if engine not in {"pdflatex", "xelatex"}:
            raise ValueError("Engine hỗ trợ: pdflatex hoặc xelatex.")
        build_dir = self.thesis_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        command = [engine, "-disable-installer", "-interaction=nonstopmode", "-halt-on-error",
                   f"-output-directory={build_dir}", "main.tex"]
        for _ in range(2):
            try:
                result = subprocess.run(
                    command, cwd=self.bundle_dir, capture_output=True, text=True, timeout=12
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"{engine} vượt thời gian build 12 giây; kiểm tra package/cài đặt TeX của môi trường."
                ) from error
            if result.returncode:
                raise RuntimeError(f"{engine} biên dịch thất bại:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}")
        return build_dir / "main.pdf"

    def build(self, compile_engine: Optional[str] = None) -> Dict[str, object]:
        """Chạy exporter, render template, thu ảnh, quality gate và tạo ZIP."""
        generated = self.exporter.export_all()
        rendered = self.render_templates()
        figures = self.collect_figures()
        gate = self.quality_gate()
        pdf = self.compile_pdf(compile_engine) if compile_engine else None
        archive = self.create_zip()
        return {"generated": generated, "rendered": rendered, "figures": figures,
                "quality_gate": gate, "pdf": pdf, "zip": archive}

    @classmethod
    def cli(cls, argv: Optional[Sequence[str]] = None) -> int:
        """Điểm vào CLI; mặc định chỉ quality gate và tạo bundle, không compile PDF."""
        parser = argparse.ArgumentParser(description="Dựng bộ thesis LaTeX cho Overleaf.")
        parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
        parser.add_argument("--compile", choices=("pdflatex", "xelatex"), default=None)
        args = parser.parse_args(argv)
        result = cls(args.project_root).build(args.compile)
        print(f"Quality Gate: {result['quality_gate']}")
        print(f"Overleaf bundle: {result['zip']}")
        if result["pdf"]:
            print(f"PDF: {result['pdf']}")
        return 0


if __name__ == "__main__":
    raise SystemExit(OverleafBundleBuilder.cli())