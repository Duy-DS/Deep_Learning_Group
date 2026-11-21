"""Tải FER2013 qua kagglehub, giải nén an toàn và xác thực trước khi cài đặt."""

import argparse
import csv
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Optional, Sequence

import kagglehub
import dotenv


class FER2013Downloader:
    """Điều phối cache, chứng thực, tải và kiểm tra toàn vẹn dữ liệu FER2013."""

    COMPETITION = "challenges-in-representation-learning-facial-expression-recognition-challenge"
    COMPETITION_URL = f"https://www.kaggle.com/c/{COMPETITION}/data"
    REQUIRED_COLUMNS = ("emotion", "pixels", "Usage")
    EXPECTED_SPLITS = {"Training": 28709, "PublicTest": 3589, "PrivateTest": 3589}
    OUTPUT_NAME = "fer2013.csv"
    MAX_MEMBER_BYTES = 1_000_000_000
    MAX_DEPTH = 5

    def __init__(self, output_dir: Path, force: bool = False, env_file: Optional[Path] = None) -> None:
        """Khởi tạo thư mục đích và chính sách tải lại.

        Args:
            output_dir: Thư mục chứa CSV sau khi xác thực.
            force: Tải lại dù CSV cục bộ đang hợp lệ.
            env_file: Tệp cấu hình tường minh, hữu ích cho kiểm thử biệt lập.
        """
        self.load_environment(env_file)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_path = self.output_dir / self.OUTPUT_NAME
        self.force = force

    @classmethod
    def load_environment(cls, env_file: Optional[Path] = None) -> None:
        """Nạp .env gốc dự án hoặc CWD, không ghi đè môi trường và không log secrets.

        Args:
            env_file: Nếu có, chỉ xét tệp này, không tìm cấu hình thật khác.
        """
        candidates = ([Path(env_file)] if env_file is not None else
                      [Path(__file__).resolve().parents[2] / ".env", Path.cwd() / ".env"])
        for candidate in candidates:
            if candidate.is_file():
                try:
                    # Không nội suy biến; token phải được giữ nguyên từng ký tự.
                    dotenv.load_dotenv(candidate, override=False, interpolate=False, encoding="utf-8-sig")
                except (OSError, UnicodeError, ValueError):
                    raise RuntimeError("Không thể nạp .env; kiểm tra định dạng và quyền đọc.") from None
                return

    @classmethod
    def from_project_root(cls, force: bool = False, output_dir: Optional[Path] = None):
        """Tính thư mục raw mặc định theo module, không phụ thuộc working directory."""
        root = Path(__file__).resolve().parents[2]
        return cls(output_dir if output_dir is not None else root / "data" / "raw", force)

    @staticmethod
    def authentication_source() -> Optional[str]:
        """Nhận diện nguồn chứng thực theo ưu tiên, không trả về giá trị bí mật."""
        token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
        if token and token not in ("your_kaggle_api_token_here", "<YOUR_TOKEN>"):
            return "KAGGLE_API_TOKEN"
        home_config = Path.home() / ".kaggle"
        access_token = home_config / "access_token"
        if access_token.is_file() and access_token.stat().st_size > 0:
            return "access_token"
        if os.environ.get("KAGGLE_USERNAME", "").strip() and os.environ.get("KAGGLE_KEY", "").strip():
            return "legacy_environment"
        config = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(home_config))) / "kaggle.json"
        if config.is_file():
            try:
                credentials = json.loads(config.read_text(encoding="utf-8"))
                if isinstance(credentials, dict) and credentials.get("username") and credentials.get("key"):
                    return "kaggle.json"
            except (OSError, ValueError):
                # Không in dữ liệu trong tệp cấu hình lỗi ra nhật ký.
                return None
        return None

    @classmethod
    def print_authentication_help(cls) -> None:
        """Hướng dẫn token mới và bước chấp nhận điều lệ; không chứa token thật."""
        print(
            "Chưa tìm thấy chứng thực Kaggle. Tạo API Access Token trong Kaggle Settings.\n"
            "Copy .env.example thành .env trong BTL02 và điền KAGGLE_API_TOKEN=<YOUR_TOKEN>.\n"
            "Bash: export KAGGLE_API_TOKEN=<YOUR_TOKEN>\n"
            "PowerShell: $env:KAGGLE_API_TOKEN='<YOUR_TOKEN>'\n"
            "Hoặc lưu token tại ~/.kaggle/access_token.\n"
            "Vẫn hỗ trợ KAGGLE_USERNAME/KAGGLE_KEY hoặc ~/.kaggle/kaggle.json.\n"
            f"Đăng nhập và nhấn Accept Rules tại: {cls.COMPETITION_URL}"
        )

    @classmethod
    def _copy_stream(cls, stream: BinaryIO, destination: Path) -> None:
        """Sao chép theo khối, giới hạn kích thước từng thành viên archive."""
        written = 0
        with destination.open("wb") as output:
            while chunk := stream.read(1024 * 1024):
                written += len(chunk)
                if written > cls.MAX_MEMBER_BYTES:
                    raise ValueError("Thành viên archive vượt giới hạn 1 GB.")
                output.write(chunk)

    @staticmethod
    def _basename(name: str) -> str:
        """Bỏ đường dẫn không tin cậy trước khi xét tên thành viên archive."""
        return PurePosixPath(name.replace("\\", "/")).name.lower()

    @classmethod
    def _candidates(cls, source: Path, name: str, workspace: Path, depth: int = 0) -> list[Path]:
        """Tìm mọi CSV ứng viên; không dùng extractall hoặc ghi đường dẫn từ archive."""
        if depth > cls.MAX_DEPTH:
            raise ValueError("Archive lồng quá sâu; từ chối giải nén.")
        basename = cls._basename(name)
        if basename == cls.OUTPUT_NAME:
            candidate = workspace / f"candidate_{len(list(workspace.glob('candidate_*')))}.csv"
            with source.open("rb") as stream:
                cls._copy_stream(stream, candidate)
            return [candidate]
        if not basename.endswith((".zip", ".tar.gz", ".tgz")):
            return []
        results = []
        if basename.endswith(".zip"):
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    child_name = cls._basename(member.filename)
                    if child_name != cls.OUTPUT_NAME and not child_name.endswith((".zip", ".tar.gz", ".tgz")):
                        continue
                    # Đường dẫn tạm do hệ thống sinh, không lấy từ archive.
                    with tempfile.TemporaryDirectory(dir=workspace) as nested:
                        child = Path(nested) / "payload"
                        with archive.open(member) as stream:
                            cls._copy_stream(stream, child)
                        results.extend(cls._candidates(child, child_name, workspace, depth + 1))
        else:
            with tarfile.open(source, "r:gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    child_name = cls._basename(member.name)
                    if child_name != cls.OUTPUT_NAME and not child_name.endswith((".zip", ".tar.gz", ".tgz")):
                        continue
                    with tempfile.TemporaryDirectory(dir=workspace) as nested:
                        child = Path(nested) / "payload"
                        with archive.extractfile(member) as stream:
                            cls._copy_stream(stream, child)
                        results.extend(cls._candidates(child, child_name, workspace, depth + 1))
        return results

    @classmethod
    def validate_csv(cls, csv_path: Path) -> dict[str, int]:
        """Kiểm tra header, nhãn, 2304 pixel/ảnh và số mẫu theo Usage.

        Args:
            csv_path: Tệp CSV cần kiểm chứng.

        Returns:
            dict[str, int]: Số mẫu từng phân tập.

        Raises:
            ValueError: Nội dung CSV không đúng đặc tả FER2013.
        """
        counts = dict.fromkeys(cls.EXPECTED_SPLITS, 0)
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or []) != cls.REQUIRED_COLUMNS:
                raise ValueError("Header CSV phải đúng thứ tự emotion, pixels, Usage.")
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Dòng {line}: sai số trường CSV.")
                if row["Usage"] not in counts:
                    raise ValueError(f"Dòng {line}: Usage không hợp lệ.")
                try:
                    label = int(row["emotion"])
                    pixels = row["pixels"].split()
                    if not 0 <= label <= 6 or len(pixels) != 2304:
                        raise ValueError
                    if any(not 0 <= int(pixel) <= 255 for pixel in pixels):
                        raise ValueError
                except ValueError:
                    raise ValueError(f"Dòng {line}: nhãn hoặc 2304 pixel [0,255] không hợp lệ.") from None
                counts[row["Usage"]] += 1
        if sum(counts.values()) != sum(cls.EXPECTED_SPLITS.values()):
            raise ValueError(f"CSV có {sum(counts.values()):,} mẫu; cần đúng 35,887.")
        if counts != cls.EXPECTED_SPLITS:
            raise ValueError("Số lượng mẫu theo Usage không đúng 28709/3589/3589.")
        return counts

    @staticmethod
    def calculate_sha256(path: Path) -> str:
        """Tính SHA-256 theo luồng, không nạp toàn bộ dữ liệu vào RAM."""
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _report_valid_file(self, path: Path) -> None:
        """Xác thực và in số mẫu cùng checksum, không in thông tin chứng thực."""
        counts = self.validate_csv(path)
        print(f"CSV hợp lệ: {path}\nTổng số mẫu: {sum(counts.values()):,}")
        for split, count in counts.items():
            print(f"  {split}: {count:,}")
        print(f"SHA-256: {self.calculate_sha256(path)}")

    def run(self, verify_only: bool = False) -> Path:
        """Kiểm tra cache hoặc tải qua kagglehub; chỉ thay CSV sau khi xác thực."""
        if self.output_path.is_file() and (verify_only or not self.force):
            try:
                self._report_valid_file(self.output_path)
                return self.output_path
            except (ValueError, UnicodeError, csv.Error):
                if verify_only:
                    raise
                print("Cache không hợp lệ; file cũ được giữ đến khi xác thực tải mới thành công.")
        if verify_only:
            raise FileNotFoundError("Không tìm thấy fer2013.csv để xác thực.")
        if self.authentication_source() is None:
            self.print_authentication_help()
            raise RuntimeError("Thiếu chứng thực Kaggle; chưa thực hiện tải.")
        try:
            cache = Path(kagglehub.competition_download(self.COMPETITION, force_download=self.force))
        except Exception:
            # Không chuyển tiếp thông báo HTTP có thể chứa token hoặc URL có chữ ký.
            raise RuntimeError("Tải Kaggle thất bại. Kiểm tra token, mạng và Accept Rules.") from None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fer2013_download_", dir=self.output_dir) as temporary:
            workspace = Path(temporary)
            sources = [cache] if cache.is_file() else sorted(path for path in cache.rglob("*") if path.is_file())
            candidates = []
            for source in sources:
                candidates.extend(self._candidates(source, source.name, workspace))
            if not candidates:
                raise ValueError("Không tìm thấy fer2013.csv trong cache Kaggle.")
            if len({self.calculate_sha256(path) for path in candidates}) != 1:
                raise ValueError("Nhiều bản fer2013.csv khác nhau trong cache; từ chối chọn tùy ý.")
            self._report_valid_file(candidates[0])
            os.replace(candidates[0], self.output_path)
        print(f"Hoàn tất: {self.output_path}")
        return self.output_path

    @classmethod
    def cli(cls, arguments: Optional[Sequence[str]] = None) -> int:
        """Phân tích CLI và trả về mã thoát an toàn, không hiển thị secrets."""
        parser = argparse.ArgumentParser(description="Tải và xác thực FER2013 bằng kagglehub.")
        parser.add_argument("--output-dir", type=Path, help="Thư mục đích chứa CSV.")
        parser.add_argument("--force", action="store_true", help="Tải lại, bỏ qua cache hợp lệ.")
        parser.add_argument("--verify-only", action="store_true", help="Chỉ xác thực; không tải.")
        options = parser.parse_args(arguments)
        try:
            cls.from_project_root(options.force, options.output_dir).run(options.verify_only)
        except (ValueError, RuntimeError, OSError, csv.Error, tarfile.TarError, zipfile.BadZipFile):
            print("Không thể tải/xác thực FER2013. Kiểm tra dữ liệu, token và Accept Rules.", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(FER2013Downloader.cli())