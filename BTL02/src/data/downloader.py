"""Tải, giải nén và xác thực bộ dữ liệu Kaggle FER2013.

Mọi thao tác được đóng gói trong ``FER2013Downloader``. Dữ liệu nén chỉ được
đọc trong thư mục tạm; chỉ CSV đã xác thực mới được thay thế vào ``data/raw``.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Dict, Iterable, List, Optional, Sequence


class FER2013Downloader:
    """Tải FER2013 bằng Kaggle API, kiểm tra cache và xác thực nội dung CSV."""

    COMPETITION = "challenges-in-representation-learning-facial-expression-recognition-challenge"
    COMPETITION_URL = (
        "https://www.kaggle.com/c/"
        "challenges-in-representation-learning-facial-expression-recognition-challenge/data"
    )
    REQUIRED_COLUMNS = ("emotion", "pixels", "Usage")
    EXPECTED_SPLITS = {"Training": 28709, "PublicTest": 3589, "PrivateTest": 3589}
    OUTPUT_NAME = "fer2013.csv"

    def __init__(self, output_dir: Path, force: bool = False) -> None:
        """Khởi tạo downloader.

        Args:
            output_dir: Thư mục đích chứa CSV chuẩn.
            force: Tải lại kể cả khi cache hiện có hợp lệ.
        """
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_path = self.output_dir / self.OUTPUT_NAME
        self.force = force

    @classmethod
    def from_project_root(cls, force: bool = False, output_dir: Optional[Path] = None) -> "FER2013Downloader":
        """Tạo downloader với thư mục raw chuẩn tính từ vị trí module."""
        project_root = Path(__file__).resolve().parents[2]
        destination = output_dir if output_dir is not None else project_root / "data" / "raw"
        return cls(destination, force=force)

    @staticmethod
    def _has_kaggle_credentials() -> bool:
        """Kiểm tra sự hiện diện thông tin xác thực mà không in bí mật ra màn hình."""
        username = os.environ.get("KAGGLE_USERNAME", "").strip()
        key = os.environ.get("KAGGLE_KEY", "").strip()
        if username and key:
            return True
        credential_file = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
        return credential_file.is_file()

    @classmethod
    def print_authentication_help(cls) -> None:
        """In hướng dẫn cấu hình Kaggle API bằng tiếng Việt."""
        print(
            "Chưa tìm thấy thông tin xác thực Kaggle.\n"
            "1. Đăng nhập Kaggle và mở trang dữ liệu cuộc thi:\n"
            f"   {cls.COMPETITION_URL}\n"
            "2. Nhấn 'Accept Rules' để chấp nhận điều lệ cuộc thi.\n"
            "3. Tạo API Token trong Kaggle Account Settings, đặt kaggle.json tại ~/.kaggle/kaggle.json\n"
            "   (Windows: %USERPROFILE%\\.kaggle\\kaggle.json), hoặc đặt KAGGLE_USERNAME và KAGGLE_KEY."
        )

    @staticmethod
    def _member_basename(member_name: str) -> str:
        """Lấy tên thành viên archive độc lập với đường dẫn do bên ngoài cung cấp."""
        return PurePosixPath(member_name.replace("\\", "/")).name.lower()

    @classmethod
    def _copy_stream(cls, source: BinaryIO, destination: Path, limit_bytes: int = 1_000_000_000) -> None:
        """Sao chép luồng theo khối và chặn thành viên giải nén quá lớn."""
        total_bytes = 0
        with Path(destination).open("wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > limit_bytes:
                    raise ValueError("Thành viên archive vượt giới hạn giải nén an toàn 1 GB.")
                target.write(chunk)

    @classmethod
    def _extract_csv_to_path(
        cls,
        source_path: Path,
        member_name: str,
        workspace: Path,
        depth: int = 0,
    ) -> Optional[Path]:
        """Trích xuất duy nhất CSV đích qua archive lồng nhau bằng streaming."""
        if depth > 5:
            raise ValueError("Archive FER2013 lồng quá sâu; dừng để bảo vệ an toàn.")

        basename = cls._member_basename(member_name)
        if basename == cls.OUTPUT_NAME:
            destination = workspace / f"candidate_{len(list(workspace.glob('candidate_*.csv')))}.csv"
            shutil.copyfile(source_path, destination)
            return destination

        nested_extensions = (".zip", ".tar.gz", ".tgz")
        if not basename.endswith(nested_extensions):
            return None

        if zipfile.is_zipfile(source_path):
            with zipfile.ZipFile(source_path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    child_name = cls._member_basename(member.filename)
                    if child_name != cls.OUTPUT_NAME and not child_name.endswith(nested_extensions):
                        continue
                    child_path = workspace / f"nested_{depth}_{len(list(workspace.glob('nested_*')))}"
                    with archive.open(member, "r") as child_stream:
                        cls._copy_stream(child_stream, child_path)
                    candidate = cls._extract_csv_to_path(child_path, child_name, workspace, depth + 1)
                    child_path.unlink(missing_ok=True)
                    if candidate is not None:
                        return candidate
            return None

        if tarfile.is_tarfile(source_path):
            with tarfile.open(source_path, mode="r:gz") as archive:
                for member in archive.getmembers():
                    # Chỉ đọc regular file; không trích xuất đường dẫn archive ra filesystem.
                    if not member.isfile():
                        continue
                    child_name = cls._member_basename(member.name)
                    if child_name != cls.OUTPUT_NAME and not child_name.endswith(nested_extensions):
                        continue
                    child_path = workspace / f"nested_{depth}_{len(list(workspace.glob('nested_*')))}"
                    child_stream = archive.extractfile(member)
                    if child_stream is None:
                        continue
                    with child_stream:
                        cls._copy_stream(child_stream, child_path)
                    candidate = cls._extract_csv_to_path(child_path, child_name, workspace, depth + 1)
                    child_path.unlink(missing_ok=True)
                    if candidate is not None:
                        return candidate
        return None

    @classmethod
    def _find_csv_file(cls, downloaded_files: Iterable[Path], workspace: Path) -> Path:
        """Tìm và trích xuất CSV chuẩn vào workspace, không nạp dataset vào RAM."""
        candidates: List[Path] = []
        for file_path in downloaded_files:
            if not file_path.is_file():
                continue
            candidate = cls._extract_csv_to_path(file_path, file_path.name, workspace)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            raise ValueError("Không tìm thấy fer2013.csv trong các file Kaggle đã tải xuống.")
        if len(candidates) > 1:
            hashes = {cls.calculate_sha256(candidate) for candidate in candidates}
            if len(hashes) != 1:
                raise ValueError("Archive chứa nhiều file fer2013.csv khác nhau; từ chối chọn dữ liệu mơ hồ.")
        selected = candidates[0]
        for duplicate in candidates[1:]:
            duplicate.unlink(missing_ok=True)
        return selected

    @classmethod
    def validate_csv(cls, csv_path: Path) -> Dict[str, int]:
        """Xác minh header, số dòng tổng và số dòng từng Usage.

        Args:
            csv_path: Tệp CSV cần kiểm tra.

        Returns:
            Dict[str, int]: Số lượng mẫu theo từng phân tập.

        Raises:
            ValueError: Khi cấu trúc hoặc số lượng mẫu không đúng đặc tả.
        """
        split_counts = {split: 0 for split in cls.EXPECTED_SPLITS}
        try:
            with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                headers = reader.fieldnames or []
                if tuple(headers) != cls.REQUIRED_COLUMNS:
                    raise ValueError(
                        "Header CSV phải đúng thứ tự "
                        f"{list(cls.REQUIRED_COLUMNS)}; nhận được: {headers}"
                    )
                total_rows = 0
                for row in reader:
                    total_rows += 1
                    usage = (row.get("Usage") or "").strip()
                    if usage not in split_counts:
                        raise ValueError(f"Giá trị Usage không hợp lệ tại dòng dữ liệu {total_rows}: {usage!r}")
                    if not (row.get("emotion") or "").strip() or not (row.get("pixels") or "").strip():
                        raise ValueError(f"Dòng dữ liệu {total_rows} thiếu emotion hoặc pixels.")
                    split_counts[usage] += 1
        except (UnicodeDecodeError, csv.Error) as error:
            raise ValueError(f"Không đọc được CSV hợp lệ tại {csv_path}: {error}") from error

        expected_total = sum(cls.EXPECTED_SPLITS.values())
        if total_rows != expected_total:
            raise ValueError(f"CSV có {total_rows:,} mẫu; cần đúng {expected_total:,}.")
        if split_counts != cls.EXPECTED_SPLITS:
            raise ValueError(
                f"Số mẫu theo Usage không đúng. Kỳ vọng {cls.EXPECTED_SPLITS}, nhận được {split_counts}."
            )
        return split_counts

    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """Tính SHA-256 theo luồng để không nạp toàn bộ CSV vào RAM."""
        digest = hashlib.sha256()
        with Path(file_path).open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _download_competition(self, destination: Path) -> None:
        """Tải toàn bộ file cuộc thi bằng Kaggle CLI vào thư mục tạm.

        Khi credentials được truyền qua biến môi trường, tạo tệp cấu hình tạm
        ngoài repository để Kaggle CLI đọc được mà không ghi bí mật vào log.
        """
        executable_name = "kaggle.exe" if os.name == "nt" else "kaggle"
        executable = Path(sys.executable).resolve().parent / executable_name
        if not executable.is_file():
            discovered = shutil.which("kaggle")
            if discovered is None:
                raise RuntimeError(
                    "Không tìm thấy lệnh Kaggle CLI. Hãy cài dependency bằng: pip install -r requirements.txt"
                )
            executable = Path(discovered)

        child_environment = os.environ.copy()
        username = child_environment.get("KAGGLE_USERNAME", "").strip()
        key = child_environment.get("KAGGLE_KEY", "").strip()
        if username and key:
            # Cấu hình tạm được xóa tự động khi Kaggle CLI kết thúc.
            with tempfile.TemporaryDirectory(prefix="fer2013_kaggle_auth_") as auth_dir:
                auth_path = Path(auth_dir) / "kaggle.json"
                auth_path.write_text(
                    json.dumps({"username": username, "key": key}),
                    encoding="utf-8",
                )
                try:
                    auth_path.chmod(0o600)
                except OSError:
                    # Windows ACL là cơ chế bảo vệ chính; chmod có thể không khả dụng.
                    pass
                child_environment["KAGGLE_CONFIG_DIR"] = auth_dir
                subprocess.run(
                    [
                        str(executable),
                        "competitions",
                        "download",
                        "-c",
                        self.COMPETITION,
                        "-p",
                        str(destination),
                        "-o",
                    ],
                    check=True,
                    env=child_environment,
                )
            return

        subprocess.run(
            [
                str(executable),
                "competitions",
                "download",
                "-c",
                self.COMPETITION,
                "-p",
                str(destination),
                "-o",
            ],
            check=True,
            env=child_environment,
        )

    def _report_valid_file(self, file_path: Path) -> str:
        """In kết quả xác thực và trả về mã SHA-256 của CSV."""
        counts = self.validate_csv(file_path)
        sha256 = self.calculate_sha256(file_path)
        print(f"CSV hợp lệ: {file_path}")
        print(f"Tổng số mẫu: {sum(counts.values()):,}")
        for split_name, count in counts.items():
            print(f"  {split_name}: {count:,}")
        print(f"SHA-256: {sha256}")
        return sha256

    def run(self, verify_only: bool = False) -> Path:
        """Xác thực cache hoặc tải, giải nén và cài đặt CSV đã kiểm chứng.

        Args:
            verify_only: Chỉ xác thực file đích hiện có, không tải dữ liệu.

        Returns:
            Path: Đường dẫn CSV chuẩn.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.output_path.is_file() and (not self.force or verify_only):
            try:
                self._report_valid_file(self.output_path)
                print("Cache hợp lệ; bỏ qua tải dữ liệu.")
                return self.output_path
            except (ValueError, csv.Error):
                if verify_only:
                    raise
                print("Cache hiện có không hợp lệ; sẽ tải lại từ Kaggle.")

        if verify_only:
            raise FileNotFoundError(f"Không tìm thấy CSV để xác thực: {self.output_path}")

        if not self._has_kaggle_credentials():
            self.print_authentication_help()
            raise RuntimeError("Cần cấu hình Kaggle API và chấp nhận điều lệ cuộc thi trước khi tải.")

        # Tải và kiểm tra trong cùng filesystem để thay thế nguyên tử sau xác thực.
        with tempfile.TemporaryDirectory(prefix="fer2013_download_", dir=self.output_dir) as temp_name:
            temp_dir = Path(temp_name)
            self._download_competition(temp_dir)
            downloaded_files = [path for path in temp_dir.iterdir() if path.is_file()]
            candidate_path = self._find_csv_file(downloaded_files, temp_dir)
            validated_path = temp_dir / "validated_fer2013.csv"
            if candidate_path != validated_path:
                candidate_path.replace(validated_path)
            candidate_path = validated_path
            self._report_valid_file(candidate_path)
            os.replace(candidate_path, self.output_path)

        self._report_valid_file(self.output_path)
        print("Tải và xác thực FER2013 hoàn tất.")
        return self.output_path

    @classmethod
    def cli(cls, arguments: Optional[Sequence[str]] = None) -> int:
        """Phân tích đối số dòng lệnh và chạy downloader."""
        parser = argparse.ArgumentParser(description="Tải và xác thực Kaggle FER2013.")
        parser.add_argument("--output-dir", type=Path, help="Thư mục đích chứa fer2013.csv.")
        parser.add_argument("--force", action="store_true", help="Bỏ qua cache hợp lệ và tải lại.")
        parser.add_argument("--verify-only", action="store_true", help="Chỉ xác thực CSV đã có.")
        options = parser.parse_args(arguments)
        downloader = cls.from_project_root(force=options.force, output_dir=options.output_dir)
        try:
            downloader.run(verify_only=options.verify_only)
        except Exception as error:
            print(f"Lỗi tải/xác thực FER2013: {error}", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(FER2013Downloader.cli())