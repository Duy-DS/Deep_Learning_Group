"""Kiểm thử downloader FER2013 bằng archive giả lập, không truy cập mạng."""

import csv
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from src.data.downloader import FER2013Downloader


class TestFER2013Downloader:
    """Kiểm tra giải nén lồng nhau, dọn dẹp và các cổng xác thực dữ liệu."""

    @staticmethod
    def _write_csv(target: io.TextIOBase, split_counts: dict[str, int]) -> None:
        """Tạo CSV tối giản nhưng đúng số mẫu/split để dùng trong unit test."""
        writer = csv.writer(target)
        writer.writerow(FER2013Downloader.REQUIRED_COLUMNS)
        for split_name, count in split_counts.items():
            for _ in range(count):
                writer.writerow(["0", "0", split_name])

    @staticmethod
    def _build_nested_archive(tmp_path: Path) -> Path:
        """Tạo ZIP chứa TAR.GZ chứa fer2013.csv giả lập."""
        csv_buffer = io.StringIO()
        TestFER2013Downloader._write_csv(csv_buffer, FER2013Downloader.EXPECTED_SPLITS)
        csv_payload = csv_buffer.getvalue().encode("utf-8")

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar_archive:
            csv_info = tarfile.TarInfo("dataset/fer2013.csv")
            csv_info.size = len(csv_payload)
            tar_archive.addfile(csv_info, io.BytesIO(csv_payload))

        outer_zip = tmp_path / "kaggle_competition.zip"
        with zipfile.ZipFile(outer_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_archive:
            zip_archive.writestr("download/fer2013.tar.gz", tar_buffer.getvalue())
        return outer_zip

    def test_nested_archives_download_and_cleanup(self, tmp_path: Path) -> None:
        """Tải giả lập ZIP→TAR.GZ→CSV, xác thực và chỉ giữ CSV đích."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        archive_path = self._build_nested_archive(tmp_path)
        downloader = FER2013Downloader(raw_dir)
        downloader._has_kaggle_credentials = lambda: True  # type: ignore[method-assign]

        def fake_download(destination: Path) -> None:
            """Mô phỏng Kaggle API bằng cách đặt archive đã tạo vào thư mục tải."""
            (destination / archive_path.name).write_bytes(archive_path.read_bytes())

        downloader._download_competition = fake_download  # type: ignore[method-assign]
        result = downloader.run()

        assert result == raw_dir / "fer2013.csv"
        assert FER2013Downloader.validate_csv(result) == FER2013Downloader.EXPECTED_SPLITS
        assert len(FER2013Downloader.calculate_sha256(result)) == 64
        assert sorted(path.name for path in raw_dir.iterdir()) == ["fer2013.csv"]

    def test_valid_cache_skips_download(self, tmp_path: Path) -> None:
        """Giữ nguyên cache hợp lệ và không gọi lại Kaggle API."""
        csv_buffer = io.StringIO()
        self._write_csv(csv_buffer, FER2013Downloader.EXPECTED_SPLITS)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        csv_path = raw_dir / FER2013Downloader.OUTPUT_NAME
        csv_path.write_text(csv_buffer.getvalue(), encoding="utf-8")
        downloader = FER2013Downloader(raw_dir)

        def fail_download(destination: Path) -> None:
            """Báo lỗi nếu cache hợp lệ mà downloader vẫn định tải."""
            raise AssertionError("Không được tải lại khi cache hợp lệ.")

        downloader._download_competition = fail_download  # type: ignore[method-assign]
        assert downloader.run() == csv_path

    @pytest.mark.parametrize(
        ("headers", "rows", "message"),
        [
            (["emotion", "pixels"], [["0", "0"]], "Header CSV phải đúng thứ tự"),
            (
                list(FER2013Downloader.REQUIRED_COLUMNS),
                [["0", "0", "Training"]],
                "CSV có 1 mẫu",
            ),
        ],
    )
    def test_integrity_validation_rejects_invalid_csv(
        self,
        tmp_path: Path,
        headers: list[str],
        rows: list[list[str]],
        message: str,
    ) -> None:
        """Báo lỗi rõ ràng khi CSV thiếu cột hoặc sai số mẫu."""
        csv_path = tmp_path / "invalid.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(rows)

        with pytest.raises(ValueError, match=message):
            FER2013Downloader.validate_csv(csv_path)