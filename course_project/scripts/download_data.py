"""Tải và xác minh bộ dữ liệu Beijing Multi-Site Air Quality từ kho UCI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict


class UciBeijingDataDownloader:
    """Tải 12 tệp CSV UCI, kiểm tra hợp đồng dữ liệu và lưu an toàn vào data/raw/."""

    ARCHIVE_URL = (
        "https://archive.ics.uci.edu/static/public/501/"
        "beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"
    )
    USER_AGENT = "DL-PM25-BJ-data-downloader/1.0"
    NESTED_ARCHIVE = "PRSA2017_Data_20130301-20170228.zip"
    EXPECTED_ROWS_PER_STATION = 35_064
    STATIONS = (
        "Aotizhongxin",
        "Changping",
        "Dingling",
        "Dongsi",
        "Guanyuan",
        "Gucheng",
        "Huairou",
        "Nongzhanguan",
        "Shunyi",
        "Tiantan",
        "Wanliu",
        "Wanshouxigong",
    )
    EXPECTED_COLUMNS = (
        "No",
        "year",
        "month",
        "day",
        "hour",
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "wd",
        "WSPM",
        "station",
    )

    def __init__(self, output_dir: Path | str, timeout: int = 60) -> None:
        """Khởi tạo downloader.

        Args:
            output_dir: Đường dẫn thư mục đích dành riêng cho dữ liệu thô.
            timeout: Thời gian chờ tối đa cho mỗi yêu cầu tải, tính bằng giây.
        """
        self.output_dir = output_dir.resolve()
        self.timeout = timeout

    def download(self) -> Dict[str, str]:
        """Tải, xác minh và lưu CSV; từ chối ghi đè file hiện có khác nội dung.

        Returns:
            Ánh xạ tên tệp CSV với trạng thái `created` hoặc `already_verified`.

        Raises:
            RuntimeError: Nếu archive, danh sách trạm hoặc CSV không đúng hợp đồng.
            FileExistsError: Nếu file đích tồn tại nhưng khác nội dung archive UCI.
            urllib.error.URLError: Nếu không thể kết nối tới UCI.
        """
        archive_bytes = self._download_archive()
        station_files = self._read_station_files(archive_bytes)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, str] = {}
        for filename, content in sorted(station_files.items()):
            destination = self.output_dir / filename
            if destination.exists():
                existing_hash = self._sha256(destination.read_bytes())
                archive_hash = self._sha256(content)
                if existing_hash != archive_hash:
                    raise FileExistsError(
                        f"Từ chối ghi đè file có nội dung khác archive UCI: {destination}"
                    )
                results[filename] = "already_verified"
                print(
                    f"[VERIFIED] {filename} | SHA-256={archive_hash} | preserved"
                )
                continue

            self._write_atomically(destination, content)
            results[filename] = "created"
            print(
                f"[SAVED] {destination} | rows={self.EXPECTED_ROWS_PER_STATION} | "
                f"SHA-256={self._sha256(content)}"
            )

        print(
            f"[SUCCESS] Verified {len(results)} station CSV files; "
            "existing files were preserved when their hashes matched."
        )
        return results

    def _download_archive(self) -> bytes:
        """Tải archive UCI chính thức vào bộ nhớ, không tạo file trung gian trong repo."""
        request = urllib.request.Request(
            self.ARCHIVE_URL,
            headers={"User-Agent": self.USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"UCI trả mã HTTP không hợp lệ: {response.status}."
                )
            payload = response.read()
        if not zipfile.is_zipfile(io.BytesIO(payload)):
            raise RuntimeError("Nội dung tải từ UCI không phải tệp ZIP hợp lệ.")
        print(f"[DOWNLOADED] UCI dataset 501 | archive_bytes={len(payload)}")
        return payload

    def _read_station_files(self, archive_bytes: bytes) -> Dict[str, bytes]:
        """Giải nén hai lớp ZIP trong bộ nhớ và chỉ nhận đúng 12 CSV theo schema."""
        try:
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer_archive:
                candidates = [
                    info
                    for info in outer_archive.infolist()
                    if PurePosixPath(info.filename).name == self.NESTED_ARCHIVE
                    and not info.is_dir()
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        "Archive UCI phải chứa đúng một archive trạm "
                        f"{self.NESTED_ARCHIVE}; tìm thấy {len(candidates)}."
                    )
                nested_bytes = outer_archive.read(candidates[0])

            station_files: Dict[str, bytes] = {}
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as station_archive:
                for info in station_archive.infolist():
                    if info.is_dir():
                        continue
                    filename = PurePosixPath(info.filename).name
                    match = re.fullmatch(
                        r"PRSA_Data_([A-Za-z]+)_20130301-20170228\.csv", filename
                    )
                    if match is None:
                        continue
                    station = match.group(1)
                    if station not in self.STATIONS:
                        raise RuntimeError(f"Tên trạm không nằm trong contract: {station}")
                    if filename in station_files:
                        raise RuntimeError(f"CSV trạm bị lặp trong archive: {filename}")
                    content = station_archive.read(info)
                    self._validate_csv(filename, station, content)
                    station_files[filename] = content
        except zipfile.BadZipFile as exc:
            raise RuntimeError("Archive ZIP lồng của UCI bị hỏng.") from exc

        expected = {
            f"PRSA_Data_{station}_20130301-20170228.csv"
            for station in self.STATIONS
        }
        actual = set(station_files)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(
                f"Danh sách CSV trạm không đúng; thiếu={missing}, dư={unexpected}."
            )
        return station_files

    def _validate_csv(self, filename: str, station: str, content: bytes) -> None:
        """Xác nhận header, số dòng và giá trị cột station trước khi lưu."""
        try:
            text = content.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if tuple(reader.fieldnames or ()) != self.EXPECTED_COLUMNS:
                raise RuntimeError(
                    f"Header CSV không đúng contract tại {filename}: {reader.fieldnames}"
                )
            row_count = 0
            for row in reader:
                row_count += 1
                if row.get("station") != station:
                    raise RuntimeError(
                        f"Cột station không khớp {station} trong {filename}, "
                        f"dòng {row_count + 1}."
                    )
            if row_count != self.EXPECTED_ROWS_PER_STATION:
                raise RuntimeError(
                    f"Số dòng không đúng tại {filename}: {row_count}, "
                    f"kỳ vọng {self.EXPECTED_ROWS_PER_STATION}."
                )
        except (UnicodeDecodeError, csv.Error) as exc:
            raise RuntimeError(f"Không đọc được CSV UTF-8 hợp lệ: {filename}.") from exc

    @staticmethod
    def _sha256(content: bytes) -> str:
        """Tính SHA-256 để xác minh file hiện có mà không ghi đè dữ liệu."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _write_atomically(destination: Path, content: bytes) -> None:
        """Ghi file mới qua file tạm cùng thư mục; không thay thế đích đã xuất hiện."""
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=".uci_download_", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.link(temporary_path, destination)
        except FileExistsError:
            raise
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def parse_args(cls) -> argparse.Namespace:
        """Đọc tham số dòng lệnh; yêu cầu chỉ rõ thư mục đích khi thực thi CLI."""
        parser = argparse.ArgumentParser(
            description="Download UCI Beijing Multi-Site Air Quality CSV files (ID 501)."
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help=(
                "Destination for the 12 station CSV files. Required explicitly; "
                "the script never defaults to the protected project data/raw directory."
            ),
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=60,
            help="Download timeout in seconds.",
        )
        return parser.parse_args()


def main() -> None:
    """Tạo downloader từ CLI và tải/xác minh dataset UCI 501."""
    args = UciBeijingDataDownloader.parse_args()
    downloader = UciBeijingDataDownloader(
        output_dir=args.output_dir,
        timeout=args.timeout,
    )
    downloader.download()


if __name__ == "__main__":
    main()