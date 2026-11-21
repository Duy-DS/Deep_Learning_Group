"""Kiểm thử downloader FER2013 bằng archive giả lập, không truy cập mạng."""

import csv
import io
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.data.downloader import FER2013Downloader


class TestFER2013Downloader:
    """Kiểm tra giải nén lồng nhau, dọn dẹp và các cổng xác thực dữ liệu."""

    @pytest.fixture(autouse=True)
    def isolate_environment_file(self):
        """Không cho các test archive đọc .env thật trong working copy."""
        with patch.object(FER2013Downloader, "load_environment"):
            yield

    @staticmethod
    def _write_csv(target: io.TextIOBase, split_counts: dict[str, int]) -> None:
        """Tạo CSV tối giản nhưng đúng số mẫu/split để dùng trong unit test."""
        writer = csv.writer(target)
        writer.writerow(FER2013Downloader.REQUIRED_COLUMNS)
        for split_name, count in split_counts.items():
            for _ in range(count):
                writer.writerow(["0", " ".join(["0"] * 2304), split_name])

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
        with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "synthetic-test-token"}, clear=True), patch(
            "src.data.downloader.kagglehub.competition_download", return_value=str(tmp_path)
        ) as download:
            result = downloader.run()
        download.assert_called_once_with(FER2013Downloader.COMPETITION, force_download=False)

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

        with patch("src.data.downloader.kagglehub.competition_download") as download:
            assert downloader.run() == csv_path
        download.assert_not_called()

    @pytest.mark.parametrize(
        ("headers", "rows", "message"),
        [
            (["emotion", "pixels"], [["0", "0"]], "Header CSV phải đúng thứ tự"),
            (
                list(FER2013Downloader.REQUIRED_COLUMNS),
                [["0", " ".join(["0"] * 2304), "Training"]],
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

    def test_authentication_priority(self, tmp_path: Path) -> None:
        """Token môi trường ưu tiên hơn access_token và credentials cũ."""
        config = tmp_path / ".kaggle"
        config.mkdir()
        (config / "access_token").write_text("synthetic-file-token", encoding="utf-8")
        with patch("src.data.downloader.Path.home", return_value=tmp_path), patch.dict(
            "os.environ", {"KAGGLE_API_TOKEN": "synthetic-env-token"}, clear=True
        ):
            assert FER2013Downloader.authentication_source() == "KAGGLE_API_TOKEN"
        with patch("src.data.downloader.Path.home", return_value=tmp_path), patch.dict(
            "os.environ", {}, clear=True
        ):
            assert FER2013Downloader.authentication_source() == "access_token"

    def test_missing_auth_and_verify_only_do_not_download(self, tmp_path: Path) -> None:
        """Thiếu token hoặc chỉ xác thực không được gọi mạng."""
        downloader = FER2013Downloader(tmp_path / "raw")
        with patch("src.data.downloader.Path.home", return_value=tmp_path), patch.dict(
            "os.environ", {}, clear=True
        ), patch("src.data.downloader.kagglehub.competition_download") as download:
            with pytest.raises(RuntimeError, match="Thiếu chứng thực"):
                downloader.run()
            with pytest.raises(FileNotFoundError):
                downloader.run(verify_only=True)
            download.assert_not_called()
        assert not downloader.output_dir.exists()

    def test_failed_download_preserves_original_and_hides_secret(self, tmp_path: Path) -> None:
        """Lỗi HTTP không được làm mất dữ liệu cũ hoặc in nội dung bí mật."""
        original = tmp_path / "fer2013.csv"
        original.write_text("invalid-original", encoding="utf-8")
        with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "synthetic-token"}, clear=True), patch(
            "src.data.downloader.kagglehub.competition_download",
            side_effect=RuntimeError("synthetic-secret-do-not-log"),
        ):
            with pytest.raises(RuntimeError) as error:
                FER2013Downloader(tmp_path, force=True).run()
        assert "synthetic-secret" not in str(error.value)
        assert original.read_text(encoding="utf-8") == "invalid-original"

    def test_invalid_candidate_cleanup_and_cache_untouched(self, tmp_path: Path) -> None:
        """CSV tải sai phải bị từ chối và thư mục tạm phải được dọn sạch."""
        cache = tmp_path / "cache"
        cache.mkdir()
        source = cache / "fer2013.csv"
        source.write_text("emotion,pixels,Usage\n", encoding="utf-8")
        raw = tmp_path / "raw"
        with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "synthetic-token"}, clear=True), patch(
            "src.data.downloader.kagglehub.competition_download", return_value=str(cache)
        ):
            with pytest.raises(ValueError, match="CSV có 0 mẫu"):
                FER2013Downloader(raw, force=True).run()
        assert source.exists()
        assert list(raw.iterdir()) == []

    def test_direct_tar_cache_and_force(self, tmp_path: Path) -> None:
        """Cache chứa TAR.GZ được đọc trực tiếp; force được truyền tới kagglehub."""
        archive = self._build_nested_archive(tmp_path)
        cache = tmp_path / "cache"
        cache.mkdir()
        with zipfile.ZipFile(archive) as outer:
            (cache / "fer2013.tar.gz").write_bytes(outer.read("download/fer2013.tar.gz"))
        with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "synthetic-token"}, clear=True), patch(
            "src.data.downloader.kagglehub.competition_download", return_value=str(cache)
        ) as download:
            result = FER2013Downloader(tmp_path / "raw", force=True).run()
        download.assert_called_once_with(FER2013Downloader.COMPETITION, force_download=True)
        assert result.is_file()
        assert (cache / "fer2013.tar.gz").is_file()

    def test_legacy_credentials_and_missing_credentials(self, tmp_path: Path) -> None:
        """Credentials cũ vẫn được nhận diện nhưng không mở tệp token thật."""
        with patch("src.data.downloader.Path.home", return_value=tmp_path), patch.dict(
            "os.environ", {"KAGGLE_USERNAME": "synthetic-user", "KAGGLE_KEY": "synthetic-key"}, clear=True
        ):
            assert FER2013Downloader.authentication_source() == "legacy_environment"
        config = tmp_path / ".kaggle"
        config.mkdir()
        (config / "kaggle.json").write_text(
            '{"username":"synthetic-user","key":"synthetic-key"}', encoding="utf-8"
        )
        with patch("src.data.downloader.Path.home", return_value=tmp_path), patch.dict("os.environ", {}, clear=True):
            assert FER2013Downloader.authentication_source() == "kaggle.json"


class TestDownloaderEnvironment:
    """Kiểm chứng nạp .env giả lập độc lập với cấu hình cá nhân."""

    def test_dotenv_loads_token_without_overriding_environment(self, tmp_path: Path) -> None:
        """Môi trường shell được ưu tiên và biến khác vẫn được nạp từ tệp."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "KAGGLE_API_TOKEN=synthetic-file-token\nKAGGLE_USERNAME=synthetic-user\n"
            "KAGGLE_KEY=synthetic-key\nMLFLOW_TRACKING_URI=synthetic-store\n", encoding="utf-8"
        )
        with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "synthetic-shell-token"}, clear=True):
            FER2013Downloader(tmp_path / "raw", env_file=env_file)
            import os

            assert os.environ["KAGGLE_API_TOKEN"] == "synthetic-shell-token"
            assert os.environ["KAGGLE_USERNAME"] == "synthetic-user"
            assert os.environ["KAGGLE_KEY"] == "synthetic-key"
            assert os.environ["MLFLOW_TRACKING_URI"] == "synthetic-store"
            assert FER2013Downloader.authentication_source() == "KAGGLE_API_TOKEN"

    def test_dotenv_token_is_available_to_kagglehub(self, tmp_path: Path) -> None:
        """Mock network kiểm tra token tệp được nạp trước lời gọi kagglehub."""
        env_file = tmp_path / ".env"
        env_file.write_text("KAGGLE_API_TOKEN=synthetic-dotenv-token\n", encoding="utf-8")
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "fer2013.csv").write_text("emotion,pixels,Usage\n", encoding="utf-8")
        with patch.dict("os.environ", {}, clear=True), patch(
            "src.data.downloader.Path.home", return_value=tmp_path
        ), patch("src.data.downloader.kagglehub.competition_download", return_value=str(cache)) as download:
            downloader = FER2013Downloader(tmp_path / "raw", env_file=env_file)
            assert downloader.authentication_source() == "KAGGLE_API_TOKEN"
            with pytest.raises(ValueError, match="CSV có 0 mẫu"):
                downloader.run()
            download.assert_called_once_with(FER2013Downloader.COMPETITION, force_download=False)

    def test_explicit_missing_dotenv_does_not_search_other_files(self, tmp_path: Path) -> None:
        """Tệp được chỉ định nhưng thiếu không kích hoạt dò tìm secrets ở nơi khác."""
        with patch("src.data.downloader.dotenv.load_dotenv") as load:
            FER2013Downloader(tmp_path / "raw", env_file=tmp_path / "missing.env")
        load.assert_not_called()

    def test_default_dotenv_search_prefers_project_then_cwd(self, tmp_path: Path) -> None:
        """Xác minh thứ tự dò đường dẫn mà không đọc .env dự án thật."""
        with patch("src.data.downloader.Path.is_file", side_effect=[False, True]), patch(
            "src.data.downloader.Path.cwd", return_value=tmp_path
        ), patch("src.data.downloader.dotenv.load_dotenv") as load:
            FER2013Downloader.load_environment()
        load.assert_called_once_with(
            tmp_path / ".env", override=False, interpolate=False, encoding="utf-8-sig"
        )

    def test_example_token_is_not_accepted(self, tmp_path: Path) -> None:
        """Copy mẫu nhưng chưa thay token phải nhận hướng dẫn, không gọi mạng."""
        with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "your_kaggle_api_token_here"}, clear=True), patch(
            "src.data.downloader.Path.home", return_value=tmp_path
        ):
            assert FER2013Downloader.authentication_source() is None