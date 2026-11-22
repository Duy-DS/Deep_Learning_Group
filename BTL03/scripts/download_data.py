"""Tải MNIST, kiểm tra tài nguyên vision và tạo dữ liệu giả lập cục bộ."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


class DataBootstrapper:
    """Tải, kiểm tra và khởi tạo tài nguyên dữ liệu BTL03."""

    MNIST_URLS = {
        "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    }

    def __init__(self, project_root: Path) -> None:
        """Khởi tạo đường dẫn dự án tuyệt đối từ vị trí đã phát hiện."""
        self.project_root = project_root.resolve()
        self.data_root = self.project_root / "data" / "raw"

    def download_file(self, url: str, destination: Path) -> None:
        """Tải một URL cấu hình vào tệp tạm rồi đổi tên nguyên tử."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "BTL03-data-bootstrap/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if temporary.stat().st_size == 0:
                raise ValueError(f"Máy chủ trả về tệp rỗng: {url}")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def read_idx_images(path: Path) -> np.ndarray:
        """Đọc tệp IDX ảnh MNIST và xác thực magic/header/kích thước."""
        with gzip.open(path, "rb") as stream:
            header = stream.read(16)
            if len(header) != 16:
                raise ValueError(f"Header IDX ảnh không đầy đủ: {path}")
            magic, count, rows, columns = struct.unpack(">IIII", header)
            if magic != 2051 or (rows, columns) != (28, 28):
                raise ValueError(f"Header IDX ảnh MNIST không hợp lệ: {path}")
            payload = stream.read()
        expected = count * rows * columns
        if len(payload) != expected:
            raise ValueError(f"Kích thước IDX ảnh sai: cần {expected}, có {len(payload)}.")
        return np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, columns).copy()

    @staticmethod
    def read_idx_labels(path: Path) -> np.ndarray:
        """Đọc tệp IDX nhãn MNIST và xác thực magic/header/kích thước."""
        with gzip.open(path, "rb") as stream:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError(f"Header IDX nhãn không đầy đủ: {path}")
            magic, count = struct.unpack(">II", header)
            if magic != 2049:
                raise ValueError(f"Header IDX nhãn MNIST không hợp lệ: {path}")
            payload = stream.read()
        if len(payload) != count:
            raise ValueError(f"Số nhãn IDX sai: cần {count}, có {len(payload)}.")
        labels = np.frombuffer(payload, dtype=np.uint8).copy()
        if labels.size and int(labels.max()) > 9:
            raise ValueError("Nhãn MNIST nằm ngoài miền 0..9.")
        return labels

    def download_mnist(self) -> None:
        """Tải bốn tệp IDX gzip và giải mã thành các mảng NumPy bị ignore."""
        destination = self.data_root / "mnist"
        destination.mkdir(parents=True, exist_ok=True)
        downloaded: dict[str, Path] = {}
        for key, url in self.MNIST_URLS.items():
            filename = url.rsplit("/", 1)[-1]
            gz_path = destination / filename
            if not gz_path.is_file():
                print(f"Tải {filename} ...")
                self.download_file(url, gz_path)
            downloaded[key] = gz_path

        arrays = {
            "train_images": self.read_idx_images(downloaded["train_images"]),
            "train_labels": self.read_idx_labels(downloaded["train_labels"]),
            "test_images": self.read_idx_images(downloaded["test_images"]),
            "test_labels": self.read_idx_labels(downloaded["test_labels"]),
        }
        for split in ("train", "test"):
            if len(arrays[f"{split}_images"]) != len(arrays[f"{split}_labels"]):
                raise ValueError(f"Số ảnh và nhãn MNIST {split} không khớp.")
        for name, array in arrays.items():
            output = destination / f"{name}.npy"
            temporary = output.with_suffix(".npy.part")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(output)
            print(f"Đã ghi {output.relative_to(self.project_root)}: {array.shape} {array.dtype}")

    def fetch_configured_assets(self, config_path: Path) -> None:
        """Tải tài nguyên WIKI/FER qua URL hoặc Google Drive ID đã cấu hình."""
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw_dir = self.project_root / config.get("raw_dir", "data/raw")
        urls = config.get("download_urls", {})
        drive_ids = config.get("google_drive_file_ids", {})
        for key, relative_name in config.get("files", {}).items():
            url = urls.get(key)
            destination = raw_dir / relative_name
            if destination.exists():
                print(f"Đã có: {destination.relative_to(self.project_root)}")
                continue
            if url:
                self.download_file(url, destination)
            elif drive_ids.get(key):
                try:
                    import gdown
                except ImportError as error:
                    raise RuntimeError(
                        "Cần cài gdown để tải Google Drive: pip install gdown"
                    ) from error
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".part")
                result = gdown.download(id=drive_ids[key], output=str(temporary), quiet=False)
                if not result or not temporary.is_file() or temporary.stat().st_size == 0:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"Tải Google Drive thất bại cho tài nguyên {key}.")
                temporary.replace(destination)
            else:
                print(
                    f"Bỏ qua {key}: cần điền download_urls.{key} hoặc "
                    f"google_drive_file_ids.{key} trong {config_path}."
                )
                continue
            print(f"Đã tải: {destination.relative_to(self.project_root)}")

    def create_dummy_data(self, samples: int = 8, seed: int = 42) -> None:
        """Tạo bộ dữ liệu nhỏ deterministic đúng schema để smoke-test offline."""
        if samples < 2:
            raise ValueError("samples phải >= 2 để có train và validation.")
        rng = np.random.default_rng(seed)
        root = self.data_root

        mnist_dir = root / "mnist_dummy"
        mnist_dir.mkdir(parents=True, exist_ok=True)
        np.save(mnist_dir / "train_images.npy", rng.integers(0, 256, (samples, 28, 28), dtype=np.uint8))
        np.save(mnist_dir / "train_labels.npy", np.arange(samples, dtype=np.uint8) % 10)

        wiki_dir = root / "wiki_gender"
        wiki_dir.mkdir(parents=True, exist_ok=True)
        # Ảnh tổng hợp được lưu NHWC trước xoay, đúng giao kèo của notebook.
        wiki_images = rng.integers(0, 256, (samples, 64, 64, 1), dtype=np.uint8)
        wiki_labels = np.eye(2, dtype=np.float32)[np.arange(samples) % 2]
        np.save(wiki_dir / "train_x.npy", wiki_images)
        np.save(wiki_dir / "train_y.npy", wiki_labels)
        np.save(wiki_dir / "validation_x.npy", wiki_images[: max(1, samples // 2)])
        np.save(wiki_dir / "validation_y.npy", wiki_labels[: max(1, samples // 2)])

        fer_dir = root / "fer2013"
        fer_dir.mkdir(parents=True, exist_ok=True)
        fer_images = rng.integers(0, 256, (samples, 48, 48), dtype=np.uint8)
        fer_labels = np.eye(7, dtype=np.float32)[np.arange(samples) % 7]
        fer_rows = np.concatenate(
            [fer_images.reshape(samples, -1).astype(np.float32), fer_labels], axis=1
        )
        np.savetxt(fer_dir / "fer2013_training_onehot.csv", fer_rows, delimiter=",", fmt="%.0f")
        np.savetxt(fer_dir / "fer2013_publictest_onehot.csv", fer_rows[: max(1, samples // 2)], delimiter=",", fmt="%.0f")
        print(f"Đã tạo dữ liệu giả lập deterministic trong {root.relative_to(self.project_root)} (N={samples}, seed={seed}).")

    def check_assets(self) -> None:
        """Báo cáo sự hiện diện và schema sơ bộ các bộ dữ liệu cục bộ."""
        required = {
            "MNIST": [
                self.data_root / "mnist" / "train_images.npy",
                self.data_root / "mnist" / "train_labels.npy",
                self.data_root / "mnist" / "test_images.npy",
                self.data_root / "mnist" / "test_labels.npy",
            ],
            "WIKI Gender": [
                self.data_root / "wiki_gender" / "train_x.npy",
                self.data_root / "wiki_gender" / "train_y.npy",
                self.data_root / "wiki_gender" / "validation_x.npy",
                self.data_root / "wiki_gender" / "validation_y.npy",
            ],
            "FER2013": [
                self.data_root / "fer2013" / "fer2013_training_onehot.csv",
                self.data_root / "fer2013" / "fer2013_publictest_onehot.csv",
            ],
        }
        for dataset_name, paths in required.items():
            missing = [path for path in paths if not path.is_file()]
            if missing:
                print(f"{dataset_name}: THIẾU " + ", ".join(p.name for p in missing))
                continue
            print(f"{dataset_name}: đủ {len(paths)} tệp cần thiết.")
            for path in paths:
                print(f"  {path.relative_to(self.project_root)} ({path.stat().st_size} bytes)")


def _project_root() -> Path:
    """Tìm thư mục BTL03 từ vị trí script, không phụ thuộc cwd."""
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào CLI tải, kiểm tra hoặc tạo dữ liệu kiểm thử."""
    # PowerShell/Windows có thể dùng code page không biểu diễn đủ tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download-mnist", help="Tải và giải mã MNIST IDX.")
    subparsers.add_parser("check", help="Kiểm tra sự hiện diện của tài nguyên thật.")
    dummy = subparsers.add_parser("dummy", help="Tạo dữ liệu nhỏ để smoke-test offline.")
    dummy.add_argument("--samples", type=int, default=8)
    dummy.add_argument("--seed", type=int, default=42)
    configured = subparsers.add_parser("fetch-configured", help="Tải URL trực tiếp khai báo trong YAML.")
    configured.add_argument("config", type=Path)

    args = parser.parse_args(argv)
    bootstrapper = DataBootstrapper(_project_root())
    try:
        if args.command == "download-mnist":
            bootstrapper.download_mnist()
        elif args.command == "check":
            bootstrapper.check_assets()
        elif args.command == "dummy":
            bootstrapper.create_dummy_data(args.samples, args.seed)
        elif args.command == "fetch-configured":
            config_path = args.config if args.config.is_absolute() else bootstrapper.project_root / args.config
            bootstrapper.fetch_configured_assets(config_path)
    except Exception as error:
        print(f"Lỗi dữ liệu: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())