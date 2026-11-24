"""Chạy kiểm tra offline DataLoader và lưu ảnh WIKI sau phép xoay legacy."""

from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

# Cho phép chạy cả `python scripts/smoke_test_datasets.py` và `python -m ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_data import DataBootstrapper
from src.datasets import FEREmotionDataset, MNISTDataset, WikiGenderDataset
from src.datasets.vision_datasets import load_fer_onehot_csv


class DatasetSmokeTest:
    """Tạo mẫu nhỏ và xác minh hợp đồng shape/dtype/giá trị của ba dataset."""

    def __init__(self, project_root: Path) -> None:
        """Nhận thư mục gốc BTL03 và khởi tạo bộ chuẩn bị dữ liệu."""
        self.project_root = project_root.resolve()
        self.bootstrapper = DataBootstrapper(self.project_root)

    @staticmethod
    def _verify_loader(name: str, loader: DataLoader, expected_shape: tuple[int, ...], classes: int) -> None:
        """Xác minh batch ảnh/nhãn, kiểu số và miền giá trị chuẩn."""
        images, labels = next(iter(loader))
        assert tuple(images.shape[1:]) == expected_shape, (name, images.shape)
        assert images.dtype == torch.float32, (name, images.dtype)
        assert labels.dtype == torch.long, (name, labels.dtype)
        assert labels.ndim == 1 and int(labels.min()) >= 0 and int(labels.max()) < classes
        assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0
        print(
            f"{name}: images={tuple(images.shape)} {images.dtype} "
            f"range=[{images.min().item():.4f},{images.max().item():.4f}]; "
            f"labels={tuple(labels.shape)} {labels.dtype} "
            f"range=[{labels.min().item()},{labels.max().item()}]"
        )

    def run(self, samples: int = 8, seed: int = 42) -> Path:
        """Tạo dummy data, chạy ba DataLoader và lưu mẫu ảnh WIKI đã xoay."""
        self.bootstrapper.create_dummy_data(samples=samples, seed=seed)
        raw = self.project_root / "data" / "raw"

        mnist_dir = raw / "mnist_dummy"
        mnist = MNISTDataset(
            mnist_dir / "train_images.npy", np.load(mnist_dir / "train_labels.npy")
        )
        self._verify_loader("MNIST", DataLoader(mnist, batch_size=4), (1, 28, 28), 10)

        wiki_dir = raw / "wiki_gender"
        wiki_images = np.load(wiki_dir / "train_x.npy")
        wiki_labels = np.load(wiki_dir / "train_y.npy")
        # So sánh sau đúng phép chuẩn hóa uint8 [0,255] -> float [0,1].
        expected_rotation = np.rot90(wiki_images, k=3, axes=(1, 2)).astype(np.float32) / 255.0
        wiki = WikiGenderDataset(wiki_images, wiki_labels)
        self._verify_loader("WIKI Gender", DataLoader(wiki, batch_size=4), (1, 64, 64), 2)
        assert np.array_equal(wiki.images.permute(0, 2, 3, 1).numpy(), expected_rotation)

        fer_dir = raw / "fer2013"
        fer_images, fer_labels = load_fer_onehot_csv(
            fer_dir / "fer2013_training_onehot.csv"
        )
        fer = FEREmotionDataset(fer_images, fer_labels, normalization="auto")
        self._verify_loader("FER2013", DataLoader(fer, batch_size=4), (1, 48, 48), 7)

        sample = wiki.images[0, 0].numpy()
        image = np.clip(sample * 255.0, 0, 255).astype(np.uint8)
        output = self.project_root / "outputs" / "figures" / "wiki_gender_rot90_smoke.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="L").save(output)
        print(f"Ảnh WIKI sau rot90(k=3) lưu tại: {output.relative_to(self.project_root)}")
        print("SMOKE TEST: PASS (dữ liệu giả lập; chưa kiểm tra dữ liệu Google Drive thật).")
        return output


def main() -> None:
    """Chạy smoke-test độc lập từ thư mục gốc BTL03."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    DatasetSmokeTest(PROJECT_ROOT).run()


if __name__ == "__main__":
    main()