"""Tiện ích dùng chung cho thí nghiệm PyTorch."""

from pathlib import Path
import os
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .logger import CheckpointManager, ExperimentLogger

__all__ = [
    "seed_everything",
    "seed_worker",
    "get_torch_generator",
    "save_checkpoint",
    "ExperimentLogger",
    "CheckpointManager",
]


def __getattr__(name: str) -> Any:
    """Nạp logger theo yêu cầu để tiện ích seed không phụ thuộc TensorBoard.

    Args:
        name: Tên API logger được yêu cầu qua ``src.utils``.

    Returns:
        Lớp logger tương ứng.

    Raises:
        AttributeError: Nếu tên không phải API được export.
    """
    if name in {"ExperimentLogger", "CheckpointManager"}:
        from .logger import CheckpointManager, ExperimentLogger

        return {
            "ExperimentLogger": ExperimentLogger,
            "CheckpointManager": CheckpointManager,
        }[name]
    raise AttributeError(f"module {__name__!r} không có thuộc tính {name!r}")


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Đặt seed cho Python, NumPy và PyTorch, đồng thời tùy chọn chế độ tất định.

    Args:
        seed: Hạt giống dùng chung cho các bộ sinh số giả ngẫu nhiên.
        deterministic: Nếu đúng, bật thuật toán tất định và tắt cuDNN benchmark.

    Ghi chú:
        ``PYTHONHASHSEED`` được đặt để tiến trình con thừa hưởng cùng giá trị.
        Python hiện tại đã khởi tạo hash seed trước khi gọi hàm; muốn cố định
        hash iteration ngay từ đầu, hãy đặt biến môi trường trước khi chạy Python.
        CUDA có thể báo lỗi nếu một toán tử không có implementation tất định.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Các hàm seed CUDA an toàn khi CUDA không khả dụng và giữ cùng API CPU/GPU.
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Strict mode: phép toán CUDA không tất định sẽ báo lỗi rõ ràng.
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    """Đặt seed riêng cho Python/NumPy trong mỗi worker của DataLoader.

    PyTorch trước khi gọi hàm này đã gán seed riêng cho worker hiện tại. Hàm
    lấy seed đó để đồng bộ Python, NumPy và PyTorch trong tiến trình con.

    Args:
        worker_id: Chỉ số worker do DataLoader truyền vào.
    """
    del worker_id  # Seed gốc của worker được PyTorch quản lý qua initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_torch_generator(seed: int = 42) -> torch.Generator:
    """Tạo generator riêng để DataLoader shuffle tái lập.

    Args:
        seed: Seed khởi tạo generator.

    Returns:
        torch.Generator: Generator CPU đã được manual_seed.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Tạo thư mục cha nếu cần và lưu checkpoint PyTorch."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)