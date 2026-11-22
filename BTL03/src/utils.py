"""Tiện ích dùng chung cho thí nghiệm PyTorch."""

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Đặt seed cho Python, NumPy và PyTorch để tăng tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Tạo thư mục cha nếu cần và lưu checkpoint PyTorch."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)