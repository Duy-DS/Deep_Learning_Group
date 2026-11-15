"""
Các hàm tiện ích dùng chung cho huấn luyện và đánh giá mô hình FER2013.
Tuân thủ chuẩn tái lập (Reproducibility) với Seed = 42 theo hiến pháp AGENTS.md.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    """
    Khóa cứng tất cả các nguồn sinh số ngẫu nhiên để tái lập kết quả 100%.

    Tham số:
        seed (int): Hạt giống ngẫu nhiên (mặc định = 42 theo quy định dự án).
    """
    # 1. Khóa môi trường hệ điều hành Python
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Khóa thư viện random mặc định của Python
    random.seed(seed)

    # 3. Khóa thư viện tính toán số học NumPy
    np.random.seed(seed)

    # 4. Khóa PyTorch trên CPU
    torch.manual_seed(seed)

    # 5. Khóa PyTorch trên tất cả các GPU (nếu có)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Bắt buộc thuật toán cuDNN chạy ở chế độ xác định (deterministic)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """
    Tính tổng số lượng tham số có thể huấn luyện (trainable parameters) của mô hình.

    Tham số:
        model (nn.Module): Mô hình mạng nơ-ron PyTorch.

    Trả về:
        int: Tổng số tham số cần cập nhật gradient.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    Tự động xác định thiết bị tính toán tối ưu nhất (GPU CUDA hoặc CPU).

    Trả về:
        torch.device: Thiết bị tính toán được chọn.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")