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
    Thiết lập nguồn ngẫu nhiên và chế độ cuDNN nhằm hỗ trợ tái lập.

    PYTHONHASHSEED chỉ có hiệu lực đầy đủ với hash khi đặt trước lúc khởi động
    Python; tính xác định giữa phần cứng/phiên bản khác nhau không được bảo đảm.

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

    # 5. Khóa PyTorch trên tất cả các GPU; hàm an toàn cả khi không có CUDA.
    torch.cuda.manual_seed_all(seed)

    # Luôn giữ trạng thái cuDNN nhất quán giữa các môi trường CPU và CUDA.
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