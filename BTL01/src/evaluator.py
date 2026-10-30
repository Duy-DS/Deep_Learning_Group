"""Đánh giá mô hình hồi quy (Evaluator).

Module này triển khai thành phần đánh giá T0304, tính toán các độ đo chuẩn:
MSE, RMSE, MAE hoàn toàn bằng PyTorch thay vì sử dụng thư viện ngoài.
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class RegressionEvaluator:
    """Thực hiện đánh giá các độ đo hồi quy (MSE, RMSE, MAE) cho dự án."""

    def __init__(self, device: torch.device) -> None:
        """Khởi tạo bộ đánh giá.

        Tham số:
            device: Thiết bị tính toán (CPU/CUDA/MPS) để đồng bộ tensor.
        """
        self.device = device
        self.mse_criterion = nn.MSELoss(reduction="mean").to(device)
        self.mae_criterion = nn.L1Loss(reduction="mean").to(device)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
        """Tính toán các độ đo trên một tập dữ liệu.

        Tham số:
            model: Mô hình PyTorch đã được huấn luyện và gọi .eval().
            dataloader: Dữ liệu cần đánh giá (kích thước batch (B, 2) và (B, 1)).

        Trả về:
            Từ điển chứa các giá trị 'mse', 'rmse', 'mae' kiểu float.

        Ngoại lệ:
            ValueError: Nếu dataloader rỗng.
        """
        model.eval()
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = model(inputs)

            batch_size = inputs.shape[0]

            # Khôi phục tổng loss thay vì trung bình batch để tránh sai số
            # khi batch cuối không đều (drop_last=False)
            batch_mse = self.mse_criterion(outputs, targets).item() * batch_size
            batch_mae = self.mae_criterion(outputs, targets).item() * batch_size

            total_mse += batch_mse
            total_mae += batch_mae
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("DataLoader không cung cấp mẫu dữ liệu nào để đánh giá.")

        final_mse = total_mse / total_samples
        final_mae = total_mae / total_samples
        final_rmse = math.sqrt(final_mse)

        return {
            "mse": final_mse,
            "rmse": final_rmse,
            "mae": final_mae,
        }

# Bi danh tuong thich theo dinh danh Class trong Hien phap AGENTS.md
Evaluator = RegressionEvaluator

__all__ = ['RegressionEvaluator', 'Evaluator']
