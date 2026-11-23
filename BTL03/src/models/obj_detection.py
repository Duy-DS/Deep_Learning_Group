"""FCN score-map detector chuyển đổi từ notebook Keras ``2_obj_detection``."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .base_model import BaseMigratedModel


class FCNObjectDetector(BaseMigratedModel):
    """Phân loại score xe/non-xe bằng fully convolutional sliding windows.

    Notebook gốc học target +1/-1 với tanh và MSE. Vì vậy forward trả score
    liên tục; confidence của lớp dương được quy đổi tuyến tính từ [-1,1] về
    [0,1], không áp dụng softmax không tương thích bài toán hồi quy này.
    """

    def __init__(self, input_size: int = 64, window_stride: int = 8) -> None:
        """Khởi tạo backbone 10 kênh và head convolution 8×8 / 1×1.

        Args:
            input_size: Kích thước patch train vuông, legacy là 64.
            window_stride: Bước MaxPool cố định của kiến trúc legacy.
        """
        super().__init__("FCNObjectDetector")
        if input_size < 16 or input_size % window_stride:
            raise ValueError("input_size phải >=16 và chia hết cho window_stride.")
        if window_stride != 8:
            raise ValueError("Kiến trúc legacy dùng MaxPool2d(kernel_size=8, stride=8).")
        self.input_size = int(input_size)
        self.window_stride = int(window_stride)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Dropout(p=0.25),
        )
        self.hidden_conv = nn.Conv2d(10, 128, kernel_size=8, padding=0)
        self.hidden_dropout = nn.Dropout(p=0.5)
        self.score_conv = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        self.output_activation = nn.Tanh()
        self.init_weights_keras_style()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả raw tanh score `(B,1,Hout,Wout)` trên mọi kích thước ảnh.

        Args:
            x: RGB NCHW float32 đã chuẩn hóa về [-1,1].

        Returns:
            Score map tanh; patch 64×64 tạo score shape `(B,1,1,1)`.
        """
        self.validate_input_shape(x, expected_channels=3)
        if not torch.is_floating_point(x):
            raise TypeError("Ảnh FCN detector phải có kiểu số thực float32.")
        features = self.backbone(x)
        if features.shape[-2] < 8 or features.shape[-1] < 8:
            raise ValueError("Ảnh phải có ít nhất 64×64 pixel cho cửa sổ detector 8×8.")
        x = torch.relu(self.hidden_conv(features))
        x = self.hidden_dropout(x)
        return self.output_activation(self.score_conv(x))

    def detect_objects(
        self,
        x: torch.Tensor,
        target_class: int = 1,
        threshold: float = 0.8,
        stride: int = 8,
        win_size: int = 64,
    ) -> tuple[torch.Tensor, List[Tuple[int, int, int, int, float]]]:
        """Suy luận heatmap và lọc các cực đại cục bộ thành bounding boxes.

        Args:
            x: Batch RGB NCHW đã chuẩn hóa theo công thức FCN.
            target_class: Chỉ hỗ trợ 1 (xe dương) hoặc 0 (nền).
            threshold: Ngưỡng confidence trong [0,1].
            stride: Khoảng dịch cửa sổ theo pixel ảnh gốc.
            win_size: Kích thước bounding box vuông trên ảnh gốc.

        Returns:
            Tuple gồm confidence heatmap `(B,1,Hout,Wout)` và danh sách
            ``(x1,y1,x2,y2,score)`` theo batch, xếp theo score giảm dần.
        """
        if target_class not in (0, 1):
            raise ValueError("target_class phải là 1 (xe) hoặc 0 (nền).")
        if not 0.0 <= threshold <= 1.0 or stride <= 0 or win_size <= 0:
            raise ValueError("threshold phải thuộc [0,1]; stride/win_size phải dương.")
        if stride != self.window_stride:
            raise ValueError("stride phải bằng bước pooling 8 của mạng legacy.")
        score_map = self.forward(x)
        # Hồi quy tanh [-1,1] của Keras được ánh xạ thành confidence [0,1].
        probabilities = ((score_map + 1.0) * 0.5).clamp_(0.0, 1.0)
        if target_class == 0:
            probabilities = 1.0 - probabilities
        pooled = torch.nn.functional.max_pool2d(probabilities, kernel_size=3, stride=1, padding=1)
        maxima = probabilities.eq(pooled) & probabilities.ge(threshold)
        boxes: list[tuple[int, int, int, int, float]] = []
        height, width = x.shape[-2:]
        for batch_index in range(x.shape[0]):
            locations = torch.nonzero(maxima[batch_index, 0], as_tuple=False)
            candidates: list[tuple[int, int, int, int, float]] = []
            for grid_y, grid_x in locations.tolist():
                x1, y1 = grid_x * stride, grid_y * stride
                x2, y2 = min(x1 + win_size, width), min(y1 + win_size, height)
                candidates.append(
                    (x1, y1, x2, y2, float(probabilities[batch_index, 0, grid_y, grid_x].item()))
                )
            # NMS vuông tham dụng loại cửa sổ chồng lấn; giữ score cao trước.
            candidates.sort(key=lambda box: box[4], reverse=True)
            kept: list[tuple[int, int, int, int, float]] = []
            for candidate in candidates:
                x1, y1, x2, y2, _ = candidate
                if all(
                    min(x2, bx2) <= max(x1, bx1) or min(bx2, x2) <= max(bx1, x1)
                    or min(y2, by2) <= max(y1, by1) or min(by2, y2) <= max(by1, y1)
                    for bx1, by1, bx2, by2, _ in kept
                ):
                    kept.append(candidate)
            boxes.extend(kept)
        return probabilities, boxes

    def count_parameters(self) -> dict[str, int]:
        """Trả bộ đếm tham số đầy đủ của kiến trúc FCN."""
        return super().count_parameters()

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Trích score map cùng activation từ các forward hook đã đăng ký."""
        return super().extract_features(x)
