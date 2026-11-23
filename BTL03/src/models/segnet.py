"""Mô hình phân vùng ngữ nghĩa SegNet theo kiến trúc Keras legacy."""

from __future__ import annotations

import torch
from torch import nn

from .base_model import BaseMigratedModel


def _conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    """Tạo chuỗi Conv2d → BatchNorm2d → ReLU tương đương thứ tự Keras."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True),
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


class SegNet(BaseMigratedModel):
    """Encoder–decoder VGG-style cho mask 7 lớp, ảnh grayscale 96×96.

    Encoder có lần lượt 2, 2, 3, 3, 3 convolution; decoder khôi phục không
    gian bằng nearest-neighbor upsampling và lần lượt 3, 3, 3, 2 convolution.
    BatchNorm cuối decoder giữ nguyên như Keras; phân loại pixel dùng logits.
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 1) -> None:
        """Khởi tạo SegNet và kiểm tra cấu hình lớp/kênh.

        Args:
            num_classes: Số lớp pixel, legacy là 7.
            in_channels: Số kênh ảnh depth grayscale, legacy là 1.
        """
        super().__init__("SegNet")
        if num_classes < 2 or in_channels != 1:
            raise ValueError("SegNet legacy yêu cầu num_classes>=2 và in_channels=1.")
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        encoder_specs = ((64, 2), (128, 2), (256, 3), (512, 3), (512, 3))
        encoders: list[nn.Module] = []
        current_channels = in_channels
        for channels, conv_count in encoder_specs:
            layers: list[nn.Module] = []
            for _ in range(conv_count):
                layers.append(_conv_bn_relu(current_channels, channels))
                current_channels = channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            encoders.append(nn.Sequential(*layers))
        self.encoders = nn.ModuleList(encoders)

        decoder_specs = (
            (512, 512, 512),
            (512, 512, 256),
            (256, 256, 128),
            (128, 64),
            (64,),
        )
        decoders: list[nn.Module] = []
        current_channels = 512
        for channels_list in decoder_specs:
            layers = [nn.Upsample(scale_factor=2, mode="nearest")]
            for channels in channels_list:
                layers.append(_conv_bn_relu(current_channels, channels))
                current_channels = channels
            decoders.append(nn.Sequential(*layers))
        self.decoders = nn.ModuleList(decoders)
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1, padding=0, bias=True)
        # Keras BatchNormalization mặc định chuẩn hóa cả logits ở tầng cuối.
        self.output_batch_norm = nn.BatchNorm2d(num_classes, eps=1e-3, momentum=0.01)
        self.init_weights_keras_style()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả logits `(B,K,H,W)` từ ảnh depth NCHW.

        Args:
            x: Ảnh `(B,1,H,W)`; H và W cần chia hết cho 32.

        Returns:
            Tensor logits chưa softmax, dùng trực tiếp với CrossEntropyLoss.
        """
        self.validate_input_shape(x, expected_channels=self.in_channels)
        input_height, input_width = x.shape[-2:]
        if input_height % 32 or input_width % 32:
            raise ValueError("Chiều cao/rộng SegNet phải chia hết cho 32 (legacy 96×96).")
        for block in self.encoders:
            x = block(x)  # Mỗi MaxPool giảm H,W một nửa; tổng cộng giảm 32 lần.
        for block in self.decoders:
            x = block(x)  # Nearest upsampling khôi phục đúng thứ tự các kích thước.
        return self.output_batch_norm(self.segmentation_head(x))

    def count_parameters(self) -> dict[str, int]:
        """Đếm weights trainable và state BN để so sánh đúng Keras summary.

        PyTorch lưu running mean/variance trong buffers thay vì Parameters;
        tổng Keras legacy tính chúng vào tổng params nhưng không tính vào
        trainable params. ``num_batches_tracked`` của PyTorch không có bản
        tương đương trong Keras nên được loại khỏi phép parity.
        """
        base = super().count_parameters()
        non_trainable = sum(
            int(buffer.numel())
            for name, buffer in self.named_buffers()
            if name.endswith(("running_mean", "running_var"))
        )
        return {
            "total_params": base["trainable_params"] + non_trainable,
            "trainable_params": base["trainable_params"],
            "non_trainable_params": non_trainable,
        }

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Trích logits và các activation đã đăng ký bằng forward hook."""
        return super().extract_features(x)
