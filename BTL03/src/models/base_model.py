"""Lớp cơ sở OOP cho các mô hình được chuyển đổi từ Keras sang PyTorch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class BaseMigratedModel(nn.Module, ABC):
    """Hợp đồng chung cho mô hình PyTorch có kiểm tra shape, summary và hooks."""

    def __init__(self, model_name: str) -> None:
        """Khởi tạo tên định danh và bộ quản lý forward hooks.

        Args:
            model_name: Tên kiến trúc dùng cho logs và báo cáo parity.
        """
        super().__init__()
        if not model_name or not model_name.strip():
            raise ValueError("model_name không được để trống.")
        self.model_name = model_name.strip()
        self._feature_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._captured_features: dict[str, torch.Tensor] = {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền tensor NCHW qua kiến trúc.

        Args:
            x: Tensor ảnh PyTorch NCHW hoặc tensor đầu vào đặc thù đã định nghĩa.

        Returns:
            Tensor logits, dự đoán hồi quy hoặc output dense-prediction.
        """
        raise NotImplementedError

    def validate_input_shape(
        self, x: torch.Tensor, expected_channels: Optional[int] = None
    ) -> None:
        """Xác thực tensor ảnh NCHW và cảnh báo lỗi thường gặp khi đưa NHWC.

        Args:
            x: Tensor đầu vào cần kiểm tra.
            expected_channels: Số channel mong đợi; bỏ qua nếu ``None``.

        Raises:
            TypeError: Nếu đầu vào không phải torch.Tensor.
            ValueError: Nếu tensor không có 4 chiều hoặc sai NCHW/channel.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Đầu vào mô hình phải là torch.Tensor.")
        if x.ndim != 4:
            raise ValueError(
                f"Ảnh phải có 4 chiều NCHW (Batch, Channels, Height, Width); "
                f"nhận shape {tuple(x.shape)}."
            )
        batch, channels, height, width = x.shape
        if min(batch, channels, height, width) <= 0:
            raise ValueError(f"Mọi chiều NCHW phải dương; nhận shape {tuple(x.shape)}.")
        # Phát hiện trường hợp Keras NHWC thường có channel 1/3 ở trục cuối.
        if x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
            raise ValueError(
                f"Có vẻ tensor đang ở NHWC {tuple(x.shape)}; PyTorch yêu cầu NCHW. "
                "Hãy dùng tensor.permute(0, 3, 1, 2).contiguous()."
            )
        if expected_channels is not None and channels != expected_channels:
            raise ValueError(
                f"Sai số channel: cần {expected_channels}, nhận {channels} "
                f"trong shape NCHW {tuple(x.shape)}."
            )

    def count_parameters(self) -> Dict[str, int]:
        """Đếm tổng, trainable và non-trainable parameters.

        Returns:
            Dict[str, int]: Ba bộ đếm tương thích báo cáo parity Keras.
        """
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_params": int(total),
            "trainable_params": int(trainable),
            "non_trainable_params": int(total - trainable),
        }

    def summary(self, input_size: Tuple[int, ...]) -> str:
        """Tạo bảng layer/output-shape/parameter count mà không cần torchinfo.

        Args:
            input_size: Shape không gồm batch, ví dụ ``(1, 28, 28)``.

        Returns:
            str: Bảng tóm tắt; đồng thời được in ra stdout.

        Raises:
            ValueError: Nếu input_size rỗng hoặc có chiều không dương.
        """
        if not input_size or any(int(size) <= 0 for size in input_size):
            raise ValueError("input_size phải gồm các chiều dương, không gồm batch.")
        rows: list[tuple[str, str, int, int]] = []
        handles = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: object) -> None:
                output_shape = self._shape_of(output)
                params = list(module.parameters(recurse=False))
                total = sum(parameter.numel() for parameter in params)
                trainable = sum(parameter.numel() for parameter in params if parameter.requires_grad)
                rows.append((name, str(output_shape), total, trainable))
            return hook

        for name, module in self.named_modules():
            if name and not any(True for _ in module.children()):
                handles.append(module.register_forward_hook(make_hook(name)))
        training_state = self.training
        device = next(self.parameters(), torch.empty(0)).device
        dtype = next(self.parameters(), torch.empty(0)).dtype
        try:
            self.eval()
            dummy = torch.zeros((1, *input_size), device=device, dtype=dtype)
            with torch.no_grad():
                output = self(dummy)
        except Exception as error:
            raise RuntimeError(f"Không thể tạo summary với input_size={input_size}: {error}") from error
        finally:
            for handle in handles:
                handle.remove()
            self.train(training_state)

        counts = self.count_parameters()
        lines = [
            f"Model: {self.model_name}",
            f"Input shape: (1, {', '.join(str(size) for size in input_size)})",
            f"Output shape: {self._shape_of(output)}",
            "Layer | Output Shape | Params | Trainable",
            "--- | --- | ---: | ---:",
        ]
        lines.extend(
            f"{name} | {shape} | {total} | {trainable}"
            for name, shape, total, trainable in rows
        )
        lines.extend(
            [
                f"Total params: {counts['total_params']}",
                f"Trainable params: {counts['trainable_params']}",
                f"Non-trainable params: {counts['non_trainable_params']}",
            ]
        )
        result = "\n".join(lines)
        print(result)
        return result

    @staticmethod
    def _shape_of(value: object) -> object:
        """Lấy shape đệ quy từ tensor hoặc cấu trúc output lồng nhau."""
        if isinstance(value, torch.Tensor):
            return tuple(value.shape)
        if isinstance(value, (tuple, list)):
            return type(value)(BaseMigratedModel._shape_of(item) for item in value)
        if isinstance(value, dict):
            return {key: BaseMigratedModel._shape_of(item) for key, item in value.items()}
        return type(value).__name__

    def register_feature_hooks(self, layer_names: List[str]) -> None:
        """Đăng ký hook forward cho các module con được chọn.

        Args:
            layer_names: Tên chính xác trong ``named_modules()``.

        Raises:
            ValueError: Nếu tên layer rỗng hoặc không tồn tại.
        """
        self.clear_hooks()
        modules = dict(self.named_modules())
        missing = [name for name in layer_names if name not in modules or not name]
        if missing:
            raise ValueError(f"Không tìm thấy layer để trích feature: {missing}.")
        for name in layer_names:
            def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: object, key: str = name) -> None:
                del module, inputs
                if isinstance(output, torch.Tensor):
                    self._captured_features[key] = output.detach()
                else:
                    raise TypeError(f"Output của layer '{key}' không phải Tensor.")

            self._feature_hook_handles.append(modules[name].register_forward_hook(hook))

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Chạy forward và trả output cùng các activation đã hook.

        Args:
            x: Tensor đầu vào đúng hợp đồng của mô hình.

        Returns:
            Tuple gồm output forward và dict activation theo tên layer.
        """
        self._captured_features.clear()
        output = self(x)
        return output, dict(self._captured_features)

    def clear_hooks(self) -> None:
        """Gỡ mọi hook đã đăng ký và xóa activation giữ trong bộ nhớ."""
        for handle in self._feature_hook_handles:
            handle.remove()
        self._feature_hook_handles.clear()
        self._captured_features.clear()

    def init_weights_keras_style(self) -> None:
        """Khởi tạo Linear/Conv2d theo Glorot Uniform và bias bằng 0.

        Phép khởi tạo này tương đương Glorot Uniform của Keras cho các layer
        Linear/Conv2D có cùng fan-in/fan-out; các layer khác giữ nguyên.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)