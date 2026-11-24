"""Trực quan hóa bộ lọc tích chập, feature maps và Grad-CAM bằng PyTorch."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMVisualizer:
    """Trích xuất convolution filters/activations và sinh bản đồ Grad-CAM."""

    def __init__(self, model: nn.Module, output_dir: Path = Path("outputs/figures/t0502"),
                 device: Optional[torch.device] = None) -> None:
        """Khởi tạo model và thiết bị suy luận."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    def _convolutions(self) -> List[Tuple[str, nn.Conv2d]]:
        """Trả về mọi Conv2d theo thứ tự duyệt kiến trúc."""
        return [(name, layer) for name, layer in self.model.named_modules() if isinstance(layer, nn.Conv2d)]

    def plot_filters(self, layer: Optional[nn.Conv2d] = None, max_filters: int = 32,
                     filename: str = "conv_filters.png") -> Path:
        """Vẽ các kernel của tầng Conv2d đầu tiên hoặc tầng do người dùng chọn."""
        if max_filters <= 0:
            raise ValueError("max_filters phải dương.")
        convolutions = self._convolutions()
        if layer is None:
            if not convolutions:
                raise ValueError("Model không có tầng nn.Conv2d.")
            layer = convolutions[0][1]
        weights = layer.weight.detach().cpu()
        count = min(max_filters, weights.shape[0])
        columns = min(8, count)
        rows = (count + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.5, rows * 1.5), squeeze=False)
        for index, axis in enumerate(axes.flat):
            axis.axis("off")
            if index >= count:
                continue
            kernel = weights[index].mean(dim=0).numpy()
            low, high = float(kernel.min()), float(kernel.max())
            shown = (kernel - low) / (high - low) if high > low else np.zeros_like(kernel)
            axis.imshow(shown, cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"Kênh {index}", fontsize=8)
        fig.suptitle("Bộ lọc tích chập — giá trị trọng số đã chuẩn hóa")
        return self._save_figure(fig, filename)

    def extract_feature_maps(self, images: torch.Tensor,
                             layers: Optional[Sequence[nn.Module]] = None) -> Dict[str, torch.Tensor]:
        """Trích xuất feature maps ở các tầng tích chập đầu/cuối bằng forward hooks."""
        if images.ndim != 4:
            raise ValueError("images phải có shape [B,C,H,W].")
        convolutions = self._convolutions()
        if not convolutions:
            raise ValueError("Model không có tầng nn.Conv2d.")
        selected = list(layers) if layers is not None else [convolutions[0][1], convolutions[-1][1]]
        captures: Dict[str, torch.Tensor] = {}
        names = {id(layer): name for name, layer in convolutions}
        handles = []
        for index, layer in enumerate(selected):
            key = names.get(id(layer), f"layer_{index}")
            handles.append(layer.register_forward_hook(
                lambda _module, _inputs, output, name=key: captures.__setitem__(name, output.detach().cpu())
            ))
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                self.model(images.to(self.device))
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)
        return captures

    def plot_feature_maps(self, feature_maps: Dict[str, torch.Tensor], sample_index: int = 0,
                          max_channels: int = 16, filename: str = "feature_maps.png") -> Path:
        """Vẽ lưới activation map cho các tầng đã trích xuất."""
        if not feature_maps or max_channels <= 0:
            raise ValueError("Cần feature maps và max_channels dương.")
        tensors = []
        for name, batch in feature_maps.items():
            if batch.ndim != 4 or not 0 <= sample_index < batch.shape[0]:
                raise ValueError(f"Feature map không hợp lệ tại {name}.")
            tensors.extend((f"{name}:{index}", batch[sample_index, index])
                           for index in range(min(max_channels, batch.shape[1])))
        columns = min(8, len(tensors))
        rows = (len(tensors) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.6, rows * 1.6), squeeze=False)
        for axis, (title, activation) in zip(axes.flat, tensors):
            axis.imshow(activation.numpy(), cmap="viridis")
            axis.set_title(title, fontsize=6)
            axis.axis("off")
        for axis in axes.flat[len(tensors):]:
            axis.axis("off")
        fig.suptitle("Feature maps tích chập")
        return self._save_figure(fig, filename)

    def _resolve_target_layer(self, target_layer: Optional[nn.Module]) -> nn.Module:
        """Chọn tầng Grad-CAM tường minh, get_target_layer hoặc Conv cuối."""
        if target_layer is not None:
            return target_layer
        getter = getattr(self.model, "get_target_layer", None)
        if callable(getter):
            layer = getter()
            if isinstance(layer, nn.Module):
                return layer
        convolutions = self._convolutions()
        if not convolutions:
            raise ValueError("Không tìm được target_layer hoặc Conv2d trong model.")
        return convolutions[-1][1]

    def compute_heatmap(self, image: torch.Tensor, class_index: Optional[int] = None,
                        target_layer: Optional[nn.Module] = None) -> torch.Tensor:
        """Tính Grad-CAM và trả heatmap nội suy đúng H×W, nằm trong [0, 1]."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Grad-CAM nhận đúng một ảnh [1,C,H,W] hoặc [C,H,W].")
        layer = self._resolve_target_layer(target_layer)
        captured: Dict[str, torch.Tensor] = {}

        def capture_activation(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured["activation"] = output
            output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

        handle = layer.register_forward_hook(capture_activation)
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        try:
            logits = self.model(image.to(self.device))
            if logits.ndim != 2 or "activation" not in captured:
                raise RuntimeError("Model phải trả logits [B,K] và target layer phải tạo feature map.")
            selected_class = int(logits.argmax(1).item()) if class_index is None else int(class_index)
            if not 0 <= selected_class < logits.shape[1]:
                raise ValueError("class_index nằm ngoài miền logits.")
            logits[0, selected_class].backward()
            activation, gradient = captured["activation"], captured.get("gradient")
            if gradient is None:
                raise RuntimeError("Không thu được gradient tại target layer.")
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = cam - cam.min()
            maximum = cam.max()
            if maximum > 0:
                cam = cam / maximum
            return cam.detach().cpu().clamp_(0.0, 1.0)
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def save_overlay(self, image: torch.Tensor, heatmap: torch.Tensor,
                      filename: str = "gradcam_overlay.png", alpha: float = 0.4) -> Path:
        """Phủ heatmap lên ảnh xám/RGB đã qua pipeline evaluation."""
        if not 0.0 <= alpha <= 1.0 or heatmap.ndim != 2:
            raise ValueError("alpha phải thuộc [0,1] và heatmap phải là ma trận 2D.")
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError("Ảnh phải có shape [C,H,W] hoặc [1,C,H,W].")
        pixels = tensor.permute(1, 2, 0).numpy()
        low, high = float(pixels.min()), float(pixels.max())
        pixels = (pixels - low) / (high - low) if high > low else np.zeros_like(pixels)
        if pixels.shape[-1] == 1:
            pixels = pixels[..., 0]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.imshow(pixels, cmap="gray" if pixels.ndim == 2 else None)
        axis.imshow(heatmap.detach().cpu().numpy(), cmap="jet", alpha=alpha, vmin=0, vmax=1)
        axis.axis("off")
        return self._save_figure(fig, filename)

    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        """Lưu figure rồi đóng để giải phóng tài nguyên Matplotlib."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / filename
        figure.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(figure)
        return destination