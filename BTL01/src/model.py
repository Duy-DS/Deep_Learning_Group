"""Mạng perceptron đa tầng (MLP) xấp xỉ hàm.

Module này triển khai thành phần T0301: một MLP có thể cấu hình bằng PyTorch.
"""

from typing import Any, Mapping

import torch
import torch.nn as nn


class FunctionApproximatorMLP(nn.Module):
    """Mạng perceptron đa tầng có thể cấu hình để xấp xỉ mật độ Gaussian 2D.

    Thuộc tính:
        layers (nn.Sequential): Container chứa các tầng của mạng lưới theo tuần tự.
    """

    def __init__(
        self, model_config: Mapping[str, Any], data_config: Mapping[str, Any]
    ) -> None:
        """Khởi tạo kiến trúc MLP dựa trên cấu hình.

        Tham số:
            model_config: Mapping chứa các tham số của mô hình (ví dụ: từ `model_baseline`).
            data_config: Mapping chứa các chiều dữ liệu (ví dụ: từ `data`).

        Ngoại lệ:
            ValueError: Nếu yêu cầu một hàm kích hoạt hoặc hàm khởi tạo không được hỗ trợ.
        """
        super().__init__()

        input_dim = data_config["input_dim"]
        output_dim = data_config["output_dim"]
        hidden_dims = model_config["hidden_dims"]
        use_bias = model_config["bias"]

        layers: list[nn.Module] = []
        current_dim = input_dim

        # Build Hidden Layers
        for hidden_dim in hidden_dims:
            linear = nn.Linear(current_dim, hidden_dim, bias=use_bias)
            self._init_weights(
                linear,
                model_config["weight_init"],
                model_config["bias_init"],
                model_config["hidden_init_gain"],
            )
            layers.append(linear)
            layers.append(
                self._get_activation(model_config["activation"], model_config)
            )
            current_dim = hidden_dim

        # Build Output Layer
        output_layer = nn.Linear(current_dim, output_dim, bias=use_bias)
        self._init_weights(
            output_layer,
            model_config["weight_init"],
            model_config["bias_init"],
            model_config["output_init_gain"],
        )
        layers.append(output_layer)
        layers.append(
            self._get_activation(model_config["output_activation"], model_config)
        )

        self.layers = nn.Sequential(*layers)

    def _get_activation(self, name: str, config: Mapping[str, Any]) -> nn.Module:
        """Ánh xạ tên hàm kích hoạt sang Module của PyTorch."""
        if name == "tanh":
            return nn.Tanh()
        elif name == "relu":
            return nn.ReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "leaky_relu":
            return nn.LeakyReLU(
                negative_slope=config.get("leaky_relu_negative_slope", 0.01)
            )
        elif name == "linear":
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def _init_weights(
        self, module: nn.Linear, weight_init: str, bias_init: str, gain: float
    ) -> None:
        """Áp dụng khởi tạo trọng số và bias."""
        if weight_init == "xavier_normal":
            nn.init.xavier_normal_(module.weight, gain=gain)
        else:
            raise ValueError(f"Unsupported weight initialization: {weight_init}")

        if module.bias is not None:
            if bias_init == "zeros":
                nn.init.zeros_(module.bias)
            else:
                raise ValueError(f"Unsupported bias initialization: {bias_init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Truyền tiến (forward pass) của MLP.

        Tham số:
            x: Tensor đầu vào có kích thước (batch_size, input_dim).

        Trả về:
            Tensor đầu ra có kích thước (batch_size, output_dim).
        """
        return self.layers(x)
