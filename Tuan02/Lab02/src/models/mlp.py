import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    Multilayer Perceptron (MLP) cho bài toán hồi quy giá nhà Ames.
    Kiến trúc mặc định: Input -> 128 (ReLU) -> 64 (ReLU) -> 1 (Linear).
    """
    def __init__(self, input_dim: int, hidden_dim1: int = 128, hidden_dim2: int = 64, dropout_rate: float = 0.0):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        layers.extend([
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
        ])
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dim2, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
