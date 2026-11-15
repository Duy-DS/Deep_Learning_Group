"""
Mô-đun định nghĩa các kiến trúc mô hình LSTM Baseline dự báo nồng độ PM2.5.

Class chính: LSTMModel — Mô hình LSTM đa tầng (Stacked LSTM) nhận chuỗi
thời gian đa biến [B, L, D] và dự báo nồng độ PM2.5 trong H bước tương lai,
trả về tensor [B, H, 1]. Kế thừa BaseTimeSeriesModel từ chuẩn AGENTS.md.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Lớp cơ sở (nhúng lại từ AGENTS.md để module độc lập)
# ---------------------------------------------------------------------------


class BaseTimeSeriesModel(nn.Module):
    """
    Lớp cơ sở trừu tượng cho tất cả các kiến trúc Deep Learning dự báo PM2.5.

    Mọi mô hình trong `src/models/` bắt buộc kế thừa lớp này để đảm bảo
    tính đồng nhất khi tích hợp vào `training_loop.py` và module XAI.
    Định nghĩa giao diện forward chuẩn: [B, L, D] → [B, H, 1].
    """

    def __init__(self, lookback: int, horizon: int, in_features: int) -> None:
        """
        Khởi tạo thông số cấu trúc không gian đầu vào và đầu ra của mạng.

        Tham số:
            lookback (int): Chiều dài chuỗi lịch sử đầu vào (L = 96 giờ).
            horizon (int): Độ dài khung dự báo tương lai (H ∈ {24, 48, 96, 192}).
            in_features (int): Số lượng đặc trưng đầu vào (đa biến hoặc đơn biến).
        """
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến (Forward pass) dự báo nồng độ PM2.5.

        Tham số:
            x (torch.Tensor): Tensor đầu vào kích thước [Batch, Lookback, In_Features].

        Trả về:
            torch.Tensor: Tensor dự báo kích thước [Batch, Horizon, 1].
        """
        raise NotImplementedError(
            "Các lớp con bắt buộc phải định nghĩa phương thức forward()."
        )


# ---------------------------------------------------------------------------
# Mô hình LSTM Baseline
# ---------------------------------------------------------------------------


class LSTMModel(BaseTimeSeriesModel):
    """
    Mô hình LSTM Baseline đa tầng (Stacked LSTM) dự báo nồng độ PM2.5 đa biến.

    Kiến trúc tổng quan:
        [B, L=96, D]
            └── LSTM (num_layers tầng, hidden_size ẩn, dropout)
                └── Lấy hidden state tại bước cuối: [B, hidden_size]
                    └── Linear → [B, H]
                        └── Reshape → [B, H, 1]

    Hỗ trợ 2 chế độ:
    - `multivariate`: D = số biến khí tượng + mã hóa thời gian (đầy đủ).
    - `univariate`:   D = 1 (chỉ PM2.5 quá khứ, dùng cho Ablation Study).

    Ví dụ sử dụng:
        >>> model = LSTMModel(lookback=96, horizon=24, in_features=15,
        ...                   hidden_size=128, num_layers=2, dropout=0.2)
        >>> x = torch.randn(32, 96, 15)   # [Batch=32, L=96, D=15]
        >>> y = model(x)                  # → [32, 24, 1]
        >>> assert y.shape == (32, 24, 1)
    """

    def __init__(
        self,
        lookback: int = 96,
        horizon: int = 24,
        in_features: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Khởi tạo mô hình LSTM Baseline.

        Tham số:
            lookback (int): Chiều dài chuỗi lịch sử đầu vào L (mặc định 96 giờ).
            horizon (int): Khung dự báo H ∈ {24, 48, 96, 192} (mặc định 24 giờ).
            in_features (int): Số biến đầu vào D (mặc định 15 biến đa biến).
            hidden_size (int): Số đơn vị ẩn trong mỗi tầng LSTM (mặc định 128).
            num_layers (int): Số tầng LSTM xếp chồng (mặc định 2).
            dropout (float): Tỷ lệ dropout giữa các tầng LSTM (mặc định 0.2).
                Lưu ý: dropout chỉ áp dụng khi num_layers > 1.
        """
        super().__init__(lookback=lookback, horizon=horizon, in_features=in_features)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Tầng LSTM đa lớp — xử lý chuỗi thời gian đầu vào [B, L, D]
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Tầng chiếu tuyến tính từ hidden state cuối → dự báo H bước
        self.linear_projection = nn.Linear(hidden_size, horizon)

        # Khởi tạo trọng số theo phân phối đều Xavier
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Khởi tạo trọng số mạng bằng phương pháp Xavier Uniform.

        Áp dụng Xavier cho các ma trận trọng số LSTM (weight_ih, weight_hh)
        và khởi tạo bias = 0. Giúp ổn định gradient khi bắt đầu huấn luyện.
        """
        for param_name, param in self.named_parameters():
            if "weight_ih" in param_name or "weight_hh" in param_name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in param_name:
                nn.init.zeros_(param.data)
                # Forget gate bias = 1.0 giúp LSTM nhớ dài hơn
                hidden_dim = param.data.size(0)
                param.data[hidden_dim // 4 : hidden_dim // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.linear_projection.weight)
        nn.init.zeros_(self.linear_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến (Forward pass) qua LSTM và lớp chiếu tuyến tính.

        Tham số:
            x (torch.Tensor): Tensor đầu vào kích thước [B, L, D].

        Trả về:
            torch.Tensor: Tensor dự báo PM2.5 kích thước [B, H, 1].

        Ngoại lệ:
            ValueError: Nếu chiều cuối cùng của x không khớp in_features.
        """
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Feature dimension mismatch: received {x.size(-1)}, "
                f"expected {self.in_features}."
            )

        # Lan truyền qua LSTM — lstm_out: [B, L, hidden_size]
        lstm_out, _ = self.lstm(x)

        # Lấy trạng thái ẩn tại bước thời gian cuối cùng: [B, hidden_size]
        last_hidden_state = lstm_out[:, -1, :]

        # Chiếu tuyến tính sang không gian dự báo: [B, H]
        predictions = self.linear_projection(last_hidden_state)

        # Thêm chiều kênh đầu ra để khớp chuẩn [B, H, 1]
        return predictions.unsqueeze(-1)

    def count_parameters(self) -> int:
        """
        Đếm tổng số tham số có thể học (trainable parameters) của mô hình.

        Trả về:
            int: Tổng số tham số trainable.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test nội bộ
# ---------------------------------------------------------------------------


def _run_smoke_test() -> None:
    """
    Kiểm tra nhanh tính đúng đắn của LSTMModel với tensor tổng hợp.
    """
    torch.manual_seed(42)
    print("[INFO] Bắt đầu smoke test LSTMModel...")

    batch_size = 4
    lookback = 96
    in_features_multi = 15  # Đa biến
    in_features_uni = 1  # Đơn biến

    for horizon in [24, 48, 96, 192]:
        # Kiểm thử chế độ đa biến
        multi_model = LSTMModel(
            lookback=lookback, horizon=horizon, in_features=in_features_multi
        )
        x_multi = torch.randn(batch_size, lookback, in_features_multi)
        y_multi = multi_model(x_multi)

        assert y_multi.shape == (batch_size, horizon, 1), (
            f"[FAIL] Multivariate H={horizon}: wrong shape {y_multi.shape}"
        )
        print(
            f"[INFO] Multivariate H={horizon:3d}: input {tuple(x_multi.shape)} "
            f"→ output {tuple(y_multi.shape)} | "
            f"Params={multi_model.count_parameters():,} → PASS"
        )

        # Kiểm thử chế độ đơn biến (Ablation)
        uni_model = LSTMModel(
            lookback=lookback, horizon=horizon, in_features=in_features_uni
        )
        x_uni = torch.randn(batch_size, lookback, in_features_uni)
        y_uni = uni_model(x_uni)

        assert y_uni.shape == (batch_size, horizon, 1), (
            f"[FAIL] Univariate H={horizon}: wrong shape {y_uni.shape}"
        )
        print(
            f"[INFO] Univariate   H={horizon:3d}: input {tuple(x_uni.shape)} "
            f"→ output {tuple(y_uni.shape)} | "
            f"Params={uni_model.count_parameters():,} → PASS"
        )

    # Kiểm thử gradient lan truyền ngược (backpropagation)
    test_model = LSTMModel(lookback=lookback, horizon=24, in_features=in_features_multi)
    x_test = torch.randn(batch_size, lookback, in_features_multi, requires_grad=False)
    y_test = test_model(x_test)
    loss = y_test.mean()
    loss.backward()

    for param_name, param in test_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"[FAIL] Gradient None at: {param_name}"
    print("[INFO] Kiểm thử gradient backpropagation: PASS")

    # Kiểm thử ValueError khi kích thước features không khớp
    try:
        err_model = LSTMModel(lookback=96, horizon=24, in_features=15)
        wrong_x = torch.randn(4, 96, 10)
        err_model(wrong_x)
        print("[FAIL] Phải ném ValueError khi sai kích thước features!")
    except ValueError as err:
        print(f"[INFO] Kiểm thử ValueError (features sai): '{err}' → PASS")

    print(
        "[SUCCESS] Toàn bộ smoke test LSTMModel: PASS. Sẵn sàng tích hợp vào training_loop.py."
    )


if __name__ == "__main__":
    _run_smoke_test()
