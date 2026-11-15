"""
Mô-đun định nghĩa kiến trúc mô hình DLinear Baseline dự báo nồng độ PM2.5.

Mô hình dựa trên nghiên cứu:
"Are Transformers Effective for Time Series Forecasting?" (Zeng et al., AAAI 2023).

Nguyên lý: Phân tách chuỗi thời gian đầu vào thành hai thành phần Trend (Xu hướng)
và Seasonal (Mùa vụ/Chu kỳ) bằng tầng Moving Average, sau đó áp dụng hai lớp Linear
riêng biệt để dự báo chuỗi thời gian dài hạn H bước tương lai.

Tuân thủ nghiêm ngặt chuẩn 100% OOP, Pydantic/PyTorch contracts và Docstring Tiếng Việt.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Lớp cơ sở (kế thừa hợp đồng kiến trúc từ AGENTS.md)
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
            horizon (int): Độ dài khung dự báo tương lai (H thuộc {24, 48, 96, 192}).
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
        raise NotImplementedError("Các lớp con bắt buộc phải định nghĩa phương thức forward().")


# ---------------------------------------------------------------------------
# Tầng trích xuất xu hướng bằng trung bình trượt (Moving Average Block)
# ---------------------------------------------------------------------------

class MovingAverageBlock(nn.Module):
    """
    Tầng trích xuất thành phần xu hướng (Trend-Cyclical) qua bộ lọc Moving Average.

    Áp dụng phép tích chập 1D (Average Pooling) với kích thước kernel lẻ.
    Để bảo toàn chiều dài chuỗi L và không làm trễ pha, kỹ thuật đệm (padding)
    sử dụng giá trị điểm đầu và điểm cuối của chuỗi thay vì zero-padding.
    """

    def __init__(self, kernel_size: int = 25) -> None:
        """
        Khởi tạo tầng trung bình trượt.

        Tham số:
            kernel_size (int): Kích thước cửa sổ trượt (bắt buộc phải là số lẻ).
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got: {kernel_size}")

        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện tính trung bình trượt dọc theo trục thời gian L.

        Tham số:
            x (torch.Tensor): Tensor đầu vào [Batch, Lookback, In_Features].

        Trả về:
            torch.Tensor: Tensor thành phần Trend có cùng kích thước [Batch, Lookback, In_Features].
        """
        padding_size = (self.kernel_size - 1) // 2

        front_padding = x[:, 0:1, :].repeat(1, padding_size, 1)
        end_padding = x[:, -1:, :].repeat(1, padding_size, 1)
        x_padded = torch.cat([front_padding, x, end_padding], dim=1)

        x_permuted = x_padded.permute(0, 2, 1)
        trend = self.avg(x_permuted)

        return trend.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Tầng phân tách chuỗi thời gian (Series Decomposition Block)
# ---------------------------------------------------------------------------

class SeriesDecompositionBlock(nn.Module):
    """
    Khối phân tách chuỗi thời gian thành hai thành phần độc lập:
    1. Trend Component (Xu hướng dài hạn): Trích xuất bởi Moving Average.
    2. Seasonal Component (Mùa vụ / Dao động ngắn hạn): Phần dư X - Trend.
    """

    def __init__(self, kernel_size: int = 25) -> None:
        """
        Khởi tạo khối phân tách chuỗi.

        Tham số:
            kernel_size (int): Kích thước cửa sổ Moving Average.
        """
        super().__init__()
        self.moving_avg = MovingAverageBlock(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phân tách tensor đầu vào thành (Seasonal, Trend).

        Tham số:
            x (torch.Tensor): Tensor đầu vào [Batch, Lookback, In_Features].

        Trả về:
            Tuple[torch.Tensor, torch.Tensor]: (seasonal, trend).
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# ---------------------------------------------------------------------------
# Mô hình DLinear Baseline chính
# ---------------------------------------------------------------------------

class DLinearModel(BaseTimeSeriesModel):
    """
    Mô hình DLinear Baseline — Dự báo chuỗi thời gian bằng phân tách Trend-Seasonal.

    Kiến trúc tổng quan:
        [B, L=96, D]
            │
            ▼ (Series Decomposition)
        ┌───────────────────────┴───────────────────────┐
        ▼ (Seasonal)                                    ▼ (Trend)
      [B, L, D]                                       [B, L, D]
        │ permute                                       │ permute
      [B, D, L]                                       [B, D, L]
        │ Linear (L → H)                                │ Linear (L → H)
      [B, D, H]                                       [B, D, H]
        │ permute                                       │ permute
      [B, H, D]                                       [B, H, D]
        └───────────────────────┬───────────────────────┘
                                ▼ (Cộng thành phần)
                            [B, H, D]
                                │ (Chiếu đặc trưng nếu D > 1)
                                ▼
                            [B, H, 1] (Dự báo nồng độ PM2.5)

    Hỗ trợ 2 chế độ:
    - `multivariate`: D = số lượng biến đầy đủ (PM2.5 + khí tượng + mã hóa thời gian).
    - `univariate`: D = 1 (chỉ nồng độ PM2.5 quá khứ, phục vụ Ablation Study RQ3).
    """

    def __init__(
        self,
        lookback: int = 96,
        horizon: int = 24,
        in_features: int = 15,
        kernel_size: int = 25,
    ) -> None:
        """
        Khởi tạo mô hình DLinear.

        Tham số:
            lookback (int): Chiều dài chuỗi lịch sử L (mặc định 96 giờ).
            horizon (int): Khung dự báo H ∈ {24, 48, 96, 192} (mặc định 24 giờ).
            in_features (int): Số lượng biến đầu vào D (mặc định 15 biến).
            kernel_size (int): Kích thước kernel Moving Average (mặc định 25).
        """
        super().__init__(lookback=lookback, horizon=horizon, in_features=in_features)

        self.kernel_size = kernel_size

        # Khối phân tách chuỗi thành Seasonal và Trend
        self.decomposition = SeriesDecompositionBlock(kernel_size=kernel_size)

        # Hai lớp ánh xạ tuyến tính độc lập theo trục thời gian: L → H
        self.linear_seasonal = nn.Linear(lookback, horizon)
        self.linear_trend = nn.Linear(lookback, horizon)

        # Lớp chiếu từ D đặc trưng về 1 kênh mục tiêu PM2.5
        if in_features > 1:
            self.projection_pm25 = nn.Linear(in_features, 1)
        else:
            self.projection_pm25 = nn.Identity()

        # Khởi tạo trọng số
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Khởi tạo trọng số các tầng Linear theo phân phối Xavier Uniform.
        """
        nn.init.xavier_uniform_(self.linear_seasonal.weight)
        nn.init.zeros_(self.linear_seasonal.bias)

        nn.init.xavier_uniform_(self.linear_trend.weight)
        nn.init.zeros_(self.linear_trend.bias)

        if isinstance(self.projection_pm25, nn.Linear):
            nn.init.xavier_uniform_(self.projection_pm25.weight)
            nn.init.zeros_(self.projection_pm25.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến (Forward pass) dự báo nồng độ PM2.5.

        Tham số:
            x (torch.Tensor): Tensor đầu vào kích thước [Batch, Lookback, In_Features].

        Trả về:
            torch.Tensor: Tensor dự báo kích thước [Batch, Horizon, 1].

        Ngoại lệ:
            ValueError: Nếu kích thước đặc trưng cuối của x khác in_features.
        """
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Feature dimension mismatch: received {x.size(-1)}, "
                f"expected {self.in_features}."
            )

        # 1. Phân tách Seasonal và Trend: [B, L, D]
        seasonal, trend = self.decomposition(x)

        # 2. Đổi chiều [B, L, D] → [B, D, L]
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        # 3. Dự báo độc lập hai thành phần sang khung H: [B, D, H]
        seasonal_output = self.linear_seasonal(seasonal)
        trend_output = self.linear_trend(trend)

        # 4. Cộng hai thành phần dự báo: [B, D, H]
        combined = seasonal_output + trend_output

        # 5. Đổi lại chiều [B, D, H] → [B, H, D]
        combined = combined.permute(0, 2, 1)

        # 6. Chiếu từ D biến về 1 kênh nồng độ PM2.5: [B, H, 1]
        predictions = self.projection_pm25(combined)

        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(-1)

        return predictions

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
    Kiểm tra nhanh tính đúng đắn của DLinearModel với tensor tổng hợp.
    """
    torch.manual_seed(42)
    print("[INFO] Bắt đầu smoke test DLinearModel...")

    batch_size = 4
    lookback = 96
    in_features_multi = 15   # Đa biến
    in_features_uni = 1      # Đơn biến

    for horizon in [24, 48, 96, 192]:
        # Kiểm thử chế độ đa biến
        multi_model = DLinearModel(
            lookback=lookback, horizon=horizon, in_features=in_features_multi, kernel_size=25
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
        uni_model = DLinearModel(
            lookback=lookback, horizon=horizon, in_features=in_features_uni, kernel_size=25
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

    # Kiểm thử lan truyền gradient (backpropagation)
    test_model = DLinearModel(lookback=lookback, horizon=24, in_features=in_features_multi)
    x_test = torch.randn(batch_size, lookback, in_features_multi)
    y_test = test_model(x_test)
    loss = y_test.mean()
    loss.backward()

    for param_name, param in test_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"[FAIL] Gradient None at: {param_name}"
    print("[INFO] Kiểm thử gradient backpropagation: PASS")

    # Kiểm thử bẫy lỗi kernel_size chẵn
    try:
        DLinearModel(lookback=96, horizon=24, in_features=15, kernel_size=24)
        print("[FAIL] Phải ném ValueError khi kernel_size là số chẵn!")
    except ValueError as err:
        print(f"[INFO] Kiểm thử ValueError (kernel_size chẵn): '{err}' → PASS")

    # Kiểm thử bẫy lỗi sai số chiều features
    try:
        err_model = DLinearModel(lookback=96, horizon=24, in_features=15)
        err_model(torch.randn(4, 96, 8))
        print("[FAIL] Phải ném ValueError khi in_features không khớp!")
    except ValueError as err:
        print(f"[INFO] Kiểm thử ValueError (features sai): '{err}' → PASS")

    print("[SUCCESS] Toàn bộ smoke test DLinearModel: PASS. Sẵn sàng tích hợp vào training_loop.py.")


if __name__ == "__main__":
    _run_smoke_test()
