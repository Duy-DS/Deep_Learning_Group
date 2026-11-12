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
        raise NotImplementedError("Các lớp con bắt buộc phải định nghĩa phương thức forward().")


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
        >>> mo_hinh = LSTMModel(lookback=96, horizon=24, in_features=15,
        ...                     hidden_size=128, num_layers=2, dropout=0.2)
        >>> x = torch.randn(32, 96, 15)   # [Batch=32, L=96, D=15]
        >>> y = mo_hinh(x)                # → [32, 24, 1]
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
        # batch_first=True để chiều batch luôn ở vị trí đầu tiên
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Tầng chiếu tuyến tính từ hidden state cuối → dự báo H bước
        self.lop_chieu_tuyen_tinh = nn.Linear(hidden_size, horizon)

        # Khởi tạo trọng số theo phân phối đều Xavier
        self._khoi_tao_trong_so()

    def _khoi_tao_trong_so(self) -> None:
        """
        Khởi tạo trọng số mạng bằng phương pháp Xavier Uniform.

        Áp dụng Xavier cho các ma trận trọng số LSTM (weight_ih, weight_hh)
        và khởi tạo bias = 0. Giúp ổn định gradient khi bắt đầu huấn luyện.
        """
        for ten, tham_so in self.named_parameters():
            if "weight_ih" in ten or "weight_hh" in ten:
                # Xavier uniform phù hợp với hàm kích hoạt tanh trong LSTM
                nn.init.xavier_uniform_(tham_so.data)
            elif "bias" in ten:
                nn.init.zeros_(tham_so.data)
                # Kỹ thuật "forget gate bias = 1" giúp LSTM nhớ dài hơn
                # Bias LSTM có cấu trúc: [bias_i, bias_f, bias_g, bias_o]
                # bias_f (forget gate) nằm ở vị trí [hidden_size : 2*hidden_size]
                n = tham_so.data.size(0)
                tham_so.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.lop_chieu_tuyen_tinh.weight)
        nn.init.zeros_(self.lop_chieu_tuyen_tinh.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến (Forward pass) qua LSTM và lớp chiếu tuyến tính.

        Luồng tính toán:
            x [B, L, D]
            → LSTM: trạng thái ẩn [B, L, hidden_size], lấy bước cuối cùng
            → hidden_last [B, hidden_size]
            → Linear: [B, H]
            → unsqueeze(-1): [B, H, 1]

        Tham số:
            x (torch.Tensor): Tensor đầu vào kích thước [B, L, D].
                - B: Kích thước batch.
                - L: Chiều dài chuỗi lịch sử (lookback = 96 giờ).
                - D: Số biến đầu vào (in_features).

        Trả về:
            torch.Tensor: Tensor dự báo PM2.5 kích thước [B, H, 1].

        Ngoại lệ:
            ValueError: Nếu chiều cuối cùng của x không khớp in_features.
        """
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Chiều features không khớp: nhận {x.size(-1)}, "
                f"mong đợi {self.in_features}."
            )

        # Lan truyền qua LSTM — lstm_out: [B, L, hidden_size]
        lstm_out, _ = self.lstm(x)

        # Lấy trạng thái ẩn tại bước thời gian cuối cùng: [B, hidden_size]
        trang_thai_cuoi = lstm_out[:, -1, :]

        # Chiếu tuyến tính sang không gian dự báo: [B, H]
        du_bao = self.lop_chieu_tuyen_tinh(trang_thai_cuoi)

        # Thêm chiều kênh đầu ra để khớp chuẩn [B, H, 1]
        return du_bao.unsqueeze(-1)

    def dem_tham_so(self) -> int:
        """
        Đếm tổng số tham số có thể học (trainable parameters) của mô hình.

        Trả về:
            int: Tổng số tham số trainable.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test nội bộ — chạy khi gọi trực tiếp module này
# ---------------------------------------------------------------------------

def _chay_smoke_test() -> None:
    """
    Kiểm tra nhanh tính đúng đắn của LSTMModel với tensor tổng hợp.

    Kiểm tra forward pass không lỗi với tất cả 4 horizons và 2 chế độ
    (multivariate + univariate). Không dùng dữ liệu thật dự án.
    """
    torch.manual_seed(42)
    print("[INFO] Bắt đầu smoke test LSTMModel...")

    BATCH = 4
    LOOKBACK = 96
    IN_FEATURES_MULTI = 15   # Số biến đa biến điển hình (PM2.5 + khí tượng + mã hóa)
    IN_FEATURES_UNI = 1      # Chỉ PM2.5

    for H in [24, 48, 96, 192]:
        # Kiểm thử chế độ đa biến
        mo_hinh_multi = LSTMModel(
            lookback=LOOKBACK, horizon=H, in_features=IN_FEATURES_MULTI
        )
        x_multi = torch.randn(BATCH, LOOKBACK, IN_FEATURES_MULTI)
        y_multi = mo_hinh_multi(x_multi)

        assert y_multi.shape == (BATCH, H, 1), (
            f"[FAIL] Multivariate H={H}: shape sai {y_multi.shape}, mong đợi {(BATCH, H, 1)}"
        )
        print(
            f"[INFO] Multivariate H={H:3d}: input {tuple(x_multi.shape)} "
            f"→ output {tuple(y_multi.shape)} | "
            f"Params={mo_hinh_multi.dem_tham_so():,} → PASS"
        )

        # Kiểm thử chế độ đơn biến (Ablation)
        mo_hinh_uni = LSTMModel(
            lookback=LOOKBACK, horizon=H, in_features=IN_FEATURES_UNI
        )
        x_uni = torch.randn(BATCH, LOOKBACK, IN_FEATURES_UNI)
        y_uni = mo_hinh_uni(x_uni)

        assert y_uni.shape == (BATCH, H, 1), (
            f"[FAIL] Univariate H={H}: shape sai {y_uni.shape}"
        )
        print(
            f"[INFO] Univariate   H={H:3d}: input {tuple(x_uni.shape)} "
            f"→ output {tuple(y_uni.shape)} | "
            f"Params={mo_hinh_uni.dem_tham_so():,} → PASS"
        )

    # Kiểm thử gradient chảy qua mạng (backprop)
    mo_hinh_test = LSTMModel(lookback=LOOKBACK, horizon=24, in_features=IN_FEATURES_MULTI)
    x_test = torch.randn(BATCH, LOOKBACK, IN_FEATURES_MULTI, requires_grad=False)
    y_test = mo_hinh_test(x_test)
    loss = y_test.mean()
    loss.backward()
    # Kiểm tra gradient không bị None sau backward
    for ten, tham_so in mo_hinh_test.named_parameters():
        if tham_so.requires_grad:
            assert tham_so.grad is not None, f"[FAIL] Gradient None tại tham số: {ten}"
    print("[INFO] Kiểm thử gradient backpropagation: PASS")

    # Kiểm thử lỗi khi shape không khớp
    try:
        mo_hinh_err = LSTMModel(lookback=96, horizon=24, in_features=15)
        x_sai = torch.randn(4, 96, 10)   # D=10 ≠ 15
        mo_hinh_err(x_sai)
        print("[FAIL] Phải ném ValueError khi shape sai!")
    except ValueError as loi:
        print(f"[INFO] Kiểm thử ValueError (shape sai): '{loi}' → PASS")

    print("[SUCCESS] Toàn bộ smoke test LSTMModel: PASS. Sẵn sàng tích hợp vào training_loop.py.")


if __name__ == "__main__":
    _chay_smoke_test()

