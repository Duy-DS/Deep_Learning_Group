"""
Mô-đun định nghĩa kiến trúc mô hình PatchTST cải tiến cho dự báo nồng độ PM2.5.

Cải tiến cốt lõi so với bản WP03 cũ (khắc phục lỗi phẳng đường dự báo & underfitting):
- Kế thừa `BaseTimeSeriesModel` và tuân thủ tuyệt đối hợp đồng `[B, L, D] -> [B, H, 1]`.
- `RevIN` chọn lọc (`num_continuous_features=11`, `min_std=0.1`): chỉ chuẩn hóa các kênh
  vật lý/ô nhiễm liên tục, KHÔNG chuẩn hóa các kênh chu kỳ thời gian (`hour/month_sin/cos`)
  và 16 kênh One-Hot hướng gió (`wd_*`) — bảo toàn tín hiệu mùa Đông/Hè và triệt tiêu hiện
  tượng nổ phương sai trên kênh thưa (`rain`, `wd_*`).
- Cơ chế kết hợp đa biến ở mức Patch (`ExogenousPatchFusion`): giữ nhánh Channel-Independent
  Transformer trên các kênh liên tục (`C_ci`) để tiết kiệm ~2.8x VRAM & thời gian, đồng thời
  hợp nhất ngữ cảnh khí tượng + hướng gió + thời gian (trung bình, giá trị cuối và độ dốc
  từng patch) vào kênh mục tiêu qua cổng phần dư (Gated Residual Fusion).
- Đầu dự báo 2 nhánh (`head_linear` + `head_mlp`) giúp bảo toàn xu hướng tuyến tính đồng thời
  tái tạo tốt biên độ phi tuyến của các đỉnh ô nhiễm (Smog Spikes).
- Tương thích 100% xuất ONNX `opset=17` (sử dụng chiều dương trong `unfold`).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTimeSeriesModel(nn.Module):
    """
    Lớp cơ sở trừu tượng cho tất cả các kiến trúc Deep Learning dự báo PM2.5.

    Định nghĩa giao diện lan truyền tiến chuẩn `[Batch, Lookback, In_Features] -> [Batch, Horizon, 1]`.
    """

    def __init__(self, lookback: int, horizon: int, in_features: int) -> None:
        """
        Khởi tạo kích thước không gian đầu vào và đầu ra.

        Tham số:
            lookback (int): Chiều dài chuỗi lịch sử đầu vào (L = 96 giờ).
            horizon (int): Tầm dự báo tương lai H ∈ {24, 48, 96, 192}.
            in_features (int): Số lượng đặc trưng đầu vào D (đa biến hoặc đơn biến).
        """
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền tiến từ chuỗi quá khứ `[B, L, D]` sang dự báo `[B, H, 1]`."""
        raise NotImplementedError(
            "Các lớp con bắt buộc phải cài đặt phương thức forward()."
        )


class RevIN(nn.Module):
    """
    Reversible Instance Normalization chọn lọc theo kênh liên tục.

    Chỉ chuẩn hóa `num_cont` kênh vật lý/ô nhiễm đầu tiên với chặn dưới độ lệch chuẩn
    `min_std` để tránh khuếch đại nhiễu ở các cửa sổ không mưa hoặc ít biến động,
    đồng thời giữ nguyên các kênh chu kỳ thời gian và hướng gió One-Hot.
    """

    def __init__(
        self,
        num_features: int,
        num_continuous_features: Optional[int] = None,
        eps: float = 1e-5,
        min_std: float = 0.1,
        affine: bool = True,
        target_idx: int = 0,
    ) -> None:
        """
        Khởi tạo lớp chuẩn hóa cục bộ RevIN chọn lọc.

        Tham số:
            num_features (int): Tổng số đặc trưng đầu vào D.
            num_continuous_features (Optional[int]): Số kênh vật lý liên tục đầu tiên cần chuẩn hóa.
            eps (float): Hằng số ổn định số học khi tính căn bậc hai.
            min_std (float): Ngưỡng chặn dưới độ lệch chuẩn tránh chia cho 0 trên kênh thưa.
            affine (bool): Bật tham số co giãn/tịnh tiến học được cho các kênh liên tục.
            target_idx (int): Chỉ số cột mục tiêu PM2.5 (mặc định 0).
        """
        super().__init__()
        self.num_features = int(num_features)
        if num_continuous_features is None:
            self.num_cont = 11 if self.num_features >= 15 else self.num_features
        else:
            self.num_cont = min(int(num_continuous_features), self.num_features)
        self.eps = float(eps)
        self.min_std = float(min_std)
        self.affine = bool(affine)
        self.target_idx = int(target_idx)

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_cont))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_cont))

    def normalize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Chuẩn hóa `num_cont` kênh liên tục của tensor đầu vào `[B, L, D]`.

        Trả về:
            Tuple `(x_out [B, L, D], mean [B, 1, C_cont], stdev [B, 1, C_cont])`.
        """
        x_cont = x[:, :, : self.num_cont]
        mean = x_cont.mean(dim=1, keepdim=True).detach()
        stdev = (
            torch.sqrt(
                torch.var(x_cont, dim=1, keepdim=True, unbiased=False) + self.eps
            )
            .clamp(min=self.min_std)
            .detach()
        )
        x_cont_norm = (x_cont - mean) / stdev
        if self.affine:
            x_cont_norm = x_cont_norm * self.affine_weight + self.affine_bias

        if self.num_cont < x.size(-1):
            x_out = torch.cat([x_cont_norm, x[:, :, self.num_cont :]], dim=-1)
        else:
            x_out = x_cont_norm
        return x_out, mean, stdev

    def denormalize_target(
        self, y_norm: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor
    ) -> torch.Tensor:
        """
        Giải chuẩn hóa dự báo kênh mục tiêu `y_norm [B, H]` về lại thang đo gốc của cửa sổ
        (khôi phục đầy đủ cả độ lệch chuẩn `stdev` lẫn trung bình `mean`).
        """
        idx = self.target_idx
        if self.affine:
            weight = self.affine_weight[:, :, idx]
            safe_weight = torch.where(
                weight >= 0,
                weight.clamp(min=self.eps),
                weight.clamp(max=-self.eps),
            )
            y_norm = (y_norm - self.affine_bias[:, :, idx]) / safe_weight
        return y_norm * stdev[:, :, idx] + mean[:, :, idx]


class FlashTransformerEncoderLayer(nn.Module):
    """
    Tầng Pre-LN Transformer Encoder dùng `F.scaled_dot_product_attention`
    tối ưu tốc độ GPU và tương thích xuất ONNX `opset=17`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.2,
    ) -> None:
        """
        Khởi tạo tầng Transformer Encoder chuẩn Pre-LayerNorm.

        Tham số:
            d_model (int): Chiều biểu diễn ẩn của mỗi patch token.
            n_heads (int): Số đầu chú ý (Attention heads).
            d_ff (int): Chiều tầng ẩn Feed-Forward Network.
            dropout (float): Tỷ lệ dropout điều chuẩn.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) phải chia hết cho n_heads ({n_heads})."
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.resid_dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền tiến trên chuỗi patch `[B_tot, N_patches, d_model]`."""
        b_tot, n_patches, d_m = x.shape

        h = self.norm1(x)
        qkv = self.qkv_proj(h).reshape(b_tot, n_patches, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_tot, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(b_tot, n_patches, d_m)
        x = x + self.resid_dropout1(self.out_proj(attn_out))
        x = x + self.ffn(self.norm2(x))
        return x


class ExogenousPatchFusion(nn.Module):
    """
    Khối hợp nhất ngữ cảnh đa biến ở mức Patch cho kênh mục tiêu PM2.5.

    Kết hợp hai nguồn thông tin ngoại sinh tại từng khung patch:
    1. Biểu diễn đã qua Transformer của các kênh vật lý liên tục (`pm10`, `co`, `no2`, `temp`, `dewp`, `wspm`...).
    2. Thống kê nội patch `[mean, last, delta]` của toàn bộ `D` đặc trưng (gồm cả `hour/month_sin/cos`
       và 16 hướng gió `wd_*`) để nhận diện tín hiệu tích tụ/phân tán bụi mịn theo mùa và hướng gió.
    """

    def __init__(
        self,
        in_features: int,
        num_cont: int,
        d_model: int,
        dropout: float = 0.2,
    ) -> None:
        """
        Khởi tạo khối hợp nhất thông tin ngoại sinh mức Patch.

        Tham số:
            in_features (int): Tổng số đặc trưng đầu vào D.
            num_cont (int): Số kênh vật lý liên tục đi qua backbone Channel-Independent.
            d_model (int): Chiều biểu diễn patch token.
            dropout (float): Tỷ lệ dropout điều chuẩn.
        """
        super().__init__()
        self.phys_channel_mixer = nn.Linear(num_cont, 1, bias=False)
        nn.init.normal_(self.phys_channel_mixer.weight, mean=0.0, std=0.05)

        self.exo_patch_encoder = nn.Sequential(
            nn.Linear(3 * in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.exo_gate = nn.Linear(2 * d_model, d_model)
        # Khởi tạo bias âm nhẹ để ban đầu ưu tiên kênh mục tiêu nhưng vẫn mở luồng gradient ngay lập tức
        nn.init.constant_(self.exo_gate.bias, -1.0)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_dropout = nn.Dropout(dropout)

    def forward(
        self,
        z_target: torch.Tensor,
        z_phys: torch.Tensor,
        exo_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hợp nhất ngữ cảnh đa biến vào chuỗi patch của kênh mục tiêu PM2.5.

        Tham số:
            z_target (torch.Tensor): Biểu diễn kênh mục tiêu `[B, N, d_model]`.
            z_phys (torch.Tensor): Biểu diễn toàn bộ kênh liên tục `[B, C_cont, N, d_model]`.
            exo_patches (torch.Tensor): Các patch của toàn bộ D kênh `[B, D, N, patch_len]`.

        Trả về:
            torch.Tensor: Biểu diễn đã làm giàu ngữ cảnh khí tượng `[B, N, d_model]`.
        """
        # Trộn biểu diễn các kênh vật lý liên tục: (B, C_cont, N, d_model) -> (B, N, d_model)
        phys_ctx = self.phys_channel_mixer(
            z_phys.permute(0, 2, 3, 1)
        ).squeeze(-1)

        # Trích thống kê [trung bình, điểm cuối, độ dốc] trong từng patch cho toàn bộ D kênh
        p_mean = exo_patches.mean(dim=-1)  # (B, D, N)
        p_last = exo_patches[:, :, :, -1]  # (B, D, N)
        p_delta = exo_patches[:, :, :, -1] - exo_patches[:, :, :, 0]  # (B, D, N)
        exo_stats = torch.cat([p_mean, p_last, p_delta], dim=1).permute(
            0, 2, 1
        )  # (B, N, 3*D)

        exo_ctx = self.exo_patch_encoder(exo_stats) + phys_ctx
        gate = torch.sigmoid(self.exo_gate(torch.cat([z_target, exo_ctx], dim=-1)))
        fused = self.fusion_norm(z_target + self.fusion_dropout(gate * exo_ctx))
        return fused


class PatchTST(BaseTimeSeriesModel):
    """
    Kiến trúc PatchTST cải tiến cho dự báo nồng độ PM2.5 đa biến và đơn biến.

    - Chế độ `multivariate` (`D=31` hoặc `D=15`): chạy Channel-Independent Transformer trên
      11 kênh vật lý liên tục, kết hợp ngữ cảnh khí tượng + hướng gió + chu kỳ thời gian
      qua `ExogenousPatchFusion`, rồi dự báo `H` bước bằng đầu 2 nhánh Linear + MLP.
    - Chế độ `univariate` (`D=1`): chạy thuần Channel-Independent PatchTST trên kênh PM2.5.
    """

    def __init__(
        self,
        lookback: int = 96,
        horizon: int = 24,
        in_features: int = 15,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        target_idx: int = 0,
        num_continuous_features: Optional[int] = None,
        use_revin: bool = False,
        min_std: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Khởi tạo mô hình PatchTST.

        Tham số:
            lookback (int): Độ dài lịch sử đầu vào L (mặc định 96 giờ).
            horizon (int): Tầm dự báo H ∈ {24, 48, 96, 192}.
            in_features (int): Số lượng đặc trưng đầu vào D.
            patch_len (int): Độ dài mỗi đoạn patch (mặc định 12 giờ — sắc nét cho đỉnh smog).
            stride (int): Bước trượt giữa các patch (mặc định 6 giờ — chồng lấn 50%).
            d_model (int): Chiều nhúng của mỗi patch token (mặc định 128).
            n_heads (int): Số đầu Multi-Head Attention (mặc định 4).
            e_layers (int): Số tầng Transformer Encoder (mặc định 3).
            d_ff (int): Chiều tầng ẩn FFN (mặc định 256).
            dropout (float): Tỷ lệ dropout điều chuẩn (mặc định 0.2).
            target_idx (int): Chỉ số kênh mục tiêu PM2.5 (mặc định 0).
            num_continuous_features (Optional[int]): Số kênh vật lý liên tục đầu tiên.
            use_revin (bool): Tùy chọn bật RevIN chọn lọc (mặc định False vì dữ liệu PM2.5 đã chuẩn hóa
                StandardScaler toàn cục và có tính hồi quy về trung bình mạnh ở tầm dự báo xa H=96/192).
            min_std (float): Ngưỡng chặn dưới độ lệch chuẩn của RevIN khi bật (mặc định 0.1).
        """
        lookback = int(kwargs.get("seq_len", lookback))
        horizon = int(kwargs.get("pred_len", horizon))
        in_features = int(kwargs.get("enc_in", kwargs.get("c_in", in_features)))
        super().__init__(lookback=lookback, horizon=horizon, in_features=in_features)

        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.target_idx = int(target_idx)
        self.use_revin = bool(use_revin)
        if num_continuous_features is None:
            self.num_cont = 11 if in_features >= 15 else in_features
        else:
            self.num_cont = min(int(num_continuous_features), in_features)

        if self.use_revin:
            self.revin = RevIN(
                num_features=in_features,
                num_continuous_features=self.num_cont,
                min_std=min_std,
                affine=True,
                target_idx=target_idx,
            )

        # Tính số lượng patch và độ dài đệm replication ở cuối chuỗi (nếu cần)
        if lookback < self.patch_len:
            self.pad_len = self.patch_len - lookback
        else:
            rem = (lookback - self.patch_len) % self.stride
            self.pad_len = (self.stride - rem) if rem != 0 else 0

        padded_len = lookback + self.pad_len
        self.num_patches = (padded_len - self.patch_len) // self.stride + 1

        # Nhúng Patch -> d_model + Positional Embedding học được
        self.patch_embedding = nn.Linear(self.patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

        # Stack các tầng FlashTransformerEncoder (Pre-LN)
        self.encoder_layers = nn.ModuleList(
            [
                FlashTransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # Khối hợp nhất ngữ cảnh ngoại sinh mức Patch (chỉ kích hoạt khi D > 1)
        if in_features > 1:
            self.exo_fusion: Optional[ExogenousPatchFusion] = ExogenousPatchFusion(
                in_features=in_features,
                num_cont=self.num_cont,
                d_model=d_model,
                dropout=dropout,
            )
        else:
            self.exo_fusion = None

        # Đầu dự báo 2 nhánh (đặt tên chứa 'head' để fine_tune.py tự nhận diện mở khóa)
        flat_dim = self.num_patches * d_model
        head_hidden = max(d_model, min(256, horizon * 2))
        self.head_dropout = nn.Dropout(dropout)
        self.head_linear = nn.Linear(flat_dim, horizon)
        self.head_mlp = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, horizon),
        )

    def _pad_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Đệm lặp giá trị bước cuối dọc trục thời gian cuối cùng nếu cần."""
        if self.pad_len <= 0:
            return x_seq
        last_val = x_seq[..., -1:].expand(*x_seq.shape[:-1], self.pad_len)
        return torch.cat([x_seq, last_val], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến dự báo nồng độ PM2.5.

        Tham số:
            x (torch.Tensor): Tensor đầu vào kích thước `[B, lookback, in_features]`.

        Trả về:
            torch.Tensor: Tensor dự báo kích thước `[B, horizon, 1]`.
        """
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Lệch số đặc trưng đầu vào: nhận {x.size(-1)}, kỳ vọng {self.in_features}."
            )

        b_size, seq_len, _ = x.shape
        if self.use_revin:
            x_norm, mean, stdev = self.revin.normalize(x)
        else:
            x_norm = x
            mean = stdev = None

        # 1. Tách nhóm kênh liên tục cho Channel-Independent Transformer: (B * C_cont, L)
        x_cont = x_norm[:, :, : self.num_cont]
        x_ci = x_cont.permute(0, 2, 1).contiguous().reshape(b_size * self.num_cont, seq_len)
        x_ci = self._pad_sequence(x_ci)

        # Dùng chiều dương (dimension=1) để tương thích nguyên bản ONNX opset=17
        patches = x_ci.unfold(dimension=1, size=self.patch_len, step=self.stride)

        z = self.patch_embedding(patches) + self.pos_embedding
        z = self.embed_dropout(z)
        for layer in self.encoder_layers:
            z = layer(z)
        z = self.encoder_norm(z)

        # Khôi phục chiều kênh liên tục: (B, C_cont, N_patches, d_model)
        z_phys = z.reshape(b_size, self.num_cont, self.num_patches, self.d_model)
        z_target = z_phys[:, self.target_idx, :, :]  # (B, N_patches, d_model)

        # 2. Hợp nhất ngữ cảnh khí tượng, hướng gió & thời gian ở mức Patch (khi đa biến)
        if self.exo_fusion is not None:
            x_all_ch = self._pad_sequence(x_norm.permute(0, 2, 1).contiguous())
            exo_patches = x_all_ch.unfold(
                dimension=2, size=self.patch_len, step=self.stride
            )  # (B, D, N_patches, patch_len)
            z_fused = self.exo_fusion(
                z_target=z_target, z_phys=z_phys, exo_patches=exo_patches
            )
        else:
            z_fused = z_target

        # 3. Trải phẳng và dự báo H bước bằng đầu 2 nhánh Linear + MLP
        z_flat = z_fused.reshape(b_size, -1)
        out = self.head_linear(self.head_dropout(z_flat)) + self.head_mlp(z_flat)

        # 4. Giải chuẩn hóa RevIN về lại đúng thang đo vật lý chuẩn hóa của cửa sổ
        if self.use_revin and mean is not None and stdev is not None:
            out = self.revin.denormalize_target(out, mean, stdev)

        return out.unsqueeze(-1)

    def count_parameters(self) -> int:
        """Đếm tổng số tham số có thể học (trainable parameters) của mô hình."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Bí danh đảm bảo tương thích với mọi module gọi PatchTSTModel
PatchTSTModel = PatchTST
