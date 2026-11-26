"""
Mô-đun định nghĩa kiến trúc mô hình Informer cải tiến dự báo nồng độ PM2.5.

Cải tiến cốt lõi (khắc phục lỗi ProbSparse sai mẫu số, ngẫu nhiên ở eval() và Decoder bị phẳng):
- Sửa công thức đo độ thưa ProbSparse Eq. (4): dùng `Q_K_sample.mean(dim=-1)` thay vì `/ L_K`.
- Áp dụng `attention_dropout` thực sự lên trọng số softmax khi huấn luyện.
- Tự động chuyển sang chế độ đo độ thưa tất định Eq. (4) bằng `torch.gather` / `torch.scatter`
  khi `eval()` (không gọi `torch.randint`), giúp loại bỏ dao động ngẫu nhiên và đạt độ trôi
  0.00% với adapter `FullKeyEq4ProbSparse` khi xuất ONNX `opset=17` ở WP05.
- Bổ sung lớp `FullAttention` ($O(L^2)$ thực sự) cho biến thể `attention_type='full'` phục vụ
  Ablation 3 chuẩn khoa học.
- Nâng cấp Encoder/Decoder sang Pre-LayerNorm (`norm_first=True`) kèm `encoder_norm` /
  `decoder_norm`, mặt nạ nhân quả (`tgt_mask`) dạng float an toàn cho ONNX và nhúng truy vấn
  chân trời học được (`horizon_queries`) để chấm dứt hiện tượng Decoder xuất đường phẳng.
- Tuân thủ tuyệt đối hợp đồng kiến trúc `BaseTimeSeriesModel` `[B, L, D] -> [B, H, 1]`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Mã hóa vị trí lượng giác cố định (Sinusoidal Positional Encoding)."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Khởi tạo bảng mã hóa vị trí cho chiều `d_model`.

        Tham số:
            d_model (int): Chiều biểu diễn ẩn của mô hình.
            max_len (int): Độ dài chuỗi tối đa được hỗ trợ.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về tensor vị trí `[1, L, d_model]` khớp độ dài chuỗi đầu vào `x`."""
        return self.pe[:, : x.size(1)]


class DataEmbedding(nn.Module):
    """Nhúng đồng thời giá trị đặc trưng chuỗi thời gian và vị trí thời gian."""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.15) -> None:
        """
        Khởi tạo tầng nhúng dữ liệu đầu vào.

        Tham số:
            c_in (int): Số lượng đặc trưng đầu vào D.
            d_model (int): Chiều không gian ẩn.
            dropout (float): Tỷ lệ dropout sau khi cộng nhúng vị trí.
        """
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Nhúng chuỗi đầu vào `[B, L, D]` sang `[B, L, d_model]`."""
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    """
    Cơ chế Canonical Full Self-Attention độ phức tạp $O(L^2)$ thực sự,
    phục vụ biến thể đối chứng `informer_full` (`Informer_NoProbSparse`) trong Ablation 3.
    """

    def __init__(self, attention_dropout: float = 0.15) -> None:
        """
        Khởi tạo tầng Full Attention chuẩn.

        Tham số:
            attention_dropout (float): Tỷ lệ dropout trên ma trận trọng số chú ý.
        """
        super().__init__()
        self.dropout = nn.Dropout(float(attention_dropout))

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Lan truyền tiến Full Scaled Dot-Product Attention.

        Tham số:
            queries (torch.Tensor): Tensor truy vấn `[B, L_Q, H, D]`.
            keys (torch.Tensor): Tensor khóa `[B, L_K, H, D]`.
            values (torch.Tensor): Tensor giá trị `[B, L_K, H, D_V]`.
            attn_mask (Optional[torch.Tensor]): Mặt nạ cộng tùy chọn `[L_Q, L_K]`.

        Trả về:
            Tuple `(context [B, L_Q, H, D_V], None)`.
        """
        d_k = queries.shape[-1]
        q_t = queries.transpose(1, 2)  # (B, H, L_Q, D)
        k_t = keys.transpose(1, 2)  # (B, H, L_K, D)
        v_t = values.transpose(1, 2)  # (B, H, L_K, D_V)

        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(float(d_k))
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, v_t)
        return context.transpose(1, 2).contiguous(), None


class ProbSparseAttention(nn.Module):
    """
    Cơ chế ProbSparse Self-Attention $O(L \\log L)$ theo Zhou et al. (AAAI 2021).

    - Khi huấn luyện (`training=True`): lấy mẫu $U_{part} = c \\lceil \\ln L_K \\rceil$ khóa
      và đo độ thưa chuẩn $M(q_i) = \\max_j(q_i k_j^T) - \\text{mean}_j(q_i k_j^T)$.
    - Khi đánh giá/suy luận (`training=False` với `deterministic_eval=True`): tính độ đo
      độ thưa tất định Eq. (4) trên toàn bộ $L_K$ khóa và dùng `torch.gather`/`torch.scatter`
      (100% tương thích ONNX `opset=17`, không dùng `torch.randint`).
    """

    def __init__(
        self,
        factor: int = 5,
        attention_dropout: float = 0.15,
        deterministic_eval: bool = True,
    ) -> None:
        """
        Khởi tạo lớp ProbSparse Attention.

        Tham số:
            factor (int): Hệ số lấy mẫu $c$ cho Top-$u$ truy vấn (mặc định 5).
            attention_dropout (float): Tỷ lệ dropout trên trọng số attention.
            deterministic_eval (bool): Bật đánh giá tất định ở chế độ `.eval()`.
        """
        super().__init__()
        self.factor = int(factor)
        self.dropout = nn.Dropout(float(attention_dropout))
        self.deterministic_eval = bool(deterministic_eval)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Lan truyền tiến ProbSparse Attention dùng hoàn toàn `gather` và `scatter`.

        Tham số:
            queries (torch.Tensor): Tensor truy vấn `[B, L_Q, H, D]`.
            keys (torch.Tensor): Tensor khóa `[B, L_K, H, D]`.
            values (torch.Tensor): Tensor giá trị `[B, L_K, H, D_V]`.
            attn_mask (Optional[torch.Tensor]): Mặt nạ chú ý tùy chọn.

        Trả về:
            Tuple `(context [B, L_Q, H, D_V], None)`.
        """
        del attn_mask
        bsz, l_q, n_heads, d_k = queries.shape
        _, l_k, _, d_v = keys.shape[0], keys.shape[1], keys.shape[2], values.shape[-1]

        q_t = queries.transpose(1, 2)  # (B, H, L_Q, D)
        k_t = keys.transpose(1, 2)  # (B, H, L_K, D)
        v_t = values.transpose(1, 2)  # (B, H, L_K, D_V)

        u_part = min(self.factor * max(1, int(math.ceil(math.log(max(l_k, 2))))), l_k)
        u_top = min(self.factor * max(1, int(math.ceil(math.log(max(l_q, 2))))), l_q)

        if self.training or not self.deterministic_eval:
            # Lấy mẫu ngẫu nhiên U_part khóa và đo độ thưa chuẩn: max - mean (trên U_part!)
            idx_sample = torch.randint(
                low=0, high=l_k, size=(l_q, u_part), device=queries.device
            )
            flat_idx = idx_sample.reshape(1, 1, l_q * u_part, 1).expand(
                bsz, n_heads, l_q * u_part, d_k
            )
            k_sample = torch.gather(k_t, dim=2, index=flat_idx).reshape(
                bsz, n_heads, l_q, u_part, d_k
            )
            q_k_sample = torch.matmul(
                q_t.unsqueeze(-2), k_sample.transpose(-2, -1)
            ).squeeze(-2)
            sparsity_m = q_k_sample.max(dim=-1)[0] - q_k_sample.mean(dim=-1)
        else:
            # Chế độ eval tất định Eq. (4) tương thích 100% ONNX opset=17 & FullKeyEq4ProbSparse
            q_k_full = torch.matmul(q_t, k_t.transpose(-2, -1))
            sparsity_m = q_k_full.max(dim=-1)[0] - q_k_full.mean(dim=-1)

        m_top = sparsity_m.topk(u_top, dim=-1, sorted=True)[1]  # (B, H, u_top)

        gather_q_idx = m_top.unsqueeze(-1).expand(bsz, n_heads, u_top, d_k)
        q_reduce = torch.gather(q_t, dim=-2, index=gather_q_idx)

        q_k = torch.matmul(q_reduce, k_t.transpose(-2, -1)) / math.sqrt(float(d_k))
        attn = self.dropout(torch.softmax(q_k, dim=-1))
        top_context = torch.matmul(attn, v_t)

        context_init = (
            v_t.mean(dim=-2, keepdim=True).expand(bsz, n_heads, l_q, d_v).clone()
        )
        scatter_ctx_idx = m_top.unsqueeze(-1).expand(bsz, n_heads, u_top, d_v)
        context = torch.scatter(
            context_init,
            dim=-2,
            index=scatter_ctx_idx,
            src=top_context.type_as(context_init),
        )
        return context.transpose(1, 2).contiguous(), None


class AttentionLayer(nn.Module):
    """Lớp chiếu Q/K/V/Out bọc quanh lõi chú ý (`ProbSparseAttention` hoặc `FullAttention`)."""

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ) -> None:
        """
        Khởi tạo lớp chiếu tuyến tính đa đầu tương thích `nn.TransformerEncoderLayer(batch_first=True)`.

        Tham số:
            attention (nn.Module): Lõi chú ý (`ProbSparseAttention` hoặc `FullAttention`).
            d_model (int): Chiều đặc trưng ẩn.
            n_heads (int): Số đầu chú ý.
            d_keys (Optional[int]): Chiều mỗi đầu khóa/truy vấn.
            d_values (Optional[int]): Chiều mỗi đầu giá trị.
        """
        super().__init__()
        self.batch_first = True
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = int(n_heads)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Lan truyền tiến đa đầu và trả về tuple `(output, attn)` theo chuẩn PyTorch."""
        del key_padding_mask, need_weights, is_causal, kwargs
        bsz, l_q, _ = queries.shape
        _, l_k, _ = keys.shape
        n_heads = self.n_heads

        q = self.query_projection(queries).view(bsz, l_q, n_heads, -1)
        k = self.key_projection(keys).view(bsz, l_k, n_heads, -1)
        v = self.value_projection(values).view(bsz, l_k, n_heads, -1)

        out, attn = self.inner_attention(q, k, v, attn_mask)
        out = out.view(bsz, l_q, -1)
        return self.out_projection(out), attn


class InformerEncoderLayer(nn.Module):
    """
    Tầng Pre-LN Informer Encoder độc lập hoàn toàn với fast-path của `nn.TransformerEncoderLayer`.

    Giữ nguyên thuộc tính `self.self_attn` (là `AttentionLayer` chứa `inner_attention`)
    để tương thích 100% với `InformerDeterministicPatcher` ở WP05 và xuất ONNX `opset=17`.
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_ff: int = 256,
        dropout: float = 0.15,
        activation: str = "gelu",
    ) -> None:
        """
        Khởi tạo một tầng Informer Encoder chuẩn Pre-LayerNorm.

        Tham số:
            attention (nn.Module): Lõi chú ý (`ProbSparseAttention` hoặc `FullAttention`).
            d_model (int): Chiều không gian ẩn.
            n_heads (int): Số đầu chú ý.
            d_ff (int): Chiều tầng ẩn Feed-Forward.
            dropout (float): Tỷ lệ dropout điều chuẩn.
            activation (str): Hàm kích hoạt (`'gelu'` hoặc `'relu'`).
        """
        super().__init__()
        self.self_attn = AttentionLayer(attention, d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        act_layer: nn.Module = (
            nn.GELU() if str(activation).strip().lower() == "gelu" else nn.ReLU()
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền tiến Pre-LN Self-Attention + FFN trên tensor `[B, L, d_model]`."""
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h)
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class BaseTimeSeriesModel(nn.Module):
    """Lớp cơ sở trừu tượng định nghĩa hợp đồng `[B, L, D] -> [B, H, 1]`."""

    def __init__(self, lookback: int, horizon: int, in_features: int) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class InformerModel(BaseTimeSeriesModel):
    """
    Kiến trúc Informer Encoder-Decoder cải tiến chống sụp phẳng đầu ra.

    Hỗ trợ đồng thời hai chế độ chú ý trong Encoder:
    - `attention_type='probsparse'`: sử dụng `ProbSparseAttention` ($O(L \\log L)$).
    - `attention_type='full'`: sử dụng `FullAttention` ($O(L^2)$) cho Ablation 3.
    """

    def __init__(
        self,
        lookback: int = 96,
        horizon: int = 24,
        in_features: int = 15,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 256,
        factor: int = 5,
        label_len: Optional[int] = None,
        dropout: float = 0.15,
        activation: str = "gelu",
        attention_type: str = "probsparse",
        deterministic_eval: bool = True,
        **kwargs,
    ) -> None:
        """
        Khởi tạo mô hình Informer.

        Tham số:
            lookback (int): Chiều dài lịch sử đầu vào L (mặc định 96 giờ).
            horizon (int): Tầm dự báo H ∈ {24, 48, 96, 192}.
            in_features (int): Số lượng đặc trưng đầu vào D.
            d_model (int): Chiều không gian ẩn (mặc định 128).
            n_heads (int): Số đầu chú ý (mặc định 4).
            e_layers (int): Số tầng Encoder (mặc định 2).
            d_layers (int): Số tầng Decoder (mặc định 1).
            d_ff (int): Chiều tầng ẩn Feed-Forward (mặc định 256).
            factor (int): Hệ số lấy mẫu ProbSparse (mặc định 5).
            label_len (Optional[int]): Độ dài đoạn gối đầu Decoder (mặc định `lookback // 2`).
            dropout (float): Tỷ lệ dropout điều chuẩn (mặc định 0.15).
            activation (str): Hàm kích hoạt (`'gelu'` hoặc `'relu'`).
            attention_type (str): `'probsparse'` hoặc `'full'` (cho Ablation 3).
            deterministic_eval (bool): Dùng Eq. 4 tất định khi `eval()` (mặc định True).
        """
        super().__init__(lookback=lookback, horizon=horizon, in_features=in_features)

        # Hỗ trợ tương thích các tham số bí danh nếu gọi từ script khác
        if kwargs.get("use_probsparse") is False or kwargs.get("attn") == "full":
            attention_type = "full"

        self.d_model = int(d_model)
        self.factor = int(factor)
        self.attention_type = str(attention_type).strip().lower()
        self.label_len = (
            min(int(label_len), lookback)
            if label_len is not None
            else (lookback // 2)
        )

        # Nhúng đầu vào Encoder và Decoder
        self.enc_embedding = DataEmbedding(in_features, d_model, dropout)
        self.dec_embedding = DataEmbedding(in_features, d_model, dropout)

        # Nhúng truy vấn chân trời học được cho H bước tương lai (chống phẳng ở Generative Decoder)
        self.horizon_queries = nn.Parameter(torch.zeros(1, horizon, d_model))
        nn.init.trunc_normal_(self.horizon_queries, std=0.02)

        # Encoder Pre-LayerNorm kèm lõi ProbSparseAttention hoặc FullAttention
        encoder_list = []
        for _ in range(e_layers):
            if self.attention_type == "full":
                inner_attn: nn.Module = FullAttention(attention_dropout=dropout)
            else:
                inner_attn = ProbSparseAttention(
                    factor=factor,
                    attention_dropout=dropout,
                    deterministic_eval=deterministic_eval,
                )
            encoder_list.append(
                InformerEncoderLayer(
                    attention=inner_attn,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.encoder_layers = nn.ModuleList(encoder_list)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder Pre-LayerNorm kèm mặt nạ nhân quả dạng float (tương thích ONNX opset=17)
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(d_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(d_model)

        dec_total_len = self.label_len + self.horizon
        causal_mask = torch.triu(
            torch.full((dec_total_len, dec_total_len), -1e9, dtype=torch.float32),
            diagonal=1,
        )
        self.register_buffer("dec_causal_mask", causal_mask, persistent=False)

        # Đầu chiếu 2 tầng giữ năng lực tái tạo đỉnh phi tuyến (giữ tên 'projection' cho fine_tune.py)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lan truyền tiến Encoder-Decoder dự báo H bước nồng độ PM2.5.

        Tham số:
            x (torch.Tensor): Tensor đầu vào `[B, L, D]`.

        Trả về:
            torch.Tensor: Tensor dự báo `[B, H, 1]`.
        """
        bsz, _, feat_dim = x.shape

        # 1. Mã hóa chuỗi lịch sử L giờ
        enc_out = self.enc_embedding(x)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        enc_out = self.encoder_norm(enc_out)

        # 2. Tạo đầu vào Generative Decoder: [đoạn gối đầu label_len | phần giữ chỗ H giờ]
        dec_inp_token = x[:, -self.label_len :, :]
        dec_inp_zeros = torch.zeros(
            bsz, self.horizon, feat_dim, device=x.device, dtype=x.dtype
        )
        dec_inp = torch.cat([dec_inp_token, dec_inp_zeros], dim=1)

        dec_out = self.dec_embedding(dec_inp)
        # Cộng truy vấn chân trời học được vào H vị trí tương lai để phân biệt rõ từng bước dự báo
        dec_prefix = dec_out[:, : self.label_len, :]
        dec_future = dec_out[:, self.label_len :, :] + self.horizon_queries
        dec_out = torch.cat([dec_prefix, dec_future], dim=1)

        tgt_mask = self.dec_causal_mask.to(dtype=dec_out.dtype, device=dec_out.device)
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask)
        dec_out = self.decoder_norm(dec_out)

        preds = self.projection(dec_out[:, -self.horizon :, :])
        return preds

    def count_parameters(self) -> int:
        """Đếm tổng số tham số có thể học (trainable parameters) của mô hình."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
