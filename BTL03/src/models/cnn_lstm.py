"""Mạng hồi quy tổng chữ số MNIST bằng CNN theo khung hình và hai LSTM."""

from __future__ import annotations

import torch
from torch import nn

from .base_model import BaseMigratedModel


class MNISTCNNLSTM(BaseMigratedModel):
    """Tương đương TimeDistributed CNN, LSTM(128)×2 và đầu hồi quy Keras."""

    def __init__(self, seq_len: int = 8, frame_dropout: float = 0.2, lstm_dropout: float = 0.22) -> None:
        """Khởi tạo CNN frame encoder và hai LSTM theo parameter parity Keras.

        Args:
            seq_len: Độ dài chuỗi thời gian đầu vào.
            frame_dropout: Dropout sau mỗi max-pooling CNN.
            lstm_dropout: Dropout sau đầu ra LSTM thứ hai.
        """
        super().__init__("MNISTCNNLSTM")
        if seq_len <= 0:
            raise ValueError("seq_len phải là số nguyên dương.")
        self.seq_len = int(seq_len)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(frame_dropout)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(frame_dropout)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(frame_dropout)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)

        self.lstm1 = nn.LSTM(input_size=2304, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        # Keras LSTM có một vector bias; PyTorch tách input/recurrent bias.
        # Bỏ recurrent bias không học để khớp tuyệt đối số tham số Keras.
        self.lstm1.register_parameter("bias_hh_l0", None)
        self.lstm2.register_parameter("bias_hh_l0", None)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.fc3_10ways_softmax = nn.Linear(128, 1)

    def init_weights_keras_style(self) -> None:
        """Khởi tạo CNN/LSTM theo Glorot, recurrent-orthogonal và forget bias Keras."""
        super().init_weights_keras_style()
        for recurrent in (self.lstm1, self.lstm2):
            nn.init.xavier_uniform_(recurrent.weight_ih_l0)
            nn.init.orthogonal_(recurrent.weight_hh_l0)
            # PyTorch và Keras cùng thứ tự cổng i, f, g/c, o; Keras mặc định
            # cộng 1 vào bias cổng forget khi unit_forget_bias=True.
            with torch.no_grad():
                recurrent.bias_ih_l0.zero_()
                hidden_size = recurrent.hidden_size
                recurrent.bias_ih_l0[hidden_size : 2 * hidden_size].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Nhận (B,T,1,28,28), trả tổng chữ số dự đoán dạng (B,1)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Đầu vào phải là torch.Tensor.")
        if x.ndim != 5:
            raise ValueError(
                "Chuỗi MNIST phải có 5 chiều (Batch,Time,Channels,Height,Width); "
                f"nhận {tuple(x.shape)}."
            )
        batch, time_steps, channels, height, width = x.shape
        if min(batch, time_steps, channels, height, width) <= 0:
            raise ValueError(f"Mọi chiều đầu vào phải dương; nhận {tuple(x.shape)}.")
        if time_steps != self.seq_len:
            raise ValueError(f"Cần chuỗi dài {self.seq_len}, nhận {time_steps}.")
        if channels != 1 or (height, width) != (28, 28):
            raise ValueError(f"MNIST cần shape (B,T,1,28,28), nhận {tuple(x.shape)}.")

        # (B,T,C,H,W) -> (B*T,C,H,W), tương đương TimeDistributed CNN.
        frames = x.reshape(batch * time_steps, channels, height, width)
        frames = self.dropout1(self.pool1(self.relu(self.conv1(frames))))
        # Sau pool đầu: (B*T,64,14,14).
        frames = self.dropout2(self.pool2(self.relu(self.conv2(frames))))
        # Sau pool hai: (B*T,128,7,7).
        frames = self.dropout3(self.pool3(self.relu(self.conv3(frames))))
        # Pool floor làm 7→3; mỗi frame có 256*3*3=2304 đặc trưng.
        frame_vectors = self.flatten(frames)
        # (B*T,2304) -> (B,T,2304), phục hồi thứ tự thời gian ban đầu.
        sequence = frame_vectors.reshape(batch, time_steps, 2304)
        # LSTM thứ nhất giữ đủ T bước: (B,T,2304) -> (B,T,128).
        sequence, _ = self.lstm1(sequence)
        # LSTM thứ hai cũng phát chuỗi; chỉ bước cuối tương đương return_sequences=False.
        sequence, _ = self.lstm2(sequence)
        last_step = sequence[:, -1, :]  # (B,128)
        last_step = self.lstm_dropout(last_step)
        return self.fc3_10ways_softmax(last_step)  # (B,1), giá trị hồi quy tuyến tính.