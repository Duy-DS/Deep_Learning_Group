"""Tiện ích tính hàm mật độ Gaussian và sinh dữ liệu xác định (deterministic).

Module này triển khai thành phần T0201 cho dự án BTL01. Quản lý công thức
toán học Gaussian 2D, lấy mẫu trên miền xác định và chia tập dữ liệu train/val/test
trong cùng một lớp OOP để các task sau tái sử dụng mà không lặp lại công thức.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ArrayLike2D = Sequence[Sequence[float]] | np.ndarray


class GaussianDataset(Dataset):
    """Đóng gói dữ liệu từ file CSV thành các Tensor float32 kèm metadata phân chia tập (split)."""

    def __init__(self, csv_path: Path | str, split: str | None = None) -> None:
        """Tải một file CSV có sẵn mà không sinh lại hoặc trộn lẫn các tập dữ liệu.

        Tham số:
            csv_path: Đường dẫn tới file CSV chứa x1, x2, y (tương đối so với thư mục gốc dự án).
            split: Tên tập dữ liệu ('train', 'val' hoặc 'test'); tự động suy ra nếu để None.

        Ngoại lệ:
            ValueError: Nếu metadata split, cấu trúc cột hoặc giá trị trong CSV không hợp lệ.
        """
        path = Path(csv_path)
        self.csv_path = path if path.is_absolute() else (
            Path(__file__).resolve().parents[1] / path
        )
        inferred = {"train_full": "train", "val": "val", "test": "test"}
        inferred_split = inferred.get(self.csv_path.stem.lower())
        self.split = inferred_split if split is None else split
        if self.split not in {"train", "val", "test"}:
            raise ValueError("Cần chỉ định split là train, val hoặc test.")
        if inferred_split is not None and self.split != inferred_split:
            raise ValueError("Split không khớp với tên file CSV chuẩn.")
        frame = pd.read_csv(self.csv_path)
        if list(frame.columns) != ["x1", "x2", "y"] or frame.empty:
            raise ValueError("Yêu cầu CSV không rỗng với các cột x1, x2, y.")
        values = frame.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("CSV chứa các giá trị không hữu hạn (NaN hoặc Inf).")
        self.features = torch.from_numpy(values[:, :2].copy())
        self.targets = torch.from_numpy(values[:, 2:].copy())

    def __len__(self) -> int:
        """Trả về số lượng mẫu trong tập dữ liệu này.

        Trả về:
            Số lượng cặp (features, target).
        """
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Lấy một mẫu dữ liệu theo chỉ số mà không làm mất chiều của nhãn target.

        Tham số:
            index: Vị trí hàng cần lấy trong tập dữ liệu.

        Trả về:
            Tuple gồm Tensor đầu vào kiểu float32 kích thước (2,) và nhãn target kích thước (1,).
        """
        return self.features[index], self.targets[index]


@dataclass(frozen=True)
class GaussianDataGenerator:
    """Sinh mẫu ngẫu nhiên và tính giá trị chính xác cho hàm mật độ xác suất Gaussian 2D.

    Tham số:
        mu: Vector kỳ vọng với kích thước ``(2,)``.
        cov: Ma trận hiệp phương sai xác định dương kích thước ``(2, 2)``.
        x1_range: Miền lấy mẫu cho tọa độ ``x1`` [min, max].
        x2_range: Miền lấy mẫu cho tọa độ ``x2`` [min, max].

    Ngoại lệ:
        ValueError: Nếu kỳ vọng, ma trận hiệp phương sai hoặc miền xác định không hợp lệ.
    """

    mu: Sequence[float] = (0.0, 0.0)
    cov: ArrayLike2D = ((0.3, -0.5), (-0.5, 2.0))
    x1_range: tuple[float, float] = (-2.0, 2.0)
    x2_range: tuple[float, float] = (-4.0, 4.0)
    _mu_array: np.ndarray = field(init=False, repr=False)
    _cov_array: np.ndarray = field(init=False, repr=False)
    _inv_cov: np.ndarray = field(init=False, repr=False)
    _det_cov: float = field(init=False, repr=False)
    _normalizer: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Kiểm tra tính hợp lệ của các tham số Gaussian và tính toán trước các hằng số."""
        mu_array = np.asarray(self.mu, dtype=np.float64)
        cov_array = np.asarray(self.cov, dtype=np.float64)
        self._validate_domain()
        if mu_array.shape != (2,):
            raise ValueError("mu phải có kích thước (2,).")
        if cov_array.shape != (2, 2):
            raise ValueError("cov phải có kích thước (2, 2).")
        if not np.allclose(cov_array, cov_array.T):
            raise ValueError("cov phải là ma trận đối xứng.")
        determinant = float(np.linalg.det(cov_array))
        if cov_array[0, 0] <= 0.0 or determinant <= 0.0:
            raise ValueError("cov phải xác định dương theo tiêu chuẩn Sylvester.")
        eigenvalues = np.linalg.eigvalsh(cov_array)
        if not np.all(eigenvalues > 0.0):
            raise ValueError("cov phải có các trị riêng hoàn toàn dương.")
        inverse = np.linalg.inv(cov_array)
        normalizer = 1.0 / (2.0 * np.pi * np.sqrt(determinant))

        object.__setattr__(self, "_mu_array", mu_array)
        object.__setattr__(self, "_cov_array", cov_array)
        object.__setattr__(self, "_inv_cov", inverse)
        object.__setattr__(self, "_det_cov", determinant)
        object.__setattr__(self, "_normalizer", float(normalizer))

    @property
    def determinant(self) -> float:
        """Trả về định thức của ma trận hiệp phương sai."""
        return self._det_cov

    @property
    def inverse_covariance(self) -> np.ndarray:
        """Trả về bản sao của ma trận nghịch đảo ma trận hiệp phương sai."""
        return self._inv_cov.copy()

    @property
    def mean(self) -> np.ndarray:
        """Trả về bản sao của vector kỳ vọng kích thước ``(2,)``."""
        return self._mu_array.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Trả về bản sao của ma trận hiệp phương sai kích thước ``(2, 2)``."""
        return self._cov_array.copy()

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """Tính giá trị mật độ xác suất Gaussian chính xác cho ma trận tọa độ đầu vào.

        Tham số:
            x: Ma trận tọa độ có kích thước ``(N, 2)`` với các cột ``x1, x2``.

        Trả về:
            Giá trị mật độ xác suất có kích thước ``(N, 1)`` và kiểu dữ liệu ``float64``.

        Ngoại lệ:
            ValueError: Nếu ``x`` không có kích thước ``(N, 2)``.
        """
        points = np.asarray(x, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("x phải có kích thước (N, 2).")
        centered = points - self._mu_array
        mahalanobis = np.einsum(
            "ni,ij,nj->n",
            centered,
            self._inv_cov,
            centered,
            optimize=True,
        )
        density = self._normalizer * np.exp(-0.5 * mahalanobis)
        return density.reshape(-1, 1)

    def sample_features(
        self,
        num_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Lấy mẫu tọa độ đầu vào phân bố đều (uniform) trong miền giá trị đã cấu hình.

        Tham số:
            num_samples: Số lượng mẫu tọa độ cần rút.
            rng: Bộ sinh số ngẫu nhiên NumPy Generator (thường dùng PCG64).

        Trả về:
            Ma trận có kích thước ``(num_samples, 2)`` và kiểu dữ liệu ``float64``.

        Ngoại lệ:
            ValueError: Nếu ``num_samples`` không phải số nguyên dương.
        """
        if num_samples <= 0:
            raise ValueError("num_samples phải là số nguyên dương.")
        x1 = rng.uniform(self.x1_range[0], self.x1_range[1], num_samples)
        x2 = rng.uniform(self.x2_range[0], self.x2_range[1], num_samples)
        return np.column_stack((x1, x2)).astype(np.float64, copy=False)

    def generate_dataframe(
        self,
        num_samples: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Sinh một tập dữ liệu có nhãn dưới dạng pandas DataFrame.

        Tham số:
            num_samples: Số lượng hàng dữ liệu cần sinh.
            rng: Bộ sinh số ngẫu nhiên NumPy Generator riêng biệt cho tập dữ liệu này.

        Trả về:
            DataFrame chứa các cột ``x1``, ``x2``, và ``y``. Ma trận đặc trưng có kích thước
            ``(N, 2)`` và nhãn target có kích thước ``(N, 1)``.
        """
        features = self.sample_features(num_samples, rng)
        targets = self.compute_density(features)
        return pd.DataFrame(
            {
                "x1": features[:, 0],
                "x2": features[:, 1],
                "y": targets[:, 0],
            }
        )

    def generate_splits(
        self,
        split_sizes: Mapping[str, int],
        seed: int = 42,
    ) -> dict[str, pd.DataFrame]:
        """Sinh các luồng tập train/val/test độc lập nhau.

        Tham số:
            split_sizes: Ánh xạ chỉ định kích thước từng tập, ví dụ: ``{"train_full": 2000,
                "val": 200, "test": 100}``.
            seed: Seed khởi tạo để sinh các luồng ``PCG64`` độc lập.

        Trả về:
            Dictionary ánh xạ tên tập dữ liệu sang DataFrame tương ứng với các cột
            ``x1``, ``x2``, và ``y``.
        """
        names = list(split_sizes)
        seed_sequence = np.random.SeedSequence(seed)
        child_sequences = seed_sequence.spawn(len(names))
        splits: dict[str, pd.DataFrame] = {}
        for name, child_sequence in zip(names, child_sequences):
            rng = np.random.Generator(np.random.PCG64(child_sequence))
            splits[name] = self.generate_dataframe(split_sizes[name], rng)
        return splits

    def fixed_train_subset_indices(
        self,
        train_full_size: int,
        subset_size: int,
        subset_seed: int = 42,
    ) -> np.ndarray:
        """Trả về danh sách chỉ số tiền tố cố định cho các thực nghiệm A/B/C sau này.

        Tham số:
            train_full_size: Tổng số dòng trong tập ``train_full.csv``.
            subset_size: Số lượng mẫu yêu cầu trong tập con.
            subset_seed: Seed dùng cho việc hoán vị cố định tập huấn luyện.

        Trả về:
            Mảng các chỉ số nguyên có kích thước ``(subset_size,)``.

        Ngoại lệ:
            ValueError: Nếu ``subset_size`` nằm ngoài khoảng [1, train_full_size].
        """
        if subset_size < 1 or subset_size > train_full_size:
            raise ValueError("subset_size phải nằm trong khoảng [1, train_full_size].")
        rng = np.random.Generator(np.random.PCG64(subset_seed))
        return rng.permutation(train_full_size)[:subset_size]

    def validate_dataframe(self, frame: pd.DataFrame) -> None:
        """Kiểm tra tính hợp lệ về cột, miền giá trị, tính hữu hạn và tính chính xác của nhãn y.

        Tham số:
            frame: DataFrame với các cột ``x1``, ``x2``, và ``y``.

        Ngoại lệ:
            ValueError: Nếu các cột, giá trị, miền xác định hoặc nhãn y không khớp với công thức.
        """
        expected = ["x1", "x2", "y"]
        if list(frame.columns) != expected:
            raise ValueError(f"Kỳ vọng các cột {expected}.")
        values = frame[expected].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Tập dữ liệu chứa giá trị NaN hoặc không hữu hạn.")
        x1 = values[:, 0]
        x2 = values[:, 1]
        if not ((self.x1_range[0] <= x1).all()
                and (x1 <= self.x1_range[1]).all()):
            raise ValueError("Giá trị x1 nằm ngoài miền xác định đã cấu hình.")
        if not ((self.x2_range[0] <= x2).all()
                and (x2 <= self.x2_range[1]).all()):
            raise ValueError("Giá trị x2 nằm ngoài miền xác định đã cấu hình.")
        expected_y = self.compute_density(values[:, :2]).ravel()
        if not np.allclose(values[:, 2], expected_y, rtol=1.0e-12,
                           atol=1.0e-15):
            raise ValueError("Nhãn mật độ y không khớp với công thức Gaussian.")

    def _validate_domain(self) -> None:
        """Kiểm tra tính hợp lệ của miền giá trị tọa độ x1, x2."""
        for name, value_range in {
            "x1_range": self.x1_range,
            "x2_range": self.x2_range,
        }.items():
            if len(value_range) != 2:
                raise ValueError(f"{name} phải gồm đúng 2 biên (min, max).")
            lower, upper = float(value_range[0]), float(value_range[1])
            if not lower < upper:
                raise ValueError(f"Biên dưới của {name} phải nhỏ hơn biên trên.")
