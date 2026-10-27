"""Gaussian density utilities and deterministic data generation.

This module implements the T0201 contract for the BTL01 project.  It keeps
the mathematical Gaussian definition, domain sampling, and split generation
in one OOP component so later tasks can reuse the same data contract without
duplicating formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


ArrayLike2D = Sequence[Sequence[float]] | np.ndarray


@dataclass(frozen=True)
class GaussianDataGenerator:
    """Generate samples and exact labels for the 2-D Gaussian density.

    Args:
        mu: Mean vector with shape ``(2,)``.
        cov: Positive definite covariance matrix with shape ``(2, 2)``.
        x1_range: Inclusive sampling range for coordinate ``x1``.
        x2_range: Inclusive sampling range for coordinate ``x2``.

    Raises:
        ValueError: If the mean, covariance, or domain is invalid.
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
        """Validate Gaussian parameters and precompute constants."""
        mu_array = np.asarray(self.mu, dtype=np.float64)
        cov_array = np.asarray(self.cov, dtype=np.float64)
        self._validate_domain()
        if mu_array.shape != (2,):
            raise ValueError("mu must have shape (2,).")
        if cov_array.shape != (2, 2):
            raise ValueError("cov must have shape (2, 2).")
        if not np.allclose(cov_array, cov_array.T):
            raise ValueError("cov must be symmetric.")
        determinant = float(np.linalg.det(cov_array))
        if cov_array[0, 0] <= 0.0 or determinant <= 0.0:
            raise ValueError("cov must be positive definite by Sylvester.")
        eigenvalues = np.linalg.eigvalsh(cov_array)
        if not np.all(eigenvalues > 0.0):
            raise ValueError("cov must have strictly positive eigenvalues.")
        inverse = np.linalg.inv(cov_array)
        normalizer = 1.0 / (2.0 * np.pi * np.sqrt(determinant))

        object.__setattr__(self, "_mu_array", mu_array)
        object.__setattr__(self, "_cov_array", cov_array)
        object.__setattr__(self, "_inv_cov", inverse)
        object.__setattr__(self, "_det_cov", determinant)
        object.__setattr__(self, "_normalizer", float(normalizer))

    @property
    def determinant(self) -> float:
        """Return the covariance determinant."""
        return self._det_cov

    @property
    def inverse_covariance(self) -> np.ndarray:
        """Return a copy of the inverse covariance matrix."""
        return self._inv_cov.copy()

    @property
    def mean(self) -> np.ndarray:
        """Return a copy of the mean vector with shape ``(2,)``."""
        return self._mu_array.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Return a copy of the covariance matrix with shape ``(2, 2)``."""
        return self._cov_array.copy()

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """Compute exact Gaussian density labels for coordinate rows.

        Args:
            x: Coordinate matrix with shape ``(N, 2)`` and columns
                ``x1, x2``.

        Returns:
            Density values with shape ``(N, 1)`` and dtype ``float64``.

        Raises:
            ValueError: If ``x`` does not have shape ``(N, 2)``.
        """
        points = np.asarray(x, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("x must have shape (N, 2).")
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
        """Sample input coordinates uniformly from the configured domain.

        Args:
            num_samples: Number of coordinate rows to draw.
            rng: NumPy generator, normally backed by ``PCG64``.

        Returns:
            Matrix with shape ``(num_samples, 2)`` and dtype ``float64``.

        Raises:
            ValueError: If ``num_samples`` is not positive.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        x1 = rng.uniform(self.x1_range[0], self.x1_range[1], num_samples)
        x2 = rng.uniform(self.x2_range[0], self.x2_range[1], num_samples)
        return np.column_stack((x1, x2)).astype(np.float64, copy=False)

    def generate_dataframe(
        self,
        num_samples: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate a labeled split as a DataFrame.

        Args:
            num_samples: Number of rows to generate.
            rng: NumPy generator used only for this split stream.

        Returns:
            DataFrame with columns ``x1``, ``x2``, and ``y``.  The feature
            matrix has shape ``(N, 2)`` and the target has shape ``(N, 1)``.
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
        """Generate independent train/validation/test split streams.

        Args:
            split_sizes: Mapping such as ``{"train_full": 2000,
                "val": 200, "test": 100}``.
            seed: Seed used to spawn independent ``PCG64`` streams.

        Returns:
            Dictionary mapping split names to DataFrames with columns
            ``x1``, ``x2``, and ``y``.
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
        """Return deterministic prefix indices for later A/B/C tasks.

        Args:
            train_full_size: Number of rows in ``train_full.csv``.
            subset_size: Requested prefix length.
            subset_seed: Seed for the fixed training permutation.

        Returns:
            Integer indices with shape ``(subset_size,)``.

        Raises:
            ValueError: If ``subset_size`` is outside the full train range.
        """
        if subset_size < 1 or subset_size > train_full_size:
            raise ValueError("subset_size must be in [1, train_full_size].")
        rng = np.random.Generator(np.random.PCG64(subset_seed))
        return rng.permutation(train_full_size)[:subset_size]

    def validate_dataframe(self, frame: pd.DataFrame) -> None:
        """Validate split columns, domain bounds, finiteness, and labels.

        Args:
            frame: DataFrame with columns ``x1``, ``x2``, and ``y``.

        Raises:
            ValueError: If columns, values, domain, or labels are invalid.
        """
        expected = ["x1", "x2", "y"]
        if list(frame.columns) != expected:
            raise ValueError(f"Expected columns {expected}.")
        values = frame[expected].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Split contains NaN or infinite values.")
        x1 = values[:, 0]
        x2 = values[:, 1]
        if not ((self.x1_range[0] <= x1).all()
                and (x1 <= self.x1_range[1]).all()):
            raise ValueError("x1 values are outside the configured range.")
        if not ((self.x2_range[0] <= x2).all()
                and (x2 <= self.x2_range[1]).all()):
            raise ValueError("x2 values are outside the configured range.")
        expected_y = self.compute_density(values[:, :2]).ravel()
        if not np.allclose(values[:, 2], expected_y, rtol=1.0e-12,
                           atol=1.0e-15):
            raise ValueError("Density labels do not match the formula.")

    def _validate_domain(self) -> None:
        """Validate coordinate ranges."""
        for name, value_range in {
            "x1_range": self.x1_range,
            "x2_range": self.x2_range,
        }.items():
            if len(value_range) != 2:
                raise ValueError(f"{name} must have two bounds.")
            lower, upper = float(value_range[0]), float(value_range[1])
            if not lower < upper:
                raise ValueError(f"{name} lower bound must be < upper bound.")
