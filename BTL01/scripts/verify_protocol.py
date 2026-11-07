"""Kiểm tra cấu hình T0102 và các bất biến toán học mà không cần huấn luyện.

Chạy trong môi trường Python của dự án. Không sinh thêm file hay dataset mới.
"""

import re
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import yaml


class ProtocolVerifier:
    """Kiểm tra tính hợp lệ của cấu hình baseline, thông số thực nghiệm và tính khả chuyển."""

    def __init__(self, root: Path) -> None:
        """Đọc và giải nạp cấu hình an toàn từ YAML.

        Tham số:
            root: Đường dẫn tuyệt đối tới thư mục gốc dự án chứa configs và AGENTS.md.
        """
        self.root = root
        self.baseline = self._load("configs/baseline.yaml")
        self.specs = self._load("configs/exp_specs.yaml")
        self.checks = 0

    def _load(self, relative_path: str) -> dict[str, Any]:
        """Đọc từ điển YAML an toàn không dùng constructor tùy biến.

        Tham số:
            relative_path: Đường dẫn file cấu hình tương đối so với thư mục gốc dự án.

        Trả về:
            Dictionary cấu hình cấp cao nhất.

        Ngoại lệ:
            ValueError: Nếu đối tượng YAML không phải là một mapping/dict.
        """
        with (self.root / relative_path).open(encoding="utf-8") as stream:
            result = yaml.safe_load(stream)
        if not isinstance(result, dict):
            raise ValueError(f"Kỳ vọng YAML mapping: {relative_path}")
        return result

    def _check(self, condition: bool, label: str) -> None:
        """Ghi nhận kiểm tra thành công hoặc raise ngoại lệ kèm thông báo chi tiết.

        Tham số:
            condition: Điều kiện cần thỏa mãn (bất biến).
            label: Chuỗi mô tả điều kiện kiểm tra.

        Ngoại lệ:
            ValueError: Nếu điều kiện không được thỏa mãn.
        """
        if not condition:
            raise ValueError(label)
        self.checks += 1

    def verify(self) -> None:
        """Kiểm tra cấu hình tĩnh, ma trận hiệp phương sai SPD và các ràng buộc tài liệu.

        Ngoại lệ:
            ValueError: Nếu cấu hình hoặc ràng buộc tài liệu không thỏa mãn.
            numpy.linalg.LinAlgError: Nếu ma trận hiệp phương sai suy biến hoặc không SPD.
        """
        b, s = self.baseline, self.specs
        self._check(b["schema_version"] == s["schema_version"] == 1, "schema")
        self._check(b["experiment_specs"] == "configs/exp_specs.yaml",
                    "spec link")
        self._check(s["baseline_config"] == "configs/baseline.yaml",
                    "baseline link")
        self._check(b["reproducibility"]["seed"] == 42, "seed")
        self._check(b["reproducibility"]["deterministic"] is True,
                    "determinism")
        self._check(b["hardware"]["device"] == "auto", "device")
        self._check(b["hardware"]["device_priority"] == ["cuda", "mps", "cpu"],
                    "priority")
        d, m, t = b["data"], b["model_baseline"], b["training_baseline"]
        expected_data = {
            "input_dim": 2, "output_dim": 1, "sampling": "uniform",
            "x1_range": [-2.0, 2.0], "x2_range": [-4.0, 4.0],
            "mu": [0.0, 0.0], "cov": [[0.3, -0.5], [-0.5, 2.0]],
            "train_full_size": 2000, "val_size": 200, "test_size": 100,
            "generation_seed": 42, "subset_seed": 42,
            "normalization": "none", "fixed_across_training_seeds": True,
            "subset_strategy": "fixed_permutation_prefix",
            "train_full_path": "data/train_full.csv",
            "val_path": "data/val.csv", "test_path": "data/test.csv",
        }
        for key, value in expected_data.items():
            self._check(d[key] == value, f"data.{key}")
        self._check(m["hidden_depth"] == len(m["hidden_dims"]) == 1, "depth")
        self._check(m["hidden_dims"] == [10], "width")
        self._check(m["activation"] == "tanh", "activation")
        self._check(m["weight_init"] == "xavier_normal", "initialization")
        self._check(m["output_activation"] == "linear", "linear output")
        self._check(set(m["supported_activations"]) == {
            "tanh", "relu", "sigmoid", "leaky_relu"
        }, "supported activations")
        for key, value in {
            "train_size": 1000, "batch_size": 32, "epochs": 1000,
            "optimizer": "adam", "learning_rate": 0.005,
            "weight_decay": 0.0, "loss_function": "mse",
        }.items():
            self._check(t[key] == value, f"training.{key}")
        self._check(set(t["supported_optimizers"]) == {"adam", "sgd"},
                    "optimizers")
        self._check(t["early_stopping"] == {
            "enabled": True, "patience": 50, "min_delta": 1.0e-6,
            "monitor": "val_loss", "mode": "min",
        }, "early stopping")
        for value in [t["early_stopping"]["min_delta"], t["adam_eps"]]:
            self._check(isinstance(value, float),
                        "scientific notation is float")
        self._check(t["checkpoint"]["monitor"] == "val_loss",
                    "checkpoint split")
        self._check(
            b["evaluation"]["test_policy"] == "final_frozen_suite_only",
            "test policy",
        )
        self._check(b["evaluation"]["metrics"] == ["mse", "rmse", "mae"],
                    "metrics")
        self._check(b["evaluation"]["std_ddof"] == 1, "sample std")
        self._check(b["visualization"]["dpi"] >= 300, "DPI")
        self._check(b["visualization"]["format"] == "png", "figure format")

        covariance = np.asarray(d["cov"], dtype=np.float64)
        self._check(covariance.shape == (2, 2), "covariance shape")
        self._check(bool(np.allclose(covariance, covariance.T)), "symmetry")
        determinant = float(np.linalg.det(covariance))
        self._check(bool(np.isclose(determinant, 0.35)), "determinant")
        self._check(covariance[0, 0] > 0 and determinant > 0, "Sylvester")
        eigenvalues = np.linalg.eigvalsh(covariance)
        self._check(bool(np.all(eigenvalues > 0)), "positive eigenvalues")
        lower = np.linalg.cholesky(covariance)
        self._check(bool(np.allclose(lower @ lower.T, covariance)), "Cholesky")
        inverse = np.linalg.inv(covariance)
        self._check(bool(np.allclose(covariance @ inverse, np.eye(2))),
                    "inverse")

        experiments = s["experiments"]
        a, width, depth = (experiments[k] for k in ("exp_A", "exp_B", "exp_C"))
        self._check(a["sample_sizes"] == list(range(200, 2001, 200)), "A grid")
        self._check(width["neuron_widths"] == list(range(1, 11)), "B grid")
        self._check(width["extended_widths"] == [16, 32, 64], "B extension")
        self._check(width["include_extensions"] is True, "B extension enabled")
        self._check(width["hidden_depth"] == 1, "B depth")
        self._check(depth["depths"] == [1, 2, 3, 4, 5], "C grid")
        self._check(depth["width_per_layer"] == 10, "C width")
        self._check(
            depth["comparison_mode"] == "fixed_width_not_parameter_matched",
            "C fairness",
        )
        self._check(width["train_size"] == depth["train_size"] == 1000,
                    "B/C train size")
        self._check(s["protocol"]["multi_seeds"] == [42, 100, 2026],
                    "protocol seeds")
        for name, experiment in experiments.items():
            self._check(experiment["multi_seeds"] == [42, 100, 2026],
                        f"{name} seeds")
        counts = [len(a["sample_sizes"]) * 3,
                  (len(width["neuron_widths"])
                   + len(width["extended_widths"])) * 3,
                  len(depth["depths"]) * 3]
        self._check(counts == [30, 39, 15], "run counts")
        for experiment, count in zip(experiments.values(), counts):
            self._check(experiment["expected_runs"] == count,
                        "declared run count")
        for path in [d["train_full_path"], d["val_path"], d["test_path"],
                     *b["outputs"].values(), b["experiment_specs"],
                     s["baseline_config"],
                     *(e[k] for e in experiments.values()
                       for k in ("summary_file", "figure_file"))]:
            self._check(not Path(path).is_absolute()
                        and not PureWindowsPath(path).is_absolute()
                        and ".." not in Path(path).parts, "portable path")
        for folder in ["configs", "data", "outputs/checkpoints",
                       "outputs/tables", "outputs/figures",
                       "docs/project_charter", "docs/reports",
                       "docs/dev-logs", "scripts"]:
            self._check((self.root / folder).is_dir(), f"directory {folder}")
        files = ["configs/baseline.yaml", "configs/exp_specs.yaml",
                 "README.md",
                 "docs/project_charter/problem_specification_and_protocol.md",
                 "scripts/verify_protocol.py"]
        for relative_path in files:
            content = (self.root / relative_path).read_text(encoding="utf-8")
            personal_path = r"(?i)(?<![a-z])(?:[a-z]:[\\/]|file:" + "/{3})"
            self._check(re.search(personal_path, content) is None,
                        f"không chứa đường dẫn tuyệt đối cá nhân: {relative_path}")
        print(f"PASS: {self.checks} protocol checks; YAML safe_load OK")
        print(f"det(C)={determinant:.12f}")
        print(f"eigenvalues={eigenvalues.tolist()}")
        print(f"inverse={inverse.tolist()}; Cholesky OK")
        print(f"runs A/B/C={counts}; total={sum(counts)}")
        print("C parameter counts:",
              [41 + 110 * (c - 1) for c in depth["depths"]])


if __name__ == "__main__":
    ProtocolVerifier(Path(__file__).resolve().parents[1]).verify()