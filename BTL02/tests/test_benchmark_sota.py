"""Kiểm thử bộ đo chi phí tính toán các backbone SOTA."""

import csv
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.evaluate.benchmark_sota import SOTAComputationalBenchmark
from src.models import PretrainedEfficientNetB0, PretrainedResNet50, PretrainedVGG16


class TestSOTAComputationalBenchmark:
    """Xác minh bộ đếm PyTorch và pipeline đo cho ba kiến trúc."""

    @pytest.mark.parametrize(
        "model_type",
        [PretrainedVGG16, PretrainedResNet50, PretrainedEfficientNetB0],
    )
    @pytest.mark.parametrize("image_size", [128, 224])
    def test_benchmark_one_reports_positive_metrics(self, model_type, image_size: int) -> None:
        """Mỗi backbone/kích thước sinh đủ tham số, compute, size và timing dương."""
        architecture = {
            PretrainedVGG16: "VGG16",
            PretrainedResNet50: "ResNet50",
            PretrainedEfficientNetB0: "EfficientNetB0",
        }[model_type]
        benchmark = SOTAComputationalBenchmark(
            device=torch.device("cpu"), warmup_steps=1, timing_steps=2, throughput_batch_size=2
        )
        result = benchmark.benchmark_one(architecture, image_size)
        assert result["warmup_runs"] == 1
        assert result["measure_runs"] == 2
        assert result["total_params"] > 0
        assert result["trainable_params"] == result["total_params"]
        assert result["frozen_params"] == 0
        assert result["model_tensor_mb"] > 0
        assert result["checkpoint_size_mb"] > 0
        assert result["macs"] > 0
        assert result["flops"] == 2 * result["macs"]
        assert result["latency_ms_bs1"] > 0
        assert result["throughput_batch_size"] == 2
        assert result["throughput_images_per_sec"] > 0

    def test_conv_and_linear_macs_match_hand_calculation(self) -> None:
        """Hook đếm MACs chính xác cho Conv2d và Linear trên mạng nhỏ xác định."""
        model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, bias=False),
            nn.Flatten(),
            nn.Linear(4 * 2 * 2, 5, bias=False),
        )
        result = SOTAComputationalBenchmark.estimate_flops_and_macs(model, image_size=4)
        conv_macs = 4 * 2 * 2 * (3 * 3 * 3)
        linear_macs = 5 * (4 * 2 * 2)
        assert result == {"macs": conv_macs + linear_macs, "flops": 2 * (conv_macs + linear_macs)}

    def test_parameter_and_buffer_size_accounting(self) -> None:
        """Thống kê counts frozen chính xác và size có cả buffer state_dict."""
        model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
        model[0].weight.requires_grad = False
        summary = SOTAComputationalBenchmark.count_parameters(model)
        assert summary["total_params"] == sum(parameter.numel() for parameter in model.parameters())
        assert summary["trainable_params"] < summary["total_params"]
        assert summary["frozen_params"] == summary["total_params"] - summary["trainable_params"]
        sizes = SOTAComputationalBenchmark.measure_model_size_mb(model)
        state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())
        assert sizes["model_tensor_mb"] == pytest.approx(state_bytes / 1_000_000)
        assert sizes["checkpoint_size_mb"] >= sizes["model_tensor_mb"]

    def test_write_csv_preserves_all_benchmark_fields(self, tmp_path) -> None:
        """CSV benchmark có đúng header và một dòng trên mỗi mẫu kết quả."""
        benchmark = SOTAComputationalBenchmark(device=torch.device("cpu"), warmup_steps=0, timing_steps=1,
                                                throughput_batch_size=1)
        result = benchmark.benchmark_one("EfficientNetB0", image_size=48)
        output_path = benchmark.write_csv([result], tmp_path / "metrics" / "benchmark.csv")
        with output_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0]["architecture"] == "EfficientNetB0"
        assert int(rows[0]["macs"]) > 0
        assert list(rows[0]) == SOTAComputationalBenchmark.CSV_COLUMNS
        assert rows[0]["warmup_runs"] == "0"
        assert rows[0]["measure_runs"] == "1"

    def test_cli_accepts_custom_warmup_and_measure_run_counts(self, tmp_path) -> None:
        """CLI chuyển số lần lặp tùy chỉnh vào cấu hình và metadata CSV."""
        with patch.object(SOTAComputationalBenchmark, "benchmark") as benchmark_mock:
            benchmark_mock.return_value = [
                {
                    "architecture": "Fake",
                    "image_size": 48,
                    "device": "cpu",
                    "warmup_runs": 3,
                    "measure_runs": 10,
                    "total_params": 1,
                    "trainable_params": 1,
                    "frozen_params": 0,
                    "model_tensor_mb": 0.1,
                    "checkpoint_size_mb": 0.1,
                    "macs": 1,
                    "flops": 2,
                    "latency_ms_bs1": 1.0,
                    "latency_std_ms_bs1": 0.0,
                    "throughput_batch_size": 32,
                    "throughput_images_per_sec": 32.0,
                }
            ]
            with patch("src.evaluate.benchmark_sota.SOTAComputationalBenchmark.write_csv") as write_mock:
                write_mock.return_value = tmp_path / "bench.csv"
                exit_code = SOTAComputationalBenchmark.cli([
                    "--image-sizes", "48", "--warmup-runs", "3", "--measure-runs", "10",
                    "--output", str(tmp_path / "bench.csv"), "--device", "cpu",
                ])
        assert exit_code == 0
        assert benchmark_mock.call_args.kwargs["image_sizes"] == [48]
        assert write_mock.call_args.args[0][0]["warmup_runs"] == 3
        assert write_mock.call_args.args[0][0]["measure_runs"] == 10

    def test_unknown_architecture_and_invalid_sizes_are_rejected(self) -> None:
        """Từ chối architecture hoặc kích thước đầu vào không hỗ trợ."""
        benchmark = SOTAComputationalBenchmark(device=torch.device("cpu"), warmup_steps=0, timing_steps=1)
        with pytest.raises(ValueError, match="không được hỗ trợ"):
            benchmark.benchmark_one("Unknown", 128)
        with pytest.raises(ValueError, match="image_sizes"):
            benchmark.benchmark(image_sizes=[])
        with pytest.raises(ValueError, match="image_sizes"):
            benchmark.benchmark(image_sizes=[128.0])

    def test_efficientnet_b0_is_smaller_and_lower_macs_than_resnet50(self) -> None:
        """EfficientNet-B0 cần ít tham số và MACs hơn ResNet50 ở cùng size."""
        benchmark = SOTAComputationalBenchmark(
            device=torch.device("cpu"), warmup_steps=0, timing_steps=1, throughput_batch_size=1
        )
        efficient = benchmark.benchmark_one("EfficientNetB0", image_size=128)
        residual = benchmark.benchmark_one("ResNet50", image_size=128)
        assert efficient["total_params"] < residual["total_params"]
        assert efficient["macs"] < residual["macs"]