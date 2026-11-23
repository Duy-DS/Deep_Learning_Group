"""Đo chi phí tài nguyên của các backbone SOTA bằng PyTorch thuần."""

import argparse
import csv
import io
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models import PretrainedEfficientNetB0, PretrainedResNet50, PretrainedVGG16
from src.train.utils import seed_everything


class SOTAComputationalBenchmark:
    """Đo tham số, kích thước state_dict, MACs/FLOPs và tốc độ suy luận.

    FLOPs ở đây dùng quy ước 2 FLOPs cho một MAC. Bộ đếm gồm phép nhân-cộng
    của Conv2d và Linear; không cộng BatchNorm, activation, pooling hoặc memory ops.
    """

    MODEL_FACTORIES = {
        "VGG16": PretrainedVGG16,
        "ResNet50": PretrainedResNet50,
        "EfficientNetB0": PretrainedEfficientNetB0,
    }
    CSV_COLUMNS = [
        "architecture",
        "image_size",
        "device",
        "warmup_runs",
        "measure_runs",
        "total_params",
        "trainable_params",
        "frozen_params",
        "model_tensor_mb",
        "checkpoint_size_mb",
        "macs",
        "flops",
        "latency_ms_bs1",
        "latency_std_ms_bs1",
        "throughput_batch_size",
        "throughput_images_per_sec",
    ]

    def __init__(
        self,
        device: Optional[torch.device] = None,
        warmup_steps: int = 10,
        timing_steps: int = 50,
        throughput_batch_size: int = 32,
        num_classes: int = 7,
    ) -> None:
        """Khởi tạo chính sách đo benchmark.

        Args:
            device: Thiết bị đo; nếu bỏ trống tự chọn CUDA hoặc CPU.
            warmup_steps: Số lượt warm-up bỏ qua trước mỗi phép đo.
            timing_steps: Số lần lặp dùng tính latency/thông lượng.
            throughput_batch_size: Batch dùng đo throughput.
            num_classes: Số lớp đầu ra của mô hình.

        Raises:
            ValueError: Tham số đo không dương hoặc không hợp lệ.
        """
        if warmup_steps < 0 or timing_steps <= 0 or throughput_batch_size <= 0:
            raise ValueError("Số warm-up không âm; timing steps và batch size phải dương.")
        if num_classes <= 0:
            raise ValueError("num_classes phải là số nguyên dương.")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.warmup_steps = warmup_steps
        self.timing_steps = timing_steps
        self.throughput_batch_size = throughput_batch_size
        self.num_classes = num_classes
        seed_everything(42)

    def _build_model(self, architecture: str) -> nn.Module:
        """Tạo model không tải pretrained weights để benchmark offline."""
        try:
            model_type = self.MODEL_FACTORIES[architecture]
        except KeyError:
            raise ValueError(f"Kiến trúc không được hỗ trợ: {architecture}") from None
        return model_type(num_classes=self.num_classes, pretrained=False).to(self.device).eval()

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Đếm tổng, trainable và frozen parameters."""
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
        }

    @staticmethod
    def measure_model_size_mb(model: nn.Module) -> Dict[str, float]:
        """Đo dung lượng tensor state_dict và serialization thực tế theo MB thập phân.

        Kích thước tensor tính toàn bộ parameter và buffer. Kích thước serialized
        state_dict được đo bằng BytesIO ở định dạng torch.save, không ghi file tạm.
        """
        state = model.state_dict()
        tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        stream = io.BytesIO()
        torch.save(state, stream)
        serialized_bytes = stream.tell()
        return {
            "model_tensor_mb": tensor_bytes / 1_000_000,
            "checkpoint_size_mb": serialized_bytes / 1_000_000,
        }

    @staticmethod
    def estimate_flops_and_macs(model: nn.Module, image_size: int) -> Dict[str, int]:
        """Đếm MACs/FLOPs của một ảnh bằng forward hooks trên Conv2d/Linear."""
        macs = 0
        handles = []

        def count_conv(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            nonlocal macs
            module = _module
            if not isinstance(module, nn.Conv2d):
                return
            batch, out_channels, out_height, out_width = output.shape
            kernel_ops = (
                module.in_channels // module.groups
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
            macs += int(batch * out_channels * out_height * out_width * kernel_ops)

        def count_linear(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            nonlocal macs
            module = _module
            if isinstance(module, nn.Linear):
                macs += int(output.numel() * module.in_features)

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(count_conv))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(count_linear))
        try:
            sample = torch.zeros(1, 3, image_size, image_size, device=next(model.parameters()).device)
            with torch.inference_mode():
                model(sample)
        finally:
            for handle in handles:
                handle.remove()
        return {"macs": macs, "flops": 2 * macs}

    def _synchronize(self) -> None:
        """Đồng bộ CUDA để tính thời gian kernel thực thay vì thời gian enqueue."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _measure_batch_latency(self, model: nn.Module, image_size: int, batch_size: int) -> List[float]:
        """Warm-up rồi trả về thời gian từng forward theo milliseconds/batch."""
        sample = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
        with torch.inference_mode():
            for _ in range(self.warmup_steps):
                model(sample)
            self._synchronize()
            durations = []
            for _ in range(self.timing_steps):
                self._synchronize()
                started = time.perf_counter()
                model(sample)
                self._synchronize()
                durations.append((time.perf_counter() - started) * 1000.0)
        return durations

    def measure_inference_speed(self, model: nn.Module, image_size: int) -> Dict[str, float]:
        """Warm-up rồi đo latency batch=1 và throughput theo các forward liên tiếp."""
        latency_samples = self._measure_batch_latency(model, image_size, batch_size=1)
        throughput_sample = torch.randn(
            self.throughput_batch_size, 3, image_size, image_size, device=self.device
        )
        with torch.inference_mode():
            for _ in range(self.warmup_steps):
                model(throughput_sample)
            self._synchronize()
            started = time.perf_counter()
            for _ in range(self.timing_steps):
                model(throughput_sample)
            self._synchronize()
            throughput_elapsed_seconds = time.perf_counter() - started
        if throughput_elapsed_seconds <= 0.0:
            raise RuntimeError("Thời gian đo throughput phải lớn hơn 0 giây.")
        return {
            "latency_ms_bs1": statistics.mean(latency_samples),
            "latency_std_ms_bs1": statistics.stdev(latency_samples) if len(latency_samples) > 1 else 0.0,
            "throughput_batch_size": self.throughput_batch_size,
            "throughput_images_per_sec": (
                self.throughput_batch_size * self.timing_steps / throughput_elapsed_seconds
            ),
        }

    def benchmark_one(self, architecture: str, image_size: int) -> Dict[str, object]:
        """Đo toàn bộ metrics cho một kiến trúc và một kích thước ảnh."""
        model = self._build_model(architecture)
        try:
            summary = self.count_parameters(model)
            size_metrics = self.measure_model_size_mb(model)
            compute_metrics = self.estimate_flops_and_macs(model, image_size)
            speed_metrics = self.measure_inference_speed(model, image_size)
            return {
                "architecture": architecture,
                "image_size": image_size,
                "device": str(self.device),
                "warmup_runs": self.warmup_steps,
                "measure_runs": self.timing_steps,
                **summary,
                **size_metrics,
                **compute_metrics,
                **speed_metrics,
            }
        finally:
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def benchmark(
        self,
        architectures: Optional[Iterable[str]] = None,
        image_sizes: Iterable[int] = (128, 224),
    ) -> List[Dict[str, object]]:
        """Chạy benchmark trên tích số kiến trúc và độ phân giải yêu cầu."""
        selected = list(architectures or self.MODEL_FACTORIES.keys())
        sizes = list(image_sizes)
        if not sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 32 for size in sizes
        ):
            raise ValueError("image_sizes phải chứa kích thước nguyên từ 32 trở lên.")
        results = []
        total_combinations = len(selected) * len(sizes)
        for index, (name, size) in enumerate(
            ((name, size) for name in selected for size in sizes), start=1
        ):
            results.append(self.benchmark_one(name, size))
            print(
                f"[{index}/{total_combinations}] Hoàn tất {name} ở {size}x{size} trên {self.device}.",
                flush=True,
            )
        return results

    @staticmethod
    def write_csv(results: List[Dict[str, object]], output_path: Path) -> Path:
        """Ghi kết quả benchmark ra CSV tại đường dẫn được yêu cầu."""
        if not results:
            raise ValueError("Không có kết quả benchmark để xuất.")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = SOTAComputationalBenchmark.CSV_COLUMNS
        if any(set(result) != set(fields) for result in results):
            raise ValueError("Các dòng benchmark phải chứa đúng bộ cột CSV đã định nghĩa.")
        with destination.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        return destination

    @classmethod
    def cli(cls, arguments: Optional[List[str]] = None) -> int:
        """Chạy benchmark CLI và xuất bảng CSV."""
        parser = argparse.ArgumentParser(description="Benchmark chi phí tính toán SOTA FER2013.")
        parser.add_argument("--image-sizes", nargs="+", type=int, default=[128, 224])
        parser.add_argument(
            "--output",
            type=str,
            default="outputs/logs/t0305_sota_computational_benchmark.csv",
        )
        parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        parser.add_argument("--warmup-runs", "--warmup-steps", dest="warmup_runs", type=int, default=10)
        parser.add_argument("--measure-runs", "--timing-steps", dest="measure_runs", type=int, default=50)
        parser.add_argument("--throughput-batch-size", type=int, default=32)
        args = parser.parse_args(arguments)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Đã yêu cầu CUDA nhưng PyTorch không phát hiện GPU.")
        device = None if args.device == "auto" else torch.device(args.device)
        benchmark = cls(
            device=device,
            warmup_steps=args.warmup_runs,
            timing_steps=args.measure_runs,
            throughput_batch_size=args.throughput_batch_size,
        )
        results = benchmark.benchmark(image_sizes=args.image_sizes)
        output = cls.write_csv(results, args.output)
        print(f"Đã ghi {len(results)} dòng benchmark vào {output}")
        return 0


if __name__ == "__main__":
    raise SystemExit(SOTAComputationalBenchmark.cli())