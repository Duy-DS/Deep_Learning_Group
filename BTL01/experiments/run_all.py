"""Chạy toàn bộ quy trình tái lập của dự án BTL01 từ sinh dữ liệu đến tổng hợp T0701/T0702."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_step(args: list[str]) -> None:
    """Chạy một bước Python trong thư mục gốc của dự án và báo lỗi nếu mã thoát khác 0."""
    command = [sys.executable, *args]
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    """Điều phối việc sinh dữ liệu, chạy các thực nghiệm A/B/C, phân tích và xuất đồ thị tổng thể."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--skip-model-figures", action="store_true")
    args = parser.parse_args()

    output_arg = ["--output-root", str(args.output_root)]
    run_step(["experiments/generate_data.py"])
    if not args.skip_training:
        common = [*output_arg]
        if args.epochs is not None:
            common.extend(["--epochs", str(args.epochs)])
        if args.smoke:
            run_step([
                "experiments/run_exp_A.py",
                "--sample-sizes", "200", "400",
                "--seeds", "42",
                *common,
            ])
            run_step([
                "experiments/run_exp_B.py",
                "--widths", "1", "2",
                "--seeds", "42",
                *common,
            ])
            run_step([
                "experiments/run_exp_C.py",
                "--depths", "1", "2",
                "--seeds", "42",
                *common,
            ])
        else:
            run_step(["experiments/run_exp_A.py", *common])
            run_step(["experiments/run_exp_B.py", *common])
            run_step(["experiments/run_exp_C.py", *common])

    run_step(["experiments/analyze_exp_A.py", *output_arg])
    run_step(["experiments/analyze_exp_B.py", *output_arg])
    run_step(["experiments/analyze_exp_C.py", *output_arg])
    analyze_args = ["experiments/analyze_overall.py", *output_arg]
    if args.skip_model_figures:
        analyze_args.append("--skip-model-figures")
    run_step(analyze_args)


if __name__ == "__main__":
    main()
