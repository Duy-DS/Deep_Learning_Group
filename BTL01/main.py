"""Command line entry point for the BTL01 function approximation project."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_script(script: str, extra_args: list[str]) -> None:
    """Run a project script with the current Python interpreter."""
    subprocess.run([sys.executable, script, *extra_args], cwd=ROOT, check=True)


def main() -> None:
    """Dispatch common project workflows through one CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "data",
            "exp_A",
            "exp_B",
            "exp_C",
            "analyze",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-model-figures", action="store_true")
    args = parser.parse_args()

    output_args = ["--output-root", str(args.output_root)]
    epoch_args = [] if args.epochs is None else ["--epochs", str(args.epochs)]
    if args.mode == "data":
        run_script("experiments/generate_data.py", [])
    elif args.mode == "exp_A":
        run_script("experiments/run_exp_A.py", [*output_args, *epoch_args])
    elif args.mode == "exp_B":
        run_script("experiments/run_exp_B.py", [*output_args, *epoch_args])
    elif args.mode == "exp_C":
        run_script("experiments/run_exp_C.py", [*output_args, *epoch_args])
    elif args.mode == "analyze":
        analyze_args = [*output_args]
        if args.skip_model_figures:
            analyze_args.append("--skip-model-figures")
        run_script("experiments/analyze_overall.py", analyze_args)
    else:
        run_all_args = [*output_args]
        if args.epochs is not None:
            run_all_args.extend(epoch_args)
        if args.smoke:
            run_all_args.append("--smoke")
        if args.skip_training:
            run_all_args.append("--skip-training")
        if args.skip_model_figures:
            run_all_args.append("--skip-model-figures")
        run_script("experiments/run_all.py", run_all_args)


if __name__ == "__main__":
    main()
