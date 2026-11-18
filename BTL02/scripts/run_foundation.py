"""Chạy tuần tự T0103–T0202 trong vùng đầu ra riêng, không ghi đè kết quả cũ."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


class FoundationRunner:
    """Điều phối kiểm thử, xác thực dữ liệu, EDA và ba thực nghiệm CNN."""

    def __init__(self, run_dir: Path) -> None:
        """Thiết lập vùng kết quả riêng và đường dẫn project từ script."""
        self.root = Path(__file__).resolve().parents[1]
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.state = {"status": "RUNNING", "started": datetime.now().isoformat(), "steps": {}}
        self.save_state()

    def save_state(self) -> None:
        """Lưu trạng thái nguyên tử để có thể theo dõi khi chạy nền."""
        temporary = self.run_dir / "status.tmp"
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.run_dir / "status.json")

    def run_tests(self) -> None:
        """Chạy pytest dependency thật; test tự tạo dữ liệu tạm ngoài repo."""
        subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
            cwd=self.root, check=True,
        )
        self.state["steps"]["tests"] = "PASSED"
        self.save_state()

    def run_data_and_eda(self) -> None:
        """Xác minh CSV thật, chạy code cell notebook và chuyển hình sang vùng riêng."""
        import matplotlib

        matplotlib.use("Agg")
        from src.data.downloader import FER2013Downloader

        csv_path = self.root / "data" / "raw" / "fer2013.csv"
        self.state["data"] = {
            "counts": FER2013Downloader.validate_csv(csv_path),
            "sha256": FER2013Downloader.calculate_sha256(csv_path),
        }
        print("DATA VERIFIED:", self.state["data"], flush=True)
        notebook = json.loads((self.root / "notebooks" / "01_eda_fer2013.ipynb").read_text(encoding="utf-8"))
        scope = {"__name__": "__eda_execution__"}
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                # Chỉ đổi đích hình trong bộ nhớ, không sửa notebook gốc.
                source = source.replace(
                    "FIGURES_DIR = project_root / 'outputs' / 'figures'",
                    f"FIGURES_DIR = Path({str(self.run_dir / 'figures')!r})",
                )
                print(f"EDA CELL {index}", flush=True)
                exec(compile(source, f"01_eda_fer2013.ipynb:cell{index}", "exec"), scope)
            self.state["eda"] = {
                "mean": scope["mean_val"], "std": scope["std_val"],
                "distribution": scope["crosstab_counts"].to_dict(),
            }
            import matplotlib.pyplot as plt

            plt.close("all")
        finally:
            os.chdir(previous)
        self.state["steps"]["T0104_T0105"] = "PASSED"
        self.save_state()

    def run_training(self) -> None:
        """Chạy ba cấu hình gốc với checkpoint/log riêng và giữ EarlyStopping."""
        import mlflow
        import yaml
        from src.train.trainer import Trainer

        previous = Path.cwd()
        os.chdir(self.run_dir)
        try:
            for name in ("exp01_small_cnn", "exp02_medium_cnn", "exp03_large_cnn"):
                config = yaml.safe_load(
                    (self.root / "experiments" / "baseline" / f"{name}.yaml").read_text(encoding="utf-8")
                )
                config["csv_path"] = str(self.root / "data" / "raw" / "fer2013.csv")
                config["checkpoint_path"] = str(self.run_dir / "checkpoints" / f"{name}.pth")
                (self.run_dir / f"{name}.yaml").write_text(
                    yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
                )
                self.state["steps"][name] = "RUNNING"
                self.save_state()
                print(f"START {name}", flush=True)
                trainer = Trainer(config)
                try:
                    result = trainer.fit()
                finally:
                    if mlflow.active_run() is not None:
                        mlflow.end_run(status="FAILED")
                self.state["steps"][name] = result
                self.save_state()
                del trainer
        finally:
            os.chdir(previous)

    def run(self) -> int:
        """Thực thi tuần tự, ghi trạng thái lỗi trung thực nếu bất kỳ bước thất bại."""
        started = time.monotonic()
        sys.path.insert(0, str(self.root))
        try:
            self.run_tests()
            self.run_data_and_eda()
            self.run_training()
            self.state["status"] = "COMPLETED"
            return 0
        except Exception as error:
            self.state["status"] = "FAILED"
            self.state["error_type"] = type(error).__name__
            traceback.print_exc()
            return 1
        finally:
            self.state["elapsed_seconds"] = time.monotonic() - started
            self.state["updated"] = datetime.now().isoformat()
            self.save_state()

    @classmethod
    def cli(cls) -> int:
        """Đọc vùng đầu ra bắt buộc để tránh khởi chạy trùng hoặc ghi đè kết quả."""
        parser = argparse.ArgumentParser(description="Chạy T0103–T0202 với seed/cấu hình gốc.")
        parser.add_argument("--run-dir", required=True, type=Path)
        return cls(parser.parse_args().run_dir).run()


if __name__ == "__main__":
    raise SystemExit(FoundationRunner.cli())