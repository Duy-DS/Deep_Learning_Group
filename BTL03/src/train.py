"""CLI thống nhất cho các runner hiện có và WP02 all-model smoke-test."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .engine import BaseTrainer, TrainerCallback

__all__ = ["BaseTrainer", "TrainerCallback", "main"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectTrainingCLI:
    """Chuyển config YAML sang runner nhóm hiện có, giữ nguyên API từng runner."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        """Giữ gốc repository để nạp YAML và tạo runner theo nhóm."""
        self.project_root = Path(project_root).resolve()

    def _load(self, config_path: str) -> tuple[Path, dict[str, Any]]:
        """Nạp YAML từ đường dẫn tuyệt đối hoặc tương đối với gốc dự án."""
        path = Path(config_path)
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy YAML cấu hình: {path}")
        return path, yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    @staticmethod
    def _member_name(env_path: Path, member_id: str) -> str:
        """Đọc chính xác biến MEMBER_XX_NAME; không suy đoán người chạy."""
        if not env_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file định danh: {env_path}")
        key = f"{member_id}_NAME"
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == key:
                result = value.strip().strip('"').strip("'")
                if result:
                    return result
        raise ValueError(f".env thiếu {key}; không thể ghi attribution/log chính xác.")

    def run(
        self,
        config_path: str | None,
        *,
        task: str = "configured",
        seed: int | None = None,
        epochs: int | None = None,
        smoke_test: bool = False,
        member_alias: str | None = None,
    ) -> dict[str, Any]:
        """Dispatch cấu hình YAML qua runner nhóm và trả metadata chạy.

        Args:
            config_path: File YAML một trong mười cấu hình WP02; None khi task=all.
            task: `configured` để dispatch config, hoặc `all` chạy mười smoke cases.
            seed: Override seed tường minh; mặc định seed trong YAML.
            epochs: Ghi đè epoch; smoke-test mặc định đúng một epoch.
            smoke_test: Dữ liệu tổng hợp schema-đúng và namespace tạm; không phải baseline.
            member_alias: Tên người chạy; thiếu thì đọc MEMBER_XX_NAME từ `.env`.
        """
        if task == "all":
            if not smoke_test:
                raise ValueError("--task all chỉ được chạy qua --smoke-test để tránh vô tình huấn luyện dữ liệu thật.")
            from scripts.wp02_smoke import main as run_all_smokes

            return {"exit_code": run_all_smokes()}
        if not config_path:
            raise ValueError("Cần --config khi không dùng --task all.")
        path, config = self._load(config_path)
        run_seed = int(seed if seed is not None else config.get("seed", 42))
        run_epochs = 1 if smoke_test and epochs is None else epochs
        model_name = str(config.get("model_name", ""))
        model_group = str(config.get("model_group", ""))

        if smoke_test:
            # Smoke-test dùng chung harness thuần `BaseTrainer`, chuyển mọi log
            # và checkpoint vào TemporaryDirectory; không gọi runner nhóm vốn
            # có đường dẫn artifact chính thức.
            from scripts.wp02_smoke import WP02_CASES, WP02SmokeHarness

            if model_group == "04_training_utilities":
                key = "gender_cnn_callbacks" if model_name == "HumanGenderCNN" else "emotion_cnn_callbacks"
            else:
                key = {
                    "FuncApxmMLP": "func_apxm_mlp",
                    "MNISTMLP": "mnist_mlp",
                    "MNISTCNN": "mnist_cnn",
                    "HumanGenderCNN": "human_gender_cnn",
                    "EmotionCNN": "emotion_cnn",
                    "MNISTCNNLSTM": "mnist_cnn_lstm",
                    "SegNet": "segnet",
                    "FCNObjectDetector": "fcn_detector",
                }.get(model_name)
            case = next((item for item in WP02_CASES if item["key"] == key), None)
            if case is None:
                raise ValueError(f"Không có cấu hình smoke cho model={model_name!r}, group={model_group!r}.")
            harness = WP02SmokeHarness(self.project_root, run_seed)
            with __import__("tempfile").TemporaryDirectory(prefix=f"wp02_{key}_") as temporary:
                summary_master = self.project_root / "experiments/csv_logs/master_experiment_runs.csv"
                master_bytes = summary_master.read_bytes()
                original_alias = os.environ.get("MEMBER_ALIAS")
                os.environ["MEMBER_ALIAS"] = member_alias or "Duy-DS"
                try:
                    result = harness.run_case(case, Path(temporary))
                finally:
                    if original_alias is None:
                        os.environ.pop("MEMBER_ALIAS", None)
                    else:
                        os.environ["MEMBER_ALIAS"] = original_alias
                if summary_master.read_bytes() != master_bytes:
                    raise RuntimeError("Smoke-test đã thay đổi master CSV chính thức.")
                result["config"] = str(path)
                result["requested_epochs"] = run_epochs
                return result

        if run_seed != int(config.get("seed", 42)):
            raise ValueError(
                "WP02 dùng seed 42 theo cấu hình; override seed khác cần cập nhật "
                "cấu hình/runners của nhóm trước khi chạy để giữ tái lập."
            )

        if model_name == "FuncApxmMLP":
            from scripts.run_group1_mlp import Group1MLPRunner

            runner = Group1MLPRunner(self.project_root, member_alias or self._member_name(self.project_root / ".env", "MEMBER_01"))
            # Sweep runner ghi vào master/project-root; chỉ parity + one-step smoke
            # được thực thi từ smoke harness để đảm bảo mọi artifacts ở tmp_path.
            result = runner.run_gaussian_sweeps(sweep_filter=None)
            return {"task": "func_apxm_mlp", "runs": len(result), "config": str(path)}

        if model_name in {"MNISTMLP", "MNISTCNN", "HumanGenderCNN", "EmotionCNN"} and model_group in {"01_mlp", "02_cnn_mlp"}:
            if model_name == "MNISTMLP":
                from scripts.run_group1_mlp import Group1MLPRunner

                return Group1MLPRunner(self.project_root, member_alias or "Duy-DS").run_mnist(epochs=run_epochs)
            key = {"MNISTCNN": "mnist", "HumanGenderCNN": "gender", "EmotionCNN": "emotion"}[model_name]
            from scripts.run_group2_cnn import Group2CNNRunner

            group_runner = Group2CNNRunner(self.project_root, member_alias or self._member_name(self.project_root / ".env", "MEMBER_02"))
            return group_runner.run(key, epochs=run_epochs, smoke=smoke_test)

        if model_name == "MNISTCNNLSTM":
            from scripts.run_group3_cnn_lstm import Group3CNNLSTMRunner

            runner = Group3CNNLSTMRunner(
                self.project_root,
                member_alias or self._member_name(self.project_root / ".env", "MEMBER_03"),
            )
            return runner.run(epochs=run_epochs, smoke_test=False)

        if model_group == "04_training_utilities" and model_name in {"HumanGenderCNN", "EmotionCNN"}:
            from scripts.run_group4_utilities import Group4UtilitiesRunner

            key = str(config.get("model_key", "gender" if model_name == "HumanGenderCNN" else "emotion"))
            runner = Group4UtilitiesRunner(
                self.project_root,
                member_alias or self._member_name(self.project_root / ".env", "MEMBER_04"),
            )
            return runner.run(key, epochs=run_epochs, smoke_test=False)

        if model_name in {"SegNet", "FCNObjectDetector"}:
            # Nhóm 5 có YAML lồng nhau; runner sở hữu cách nạp path/data cụ thể.
            from scripts.run_group5_others import Group5OthersRunner

            runner = Group5OthersRunner(self.project_root, member_alias or self._member_name(self.project_root / ".env", "MEMBER_05"))
            task = "segnet" if model_name == "SegNet" else "detection"
            return runner.run(task=task, synthetic=smoke_test, epochs=run_epochs)[0]

        raise ValueError(f"Không có runner WP02 cho model_name={model_name!r}, model_group={model_group!r} trong {path}.")

def main(argv: list[str] | None = None) -> int:
    """Điểm vào `python -m src.train --config ...` cho runner WP02."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Đường dẫn YAML trong configs/.")
    parser.add_argument("--task", choices=("configured", "all"), default="configured")
    parser.add_argument("--seed", type=int, default=None, help="Override seed; mặc định YAML.")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch; --smoke-test mặc định 1.")
    parser.add_argument("--smoke-test", action="store_true", help="Dữ liệu giả đúng schema; không phải metric khoa học.")
    parser.add_argument("--member-alias", default=None, help="Override tên người chạy bằng giá trị thật.")
    args = parser.parse_args(argv)
    if args.task == "configured" and not args.config:
        parser.error("--config là bắt buộc khi --task=configured.")
    if args.task == "all" and args.config:
        parser.error("Không truyền --config cùng --task=all.")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs phải là số nguyên dương.")
    result = ProjectTrainingCLI().run(
        args.config, task=args.task, seed=args.seed, epochs=args.epochs,
        smoke_test=args.smoke_test, member_alias=args.member_alias,
    )
    print(result)
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
