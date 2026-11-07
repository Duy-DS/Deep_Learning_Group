"""Sinh các tập dữ liệu Gaussian xác định và các artifacts khảo sát phân bố."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import GaussianDataGenerator
from src.visualizer import Visualizer


def load_yaml(relative_path: str) -> dict[str, Any]:
    """Tải file YAML cấu hình dự án bằng ``yaml.safe_load``.

    Tham số:
        relative_path: Đường dẫn file cấu hình tương đối so với ``PROJECT_ROOT``.

    Trả về:
        Dictionary cấu hình đã giải nạp.

    Ngoại lệ:
        ValueError: Nếu nội dung YAML cấp cao nhất không phải là một mapping/dict.
    """
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as stream:
        content = yaml.safe_load(stream)
    if not isinstance(content, dict):
        raise ValueError(f"Kỳ vọng YAML mapping: {relative_path}")
    return content


def sha256_file(path: Path) -> str:
    """Tính mã băm SHA-256 cho một file.

    Tham số:
        path: Đường dẫn file cần tính hash.

    Trả về:
        Chuỗi biểu diễn thập lục phân (hex digest).
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_splits(
    splits: Mapping[str, pd.DataFrame],
    generator: GaussianDataGenerator,
) -> pd.DataFrame:
    """Kiểm tra tính hợp lệ của các tập dữ liệu đã sinh và xây dựng bảng kiểm tra toàn vẹn.

    Tham số:
        splits: Ánh xạ chứa các DataFrame đã sinh theo tên từng tập dữ liệu.
        generator: Đối tượng sinh dữ liệu Gaussian dùng để đối chiếu công thức.

    Trả về:
        DataFrame chứa kết quả kiểm tra PASS/FAIL cho từng hạng mục.
    """
    rows: list[dict[str, str]] = []
    coordinate_sets: dict[str, set[tuple[float, float]]] = {}
    for split_name, frame in splits.items():
        generator.validate_dataframe(frame)
        coordinates = set(map(tuple, frame[["x1", "x2"]].to_numpy()))
        coordinate_sets[split_name] = coordinates
        rows.extend(
            [
                {
                    "check": f"{split_name}_row_count",
                    "status": "PASS",
                    "details": str(len(frame)),
                },
                {
                    "check": f"{split_name}_finite_domain_density",
                    "status": "PASS",
                    "details": "columns x1,x2,y finite and formula-valid",
                },
            ]
        )

    names = list(coordinate_sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            overlap = coordinate_sets[left_name] & coordinate_sets[right_name]
            rows.append(
                {
                    "check": f"{left_name}_vs_{right_name}_coordinate_overlap",
                    "status": "PASS" if not overlap else "FAIL",
                    "details": str(len(overlap)),
                }
            )
            if overlap:
                raise ValueError(
                    f"Phát hiện trùng lặp tọa độ giữa {left_name} và {right_name}."
                )
    return pd.DataFrame(rows)


def write_splits(
    splits: Mapping[str, pd.DataFrame],
    data_config: Mapping[str, Any],
) -> dict[str, Path]:
    """Ghi dữ liệu các tập train, validation, và test ra các file CSV.

    Tham số:
        splits: Ánh xạ chứa các DataFrame theo tên tập dữ liệu.
        data_config: Mục ``data`` trong file cấu hình ``baseline.yaml``.

    Trả về:
        Ánh xạ tên tập dữ liệu sang đường dẫn Path của file CSV đã ghi.
    """
    path_keys = {
        "train_full": "train_full_path",
        "val": "val_path",
        "test": "test_path",
    }
    written: dict[str, Path] = {}
    for split_name, frame in splits.items():
        output_path = PROJECT_ROOT / data_config[path_keys[split_name]]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(
            output_path,
            index=False,
            float_format=data_config["csv_float_format"],
            encoding="utf-8",
        )
        written[split_name] = output_path
    return written


def main() -> None:
    """Thực thi luồng sinh dữ liệu và trực quan hóa cho các nhiệm vụ T0201-T0203."""
    baseline = load_yaml("configs/baseline.yaml")
    data_config = baseline["data"]
    output_config = baseline["outputs"]
    visual_config = baseline["visualization"]

    split_sizes = {
        "train_full": int(data_config["train_full_size"]),
        "val": int(data_config["val_size"]),
        "test": int(data_config["test_size"]),
    }
    generator = GaussianDataGenerator(
        mu=data_config["mu"],
        cov=data_config["cov"],
        x1_range=tuple(data_config["x1_range"]),
        x2_range=tuple(data_config["x2_range"]),
    )
    splits = generator.generate_splits(
        split_sizes=split_sizes,
        seed=int(data_config["generation_seed"]),
    )
    audit_table = audit_splits(splits, generator)
    written_splits = write_splits(splits, data_config)

    tables_dir = PROJECT_ROOT / output_config["tables_dir"]
    figures_dir = PROJECT_ROOT / output_config["figures_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    visualizer = Visualizer(
        figures_dir=figures_dir,
        dpi=int(visual_config["dpi"]),
        figure_format=str(visual_config["format"]),
    )
    summary_table = visualizer.build_summary_table(splits)
    summary_path = tables_dir / "data_summary.csv"
    audit_path = tables_dir / "data_integrity_checks.csv"
    summary_table.to_csv(summary_path, index=False, encoding="utf-8")
    audit_table.to_csv(audit_path, index=False, encoding="utf-8")

    data_distribution_path = visualizer.plot_data_distribution(splits)
    true_surface_path = visualizer.plot_true_surface(
        generator=generator,
        grid_size=tuple(visual_config["surface_grid_size"]),
    )

    manifest_path = tables_dir / "data_generation_manifest.json"
    manifest = {
        "task_ids": ["T0201", "T0202", "T0203"],
        "schema_version": baseline["schema_version"],
        "generation_seed": data_config["generation_seed"],
        "rng_algorithm": data_config["rng_algorithm"],
        "split_strategy": data_config["split_strategy"],
        "split_sizes": split_sizes,
        "mu": data_config["mu"],
        "cov": data_config["cov"],
        "determinant": generator.determinant,
        "inverse_covariance": generator.inverse_covariance.tolist(),
        "data_files": {
            split: {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for split, path in written_splits.items()
        },
        "tables": {
            "summary": summary_path.relative_to(PROJECT_ROOT).as_posix(),
            "integrity": audit_path.relative_to(PROJECT_ROOT).as_posix(),
        },
        "figures": {
            "data_distribution": (
                data_distribution_path.relative_to(PROJECT_ROOT).as_posix()
            ),
            "true_surface": true_surface_path.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Đã sinh dữ liệu Gaussian và các artifacts cho T0203")
    for split_name, path in written_splits.items():
        print(f"{split_name}: {path.relative_to(PROJECT_ROOT)}")
    print(f"Bảng tổng kết: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Kiểm tra toàn vẹn: {audit_path.relative_to(PROJECT_ROOT)}")
    print(f"Manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"Đồ thị phân bố: {data_distribution_path.relative_to(PROJECT_ROOT)}")
    print(f"Đồ thị bề mặt: {true_surface_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
