"""Kiểm thử controlled augmentation RQ3 và bảo vệ split đánh giá FER2013."""

from pathlib import Path

import ast
import json
import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms
import yaml

from src.data.transforms import FER2013TransformBuilder
from src.data.transforms_sota import SOTATransformBuilder
from src.data import dataset as dataset_module
from src.train.trainer import Trainer


@pytest.fixture
def facial_image():
    """Ảnh tổng hợp có cấu trúc không gian để kiểm tra phép biến đổi."""
    array = np.zeros((48, 48), dtype=np.uint8)
    array[8:40, 12:36] = np.arange(32, dtype=np.uint8)[:, None] * 7
    return Image.fromarray(array, mode="L")


@pytest.mark.parametrize("mode", ["none", "basic", "advanced"])
def test_sota_augmentation_modes_preserve_rgb_tensor_shape(facial_image, mode):
    """Ba mode đều tạo tensor RGB 128×128 đúng dtype."""
    builder = SOTATransformBuilder(
        image_size=128,
        aug_config={
            "mode": mode,
            "horizontal_flip_p": 0.5,
            "rotation_degrees": 10,
            "translate": (0.08, 0.08),
            "color_jitter": 0,
            "random_erasing_p": 1.0,
        },
    )
    result = builder.build_train_transform()(facial_image)
    assert result.shape == (3, 128, 128)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_sota_none_mode_contains_no_random_training_transforms(facial_image):
    """Mode none không đưa phép biến đổi ngẫu nhiên vào train transform."""
    builder = SOTATransformBuilder(image_size=128, aug_config={"mode": "none"})
    random_types = (transforms.RandomHorizontalFlip, transforms.RandomRotation,
                    transforms.RandomAffine, transforms.ColorJitter, transforms.RandomErasing)
    assert not any(isinstance(item, random_types) for item in builder.build_train_transform().transforms)


def test_sota_basic_uses_requested_flip_rotation_and_translation():
    """Mode basic giới hạn ở flip, rotation 10° và affine shift 8%."""
    builder = SOTATransformBuilder(
        image_size=128,
        aug_config={"mode": "basic", "horizontal_flip_p": 0.5,
                    "rotation_degrees": 10, "translate": (0.08, 0.08),
                    "random_erasing_p": 1.0},
    )
    pipeline = builder.build_train_transform().transforms
    assert any(isinstance(item, transforms.RandomHorizontalFlip) and item.p == 0.5 for item in pipeline)
    rotation = next(item for item in pipeline if isinstance(item, transforms.RandomRotation))
    assert rotation.degrees == [-10.0, 10.0]
    affine = next(item for item in pipeline if isinstance(item, transforms.RandomAffine))
    assert affine.translate == (0.08, 0.08)
    assert not any(isinstance(item, transforms.RandomErasing) for item in pipeline)


def test_advanced_random_erasing_operates_on_tensor_after_normalize(facial_image):
    """RandomErasing chỉ nhận Tensor và dùng đúng xác suất/diện tích/ratio/value."""
    builder = SOTATransformBuilder(
        image_size=128,
        aug_config={"mode": "advanced", "random_erasing_p": 1.0,
                    "random_erasing_scale": (0.02, 0.15),
                    "random_erasing_ratio": (0.3, 3.3),
                    "random_erasing_value": "random"},
    )
    pipeline = builder.build_train_transform().transforms
    erasing = next(item for item in pipeline if isinstance(item, transforms.RandomErasing))
    assert pipeline.index(erasing) > next(i for i, item in enumerate(pipeline) if isinstance(item, transforms.ToTensor))
    assert erasing.p == 1.0
    assert erasing.scale == (0.02, 0.15)
    assert erasing.ratio == (0.3, 3.3)
    assert erasing.value == "random"
    torch.manual_seed(7)
    plain = transforms.Compose(pipeline[:pipeline.index(erasing)])(facial_image)
    torch.manual_seed(7)
    erased = builder.build_train_transform()(facial_image)
    assert erased.shape == plain.shape
    assert not torch.equal(erased, plain)


@pytest.mark.parametrize("mode", ["none", "basic", "advanced"])
def test_public_and_private_eval_transform_is_deterministic_and_unaugmented(facial_image, mode):
    """Eval transform không chứa random ops dù bật mode augmentation cho train."""
    builder = SOTATransformBuilder(
        image_size=128,
        aug_config={"mode": mode, "random_erasing_p": 1.0},
    )
    evaluation = builder.build_eval_transform()
    random_types = (transforms.RandomCrop, transforms.RandomHorizontalFlip,
                    transforms.RandomRotation, transforms.RandomAffine,
                    transforms.ColorJitter, transforms.RandomErasing)
    assert not any(isinstance(item, random_types) for item in evaluation.transforms)
    first, second = evaluation(facial_image), evaluation(facial_image)
    assert torch.equal(first, second)


@pytest.mark.parametrize("mode", ["none", "basic", "advanced"])
def test_custom_pipeline_accepts_yaml_mode_mapping(facial_image, mode):
    """Pipeline ảnh xám legacy cũng đọc được augmentation.mode từ YAML."""
    pipeline = FER2013TransformBuilder().build_train_transforms({"mode": mode})
    result = pipeline(facial_image)
    assert result.shape == (1, 48, 48)
    assert any(isinstance(item, transforms.RandomErasing) for item in pipeline.transforms) == (mode == "advanced")


@pytest.mark.parametrize("mode", ["none", "basic", "advanced"])
def test_dataloaders_apply_augmentation_only_to_training_split(mode, monkeypatch):
    """Tạo ba split qua get_dataloaders và xác nhận eval không nhận train pipeline."""
    seen = {}

    class DatasetSpy(torch.utils.data.Dataset):
        """Dataset giả chỉ ghi nhận split/transform, không đọc dữ liệu thực."""

        def __init__(self, csv_path, split, transform):
            self.split = split
            self.transform = transform
            seen[split] = transform

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.zeros(3, 128, 128), torch.tensor(index % 2)

    monkeypatch.setattr(dataset_module, "FER2013Dataset", DatasetSpy)
    dataset_module.get_dataloaders(
        csv_path="unused.csv", batch_size=2, image_size=128, in_channels=3,
        transform_mode="sota", augmentation_mode=mode,
        aug_config={"mode": mode, "random_erasing_p": 1.0},
    )
    assert seen["train"].__class__ is transforms.Compose
    assert seen["val"] is seen["test"]
    random_types = (transforms.RandomHorizontalFlip, transforms.RandomRotation,
                    transforms.RandomAffine, transforms.ColorJitter, transforms.RandomErasing)
    assert any(isinstance(item, random_types) for item in seen["train"].transforms) == (mode != "none")
    assert not any(isinstance(item, random_types) for item in seen["val"].transforms)


def test_three_yaml_recipes_differ_only_in_augmentation_mode():
    """Ba YAML giữ nguyên kiến trúc/recipe; chỉ mode augmentation và experiment label đổi."""
    root = Path(__file__).resolve().parents[1] / "experiments" / "ablation"
    filenames = ("exp09_aug_none.yaml", "exp09_aug_basic.yaml", "exp10_aug_advanced.yaml")
    configs = [yaml.safe_load((root / name).read_text(encoding="utf-8")) for name in filenames]
    assert [config["augmentation_mode"] for config in configs] == ["none", "basic", "advanced"]
    assert [config["exp_id"] for config in configs] == ["EXP-09", "EXP-09", "EXP-10"]
    controlled = (
        "architecture", "pretrained", "image_size", "seed", "split_counts", "optimizer", "lr",
        "weight_decay", "scheduler", "scheduler_params", "batch_size", "epochs", "patience",
    )
    for key in controlled:
        assert len({str(config[key]) for config in configs}) == 1, key
    for config in configs:
        assert config["augmentation"]["mode"] == config["augmentation_mode"]


@pytest.mark.parametrize(
    "filename, expected_mode",
    [
        ("exp09_aug_none.yaml", "none"),
        ("exp09_aug_basic.yaml", "basic"),
        ("exp10_aug_advanced.yaml", "advanced"),
    ],
)
def test_trainer_yaml_builds_efficientnet_and_runs_forward_backward(
    filename, expected_mode, monkeypatch
):
    """Mỗi recipe khởi tạo Trainer và chạy một bước forward/backward CPU."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "experiments" / "ablation" / filename).read_text(encoding="utf-8"))
    config["pretrained"] = False
    monkeypatch.setattr("src.train.trainer.get_dataloaders", lambda **kwargs: ([], [], []))
    trainer = Trainer(config)
    assert trainer.augmentation_mode == expected_mode
    assert trainer.use_augmentation is (expected_mode != "none")
    assert "aug-" + expected_mode in trainer._format_run_name(
        trainer.exp_id, trainer.architecture, trainer.optimizer_name,
        pretrained=trainer.pretrained, image_size=trainer.image_size,
        augmentation_mode=trainer.augmentation_mode,
    )
    trainer.model.train()
    images = torch.randn(2, 3, 128, 128, device=trainer.device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=trainer.device)
    logits = trainer.model(images)
    loss = trainer.criterion(logits, labels)
    loss.backward()
    assert logits.shape == (2, 7)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in trainer.model.parameters() if parameter.requires_grad)


def test_kaggle_backup_restores_only_finished_missing_runs_without_overwrite(tmp_path):
    """Hàm recovery notebook chỉ thêm run FINISHED còn thiếu và giữ nguyên local run."""
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "notebooks/kaggle/run_t0401_kaggle.ipynb")
        .read_text(encoding="utf-8")
    )
    source = "".join(notebook["cells"][3]["source"])
    start = source.index("def restore_finished_runs")
    end = source.index("restored_run_ids = []")
    function_source = source[start:end]
    assert 'filter_string="attributes.status = \'FINISHED\'"' in function_source
    assert "destination_run.exists()" in function_source

    backup = tmp_path / "backup" / "mlruns"
    destination = tmp_path / "destination" / "mlruns"
    from mlflow.tracking import MlflowClient

    source_client = MlflowClient(tracking_uri=backup.resolve().as_uri())
    experiment_id = source_client.create_experiment("backup-test")
    finished_missing = source_client.create_run(experiment_id).info.run_id
    source_client.set_terminated(finished_missing, status="FINISHED")
    finished_existing = source_client.create_run(experiment_id).info.run_id
    source_client.set_terminated(finished_existing, status="FINISHED")
    running = source_client.create_run(experiment_id).info.run_id
    source_client.set_terminated(running, status="KILLED")
    source_experiment = backup / experiment_id
    for run_id, value in ((finished_missing, "backup-finished"),
                          (finished_existing, "backup-should-not-overwrite"),
                          (running, "not-finished")):
        (source_experiment / run_id / "marker.txt").write_text(value, encoding="utf-8")

    destination_experiment = destination / experiment_id
    destination_experiment.mkdir(parents=True)
    (destination_experiment / "meta.yaml").write_bytes(
        (source_experiment / "meta.yaml").read_bytes()
    )
    local_finished = destination_experiment / finished_existing
    local_finished.mkdir(parents=True)
    (local_finished / "marker.txt").write_text("local-finished", encoding="utf-8")

    namespace = {"Path": Path, "shutil": __import__("shutil")}
    exec(function_source, namespace)
    restored = namespace["restore_finished_runs"](backup, destination)
    assert restored == [finished_missing]
    assert (local_finished / "marker.txt").read_text(encoding="utf-8") == "local-finished"
    assert not (destination_experiment / running).exists()
    assert (destination_experiment / finished_missing / "marker.txt").read_text(encoding="utf-8") == "backup-finished"

def test_colab_and_kaggle_runner_notebooks_are_valid_and_have_eight_cells():
    """Hai notebook JSON h?p l?, ??ng t?m cell v? code cell parse ???c."""
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for platform in ("colab", "kaggle"):
        notebook = json.loads((root / platform / f"run_t0401_{platform}.ipynb").read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert len(notebook["cells"]) == 8
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert len(code_cells) == 7
        for index, cell in enumerate(code_cells, 1):
            ast.parse("".join(cell["source"]), filename=f"t0401-{platform}-cell-{index}")
        source = "\n".join("".join(cell["source"]) for cell in code_cells)
        assert all(name in source for name in ("exp09_aug_none.yaml", "exp09_aug_basic.yaml", "exp10_aug_advanced.yaml"))
        assert "sanitize_to_python_scalars" in source
        assert "FINISHED" in source and "MLflowLogger.CSV_COLUMNS" in source
        if platform == "kaggle":
            assert "UserSecretsClient" in source and "x-access-token:" in source
            assert "/kaggle/working/BTL02/T0401" in source
            assert "fer2013.tar.gz" in source and "KAGGLE_API_TOKEN" in source
            assert "meta.yaml" in source
        else:
            assert "LOCAL_MLRUNS" in source and "stable_age_seconds" in source
            assert "drive.mount" in source and "torchvision" in source
            assert "sync_and_export_results(sync_checkpoints=True)" in source
            assert "sync_tree_new_files(LOCAL_CHECKPOINTS, DRIVE_CHECKPOINTS)" in source
            assert source.count('print("Drive CSV:", DRIVE_CSV)') == 1
        if platform == "kaggle":
            assert "filter_string=\"attributes.status = 'FINISHED'\"" in source
            assert "shutil.copytree(source_run, destination_run)" in source
            assert "if not source_run.is_dir() or destination_run.exists():" in source
            assert '(current_path / "0" / "meta.yaml").is_file()' in source
