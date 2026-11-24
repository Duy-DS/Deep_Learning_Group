"""Kiểm thử scheduler LR và controlled recipes EXP-12 cho FER2013."""

import ast
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.train.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments" / "ablation"
CONFIG_NAMES = (
    "exp12_sched_step_or_plateau.yaml",
    "exp12_sched_cosine.yaml",
    "exp12_sched_warmup_cosine.yaml",
)


class DatasetStub:
    """Dataset giả với nhãn đủ bảy lớp, không đọc CSV thật."""

    labels = list(range(7))


class LoaderStub:
    """DataLoader giả chỉ cung cấp dataset cho bước khởi tạo Trainer."""

    dataset = DatasetStub()


def _make_trainer(config, monkeypatch, tmp_path):
    """Khởi tạo Trainer CPU không tải pretrained weights hoặc FER2013."""
    local_config = dict(config)
    local_config.update(
        pretrained=False,
        checkpoint_path=str(tmp_path / "best.pth"),
        mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
    )
    monkeypatch.setattr(
        "src.train.trainer.get_dataloaders",
        lambda **_: (LoaderStub(), [], []),
    )
    monkeypatch.setattr("src.train.trainer.MLflowLogger", lambda **_: object())
    return Trainer(local_config)


@pytest.mark.parametrize(
    "name, expected_type",
    [
        ("step", torch.optim.lr_scheduler.StepLR),
        ("ReduceLROnPlateau", torch.optim.lr_scheduler.ReduceLROnPlateau),
        ("cosine", torch.optim.lr_scheduler.CosineAnnealingLR),
        ("WarmupCosine", torch.optim.lr_scheduler.SequentialLR),
    ],
)
def test_scheduler_aliases_build_and_update_learning_rate(
    name, expected_type, monkeypatch, tmp_path
):
    """Alias scheduler khởi tạo đúng lớp PyTorch và thay đổi LR theo epoch."""
    config = yaml.safe_load((CONFIG_DIR / CONFIG_NAMES[1]).read_text(encoding="utf-8"))
    config["scheduler"] = name
    config["scheduler_params"] = {
        "step_size": 1,
        "gamma": 0.5,
        "mode": "min",
        "factor": 0.5,
        "patience": 0,
        "min_lr": 1e-8,
        "T_max": 4,
        "eta_min": 1e-8,
        "warmup_epochs": 2,
        "start_factor": 0.25,
    }
    trainer = _make_trainer(config, monkeypatch, tmp_path)
    scheduler = trainer.lr_scheduler
    assert isinstance(scheduler, expected_type)

    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(1.0)
        scheduler.step(2.0)
    else:
        for _ in range(2):
            trainer.optimizer.step()
            scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] != pytest.approx(initial_lr)


@pytest.mark.parametrize(
    "alias, expected_type",
    [
        ("plateau", torch.optim.lr_scheduler.ReduceLROnPlateau),
        ("reduce_on_plateau", torch.optim.lr_scheduler.ReduceLROnPlateau),
        ("StepLR", torch.optim.lr_scheduler.StepLR),
        ("cosine_annealing", torch.optim.lr_scheduler.CosineAnnealingLR),
    ],
)
def test_scheduler_legacy_and_documented_aliases(
    alias, expected_type, monkeypatch, tmp_path
):
    """Các tên viết tắt và YAML ReduceLROnPlateau cũ tiếp tục tương thích."""
    config = yaml.safe_load((CONFIG_DIR / CONFIG_NAMES[0]).read_text(encoding="utf-8"))
    config["scheduler"] = alias
    trainer = _make_trainer(config, monkeypatch, tmp_path)
    assert isinstance(trainer.lr_scheduler, expected_type)


def test_warmup_cosine_ramps_then_decays(monkeypatch, tmp_path):
    """Warmup tăng LR tuyến tính tới mức nền rồi cosine giảm theo epoch."""
    config = yaml.safe_load((CONFIG_DIR / CONFIG_NAMES[2]).read_text(encoding="utf-8"))
    trainer = _make_trainer(config, monkeypatch, tmp_path)
    rates = [trainer.optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        rates.append(trainer.optimizer.param_groups[0]["lr"])
    assert rates[0] < rates[1] < rates[2] < rates[3]
    assert rates[4] < rates[3]


def test_exp12_recipes_change_only_scheduler_and_have_unique_run_names(
    monkeypatch, tmp_path
):
    """Recipe EXP-12 giữ cố định mọi tham số, chỉ đổi lịch LR và định danh."""
    configs = [yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8")) for name in CONFIG_NAMES]
    invariant_keys = (
        "architecture", "pretrained", "freeze_backbone", "dropout_rate", "seed",
        "image_size", "in_channels", "num_classes", "normalization", "transform_mode",
        "use_augmentation", "augmentation_mode", "augmentation", "split_counts",
        "optimizer", "lr", "weight_decay", "batch_size", "epochs", "patience", "loss",
    )
    for key in invariant_keys:
        assert all(config[key] == configs[0][key] for config in configs), key
    assert {config["exp_id"] for config in configs} == {"EXP-12"}
    assert {config["rq_tag"] for config in configs} == {"RQ3_Ablation_LRScheduler"}
    trainers = [_make_trainer(config, monkeypatch, tmp_path / str(index)) for index, config in enumerate(configs)]
    names = [
        Trainer._format_run_name(
            exp_id=item.exp_id,
            architecture=item.architecture,
            optimizer=item.optimizer_name,
            pretrained=item.pretrained,
            image_size=item.image_size,
            augmentation_mode=item.augmentation_mode,
            loss_name=item.loss_name,
            scheduler_name=item.config["scheduler"],
        )
        for item in trainers
    ]
    assert len(set(names)) == len(configs)

    for trainer in trainers:
        trainer.model.train()
        logits = trainer.model(torch.randn(2, 3, 128, 128, device=trainer.device))
        loss = trainer.criterion(logits, torch.tensor([0, 1], device=trainer.device))
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        if isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            trainer.lr_scheduler.step(1.0)
        else:
            trainer.lr_scheduler.step()
        assert logits.shape == (2, 7)
        assert torch.isfinite(loss)


def test_t0403_notebooks_are_valid_json_and_code_cells_parse():
    """Notebook Colab/Kaggle hợp lệ, có recipe T0403 và CSV schema chuẩn."""
    notebooks = ROOT / "notebooks"
    for platform in ("colab", "kaggle"):
        path = notebooks / platform / f"run_t0403_{platform}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        sources = ["".join(cell["source"]) for cell in cells]
        for index, source in enumerate(sources):
            ast.parse(source, filename=f"t0403-{platform}-{index}")
        joined = "\n".join(sources)
        assert all(name in joined for name in CONFIG_NAMES)
        assert "t0403_ablation_scheduler_24col.csv" in joined
        assert "MLflowLogger.CSV_COLUMNS" in joined
        assert "scheduler_name=cfg['scheduler']" in joined
        assert "FINISHED" in joined
        if platform == "colab":
            assert "/content/drive/MyDrive/BTL02/T0403_Scheduler" in joined
            assert "stable_age_seconds" in joined and "torchvision" in joined
        else:
            assert "/kaggle/working/BTL02/T0403_Scheduler" in joined
            assert "x-access-token:" in joined and "KAGGLE_API_TOKEN" in joined
            assert "/kaggle/input/datasets" in joined and "restore_finished_runs" in joined