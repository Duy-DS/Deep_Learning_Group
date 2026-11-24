"""Kiểm thử hàm mất mát và recipe EXP-11 xử lý mất cân bằng FER2013."""

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import yaml

from src.train.losses import FocalLoss, LossFactory, compute_class_weights
from src.train.trainer import Trainer


def test_class_weights_inverse_frequency_are_normalized():
    """Lớp ít mẫu có trọng số cao hơn và trung bình trọng số bằng một."""
    labels = [0] * 8 + [1] * 2 + [2] * 1
    weights = compute_class_weights(labels, num_classes=3)
    assert weights[2] > weights[1] > weights[0]
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize("labels", [[], [0, 1, 3], [-1, 0, 1]])
def test_class_weight_rejects_empty_missing_or_invalid_training_labels(labels):
    """Tính trọng số dừng an toàn nếu tập Training thiếu lớp hoặc nhãn sai."""
    with pytest.raises(ValueError):
        compute_class_weights(labels, num_classes=3)


def test_focal_loss_gamma_zero_equals_cross_entropy():
    """Khi gamma bằng không, focal modulation bằng một và còn lại CE chuẩn."""
    logits = torch.randn(9, 7, requires_grad=True)
    targets = torch.randint(0, 7, (9,))
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, targets), nn.CrossEntropyLoss()(logits, targets))


def test_focal_loss_downweights_confident_correct_examples_and_has_finite_gradient():
    """Mẫu dự đoán tự tin nhận focal loss nhỏ hơn CE và gradient hữu hạn."""
    targets = torch.tensor([0, 1])
    logits = torch.tensor([[8.0, 0.0], [0.2, 0.0]], requires_grad=True)
    ce = nn.CrossEntropyLoss(reduction="none")(logits, targets)
    focal = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
    assert focal[0] < ce[0] * 1e-5
    assert focal[1] < ce[1]
    focal.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_supports_class_vector_alpha_and_reductions():
    """Alpha vector theo lớp, reduction none/sum và device buffer hoạt động."""
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    criterion = FocalLoss(gamma=2, alpha=torch.tensor([0.25, 2.0]), reduction="none")
    assert criterion.alpha.device == logits.device
    values = criterion(logits, targets)
    assert values.shape == (2,) and values[1] > values[0]
    assert torch.isfinite(FocalLoss(gamma=2, alpha=0.25, reduction="sum")(logits, targets))


def test_loss_factory_builds_all_modes_and_requires_training_labels_for_weights():
    """Factory tạo CE/Weighted CE/Focal và không ước lượng weight khi thiếu train labels."""
    labels = [0] * 6 + [1] * 2 + [2]
    assert isinstance(LossFactory.create({"name": "cross_entropy"}), nn.CrossEntropyLoss)
    weighted = LossFactory.create({"name": "weighted_ce"}, labels, num_classes=3, device="cpu")
    assert isinstance(weighted, nn.CrossEntropyLoss)
    assert weighted.weight.device.type == "cpu"
    focal = LossFactory.create({"name": "focal", "gamma": 2.0}, num_classes=3)
    assert isinstance(focal, FocalLoss) and focal.gamma == 2.0
    with pytest.raises(ValueError, match="nhãn Training"):
        LossFactory.create({"name": "weighted_ce"}, num_classes=3)


def test_trainer_uses_only_train_dataset_labels_to_create_weighted_loss(monkeypatch, tmp_path):
    """Trainer chuyển nhãn train split vào factory; không truyền nhãn validation/test."""
    configs_dir = Path(__file__).resolve().parents[1] / "experiments/ablation"
    config = yaml.safe_load((configs_dir / "exp11_loss_weighted_ce.yaml").read_text(encoding="utf-8"))
    config["pretrained"] = False

    class DatasetStub:
        """Dataset giả chỉ chứa nhãn Training để kiểm tra ranh giới nguồn thống kê."""

        labels = [0, 0, 1, 2, 2, 2, 3, 4, 5, 6]

    class LoaderStub:
        dataset = DatasetStub()

    validation = object()
    testing = object()
    config["mlflow_tracking_uri"] = (tmp_path / "mlruns").as_uri()
    monkeypatch.setattr("src.train.trainer.get_dataloaders", lambda **_: (LoaderStub(), validation, testing))
    with patch("src.train.losses.compute_class_weights", wraps=compute_class_weights) as compute:
        trainer = Trainer(config)
    assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
    assert trainer.criterion.weight is not None
    assert compute.call_count == 1
    assert compute.call_args.args[0] is DatasetStub.labels


def test_legacy_yaml_without_loss_uses_standard_cross_entropy(monkeypatch, tmp_path):
    """Cấu hình cũ không có khóa loss vẫn khởi tạo CrossEntropy chuẩn."""
    class DatasetStub:
        labels = list(range(7))

    class LoaderStub:
        dataset = DatasetStub()

    monkeypatch.setattr("src.train.trainer.get_dataloaders", lambda **_: (LoaderStub(), [], []))
    with patch("src.train.trainer.MLflowLogger"):
        trainer = Trainer({
            "architecture": "SmallCNN",
            "pretrained": False,
            "mlflow_tracking_uri": (tmp_path / "legacy-mlruns").as_uri(),
        })
    assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
    assert trainer.criterion.weight is None
    assert isinstance(trainer.eval_criterion, nn.CrossEntropyLoss)


def test_exp11_configs_change_only_loss_and_trainer_smoke_backward(monkeypatch, tmp_path):
    """Ba YAML giữ nguyên recipe augmentation/model và chạy forward/backward CPU."""
    root = Path(__file__).resolve().parents[1]
    names = ("exp11_loss_ce.yaml", "exp11_loss_weighted_ce.yaml", "exp11_loss_focal.yaml")
    configs = [yaml.safe_load((root / "experiments/ablation" / name).read_text(encoding="utf-8")) for name in names]
    assert [item["loss"]["name"] for item in configs] == ["cross_entropy", "weighted_ce", "focal"]
    invariant_keys = (
        "exp_id", "exp_group", "rq_tag", "architecture", "pretrained", "image_size", "seed",
        "split_counts", "augmentation_mode", "augmentation", "optimizer", "lr", "weight_decay",
        "scheduler", "scheduler_params", "batch_size", "epochs", "patience",
    )
    for key in invariant_keys:
        assert len({str(config[key]) for config in configs}) == 1, key
    assert configs[0]["augmentation_mode"] == "basic"
    assert configs[2]["loss"]["gamma"] == 2.0

    class DatasetStub:
        labels = [0, 0, 1, 2, 3, 4, 5, 6]

    class LoaderStub:
        dataset = DatasetStub()

    monkeypatch.setattr("src.train.trainer.get_dataloaders", lambda **_: (LoaderStub(), [], []))
    for raw in configs:
        config = dict(raw, pretrained=False, mlflow_tracking_uri=(tmp_path / raw["loss"]["name"] / "mlruns").as_uri())
        trainer = Trainer(config)
        images = torch.randn(2, 3, 128, 128, device=trainer.device)
        targets = torch.tensor([0, 1], dtype=torch.long, device=trainer.device)
        loss = trainer.criterion(trainer.model(images), targets)
        eval_loss = trainer.eval_criterion(trainer.model(images), targets)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(eval_loss)
        assert any(parameter.grad is not None for parameter in trainer.model.parameters() if parameter.requires_grad)


def test_evaluation_classification_report_contains_all_fer_classes():
    """Evaluation sinh precision/recall/F1 cho đủ 7 lớp cảm xúc, kể cả lớp vắng dự đoán."""
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.num_classes = 7
    trainer.eval_criterion = nn.CrossEntropyLoss()
    trainer.model = nn.Identity()
    logits = torch.eye(7)
    targets = torch.arange(7)
    _, _, macro_f1 = trainer.evaluate([(logits, targets)])
    assert macro_f1 == pytest.approx(1.0)
    assert set(trainer.last_classification_report) >= {
        "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "macro avg"
    }
    assert trainer.last_classification_report["Disgust"]["recall"] == 1.0


def test_loss_run_names_are_distinct_without_changing_legacy_names():
    """Loss suffix phân biệt EXP-11, còn YAML cũ giữ nguyên định dạng tên run."""
    base = dict(exp_id="EXP-11", architecture="EfficientNetB0", optimizer="adamw", pretrained=True,
                image_size=128, augmentation_mode="basic")
    names = {Trainer._format_run_name(**base, loss_name=loss) for loss in ("cross_entropy", "weighted_ce", "focal")}
    assert len(names) == 3
    assert Trainer._format_run_name(**base).endswith("aug-basic")


def test_t0402_notebooks_are_valid_json_and_code_cells_parse():
    """Hai notebook EXP-11 có JSON hợp lệ và mọi code cell phân tích cú pháp được."""
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for platform in ("colab", "kaggle"):
        path = root / platform / f"run_t0402_{platform}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for index, source in enumerate(sources):
            ast.parse(source, filename=f"t0402-{platform}-cell-{index}")
        joined = "\n".join(sources)
        for config_name in ("exp11_loss_ce.yaml", "exp11_loss_weighted_ce.yaml", "exp11_loss_focal.yaml"):
            assert config_name in joined
        assert "t0402_ablation_imbalance_24col.csv" in joined
        assert "metrics.macro_f1" in joined
        assert "metrics.private_test_acc" in joined
        assert "MLflowLogger.CSV_COLUMNS" in joined
        if platform == "colab":
            assert "/content/mlruns_t0402" in joined
            assert "/content/drive/MyDrive/BTL02/T0402_Imbalance" in joined
            assert "torchvision" in joined and "stable_age_seconds" in joined
        else:
            assert "/kaggle/working/BTL02/T0402_Imbalance" in joined
            assert "x-access-token:" in joined and "KAGGLE_API_TOKEN" in joined
            assert "/kaggle/input/datasets" in joined and "restore_finished_runs" in joined