"""
Bộ kiểm thử tự động xác minh các bản sửa lỗi WP03 (PatchTST, Informer, training_loop,
fine_tune, run_ablation, configs/default.yaml) và tương thích 100% với WP04 (XAI) & WP05 (ONNX opset=17).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

COURSE_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(COURSE_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_PROJECT_DIR))

from fine_tune import (
    build_finetune_param_groups,
    configure_trainable_layers,
    extract_state_dict,
)
from src.models.informer import FullAttention, InformerModel, ProbSparseAttention
from src.models.patchtst import PatchTST, PatchTSTModel
from training_loop import (
    PeakShapeLoss,
    RevINModelWrapper,
    build_criterion,
    build_model,
    build_optimizer_param_groups,
    compute_selection_score,
    load_model_configs,
)


@pytest.mark.parametrize("model_name", ["lstm", "dlinear", "informer", "informer_full", "patchtst"])
@pytest.mark.parametrize("in_features", [31, 1])
@pytest.mark.parametrize("horizon", [24, 48, 96, 192])
def test_all_models_shape_and_gradients(
    model_name: str, in_features: int, horizon: int
) -> None:
    """Mọi mô hình phải tuân thủ hợp đồng [B, 96, D] -> [B, H, 1] và lan truyền gradient sạch."""
    torch.manual_seed(42)
    cfg = load_model_configs(COURSE_PROJECT_DIR / "configs" / "default.yaml")
    model = build_model(
        model_name=model_name,
        lookback=96,
        horizon=horizon,
        in_features=in_features,
        device=torch.device("cpu"),
        use_wrappers=True,
        use_residual_anchor=False,
        model_kwargs=cfg.get(model_name, {}),
    )
    x = torch.randn(2, 96, in_features)
    y = model(x)
    assert y.shape == (2, horizon, 1), f"{model_name} sai shape: {y.shape}"
    assert torch.isfinite(y).all(), f"{model_name} xuất NaN/Inf"

    loss = y.pow(2).mean()
    loss.backward()
    for p_name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{model_name} bị None grad tại {p_name}"
            assert torch.isfinite(param.grad).all(), f"{model_name} bị NaN/Inf grad tại {p_name}"


def test_patchtst_selective_revin_handles_constant_channels() -> None:
    """PatchTST RevIN (tùy chọn use_revin=True) chỉ chuẩn hóa 11 kênh liên tục, giữ nguyên chu kỳ/gió."""
    torch.manual_seed(42)
    model = PatchTST(lookback=96, horizon=24, in_features=31, use_revin=True)
    x = torch.randn(4, 96, 31)
    # Cố định toàn bộ kênh rain (9), month_sin/cos (13,14) và 16 cột wd_* (15..30)
    x[:, :, 9] = -0.0782
    x[:, :, 13] = 0.866
    x[:, :, 14] = -0.500
    x[:, :, 15:] = -0.326

    x_norm, mean, stdev = model.revin.normalize(x)
    assert mean.shape == (4, 1, 11)
    assert stdev.shape == (4, 1, 11)
    assert (stdev >= 0.1 - 1e-6).all(), "stdev không được nhỏ hơn min_std=0.1"
    # Các kênh >= 11 (chu kỳ thời gian + 16 hướng gió) phải được giữ nguyên bản 100%
    assert torch.allclose(x_norm[:, :, 11:], x[:, :, 11:], atol=1e-6)

    y = model(x)
    assert y.shape == (4, 24, 1)
    assert torch.isfinite(y).all()


def test_informer_probsparse_vs_full_and_deterministic_eval() -> None:
    """Informer dùng ProbSparse tất định ở eval(), còn informer_full dùng FullAttention O(L^2) thực sự."""
    torch.manual_seed(42)
    inf_sparse = InformerModel(
        lookback=96, horizon=24, in_features=31, attention_type="probsparse"
    )
    inf_full = InformerModel(
        lookback=96, horizon=24, in_features=31, attention_type="full"
    )
    assert isinstance(inf_sparse.encoder_layers[0].self_attn.inner_attention, ProbSparseAttention)
    assert isinstance(inf_full.encoder_layers[0].self_attn.inner_attention, FullAttention)

    inf_sparse.eval()
    x = torch.randn(3, 96, 31)
    torch.manual_seed(101)
    y1 = inf_sparse(x)
    torch.manual_seed(202)
    y2 = inf_sparse(x)
    assert torch.allclose(y1, y2, atol=0.0), "Informer ở eval() phải tất định 100%!"


def test_yaml_and_default_constructor_strict_compatibility_for_wp04_wp05() -> None:
    """
    Đảm bảo trọng số huấn luyện từ configs/default.yaml nạp strict=True thành công khi WP04
    (ig_explainer.py gọi PatchTSTModel mặc định) hoặc WP05 (PyTorchCheckpointLoader gọi
    build_model không truyền model_kwargs) khởi tạo mô hình.
    """
    cfg = load_model_configs(COURSE_PROJECT_DIR / "configs" / "default.yaml")
    for m_name in ("patchtst", "informer", "dlinear", "lstm"):
        for in_feat in (31, 1):
            trained_model = build_model(
                model_name=m_name,
                lookback=96,
                horizon=24,
                in_features=in_feat,
                device=torch.device("cpu"),
                use_wrappers=True,
                use_residual_anchor=False,
                model_kwargs=cfg.get(m_name, {}),
            )
            # Mô phỏng WP05 PyTorchCheckpointLoader.load() gọi build_model không có model_kwargs
            serving_model = build_model(
                model_name=m_name,
                lookback=96,
                horizon=24,
                in_features=in_feat,
                device=torch.device("cpu"),
                use_wrappers=True,
            )
            serving_model.load_state_dict(trained_model.state_dict(), strict=True)

            # Mô phỏng WP04 ig_explainer.py gọi PatchTSTModel(lookback, horizon, in_features)
            if m_name == "patchtst":
                wp04_model = PatchTSTModel(lookback=96, horizon=24, in_features=in_feat)
                wp04_model.load_state_dict(trained_model.state_dict(), strict=True)


@pytest.mark.parametrize("model_name", ["patchtst", "informer", "dlinear", "lstm"])
@pytest.mark.parametrize("in_features", [31, 1])
def test_onnx_opset17_export_and_parity(model_name: str, in_features: int) -> None:
    """Kiểm định xuất ONNX opset=17 và độ lệch số học PyTorch ↔ ONNXRuntime < 1e-4."""
    if hasattr(torch.backends, "mha") and hasattr(
        torch.backends.mha, "set_fastpath_enabled"
    ):
        torch.backends.mha.set_fastpath_enabled(False)

    torch.manual_seed(42)
    cfg = load_model_configs(COURSE_PROJECT_DIR / "configs" / "default.yaml")
    model = build_model(
        model_name=model_name,
        lookback=96,
        horizon=24,
        in_features=in_features,
        device=torch.device("cpu"),
        use_wrappers=True,
        use_residual_anchor=False,
        model_kwargs=cfg.get(model_name, {}),
    )
    model.eval()
    x_dummy = torch.randn(1, 96, in_features, dtype=torch.float32)

    with torch.no_grad():
        pt_out = model(x_dummy).cpu().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / f"{model_name}_D{in_features}.onnx"
        torch.onnx.export(
            model,
            x_dummy,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": x_dummy.numpy()})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    assert max_diff < 1e-4, f"ONNX parity thất bại cho {model_name} (D={in_features}): diff={max_diff}"


def test_finetune_layer_unfreezing_and_param_groups() -> None:
    """Fine-tune phải mở khóa đúng head/adapter thực sự (không còn lỗi 25 tham số của LSTM/DLinear)."""
    cfg = load_model_configs(COURSE_PROJECT_DIR / "configs" / "default.yaml")
    for m_name in ("lstm", "dlinear", "informer", "patchtst"):
        model = build_model(
            model_name=m_name,
            lookback=96,
            horizon=24,
            in_features=31,
            device=torch.device("cpu"),
            use_wrappers=True,
            use_residual_anchor=False,
            model_kwargs=cfg.get(m_name, {}),
        )
        unfrozen = configure_trainable_layers(
            model, model_name=m_name, freeze_backbone=True, use_wrappers=True
        )
        assert unfrozen > 1000, f"{m_name} chỉ mở khóa {unfrozen} tham số (quá ít)!"

        # Kiểm tra decoupled weight decay và discriminative LR khi mở khóa toàn bộ
        configure_trainable_layers(
            model, model_name=m_name, freeze_backbone=False, use_wrappers=True
        )
        groups = build_finetune_param_groups(
            model,
            model_name=m_name,
            lr=1e-4,
            weight_decay=1e-2,
            freeze_backbone=False,
            backbone_lr_ratio=0.1,
        )
        assert any(g["weight_decay"] == 0.0 for g in groups)
        assert any(abs(g["lr"] - 1e-4) < 1e-12 for g in groups)
        if m_name != "dlinear":
            assert any(abs(g["lr"] - 1e-5) < 1e-12 for g in groups)


def test_peak_shape_loss_and_balanced_selection() -> None:
    """PeakShapeLoss phải phạt mạnh sai số vùng đỉnh & đường phẳng; balanced_dev cân bằng 50/50."""
    loss_fn = build_criterion(
        "PeakShapeLoss", peak_threshold_z=0.84, peak_alpha=0.5, diff_weight=0.15
    )
    assert isinstance(loss_fn, PeakShapeLoss)

    # Hai dự báo có cùng sai số điểm ở nền thấp vs đỉnh cao -> đỉnh cao bị phạt nặng hơn
    y_low = torch.zeros(2, 24)
    y_high = torch.full((2, 24), 3.0)
    err = torch.full((2, 24), 0.5)
    assert loss_fn(y_high - err, y_high) > loss_fn(y_low - err, y_low)

    # Kiểm tra cân bằng mùa: cùng tỷ lệ MSE/var = 0.5 cho val_score = 0.5
    score = compute_selection_score(
        train_dev_mse=0.50, dev_mse=0.15, var_train_dev=1.00, var_dev=0.30, mode="balanced_dev"
    )
    assert abs(score - 0.5) < 1e-9

    # Kiểm tra decoupled optimizer param groups
    m = PatchTST(lookback=96, horizon=24, in_features=31)
    opt_groups = build_optimizer_param_groups(m, lr=5e-4, weight_decay=1e-2)
    assert len(opt_groups) == 2
    assert opt_groups[0]["weight_decay"] == 1e-2
    assert opt_groups[1]["weight_decay"] == 0.0


def test_legacy_checkpoint_detection_and_reload() -> None:
    """extract_state_dict phải nhận diện chính xác cả checkpoint v2 mới lẫn checkpoint v1 cũ có res_scale."""
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 1. Checkpoint mới (use_residual_anchor=False)
        m_new = build_model(
            "patchtst", 96, 24, 31, device, use_wrappers=True, use_residual_anchor=False
        )
        p_new = tmp / "new.pt"
        torch.save(
            {
                "use_wrappers": True,
                "use_residual_anchor": False,
                "model_state_dict": m_new.state_dict(),
            },
            p_new,
        )
        sd_new, _, uw_new, ura_new = extract_state_dict(p_new, device)
        assert uw_new is True and ura_new is False
        m_new.load_state_dict(sd_new, strict=True)

        # 2. Checkpoint legacy (có res_scale & station_bias & RevINModelWrapper)
        m_old = build_model(
            "informer",
            96,
            24,
            31,
            device,
            use_wrappers=True,
            use_residual_anchor=True,
            use_revin_wrapper=True,
        )
        p_old = tmp / "old.pt"
        torch.save({"use_wrappers": True, "model_state_dict": m_old.state_dict()}, p_old)
        sd_old, _, uw_old, ura_old = extract_state_dict(p_old, device)
        assert uw_old is True and ura_old is True
        m_old.load_state_dict(sd_old, strict=True)

        # 3. Checkpoint bọc RevINModelWrapper độc lập (không có res_scale)
        m_rev = build_model(
            "lstm",
            96,
            24,
            31,
            device,
            use_wrappers=True,
            use_residual_anchor=False,
            use_revin_wrapper=True,
        )
        p_rev = tmp / "rev.pt"
        torch.save(
            {
                "use_wrappers": True,
                "use_residual_anchor": False,
                "use_revin_wrapper": True,
                "model_state_dict": m_rev.state_dict(),
            },
            p_rev,
        )
        sd_rev, meta_rev, uw_rev, ura_rev = extract_state_dict(p_rev, device)
        assert uw_rev is True and ura_rev is False and meta_rev.get("use_revin_wrapper") is True
        m_rev.load_state_dict(sd_rev, strict=True)


def test_revin_wrapper_restores_mean_and_preserves_exogenous() -> None:
    """RevINModelWrapper chuẩn phải khôi phục +mean trên đầu ra và không chuẩn hóa cột >= 11."""
    torch.manual_seed(0)
    backbone = InformerModel(lookback=96, horizon=24, in_features=31)
    wrapped = RevINModelWrapper(backbone=backbone, in_features=31, affine=True, restore_mean=True)
    wrapped.eval()
    x1 = torch.randn(2, 96, 31)
    x2 = x1.clone()
    x2[:, :, 0] += 5.0  # Dịch chuyển phân phối PM2.5 thêm +5 sigma
    with torch.no_grad():
        y1 = wrapped(x1)
        y2 = wrapped(x2)
    # Với RevIN chuẩn có +mean, dịch đầu vào +5 sigma phải kéo đầu ra lên đúng +5 sigma!
    assert torch.allclose(y2 - y1, torch.full_like(y1, 5.0), atol=1e-4)



@pytest.mark.parametrize("model_variant", ["informer", "informer_full"])
def test_informer_attention_projections_frozen_during_finetune(model_variant: str) -> None:
    """
    Các lớp Attention projection của Informer phải ở trạng thái requires_grad=False
    khi freeze_backbone=True.

    Kiểm chứng bản sửa lỗi BPM25_T0305: thay keyword quá rộng "projection" bằng
    "projection.0" và "projection.3" để tránh vô tình mở khóa query_projection,
    key_projection, value_projection, out_projection bên trong các lớp Attention.
    """
    torch.manual_seed(42)
    cfg = load_model_configs(COURSE_PROJECT_DIR / "configs" / "default.yaml")
    model = build_model(
        model_name=model_variant,
        lookback=96,
        horizon=24,
        in_features=31,
        device=torch.device("cpu"),
        use_wrappers=False,
        use_residual_anchor=False,
        model_kwargs=cfg.get(model_variant.replace("_full", ""), {}),
    )

    configure_trainable_layers(model, model_variant, freeze_backbone=True)

    ATTENTION_PROJECTION_KEYWORDS = (
        "query_projection",
        "key_projection",
        "value_projection",
    )
    for name, param in model.named_parameters():
        if any(kw in name for kw in ATTENTION_PROJECTION_KEYWORDS):
            assert not param.requires_grad, (
                f"[BUG] Lớp Attention projection '{name}' bị mở khóa nhầm "
                f"khi freeze_backbone=True -- se gay catastrophic forgetting."
            )

    FINAL_HEAD_KEYWORDS = ("projection.0", "projection.3", "horizon_queries")
    head_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and any(kw in name for kw in FINAL_HEAD_KEYWORDS)
    ]
    assert len(head_trainable) > 0, (
        f"[BUG] Khong co lop dau ra nao cua {model_variant} duoc mo khoa -- "
        f"fine-tuning se khong cap nhat duoc tham so."
    )
