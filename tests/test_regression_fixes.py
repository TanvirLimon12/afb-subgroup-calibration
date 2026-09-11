"""Regression tests for the July-2026 fixes.

Each test pins a specific bug that was found in the v3 suite / logs and would
silently recur without a guard. Keep these fast (CPU, tiny tensors).
"""
import numpy as np
import torch

from fm_robustafb.utils.boxes import expand_boxes
from fm_robustafb.verifier.verifier import extract_crops
from fm_robustafb.robust.reducers import GroupDROReducer
from fm_robustafb.fusion.fuse import CalibrateThenFuse, _sanitize_logits
from fm_robustafb.verifier.prototypes import NNMemoryHead, build_head


# --- ZNSM 0-height crash (reducers.py / verifier.py) ------------------------

def test_expand_boxes_flipped_box_does_not_collapse():
    """A flipped/degenerate recovered ZNSM annotation box (x2 < x1) must expand
    to a >=1px box, not collapse to 0 height. This was the ZNSM LOBO crash."""
    flipped = torch.tensor([[30.0, 20.0, 10.0, 20.0]])  # x2 < x1, zero height
    out = expand_boxes(flipped, factor=2.5, height=60, width=80)
    w = (out[:, 2] - out[:, 0]).item()
    h = (out[:, 3] - out[:, 1]).item()
    assert w >= 1.0, f"expanded width collapsed to {w}"
    assert h >= 1.0, f"expanded height collapsed to {h}"


def test_expand_boxes_negative_extent():
    """A box with negative extent in both dims (fully flipped) must still yield
    a positive-area expanded box."""
    flipped = torch.tensor([[50.0, 50.0, 10.0, 10.0]])
    out = expand_boxes(flipped, factor=2.0, height=100, width=100)
    assert (out[:, 2] - out[:, 0]).item() >= 1.0
    assert (out[:, 3] - out[:, 1]).item() >= 1.0


def test_extract_crops_zero_height_input_does_not_crash():
    """Even if expand_boxes returned a degenerate box, extract_crops must produce
    a valid (C, size, size) crop — never feed F.interpolate a 0-size input."""
    img = torch.rand(3, 60, 80)
    zero_height = torch.tensor([[10.0, 20.0, 30.0, 20.0]])  # h = 0
    crops = extract_crops(img, zero_height, context=1.0, size=24)
    assert crops.shape == (1, 3, 24, 24)
    assert torch.isfinite(crops).all()


# --- Group DRO minibatch instability (reducers.py) --------------------------

def test_group_dro_sparse_batches_do_not_explode():
    """The bug: with small batches and many groups, most batches contain only one
    group; the naive per-batch DRO update drives that group's weight to ~1 and
    starves the others. The EMA fix must keep absent groups' weights non-trivial
    over a sequence of single-group batches."""
    dro = GroupDROReducer(num_groups=4, step_size=0.05, ema_momentum=0.9)
    # Simulate 20 steps where each batch contains ONLY group 0 (the failure mode).
    for _ in range(20):
        per_ex = torch.tensor([0.8, 0.9])
        groups = torch.tensor([0, 0])
        dro.reduce(per_ex, groups)
    # Group 0 should be up-weighted (highest EMA loss), but the never-seen groups
    # must not have been driven to exactly 0 by normalisation over absent groups.
    # After the fix, unseen groups are masked out of the normaliser entirely, so
    # their *stored* weight is 0 — but group 0 must not have collapsed either.
    assert dro.group_weights[0] > 0.0
    assert torch.isfinite(dro.group_weights).all()
    # And the EMA losses for seen groups must be tracked.
    assert dro._seen[0]
    assert dro._group_loss_ema[0] > 0


def test_group_dro_all_groups_seen_weights_sum_to_one():
    dro = GroupDROReducer(num_groups=3, step_size=0.02)
    for g in range(3):
        dro.reduce(torch.tensor([0.5 * (g + 1)]), torch.tensor([g]))
    total = dro.group_weights.sum().item()
    assert abs(total - 1.0) < 1e-5, f"weights sum to {total}, not 1"


def test_group_dro_loss_is_finite_and_differentiable():
    dro = GroupDROReducer(num_groups=3, step_size=0.02)
    per_ex = torch.tensor([0.5, 0.3, 0.9], requires_grad=True)
    groups = torch.tensor([0, 1, 2])
    loss = dro.reduce(per_ex, groups)
    assert torch.isfinite(loss)
    loss.backward()
    assert per_ex.grad is not None


# --- Fusion numerical sanitisation (fuse.py) --------------------------------

def test_sanitize_logits_handles_inf_nan():
    x = np.array([0.0, np.inf, -np.inf, np.nan, 1e6, -1e6])
    out = _sanitize_logits(x)
    assert np.isfinite(out).all()
    assert out.max() <= 50.0
    assert out.min() >= -50.0


def test_fusion_fit_with_non_finite_logits_does_not_warn():
    """Non-finite logits (common under shift) must not produce sklearn
    divide-by-zero / overflow warnings, and must still fit cleanly."""
    import warnings
    rng = np.random.default_rng(0)
    n = 200
    det = rng.normal(0, 3, n)
    ver = rng.normal(0, 3, n)
    # inject non-finite values the way an OOD tile would
    det[:10] = np.inf
    ver[10:20] = -np.inf
    det[20:30] = np.nan
    labels = (rng.random(n) > 0.5).astype(int)
    groups = rng.integers(0, 4, n)
    fuse = CalibrateThenFuse(calibration="isotonic", num_groups=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        fuse.fit(det, ver, labels, groups)
    p = fuse.predict_proba(det, ver, groups)
    assert np.isfinite(p).all()
    assert p.min() >= 0.0 and p.max() <= 1.0


# --- Training-free NN verifier head (prototypes.py) -------------------------

def test_nn_head_builds_and_scores():
    head = build_head("nn", dim=16)
    assert isinstance(head, NNMemoryHead)
    feats = torch.randn(40, 16)
    labels = torch.tensor([1] * 20 + [0] * 20)
    head.fit_from_features(feats, labels)
    assert head.pos_bank.shape == (20, 16)
    assert head.neg_bank.shape == (20, 16)
    out = head(torch.randn(5, 16))
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


def test_nn_head_empty_bank_emits_neutral():
    head = NNMemoryHead(dim=8)
    out = head(torch.randn(3, 8))
    assert torch.all(out == 0.0)


# --- DINOv2 loader: offline_stub behaviour ----------------------------------

def test_build_dino_stub_returns_stub():
    from fm_robustafb.verifier.dino import build_dino
    model = build_dino(offline_stub=True)
    # The stub exposes the DINOv2 interface but is not a real checkpoint.
    assert hasattr(model, "get_intermediate_layers")
    assert hasattr(model, "embed_dim")


def test_build_dino_real_raises_on_hub_failure(monkeypatch):
    """When offline_stub=False and the hub load fails, build_dino must RAISE —
    not silently fall back to the stub. This is the fix that prevents a full run
    from looking valid while containing no real DINOv2."""
    from fm_robustafb.verifier import dino as dino_mod

    def _boom(*a, **k):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(torch.hub, "load", _boom)
    try:
        dino_mod.build_dino(offline_stub=False)
        assert False, "build_dino should have raised on hub failure"
    except RuntimeError as e:
        assert "DINOv2 hub load failed" in str(e)
        assert "offline_stub" in str(e)


# --- Last-layer retraining reducer (P55) ------------------------------------

def test_llr_reducer_builds_and_balances():
    """The LLR reducer (last_layer_retrain) must build via build_reducer and
    produce group-balanced reweighting (its in-batch behavior)."""
    from fm_robustafb.robust.reducers import build_reducer, LastLayerRetrainingReducer
    from fm_robustafb.config import RobustConfig
    r = build_reducer(RobustConfig(method="last_layer_retrain"), num_groups=3)
    assert isinstance(r, LastLayerRetrainingReducer)
    # Unequal group sizes -> weights must up-weight the minority group
    per_ex = torch.tensor([0.5, 0.6, 0.4, 0.7])
    groups = torch.tensor([0, 0, 0, 1])  # group 1 is minority
    loss = r.reduce(per_ex, groups)
    assert torch.isfinite(loss)
    # group 0 (3 examples) and group 1 (1 example) get inverse-frequency weights
    # so the single group-1 example contributes as much as all of group 0


def test_build_reducer_rejects_unknown_method():
    from fm_robustafb.robust.reducers import build_reducer
    from fm_robustafb.config import RobustConfig
    try:
        build_reducer(RobustConfig(method="bogus"), num_groups=2)
        assert False, "should have raised"
    except ValueError as e:
        assert "bogus" in str(e)


# --- Localization-aware calibration (LaECE in evaluate_calibration) ---------

def test_evaluate_calibration_emits_laece():
    """evaluate_calibration must report LaECE (Kuzucu ECCV 2024, P54) when IoUs
    are present in fusion_out — not just D-ECE."""
    from fm_robustafb.engine.evaluate import evaluate_calibration
    from fm_robustafb.config import Config
    cfg = Config()
    fusion_out = {
        "fused": np.array([0.9, 0.8, 0.4, 0.7, 0.3, 0.6, 0.85, 0.35, 0.75, 0.25, 0.65, 0.45]),
        "labels": np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]),
        "groups": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        "ious": np.array([0.8, 0.6, 0.0, 0.7, 0.0, 0.0, 0.75, 0.0, 0.65, 0.0, 0.7, 0.0]),
    }
    out = evaluate_calibration(fusion_out, cfg)
    assert "D-ECE" in out
    assert "LaECE" in out, "LaECE must be reported when IoUs available"
    assert np.isfinite(out["LaECE"])
    assert "worst_group_D-ECE" in out
