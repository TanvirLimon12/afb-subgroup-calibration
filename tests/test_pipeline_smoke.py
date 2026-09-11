import copy

from afb_calibration.data.audit import run_audit
from afb_calibration.engine import run_pipeline


def test_audit_runs(smoke_config):
    report = run_audit(smoke_config.data.ann_train, smoke_config.data.root)
    assert report["num_images"] > 0
    assert "group_distribution" in report
    assert "recommend_p2" in report["p2_recommendation"]


def test_full_pipeline_smoke(smoke_config):
    art = run_pipeline(smoke_config, use_verifier=True, use_fusion=True, verifier_epochs=1)
    assert "detection" in art.report
    det = art.report["detection"]
    assert "worst_group_AP50" in det
    assert det["inference_cost_s_per_image"] >= 0


def test_full_pipeline_group_dro(smoke_config):
    """The group_dro path must run end-to-end after the EMA/softmax reducer fix.
    This is the method that produced worst-group AP 0.087 in the v3 suite."""
    cfg = copy.deepcopy(smoke_config)
    cfg.robust.method = "group_dro"
    art = run_pipeline(cfg, use_verifier=False, use_fusion=False)
    assert "detection" in art.report
    assert art.report["detection"]["worst_group_AP50"] >= 0  # finite, not crashed


def test_full_pipeline_nn_verifier_head(smoke_config):
    """The training-free NN verifier head (AnomalyDINO-style) must run through
    the real train_verifier path — populate banks from crops and score val boxes."""
    cfg = copy.deepcopy(smoke_config)
    cfg.verifier.head = "nn"
    art = run_pipeline(cfg, use_verifier=True, use_fusion=True, verifier_epochs=1)
    assert "detection" in art.report
    # NN head should have populated its memory bank (needs >=1 positive candidate).
    assert art.verifier.head.pos_bank.shape[0] + art.verifier.head.neg_bank.shape[0] > 0
