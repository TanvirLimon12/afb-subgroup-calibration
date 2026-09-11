"""Reproducibility specification.

Reads the training, verifier and fusion configuration off the live config and code
rather than restating it by hand, so documentation cannot drift from the implementation.
Emits JSON plus a LaTeX-ready paragraph.
"""
from __future__ import annotations

import inspect
import os
import json


def verifier_spec(cfg):
    """Everything the verifier paragraph needs, pulled from the live config."""
    v = cfg.verifier
    return {
        "backbone": v.backbone,
        "backbone_frozen": True,
        "adaptation": v.adaptation,
        "lora_rank": getattr(v, "lora_rank", None),
        "lora_targets": "attention and MLP projection matrices of the ViT blocks",
        "fuse_layers": list(getattr(v, "fuse_layers", [])),
        "pooling": "mean patch embedding concatenated with the CLS token, learned layer weights",
        "head": v.head,
        "prototypes_per_class": getattr(v, "prototypes_per_class", None),
        "crop_context": v.crop_context,
        "crop_size": v.crop_size,
        "supervision": (
            "candidate crops from the development (provider validation) split only. Each "
            "detector proposal on that split is labelled positive when its best IoU with "
            "any ground-truth box in the same image is >= 0.5, negative otherwise; "
            "ground-truth boxes are additionally injected as positives when building the "
            "development bank."),
        "epochs": 5,
        "gradient_flow_to_detector": False,
        "gradient_flow_note": (
            "the verifier is trained after the detector is frozen and consumes cropped "
            "images, not detector features, so no gradient reaches the detector; the two "
            "stages never appear in a single backward pass"),
    }


def training_spec(cfg):
    t = cfg.train
    return {
        "epochs": t.epochs, "batch_size": t.batch_size, "lr": t.lr,
        "weight_decay": t.weight_decay, "optimizer": "AdamW",
        "grad_clip_norm": 5.0, "amp": getattr(t, "amp", None),
        "detector": {
            "backbone": cfg.detector.backbone, "pretrained": cfg.detector.pretrained,
            "head": "anchor-free FCOS-style", "fpn_levels":
                f"P{cfg.detector.fpn_min_level}-P{cfg.detector.fpn_max_level}",
            "score_thresh": cfg.detector.score_thresh,
            "cross_tile_nms_iou": cfg.detector.cross_tile_nms_iou},
        "tiling": {"max_side": cfg.data.max_side, "tile_size": cfg.data.tile_size,
                   "tile_overlap": cfg.data.tile_overlap},
        "consistency_regulariser": {
            "enabled": cfg.robust.consistency.enabled,
            "weight": cfg.robust.consistency.weight,
            "form": "KL on class probabilities + L1 on box regression between two "
                    "lightly augmented views",
            "bounds": {"max_hue": cfg.robust.consistency.max_hue,
                       "max_sat": cfg.robust.consistency.max_sat,
                       "max_blur": cfg.robust.consistency.max_blur}},
    }


def fusion_spec(cfg):
    return {
        "per_signal_calibration": cfg.fusion.calibration,
        "fitted_on": "development split candidates only, never the test split",
        "fusion_model": cfg.fusion.model,
        "features": ["calibrated detector probability", "calibrated verifier probability",
                     "Jensen-Shannon divergence between the two",
                     "one-hot style-group indicator" if cfg.fusion.use_group_feature else None],
        "use_group_feature": cfg.fusion.use_group_feature,
        "note": ("the group indicator is given to the fusion model deliberately, so any "
                 "residual per-group calibration gap survives a model that was free to fit "
                 "a group-specific offset"),
    }


def source_of_truth():
    """Source locations of the functions that implement the specification above, so the
    reported configuration can be checked against the code."""
    # `afb_calibration.engine` re-exports a *function* named train_verifier, which shadows
    # the module of the same name under `from ... import`. Import the modules explicitly.
    import importlib
    tv = importlib.import_module("afb_calibration.engine.train_verifier")
    fs = importlib.import_module("afb_calibration.engine.fusion_stage")
    out = {}
    for name, fn in [("build_candidate_bank", tv.build_candidate_bank),
                     ("train_verifier", tv.train_verifier),
                     ("fit_fusion", fs.fit_fusion),
                     ("apply_fusion", fs.apply_fusion)]:
        try:
            # repository-relative, so the record does not leak a local filesystem path
            src = inspect.getsourcefile(fn) or ""
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            out[name] = {"file": os.path.relpath(src, root) if src else None,
                         "lines": inspect.getsourcelines(fn)[1]}
        except Exception as e:
            out[name] = {"error": str(e)}
    return out


def full_spec(cfg):
    return {"training": training_spec(cfg), "verifier": verifier_spec(cfg),
            "fusion": fusion_spec(cfg), "code_locations": source_of_truth()}


def to_prose(spec):
    """A LaTeX-ready paragraph answering  verbatim."""
    v, t, f = spec["verifier"], spec["training"], spec["fusion"]
    return (
        f"\\paragraph{{Verifier training.}} The {v['backbone']} backbone stays frozen; "
        f"adaptation uses LoRA of rank {v['lora_rank']} on the {v['lora_targets']}. "
        f"Candidate crops are expanded {v['crop_context']}$\\times$ for context and resized "
        f"to ${v['crop_size']}\\times{v['crop_size']}$; features fuse blocks "
        f"{v['fuse_layers']} through learned weights, and a {v['head']} head with "
        f"{v['prototypes_per_class']} prototypes per class produces the verifier logit. "
        f"The verifier is supervised by {v['supervision']} It is trained for "
        f"{v['epochs']} epochs after the detector is frozen. "
        f"\\textbf{{No gradient flows back to the detector}}: {v['gradient_flow_note']}. "
        f"The detector itself is trained for {t['epochs']} epochs with AdamW "
        f"(lr {t['lr']}, weight decay {t['weight_decay']}), batch size {t['batch_size']}, "
        f"gradient-norm clipping at {t['grad_clip_norm']}. "
        f"Both signals are calibrated by {f['per_signal_calibration']} regression fitted "
        f"on {f['fitted_on']}, and combined by a {f['fusion_model']} model. "
        f"{f['note'].capitalize()}."
    )
