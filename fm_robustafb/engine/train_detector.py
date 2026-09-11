"""Stage-1b / Stage-3 detector training with group-robust reduction and a
bounded cross-style consistency regulariser.

This version adds a *disk checkpoint cache* so the detector is trained only
once for a given training configuration. Every entry point that trains a
detector (scripts.train_detector, scripts.run_main, and every experiment) goes
through ``train_detector`` below, so the cache is shared across all of them.

Cache behaviour
---------------
The cache key is a hash of only the things that actually change the trained
detector: the detector architecture, the robust method + its params, the train
hyper-parameters, the seed, the relevant data settings, the number of groups,
and the size of the training set (which captures label-efficiency subsampling).
It deliberately does NOT depend on the *test* camera/background selection, so
experiments that only change what they evaluate on (cross_camera, lobo,
hard_negative, calibration) reuse a single trained detector instead of
retraining once per fold.

Experiments that genuinely change training (benchmark / ablation vary
robust.method; label_efficiency varies the train fraction) get a different key
and correctly retrain only the variants that differ.

Controls
--------
- Set env var ``AFB_DET_CACHE`` to change the cache directory
  (default: ``runs/_detector_cache``).
- Set env var ``AFB_DET_CACHE_DISABLE=1`` to force training every time.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import Config, to_dict
from ..data.augment import CrossStyleAugment
from ..detector import AFBDetector
from ..detector.detector import pad_batch
from ..robust import build_reducer
from .memory import enable_gradient_checkpointing
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _consistency(det: AFBDetector, images, aug: CrossStyleAugment, seed: int) -> torch.Tensor:
    # Build both augmented views, then pad every view to a common (H, W) so they
    # can be stacked even when CrossStyleAugment returns slightly different sizes.
    views1 = [aug.two_views(im, seed + i)[0] for i, im in enumerate(images)]
    views2 = [aug.two_views(im, seed + i)[1] for i, im in enumerate(images)]
    max_h = max(max(im.shape[1] for im in views1), max(im.shape[1] for im in views2))
    max_w = max(max(im.shape[2] for im in views1), max(im.shape[2] for im in views2))
    v1 = torch.stack([F.pad(im, (0, max_w - im.shape[2], 0, max_h - im.shape[1])) for im in views1])
    v2 = torch.stack([F.pad(im, (0, max_w - im.shape[2], 0, max_h - im.shape[1])) for im in views2])
    o1, o2 = det(v1), det(v2)
    cls1 = torch.sigmoid(det._flatten(o1.cls_logits))
    cls2 = torch.sigmoid(det._flatten(o2.cls_logits))
    reg1 = det._flatten(o1.bbox_reg)
    reg2 = det._flatten(o2.bbox_reg)
    kl = (cls1 * (cls1.clamp(1e-6) / cls2.clamp(1e-6)).log()).mean()
    l1 = F.l1_loss(reg1, reg2)
    return kl + l1


def _detector_cache_key(cfg: Config, dataset, num_groups: int) -> str:
    """A stable hash of everything that affects the trained detector weights."""
    payload = {
        "detector": to_dict(cfg.detector),
        "robust": to_dict(cfg.robust),
        "train": to_dict(cfg.train),
        "seed": cfg.seed,
        "data": {
            "root": cfg.data.root,
            "ann_train": cfg.data.ann_train,
            "max_side": cfg.data.max_side,
            "normalize_stain": cfg.data.normalize_stain,
            "num_classes": cfg.data.num_classes,
        },
        "num_groups": int(num_groups),
        "train_size": int(len(dataset)),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _load_cached_detector(cfg: Config, device: torch.device, ckpt: Path) -> AFBDetector:
    # Build the architecture without triggering the pretrained-backbone download
    # (weights are overwritten by the checkpoint anyway).
    det_cfg = copy.deepcopy(cfg.detector)
    det_cfg.pretrained = False
    det = AFBDetector(det_cfg).to(device)
    det.load_state_dict(torch.load(ckpt, map_location=device))
    return det


def train_detector(cfg: Config, dataset, device: torch.device, group_index, log_every: int = 20):
    from .build import build_loader

    use_cache = os.environ.get("AFB_DET_CACHE_DISABLE", "0") != "1"
    cache_dir = Path(os.environ.get("AFB_DET_CACHE", "runs/_detector_cache"))
    key = _detector_cache_key(cfg, dataset, group_index.num_groups)
    ckpt = cache_dir / f"det_{key}.pt"

    if use_cache and ckpt.exists():
        logger.info("reusing cached detector %s (skipping training; "
                    "set AFB_DET_CACHE_DISABLE=1 to force retrain)", ckpt)
        return _load_cached_detector(cfg, device, ckpt)

    det = AFBDetector(cfg.detector).to(device)
    if os.environ.get("AFB_GRAD_CHECKPOINT", "1") != "0":
        n = enable_gradient_checkpointing(det)
        if n:
            logger.info("gradient checkpointing enabled on %d backbone stages "
                        "(set AFB_GRAD_CHECKPOINT=0 to disable)", n)
    reducer = build_reducer(cfg.robust, group_index.num_groups)
    balanced = cfg.robust.method == "group_balanced"
    weight_decay = (cfg.robust.group_dro.l2 if cfg.robust.method == "group_dro"
                    else cfg.train.weight_decay)
    loader = build_loader(cfg, dataset, train=True, balanced=balanced)
    opt = torch.optim.AdamW(det.parameters(), lr=cfg.train.lr, weight_decay=weight_decay)

    cons_cfg = cfg.robust.consistency
    aug = CrossStyleAugment(cons_cfg.max_hue, cons_cfg.max_sat, cons_cfg.max_blur) if cons_cfg.enabled else None

    step = 0
    det.train()
    for epoch in range(cfg.train.epochs):
        for batch in loader:
            images = [im.to(device) for im in batch["images"]]
            targets = [{"boxes": b, "labels": l} for b, l in zip(batch["boxes"], batch["labels"])]
            batched, _ = pad_batch(images)
            outputs = det(batched)
            loss_dict = det.loss(outputs, targets)
            loss = reducer.reduce(loss_dict["per_image"], batch["groups"].to(device))
            if aug is not None:
                loss = loss + cons_cfg.weight * _consistency(det, images, aug, seed=step)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(det.parameters(), 5.0)
            opt.step()
            if step % log_every == 0:
                logger.info("detector epoch %d step %d loss %.4f (cls %.3f box %.3f ctr %.3f)",
                            epoch, step, float(loss.detach()), float(loss_dict["loss_cls"].detach()),
                            float(loss_dict["loss_box"].detach()), float(loss_dict["loss_ctr"].detach()))
            step += 1

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(det.state_dict(), ckpt)
        logger.info("cached trained detector to %s", ckpt)
    return det
