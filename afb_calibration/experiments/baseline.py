"""Experiment 8 - external baseline comparison (Faster R-CNN).

Provides an external state-of-the-art comparison alongside internal ablations. This runs a
standard Faster R-CNN ResNet-50 FPN (matching Rulaningtyas et al. 2026, P43 — the
closest AFB competitor) through the SAME tiled-inference + evaluate_detection path
as the AFB Subgroup Calibration rows, producing a directly comparable worst-group AP@0.5 row.

The baseline trains via torchvision's own loss (the adapter forwards
net(images, targets)); group-robust methods do not apply (it is a plain ERM
baseline by design — the point is "what does a standard detector achieve, with no
FM verifier and no group-robust training"). The baseline does NOT use the verifier
or fusion stages; it is detector-only.
"""
from __future__ import annotations

import time

import torch

from ..config import Config
from ..data.augment import CrossStyleAugment
from ..engine.build import build_group_index, build_loader
from ..engine.evaluate import evaluate_detection_results
from ..engine.infer import tiled_inference
from ..detector.baseline import build_baseline
from ..detector.detector import pad_batch
from ..utils.logging import get_logger
from .common import clone_config

logger = get_logger(__name__)


def _train_baseline(cfg: Config, dataset, device, epochs: int):
    """Train the torchvision baseline via its native loss. Plain ERM (no group robustness)."""
    det = build_baseline("faster_rcnn", pretrained=cfg.detector.pretrained).to(device)
    loader = build_loader(cfg, dataset, train=True, balanced=False)
    opt = torch.optim.AdamW([p for p in det.parameters() if p.requires_grad],
                            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    step = 0
    det.train()
    for epoch in range(epochs):
        for batch in loader:
            images = [im.to(device) for im in batch["images"]]
            targets = [{"boxes": b.to(device), "labels": l.to(device)}
                       for b, l in zip(batch["boxes"], batch["labels"])]
            batched = pad_batch(images)[0]
            stub = det(batched)            # caches images
            loss_dict = det.loss(stub, targets)
            loss = loss_dict["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(det.parameters(), 5.0)
            opt.step()
            if step % 20 == 0:
                logger.info("baseline epoch %d step %d loss %.4f", epoch, step, float(loss))
            step += 1
    return det


def run(cfg: Config, max_images=None, train_image_fraction: float = 1.0,
        epochs: int | None = None) -> dict:
    c = clone_config(cfg)
    device = torch.device(cfg.device if cfg.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    gi = build_group_index(c)
    from ..engine.build import build_dataset
    train_ds = build_dataset(c, "train", gi, image_fraction=train_image_fraction,
                             fraction_seed=c.seed)
    test_ds = build_dataset(c, "test", gi)
    n_epochs = epochs or c.train.epochs

    logger.info("training Faster R-CNN baseline (%d epochs)", n_epochs)
    det = _train_baseline(c, train_ds, device, n_epochs)

    t0 = time.perf_counter()
    results = tiled_inference(det, test_ds, device, c.data.tile_size, c.data.tile_overlap,
                              c.detector.cross_tile_nms_iou, max_images=max_images)
    inference_cost = (time.perf_counter() - t0) / max(1, len(results))
    report = evaluate_detection_results(results, c)
    report["inference_cost_s_per_image"] = inference_cost
    return {
        "variant": "faster_rcnn_baseline",
        "worst_group_AP50": report.get("worst_group_AP50"),
        "mean_AP50": report.get("mean_AP50"),
        "AP75": report.get("AP75"),
        "AP50:95": report.get("AP50:95"),
        "count_MAE": report.get("count_MAE"),
        "inference_cost_s": inference_cost,
        "note": "torchvision Faster R-CNN ResNet-50 FPN (ERM, no verifier/fusion)",
    }
