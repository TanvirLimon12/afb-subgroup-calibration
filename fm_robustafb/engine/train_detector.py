"""Stage-1b / Stage-3 detector training with group-robust reduction and a
bounded cross-style consistency regulariser."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import Config
from ..data.augment import CrossStyleAugment
from ..detector import AFBDetector
from ..detector.detector import pad_batch
from ..robust import build_reducer
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _consistency(det: AFBDetector, images, aug: CrossStyleAugment, seed: int) -> torch.Tensor:
    v1 = [aug.two_views(im, seed + i)[0] for i, im in enumerate(images)]
    v2 = [aug.two_views(im, seed + i)[1] for i, im in enumerate(images)]
    v1, _ = pad_batch(v1)
    v2, _ = pad_batch(v2)
    o1, o2 = det(v1), det(v2)
    cls1 = torch.sigmoid(det._flatten(o1.cls_logits))
    cls2 = torch.sigmoid(det._flatten(o2.cls_logits))
    reg1 = det._flatten(o1.bbox_reg)
    reg2 = det._flatten(o2.bbox_reg)
    kl = (cls1 * (cls1.clamp(1e-6) / cls2.clamp(1e-6)).log()).mean()
    l1 = F.l1_loss(reg1, reg2)
    return kl + l1


def train_detector(cfg: Config, dataset, device: torch.device, group_index, log_every: int = 20):
    from .build import build_loader

    det = AFBDetector(cfg.detector).to(device)
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
                            epoch, step, float(loss), float(loss_dict["loss_cls"]),
                            float(loss_dict["loss_box"]), float(loss_dict["loss_ctr"]))
            step += 1
    return det
