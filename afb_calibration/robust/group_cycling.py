"""A Group-DRO baseline that is not sabotaged by the batch-size / group-count mismatch.

R2's diagnosis of the submitted DRO row:

    "this instability stems largely from a mismatch between a batch size of two and three
     target groups - meaning most training batches contained only one or two groups.
     Consequently, the finding reflects a failure of this specific small-batch setup
     rather than a meaningful insight into Group-DRO's applicability to object detection."

Two changes, neither of which raises peak memory above the original batch of 2:

1. ``GroupCycleSampler`` - consecutive micro-batches rotate through group ids, so any
   window of >= num_groups batches contains every group.
2. Gradient accumulation over ``accum_steps`` micro-batches. The optimiser sees an
   effective batch of ``accum_steps * batch_size``, which spans all three groups, while
   each micro-batch's graph is freed immediately after its backward pass.

The DRO reducer's EMA is updated on every micro-batch, so by the time an optimiser step
fires its group weights already reflect all groups - which is precisely what the
batch-of-2 setup made impossible.

If DRO still collapses under this, that is a real negative result about Group-DRO on
detection. If it does not, the paper's "unstable" framing has to be withdrawn.
"""
from __future__ import annotations

import copy
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


class GroupCycleSampler(Sampler):
    """Round-robin over group ids; reshuffles a group's pool when exhausted."""

    def __init__(self, group_ids, seed: int = 0):
        self.by = defaultdict(list)
        for i, g in enumerate(np.asarray(group_ids)):
            self.by[int(g)].append(i)
        self.groups = sorted(self.by)
        self.n = len(group_ids)
        self.seed = seed

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        pools = {g: list(rng.permutation(self.by[g])) for g in self.groups}
        order, k = [], 0
        while len(order) < self.n:
            g = self.groups[k % len(self.groups)]
            k += 1
            if not pools[g]:
                pools[g] = list(rng.permutation(self.by[g]))
            order.append(int(pools[g].pop()))
        return iter(order[:self.n])


def group_coverage(dataset, batch_size, accum_steps, num_groups, seed=0, windows=200):
    """Sanity check: fraction of optimiser windows that actually see every group.

    Run this before committing GPU hours - it is the whole point of the fix, and it
    costs nothing. The submitted setup (shuffle, batch 2, accum 1) scores near zero.
    """
    from afb_calibration.data.groups import compute_group_ids
    gids = np.asarray(compute_group_ids(dataset))
    order = list(GroupCycleSampler(gids, seed=seed))
    per_window = batch_size * accum_steps
    full = 0
    n = min(windows, len(order) // per_window)
    for w in range(n):
        chunk = order[w * per_window:(w + 1) * per_window]
        full += len(set(gids[chunk].tolist())) == num_groups
    return full / max(1, n)


def train_group_cycling(cfg, dataset, device, group_index, accum_steps: int = 4,
                    log_every: int = 100):
    from afb_calibration.data.augment import CrossStyleAugment
    from afb_calibration.data.dataset import collate_detection
    from afb_calibration.data.groups import compute_group_ids
    from afb_calibration.detector import AFBDetector
    from afb_calibration.detector.detector import pad_batch
    from afb_calibration.engine.train_detector import _consistency
    from afb_calibration.robust import build_reducer

    det = AFBDetector(cfg.detector).to(device)
    reducer = build_reducer(cfg.robust, group_index.num_groups)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                        sampler=GroupCycleSampler(compute_group_ids(dataset), seed=cfg.seed),
                        num_workers=cfg.train.num_workers, collate_fn=collate_detection)
    opt = torch.optim.AdamW(det.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.robust.group_dro.l2)
    cc = cfg.robust.consistency
    aug = CrossStyleAugment(cc.max_hue, cc.max_sat, cc.max_blur) if cc.enabled else None

    det.train()
    step = 0
    opt.zero_grad()
    for epoch in range(cfg.train.epochs):
        for batch in loader:
            images = [im.to(device) for im in batch["images"]]
            targets = [{"boxes": b, "labels": l}
                       for b, l in zip(batch["boxes"], batch["labels"])]
            batched, _ = pad_batch(images)
            loss_dict = det.loss(det(batched), targets)
            loss = reducer.reduce(loss_dict["per_image"], batch["groups"].to(device))
            if aug is not None:
                loss = loss + cc.weight * _consistency(det, images, aug, seed=step)
            (loss / accum_steps).backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(det.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
            if step % log_every == 0:
                print(f"  epoch {epoch} step {step} loss {float(loss.detach()):.4f}",
                      flush=True)
            step += 1
    # flush a partial accumulation window
    if step % accum_steps:
        torch.nn.utils.clip_grad_norm_(det.parameters(), 5.0)
        opt.step()
        opt.zero_grad()
    return det


def build_or_load_group_cycling(cfg, dataset, device, group_index, cache_dir,
                            accum_steps: int = 4):
    """Cached wrapper so a killed Colab session does not lose a finished detector."""
    from afb_calibration.detector import AFBDetector

    os.makedirs(cache_dir, exist_ok=True)
    ck = os.path.join(cache_dir, f"group_cycling_accum{accum_steps}_seed{cfg.seed}.pt")
    if os.path.exists(ck):
        dc = copy.deepcopy(cfg.detector)
        dc.pretrained = False
        det = AFBDetector(dc).to(device)
        det.load_state_dict(torch.load(ck, map_location=device))
        print(f"loaded cached {os.path.basename(ck)}")
        return det
    det = train_group_cycling(cfg, dataset, device, group_index, accum_steps=accum_steps)
    torch.save(det.state_dict(), ck)
    print(f"cached {ck}")
    return det
