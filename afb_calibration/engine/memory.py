"""Activation-memory control for high-resolution detector training.

Training runs on full fields (up to 1600 px on the long side), and the bounded
cross-style consistency term adds two further forward passes per step. At batch size 2
the peak activation footprint is roughly 21 GiB, which does not fit a 16 GB accelerator.

``enable_gradient_checkpointing`` trades compute for memory on the backbone stages:
activations are recomputed during the backward pass instead of being stored. The
computation is mathematically identical, so trained weights are unaffected; the measured
cost is roughly a 40% slowdown for a peak of about 7.6 GiB.

Checkpointing is applied only while gradients are enabled, so inference is untouched.
"""
from __future__ import annotations

import torch
import torch.utils.checkpoint as cp

BACKBONE_STAGES = ("layer1", "layer2", "layer3", "layer4")


def enable_gradient_checkpointing(detector) -> int:
    """Wrap the backbone stages of ``detector`` in activation checkpoints.

    Returns the number of stages wrapped. Safe to call once per model; calling it twice
    would nest the wrappers, so callers should apply it immediately after construction.
    """
    body = getattr(getattr(detector, "backbone", None), "body", None)
    if body is None:
        return 0
    wrapped = 0
    for name in BACKBONE_STAGES:
        module = getattr(body, name, None)
        if module is None:
            continue
        original = module.forward  # bind the original forward, never the module itself,
        # otherwise checkpoint(module, x) re-enters this wrapper and recurses forever.
        module.forward = (
            lambda fn: (lambda x: cp.checkpoint(fn, x, use_reentrant=False)
                        if torch.is_grad_enabled() else fn(x))
        )(original)
        wrapped += 1
    return wrapped
