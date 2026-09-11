"""Verifier classifier heads.

A single global-mean prototype under-represents the diversity of both bacilli and
artifacts, so the verifier uses multiple positive/negative prototypes or a small
learned classifier over the fused DINO feature.

A training-free nearest-neighbour head (``nn``) is also provided — this is the
AnomalyDINO-style (Baitieva et al. 2024, arXiv:2405.14529) memory bank: score each
candidate by its distance to the nearest confirmed-bacilli feature vs. the nearest
artifact feature. It is the cleanest empirical answer to D02 ("do frozen FM features
transfer to raw ZN without adaptation?") and is the cheapest ablation row. Per
Khan & Krawczyk 2025 (arXiv:2510.13643) the raw NN score is poorly calibrated and
should be post-hoc scaled before going into fusion — that is handled downstream by
the per-signal calibrator in ``fusion/calibration.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats).squeeze(-1)


class PrototypeHead(nn.Module):
    def __init__(self, dim: int, prototypes_per_class: int = 8):
        super().__init__()
        k = prototypes_per_class
        self.pos = nn.Parameter(torch.randn(k, dim) * 0.01)
        self.neg = nn.Parameter(torch.randn(k, dim) * 0.01)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        f = F.normalize(feats, dim=-1)
        pos = F.normalize(self.pos, dim=-1)
        neg = F.normalize(self.neg, dim=-1)
        s_pos = (f @ pos.t()).max(dim=1).values
        s_neg = (f @ neg.t()).max(dim=1).values
        return self.scale * (s_pos - s_neg) + self.bias

    @torch.no_grad()
    def init_from_features(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        pos = feats[labels == 1]
        neg = feats[labels == 0]
        if pos.numel():
            self.pos.copy_(_kmeans(pos, self.pos.shape[0]))
        if neg.numel():
            self.neg.copy_(_kmeans(neg, self.neg.shape[0]))


class NNMemoryHead(nn.Module):
    """Training-free nearest-neighbour verifier head.

    Stores a memory bank of L2-normalised feature vectors for confirmed positives
    (bacilli) and negatives (artifacts), and scores each candidate by the signed
    margin between its similarity to the nearest positive and its similarity to
    the nearest negative. No learnable parameters are fit by gradient descent; the
    only state is the two banks, populated via :meth:`fit_from_features`.

    This is the AnomalyDINO-style design (frozen DINOv2 + nearest-neighbour
    scoring, no adaptation) — it directly tests whether frozen FM features carry
    enough signal to verify AFB candidates on raw ZN without any fine-tuning.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("pos_bank", torch.empty(0, dim))
        self.register_buffer("neg_bank", torch.empty(0, dim))

    @torch.no_grad()
    def fit_from_features(self, feats: torch.Tensor, labels: torch.Tensor,
                          max_per_class: int = 512) -> "NNMemoryHead":
        """Populate the memory banks from labelled fused features.

        ``feats``: (N, dim) fused DINO features for confirmed candidates.
        ``labels``: (N,) {0=artifact/negative, 1=bacilli/positive}.
        ``max_per_class`` caps each bank for inference cost (kNN scales linearly).
        """
        feats = F.normalize(feats.float(), dim=-1)
        labels = labels.to(torch.long)
        pos = feats[labels == 1]
        neg = feats[labels == 0]
        if pos.numel() and pos.shape[0] > max_per_class:
            pos = pos[torch.randperm(pos.shape[0])[:max_per_class]]
        if neg.numel() and neg.shape[0] > max_per_class:
            neg = neg[torch.randperm(neg.shape[0])[:max_per_class]]
        self.pos_bank = pos if pos.numel() else torch.empty(0, self.dim, device=feats.device)
        self.neg_bank = neg if neg.numel() else torch.empty(0, self.dim, device=feats.device)
        return self

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.pos_bank.numel() == 0:
            # No references yet — emit a neutral logit so the calibrator handles it.
            return feats.new_zeros(feats.shape[0])
        f = F.normalize(feats.float(), dim=-1)
        # Banks are populated by reassignment in fit_from_features and may sit on a
        # different device than the inference features (e.g. banks fit on CPU, f on
        # CUDA). Align them to f's device before the matmul.
        pos_bank = self.pos_bank.to(f.device)
        s_pos = (f @ pos_bank.t()).max(dim=1).values   # cosine sim to nearest bacillus
        if self.neg_bank.numel():
            s_neg = (f @ self.neg_bank.to(f.device).t()).max(dim=1).values
        else:
            s_neg = feats.new_zeros(f.shape[0])
        # Margin in [-2, 2]; multiply to spread logits for the calibrator.
        return 5.0 * (s_pos - s_neg)


def _kmeans(x: torch.Tensor, k: int, iters: int = 25) -> torch.Tensor:
    if x.shape[0] <= k:
        pad = x[torch.randint(x.shape[0], (k - x.shape[0],))] if x.shape[0] < k else x
        return torch.cat([x, pad], 0)[:k].clone()
    centers = x[torch.randperm(x.shape[0])[:k]].clone()
    for _ in range(iters):
        d = torch.cdist(x, centers)
        assign = d.argmin(dim=1)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(dim=0)
    return centers


def build_head(kind: str, dim: int, prototypes_per_class: int = 8) -> nn.Module:
    if kind == "linear":
        return LinearHead(dim)
    if kind == "prototype":
        return PrototypeHead(dim, prototypes_per_class)
    if kind == "nn":
        return NNMemoryHead(dim)
    raise ValueError(f"unknown head: {kind}")
