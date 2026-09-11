"""Camera x background groups and group-aware sampling.

A group is (camera, background_profile). Group IDs index every worst-group metric
and every group-robust training method (Stage 3).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class GroupIndex:
    def __init__(self, cameras: Sequence[str], backgrounds: Sequence[str]):
        self.cameras = list(cameras)
        self.backgrounds = list(backgrounds)
        self._cam = {c: i for i, c in enumerate(self.cameras)}
        self._bg = {b: i for i, b in enumerate(self.backgrounds)}

    @property
    def num_groups(self) -> int:
        return len(self.cameras) * len(self.backgrounds)

    def id(self, camera: str, background: str) -> int:
        return self._cam[camera] * len(self.backgrounds) + self._bg[background]

    def name(self, group_id: int) -> str:
        cam = group_id // len(self.backgrounds)
        bg = group_id % len(self.backgrounds)
        return f"{self.cameras[cam]}/{self.backgrounds[bg]}"

    def camera_of(self, group_id: int) -> str:
        return self.cameras[group_id // len(self.backgrounds)]

    def background_of(self, group_id: int) -> str:
        return self.backgrounds[group_id % len(self.backgrounds)]

    def present_groups(self, group_ids: Sequence[int]) -> List[int]:
        return sorted(set(int(g) for g in group_ids))


def compute_group_ids(dataset) -> torch.Tensor:
    return torch.tensor([dataset.group_id(i) for i in range(len(dataset))], dtype=torch.long)


class GroupBalancedSampler(Sampler[int]):
    """Sample each group with equal probability (Stage 3 balanced baseline).

    Simple balancing frequently matches or beats Group DRO under noisy/unknown
    groups (P16, P39); it is the required comparison, not an afterthought.
    """

    def __init__(self, group_ids: torch.Tensor, num_samples: int | None = None, seed: int = 0):
        self.group_ids = group_ids.long()
        self.num_samples = num_samples or len(group_ids)
        self.generator = torch.Generator().manual_seed(seed)
        groups = torch.unique(self.group_ids)
        self.group_to_indices = {
            int(g): torch.nonzero(self.group_ids == g, as_tuple=False).flatten() for g in groups
        }
        self.groups = list(self.group_to_indices.keys())

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        picks = []
        for _ in range(self.num_samples):
            g = self.groups[torch.randint(len(self.groups), (1,), generator=self.generator).item()]
            pool = self.group_to_indices[g]
            j = torch.randint(len(pool), (1,), generator=self.generator).item()
            picks.append(int(pool[j]))
        return iter(picks)


def group_counts(group_ids: torch.Tensor, num_groups: int) -> np.ndarray:
    counts = np.zeros(num_groups, dtype=np.int64)
    for g in group_ids.tolist():
        counts[g] += 1
    return counts
