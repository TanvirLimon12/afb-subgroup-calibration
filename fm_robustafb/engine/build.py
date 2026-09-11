"""Dataset / dataloader construction."""
from __future__ import annotations

from typing import Optional, Sequence

from torch.utils.data import DataLoader

from ..config import Config
from ..data import (AFBDetectionDataset, GroupBalancedSampler, GroupIndex, collate_detection,
                    compute_group_ids)


def build_group_index(cfg: Config) -> GroupIndex:
    return GroupIndex(cfg.data.cameras, cfg.data.backgrounds)


def build_dataset(
    cfg: Config, split: str, group_index: Optional[GroupIndex] = None,
    cameras: Optional[Sequence[str]] = None, backgrounds: Optional[Sequence[str]] = None,
    image_fraction: float = 1.0, fraction_seed: int = 0,
) -> AFBDetectionDataset:
    group_index = group_index or build_group_index(cfg)
    ann = {"train": cfg.data.ann_train, "val": cfg.data.ann_val, "test": cfg.data.ann_test}[split]
    return AFBDetectionDataset(
        ann_file=ann, image_root=cfg.data.root, group_index=group_index,
        cameras=cameras, backgrounds=backgrounds,
        image_fraction=image_fraction, fraction_seed=fraction_seed,
        max_side=cfg.data.max_side,
    )


def build_loader(cfg: Config, dataset: AFBDetectionDataset, train: bool, balanced: bool = False):
    if train and balanced:
        sampler = GroupBalancedSampler(compute_group_ids(dataset), seed=cfg.seed)
        return DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler,
                          num_workers=cfg.train.num_workers, collate_fn=collate_detection)
    return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=train,
                      num_workers=cfg.train.num_workers, collate_fn=collate_detection)
