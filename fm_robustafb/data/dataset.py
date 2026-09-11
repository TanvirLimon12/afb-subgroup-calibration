"""COCO-style AFB detection dataset with camera/background group metadata.

Annotation JSON schema::

    {
      "images": [{"id", "file_name", "width", "height", "camera", "background"}],
      "annotations": [{"id", "image_id", "bbox": [x, y, w, h], "category_id": 1}],
      "categories": [{"id": 1, "name": "afb"}]
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .groups import GroupIndex


class AFBDetectionDataset(Dataset):
    def __init__(
        self,
        ann_file: str | Path,
        image_root: str | Path,
        group_index: GroupIndex,
        cameras: Optional[Sequence[str]] = None,
        backgrounds: Optional[Sequence[str]] = None,
        image_fraction: float = 1.0,
        fraction_seed: int = 0,
        transform: Optional[Callable] = None,
        max_side: int = 0,
    ):
        self.image_root = Path(image_root)
        self.group_index = group_index
        self.transform = transform
        self.max_side = max_side
        with open(ann_file, "r") as f:
            coco = json.load(f)

        by_image: dict = {img["id"]: [] for img in coco["images"]}
        for ann in coco["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)

        images = coco["images"]
        if cameras is not None:
            images = [im for im in images if im["camera"] in set(cameras)]
        if backgrounds is not None:
            images = [im for im in images if im["background"] in set(backgrounds)]

        images = self._subsample(images, image_fraction, fraction_seed)
        self.images = images
        self.anns = by_image

    def _subsample(self, images: List[dict], fraction: float, seed: int) -> List[dict]:
        if fraction >= 1.0:
            return images
        rng = np.random.default_rng(seed)
        buckets: dict = {}
        for im in images:
            buckets.setdefault((im["camera"], im["background"]), []).append(im)
        kept: List[dict] = []
        for group_images in buckets.values():
            idx = rng.permutation(len(group_images))
            n = max(1, int(round(len(group_images) * fraction)))
            kept.extend(group_images[i] for i in idx[:n])
        kept.sort(key=lambda im: im["id"])
        return kept

    def __len__(self) -> int:
        return len(self.images)

    def group_id(self, index: int) -> int:
        im = self.images[index]
        return self.group_index.id(im["camera"], im["background"])

    def __getitem__(self, index: int) -> dict:
        im = self.images[index]
        path = self.image_root / im["file_name"]
        image = Image.open(path).convert("RGB")
        scale = 1.0
        if self.max_side and max(image.size) > self.max_side:
            scale = self.max_side / max(image.size)
            image = image.resize((max(1, round(image.size[0] * scale)),
                                  max(1, round(image.size[1] * scale))))
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

        boxes, labels = [], []
        for ann in self.anns.get(im["id"], []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x * scale, y * scale, (x + w) * scale, (y + h) * scale])
            labels.append(ann["category_id"] - 1)
        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.long).reshape(-1)

        sample = {
            "image": tensor,
            "boxes": boxes_t,
            "labels": labels_t,
            "group": self.group_id(index),
            "image_id": im["id"],
            "meta": {"camera": im["camera"], "background": im["background"], "file_name": im["file_name"]},
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def collate_detection(batch: List[dict]) -> dict:
    return {
        "images": [b["image"] for b in batch],
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "groups": torch.tensor([b["group"] for b in batch], dtype=torch.long),
        "image_ids": [b["image_id"] for b in batch],
        "metas": [b["meta"] for b in batch],
    }
