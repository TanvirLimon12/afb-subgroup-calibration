"""Unsupervised staining-style group construction.

The Raw Sputum corpus carries no camera or staining-batch identifiers, so the
camera x background groups the pipeline needs cannot be read from metadata. In
their absence we partition images by low-level acquisition statistics -
colour-cast, brightness, and focus - into K staining-style groups with k-means.
This is a data-audit heuristic, not a model: it lets the worst-group AP and the
Group-DRO-vs-balanced comparison run non-trivially, and it is reported as an
unsupervised style partition, never as true device/patient grouping.

The k-means is fit on the training split only and applied to val/test, so the
grouping introduces no cross-split information leak.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def style_features(path: Path, resize: int = 128) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((resize, resize))
    arr = np.asarray(img, dtype=np.float64) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    lap = _laplacian_var(luma)
    return np.array([r.mean(), g.mean(), b.mean(), luma.mean(), luma.std(), colorfulness, lap])


def _laplacian_var(gray: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = gray.shape
    out = np.zeros_like(gray)
    for dy, dx, wt in [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (0, 0, -4)]:
        out += wt * np.roll(np.roll(gray, dy, 0), dx, 1)
    return float(out[1:h - 1, 1:w - 1].var())


class StyleGrouper:
    def __init__(self, k: int = 4, seed: int = 0):
        self.k = k
        self.seed = seed
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)

    def fit(self, coco: dict, image_root: str) -> "StyleGrouper":
        feats = self._features(coco, image_root)
        self.kmeans.fit(self.scaler.fit_transform(feats))
        return self

    def predict(self, coco: dict, image_root: str) -> List[int]:
        feats = self._features(coco, image_root)
        return self.kmeans.predict(self.scaler.transform(feats)).tolist()

    def _features(self, coco: dict, image_root: str) -> np.ndarray:
        root = Path(image_root)
        return np.stack([style_features(root / im["file_name"]) for im in coco["images"]])


def assign_style_groups(ann_files: Dict[str, str], image_root: str, k: int = 4, seed: int = 0) -> Dict[str, str]:
    """Fit style groups on train, apply to all splits, and rewrite the JSONs in
    place with ``background = style{c}``. Returns the split->path map."""
    with open(ann_files["train"]) as f:
        train = json.load(f)
    grouper = StyleGrouper(k, seed).fit(train, image_root)

    for split, path in ann_files.items():
        with open(path) as f:
            coco = json.load(f)
        clusters = grouper.predict(coco, image_root)
        for im, c in zip(coco["images"], clusters):
            im["background"] = f"style{c}"
        with open(path, "w") as f:
            json.dump(coco, f)
    return ann_files


def group_sizes(ann_files: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    out = {}
    for split, path in ann_files.items():
        with open(path) as f:
            coco = json.load(f)
        counts: Dict[str, int] = {}
        for im in coco["images"]:
            counts[im["background"]] = counts.get(im["background"], 0) + 1
        out[split] = dict(sorted(counts.items()))
    return out
