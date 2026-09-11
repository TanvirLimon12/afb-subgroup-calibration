"""Stage-1a dataset audit: duplication, leakage, and distribution reporting.

Produces the camera x background distribution table and box-size statistics that
decide tile size, input resolution, and whether a P2 feature level is justified,
rather than assuming small-object loss categorically.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import imagehash
except ImportError:  # optional
    imagehash = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_coco(ann_file: str) -> dict:
    with open(ann_file) as f:
        return json.load(f)


def exact_duplicates(image_paths: Dict[int, Path]) -> List[List[int]]:
    by_hash: Dict[str, List[int]] = defaultdict(list)
    for img_id, path in image_paths.items():
        by_hash[_sha256(path)].append(img_id)
    return [ids for ids in by_hash.values() if len(ids) > 1]


def near_duplicates_phash(image_paths: Dict[int, Path], hamming_max: int = 6) -> List[tuple]:
    if imagehash is None:
        return []
    hashes = {img_id: imagehash.phash(Image.open(p).convert("L")) for img_id, p in image_paths.items()}
    ids = list(hashes)
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if hashes[ids[i]] - hashes[ids[j]] <= hamming_max:
                pairs.append((ids[i], ids[j]))
    return pairs


def near_duplicates_embedding(
    image_paths: Dict[int, Path], embed_fn: Callable[[Image.Image], np.ndarray], cosine_min: float = 0.98
) -> List[tuple]:
    ids = list(image_paths)
    embeds = np.stack([_l2(embed_fn(Image.open(image_paths[i]).convert("RGB"))) for i in ids])
    sim = embeds @ embeds.T
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim[i, j] >= cosine_min:
                pairs.append((ids[i], ids[j], float(sim[i, j])))
    return pairs


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def group_distribution(coco: dict) -> Dict[str, int]:
    dist: Dict[str, int] = defaultdict(int)
    for im in coco["images"]:
        dist[f"{im['camera']}/{im['background']}"] += 1
    return dict(sorted(dist.items()))


def box_statistics(coco: dict) -> dict:
    per_image = defaultdict(int)
    ws, hs, areas, ars = [], [], [], []
    img_group = {im["id"]: f"{im['camera']}/{im['background']}" for im in coco["images"]}
    by_group = defaultdict(lambda: {"w": [], "h": [], "area": [], "ar": []})
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        per_image[ann["image_id"]] += 1
        ar = max(w, h) / max(1e-6, min(w, h))
        ws.append(w); hs.append(h); areas.append(w * h); ars.append(ar)
        g = img_group[ann["image_id"]]
        by_group[g]["w"].append(w); by_group[g]["h"].append(h)
        by_group[g]["area"].append(w * h); by_group[g]["ar"].append(ar)

    counts = [per_image.get(im["id"], 0) for im in coco["images"]]

    def summ(v):
        v = np.asarray(v, dtype=np.float64)
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
                "min": float(v.min()), "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    return {
        "width": summ(ws), "height": summ(hs), "area": summ(areas), "aspect_ratio": summ(ars),
        "boxes_per_image": summ(counts),
        "per_group": {g: {k: summ(vals) for k, vals in d.items()} for g, d in by_group.items()},
    }


def p2_justified(box_stats: dict, p3_stride: int = 8) -> dict:
    """P2 (stride 4) is justified only if a meaningful fraction of objects fall
    below the P3 effective stride. Empirical decision, not an assumption."""
    w = box_stats["width"]
    h = box_stats["height"]
    below = w.get("p50", 1e9) < p3_stride or h.get("p50", 1e9) < p3_stride
    return {"median_w": w.get("p50"), "median_h": h.get("p50"), "p3_stride": p3_stride,
            "recommend_p2": bool(below)}


def run_audit(
    ann_file: str, image_root: str, phash_hamming_max: int = 6, cosine_min: float = 0.98,
    embed_fn: Optional[Callable] = None,
) -> dict:
    coco = _load_coco(ann_file)
    root = Path(image_root)
    image_paths = {im["id"]: root / im["file_name"] for im in coco["images"]}
    stats = box_statistics(coco)
    report = {
        "num_images": len(coco["images"]),
        "num_boxes": len(coco["annotations"]),
        "group_distribution": group_distribution(coco),
        "exact_duplicates": exact_duplicates(image_paths),
        "near_duplicates_phash": near_duplicates_phash(image_paths, phash_hamming_max),
        "box_statistics": stats,
        "p2_recommendation": p2_justified(stats),
    }
    if embed_fn is not None:
        report["near_duplicates_embedding"] = near_duplicates_embedding(image_paths, embed_fn, cosine_min)
    return report
