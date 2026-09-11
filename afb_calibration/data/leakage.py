"""Cross-split leakage detection.

The audit flagged exact-duplicate images and png/jpg pairs; if a duplicated image
straddles the train/test boundary the reported numbers are inflated. This checks
for exact (content hash) and near (perceptual hash) duplicates that appear in more
than one split, which is the leak that actually matters.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image

try:
    import imagehash
except ImportError:
    imagehash = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_images(ann_files: Dict[str, str], image_root: str):
    root = Path(image_root)
    records = []
    for split, ann in ann_files.items():
        with open(ann) as f:
            coco = json.load(f)
        for im in coco["images"]:
            records.append((split, im["id"], root / im["file_name"], im["file_name"]))
    return records


def cross_split_leakage(ann_files: Dict[str, str], image_root: str, phash_hamming_max: int = 4) -> dict:
    records = _split_images(ann_files, image_root)

    by_hash: Dict[str, list] = defaultdict(list)
    for split, img_id, path, name in records:
        by_hash[_sha256(path)].append((split, name))
    exact = [v for v in by_hash.values() if len({s for s, _ in v}) > 1]

    near = []
    if imagehash is not None:
        hashes = [(split, name, imagehash.phash(Image.open(path).convert("L")))
                  for split, _, path, name in records]
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if hashes[i][0] == hashes[j][0]:
                    continue
                if hashes[i][2] - hashes[j][2] <= phash_hamming_max:
                    near.append((f"{hashes[i][0]}:{hashes[i][1]}", f"{hashes[j][0]}:{hashes[j][1]}"))

    return {
        "exact_cross_split": exact,
        "near_cross_split": near,
        "num_exact_leaks": len(exact),
        "num_near_leaks": len(near),
        "clean": len(exact) == 0 and len(near) == 0,
    }


def leaked_test_ids(ann_files: Dict[str, str], image_root: str) -> List[int]:
    """Test image ids whose content also appears in train (drop these before eval)."""
    root = Path(image_root)
    with open(ann_files["train"]) as f:
        train_hashes = {_sha256(root / im["file_name"]) for im in json.load(f)["images"]}
    with open(ann_files["test"]) as f:
        test = json.load(f)["images"]
    return [im["id"] for im in test if _sha256(root / im["file_name"]) in train_hashes]


def remove_cross_split_leaks(ann_files: Dict[str, str], image_root: str) -> Dict[str, int]:
    """Enforce split priority train > val > test: drop any image whose content
    already appeared in a higher-priority split. Rewrites the JSONs in place and
    returns removed counts."""
    root = Path(image_root)
    seen: set = set()
    removed = {}
    for split in ("train", "val", "test"):
        with open(ann_files[split]) as f:
            coco = json.load(f)
        keep_ids, kept_images = set(), []
        for im in coco["images"]:
            h = _sha256(root / im["file_name"])
            if h in seen:
                continue
            seen.add(h)
            keep_ids.add(im["id"])
            kept_images.append(im)
        removed[split] = len(coco["images"]) - len(kept_images)
        coco["images"] = kept_images
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
        with open(ann_files[split], "w") as f:
            json.dump(coco, f)
    return removed
