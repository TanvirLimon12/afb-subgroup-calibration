"""Convert a YOLO-format detection dataset to the COCO-style schema used here.

The Raw Sputum Microscopy corpus ships as ``images/{split}`` + ``labels/{split}``
with YOLO ``class cx cy w h`` (normalised) annotations and a single AFB class.
Filenames encode no camera or slide identity, so - consistent with the pipeline's
Phase-1 framing - every image is assigned a single ``unknown`` group and results
are reported as image-level validation, not patient/camera-independent. An
optional metadata CSV (file_name,camera,background) overrides this once slide or
device identifiers arrive.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _resolve_image(labels_stem: str, image_dir: Path) -> Optional[Path]:
    # Prefer lossless PNG (raw) when both formats are present for a basename.
    for ext in IMAGE_EXTS:
        cand = image_dir / f"{labels_stem}{ext}"
        if cand.exists():
            return cand
    return None


def _read_metadata(csv_path: Optional[str]) -> Dict[str, Tuple[str, str]]:
    if not csv_path:
        return {}
    mapping = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["file_name"]] = (row.get("camera", "unknown"), row.get("background", "b0"))
    return mapping


def yolo_split_to_coco(
    dataset_root: str | Path, split: str, single_class: bool = True,
    metadata_csv: Optional[str] = None, default_group: Tuple[str, str] = ("unknown", "b0"),
) -> dict:
    root = Path(dataset_root)
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    metadata = _read_metadata(metadata_csv)

    images, annotations = [], []
    ann_id = 0
    categories = [{"id": 1, "name": "afb"}]
    if not single_class:
        categories = [{"id": 1, "name": "afb"}, {"id": 2, "name": "class1"}]

    for img_id, label_file in enumerate(sorted(label_dir.glob("*.txt"))):
        image_path = _resolve_image(label_file.stem, image_dir)
        if image_path is None:
            continue
        w, h = Image.open(image_path).size
        camera, background = metadata.get(image_path.name, default_group)
        rel = image_path.relative_to(root).as_posix()
        images.append({"id": img_id, "file_name": rel, "width": w, "height": h,
                       "camera": camera, "background": background})
        for line in label_file.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = (float(p) for p in parts)
            category_id = 1 if single_class else int(cls) + 1
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            annotations.append({"id": ann_id, "image_id": img_id,
                                "bbox": [x, y, bw * w, bh * h], "category_id": category_id,
                                "iscrowd": 0, "area": bw * w * bh * h})
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def convert_dataset(
    dataset_root: str | Path, out_dir: str | Path, single_class: bool = True,
    metadata_csv: Optional[str] = None,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "val", "test"):
        coco = yolo_split_to_coco(dataset_root, split, single_class, metadata_csv)
        p = out_dir / f"{split}.json"
        with open(p, "w") as f:
            json.dump(coco, f)
        paths[split] = str(p)
    return paths
