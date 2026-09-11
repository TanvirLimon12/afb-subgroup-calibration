"""ZNSM-iDB ingestion: recover box annotations from drawn geometric shapes.

ZNSM-iDB (Shah et al., 2017; P05) is a multi-microscope ZN corpus. Its manually
annotated category marks bacilli with geometric shapes drawn directly on the
image - per the paper: circle/oval for a single bacillus, square/rectangle for
occluded bacilli, diamond for unclassified red structures, hexagon for artifacts.
No coordinate files are shipped. This module recovers machine-readable boxes by
detecting the (near-black) annotation ink and classifying each shape:

    oval / rectangle -> AFB positive box
    diamond / hexagon -> artifact (hard-negative) box

The microscope id in the path becomes the camera group, giving a genuine
cross-sensor axis (three bright-field microscopes at different resolutions).
Recovered boxes approximate the drawn shapes; this is stated as annotation
recovery, not original ground truth.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

POSITIVE = "afb"
ARTIFACT = "artifact"


def _shape_label(contour) -> str:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    extent = area / max(1.0, w * h)
    peri = cv2.arcLength(contour, True)
    n = len(cv2.approxPolyDP(contour, 0.03 * peri, True))
    circularity = 4 * np.pi * area / (peri * peri + 1e-9)
    if n <= 5 and extent >= 0.82:
        return "rectangle"      # occluded bacilli -> positive
    if n <= 5 and extent <= 0.62:
        return "diamond"        # unclassified red structure -> artifact
    if circularity >= 0.78:
        return "oval"           # single bacillus -> positive
    return "hexagon"            # low-circularity marker -> artifact


_POS = {"oval", "rectangle"}


def extract_boxes(image_path: str | Path, black_thresh: int = 70, min_area: int = 30,
                 max_area_frac: float = 0.2, filled_ink_frac: float = 0.15
                 ) -> Tuple[List[list], List[list]]:
    """Return (positive_boxes, artifact_boxes) as xyxy lists for one image.

    ``filled_ink_frac`` guards against filled-mask annotations: some ZNSM
    Microscope-3 'Mannual' images mark bacilli as filled black regions covering
    ~35% of the image, where individual bacilli are not separable as contours.
    When the total ink fraction exceeds this threshold, the image is treated as
    a filled-mask annotation and no boxes are recovered (returns empty), rather
    than producing one giant garbage box. Clean drawn-shape outlines are ~1-2%
    ink, so the default 15% threshold cleanly separates the two regimes.
    """
    arr = cv2.imread(str(image_path))
    if arr is None:
        return [], []
    h, w = arr.shape[:2]
    ink = (arr.max(axis=2) < black_thresh).astype(np.uint8) * 255
    if float(ink.mean() / 255) > filled_ink_frac:
        return [], []  # filled-mask annotation; not recoverable as boxes
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pos, art = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > max_area_frac * w * h:
            continue
        label = _shape_label(c)
        box = [float(x), float(y), float(x + bw), float(y + bh)]
        (pos if label in _POS else art).append(box)
    return pos, art


def _microscope_of(path: Path) -> str:
    m = re.search(r"Microscope-(\d)", str(path))
    return f"micro{m.group(1)}" if m else "unknown"


def _category_of(path: Path) -> str:
    """The ZNSM distribution category (Autofocus, Mannual, Without_bacilli, ...).

    Determined from the parent directory name (e.g. 'Mannual_Seg_Microscope-1_set6'
    -> 'Mannual_Seg'). Determines whether an image carries recoverable annotations.
    Only the 'Mannual' / 'Mannual_Seg' categories have drawn annotations (ink on
    the image); the rest are raw fields used as training backgrounds / hard
    negatives. Microscope-3 'Mannual' images use heavy filled-mask annotations
    (ink covers ~35% of the image) that cannot be cleanly recovered as boxes; they
    are skipped from box recovery and treated as raw backgrounds.
    """
    name = Path(path).parent.name
    for part in re.split(r"_Microscope-\d", name):
        if part:
            return part
    return name


# Categories whose images carry recoverable drawn-shape annotations.
_ANNOTATED_CATEGORIES = {"Mannual", "Mannual_Seg"}


def convert_znsm(
    images_root: str | Path, out_dir: str | Path, splits: Dict[str, float] = None, seed: int = 0,
    include_artifacts: bool = False, include_backgrounds: bool = False,
) -> Dict[str, str]:
    """Build COCO-style splits from ZNSM images.

    camera = microscope id, background = "zn".

    Annotation recovery:
      - ``Mannual_Seg`` (micro1) and ``Mannual`` (micro2) carry clean drawn-shape
        annotations (oval/rectangle = bacilli, diamond/hexagon = artifact); boxes
        are recovered from the ink via :func:`extract_boxes`.
      - ``Mannual`` (micro3) uses heavy filled-mask annotations (~35% ink) that
        cannot be cleanly recovered as individual boxes; these images are treated
        as backgrounds only (no boxes), not as positives.
      - All other categories (Autofocus, Overlapping, ...) are raw unannotated
        fields.

    Backgrounds and hard negatives:
      - When ``include_backgrounds`` is True, unannotated images are kept in the
        splits as zero-box images. This adds stain/artifact variety to detector
        training. They contribute to background learning only, never to AP.
      - ``Without_bacilli`` images are tagged ``background="zn_empty"`` so the
        verifier can draw hard-negative crops from confirmed-bacillus-free fields.
      - When ``include_backgrounds`` is False (default, the original behaviour),
        only images with at least one recovered box are kept, matching the
        annotated-only corpus (micro1+micro2, ~194 images with recovered boxes).

    Artifact boxes (diamond/hexagon) are always exported to ``artifacts.json``
    for the hard-negative experiment.
    """
    images_root = Path(images_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = splits or {"train": 0.6, "val": 0.2, "test": 0.2}
    rng = np.random.default_rng(seed)

    paths = sorted(p for p in images_root.rglob("*.jpg") if "Thumbs" not in p.name)
    by_cam: Dict[str, list] = {}
    for p in paths:
        by_cam.setdefault(_microscope_of(p), []).append(p)

    records, artifacts = [], {}
    img_id, ann_id = 0, 0
    for cam, cam_paths in by_cam.items():
        for p in cam_paths:
            cat = _category_of(p)
            # Only annotated categories get box recovery. extract_boxes itself
            # rejects filled-mask annotations (e.g. micro3 Mannual, ~35% ink) by
            # returning empty when the ink fraction is too high, so those images
            # yield no boxes and are kept only when include_backgrounds is set.
            recoverable = cat in _ANNOTATED_CATEGORIES
            pos, art = (extract_boxes(p) if recoverable else ([], []))
            # Decide whether to keep this image:
            #  - if it has recovered boxes -> always keep (annotated positive)
            #  - if no boxes recovered -> keep only when include_backgrounds
            #    (otherwise it is a recall-free image that shouldn't be in an AP table)
            if not pos and not include_backgrounds:
                continue
            from PIL import Image
            w, h = Image.open(p).size
            rel = p.relative_to(images_root).as_posix()
            # Without_bacilli images get a distinct background tag for hard-neg mining.
            bg = "zn_empty" if cat == "Without_bacilli" else "zn"
            anns = [{"id": None, "image_id": img_id, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                     "category_id": 1, "iscrowd": 0, "area": (b[2] - b[0]) * (b[3] - b[1])} for b in pos]
            if include_artifacts:
                anns += [{"id": None, "image_id": img_id, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                          "category_id": 2, "iscrowd": 0} for b in art]
            records.append({"image": {"id": img_id, "file_name": rel, "width": w, "height": h,
                                      "camera": cam, "background": bg}, "annotations": anns})
            artifacts[img_id] = art
            img_id += 1

    perm = rng.permutation(len(records))
    n_train = int(len(records) * splits["train"])
    n_val = int(len(records) * splits["val"])
    assign = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": perm[n_train + n_val:]}

    categories = [{"id": 1, "name": POSITIVE}] + ([{"id": 2, "name": ARTIFACT}] if include_artifacts else [])
    out = {}
    for split, idxs in assign.items():
        coco = {"images": [], "annotations": [], "categories": categories}
        for i in idxs:
            coco["images"].append(records[i]["image"])
            for a in records[i]["annotations"]:
                a = dict(a, id=ann_id)
                ann_id += 1
                coco["annotations"].append(a)
        path = out_dir / f"{split}.json"
        with open(path, "w") as f:
            json.dump(coco, f)
        out[split] = str(path)

    with open(out_dir / "artifacts.json", "w") as f:
        json.dump({str(k): v for k, v in artifacts.items()}, f)
    return out
