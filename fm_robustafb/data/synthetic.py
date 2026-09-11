"""Synthetic ZN-microscopy generator.

Produces raw (deliberately unnormalised) Ziehl-Neelsen-like fields with tiny
rod-shaped reddish bacilli on stained backgrounds, plus hard-negative artifacts
(stain deposits, fibers, debris) that are NOT annotated. Per-camera and
per-background style shifts induce the camera x background groups the pipeline
targets. This exists to exercise the full pipeline end-to-end and for CI; it is
not a substitute for the real corpus.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

# Per-camera style: resolution scale, blur, colour gain (RGB), noise.
CAMERA_STYLE: Dict[str, dict] = {
    "hayear": {"blur": 0.6, "gain": (1.05, 0.98, 1.02), "noise": 0.015, "gamma": 1.0},
    "optilab": {"blur": 1.2, "gain": (0.95, 1.0, 1.12), "noise": 0.030, "gamma": 1.15},
}

# Per-background profile: base field colour and artifact density.
BACKGROUND_STYLE: Dict[str, dict] = {
    "b0": {"base": (0.62, 0.70, 0.88), "artifacts": 6},
    "b1": {"base": (0.70, 0.78, 0.80), "artifacts": 14},
    "b2": {"base": (0.55, 0.66, 0.85), "artifacts": 10},
    "b3": {"base": (0.75, 0.72, 0.70), "artifacts": 20},
}


def _draw_rod(canvas: np.ndarray, cx, cy, length, width, angle, color, rng) -> Tuple[int, int, int, int]:
    h, w, _ = canvas.shape
    ca, sa = math.cos(angle), math.sin(angle)
    half = length / 2
    xs, ys = [], []
    steps = max(6, int(length))
    for t in np.linspace(-half, half, steps):
        px = cx + t * ca
        py = cy + t * sa
        for du in np.linspace(-width / 2, width / 2, max(2, int(width))):
            qx = int(round(px - du * sa))
            qy = int(round(py + du * ca))
            if 0 <= qx < w and 0 <= qy < h:
                jitter = 1.0 + rng.normal(0, 0.08)
                canvas[qy, qx] = np.clip(np.array(color) * jitter, 0, 1)
                xs.append(qx)
                ys.append(qy)
    if not xs:
        return None
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def render_image(
    rng: np.random.Generator, camera: str, background: str, size: int = 512,
    n_bacilli: Tuple[int, int] = (3, 18),
) -> Tuple[np.ndarray, List[List[float]]]:
    cam = CAMERA_STYLE[camera]
    bg = BACKGROUND_STYLE[background]
    base = np.array(bg["base"], dtype=np.float32)
    field = np.ones((size, size, 3), dtype=np.float32) * base
    field += rng.normal(0, 0.03, field.shape).astype(np.float32)

    gx, gy = rng.uniform(-0.15, 0.15, 2)
    ax = np.linspace(-1, 1, size)
    grad = 1.0 + gx * ax[None, :, None] + gy * ax[:, None, None]
    field = field * grad

    # Hard-negative artifacts (unannotated distractors).
    for _ in range(bg["artifacts"]):
        kind = rng.integers(0, 3)
        cx, cy = rng.uniform(0, size, 2)
        if kind == 0:  # stain deposit blob
            r = rng.uniform(3, 9)
            color = np.array([0.75, 0.25, 0.35]) + rng.normal(0, 0.05, 3)
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            field[mask] = np.clip(color, 0, 1)
        elif kind == 1:  # fiber
            _draw_rod(field, cx, cy, rng.uniform(30, 90), rng.uniform(1, 2),
                      rng.uniform(0, math.pi), (0.4, 0.45, 0.6), rng)
        else:  # debris speck
            r = rng.uniform(1, 3)
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            field[mask] = np.clip(rng.uniform(0.2, 0.4, 3), 0, 1)

    # Bacilli (annotated positives): thin reddish/magenta rods.
    boxes: List[List[float]] = []
    n = int(rng.integers(n_bacilli[0], n_bacilli[1] + 1))
    for _ in range(n):
        cx, cy = rng.uniform(20, size - 20, 2)
        length = rng.uniform(8, 22)
        width = rng.uniform(1.5, 3.0)
        angle = rng.uniform(0, math.pi)
        color = np.array([0.85, 0.15, 0.30]) + rng.normal(0, 0.04, 3)
        box = _draw_rod(field, cx, cy, length, width, angle, color, rng)
        if box is not None:
            x0, y0, x1, y1 = box
            boxes.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])

    # Camera style: colour gain, gamma, blur, sensor noise.
    field = field * np.array(cam["gain"], dtype=np.float32)
    field = np.clip(field, 0, 1) ** cam["gamma"]
    if cam["blur"] > 0:
        field = _blur(field, cam["blur"])
    field = field + rng.normal(0, cam["noise"], field.shape).astype(np.float32)
    field = np.clip(field, 0, 1)
    return (field * 255).astype(np.uint8), boxes


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image, sigma=(sigma, sigma, 0))


def generate_dataset(
    out_dir: str | Path,
    n_per_group: int = 8,
    cameras: Sequence[str] = ("hayear", "optilab"),
    backgrounds: Sequence[str] = ("b0", "b1", "b2", "b3"),
    size: int = 512,
    splits: Dict[str, float] = None,
    seed: int = 0,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    splits = splits or {"train": 0.6, "val": 0.2, "test": 0.2}
    rng = np.random.default_rng(seed)

    records: List[dict] = []
    img_id = 0
    ann_id = 0
    for camera in cameras:
        for background in backgrounds:
            for _ in range(n_per_group):
                arr, boxes = render_image(rng, camera, background, size=size)
                fname = f"img_{img_id:06d}.png"
                Image.fromarray(arr).save(out_dir / "images" / fname)
                anns = []
                for b in boxes:
                    anns.append({"id": ann_id, "image_id": img_id, "bbox": b,
                                 "category_id": 1, "iscrowd": 0, "area": b[2] * b[3]})
                    ann_id += 1
                records.append({
                    "image": {"id": img_id, "file_name": f"images/{fname}", "width": size,
                              "height": size, "camera": camera, "background": background},
                    "annotations": anns,
                })
                img_id += 1

    perm = rng.permutation(len(records))
    bounds = np.cumsum([int(len(records) * splits[s]) for s in ["train", "val"]])
    assign = {"train": perm[: bounds[0]], "val": perm[bounds[0]: bounds[1]], "test": perm[bounds[1]:]}

    paths = {}
    categories = [{"id": 1, "name": "afb"}]
    for split, idxs in assign.items():
        coco = {"images": [], "annotations": [], "categories": categories}
        for i in idxs:
            coco["images"].append(records[i]["image"])
            coco["annotations"].extend(records[i]["annotations"])
        p = out_dir / f"{split}.json"
        with open(p, "w") as f:
            json.dump(coco, f)
        paths[split] = str(p)
    return paths
