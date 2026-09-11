"""Ingest ZNSM-iDB images into COCO with per-microscope camera groups.

Point --images-root at the extracted ZNSM corpus (folders named
<Category>_Microscope-<N>_set<M>). Box annotations are recovered from drawn
shapes for the clean-outline categories (Mannual_Seg micro1, Mannual micro2);
micro3 Mannual uses filled masks that cannot be cleanly recovered and is treated
as background. See afb_calibration.data.znsm for the recovery details.

Modes:
  annotated-only (default): keep only images with recovered boxes (~500 images).
      Use this for detection-AP tables — every image contributes to recall.
  --include-backgrounds : also keep unannotated fields as zero-box images and tag
      Without_bacilli images as background="zn_empty" for hard-negative mining.
      Use this for detector training where extra stain/artifact variety helps and
      for the hard-negative verifier experiment.
"""
from __future__ import annotations

import argparse
import json

from afb_calibration.data.znsm import convert_znsm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-root", required=True,
                    help="dir with extracted ZNSM images (Microscope-* in the paths)")
    ap.add_argument("--out-dir", default="data/znsm")
    ap.add_argument("--include-artifacts", action="store_true",
                    help="export diamond/hexagon artifact boxes as a second category")
    ap.add_argument("--include-backgrounds", action="store_true",
                    help="also keep unannotated fields (training backgrounds) and tag "
                         "Without_bacilli images as zn_empty for hard-negative mining")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = convert_znsm(args.images_root, args.out_dir, seed=args.seed,
                         include_artifacts=args.include_artifacts,
                         include_backgrounds=args.include_backgrounds)
    for split, path in paths.items():
        coco = json.load(open(path))
        cams = sorted({im["camera"] for im in coco["images"]})
        bgs = sorted({im["background"] for im in coco["images"]})
        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} boxes, "
              f"cameras={cams}, backgrounds={bgs}")


if __name__ == "__main__":
    main()
