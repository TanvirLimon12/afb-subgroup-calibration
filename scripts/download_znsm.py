"""Download ZNSM-iDB from the Google Drive mirror and extract it.

The original ZNSM-iDB host (14.139.240.55/znsm) is offline. By default this pulls
the six annotated zips (microscopes 1-3, ~300 images with recoverable drawn-shape
annotations); ``--full`` additionally attempts the whole mirror folder (all
categories, incl. background / Without_bacilli fields). Boxes are recovered later
by scripts/ingest_znsm.py, which reads the extracted tree under ``images_full``.
Requires internet access (Kaggle: Settings -> Internet: ON).
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

FOLDER_ID = "1HPcJzwKi76WwCFYj7dHUgVA31dAyFyTF"

ANNOTATED = {
    "Mannual_Seg_Microscope-1_set6.zip": "1wUBUn5QOeKiagITBnHqOZekPPKtZdUfv",
    "Mannual_Seg_Microscope-1_set7.zip": "16IkPKi0rf3Pkf4pkCY6cyHWTNuExW02o",
    "Mannual_Microscope-2_set1.zip": "14p-L0fAYLW7PQiTnFqv9LvxV_G2zO6az",
    "Mannual_Microscope-2_set2.zip": "1nW7e22SifZHz-veRfFRznWbTakLyhUIz",
    "Mannual_Microscope-3_set1.zip": "17wBILCPcG9DqTcjg0DB10ELLO-ihSUcT",
    "Mannual_Microscope-3_set2.zip": "1fBcuke_ErQSQ49izn7998kTdYwVDioeP",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="Datasets/ZNSM-iDB")
    ap.add_argument("--full", action="store_true",
                    help="also fetch the whole mirror folder (backgrounds, all categories)")
    args = ap.parse_args()

    import gdown

    out = Path(args.out)
    zips = out / "zips"
    images = out / "images_full"
    zips.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    for name, file_id in ANNOTATED.items():
        dest = zips / name
        if not dest.exists():
            gdown.download(id=file_id, output=str(dest), quiet=False)

    if args.full:
        try:
            gdown.download_folder(id=FOLDER_ID, output=str(zips), quiet=False, use_cookies=False)
        except Exception as exc:
            print(f"folder download incomplete ({exc}); continuing with annotated zips")

    for z in sorted(zips.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(images)
        except zipfile.BadZipFile:
            print(f"skip corrupt zip: {z.name}")

    n = sum(1 for _ in images.rglob("*.jpg") if "Thumbs" not in _.name)
    print(f"done: {n} images under {images}")


if __name__ == "__main__":
    main()
