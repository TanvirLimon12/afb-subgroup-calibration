"""Convert a YOLO-format dataset (images/{split} + labels/{split}) to COCO JSON."""
from __future__ import annotations

import argparse

from afb_calibration.data.convert import convert_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True,
                    help="root containing images/{train,val,test} and labels/{...}")
    ap.add_argument("--out-dir", required=True, help="where to write {train,val,test}.json")
    ap.add_argument("--metadata-csv", default=None,
                    help="optional file_name,camera,background CSV for group labels")
    ap.add_argument("--multi-class", action="store_true",
                    help="keep original YOLO class ids (default: collapse to single AFB class)")
    args = ap.parse_args()

    paths = convert_dataset(args.dataset_root, args.out_dir,
                            single_class=not args.multi_class, metadata_csv=args.metadata_csv)
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
