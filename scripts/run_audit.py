"""Stage-1a dataset audit CLI."""
from __future__ import annotations

import argparse
import json

from fm_robustafb.config import load_config
from fm_robustafb.data.audit import run_audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("opts", nargs="*", help="config overrides key.sub=value")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    ann = {"train": cfg.data.ann_train, "val": cfg.data.ann_val, "test": cfg.data.ann_test}[args.split]
    report = run_audit(ann, cfg.data.root, cfg.audit.phash_hamming_max, cfg.audit.embed_cosine_min)
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
