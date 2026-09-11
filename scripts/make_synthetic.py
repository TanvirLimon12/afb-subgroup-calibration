"""Generate a synthetic ZN dataset for pipeline development and CI."""
from __future__ import annotations

import argparse

from fm_robustafb.data.synthetic import generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/afb")
    ap.add_argument("--per-group", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = generate_dataset(args.out, n_per_group=args.per_group, size=args.size, seed=args.seed)
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
