#!/usr/bin/env python
"""Throwaway: scan every crop .npz under data/processed/*/crops and report
distinct (H, W) shapes with counts, plus global min/max of bt.

Not part of the pipeline -- just answers "what does the dataset class need
to handle" before writing it. Safe to delete.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from hurricane_intensity import config


def main() -> None:
    crop_paths = sorted(config.PROCESSED_DIR.glob("*/crops/*.npz"))
    print(f"found {len(crop_paths)} crop files under {config.PROCESSED_DIR}")

    shape_counts: Counter[tuple[int, int]] = Counter()
    bt_min = np.inf
    bt_max = -np.inf
    n_ok = 0
    n_failed = 0

    for path in crop_paths:
        try:
            with np.load(path) as npz:
                bt = npz["bt"]
                shape_counts[bt.shape] += 1
                bt_min = min(bt_min, float(bt.min()))
                bt_max = max(bt_max, float(bt.max()))
            n_ok += 1
        except Exception as e:
            print(f"  FAILED to load {path}: {e}")
            n_failed += 1

    print(f"\nloaded ok: {n_ok}, failed: {n_failed}")

    print(f"\ndistinct (H, W) shapes across {n_ok} crops:")
    for shape, count in shape_counts.most_common():
        print(f"  {shape}: {count}")

    print(f"\nbt min: {bt_min}")
    print(f"bt max: {bt_max}")


if __name__ == "__main__":
    main()
