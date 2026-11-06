#!/usr/bin/env python
"""Write configs/analog_splits.json: the analog-forecasting split by storm year (ATCF ID's last four digits).

    .venv\\Scripts\\python.exe scripts\\make_analog_splits.py

train 2017-2022, val 2023, test 2024-2025. Storms: the usable storms of configs/analog_storms.csv that have
files in data/storm_windows/ (a storm whose crops all failed is listed under "missing" instead). Each storm
is in exactly one split, and every split must hold both basins (AL and EP), or this exits with an error.
Cross-basin storms carry two IDs (e.g. AL022022 / EP042022, different track segments); both share a year
and so a split.

Also writes configs/analog_splits_seen.csv (atcf_id, split, resnet_role, seen_by_resnet_ensemble) so that
storms the ResNet-18 ensemble trained or selected checkpoints on can be left out of analog-vs-ensemble
comparisons. Storms are not moved between splits.
"""
from __future__ import annotations

import json

import pandas as pd

from hurricane_intensity.config import PROJECT_ROOT
from hurricane_intensity.splits import load_splits, load_storm_list

STORMS = PROJECT_ROOT / "configs" / "analog_storms.csv"
OUT = PROJECT_ROOT / "configs" / "analog_splits.json"
SEEN = PROJECT_ROOT / "configs" / "analog_splits_seen.csv"
DATA = PROJECT_ROOT / "data" / "storm_windows"
YEARS = {"train": range(2017, 2023), "val": range(2023, 2024), "test": range(2024, 2026)}


def main():
    df = pd.read_csv(STORMS)
    df = df[df["usable"] == 1]
    have = df["atcf_id"].map(lambda s: (DATA / f"{s}.npy").is_file() and (DATA / f"{s}.csv").is_file())
    split = {}
    for name, years in YEARS.items():
        ids = sorted(df.loc[have & df["year"].isin(years), "atcf_id"])
        basins = {s[:2] for s in ids}
        if basins != {"AL", "EP"}:
            raise SystemExit(f"{name} has basins {sorted(basins)}, needs AL and EP")
        split[name] = ids
    flat = [s for ids in split.values() for s in ids]
    assert len(flat) == len(set(flat)) == int(have.sum())
    out = {
        "description": "Analog-forecasting split by storm year: train 2017-2022, val 2023, test 2024-2025. "
                       "Built by scripts/make_analog_splits.py from configs/analog_storms.csv (usable storms "
                       "with data/storm_windows files).",
        **split,
        "missing": sorted(df.loc[~have, "atcf_id"]),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    # Sidecar: which analog storms the ResNet-18 ensemble saw. It was fitted on storms_train_expanded.txt
    # minus storms_val_expanded.txt and its checkpoints were selected on storms_val_expanded.txt (README);
    # the train list names the val storms, so "seen" = listed in storms_train_expanded.txt. H9 storms only
    # compared experiments: role h9_dev, not seen.
    fit, val = load_splits(PROJECT_ROOT / "storms_train_expanded.txt", PROJECT_ROOT / "storms_val_expanded.txt",
                           PROJECT_ROOT / "storms_holdout_h9.txt")[:2]
    h9 = set(load_storm_list(PROJECT_ROOT / "storms_holdout_h9.txt"))
    role = {**{s: "h9_dev" for s in h9}, **{s: "checkpoint_selection" for s in val}, **{s: "fit" for s in fit}}
    seen = pd.DataFrame([
        {"atcf_id": s, "split": name, "resnet_role": role.get(s, "none"),
         "seen_by_resnet_ensemble": int(role.get(s) in ("fit", "checkpoint_selection"))}
        for name, ids in split.items() for s in ids])
    seen.to_csv(SEEN, index=False)
    print(seen.groupby("split")["seen_by_resnet_ensemble"].sum().to_string())
    print("test storms seen:", list(seen.query("split == 'test' and seen_by_resnet_ensemble == 1")["atcf_id"]))
    for name, ids in split.items():
        print(f"{name}: {len(ids)} storms (AL {sum(s[:2] == 'AL' for s in ids)}, EP {sum(s[:2] == 'EP' for s in ids)})")
    print(f"missing (no storm_windows files): {out['missing']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
