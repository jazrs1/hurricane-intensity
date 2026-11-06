#!/usr/bin/env python
"""Write configs/analog_splits.json: the analog-forecasting split by storm year (ATCF ID's last four digits).

    .venv\\Scripts\\python.exe scripts\\make_analog_splits.py

train 2017-2022, val 2023, test 2024-2025. Storms: the usable storms of configs/analog_storms.csv that have
files in data/storm_windows/ (a storm whose crops all failed is listed under "missing" instead). Each storm
is in exactly one split, and every split must hold both basins (AL and EP), or this exits with an error.
Cross-basin storms carry two IDs (e.g. AL022022 / EP042022, different track segments); both share a year
and so a split.
"""
from __future__ import annotations

import json

import pandas as pd

from hurricane_intensity.config import PROJECT_ROOT

STORMS = PROJECT_ROOT / "configs" / "analog_storms.csv"
OUT = PROJECT_ROOT / "configs" / "analog_splits.json"
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
    for name, ids in split.items():
        print(f"{name}: {len(ids)} storms (AL {sum(s[:2] == 'AL' for s in ids)}, EP {sum(s[:2] == 'EP' for s in ids)})")
    print(f"missing (no storm_windows files): {out['missing']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
