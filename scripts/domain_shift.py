"""Domain-shift diagnostic for the 2025 GOES-19 final test: crop brightness-temperature statistics, 2025 vs H9.

Pre-registered in logs/decisions.md (2026-10-15) and hashed before any 2025 crop was built. Descriptive
only; computed before any model is run on 2025 data, and it never loads a model.

Frame set per storm list: manifest rows with a crop, minus excluded_crops.txt (the same frames TCDataset
uses). Per frame, on the native crop (before resampling and before the 180-320 K clip): mean BT, BT sd,
5th percentile, fraction of pixels below 200 K, fraction outside the model's 180-320 K scaling range.
Each statistic is then averaged over frames, per Saffir-Simpson band (from the frame's best-track label)
and over all frames.

usage: domain_shift.py --set H9 storms_holdout_h9.txt --set 2025 storms_test_2025.txt --out <csv>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hurricane_intensity.splits import load_storm_list

parser = argparse.ArgumentParser()
parser.add_argument("--set", nargs=2, action="append", metavar=("NAME", "LIST"), required=True)
parser.add_argument("--processed-root", default="data/processed")
parser.add_argument("--out", required=True)
args = parser.parse_args()

BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
STATS = ["bt mean", "bt sd", "bt p5", "frac < 200 K", "frac outside 180-320 K"]
root = Path(args.processed_root)
excluded = set(Path("excluded_crops.txt").read_text().split())

per_frame = []
for name, lst in args.set:
    for storm in load_storm_list(lst):
        m = pd.read_csv(root / storm / "manifest.csv")
        m = m[m.crop_path.notna() & ~m.crop_path.isin(excluded)]
        for r in m.itertuples(index=False):
            bt = np.load(root / r.crop_path)["bt"].astype(np.float64)
            per_frame.append({"set": name, "atcf_id": storm, "wind_kt": r.wind_kt,
                              "band": next(b for lo, hi, b in BANDS if lo <= r.wind_kt < hi),
                              "bt mean": bt.mean(), "bt sd": bt.std(), "bt p5": np.percentile(bt, 5),
                              "frac < 200 K": (bt < 200).mean(), "frac outside 180-320 K": ((bt < 180) | (bt > 320)).mean()})
pf = pd.DataFrame(per_frame)

rows = []
order = [b for *_, b in BANDS] + ["all"]
for name, _ in args.set:
    s = pf[pf.set == name]
    for band in order:
        k = s if band == "all" else s[s.band == band]
        rows.append({"set": name, "band": band, "frames": len(k), "storms": k.atcf_id.nunique(),
                     **{c: k[c].mean() if len(k) else np.nan for c in STATS}})
table = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.out, index=False)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(table.to_string(index=False))
names = [n for n, _ in args.set]
if len(names) == 2:
    a, b = (table[table.set == n].set_index("band")[STATS] for n in names)
    print(f"\n{names[1]} - {names[0]}, by band:")
    print((b - a).reindex(order).to_string())
