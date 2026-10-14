"""Read-only: training labels by band per training list (via the real loader), per-storm Cat 3 / Cat 4 / Cat 5
bias per arm, and the arm C sensitivity without seeds 0-2. Run from the repo root after imbalance_analysis.py."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
from hurricane_intensity.dataset import TCDataset  # noqa: E402
from hurricane_intensity.splits import load_splits  # noqa: E402

BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")


def band(w):
    return next(n for lo, hi, n in BANDS if lo <= w < hi)


print("== training labels by band (frames the loader actually uses) ==")
tabs = {}
for lst in ("storms_train_baseline.txt", "storms_train.txt"):
    tr, _, _ = load_splits(train_list=lst)
    df = TCDataset("data/processed", tr).df.copy()
    df["band"] = df.wind_kt.apply(band)
    sid = df.crop_path.str.split("[/\\\\]", regex=True).str[0]
    df["storm"] = sid
    tabs[lst] = df
    c = df.band.value_counts().reindex([b for *_, b in BANDS]).fillna(0).astype(int)
    print(f"{lst}: {len(df)} frames | " + "  ".join(f"{b} {n}" for b, n in c.items()) + f" | >96: {(df.wind_kt > 96).sum()}  >=113: {(df.wind_kt >= 113).sum()}  >=137: {(df.wind_kt >= 137).sum()}")
new = tabs["storms_train.txt"][tabs["storms_train.txt"].storm.isin(["AL022024", "AL132023"])]
print("added frames by storm and band:")
print(pd.crosstab(new.storm, new.band).reindex(columns=[b for *_, b in BANDS], fill_value=0).to_string())
print("added storms' peak label in training data:", new.groupby("storm").wind_kt.max().to_dict())

print("\n== per-storm Cat 3 / Cat 4 bias and MAE, four-storm CSVs ==")
rows = []
for arm in ("A_baseline_40ep", "B_expanded_29ep", "C_expanded_40ep"):
    for s in range(10):
        d = pd.read_csv(REPO / "results" / "imbalance" / f"{arm}_seed{s}_holdout4.csv")
        d["e"] = d.pred - d.truth
        d["band"] = d.truth.apply(band)
        for (storm, b), g in d[d.band.isin(["Cat 3", "Cat 4", "Cat 5"])].groupby([d.storm_name.str.title(), "band"]):
            rows.append(dict(arm=arm[0], seed=s, storm=storm, band=b, n=len(g), bias=g.e.mean(), mae=g.e.abs().mean()))
r = pd.DataFrame(rows)
out = r.groupby(["storm", "band", "arm"]).agg(n=("n", "first"), bias_mean=("bias", "mean"), bias_sd=("bias", "std"), mae_mean=("mae", "mean"), mae_sd=("mae", "std")).unstack("arm")
print(out.to_string())

print("\n== sensitivity: arm C without seeds 0-2 (ran first; within-C order spearman -0.75) ==")
runs = pd.read_csv(REPO / "results" / "imbalance" / "summary_per_run.csv")
for st in ("h3", "h4"):
    x = runs[runs.set == st]
    a = x[x.arm == "A_baseline_40ep"]
    c_all, c_late = x[x.arm == "C_expanded_40ep"], x[(x.arm == "C_expanded_40ep") & (x.seed >= 3)]
    for m in ("overall MAE", "fixes-only MAE", ">96 MAE", ">96 bias"):
        print(f"  {st} {m:15s} C-A all seeds {c_all[m].mean() - a[m].mean():+.2f} | seeds 3-9 {c_late[m].mean() - a[m].mean():+.2f}")
