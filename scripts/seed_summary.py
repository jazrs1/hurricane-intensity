"""Summarize the seeded baseline runs from their prediction CSVs and training logs. Read-only."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1])
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"),
         (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
runs = {"original": REPO / "results" / "heldout_predictions.csv"}
runs |= {f"seed {s}": REPO / "results" / "seed_baseline" / f"heldout_predictions_seed{s}.csv" for s in (0, 1, 2)}

print("== training logs ==")
for s in (0, 1, 2):
    log = (REPO / "logs" / "seed_baseline" / f"train_seed{s}.log").read_text()
    ep = [(int(e), float(tr), float(va)) for e, tr, va in re.findall(r"epoch (\d+): train ([\d.]+) kt\s+val ([\d.]+) kt", log)]
    best = min(ep, key=lambda r: r[2])
    n_train = re.search(r"train samples: (\d+)\s+val samples: (\d+)", log).groups()
    storms = re.search(r"train storms: (\[.*\])", log).group(1).count("AL")
    print(f"seed {s}: epochs={len(ep)} train_storms={storms} samples train/val={n_train} "
          f"best val {best[2]:.2f} kt at epoch {best[0]} (train {best[1]:.2f}) | final-epoch val {ep[-1][2]:.2f}")

frames = {}
ref = None
for name, path in runs.items():
    d = pd.read_csv(path)
    key = d[["atcf_id", "sample_time", "truth"]].round(4)
    if ref is None:
        ref = key
    assert key.equals(ref), f"{name}: evaluated on different frames than the original"
    frames[name] = d
print(f"\nall four runs evaluated on the same {len(ref)} frames: True; storms {sorted(set(ref.atcf_id))}")


def stats(d):
    t, p = d.truth.to_numpy(), d.pred.to_numpy()
    fix = ~d.is_interpolated.astype(bool).to_numpy()
    out = {"overall MAE": np.abs(p - t).mean(), "fixes MAE": np.abs(p - t)[fix].mean(), "overall bias": (p - t).mean()}
    for lo, hi, name in BANDS:
        m = (t >= lo) & (t < hi)
        out[f"{name} n"] = m.sum()
        out[f"{name} MAE"] = np.abs(p - t)[m].mean()
        out[f"{name} bias"] = (p - t)[m].mean()
    m = t > 96
    out[">96 n"], out[">96 MAE"], out[">96 bias"] = m.sum(), np.abs(p - t)[m].mean(), (p - t)[m].mean()
    return out


tab = pd.DataFrame({k: stats(d) for k, d in frames.items()})
seeds = tab[["seed 0", "seed 1", "seed 2"]]
tab["seed mean"] = seeds.mean(axis=1)
tab["seed sd"] = seeds.std(axis=1, ddof=1)
tab["seed range"] = seeds.max(axis=1) - seeds.min(axis=1)
tab["orig - seed mean"] = tab["original"] - tab["seed mean"]
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")
print(tab.to_string())

print("\n== per storm MAE ==")
ps = pd.DataFrame({k: d.groupby("storm_name").abs_error.mean() for k, d in frames.items()})
ps["seed range"] = ps[["seed 0", "seed 1", "seed 2"]].max(axis=1) - ps[["seed 0", "seed 1", "seed 2"]].min(axis=1)
print(ps.to_string())

print("\n== bootstrap: how much of a seed-to-seed MAE gap is frame sampling? (paired, 10000 resamples of the 393 frames) ==")
rng = np.random.default_rng(0)
idx = rng.integers(0, len(ref), size=(10000, len(ref)))
names = ["seed 0", "seed 1", "seed 2"]
errs = {n: frames[n].abs_error.to_numpy() for n in names}
for i in range(3):
    for j in range(i + 1, 3):
        diff = (errs[names[i]] - errs[names[j]])[idx].mean(axis=1)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"{names[i]} - {names[j]}: {errs[names[i]].mean() - errs[names[j]].mean():+.2f} kt  95% CI [{lo:+.2f}, {hi:+.2f}]")
print("\nper-frame prediction spread across the three seeds (max - min), kt:")
P = np.stack([frames[n].pred.to_numpy() for n in names])
sp = P.max(0) - P.min(0)
print(f"  median {np.median(sp):.2f}  mean {sp.mean():.2f}  p90 {np.percentile(sp, 90):.2f}  max {sp.max():.2f}")
t = frames["seed 0"].truth.to_numpy()
print(f"  frames >96 kt: median {np.median(sp[t > 96]):.2f}  mean {sp[t > 96].mean():.2f}")
