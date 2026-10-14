"""Recompute every figure on site/index.html that came from the single 9.00 kt run, for each baseline rerun.

Sources (per-sample held-out predictions on Ian, Nicole and Claudette, 393 frames):
- the published run: results/heldout_predictions.csv
- 33 baseline reruns at 20 epochs (listed in results/run_order/all_baseline_reruns.csv): results/seed_baseline/,
  results/epoch_traces/ (traced runs at their selected epoch; control runs), results/epoch_traces_60/ (selected
  within the first 20 epochs), results/run_order/
- arm A of the class-imbalance experiment (original nine storms, 40 epochs): results/imbalance/A_*_holdout3.csv

The site's category table was made with the old evaluate.py bands (inclusive integer edges, which leave
fractional labels such as 33.5 kt in no band); "site bands" reproduces those, "contiguous" uses the current ones.
Writes results/site_figures/figures_by_run.csv and summary.csv. Read-only otherwise. Run from the repo root.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
R = REPO / "results"
SITE_BANDS = [(0, 33, "TD"), (34, 63, "TS"), (64, 82, "Cat 1"), (83, 95, "Cat 2"), (96, 112, "Cat 3"), (113, 136, "Cat 4"), (137, 999, "Cat 5")]
CONT_BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
KEY = ["atcf_id", "sample_time"]

ref = pd.read_csv(R / "heldout_predictions.csv")
meta = ref[KEY + ["storm_name", "is_interpolated", "truth"]]


def from_epoch_file(path, seed, epoch):
    e = pd.read_csv(path)
    e = e[(e.seed == seed) & (e.epoch == epoch)][KEY + ["pred"]]
    return meta.merge(e, on=KEY, how="inner")


reruns = pd.read_csv(R / "run_order" / "all_baseline_reruns.csv")
runs = [("published 9.00 run", "published", ref)]
for _, r in reruns.iterrows():
    name, s = r.run, int(r.run.split("seed ")[1].split()[0])
    if name.startswith("first batch"):
        d = pd.read_csv(R / "seed_baseline" / f"heldout_predictions_seed{s}{'_repeat' if 'repeat' in name else ''}.csv")
    elif name.startswith("traced"):
        d = from_epoch_file(R / "epoch_traces" / "heldout_predictions_by_epoch.csv", s, int(r.selected_epoch))
    elif name.startswith("control"):
        d = pd.read_csv(R / "epoch_traces" / f"control_heldout_predictions_seed{s}.csv")
    elif name.startswith("long60"):
        d = from_epoch_file(R / "epoch_traces_60" / "heldout_predictions_by_epoch.csv", s, int(r.selected_epoch))
    else:
        prefix = {"spaced": "spaced", "b2b": "b2b", "replication": "rep"}[name.split()[0]]
        d = pd.read_csv(R / "run_order" / f"{prefix}_seed{s}.csv")
    runs.append((name, "33 reruns, 20 ep", d))
for s in range(10):
    runs.append((f"A seed {s}", "A, 40 ep", pd.read_csv(R / "imbalance" / f"A_baseline_40ep_seed{s}_holdout3.csv")))


def figures(d):
    d = d.sort_values(KEY).reset_index(drop=True)
    t, p = d.truth.to_numpy(), d.pred.to_numpy()
    e = p - t
    out = {"n": len(d), "overall MAE": np.abs(e).mean(), "fixes-only MAE": np.abs(e)[~d.is_interpolated.astype(bool).to_numpy()].mean()}
    for tag, bands, inclusive in (("site", SITE_BANDS, True), ("cont", CONT_BANDS, False)):
        for lo, hi, b in bands:
            m = (t >= lo) & ((t <= hi) if inclusive else (t < hi))
            out[f"{tag} n {b}"], out[f"{tag} MAE {b}"], out[f"{tag} bias {b}"] = m.sum(), np.abs(e)[m].mean(), e[m].mean()
    out["site Cat1/Cat4 MAE ratio"] = out["site MAE Cat 1"] / out["site MAE Cat 4"]
    out["bias above 120 kt"] = e[t > 120].mean()
    out["bias 96-120 kt"] = e[(t > 96) & (t <= 120)].mean()
    ian = d[d.atcf_id == "AL092022"].reset_index(drop=True)
    pk = ian.truth.idxmax()
    out["Ian error at 140 kt peak frame"] = ian.pred[pk] - ian.truth[pk]
    out["Ian mean error after peak"] = (ian.pred - ian.truth)[pk + 1:].mean()
    return out


rows = [{"run": n, "group": g, **figures(d)} for n, g, d in runs]
t = pd.DataFrame(rows)
chk = t[t.group == "33 reruns, 20 ep"].set_index("run")["overall MAE"].round(2) - reruns.set_index("run").heldout_mae.round(2)
assert (t.n == 393).all(), t[t.n != 393]
assert chk.abs().max() <= 0.011, chk
out = R / "site_figures"
out.mkdir(exist_ok=True)
t.to_csv(out / "figures_by_run.csv", index=False)

cols = [c for c in t.columns if c not in ("run", "group", "n") and not c.startswith(("site n", "cont n"))]
pub = t[t.group == "published"].iloc[0]
summ = []
for c in cols:
    r = {"figure": c, "published run": pub[c]}
    for g in ("33 reruns, 20 ep", "A, 40 ep"):
        x = t[t.group == g][c]
        r[f"{g} mean"], r[f"{g} sd"], r[f"{g} min"], r[f"{g} max"] = x.mean(), x.std(), x.min(), x.max()
        r[f"{g} rank of published (runs at or below)"] = int((x <= pub[c] + 1e-9).sum())
    summ.append(r)
s = pd.DataFrame(summ)
s.to_csv(out / "summary.csv", index=False)
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")
print(f"runs: {t.group.value_counts().to_dict()}; rerun MAEs match all_baseline_reruns.csv to {chk.abs().max():.3f} kt")
print("band sizes, site bands:", {b: int(pub[f'site n {b}']) for *_, b in SITE_BANDS}, "| contiguous:", {b: int(pub[f'cont n {b}']) for *_, b in CONT_BANDS})
print(s.to_string(index=False))
