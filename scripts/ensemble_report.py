"""Metrics for an ensemble predictions CSV written by scripts/ensemble_eval.py.

Metric definitions are those of scripts/storms_analysis.py (high end is truth > 96 kt; fixes are
non-interpolated frames; bands lower-inclusive). Per-storm rows are keyed by ATCF id, because storm
names are reused (Humberto 2019 trains; Humberto 2025 is in the 2025 test set).

Optional:
  --member-csv-pattern  cross-check: each member column pred_<stem> must equal the `pred` column of
                        the matching single-model CSV (joined on crop_path), e.g.
                        results/storms/expanded_{stem}_h9.csv
  --reference           summary_per_run.csv to report the member runs' mean and sd beside the ensemble
  --reference-arm, --reference-set   rows of the reference to use
No inferential tests: one ensemble is one prediction set, not a sample of runs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--member-csv-pattern", default=None)
parser.add_argument("--reference", default=None)
parser.add_argument("--reference-arm", default="expanded")
parser.add_argument("--reference-set", default="h9")
parser.add_argument("--reweight-to", default=None,
                    help="predictions CSV whose band composition defines the weights for a composition-adjusted overall MAE")
parser.add_argument("--out", required=True, help="metrics table (CSV)")
args = parser.parse_args()

BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
REWEIGHT = None
if args.reweight_to:
    _t = pd.read_csv(args.reweight_to).truth.to_numpy()
    REWEIGHT = {name: float(((_t >= a) & (_t < b)).mean()) for a, b, name in BANDS}
    print(f"band weights from {args.reweight_to}: " + ", ".join(f"{k} {v:.3f}" for k, v in REWEIGHT.items()))


def metrics(t, p, fix, storm):
    e = p - t
    hi, lo = t > 96, t < 64
    m = {"n": len(t), ">96 n": int(hi.sum()), "overall MAE": np.abs(e).mean(), "fixes-only MAE": np.abs(e)[fix].mean(),
         "overall bias": e.mean(), ">96 MAE": np.abs(e)[hi].mean() if hi.any() else np.nan,
         ">96 bias": e[hi].mean() if hi.any() else np.nan,
         "spearman >96": stats.spearmanr(p[hi], t[hi])[0] if hi.sum() > 2 else np.nan,
         "<64 MAE": np.abs(e)[lo].mean() if lo.any() else np.nan, "<64 bias": e[lo].mean() if lo.any() else np.nan}
    for a, b, name in BANDS:
        k = (t >= a) & (t < b)
        m[f"bias {name}"] = e[k].mean() if k.any() else np.nan
        m[f"MAE {name}"] = np.abs(e)[k].mean() if k.any() else np.nan
        m[f"n {name}"] = int(k.sum())
    if REWEIGHT is not None:  # overall MAE with this set's per-band MAE weighted by the reference set's band shares
        m["MAE reweighted to reference bands"] = sum(REWEIGHT[name] * m[f"MAE {name}"] for *_, name in BANDS if REWEIGHT[name] > 0)
    for s in np.unique(storm):
        k = storm == s
        m[f"MAE {s}"] = np.abs(e)[k].mean()
        if (k & hi).sum() >= 10:
            m[f">96 MAE {s}"], m[f">96 bias {s}"] = np.abs(e)[k & hi].mean(), e[k & hi].mean()
            m[f">96 spearman {s}"] = stats.spearmanr(p[k & hi], t[k & hi])[0]
    return m


d = pd.read_csv(args.csv)
members = [c[len("pred_"):] for c in d.columns if c.startswith("pred_")]
t = d.truth.to_numpy()
fix = ~d.is_interpolated.astype(bool).to_numpy()
storm = (d.atcf_id + " " + d.storm_name.str.title()).to_numpy()
print(f"{args.csv}: {len(d)} frames, {int((t > 96).sum())} above 96 kt, {len(members)} members")

if args.member_csv_pattern:
    worst = 0.0
    for s in members:
        single = pd.read_csv(args.member_csv_pattern.format(stem=s))[["crop_path", "pred"]]
        j = d[["crop_path", f"pred_{s}"]].merge(single, on="crop_path", how="outer", validate="1:1", indicator=True)
        assert (j._merge == "both").all(), f"{s}: frame sets differ"
        worst = max(worst, float(np.abs(j[f"pred_{s}"] - j.pred).max()))
    print(f"cross-check: every member matches its single-model CSV on the same {len(d)} frames; "
          f"largest difference {worst:.2e} kt")
    assert worst < 1e-3, "members do not reproduce the single-model CSVs"

ens = metrics(t, d.pred.to_numpy(), fix, storm)
mem = pd.DataFrame([metrics(t, d[f"pred_{s}"].to_numpy(), fix, storm) for s in members], index=members)

rows = []
for k in ens:
    if k.startswith("n ") or k in ("n", ">96 n"):
        continue
    r = {"metric": k, "ensemble": ens[k], "member mean": mem[k].mean(), "member sd": mem[k].std(ddof=1),
         "member min": mem[k].min(), "member max": mem[k].max()}
    lower_better = "MAE" in k
    if "spearman" in k:
        r["members ensemble beats"] = int((mem[k] < ens[k]).sum())
    elif lower_better:
        r["members ensemble beats"] = int((mem[k] > ens[k]).sum())
    rows.append(r)
table = pd.DataFrame(rows)

if args.reference:
    ref = pd.read_csv(args.reference)
    ref = ref[(ref.arm == args.reference_arm) & (ref.set == args.reference_set)]
    print(f"reference: {args.reference}, arm {args.reference_arm}, set {args.reference_set}, {len(ref)} runs")
    # the reference keys per-storm metrics by title-case name only
    def ref_key(k):
        for pre in (">96 MAE ", ">96 bias ", ">96 spearman ", "MAE "):
            if k.startswith(pre) and k[len(pre):].startswith("AL"):
                return pre + k[len(pre):].split(" ", 1)[1]
        return k
    table["reference mean"] = [ref[ref_key(k)].mean() if ref_key(k) in ref else np.nan for k in table.metric]
    table["reference sd"] = [ref[ref_key(k)].std(ddof=1) if ref_key(k) in ref else np.nan for k in table.metric]
    gap = (table["reference mean"] - table["member mean"]).abs().max()
    print(f"members recomputed here vs reference summary: largest difference {gap:.2e}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.out, index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(f"\nband counts: " + ", ".join(f"{name} {ens[f'n {name}']}" for *_, name in BANDS))
print(table.to_string(index=False))
