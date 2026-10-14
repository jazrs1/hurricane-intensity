"""Analysis of the rotation-augmentation experiment. Read-only, CPU only. Run from the repo root.

Written before launch (logs/decisions.md, 2026-10-14). The three primary tests, the Bonferroni threshold, the
exact permutation test and the secondary are fixed below and must not be edited after results exist.
Everything else printed is exploratory and labelled that way.

Inputs: logs/rotation/<arm>_seed<s>.log, results/rotation/<arm>_seed<s>_h6.csv and _storm_fraction.json;
for the masked-vs-unmasked comparison, results/loss/l1_seed<s>_h6.csv.
Writes results/rotation/summary_per_run.csv, summary_by_arm.csv, primary_tests.csv, exploratory_differences.csv.
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RES, LOG = REPO / "results" / "rotation", REPO / "logs" / "rotation"
ARMS = ["control", "rotation"]
SEEDS = range(10)
H3_IDS = ["AL092022", "AL172022", "AL032021"]
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
HIGH_STORMS = ["Ian", "Beryl", "Lee", "Sam"]

# ---- fixed before launch -----------------------------------------------------------------------------------
PRIMARY = [(">96 bias", "rotation"), (">96 MAE", "rotation"), ("overall MAE", "rotation")]  # on H6 pooled, vs control
THRESHOLD = 0.0167  # 0.05 / 3, Bonferroni over the three primary tests
SECONDARY = "same_storm_frac_6h"  # one test, alpha 0.05, outside the Bonferroni family; expected to drop
SECONDARY_ALPHA = 0.05
assert len(PRIMARY) == 3 and abs(THRESHOLD - 0.05 / 3) < 1e-4
# ------------------------------------------------------------------------------------------------------------

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")
_SPLITS = None


def exact_p(a, b):
    """Two-sided exact permutation p for the difference of means, enumerating every split of the pooled runs."""
    global _SPLITS
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a) + len(b)
    if _SPLITS is None or _SPLITS.shape[1] != n:
        combs = np.array(list(combinations(range(n), len(b))))
        _SPLITS = np.zeros((len(combs), n), bool)
        _SPLITS[np.arange(len(combs))[:, None], combs] = True
    pool = np.concatenate([a, b])
    mb = (_SPLITS * pool).sum(1) / len(b)
    ma = (pool.sum() - mb * len(b)) / len(a)
    obs = b.mean() - a.mean()
    return float((np.abs(mb - ma) >= abs(obs) - 1e-12).mean())


def metrics(d):
    t, p = d.truth.to_numpy(), d.pred.to_numpy()
    e = p - t
    fix = ~d.is_interpolated.astype(bool).to_numpy()
    storm = d.storm_name.str.title().to_numpy()
    hi, lo = t > 96, t < 64
    m = {"n": len(d), "overall MAE": np.abs(e).mean(), "fixes-only MAE": np.abs(e)[fix].mean(), "overall bias": e.mean(),
         ">96 n": hi.sum(), ">96 MAE": np.abs(e)[hi].mean(), ">96 bias": e[hi].mean(), "<64 MAE": np.abs(e)[lo].mean(), "<64 bias": e[lo].mean()}
    for a, b_, name in BANDS:
        k = (t >= a) & (t < b_)
        m[f"bias {name}"], m[f"MAE {name}"] = e[k].mean(), np.abs(e)[k].mean()
    for s in np.unique(storm):
        k = storm == s
        m[f"MAE {s}"] = np.abs(e)[k].mean()
        if (k & hi).any():
            m[f">96 MAE {s}"], m[f">96 bias {s}"] = np.abs(e)[k & hi].mean(), e[k & hi].mean()
        k45, k3 = k & (t >= 113), k & (t >= 96) & (t < 113)
        if k45.any() and k3.any():
            m[f"{s} Cat 3 bias"], m[f"{s} Cat 4/5 bias"] = e[k3].mean(), e[k45].mean()
            m[f"{s} pred gap Cat4/5-Cat3"] = p[k45].mean() - p[k3].mean()
    return m


rows, problems = [], []
for arm in ARMS:
    for seed in SEEDS:
        log = LOG / f"{arm}_seed{seed}.log"
        if not log.exists() or "END " not in log.read_text():
            problems.append(f"{arm} seed {seed}: missing or unfinished")
            continue
        txt = log.read_text()
        ep = [(int(e), float(tr), float(va)) for e, tr, va in re.findall(r"epoch (\d+): train ([\d.]+) kt\s+val ([\d.]+) kt", txt)]
        sel = min(ep, key=lambda r: r[2])
        info = dict(arm=arm, seed=seed, order=int(re.search(r"order=(\d+)", txt).group(1)), epochs=len(ep),
                    mask_logged=re.search(r"circular_mask: (\w+)", txt).group(1), rotate_logged=re.search(r"rotate: (\w+)", txt).group(1),
                    train_storms=re.search(r"train storms: (\[.*\])", txt).group(1).count("AL"),
                    train_frames=int(re.search(r"train samples: (\d+)", txt).group(1)), val_frames=int(re.search(r"val samples: (\d+)", txt).group(1)),
                    sel_epoch=sel[0], val_mae=sel[2], train_mae_final=ep[-1][1], failed="TRAIN_FAILED" in txt)
        info["position_in_round"] = (info["order"] - 1) % 2 + 1
        sf = json.loads((RES / f"{arm}_seed{seed}_storm_fraction.json").read_text())
        info.update({"same_storm_frac_6h": sf["same_storm_frac_6h"], "same_storm_frac_24h": sf["same_storm_frac_24h"],
                     **{f"same_storm_6h {b}": v for b, v in sf["by_band_6h"].items()}, "sf_masked": sf["circular_mask"]})
        h6 = pd.read_csv(RES / f"{arm}_seed{seed}_h6.csv")
        rows.append({**info, "set": "h6", **metrics(h6)})
        rows.append({**info, "set": "h3", **metrics(h6[h6.atcf_id.isin(H3_IDS)])})
runs = pd.DataFrame(rows)
RES.mkdir(parents=True, exist_ok=True)
runs.to_csv(RES / "summary_per_run.csv", index=False)
d6 = runs[runs.set == "h6"]
print(f"runs analysed: {len(d6)} of 20; problems: {problems or 'none'}")

print("\n== integrity ==")
print(d6.groupby("arm").agg(runs=("seed", "count"), mask=("mask_logged", lambda s: sorted(set(s))), rotate=("rotate_logged", lambda s: sorted(set(s))),
                            sf_masked=("sf_masked", lambda s: sorted(set(s))), epochs=("epochs", lambda s: sorted(set(s))),
                            train_storms=("train_storms", lambda s: sorted(set(s))), train_frames=("train_frames", lambda s: sorted(set(s))),
                            val_frames=("val_frames", lambda s: sorted(set(s))), failed=("failed", "sum"), sel_epoch_mean=("sel_epoch", "mean"),
                            val_mae=("val_mae", "mean"), train_mae_final=("train_mae_final", "mean")).reindex(ARMS).to_string())
print("H6 frames:", sorted(set(d6.n)), "| >96 kt frames:", sorted(set(d6[">96 n"])))
print("selected epochs:", {a: sorted(d6[d6.arm == a].sel_epoch.tolist()) for a in ARMS},
      "| rotation runs selecting epoch >= 38:", int((d6[d6.arm == "rotation"].sel_epoch >= 38).sum()), "of 10")


def compare(df, metric, x, y, label_x=None, label_y=None):
    a, b = df[df.arm == x][metric].dropna().to_numpy(), df[df.arm == y][metric].dropna().to_numpy()
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"metric": metric, "comparison": f"{label_y or y} - {label_x or x}", f"first mean": a.mean(), "second mean": b.mean(), "difference": diff,
            "SE": se, "diff/SE": diff / se, "exact p": exact_p(a, b), "Welch p": stats.ttest_ind(b, a, equal_var=False).pvalue,
            "runs of second beyond first's range": int(((b < a.min()) | (b > a.max())).sum())}


def summarize(df, cols, arms=ARMS):
    out = []
    for c in cols:
        r = {"metric": c}
        for arm in arms:
            x = df[df.arm == arm][c]
            r[f"{arm} mean"], r[f"{arm} sd"], r[f"{arm} min"], r[f"{arm} max"] = x.mean(), x.std(), x.min(), x.max()
        out.append(r)
    return pd.DataFrame(out)


print(f"\n\n################ PRIMARY TESTS (fixed before launch; H6 pooled; rotation - control; two-sided exact permutation; "
      f"Bonferroni threshold {THRESHOLD}) ################")
prim = pd.DataFrame([compare(d6, m, "control", y) for m, y in PRIMARY])
prim["survives correction"] = prim["exact p"] < THRESHOLD
print(prim.to_string(index=False, formatters={"exact p": "{:.5f}".format, "Welch p": "{:.5f}".format}))
surv = prim[prim["survives correction"]]
print("\nSurviving correction: " + ("none." if surv.empty else "; ".join(f"{r.metric} {r.difference:+.2f} kt (exact p {r['exact p']:.5f})" for _, r in surv.iterrows())))
prim.to_csv(RES / "primary_tests.csv", index=False)

print(f"\n################ PRE-REGISTERED SECONDARY: same-storm neighbour fraction (k=20, +-6 h excluded), rotation - control; "
      f"exact permutation, alpha {SECONDARY_ALPHA} ################")
sec = compare(d6, SECONDARY, "control", "rotation")
sec["significant"] = sec["exact p"] < SECONDARY_ALPHA
print(pd.DataFrame([sec]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("chance levels (from storm_clustering.py, the same for every run): naive 0.147, intensity-matched 0.204")

print("\n\n################ EVERYTHING BELOW IS EXPLORATORY (uncorrected) ################")
BIAS = [f"bias {b}" for *_, b in BANDS]
for set_name, label in (("h6", "H6 (1197 frames)"), ("h3", "H3 rows of the H6 CSVs (393 frames)")):
    d = runs[runs.set == set_name]
    print(f"\n==== {label}: mean / sd / min / max over ten runs ====")
    print(summarize(d, ["overall MAE", "fixes-only MAE", "overall bias", ">96 MAE", ">96 bias", "<64 MAE", "<64 bias"]).to_string(index=False))
    print("-- per-category bias --")
    print(summarize(d, BIAS).to_string(index=False))
summarize(d6, ["overall MAE", "fixes-only MAE", "overall bias", ">96 MAE", ">96 bias", "<64 MAE", "<64 bias"] + BIAS + [f"MAE {b}" for *_, b in BANDS]
          + [f">96 {k} {s}" for s in HIGH_STORMS for k in ("MAE", "bias")] + ["same_storm_frac_6h", "same_storm_frac_24h"]).to_csv(RES / "summary_by_arm.csv", index=False)

print("\n==== >96 kt band per storm (H6) ====")
print(summarize(d6, [f">96 {k} {s}" for s in HIGH_STORMS for k in ("MAE", "bias")]).to_string(index=False))

print("\n==== Ian as the check: does rotation separate Cat 3 from Cat 4/5, or move the high end together? ====")
for s in HIGH_STORMS:
    r = {c: compare(d6, f"{s} {c}", "control", "rotation") for c in ("Cat 3 bias", "Cat 4/5 bias", "pred gap Cat4/5-Cat3")}
    print(f"  {s}: Cat 3 bias shift {r['Cat 3 bias']['difference']:+.2f} (p {r['Cat 3 bias']['exact p']:.3f}), "
          f"Cat 4/5 bias shift {r['Cat 4/5 bias']['difference']:+.2f} (p {r['Cat 4/5 bias']['exact p']:.3f}), "
          f"predicted gap {r['pred gap Cat4/5-Cat3']['first mean']:+.1f} -> {r['pred gap Cat4/5-Cat3']['second mean']:+.1f} (p {r['pred gap Cat4/5-Cat3']['exact p']:.3f})")

print("\n==== same-storm neighbour fraction, by band and at +-24 h ====")
print(summarize(d6, ["same_storm_frac_6h", "same_storm_frac_24h"] + [f"same_storm_6h {b}" for *_, b in BANDS]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n==== all rotation - control differences ====")
cols = ["overall MAE", "fixes-only MAE", "overall bias", ">96 MAE", ">96 bias", "<64 MAE", "<64 bias"] + BIAS + [f">96 {k} {s}" for s in HIGH_STORMS for k in ("MAE", "bias")] + ["same_storm_frac_24h"] + [f"same_storm_6h {b}" for *_, b in BANDS]
exp = pd.DataFrame([dict(set=s, **compare(runs[runs.set == s], c, "control", "rotation")) for s in ("h6", "h3") for c in cols if not (s == "h3" and (c.startswith("same_storm") or any(c.endswith(x) for x in ("Beryl", "Lee", "Sam"))))])
exp = exp[~((exp.set == "h6") & exp.metric.isin([m for m, _ in PRIMARY] + [SECONDARY]))]

print("\n==== pre-registered exploratory: masked control vs the unmasked L1 arm of the loss experiment (H6; different sessions) ====")
l1 = []
for seed in SEEDS:
    f = REPO / "results" / "loss" / f"l1_seed{seed}_h6.csv"
    if f.exists():
        l1.append({"arm": "unmasked L1 (loss exp.)", "seed": seed, **metrics(pd.read_csv(f))})
both = pd.concat([d6[d6.arm == "control"].assign(arm="masked control"), pd.DataFrame(l1)], ignore_index=True)
mm = pd.DataFrame([compare(both, c, "unmasked L1 (loss exp.)", "masked control") for c in ["overall MAE", ">96 MAE", ">96 bias", "<64 MAE", "<64 bias"]])
print(mm.to_string(index=False, formatters={"exact p": "{:.4f}".format, "Welch p": "{:.4f}".format}))
exp = pd.concat([exp, mm.assign(set="h6, across experiments")])
exp.to_csv(RES / "exploratory_differences.csv", index=False)
print("\n(all differences)")
print(exp[["set", "metric", "comparison", "first mean", "second mean", "difference", "SE", "exact p"]].to_string(index=False, formatters={"exact p": "{:.3f}".format}))

print("\n==== run order (fixed in advance) ====")
d = d6.copy()
d["resid"] = d["overall MAE"] - d.groupby("arm")["overall MAE"].transform("mean")
rho, p = stats.spearmanr(d.order, d.resid)
print(f"H6 overall MAE vs seed, Spearman within arm: " + ", ".join(f"{a} {d[d.arm == a].seed.corr(d[d.arm == a]['overall MAE'], method='spearman'):+.2f}" for a in ARMS)
      + f" | pooled (arm means removed) vs run order {rho:+.2f} (p {p:.2f}) | mean residual 1st/2nd in round: "
      + ", ".join(f"{k}: {v:+.2f}" for k, v in d.groupby("position_in_round").resid.mean().items()))

print("\n==== per-run values ====")
print(d6.set_index(["arm", "seed"])[["order", "sel_epoch", "val_mae", "overall MAE", ">96 MAE", ">96 bias", "<64 MAE", "same_storm_frac_6h", "Ian Cat 3 bias", "Ian Cat 4/5 bias"]].to_string())
