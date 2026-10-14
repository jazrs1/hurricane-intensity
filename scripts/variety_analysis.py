"""Analysis of the variety-against-volume experiment. Read-only, CPU only. Run from the repo root.

Written before launch (logs/decisions.md, 2026-10-14, "Variety-against-volume experiment"); its SHA-256 at launch is
in logs/variety/analysis_sha256_at_launch.txt. The three primary tests, the Bonferroni threshold, the exact
permutation test, the secondary and the robustness check are fixed below and must not be edited after results exist.
Everything else printed is exploratory and labelled that way.

Inputs: logs/variety/<arm>_seed<s>.log, results/variety/<arm>_seed<s>_h9.csv and _storm_fraction_h9.json,
results/variety/frame_lists/ (summary.csv, matching.csv). Writes results/variety/summary_per_run.csv,
summary_by_arm.csv, primary_tests.csv, robustness.csv, exploratory_differences.csv.
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
RES, LOG = REPO / "results" / "variety", REPO / "logs" / "variety"
ARMS = ["dense", "sparse"]  # comparisons are sparse - dense
SEEDS = range(10)
H6_IDS = ["AL092022", "AL172022", "AL032021", "AL022024", "AL132023", "AL182021"]
H3_IDS = ["AL092022", "AL172022", "AL032021"]
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
HIGH_STORMS = ["Ian", "Beryl", "Lee", "Sam", "Fiona", "Kirk", "Iota"]
EXPECTED = {"dense": dict(epochs=51, frame_storms=9, train_frames=1013, val_frames=940),
            "sparse": dict(epochs=51, frame_storms=19, train_frames=1013, val_frames=940)}

# ---- fixed before launch -----------------------------------------------------------------------------------
PRIMARY = ["spearman >96", ">96 bias", "overall MAE"]  # on H9 pooled, sparse - dense
THRESHOLD = 0.0167  # 0.05 / 3, Bonferroni over the three primary tests
SECONDARY = "same_storm_frac_6h"  # on the H9 frames (neither arm trains on them); one test, alpha 0.05; expected down
SECONDARY_ALPHA = 0.05
assert len(PRIMARY) == 3 and abs(THRESHOLD - 0.05 / 3) < 1e-4
# robustness (not primary): each arm's seeds 0-4 against the other arm's ten runs, same exact test and threshold
# ------------------------------------------------------------------------------------------------------------

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")


def exact_p(a, b):
    """Two-sided exact permutation p for the difference of means, enumerating every split of the pooled runs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, k = len(a) + len(b), len(b)
    combs = np.array(list(combinations(range(n), k)))
    split = np.zeros((len(combs), n), bool)
    split[np.arange(len(combs))[:, None], combs] = True
    pool = np.concatenate([a, b])
    mb = (split * pool).sum(1) / k
    ma = (pool.sum() - mb * k) / len(a)
    return float((np.abs(mb - ma) >= abs(b.mean() - a.mean()) - 1e-12).mean())


def metrics(d):
    t, p = d.truth.to_numpy(), d.pred.to_numpy()
    e = p - t
    fix = ~d.is_interpolated.astype(bool).to_numpy()
    storm = d.storm_name.str.title().to_numpy()
    hi, lo = t > 96, t < 64
    m = {"n": len(d), "overall MAE": np.abs(e).mean(), "fixes-only MAE": np.abs(e)[fix].mean(), "overall bias": e.mean(),
         ">96 n": hi.sum(), ">96 MAE": np.abs(e)[hi].mean(), ">96 bias": e[hi].mean(),
         "spearman >96": stats.spearmanr(p[hi], t[hi])[0], "pearson >96": stats.pearsonr(p[hi], t[hi])[0],
         "<64 MAE": np.abs(e)[lo].mean(), "<64 bias": e[lo].mean()}
    for a, b_, name in BANDS:
        k = (t >= a) & (t < b_)
        m[f"bias {name}"] = e[k].mean() if k.any() else np.nan
        m[f"MAE {name}"] = np.abs(e)[k].mean() if k.any() else np.nan
    rho = []
    for s in np.unique(storm):
        k = storm == s
        m[f"MAE {s}"] = np.abs(e)[k].mean()
        if (k & hi).sum() >= 10:
            m[f">96 MAE {s}"], m[f">96 bias {s}"] = np.abs(e)[k & hi].mean(), e[k & hi].mean()
            m[f">96 spearman {s}"] = stats.spearmanr(p[k & hi], t[k & hi])[0]
            rho.append(m[f">96 spearman {s}"])
        k45, k3 = k & (t >= 113), k & (t >= 96) & (t < 113)
        if k45.any() and k3.any():
            m[f"{s} Cat 3 bias"], m[f"{s} Cat 4/5 bias"] = e[k3].mean(), e[k45].mean()
            m[f"{s} pred gap Cat4/5-Cat3"] = p[k45].mean() - p[k3].mean()
    m["mean per-storm spearman >96"] = np.mean(rho) if rho else np.nan
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
        fl = re.search(r"frame list: (\S+): (\d+) frames from (\d+) storms", txt)
        info = dict(arm=arm, seed=seed, order=int(re.search(r"order=(\d+)", txt).group(1)), epochs=len(ep),
                    frame_list=Path(fl.group(1)).name, frame_storms=int(fl.group(3)),
                    val_logged=re.search(r"val=(\S+)", txt).group(1), loss_logged=re.search(r"loss: (\w+)", txt).group(1),
                    mask_logged=re.search(r"circular_mask: (\w+)", txt).group(1), rotate_logged=re.search(r"rotate: (\w+)", txt).group(1),
                    val_storms=re.search(r"val storms:\s+(\[.*\])", txt).group(1).count("AL"),
                    train_frames=int(re.search(r"train samples: (\d+)", txt).group(1)), val_frames=int(re.search(r"val samples: (\d+)", txt).group(1)),
                    sel_epoch=sel[0], val_mae=sel[2], train_mae_final=ep[-1][1], failed="TRAIN_FAILED" in txt)
        info["frame_list_matches_seed"] = info["frame_list"] == f"{arm}_seed{seed}.txt"
        info["position_in_round"] = (info["order"] - 1) % 2 + 1
        sf = json.loads((RES / f"{arm}_seed{seed}_storm_fraction_h9.json").read_text())
        info.update({"same_storm_frac_6h": sf["same_storm_frac_6h"], "same_storm_frac_24h": sf["same_storm_frac_24h"],
                     **{f"same_storm_6h {b}": v for b, v in sf["by_band_6h"].items()}, "sf_frame_set": sf.get("frame_set"), "sf_frames": sf["n_frames"]})
        h9 = pd.read_csv(RES / f"{arm}_seed{seed}_h9.csv")
        for name, d in (("h9", h9), ("h6", h9[h9.atcf_id.isin(H6_IDS)]), ("h3", h9[h9.atcf_id.isin(H3_IDS)])):
            rows.append({**info, "set": name, **metrics(d)})
runs = pd.DataFrame(rows)
RES.mkdir(parents=True, exist_ok=True)
runs.to_csv(RES / "summary_per_run.csv", index=False)
d9 = runs[runs.set == "h9"]
print(f"runs analysed: {len(d9)} of 20; problems: {problems or 'none'}")

print("\n== integrity ==")
print(d9.groupby("arm").agg(runs=("seed", "count"), list_matches_seed=("frame_list_matches_seed", "all"), val_list=("val_logged", lambda s: sorted(set(s))),
                            loss=("loss_logged", lambda s: sorted(set(s))), mask=("mask_logged", lambda s: sorted(set(s))), rotate=("rotate_logged", lambda s: sorted(set(s))),
                            epochs=("epochs", lambda s: sorted(set(s))), frame_storms=("frame_storms", lambda s: sorted(set(s))),
                            train_frames=("train_frames", lambda s: sorted(set(s))), val_frames=("val_frames", lambda s: sorted(set(s))),
                            sf_frame_set=("sf_frame_set", lambda s: sorted(set(s))), sf_frames=("sf_frames", lambda s: sorted(set(s))),
                            failed=("failed", "sum")).reindex(ARMS).to_string())
for arm, exp in EXPECTED.items():
    got = d9[d9.arm == arm]
    bad = {k: sorted(set(got[k])) for k, v in exp.items() if set(got[k]) != {v}}
    print(f"  {arm}: matches the recorded design" if not bad else f"  {arm}: DIFFERS from the recorded design: {bad} (expected {exp})")
print("H9 frames:", sorted(set(d9.n)), "| >96 kt frames:", sorted(set(d9[">96 n"])))
print("selected epochs:", {a: sorted(d9[d9.arm == a].sel_epoch.tolist()) for a in ARMS})
for a in ARMS:
    late = int((d9[d9.arm == a].sel_epoch >= 50).sum())
    print(f"  {a} runs selecting epoch 50 or 51: {late} of 10" + (" -> FLAG: possibly under-trained (pre-set rule; not rerun)" if late > 5 else ""))

fl = pd.read_csv(RES / "frame_lists" / "summary.csv")
print("\n== training frames: per-band count and mean intensity, mean over the ten draws (matching on 5 kt bins) ==")
fb = fl[fl.band != "all"].groupby(["band", "arm"]).agg(n=("n", "mean"), mean_kt=("mean_kt", "mean"), storms=("storms", "mean")).unstack("arm")
fb[("mean_kt", "sparse - dense")] = fb[("mean_kt", "sparse")] - fb[("mean_kt", "dense")]
print(fb.reindex([b[2] for b in BANDS]).to_string(float_format=lambda v: f"{v:.2f}"))
print(fl[fl.band == "all"].groupby("arm")[["n", "mean_kt", "storms", "ge96", "ge113", "ge137", "largest_storm_share"]].mean().to_string())
mt = pd.read_csv(RES / "frame_lists" / "matching.csv")
print(f"bins matched exactly: {int((~mt.orphan & (mt.matched_in_bin > 0)).sum())} of {len(mt)}; orphan bins: {mt[mt.orphan][['bin', 'band', 'dense_pool', 'sparse_pool', 'band_fallback_matched']].to_dict('records')}")


def compare(df, metric, x="dense", y="sparse"):
    a, b = df[df.arm == x][metric].dropna().to_numpy(), df[df.arm == y][metric].dropna().to_numpy()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"metric": metric, "comparison": f"{y} - {x}", x: a.mean(), f"{x} sd": a.std(ddof=1), y: b.mean(), f"{y} sd": b.std(ddof=1),
            "difference": b.mean() - a.mean(), "SE": se, "exact p": exact_p(a, b), "Welch p": stats.ttest_ind(b, a, equal_var=False).pvalue,
            f"runs of {y} beyond {x}'s range": int(((b < a.min()) | (b > a.max())).sum())}


def summarize(df, cols):
    out = []
    for c in cols:
        r = {"metric": c}
        for arm in ARMS:
            x = df[df.arm == arm][c]
            r[f"{arm} mean"], r[f"{arm} sd"], r[f"{arm} min"], r[f"{arm} max"] = x.mean(), x.std(), x.min(), x.max()
        out.append(r)
    return pd.DataFrame(out)


print(f"\n\n################ PRIMARY TESTS (fixed before launch; H9 pooled; sparse - dense; two-sided exact permutation; "
      f"Bonferroni threshold {THRESHOLD}) ################")
prim = pd.DataFrame([compare(d9, m) for m in PRIMARY])
prim["survives correction"] = prim["exact p"] < THRESHOLD
print(prim.to_string(index=False, formatters={"exact p": "{:.5f}".format, "Welch p": "{:.5f}".format}))
surv = prim[prim["survives correction"]]
print("\nSurviving correction: " + ("none." if surv.empty else "; ".join(f"{r.metric} {r.difference:+.3f} (exact p {r['exact p']:.5f})" for _, r in surv.iterrows())))
prim.to_csv(RES / "primary_tests.csv", index=False)

print(f"\n################ PRE-REGISTERED SECONDARY: same-storm neighbour fraction on the H9 frames (k=20, +-6 h excluded), "
      f"sparse - dense; exact permutation, alpha {SECONDARY_ALPHA} ################")
sec = compare(d9, SECONDARY)
sec["significant"] = sec["exact p"] < SECONDARY_ALPHA
print(pd.DataFrame([sec]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print(f"\n################ ROBUSTNESS (fixed before launch; not primary): each arm's seeds 0-4 against the other arm's ten runs ################")
rob = []
for m in PRIMARY:
    for early, full in (("dense", "sparse"), ("sparse", "dense")):
        a = d9[(d9.arm == early) & (d9.seed <= 4)][m].to_numpy()
        b = d9[d9.arm == full][m].to_numpy()
        diff = (b.mean() - a.mean()) if early == "dense" else (a.mean() - b.mean())  # always sparse - dense
        p = exact_p(a, b)
        rob.append({"metric": m, "check": f"{early} seeds 0-4 vs {full} all ten", "sparse - dense": diff, "exact p": p, "passes 0.0167": p < THRESHOLD})
rob = pd.DataFrame(rob)
rob.to_csv(RES / "robustness.csv", index=False)
print(rob.to_string(index=False, formatters={"exact p": "{:.5f}".format, "sparse - dense": "{:+.3f}".format}))

print("\n\n################ EVERYTHING BELOW IS EXPLORATORY (uncorrected) ################")
BIAS = [f"bias {b}" for *_, b in BANDS]
HEAD = ["overall MAE", "fixes-only MAE", "overall bias", ">96 MAE", ">96 bias", "spearman >96", "pearson >96", "mean per-storm spearman >96", "<64 MAE", "<64 bias"]
for set_name, label in (("h9", "H9 (1752 frames)"), ("h6", "H6 rows of the H9 CSVs"), ("h3", "H3 rows of the H9 CSVs")):
    d = runs[runs.set == set_name]
    print(f"\n==== {label}: mean / sd / min / max over ten runs ====")
    print(summarize(d, HEAD).to_string(index=False))
    print("-- per-category bias --")
    print(summarize(d, BIAS).to_string(index=False))
STORMCOLS = [f">96 {k} {s}" for s in HIGH_STORMS for k in ("spearman", "MAE", "bias")]
summarize(d9, HEAD + BIAS + [f"MAE {b}" for *_, b in BANDS] + STORMCOLS + ["same_storm_frac_6h", "same_storm_frac_24h"]).to_csv(RES / "summary_by_arm.csv", index=False)

print("\n==== >96 kt band per storm (H9): Spearman, MAE, bias ====")
print(summarize(d9, STORMCOLS).to_string(index=False))

print("\n==== Ian as the check: does a high-end change separate Cat 3 from Cat 4/5, or move the high end together? ====")
for s in HIGH_STORMS:
    if f"{s} pred gap Cat4/5-Cat3" not in d9.columns or d9[f"{s} pred gap Cat4/5-Cat3"].isna().all():
        continue
    r = {c: compare(d9, f"{s} {c}") for c in ("Cat 3 bias", "Cat 4/5 bias", "pred gap Cat4/5-Cat3")}
    print(f"  {s}: Cat 3 bias shift {r['Cat 3 bias']['difference']:+.2f} (p {r['Cat 3 bias']['exact p']:.3f}), "
          f"Cat 4/5 bias shift {r['Cat 4/5 bias']['difference']:+.2f} (p {r['Cat 4/5 bias']['exact p']:.3f}), "
          f"predicted gap {r['pred gap Cat4/5-Cat3']['dense']:+.1f} -> {r['pred gap Cat4/5-Cat3']['sparse']:+.1f} (p {r['pred gap Cat4/5-Cat3']['exact p']:.3f})")

print("\n==== same-storm neighbour fraction on H9, by band and at +-24 h ====")
print(summarize(d9, ["same_storm_frac_6h", "same_storm_frac_24h"] + [f"same_storm_6h {b}" for *_, b in BANDS]).to_string(index=False))

print("\n==== all sparse - dense differences ====")
cols = [c for c in HEAD + BIAS + STORMCOLS if c not in PRIMARY]
exp = [dict(set="h9", **compare(d9, c)) for c in cols if d9[c].notna().any()]
for s in ("h6", "h3"):
    d = runs[runs.set == s]
    exp += [dict(set=s, **compare(d, c)) for c in HEAD + BIAS if d[c].notna().any()]
exp += [dict(set="h9", **compare(d9, c)) for c in ["same_storm_frac_24h"] + [f"same_storm_6h {b}" for *_, b in BANDS] if d9[c].notna().any()]
exp = pd.DataFrame(exp)
exp.to_csv(RES / "exploratory_differences.csv", index=False)
print(exp[["set", "metric", "dense", "sparse", "difference", "SE", "exact p"]].to_string(index=False, formatters={"exact p": "{:.4f}".format}))

print("\n==== run order (fixed in advance) ====")
d = d9.copy()
d["resid"] = d["overall MAE"] - d.groupby("arm")["overall MAE"].transform("mean")
rho, p = stats.spearmanr(d.order, d.resid)
a1, a2 = d[d.position_in_round == 1].resid.to_numpy(), d[d.position_in_round == 2].resid.to_numpy()
print("H9 overall MAE vs seed, Spearman within arm: " + ", ".join(f"{a} {d[d.arm == a].seed.corr(d[d.arm == a]['overall MAE'], method='spearman'):+.2f}" for a in ARMS)
      + f" | pooled (arm means removed) vs run order {rho:+.2f} (p {p:.2f}) | first-in-round minus second {a1.mean() - a2.mean():+.2f} kt (exact p {exact_p(a2, a1):.3f})")

print("\n==== per-run values ====")
print(d9.set_index(["arm", "seed"])[["order", "sel_epoch", "val_mae", "overall MAE", ">96 MAE", ">96 bias", "spearman >96", "same_storm_frac_6h"]].to_string())
