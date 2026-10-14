"""Read-only analysis of the class-imbalance experiment. CPU only. Run from the repo root.

Everything is recomputed from the per-sample prediction CSVs evaluate.py wrote to results/imbalance/ and the
training logs in logs/imbalance/. Writes summary_per_run.csv, summary_by_arm_h3.csv, summary_by_arm_h4.csv and
arm_differences.csv; the numbers in results/imbalance/findings.md come from here and from imbalance_extra.py.
"""
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RES, LOG = REPO / "results" / "imbalance", REPO / "logs" / "imbalance"
ARMS = {"A_baseline_40ep": "A  9 storms, 40 ep", "B_expanded_29ep": "B 11 storms, 29 ep", "C_expanded_40ep": "C 11 storms, 40 ep"}
SHORT = {"A_baseline_40ep": "A", "B_expanded_29ep": "B", "C_expanded_40ep": "C"}
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
REF_SD = 0.59
pd.set_option("display.width", 280)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")


def metrics(d):
    t, p = d.truth.to_numpy(), d.pred.to_numpy()
    e = p - t
    fix = ~d.is_interpolated.astype(bool).to_numpy()
    storm = d.storm_name.str.title().to_numpy()
    m = {"n": len(d), "overall MAE": np.abs(e).mean(), "fixes-only MAE": np.abs(e)[fix].mean(), "overall bias": e.mean()}
    hi = t > 96
    m[">96 n"], m[">96 MAE"], m[">96 bias"] = hi.sum(), np.abs(e)[hi].mean(), e[hi].mean()
    m[">96 fixes MAE"] = np.abs(e)[hi & fix].mean()
    for s in np.unique(storm):
        k = storm == s
        m[f"MAE {s}"] = np.abs(e)[k].mean()
        if (k & hi).any():
            m[f">96 n {s}"], m[f">96 MAE {s}"], m[f">96 bias {s}"] = (k & hi).sum(), np.abs(e)[k & hi].mean(), e[k & hi].mean()
    lo_ = t < 64
    m["<64 MAE"], m["<64 bias"] = np.abs(e)[lo_].mean(), e[lo_].mean()
    for lo, hi_, name in BANDS:
        k = (t >= lo) & (t < hi_)
        m[f"n {name}"] = k.sum()
        m[f"bias {name}"] = e[k].mean() if k.any() else np.nan
        m[f"MAE {name}"] = np.abs(e)[k].mean() if k.any() else np.nan
    return m


rows, missing = [], []
for arm in ARMS:
    for seed in range(10):
        log = LOG / f"{arm}_seed{seed}.log"
        if not log.exists() or "END " not in log.read_text():
            missing.append((arm, seed))
            continue
        txt = log.read_text()
        ep = [(int(e), float(tr), float(va)) for e, tr, va in re.findall(r"epoch (\d+): train ([\d.]+) kt\s+val ([\d.]+) kt", txt)]
        sel = min(ep, key=lambda r: r[2])
        info = dict(arm=arm, seed=seed, order=int(re.search(r"order=(\d+)", txt).group(1)), epochs=len(ep),
                    train_storms=re.search(r"train storms: (\[.*\])", txt).group(1).count("AL"),
                    train_frames=int(re.search(r"train samples: (\d+)", txt).group(1)), val_frames=int(re.search(r"val samples: (\d+)", txt).group(1)),
                    sel_epoch=sel[0], val_mae=sel[2], train_mae_final=ep[-1][1], failed="TRAIN_FAILED" in txt)
        h3 = pd.read_csv(RES / f"{arm}_seed{seed}_holdout3.csv")
        h4 = pd.read_csv(RES / f"{arm}_seed{seed}_holdout4.csv")
        sub = h4[h4.atcf_id != "AL182021"].reset_index(drop=True)
        info["h3_h4_max_pred_diff"] = float(np.abs(sub.pred.to_numpy() - h3.pred.to_numpy()).max())
        for name, d in (("h3", h3), ("h4", h4)):
            rows.append({**info, "set": name, **metrics(d)})
runs = pd.DataFrame(rows)
runs.to_csv(RES / "summary_per_run.csv", index=False)
print(f"runs analysed: {runs[['arm', 'seed']].drop_duplicates().shape[0]} of 30; missing: {missing}")

print("\n== integrity ==")
chk = runs[runs.set == "h3"]
print(chk.groupby("arm").agg(runs=("seed", "count"), epochs=("epochs", lambda s: sorted(set(s))), train_storms=("train_storms", lambda s: sorted(set(s))),
                             train_frames=("train_frames", lambda s: sorted(set(s))), val_frames=("val_frames", lambda s: sorted(set(s))),
                             failed=("failed", "sum"), sel_epoch_mean=("sel_epoch", "mean"), sel_epoch_min=("sel_epoch", "min"), sel_epoch_max=("sel_epoch", "max"),
                             val_mae=("val_mae", "mean"), train_mae_final=("train_mae_final", "mean")).to_string())
print("held-out frames:", runs.groupby("set").n.agg(lambda s: sorted(set(s))).to_dict(), "| >96 kt frames:", runs.groupby("set")[">96 n"].agg(lambda s: sorted(set(s))).to_dict())
print(f"same checkpoint, three-storm rows of the four-storm CSV vs the three-storm CSV: max |pred difference| {runs.h3_h4_max_pred_diff.max():.2e} kt")
print("selected epochs per arm:", {SHORT[a]: sorted(chk[chk.arm == a].sel_epoch.tolist()) for a in ARMS})


def summarize(set_name, cols):
    d = runs[runs.set == set_name]
    out = []
    for c in cols:
        r = {"metric": c}
        for arm in ARMS:
            x = d[d.arm == arm][c]
            r[f"{SHORT[arm]} mean"], r[f"{SHORT[arm]} sd"], r[f"{SHORT[arm]} min"], r[f"{SHORT[arm]} max"] = x.mean(), x.std(), x.min(), x.max()
        out.append(r)
    return pd.DataFrame(out)


def perm_p(a, b, n=40000, rng=np.random.default_rng(0)):
    pool = np.concatenate([a, b])
    obs = b.mean() - a.mean()
    idx = np.array([rng.permutation(len(pool)) for _ in range(n)])
    d = pool[idx[:, len(a):]].mean(axis=1) - pool[idx[:, :len(a)]].mean(axis=1)
    return (np.abs(d) >= abs(obs) - 1e-12).mean()


def compare(set_name, cols):
    d = runs[runs.set == set_name]
    out = []
    for c in cols:
        for x, y in (("A_baseline_40ep", "B_expanded_29ep"), ("A_baseline_40ep", "C_expanded_40ep"), ("B_expanded_29ep", "C_expanded_40ep")):
            a, b = d[d.arm == x][c].to_numpy(), d[d.arm == y][c].to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            diff = b.mean() - a.mean()
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            psd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            out.append({"metric": c, "comparison": f"{SHORT[y]} - {SHORT[x]}", "difference": diff, "SE": se, "diff/SE": diff / se, "perm p": perm_p(a, b),
                        "pooled arm sd": psd, "diff / pooled sd": diff / psd, "exceeds 0.59": abs(diff) > REF_SD,
                        "runs of second arm beyond first arm's range": int(((b < a.min()) | (b > a.max())).sum())})
    return pd.DataFrame(out)


HEAD = ["overall MAE", "fixes-only MAE", "overall bias", ">96 MAE", ">96 bias", ">96 fixes MAE", "<64 MAE", "<64 bias"]
BIAS = [f"bias {b}" for _, _, b in BANDS]
BMAE = [f"MAE {b}" for _, _, b in BANDS]
all_cmp = []
for set_name, label in (("h3", "ORIGINAL THREE held-out storms (Ian, Nicole, Claudette; 393 frames)"), ("h4", "EXPANDED FOUR held-out storms (adds Sam; 688 frames)")):
    d = runs[runs.set == set_name]
    extra = [c for c in runs.columns if (c.startswith(">96 MAE ") or c.startswith(">96 bias ") or c.startswith("MAE ")) and c not in BMAE and d[c].notna().any()]
    print(f"\n\n################ {label} ################")
    print("band sizes:", {b: int(d[f'n {b}'].iloc[0]) for _, _, b in BANDS}, "| >96 kt by storm:", {c[6:]: int(d[c].iloc[0]) for c in d.columns if c.startswith('>96 n ') and d[c].notna().any()})
    print("\n-- >96 kt band (the experiment's target) --")
    hi_cols = [">96 MAE", ">96 bias", ">96 fixes MAE"] + [c for c in extra if c.startswith(">96")] + ["MAE Cat 3", "bias Cat 3", "MAE Cat 4", "bias Cat 4", "MAE Cat 5", "bias Cat 5"]
    print(summarize(set_name, hi_cols).to_string(index=False))
    print("\n-- overall --")
    print(summarize(set_name, ["overall MAE", "fixes-only MAE", "overall bias", "<64 MAE", "<64 bias"] + [c for c in extra if c.startswith("MAE ")]).to_string(index=False))
    print("\n-- per-category bias --")
    print(summarize(set_name, BIAS).to_string(index=False))
    print("\n-- per-category MAE --")
    print(summarize(set_name, BMAE).to_string(index=False))
    print("\n-- differences between arms --")
    cmp = compare(set_name, ["overall MAE", "fixes-only MAE", ">96 MAE", ">96 bias"] + [c for c in extra if c.startswith(">96")] + ["<64 MAE", "overall bias"] + BIAS + ["MAE Cat 3", "MAE Cat 4", "MAE Cat 5"])
    print(cmp.to_string(index=False))
    cmp.insert(0, "set", set_name)
    all_cmp.append(cmp)
    summarize(set_name, HEAD + BIAS + BMAE + extra).to_csv(RES / f"summary_by_arm_{set_name}.csv", index=False)
pd.concat(all_cmp).to_csv(RES / "arm_differences.csv", index=False)

print("\n\n== per-run values, three-storm and four-storm ==")
w = runs.pivot_table(index=["arm", "seed"], columns="set", values=["overall MAE", ">96 MAE", ">96 bias"]).round(2)
w.columns = [f"{m} {s}" for m, s in w.columns]
w = w.join(runs[runs.set == "h4"].set_index(["arm", "seed"])[["sel_epoch", "val_mae", ">96 MAE Sam", ">96 bias Sam", ">96 MAE Ian", ">96 bias Ian"]])
print(w.to_string())

print("\n== does run order line up with anything? ==")
h3 = runs[runs.set == "h3"]
print(f"held-out MAE vs position in session, all runs: spearman {h3.order.corr(h3['overall MAE'], method='spearman'):+.2f}; within arm: " +
      ", ".join(f"{SHORT[a]} {h3[h3.arm == a].order.corr(h3[h3.arm == a]['overall MAE'], method='spearman'):+.2f}" for a in ARMS))
