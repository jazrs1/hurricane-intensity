"""Analysis of the loss-function experiment (L1 / Huber / MSE). Read-only, CPU only. Run from the repo root.

Written and committed to before launch (logs/decisions.md, 2026-10-14). The six primary tests and the
Bonferroni threshold are fixed below and must not be edited after results exist. Everything else this script
prints is exploratory and is labelled that way.

Inputs: logs/loss/<loss>_seed<s>.log, results/loss/<loss>_seed<s>_h3.csv and _h6.csv.
Writes results/loss/summary_per_run.csv, summary_by_arm_h3.csv, summary_by_arm_h6.csv, primary_tests.csv,
exploratory_differences.csv.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RES, LOG = REPO / "results" / "loss", REPO / "logs" / "loss"
ARMS = ["l1", "huber", "mse"]
SEEDS = range(10)
H3_IDS = ["AL092022", "AL172022", "AL032021"]
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
HIGH_STORMS = ["Ian", "Beryl", "Lee", "Sam"]

# ---- primary tests: fixed before launch -------------------------------------------------------------------
PRIMARY = [  # (held-out set, metric, arm compared against l1)
    ("h6", ">96 bias", "huber"),
    ("h6", ">96 MAE", "huber"),
    ("h6", "overall MAE", "huber"),
    ("h6", ">96 bias", "mse"),
    ("h6", ">96 MAE", "mse"),
    ("h6", "overall MAE", "mse"),
]
ALPHA = 0.05
THRESHOLD = 0.0083  # 0.05 / 6, Bonferroni over the six primary tests
N_PERM = 40000  # two-sided permutation test on the difference of arm means
assert len(PRIMARY) == 6 and abs(THRESHOLD - ALPHA / len(PRIMARY)) < 1e-4
# ------------------------------------------------------------------------------------------------------------

pd.set_option("display.width", 280)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")


def metrics(d):
    t, p = d.truth.to_numpy(), d.pred.to_numpy()
    e = p - t
    fix = ~d.is_interpolated.astype(bool).to_numpy()
    storm = d.storm_name.str.title().to_numpy()
    hi = t > 96
    m = {"n": len(d), "overall MAE": np.abs(e).mean(), "fixes-only MAE": np.abs(e)[fix].mean(), "overall bias": e.mean(),
         ">96 n": hi.sum(), ">96 MAE": np.abs(e)[hi].mean(), ">96 bias": e[hi].mean(), ">96 fixes MAE": np.abs(e)[hi & fix].mean()}
    for lo, up, b in BANDS:
        k = (t >= lo) & (t < up)
        m[f"n {b}"], m[f"bias {b}"], m[f"MAE {b}"] = k.sum(), (e[k].mean() if k.any() else np.nan), (np.abs(e)[k].mean() if k.any() else np.nan)
    for s in np.unique(storm):
        k = storm == s
        m[f"MAE {s}"], m[f"bias {s}"] = np.abs(e)[k].mean(), e[k].mean()
        if (k & hi).any():
            m[f">96 n {s}"], m[f">96 MAE {s}"], m[f">96 bias {s}"] = (k & hi).sum(), np.abs(e)[k & hi].mean(), e[k & hi].mean()
        for lo, up, b in BANDS[4:]:
            kb = k & (t >= lo) & (t < up)
            if kb.any():
                m[f"{s} {b} bias"], m[f"{s} {b} MAE"], m[f"{s} {b} mean pred"] = e[kb].mean(), np.abs(e)[kb].mean(), p[kb].mean()
        k45 = k & (t >= 113)
        k3 = k & (t >= 96) & (t < 113)
        if k45.any() and k3.any():
            m[f"{s} Cat 4/5 bias"] = e[k45].mean()
            # separation: how far apart the model puts this storm's Cat 4/5 and Cat 3 frames, against the best track
            m[f"{s} pred gap Cat4/5-Cat3"] = p[k45].mean() - p[k3].mean()
            m[f"{s} true gap Cat4/5-Cat3"] = t[k45].mean() - t[k3].mean()
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
        sel = min(ep, key=lambda r: r[2])  # train.py saves on strict improvement, so the first minimum
        info = dict(arm=arm, seed=seed, order=int(re.search(r"order=(\d+)", txt).group(1)), epochs=len(ep),
                    loss_logged=re.search(r"loss: (\w+)", txt).group(1), train_storms=re.search(r"train storms: (\[.*\])", txt).group(1).count("AL"),
                    train_frames=int(re.search(r"train samples: (\d+)", txt).group(1)), val_frames=int(re.search(r"val samples: (\d+)", txt).group(1)),
                    sel_epoch=sel[0], val_mae=sel[2], train_mae_final=ep[-1][1], nan_in_log=("nan" in txt.lower()), failed="TRAIN_FAILED" in txt)
        info["round"], info["position_in_round"] = seed, (info["order"] - 1) % 3 + 1
        h3, h6 = pd.read_csv(RES / f"{arm}_seed{seed}_h3.csv"), pd.read_csv(RES / f"{arm}_seed{seed}_h6.csv")
        sub = h6[h6.atcf_id.isin(H3_IDS)].reset_index(drop=True)
        info["h3_rows_match_h6"] = len(sub) == len(h3) and (sub.sample_time.to_numpy() == h3.sample_time.to_numpy()).all()
        info["h3_h6_max_pred_diff"] = float(np.abs(sub.pred.to_numpy() - h3.pred.to_numpy()).max()) if info["h3_rows_match_h6"] else np.nan
        for name, d in (("h3", h3), ("h6", h6)):
            rows.append({**info, "set": name, **metrics(d)})
runs = pd.DataFrame(rows)
RES.mkdir(parents=True, exist_ok=True)
runs.to_csv(RES / "summary_per_run.csv", index=False)
n_runs = runs[["arm", "seed"]].drop_duplicates().shape[0]
print(f"runs analysed: {n_runs} of 30; problems: {problems or 'none'}")

print("\n== integrity ==")
chk = runs[runs.set == "h6"]
print(chk.groupby("arm").agg(runs=("seed", "count"), loss_logged=("loss_logged", lambda s: sorted(set(s))), epochs=("epochs", lambda s: sorted(set(s))),
                             train_storms=("train_storms", lambda s: sorted(set(s))), train_frames=("train_frames", lambda s: sorted(set(s))),
                             val_frames=("val_frames", lambda s: sorted(set(s))), failed=("failed", "sum"), nan_in_log=("nan_in_log", "sum"),
                             sel_epoch_mean=("sel_epoch", "mean"), sel_epoch_min=("sel_epoch", "min"), sel_epoch_max=("sel_epoch", "max"),
                             val_mae=("val_mae", "mean"), train_mae_final=("train_mae_final", "mean")).reindex(ARMS).to_string())
print("held-out frames:", runs.groupby("set").n.agg(lambda s: sorted(set(s))).to_dict(), "| >96 kt frames:", runs.groupby("set")[">96 n"].agg(lambda s: sorted(set(s))).to_dict())
print(f"H3 rows of every H6 CSV match the H3 CSV: {bool(chk.h3_rows_match_h6.all())}; max |pred difference| {chk.h3_h6_max_pred_diff.max():.2e} kt")
print("selected epochs:", {a: sorted(chk[chk.arm == a].sel_epoch.tolist()) for a in ARMS})


def perm_p(a, b, n=N_PERM, seed=0):
    rng = np.random.default_rng(seed)
    pool, obs = np.concatenate([a, b]), b.mean() - a.mean()
    idx = np.argsort(rng.random((n, len(pool))), axis=1)
    d = pool[idx[:, len(a):]].mean(axis=1) - pool[idx[:, :len(a)]].mean(axis=1)
    return (np.abs(d) >= abs(obs) - 1e-12).mean()


def compare(set_name, metric, x, y):
    d = runs[runs.set == set_name]
    a, b = d[d.arm == x][metric].dropna().to_numpy(), d[d.arm == y][metric].dropna().to_numpy()
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return dict(set=set_name, metric=metric, comparison=f"{y} - {x}", **{f"{x} mean": a.mean(), f"{y} mean": b.mean()}, difference=diff, SE=se,
                **{"diff/SE": diff / se, "perm p": perm_p(a, b), "Welch p": stats.ttest_ind(b, a, equal_var=False).pvalue,
                   "runs of second arm beyond first arm's range": int(((b < a.min()) | (b > a.max())).sum())})


def summarize(set_name, cols):
    d = runs[runs.set == set_name]
    out = []
    for c in cols:
        r = {"metric": c}
        for arm in ARMS:
            x = d[d.arm == arm][c]
            r[f"{arm} mean"], r[f"{arm} sd"], r[f"{arm} min"], r[f"{arm} max"] = x.mean(), x.std(), x.min(), x.max()
        out.append(r)
    return pd.DataFrame(out)


print("\n\n################ PRIMARY TESTS (fixed before launch; H6 pooled; two-sided; Bonferroni threshold "
      f"{THRESHOLD}) ################")
prim = pd.DataFrame([compare(s, m, "l1", y) for s, m, y in PRIMARY])
prim["survives correction"] = prim["perm p"] < THRESHOLD
prim["arm mean"] = [r[f"{y} mean"] for (_, _, y), (_, r) in zip(PRIMARY, prim.iterrows())]
print(prim[["metric", "comparison", "l1 mean", "arm mean", "difference", "SE", "diff/SE", "perm p", "Welch p", "survives correction"]]
      .to_string(index=False, formatters={"perm p": "{:.4f}".format, "Welch p": "{:.4f}".format}))
surv = prim[prim["survives correction"]]
print("\nSurviving correction: " + ("none." if surv.empty else "; ".join(f"{r.metric} {r.comparison} {r.difference:+.2f} kt (p {r['perm p']:.4f})" for _, r in surv.iterrows())))
prim.to_csv(RES / "primary_tests.csv", index=False)

print("\n\n################ EVERYTHING BELOW IS EXPLORATORY (uncorrected) ################")
BIAS = [f"bias {b}" for *_, b in BANDS]
BMAE = [f"MAE {b}" for *_, b in BANDS]
for set_name, label in (("h3", "H3: Ian, Nicole, Claudette (393 frames)"), ("h6", "H6: + Beryl, Lee, Sam (1197 frames)")):
    d = runs[runs.set == set_name]
    print(f"\n==== {label} ====")
    print("band sizes:", {b: int(d[f'n {b}'].iloc[0]) for *_, b in BANDS})
    print("\n-- overall (mean / sd / min / max over ten runs) --")
    print(summarize(set_name, ["overall MAE", "fixes-only MAE", "overall bias", ">96 MAE", ">96 bias", ">96 fixes MAE"]).to_string(index=False))
    print("\n-- per-category bias --")
    print(summarize(set_name, BIAS).to_string(index=False))
    print("\n-- per-category MAE --")
    print(summarize(set_name, BMAE).to_string(index=False))
    stormcols =[f"{k} {s}" for s in sorted(set(c.split(' ', 1)[1] for c in d.columns if c.startswith('MAE ') and c not in BMAE)) for k in ("MAE", "bias")]
    print("\n-- per storm, all frames --")
    print(summarize(set_name, [c for c in stormcols if d[c].notna().any()]).to_string(index=False))
    summarize(set_name, ["overall MAE", "fixes-only MAE", "overall bias", ">96 MAE", ">96 bias", ">96 fixes MAE"] + BIAS + BMAE + stormcols).to_csv(RES / f"summary_by_arm_{set_name}.csv", index=False)

print("\n==== >96 kt band per storm (H6 CSVs; Ian's figures are identical on H3) ====")
d6 = runs[runs.set == "h6"]
print("frames >96 kt:", {s: int(d6[f'>96 n {s}'].iloc[0]) for s in HIGH_STORMS})
print(summarize("h6", [f">96 {k} {s}" for s in HIGH_STORMS for k in ("MAE", "bias")]).to_string(index=False))
print("\nper storm and band (Cat 3 / Cat 4 / Cat 5 bias):")
print(summarize("h6", [c for s in HIGH_STORMS for c in (f"{s} Cat 3 bias", f"{s} Cat 4 bias", f"{s} Cat 5 bias") if c in d6.columns and d6[c].notna().any()]).to_string(index=False))

print("\n==== Does a high-end change separate Cat 3 from Cat 4/5, or move the high end together? (exploratory; Ian is the check) ====")
print("Ian is over-predicted at Cat 3 and under-predicted at Cat 4/5, so a change that separates the bands moves Cat 3 bias down and Cat 4/5")
print("bias up (the predicted gap between them widens); a change that moves the high end together shifts both the same way.")
sep_rows = []
for s in HIGH_STORMS:
    if f"{s} pred gap Cat4/5-Cat3" not in d6.columns or d6[f"{s} pred gap Cat4/5-Cat3"].isna().all():
        continue
    true_gap = d6[f"{s} true gap Cat4/5-Cat3"].iloc[0]
    for arm in ["huber", "mse"]:
        r = {"storm": s, "arm": arm, "true gap": true_gap}
        for c, k in ((f"{s} Cat 3 bias", "Cat 3 bias shift"), (f"{s} Cat 4/5 bias", "Cat 4/5 bias shift"), (f"{s} pred gap Cat4/5-Cat3", "pred gap shift")):
            x = compare("h6", c, "l1", arm)
            r[k], r[f"{k} p"] = x["difference"], x["perm p"]
        r["l1 pred gap"] = d6[d6.arm == "l1"][f"{s} pred gap Cat4/5-Cat3"].mean()
        r["arm pred gap"] = d6[d6.arm == arm][f"{s} pred gap Cat4/5-Cat3"].mean()
        same_dir = np.sign(r["Cat 3 bias shift"]) == np.sign(r["Cat 4/5 bias shift"])
        r["pattern"] = ("moves together " + ("up" if r["Cat 3 bias shift"] > 0 else "down")) if same_dir else (
            "separates (gap widens)" if r["pred gap shift"] > 0 else "compresses (gap narrows)")
        sep_rows.append(r)
sep = pd.DataFrame(sep_rows)
print(sep.to_string(index=False))

print("\n==== all pairwise differences, exploratory ====")
EXP = ["overall MAE", "fixes-only MAE", "overall bias", ">96 MAE", ">96 bias"] + BIAS + BMAE
exp_rows = []
for set_name in ("h3", "h6"):
    cols = EXP + ([f">96 {k} {s}" for s in HIGH_STORMS for k in ("MAE", "bias")] if set_name == "h6" else [">96 MAE Ian", ">96 bias Ian"])
    for c in cols:
        for x, y in (("l1", "huber"), ("l1", "mse"), ("huber", "mse")):
            if (set_name, c, y) in PRIMARY and x == "l1":
                continue
            exp_rows.append(compare(set_name, c, x, y))
exp = pd.DataFrame(exp_rows)
exp.to_csv(RES / "exploratory_differences.csv", index=False)
print(exp[["set", "metric", "comparison", "difference", "SE", "perm p", "runs of second arm beyond first arm's range"]].to_string(index=False))

print("\n==== L1 arm against arm A of the class-imbalance experiment (same configuration; consistency check) ====")
a = pd.read_csv(REPO / "results" / "imbalance" / "summary_per_run.csv")
a = a[(a.arm == "A_baseline_40ep") & (a.set == "h3")]["overall MAE"]
l1 = runs[(runs.arm == "l1") & (runs.set == "h3")]["overall MAE"]
print(f"H3 overall MAE: L1 {l1.mean():.2f} +- {l1.std():.2f} [{l1.min():.2f}-{l1.max():.2f}] vs A {a.mean():.2f} +- {a.std():.2f} [{a.min():.2f}-{a.max():.2f}]; "
      f"difference {l1.mean() - a.mean():+.2f}, perm p {perm_p(a.to_numpy(), l1.to_numpy()):.3f}")

print("\n==== run order (check fixed in advance) ====")
for set_name in ("h3", "h6"):
    d = runs[runs.set == set_name].copy()
    d["resid"] = d["overall MAE"] - d.groupby("arm")["overall MAE"].transform("mean")
    per_arm = ", ".join(f"{a} {d[d.arm == a]['round'].corr(d[d.arm == a]['overall MAE'], method='spearman'):+.2f}" for a in ARMS)
    rho, p = stats.spearmanr(d["order"], d["resid"])
    print(f"{set_name}: overall MAE vs round, Spearman within arm: {per_arm} | pooled (arm means removed) vs run order {rho:+.2f} (p {p:.2f})")
    print(f"    mean residual by position in round (1st/2nd/3rd): " + ", ".join(f"{k}: {v:+.2f}" for k, v in d.groupby('position_in_round').resid.mean().items()))
    print("    mean residual by round: " + " ".join(f"{k}:{v:+.2f}" for k, v in d.groupby("round").resid.mean().items()))

print("\n==== per-run values ====")
w = runs.pivot_table(index=["arm", "seed"], columns="set", values=["overall MAE", ">96 MAE", ">96 bias"]).round(2)
w.columns = [f"{m} {s}" for m, s in w.columns]
w = w.join(runs[runs.set == "h6"].set_index(["arm", "seed"])[["order", "sel_epoch", "val_mae", "Ian Cat 3 bias", "Ian Cat 4/5 bias"]])
print(w.reindex(ARMS, level=0).to_string())
