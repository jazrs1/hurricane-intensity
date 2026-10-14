"""Read-only analysis of results/epoch_traces/epoch_trace.csv: does validation MAE track held-out MAE?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1])
LATE = int(sys.argv[2]) if len(sys.argv) > 2 else 11  # first epoch of the "late" window
t = pd.read_csv(REPO / "results" / "epoch_traces" / "epoch_trace.csv")
seeds = sorted(t.seed.unique())
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")

print("== per-epoch validation vs held-out MAE (kt); * = epoch validation selects, o = lowest held-out ==")
wide = t.pivot(index="epoch", columns="seed", values=["train_mae_log", "val_mae", "heldout_mae", "heldout_bias_gt96"])
lines = []
for e in wide.index:
    cells = []
    for s in seeds:
        d = t[t.seed == s]
        sel, orc = d.loc[d.val_mae.idxmin(), "epoch"], d.loc[d.heldout_mae.idxmin(), "epoch"]
        mark = ("*" if e == sel else " ") + ("o" if e == orc else " ")
        cells.append(f"{wide.loc[e, ('train_mae_log', s)]:6.2f} {wide.loc[e, ('val_mae', s)]:6.2f} {wide.loc[e, ('heldout_mae', s)]:6.2f}{mark} {wide.loc[e, ('heldout_bias_gt96', s)]:+7.2f}")
    lines.append(f"{e:5d} | " + " | ".join(cells))
print("epoch | " + " | ".join(f"seed {s}: train    val   held   >96bias" for s in seeds))
print("\n".join(lines))


def corr(a, b, method):
    return pd.Series(a).corr(pd.Series(b), method=method)


print(f"\n== correlation of validation MAE with held-out MAE across epochs (late = epochs {LATE}-20) ==")
rows = []
for s in seeds:
    d = t[t.seed == s].sort_values("epoch")
    late = d[d.epoch >= LATE]
    dv, dh = np.diff(d.val_mae), np.diff(d.heldout_mae)
    lv, lh = np.diff(late.val_mae), np.diff(late.heldout_mae)
    rows.append(dict(seed=s,
                     spearman_all=corr(d.val_mae.values, d.heldout_mae.values, "spearman"),
                     pearson_all=corr(d.val_mae.values, d.heldout_mae.values, "pearson"),
                     spearman_late=corr(late.val_mae.values, late.heldout_mae.values, "spearman"),
                     pearson_late=corr(late.val_mae.values, late.heldout_mae.values, "pearson"),
                     spearman_steps_all=corr(dv, dh, "spearman"), spearman_steps_late=corr(lv, lh, "spearman"),
                     same_direction_steps_late=f"{int((np.sign(lv) == np.sign(lh)).sum())}/{len(lv)}"))
print(pd.DataFrame(rows).to_string(index=False))
late = t[t.epoch >= LATE].copy()
late["v"] = late.val_mae - late.groupby("seed").val_mae.transform("mean")
late["h"] = late.heldout_mae - late.groupby("seed").heldout_mae.transform("mean")
print(f"pooled late epochs, each run's mean removed (n={len(late)}): pearson {late.v.corr(late.h):+.2f}, spearman {late.v.corr(late.h, method='spearman'):+.2f}")
rng = np.random.default_rng(0)
perm = []
for _ in range(20000):
    hh = np.concatenate([rng.permutation(g.h.values) for _, g in late.groupby("seed")])
    perm.append(np.corrcoef(late.v.values, hh)[0, 1])
perm = np.array(perm)
obs = late.v.corr(late.h)
print(f"  permutation test (held-out shuffled across epochs within run): P(r >= observed) = {(perm >= obs).mean():.3f}; 95% of shuffles fall in [{np.percentile(perm, 2.5):+.2f}, {np.percentile(perm, 97.5):+.2f}]")
print(f"late-epoch spread (sd across epochs, mean over runs): val {late.groupby('seed').val_mae.std().mean():.2f} kt, held-out {late.groupby('seed').heldout_mae.std().mean():.2f} kt")

print("\n== what validation selects vs what was available ==")
rows = []
for s in seeds:
    d = t[t.seed == s].sort_values("epoch").reset_index(drop=True)
    sel, orc = d.loc[d.val_mae.idxmin()], d.loc[d.heldout_mae.idxmin()]
    lt = d[d.epoch >= LATE]
    rows.append(dict(seed=s, selected_epoch=int(sel.epoch), val_at_selected=sel.val_mae, heldout_at_selected=sel.heldout_mae,
                     best_heldout_epoch=int(orc.epoch), best_heldout=orc.heldout_mae, regret=sel.heldout_mae - orc.heldout_mae,
                     val_at_best_heldout=orc.val_mae, heldout_final_epoch=d.heldout_mae.iloc[-1],
                     late_mean=lt.heldout_mae.mean(), late_min=lt.heldout_mae.min(), late_max=lt.heldout_mae.max(),
                     selected_rank_late=f"{int((lt.heldout_mae < sel.heldout_mae).sum()) + 1}/{len(lt)}" if sel.epoch >= LATE else "n/a",
                     selected_minus_late_mean=sel.heldout_mae - lt.heldout_mae.mean()))
r = pd.DataFrame(rows)
print(r.to_string(index=False))
print(f"mean over runs: held-out at selected {r.heldout_at_selected.mean():.2f}, mean of late epochs {r.late_mean.mean():.2f}, "
      f"best available {r.best_heldout.mean():.2f}, final epoch {r.heldout_final_epoch.mean():.2f}")

print("\n== first epoch within 0.25 / 0.5 kt of each run's best held-out MAE ==")
for s in seeds:
    d = t[t.seed == s].sort_values("epoch")
    b = d.heldout_mae.min()
    print(f"seed {s}: best {b:.2f} at epoch {int(d.loc[d.heldout_mae.idxmin(), 'epoch'])}; within 0.25 kt first at epoch {int(d[d.heldout_mae <= b + 0.25].epoch.min())}; "
          f"within 0.5 kt first at epoch {int(d[d.heldout_mae <= b + 0.5].epoch.min())}; epochs within 0.5 kt: {list(d[d.heldout_mae <= b + 0.5].epoch)}")

print("\n== where the noise is: per-storm MAE spread across late epochs (sd, kt), and what validation correlates with ==")
storm_cols = [c for c in t.columns if c.startswith(("val_mae_", "heldout_mae_")) and c.split("_")[-1][0].isupper() and not c.endswith(("TD", "TS")) and "Cat" not in c]
print(late.groupby("seed")[storm_cols].std().mean().to_string())
print("\nlate-epoch pearson, run mean removed, validation MAE vs:")
for c in ["heldout_mae"] + [c for c in storm_cols if c.startswith("heldout")] + ["heldout_mae_gt96", "heldout_bias_gt96", "heldout_bias", "val_bias"]:
    x = late[c] - late.groupby("seed")[c].transform("mean")
    print(f"  {c:26s} {late.v.corr(x):+.2f}")
print("late-epoch pearson, run mean removed, held-out storms against each other:")
hs = [c for c in storm_cols if c.startswith("heldout")]
dm = late[hs] - late.groupby("seed")[hs].transform("mean")
print(dm.corr().to_string())
print("\nlate-epoch pearson, run mean removed: overall validation bias vs overall held-out bias:",
      f"{(late.val_bias - late.groupby('seed').val_bias.transform('mean')).corr(late.heldout_bias - late.groupby('seed').heldout_bias.transform('mean')):+.2f}")

print("\n== >96 kt held-out bias across late epochs, per run ==")
print(late.groupby("seed").heldout_bias_gt96.agg(["min", "max", "mean", "std"]).to_string())
print("at the validation-selected epoch:", {int(s): round(float(t[t.seed == s].sort_values('val_mae').heldout_bias_gt96.iloc[0]), 2) for s in seeds})
