"""Read-only analysis of results/epoch_traces_60/epoch_trace.csv: where does held-out MAE bottom out, and does it rise?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1])
t = pd.read_csv(REPO / "results" / "epoch_traces_60" / "epoch_trace.csv")
seeds = sorted(t.seed.unique())
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")
WINDOWS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)]
STEPS_BASE, STEPS_EXP = 41, 57  # batches per epoch: ceil(1307/32), ceil(1816/32)

print("== per-epoch held-out MAE (kt); v = validation MAE; * = validation's pick over 60 epochs, o = lowest held-out ==")
print("epoch | " + " | ".join(f"seed {s}: train    val   held  >96bias" for s in seeds))
for e in range(1, 61):
    cells = []
    for s in seeds:
        d = t[t.seed == s].set_index("epoch")
        mark = ("*" if e == d.val_mae.idxmin() else " ") + ("o" if e == d.heldout_mae.idxmin() else " ")
        cells.append(f"{d.train_mae_log[e]:6.2f} {d.val_mae[e]:6.2f} {d.heldout_mae[e]:6.2f}{mark} {d.heldout_bias_gt96[e]:+7.2f}")
    print(f"{e:5d} | " + " | ".join(cells))

print("\n== ten-epoch windows: mean / median / min / sd of held-out MAE across the epochs in the window ==")
for col, label in (("heldout_mae", "held-out MAE"), ("val_mae", "validation MAE"), ("train_mae_log", "training MAE"), ("heldout_bias_gt96", ">96 kt held-out bias"), ("heldout_mae_gt96", ">96 kt held-out MAE")):
    rows = []
    for lo, hi in WINDOWS:
        w = t[(t.epoch >= lo) & (t.epoch <= hi)]
        r = {"epochs": f"{lo}-{hi}"}
        for s in seeds:
            x = w[w.seed == s][col]
            r[f"s{s} mean"], r[f"s{s} med"], r[f"s{s} min"], r[f"s{s} sd"] = x.mean(), x.median(), x.min(), x.std()
        r["all mean"], r["all med"] = w[col].mean(), w[col].median()
        r["all sd within run"] = w.groupby("seed")[col].std().mean()
        rows.append(r)
    print(f"\n{label}:")
    print(pd.DataFrame(rows).to_string(index=False))

print("\n== where it bottoms out, per run ==")
rows = []
for s in seeds:
    d = t[t.seed == s].sort_values("epoch").reset_index(drop=True)
    roll = d.heldout_mae.rolling(5, center=True).median()
    rollm = d.heldout_mae.rolling(5, center=True).mean()
    late = d[d.epoch >= 21]
    slope = np.polyfit(late.epoch, late.heldout_mae, 1)[0]
    # robust slope: Theil-Sen
    x, y = late.epoch.to_numpy(), late.heldout_mae.to_numpy()
    ts = np.median([(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))])
    rows.append(dict(seed=s, raw_min_epoch=int(d.epoch[d.heldout_mae.idxmin()]), raw_min=d.heldout_mae.min(),
                     roll5_median_min_epoch=int(d.epoch[roll.idxmin()]), roll5_median_min=roll.min(),
                     roll5_mean_min_epoch=int(d.epoch[rollm.idxmin()]), roll5_mean_min=rollm.min(),
                     roll5_median_at_20=roll[d.epoch == 18].iloc[0], roll5_median_last=roll[d.epoch == 58].iloc[0],
                     slope_21_60_kt_per_10ep=slope * 10, theil_sen_21_60_per_10ep=ts * 10,
                     train_at_20=d.train_mae_log[d.epoch == 20].iloc[0], train_at_60=d.train_mae_log.iloc[-1],
                     val_min_epoch=int(d.epoch[d.val_mae.idxmin()])))
print(pd.DataFrame(rows).to_string(index=False))

print("\n== the number the experiment would use: held-out MAE of the validation-selected checkpoint, by epoch budget ==")
rows = []
for budget in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60):
    r = {"budget": budget, "base steps": budget * STEPS_BASE, "= experiment epochs at equal steps": budget * STEPS_BASE / STEPS_EXP}
    hs = []
    for s in seeds:
        d = t[(t.seed == s) & (t.epoch <= budget)]
        sel = d.loc[d.val_mae.idxmin()]
        r[f"s{s} pick"], r[f"s{s} held"] = int(sel.epoch), sel.heldout_mae
        hs.append(sel.heldout_mae)
    r["mean held"], r["oracle mean"] = np.mean(hs), np.mean([t[(t.seed == s) & (t.epoch <= budget)].heldout_mae.min() for s in seeds])
    r["mean >96 bias at pick"] = np.mean([t[(t.seed == s) & (t.epoch <= budget)].sort_values("val_mae").heldout_bias_gt96.iloc[0] for s in seeds])
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== does validation still track held-out late? each run's mean removed, by window ==")
for lo, hi in ((11, 20), (21, 40), (41, 60), (21, 60)):
    w = t[(t.epoch >= lo) & (t.epoch <= hi)].copy()
    v = w.val_mae - w.groupby("seed").val_mae.transform("mean")
    h = w.heldout_mae - w.groupby("seed").heldout_mae.transform("mean")
    print(f"  epochs {lo}-{hi}: pearson {v.corr(h):+.2f}, spearman {v.corr(h, method='spearman'):+.2f}, held-out sd within run {w.groupby('seed').heldout_mae.std().mean():.2f} kt, "
          f"worst epoch {w.heldout_mae.max():.2f}, epochs above 12 kt: {int((w.heldout_mae > 12).sum())}/{len(w)}")

print("\n== generalization gap: held-out MAE minus training MAE, window means over the three runs ==")
for lo, hi in WINDOWS:
    w = t[(t.epoch >= lo) & (t.epoch <= hi)]
    print(f"  epochs {lo}-{hi}: train {w.train_mae_log.mean():5.2f}  val {w.val_mae.mean():5.2f}  held-out {w.heldout_mae.mean():5.2f}  held-out fixes {w.heldout_fix_mae.mean():5.2f}  gap {w.heldout_mae.mean() - w.train_mae_log.mean():5.2f}")
