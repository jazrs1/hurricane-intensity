"""Supplementary read-only checks on the epoch traces and on all baseline reruns."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
t = pd.read_csv(REPO / "results" / "epoch_traces" / "epoch_trace.csv")
late = t[t.epoch >= 11].copy()


def dm(d, c):
    return d[c] - d.groupby("seed")[c].transform("mean")


print("== is the late-epoch correlation only the spike epochs? ==")
for label, d in (("all late epochs", late),
                 ("late epochs with val within 2 kt of the run's best val", late[late.val_mae <= late.groupby("seed").val_mae.transform("min") + 2]),
                 ("late epochs with val within 1 kt of the run's best val", late[late.val_mae <= late.groupby("seed").val_mae.transform("min") + 1])):
    v, h = dm(d, "val_mae"), dm(d, "heldout_mae")
    print(f"  {label}: n={len(d)}  pearson {v.corr(h):+.2f}  spearman {v.corr(h, method='spearman'):+.2f}  held-out sd {h.std():.2f} kt")

print("\n== does validation see the >96 kt bias? late epochs, run mean removed ==")
print(f"  val >96 bias (25 Laura frames) vs held-out >96 bias (47 Ian frames): pearson {dm(late, 'val_bias_gt96').corr(dm(late, 'heldout_bias_gt96')):+.2f}")
print(f"  overall val bias vs held-out >96 bias: {dm(late, 'val_bias').corr(dm(late, 'heldout_bias_gt96')):+.2f}")
late["hb_resid"] = late.heldout_bias_gt96 - late.heldout_bias
late["vb_resid"] = late.val_bias_gt96 - late.val_bias
print(f"  >96 bias relative to the overall bias, val vs held-out: {dm(late, 'vb_resid').corr(dm(late, 'hb_resid')):+.2f}")
print(f"  val MAE vs |held-out >96 bias|: {dm(late, 'val_mae').corr(late.heldout_bias_gt96.abs() - late.assign(a=late.heldout_bias_gt96.abs()).groupby('seed').a.transform('mean')):+.2f}")

print("\n== selection rules, mean held-out MAE over the three traced runs ==")
rules = {}
for s, d in t.groupby("seed"):
    d = d.sort_values("epoch").reset_index(drop=True)
    rules.setdefault("validation-selected (train.py)", []).append(d.loc[d.val_mae.idxmin(), "heldout_mae"])
    rules.setdefault("final epoch (20)", []).append(d.heldout_mae.iloc[-1])
    rules.setdefault("random epoch from 11-20 (expected)", []).append(d[d.epoch >= 11].heldout_mae.mean())
    rules.setdefault("worst epoch from 11-20", []).append(d[d.epoch >= 11].heldout_mae.max())
    rules.setdefault("best held-out epoch (oracle, not usable)", []).append(d.heldout_mae.min())
for k, v in rules.items():
    print(f"  {k:42s} {np.mean(v):6.2f}   per run: {[round(x, 2) for x in v]}")

print("\n== every baseline rerun so far: validation MAE of the selected checkpoint vs its held-out MAE ==")
runs = [("original 9.00 run", 20, 9.44, 9.00)]


def from_log(name, log_path, heldout):
    log = Path(log_path).read_text()
    ep = [(int(e), float(va)) for e, va in re.findall(r"epoch (\d+): train [\d.]+ kt\s+val ([\d.]+) kt", log)]
    e, v = min(ep, key=lambda r: r[1])
    runs.append((name, e, v, heldout))


def heldout_from_eval(path):
    return float(re.search(r"overall MAE:\s+([\d.]+)", Path(path).read_text()).group(1))


for s in (0, 1, 2):
    from_log(f"first batch seed {s}", REPO / "logs/seed_baseline" / f"train_seed{s}.log", heldout_from_eval(REPO / "logs/seed_baseline" / f"eval_seed{s}.log"))
from_log("first batch seed 0 repeat", REPO / "logs/seed_baseline/train_seed0_repeat.log", heldout_from_eval(REPO / "logs/seed_baseline/eval_seed0_repeat.log"))
for s in (0, 1, 2):
    d = t[t.seed == s]
    r = d.loc[d.val_mae.idxmin()]
    runs.append((f"traced seed {s}", int(r.epoch), round(r.val_mae, 2), round(r.heldout_mae, 2)))
for s in (0, 1, 2):
    # control runs were first logged in a session scratch folder; identical copies live in logs/epoch_traces/
    from_log(f"control seed {s}", REPO / "logs/epoch_traces" / f"control_train_seed{s}.log", heldout_from_eval(REPO / "logs/epoch_traces" / f"control_eval_seed{s}.log"))
r = pd.DataFrame(runs, columns=["run", "selected_epoch", "val_mae", "heldout_mae"])
r["heldout_minus_val"] = r.heldout_mae - r.val_mae
print(r.to_string(index=False))
re_ = r[r.run != "original 9.00 run"]
print(f"\n10 reruns: held-out mean {re_.heldout_mae.mean():.2f}, sd {re_.heldout_mae.std():.2f}, min {re_.heldout_mae.min():.2f}, max {re_.heldout_mae.max():.2f}; "
      f"reruns at or below 9.00: {int((re_.heldout_mae <= 9.0).sum())}")
print(f"selected val: mean {re_.val_mae.mean():.2f}, sd {re_.val_mae.std():.2f}")
print(f"across runs, selected val vs held-out: pearson {re_.val_mae.corr(re_.heldout_mae):+.2f}, spearman {re_.val_mae.corr(re_.heldout_mae, method='spearman'):+.2f} (n=10)")
rng = np.random.default_rng(0)
boot = [re_.sample(len(re_), replace=True, random_state=int(rng.integers(1e9))).pipe(lambda d: d.val_mae.corr(d.heldout_mae)) for _ in range(5000)]
print(f"  bootstrap 95% interval for that pearson: [{np.nanpercentile(boot, 2.5):+.2f}, {np.nanpercentile(boot, 97.5):+.2f}]")
print(f"selected epochs: {sorted(re_.selected_epoch.tolist())}")
flag_on = re_[re_.run.str.startswith("traced")].heldout_mae
flag_off = re_[~re_.run.str.startswith("traced")].heldout_mae
print(f"per-epoch saving on: {sorted(flag_on.round(2))}  off: {sorted(flag_off.round(2))}")
r.to_csv(REPO / "results" / "epoch_traces" / "baseline_reruns.csv", index=False)
