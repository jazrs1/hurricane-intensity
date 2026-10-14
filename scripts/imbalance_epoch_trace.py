"""Evaluate every saved epoch of the class-imbalance runs on validation and on the four held-out storms.

Reads models/imbalance/<arm>/seed<s>/epoch_NN.pt; writes results/imbalance/epoch_trace.csv (one row per
arm, seed and epoch, with metrics on both the three-storm and four-storm held-out sets) and
heldout_predictions_by_epoch.csv.gz (per-sample predictions on the four storms, which contain the three).
Trains nothing. Run from the repo root.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
from hurricane_intensity.dataset import TCDataset  # noqa: E402
from hurricane_intensity.splits import load_splits  # noqa: E402

ARMS = ["A_baseline_40ep", "B_expanded_29ep", "C_expanded_40ep"]
SAM = "AL182021"
_, VAL, HOLD = load_splits(train_list=REPO / "storms_train_baseline.txt", holdout_list=REPO / "storms_holdout.txt")
print("val storms:", VAL, "| held-out storms:", HOLD)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat1"), (83, 96, "Cat2"), (96, 113, "Cat3"), (113, 137, "Cat4"), (137, np.inf, "Cat5")]


def load(ids):
    ds = TCDataset("data/processed", ids)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([ds[i][1] for i in range(len(ds))]).numpy() * 185.0
    return ds.df.copy().reset_index(drop=True), x, y


val_df, val_x, val_y = load(VAL)
hold_df, hold_x, hold_y = load(HOLD)
fix = ~hold_df["is_interpolated"].to_numpy().astype(bool)
storm = hold_df["storm_name"].str.title().to_numpy()
in3 = (hold_df["atcf_id"] != SAM).to_numpy()

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device).eval()


def predict(x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):  # same batch size as train.py / evaluate.py
            out.append(model(x[i:i + 32].to(device)).squeeze(1).cpu())
    return torch.cat(out).numpy() * 185.0


def held_metrics(prefix, e, t, k):
    e, t, f, s = e[k], t[k], fix[k], storm[k]
    r = {f"{prefix}_mae": np.abs(e).mean(), f"{prefix}_fix_mae": np.abs(e)[f].mean(), f"{prefix}_bias": e.mean()}
    hi = t > 96
    r[f"{prefix}_mae_gt96"], r[f"{prefix}_bias_gt96"] = np.abs(e)[hi].mean(), e[hi].mean()
    for lo, up, b in BANDS:
        m = (t >= lo) & (t < up)
        r[f"{prefix}_mae_{b}"], r[f"{prefix}_bias_{b}"] = np.abs(e)[m].mean(), e[m].mean()
    for name in np.unique(s):
        m = s == name
        r[f"{prefix}_mae_{name}"] = np.abs(e)[m].mean()
        if (m & hi).any():
            r[f"{prefix}_mae_gt96_{name}"], r[f"{prefix}_bias_gt96_{name}"] = np.abs(e)[m & hi].mean(), e[m & hi].mean()
    return r


rows, frames, checks = [], [], []
for seed in range(10):
    for arm in ARMS:
        log = (REPO / "logs" / "imbalance" / f"{arm}_seed{seed}.log").read_text()
        logged = {int(e): (float(tr), float(va)) for e, tr, va in re.findall(r"epoch (\d+): train ([\d.]+) kt\s+val ([\d.]+) kt", log)}
        sel = min(logged, key=lambda e: logged[e][1])
        d = REPO / "models" / "imbalance" / arm / f"seed{seed}"
        for epoch in sorted(logged):
            model.load_state_dict(torch.load(d / f"epoch_{epoch:02d}.pt"))
            pv, ph = predict(val_x), predict(hold_x)
            ev, eh = pv - val_y, ph - hold_y
            r = dict(arm=arm, seed=seed, epoch=epoch, selected=epoch == sel, train_mae_log=logged[epoch][0], val_mae_log=logged[epoch][1],
                     val_mae=np.abs(ev).mean(), val_bias=ev.mean(), val_mae_gt96=np.abs(ev)[val_y > 96].mean(), val_bias_gt96=ev[val_y > 96].mean())
            r.update(held_metrics("h3", eh, hold_y, in3))
            r.update(held_metrics("h4", eh, hold_y, np.ones_like(in3)))
            rows.append(r)
            frames.append(pd.DataFrame(dict(arm=arm, seed=seed, epoch=epoch, atcf_id=hold_df["atcf_id"].to_numpy(),
                                            sample_time=hold_df["sample_time"].to_numpy(), truth=hold_y, pred=ph.astype(np.float32))))
        best = torch.load(d / "best_val.pt")
        same = all(torch.equal(best[k].cpu(), v.cpu()) for k, v in torch.load(d / f"epoch_{sel:02d}.pt").items())
        ref = pd.read_csv(REPO / "results" / "imbalance" / f"{arm}_seed{seed}_holdout4.csv")
        recomputed = [x for x in rows if x["arm"] == arm and x["seed"] == seed and x["selected"]][0]["h4_mae"]
        checks.append(dict(arm=arm, seed=seed, epochs=len(logged), selected=sel, best_val_bit_identical=same,
                           h4_mae_evaluate_py=ref.abs_error.mean(), h4_mae_recomputed=recomputed))
        print(f"{arm} seed {seed}: {len(logged)} epochs, selected {sel}, best_val.pt == epoch_{sel:02d}.pt: {same}, "
              f"4-storm MAE evaluate.py {ref.abs_error.mean():.3f} vs recomputed {recomputed:.3f}", flush=True)

out = REPO / "results" / "imbalance"
t = pd.DataFrame(rows)
t.to_csv(out / "epoch_trace.csv", index=False)
pd.concat(frames, ignore_index=True).to_csv(out / "heldout_predictions_by_epoch.csv.gz", index=False, float_format="%.4f")
c = pd.DataFrame(checks)
print(f"wrote {out / 'epoch_trace.csv'} ({len(t)} rows) and heldout_predictions_by_epoch.csv.gz")
print(f"all best_val.pt bit-identical to the selected epoch: {c.best_val_bit_identical.all()}")
print(f"max |recomputed - evaluate.py| 4-storm MAE at the selected epoch: {(c.h4_mae_recomputed - c.h4_mae_evaluate_py).abs().max():.3f} kt")
print(f"max |recomputed val MAE - val MAE printed by train.py|: {(t.val_mae - t.val_mae_log).abs().max():.3f} kt")
