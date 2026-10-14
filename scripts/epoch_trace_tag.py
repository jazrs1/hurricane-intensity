"""Evaluate every saved epoch of the traced baseline runs on validation and on the original held-out storms.

Reads checkpoints from models/<tag>/seedN/epoch_NN.pt; writes results/<tag>/*.csv. Trains nothing.
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

REPO = Path(sys.argv[1])
TAG = sys.argv[2]  # e.g. epoch_traces or epoch_traces_60
SEEDS = [int(s) for s in sys.argv[3:]] or [0, 1, 2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
from hurricane_intensity.dataset import TCDataset  # noqa: E402
from hurricane_intensity.splits import load_splits  # noqa: E402

_, VAL, HOLD = load_splits(train_list=REPO / "storms_train_baseline.txt", holdout_list=REPO / "storms_holdout_baseline.txt")
print("val storms:", VAL, "| held-out storms:", HOLD)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat1"), (83, 96, "Cat2"), (96, 113, "Cat3"), (113, 137, "Cat4"), (137, np.inf, "Cat5")]


def load(ids):
    ds = TCDataset("data/processed", ids)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([ds[i][1] for i in range(len(ds))]).numpy() * 185.0
    return ds.df.copy(), x, y


val_df, val_x, val_y = load(VAL)
hold_df, hold_x, hold_y = load(HOLD)
hold_fix = ~hold_df["is_interpolated"].to_numpy().astype(bool)

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device).eval()


def predict(x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):  # same batch size as train.py / evaluate.py
            out.append(model(x[i:i + 32].to(device)).squeeze(1).cpu())
    return torch.cat(out).numpy() * 185.0


rows, frames = [], []
for seed in SEEDS:
    log = (REPO / "logs" / TAG / f"train_seed{seed}.log").read_text()
    logged = {int(e): (float(tr), float(va)) for e, tr, va in re.findall(r"epoch (\d+): train ([\d.]+) kt\s+val ([\d.]+) kt", log)}
    for epoch in sorted(logged):
        model.load_state_dict(torch.load(REPO / "models" / TAG / f"seed{seed}" / f"epoch_{epoch:02d}.pt"))
        pv, ph = predict(val_x), predict(hold_x)
        ev, eh = pv - val_y, ph - hold_y
        r = dict(seed=seed, epoch=epoch, train_mae_log=logged[epoch][0], val_mae_log=logged[epoch][1],
                 val_mae=np.abs(ev).mean(), val_bias=ev.mean(), heldout_mae=np.abs(eh).mean(),
                 heldout_fix_mae=np.abs(eh)[hold_fix].mean(), heldout_bias=eh.mean())
        for name, df_, e in (("val", val_df, ev), ("heldout", hold_df, eh)):
            for storm in df_["storm_name"].unique():
                r[f"{name}_mae_{storm.title()}"] = np.abs(e)[(df_["storm_name"] == storm).to_numpy()].mean()
        for lo, hi, b in BANDS:
            m = (hold_y >= lo) & (hold_y < hi)
            r[f"heldout_mae_{b}"], r[f"heldout_bias_{b}"] = np.abs(eh)[m].mean(), eh[m].mean()
        m = hold_y > 96
        r["heldout_mae_gt96"], r["heldout_bias_gt96"] = np.abs(eh)[m].mean(), eh[m].mean()
        mv = val_y > 96
        r["val_mae_gt96"], r["val_bias_gt96"] = np.abs(ev)[mv].mean(), ev[mv].mean()
        rows.append(r)
        frames.append(pd.DataFrame(dict(seed=seed, epoch=epoch, atcf_id=hold_df["atcf_id"].to_numpy(),
                                        sample_time=hold_df["sample_time"].to_numpy(), truth=hold_y, pred=ph)))
    best_ckpt = torch.load(REPO / "models" / TAG / f"seed{seed}" / "best_val.pt")
    sel = min(logged, key=lambda e: logged[e][1])
    same = all(torch.equal(best_ckpt[k].cpu(), torch.load(REPO / "models" / TAG / f"seed{seed}" / f"epoch_{sel:02d}.pt")[k].cpu()) for k in best_ckpt)
    print(f"seed {seed}: {len(logged)} epochs; train.py's saved checkpoint is bit-identical to epoch_{sel:02d}.pt: {same}")

t = pd.DataFrame(rows)
out = REPO / "results" / TAG
out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / "epoch_trace.csv", index=False)
pd.concat(frames, ignore_index=True).to_csv(out / "heldout_predictions_by_epoch.csv", index=False)
print(f"wrote {out / 'epoch_trace.csv'} ({len(t)} rows) and heldout_predictions_by_epoch.csv")
print(f"max |recomputed val MAE - val MAE printed by train.py|: {(t.val_mae - t.val_mae_log).abs().max():.3f} kt")
