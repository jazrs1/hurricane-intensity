#!/usr/bin/env python
"""Analog forecasting Version 1: does PDAL + kNN retrieval estimate current intensity as well as MSE regression?

    .venv\\Scripts\\python.exe scripts\\pdal_v1.py train --arm A --seed 0   # PDAL (scale-invariant) + kNN
    .venv\\Scripts\\python.exe scripts\\pdal_v1.py train --arm B --seed 0   # MSE regression
    .venv\\Scripts\\python.exe scripts\\pdal_v1.py frozen                   # C: ImageNet ResNet-34 features + kNN
    .venv\\Scripts\\python.exe scripts\\pdal_v1.py ensemble                 # D: the frozen ResNet-18 ensemble
    .venv\\Scripts\\python.exe scripts\\pdal_v1.py report                   # tables, figures, results.json

Data: configs/analog_splits.json, data/storm_windows/. Each sample is a StormWindowDataset window with history=2
(t-6 h .. t+24 h at 3 h steps, land/status filters at t and t+24 h); the model input is the 3-frame stack
[t-6 h, t-3 h, t] as channels, the label the intensity at t. The 2024-2025 test split is never read.

Outputs (gitignored) under data/exports/pdal_v1/: <arm>_s<seed>/ (log.csv, best.pt, result.npz), C/, D/, figures.
A run whose result.npz exists is skipped, so an interrupted batch of runs can be restarted.

Selection: A stops early on validation kNN MAE (k=10, all filtered validation windows, patience 5), B on validation
regression MAE; k is then tuned in {5, 10, 20, 50} on the same windows. Metrics are reported on the validation
windows whose anchor t is a real best-track fix (is_fix), so selection and reporting share the 2023 split.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hurricane_intensity.analog import EmbeddingNet, FrozenFeatures, RegressionNet, StormCappedBatchSampler, knn_predict
from hurricane_intensity.config import PROJECT_ROOT
from hurricane_intensity.pdal import pdal_loss
from hurricane_intensity.storm_windows import StormWindowDataset

DATA = PROJECT_ROOT / "data" / "storm_windows"
SPLIT = PROJECT_ROOT / "configs" / "analog_splits.json"
SEEN = PROJECT_ROOT / "configs" / "analog_splits_seen.csv"
OUT = PROJECT_ROOT / "data" / "exports" / "pdal_v1"
KS = (5, 10, 20, 50)
K_EARLY = 10
EPOCHS, PATIENCE, LR, BATCH, PER_STORM = 30, 5, 4e-4, 128, 4
HISTORY = 2
WORKERS = 6


def splits():
    s = json.loads(SPLIT.read_text())
    return s["train"], s["val"]


def dataset(ids, augment=False):
    return StormWindowDataset(DATA, ids, history=HISTORY, load_positions=(0, 1, 2), augment=augment)


def window_table(ds):
    """Per-window storm, anchor time, intensity at t, is_fix at t."""
    a = ds.anchor
    storms = np.array([w[0] for w in ds.windows])
    t = np.array([ds.meta[s]["time"][w[1]] for s, w in zip(storms, ds.windows)])
    kt = np.array([ds.meta[s]["intensity"][w[1]] for s, w in zip(storms, ds.windows)], dtype=np.float64)
    fix = np.array([ds.meta[s]["is_fix"][w[2][a]] for s, w in zip(storms, ds.windows)], dtype=bool)
    return storms, t, kt, fix


def collate(batch):
    x = torch.stack([b["frames"][:, 0] for b in batch])  # [B, 3, H, W]
    return x


def loader(ds, batch=256):
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=WORKERS, pin_memory=True,
                      collate_fn=collate, persistent_workers=True)


@torch.no_grad()
def forward_all(model, dl, device):
    model.eval()
    out = []
    for x in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            y = model(x.to(device, non_blocking=True))
        out.append(y.float().cpu())
    return torch.cat(out).numpy()


def tune_k(val_emb, train_emb, train_kt, train_storms, val_kt):
    """{k: val predictions} and the k with the lowest MAE on all filtered validation windows."""
    preds = {k: knn_predict(torch.as_tensor(val_emb).cuda() if torch.cuda.is_available() else val_emb,
                            train_emb, train_kt, train_storms, k)[0] for k in KS}
    maes = {k: float(np.abs(p - val_kt).mean()) for k, p in preds.items()}
    return preds, maes, min(maes, key=maes.get)


class Indexed(torch.utils.data.Dataset):
    """Training items as (frames [3, H, W], index), the index for the label lookup. Module level so spawned
    DataLoader workers on Windows can pickle it."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]["frames"][:, 0], i


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train(arm, seed):
    run = OUT / f"{arm}_s{seed}"
    if (run / "result.npz").exists():
        print(f"{run} done; skipping")
        return
    run.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, val_ids = splits()
    tr_aug, tr, va = dataset(train_ids, augment=True), dataset(train_ids), dataset(val_ids)
    tr_storms, _, tr_kt, _ = window_table(tr)
    va_storms, va_t, va_kt, va_fix = window_table(va)
    mu, sd = float(tr_kt.mean()), float(tr_kt.std())
    tr_z = torch.tensor((tr_kt - mu) / sd, dtype=torch.float32)
    print(f"{arm} seed {seed}: {len(tr)} train windows ({len(set(tr_storms))} storms), {len(va)} val; "
          f"label mean {mu:.2f} sd {sd:.2f}", flush=True)

    sampler = StormCappedBatchSampler(tr_storms, BATCH, PER_STORM, generator=torch.Generator().manual_seed(seed))
    train_dl = DataLoader(Indexed(tr_aug), batch_sampler=sampler, num_workers=WORKERS, pin_memory=True,
                          persistent_workers=True)
    tr_eval_dl, va_dl = loader(tr), loader(va)
    model = (EmbeddingNet() if arm == "A" else RegressionNet()).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best, best_epoch, wait, log = np.inf, -1, 0, []
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, losses = time.time(), []
        for x, idx in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            z = tr_z[idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
            out = out.float()
            loss = pdal_loss(out, z[:, None], "scale_invariant") if arm == "A" else torch.mean((out - z) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if arm == "A":
            va_emb, tr_emb = forward_all(model, va_dl, device), forward_all(model, tr_eval_dl, device)
            va_pred = knn_predict(va_emb, tr_emb, tr_kt, tr_storms, K_EARLY)[0]
        else:
            va_pred = forward_all(model, va_dl, device) * sd + mu
        mae = float(np.abs(va_pred - va_kt).mean())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": mae, "dropped": sampler.dropped,
               "epoch_s": round(time.time() - t0, 1),
               "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0}
        log.append(row)
        print(row, flush=True)
        if mae < best:
            best, best_epoch, wait = mae, epoch, 0
            torch.save(model.state_dict(), run / "best.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    train_s = time.time() - t_start
    with open(run / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)

    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    save = dict(val_storms=va_storms, val_time=va_t, val_kt=va_kt, val_fix=va_fix, best_epoch=best_epoch,
                train_s=train_s, label_mu=mu, label_sd=sd)
    if arm == "A":
        va_emb, tr_emb = forward_all(model, va_dl, device), forward_all(model, tr_eval_dl, device)
        preds, maes, k = tune_k(va_emb, tr_emb, tr_kt, tr_storms, va_kt)
        save.update(val_emb=va_emb, k=k, val_pred=preds[k], **{f"val_pred_k{kk}": p for kk, p in preds.items()},
                    k_maes=json.dumps(maes))
    else:
        save.update(val_pred=forward_all(model, va_dl, device) * sd + mu)
    np.savez(run / "result.npz", **save)
    print(f"{arm} seed {seed}: best epoch {best_epoch}, val MAE {best:.2f} kt, {train_s / 60:.1f} min", flush=True)


def frozen():
    run = OUT / "C"
    if (run / "result.npz").exists():
        print(f"{run} done; skipping")
        return
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, val_ids = splits()
    tr, va = dataset(train_ids), dataset(val_ids)
    tr_storms, _, tr_kt, _ = window_table(tr)
    va_storms, va_t, va_kt, va_fix = window_table(va)
    model = FrozenFeatures().to(device)
    t0 = time.time()
    tr_emb, va_emb = forward_all(model, loader(tr), device), forward_all(model, loader(va), device)
    preds, maes, k = tune_k(va_emb, tr_emb, tr_kt, tr_storms, va_kt)
    np.savez(run / "result.npz", val_storms=va_storms, val_time=va_t, val_kt=va_kt, val_fix=va_fix, val_emb=va_emb,
             k=k, val_pred=preds[k], k_maes=json.dumps(maes), train_s=time.time() - t0,
             **{f"val_pred_k{kk}": p for kk, p in preds.items()})
    print(f"C: k={k}, val MAEs {maes}")


def ensemble():
    """Arm D: scripts/ensemble_eval.py's preprocessing and members, on the validation storms the ensemble never
    saw (seen_by_resnet_ensemble == 0), joined to the validation windows by (storm, t)."""
    import hashlib

    import pandas as pd
    import torch.nn as nn
    from torchvision.models import resnet18

    from hurricane_intensity.dataset import TCDataset

    run = OUT / "D"
    if (run / "result.npz").exists():
        print(f"{run} done; skipping")
        return
    run.mkdir(parents=True, exist_ok=True)
    members = [PROJECT_ROOT / f"models/storms/expanded/seed{s}.pt" for s in range(10)]
    expected = [ln for ln in (PROJECT_ROOT / "results/ensemble/members_sha256.txt").read_text().splitlines() if ln.strip()]
    got = [f"{hashlib.sha256(m.read_bytes()).hexdigest()}  {m.relative_to(PROJECT_ROOT).as_posix()}" for m in members]
    if got != expected:
        raise SystemExit("ensemble members do not match results/ensemble/members_sha256.txt")
    seen = pd.read_csv(SEEN)
    unseen = sorted(seen.query("split == 'val' and seen_by_resnet_ensemble == 0")["atcf_id"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TCDataset(str(PROJECT_ROOT / "data" / "processed"), unseen)
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    P = []
    for c in members:
        model.load_state_dict(torch.load(c, map_location=device))
        model.eval()
        with torch.no_grad():
            P.append(torch.cat([model(x.to(device)).squeeze(1).cpu() for x, _ in dl]).numpy() * 185.0)
    pred = np.stack(P).mean(0)
    key = {(a, pd.Timestamp(t)): p for a, t, p in zip(ds.df["atcf_id"], ds.df["sample_time"], pred)}
    _, val_ids = splits()
    va = dataset(val_ids)
    va_storms, va_t, va_kt, va_fix = window_table(va)
    d_pred = np.array([key.get((s, pd.Timestamp(t)), np.nan) if s in unseen else np.nan
                       for s, t in zip(va_storms, va_t)])
    missing = int(np.isnan(d_pred[np.isin(va_storms, unseen)]).sum())
    if missing:
        raise SystemExit(f"{missing} unseen validation windows have no ensemble prediction")
    np.savez(run / "result.npz", val_storms=va_storms, val_time=va_t, val_kt=va_kt, val_fix=va_fix, val_pred=d_pred,
             unseen=np.array(unseen))
    print(f"D: {np.isfinite(d_pred).sum()} windows from {len(unseen)} unseen validation storms")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=("A", "B"), required=True)
    t.add_argument("--seed", type=int, required=True)
    sub.add_parser("frozen")
    sub.add_parser("ensemble")
    sub.add_parser("report")
    a = p.parse_args()
    if a.cmd == "train":
        train(a.arm, a.seed)
    elif a.cmd == "frozen":
        frozen()
    elif a.cmd == "ensemble":
        ensemble()
    else:
        import pdal_v1_report
        pdal_v1_report.main()
