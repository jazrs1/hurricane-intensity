"""Evaluate an ensemble of checkpoints: average their predictions per frame.

Default members are the ten expanded-arm checkpoints of the more-distinct-storms experiment
(models/storms/expanded/seed0-9.pt). Same dataset, ordering and train/held-out refusal as
scripts/evaluate.py. Writes one CSV in evaluate.py's columns, where `pred` is the member mean,
plus each member's prediction (pred_<member>), and a SHA-256 file for the members.
Never writes best_model.pt, models/ or site/.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from hurricane_intensity.dataset import TCDataset
from hurricane_intensity.splits import load_splits

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", nargs="+", default=[f"models/storms/expanded/seed{s}.pt" for s in range(10)])
parser.add_argument("--train-list", default="storms_train_expanded.txt",
                    help="list the members were trained on; evaluation refuses any held-out storm that appears in it")
parser.add_argument("--holdout-list", default="storms_holdout_h9.txt", help="storm list to evaluate on")
parser.add_argument("--processed-root", default="data/processed")
parser.add_argument("--out", required=True, help="per-sample predictions CSV")
parser.add_argument("--sha-out", required=True, help="SHA-256 of each member checkpoint")
parser.add_argument("--expect-sha", default=None, help="refuse to run unless the members match this SHA-256 file")
args = parser.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


members = [Path(c) for c in args.checkpoints]
sha_lines = [f"{sha256(c)} *{c.as_posix()}" for c in members]
if args.expect_sha:
    expected = Path(args.expect_sha).read_text().split("\n")
    if [l for l in expected if l.strip()] != sha_lines:
        raise SystemExit(f"members do not match {args.expect_sha}")
names = [c.stem for c in members]
assert len(set(names)) == len(names), "member names (file stems) must be unique"

HOLDOUT_STORMS = load_splits(train_list=args.train_list, holdout_list=args.holdout_list)[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"members: {len(members)}")
for line in sha_lines:
    print(f"  {line}")
print(f"held-out storms: {HOLDOUT_STORMS}")

ds = TCDataset(args.processed_root, HOLDOUT_STORMS)
loader = DataLoader(ds, batch_size=32, shuffle=False)

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)
member_preds = []
for c in members:
    model.load_state_dict(torch.load(c, map_location=device))
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds.append(model(images.to(device)).squeeze(1).cpu())
            truths.append(labels)
    member_preds.append(torch.cat(preds).numpy() * 185.0)
    truths = torch.cat(truths).numpy() * 185.0
    print(f"  {c.as_posix()}: MAE {np.abs(member_preds[-1] - truths).mean():.3f} kt")

P = np.stack(member_preds)  # (members, frames)
pred = P.mean(0)
errors = np.abs(pred - truths)
is_fix = ~ds.df["is_interpolated"].to_numpy().astype(bool)
print(f"held-out samples: {len(ds)}")
print(f"ensemble overall MAE: {errors.mean():.3f} kt")
print(f"ensemble fixes-only MAE: {errors[is_fix].mean():.3f} kt  (n={is_fix.sum()})")

out = ds.df.copy()
out["pred"] = pred
out["truth"] = truths
out["signed_error"] = pred - truths
out["abs_error"] = errors
for name, p in zip(names, P):
    out[f"pred_{name}"] = p
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out, index=False)
Path(args.sha_out).write_text("\n".join(sha_lines) + "\n")
print(f"wrote {args.out} ({len(out)} rows) and {args.sha_out}")
