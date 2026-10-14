import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from hurricane_intensity.dataset import TCDataset
from hurricane_intensity.splits import HOLDOUT_LIST, TRAIN_LIST, load_splits
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="best_model.pt")
parser.add_argument("--holdout-list", default=str(HOLDOUT_LIST), help="storm list to evaluate on")
parser.add_argument("--train-list", default=str(TRAIN_LIST),
                    help="list the checkpoint was trained on; evaluation refuses any held-out storm that appears in it")
parser.add_argument("--out", default="results/heldout_predictions.csv", help="per-sample predictions CSV")
parser.add_argument("--circular-mask", action="store_true", help="fill outside the inscribed circle with 300 K (no rotation)")
args = parser.parse_args()

HOLDOUT_STORMS = load_splits(train_list=args.train_list, holdout_list=args.holdout_list)[2]
PROCESSED_ROOT = "data/processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"checkpoint: {args.checkpoint}")
print(f"held-out storms: {HOLDOUT_STORMS}")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(args.checkpoint))
model = model.to(device)
model.eval()

ds = TCDataset(PROCESSED_ROOT, HOLDOUT_STORMS, circular_mask=args.circular_mask)
print(f"circular mask: {args.circular_mask}")
loader = DataLoader(ds, batch_size=32, shuffle=False)

preds, truths = [], []
with torch.no_grad():
    for images, labels in loader:
        out = model(images.to(device)).squeeze(1).cpu()
        preds.append(out)
        truths.append(labels)

preds = torch.cat(preds).numpy() * 185.0
truths = torch.cat(truths).numpy() * 185.0
errors = np.abs(preds - truths)

is_fix = ~ds.df["is_interpolated"].to_numpy().astype(bool)

print(f"held-out samples: {len(ds)}")
print(f"overall MAE:      {errors.mean():.2f} kt")
print(f"true fixes only:  {errors[is_fix].mean():.2f} kt  (n={is_fix.sum()})")
print()

# Contiguous, lower bound inclusive, upper bound exclusive: interpolated
# labels are fractional (e.g. 33.5 kt) and must land in exactly one band.
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"),
         (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, float("inf"), "Cat 5")]

for lo, hi, name in BANDS:
    mask = (truths >= lo) & (truths < hi)
    if mask.sum() > 0:
        print(f"{name:6s} {mask.sum():4d} samples   MAE {errors[mask].mean():6.2f} kt   bias {(preds[mask] - truths[mask]).mean():+6.2f} kt")
cat1 = (truths >= 64) & (truths < 83)
sub = ds.df[cat1].copy()
sub["error"] = errors[cat1]
print(sub.groupby("storm_name")["error"].agg(["count", "mean"]))


out = ds.df.copy()
out["pred"] = preds
out["truth"] = truths
out["signed_error"] = preds - truths
out["abs_error"] = errors

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out, index=False)
print(f"\nwrote {args.out}  ({len(out)} rows)")