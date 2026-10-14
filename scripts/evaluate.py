import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from hurricane_intensity.dataset import TCDataset
from pathlib import Path

HOLDOUT_STORMS = ["AL092022", "AL172022", "AL032021"]
PROCESSED_ROOT = "data/processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("best_model.pt"))
model = model.to(device)
model.eval()

ds = TCDataset(PROCESSED_ROOT, HOLDOUT_STORMS)
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

Path("results").mkdir(exist_ok=True)
out.to_csv("results/heldout_predictions.csv", index=False)
print(f"\nwrote results/heldout_predictions.csv  ({len(out)} rows)")