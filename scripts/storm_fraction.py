"""Same-storm neighbour fraction for one checkpoint: the rotation experiment's pre-registered secondary metric.

Embeds the 1307 training frames of the nine-storm list (512-d avgpool features; optionally circular-masked,
never rotated), and for each frame takes its 20 nearest training frames by Euclidean distance, excluding itself
and every same-storm frame within 6 hours. Metric = mean over frames of the fraction from the same storm.
Also records the +-24 h version and the by-band breakdown (exploratory). Chance levels depend only on labels and
timing, not on the checkpoint, so they are the same for every run (see storm_clustering.py).
Inference only. Run from the repo root.

  python scripts/storm_fraction.py --checkpoint X [--circular-mask] --out Y.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hurricane_intensity.dataset import TCDataset  # noqa: E402
from hurricane_intensity.splits import load_splits, load_storm_list  # noqa: E402

K = 20
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--circular-mask", action="store_true")
parser.add_argument("--out", required=True)
parser.add_argument("--storms-list", default=None,
                    help="storm list whose frames make up the frame set (neighbours are searched among them); "
                         "default: the original nine training storms (storms_train_baseline.txt minus validation)")
args = parser.parse_args()

if args.storms_list:
    frame_ids = load_storm_list(args.storms_list)
else:
    frame_ids = load_splits(train_list=REPO / "storms_train_baseline.txt")[0]
ds = TCDataset("data/processed", frame_ids, circular_mask=args.circular_mask)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(args.checkpoint))
model.fc = nn.Identity()
model = model.to(device).eval()
feats = []
with torch.no_grad():
    for i in range(0, len(ds), 32):
        feats.append(model(torch.stack([ds[j][0] for j in range(i, min(i + 32, len(ds)))]).to(device)).cpu())
Fe = torch.cat(feats).numpy().astype(np.float64)

df = ds.df.reset_index(drop=True)
storm = df.atcf_id.to_numpy()
ts = pd.to_datetime(df.sample_time)
t = ((ts - ts.min()).dt.total_seconds() / 3600).to_numpy()
y = df.wind_kt.to_numpy(float)
band = pd.cut(y, [b[0] for b in BANDS] + [np.inf], right=False, labels=[b[2] for b in BANDS]).astype(str)
D = (Fe ** 2).sum(1)[:, None] + (Fe ** 2).sum(1)[None, :] - 2 * Fe @ Fe.T
same = storm[:, None] == storm[None, :]


def frac(window_h):
    d = np.where(same & (np.abs(t[:, None] - t[None, :]) <= window_h), np.inf, D)
    idx = np.argsort(d, axis=1)[:, :K]
    return same[np.arange(len(df))[:, None], idx].mean(1)


f6, f24 = frac(6), frac(24)
out = {"checkpoint": str(args.checkpoint), "circular_mask": args.circular_mask, "n_frames": len(df),
       "frame_set": args.storms_list or "original nine training storms", "n_storms": int(df.atcf_id.nunique()),
       "same_storm_frac_6h": float(f6.mean()), "same_storm_frac_24h": float(f24.mean()),
       "by_band_6h": {b: float(f6[band == b].mean()) for *_, b in BANDS}, "by_band_24h": {b: float(f24[band == b].mean()) for *_, b in BANDS}}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=1))
print(f"same-storm fraction (k={K}, +-6 h excluded): {out['same_storm_frac_6h']:.4f}; +-24 h: {out['same_storm_frac_24h']:.4f} -> {args.out}")
