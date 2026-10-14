"""Pre-launch checks for the rotation experiment. No training.
1. angle-0 resample() vs the current F.interpolate resize, on every train / val / H6 frame.
2. resample() rotation is proper (matches torch.rot90 one way or the other at 90 deg; a reflection matches neither).
3. Renders a few masked and rotated frames for a visual check.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
from hurricane_intensity.dataset import BT_MAX, BT_MIN, MASK_FILL, OUTSIDE_CIRCLE, TCDataset, resample  # noqa: E402
from hurricane_intensity.splits import load_splits  # noqa: E402

tr, va, ho = load_splits(train_list="storms_train_baseline.txt", holdout_list="storms_holdout_h6.txt")
ds = TCDataset("data/processed", tr + va + ho)
worst, exact, n = 0.0, 0, len(ds)
for i in range(n):
    bt = np.load(Path("data/processed") / ds.df.crop_path[i])["bt"]
    bt = torch.from_numpy((np.clip(bt, BT_MIN, BT_MAX) - BT_MIN) / (BT_MAX - BT_MIN)).float()[None, None]
    a = F.interpolate(bt, size=(224, 224), mode="bilinear", align_corners=False)
    b = resample(bt, 0.0)
    d = (a - b).abs().max().item()
    worst = max(worst, d)
    exact += int(d == 0.0)
print(f"1. angle-0 resample vs F.interpolate over {n} frames (train+val+H6): max |diff| {worst:.3e} normalized "
      f"= {worst * (BT_MAX - BT_MIN):.3e} K; bit-identical frames {exact}/{n}")

# default dataset path unchanged: unmasked TCDataset still uses F.interpolate
x0 = TCDataset("data/processed", tr[:1])[5][0]
bt = np.load(Path("data/processed") / TCDataset("data/processed", tr[:1]).df.crop_path[5])["bt"]
bt = torch.from_numpy((np.clip(bt, BT_MIN, BT_MAX) - BT_MIN) / (BT_MAX - BT_MIN)).float()[None, None]
print("   default (unmasked) dataset output equals the old path:", torch.equal(x0, F.interpolate(bt, size=(224, 224), mode="bilinear", align_corners=False).squeeze(0).repeat(3, 1, 1)))

# 2. chirality: an asymmetric 224 x 224 pattern (an 'L' plus a dot), rotated 90 degrees
img = torch.zeros(1, 1, 224, 224)
img[0, 0, 40:180, 60:80] = 1.0
img[0, 0, 160:180, 60:150] = 1.0
img[0, 0, 50:60, 140:150] = 0.5
r = resample(img, np.pi / 2)
inner = ~OUTSIDE_CIRCLE
m_ccw = (r[0, 0] - torch.rot90(img[0, 0], 1, (0, 1))).abs()[inner].max().item()
m_cw = (r[0, 0] - torch.rot90(img[0, 0], -1, (0, 1))).abs()[inner].max().item()
m_flip = min((r[0, 0] - img[0, 0].flip(d)).abs()[inner].max().item() for d in (0, 1))
m_tr = (r[0, 0] - img[0, 0].T).abs()[inner].max().item()
print(f"2. 90-deg rotation vs rot90(+1) {m_ccw:.2e}, rot90(-1) {m_cw:.2e}; vs flips {m_flip:.2e}, transpose {m_tr:.2e} "
      f"-> proper rotation: {min(m_ccw, m_cw) < 1e-5 and m_flip > 0.1 and m_tr > 0.1}")
th = np.array([[np.cos(1.0), -np.sin(1.0)], [np.sin(1.0), np.cos(1.0)]])
print(f"   determinant of the rotation matrix: {np.linalg.det(th):+.6f}")

# 3. render
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
g = torch.Generator().manual_seed(0)
masked = TCDataset("data/processed", ["AL142018"], circular_mask=True)
rot = TCDataset("data/processed", ["AL142018"], circular_mask=True, rotate=True, generator=g)
plain = TCDataset("data/processed", ["AL142018"])
k = int((masked.df.wind_kt >= 135).to_numpy().argmax())
fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
for a_, (t, im) in zip(ax, [("unmasked (current)", plain[k][0]), ("masked control", masked[k][0])] + [(f"rotated #{j + 1}", rot[k][0]) for j in range(3)]):
    a_.imshow(im[0].numpy() * (BT_MAX - BT_MIN) + BT_MIN, cmap="gray_r", vmin=180, vmax=320)
    a_.set_title(t, fontsize=9)
    a_.axis("off")
fig.suptitle(f"Michael {masked.df.sample_time[k]} {masked.df.wind_kt[k]:.0f} kt; fill {MASK_FILL * (BT_MAX - BT_MIN) + BT_MIN:.0f} K outside the circle")
fig.savefig(OUT / "rotation_check.png", dpi=80, bbox_inches="tight")
print("3. rendered", OUT / "rotation_check.png", "| fill normalized", round(MASK_FILL, 4), "| masked pixels", int(OUTSIDE_CIRCLE.sum()), "of", 224 * 224)
