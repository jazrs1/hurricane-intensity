import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from hurricane_intensity.dataset import TCDataset
from hurricane_intensity.splits import load_splits

parser = argparse.ArgumentParser()
parser.add_argument("--train-list", required=True,
                    help="storm list to train on; its validation storms (--val-list) are held out of training")
parser.add_argument("--val-list", required=True,
                    help="validation storm list; every storm in it must also be in --train-list")
parser.add_argument("--seed", type=int, default=None,
                    help="seed python, numpy and torch (weight init and shuffle order); default: unseeded")
parser.add_argument("--out", required=True, help="where to save the best-validation checkpoint")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--epoch-dir", default=None,
                    help="also save every epoch's weights here as epoch_NN.pt (diagnostics); default: off")
parser.add_argument("--loss", choices=["l1", "huber", "mse"], default="l1",
                    help="training loss; the checkpoint is always selected on validation MAE, whatever the loss")
parser.add_argument("--circular-mask", action="store_true",
                    help="fill outside the inscribed circle with 300 K, on training and validation frames")
parser.add_argument("--rotate", action="store_true",
                    help="rotate each training sample by a uniform random angle (requires --circular-mask); validation is not rotated")
parser.add_argument("--frame-list", default=None,
                    help="text file of crop paths (as in the manifests); train only on these frames, which must all "
                         "belong to the training storms of --train-list; default: every training frame")
args = parser.parse_args()
if args.rotate and not args.circular_mask:
    parser.error("--rotate requires --circular-mask")

# Labels are wind_kt / 185, so the Huber delta is 10 kt in those units. Fixed in advance; not tuned.
HUBER_DELTA = 10 / 185

TRAIN_STORMS, VAL_STORMS, _ = load_splits(train_list=args.train_list, val_list=args.val_list)

PROCESSED_ROOT = "data/processed"

# Seeded before the model is built, so the new fc layer's init is covered.
# The shuffle gets its own generator, so batch order depends on the seed alone.
# Rotation angles get a third generator, offset from the seed, so they don't repeat the shuffle stream.
shuffle_generator = None
rotation_generator = None
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    rotation_generator = torch.Generator().manual_seed(10_000 + args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
print(f"train storms: {TRAIN_STORMS}")
print(f"val storms:   {VAL_STORMS}")
print(f"seed: {args.seed}  checkpoint: {args.out}  loss: {args.loss}  circular_mask: {args.circular_mask}  rotate: {args.rotate}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
if args.epoch_dir:
    Path(args.epoch_dir).mkdir(parents=True, exist_ok=True)

train_ds = TCDataset(PROCESSED_ROOT, TRAIN_STORMS, circular_mask=args.circular_mask, rotate=args.rotate, generator=rotation_generator)
val_ds = TCDataset(PROCESSED_ROOT, VAL_STORMS, circular_mask=args.circular_mask)
if args.frame_list:
    wanted = [line.strip() for line in Path(args.frame_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(wanted)) != len(wanted):
        raise SystemExit(f"{args.frame_list} lists a frame more than once")
    missing = sorted(set(wanted) - set(train_ds.df["crop_path"]))
    if missing:
        raise SystemExit(f"{len(missing)} frames in {args.frame_list} are not training frames of {args.train_list} (first: {missing[0]})")
    train_ds.df = train_ds.df[train_ds.df["crop_path"].isin(set(wanted))].reset_index(drop=True)
    print(f"frame list: {args.frame_list}: {len(train_ds)} frames from {train_ds.df['atcf_id'].nunique()} storms")
print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, generator=shuffle_generator)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = {"l1": nn.L1Loss(), "huber": nn.HuberLoss(delta=HUBER_DELTA), "mse": nn.MSELoss()}[args.loss]
mae = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

NUM_EPOCHS = args.epochs
best_val_mae = float("inf")

# Logged train/val figures are MAE in kt for every loss (for l1 they are the loss itself, as before),
# and the checkpoint is chosen on validation MAE, so arms with different losses pick epochs by one rule.
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_mae = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images).squeeze(1)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_mae += mae(preds.detach(), labels).item() * images.size(0)

    train_loss /= len(train_ds)
    train_mae /= len(train_ds)

    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).squeeze(1)
            val_loss += criterion(preds, labels).item() * images.size(0)
            val_mae += mae(preds, labels).item() * images.size(0)

    val_loss /= len(val_ds)
    val_mae /= len(val_ds)

    line = f"epoch {epoch+1}: train {train_mae*185:.2f} kt  val {val_mae*185:.2f} kt"
    if args.loss != "l1":
        line += f"  ({args.loss} loss: train {train_loss:.6f}  val {val_loss:.6f})"
    print(line)
    if args.epoch_dir:
        torch.save(model.state_dict(), Path(args.epoch_dir) / f"epoch_{epoch+1:02d}.pt")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), args.out)
        print("  saved")