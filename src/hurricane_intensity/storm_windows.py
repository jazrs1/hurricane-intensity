"""Windows of consecutive IR frames from one storm, for learned analog forecasting of intensity change.

Storage (built from data/processed by scripts/build_storm_windows.py), one pair of files per storm:
    <ID>.npy  float32 [N, H, W] brightness temperature in kelvin, sorted by time
    <ID>.csv  time (ISO UTC), intensity_kt (HURDAT2 interpolated to the frame), is_fix (0/1)

A window is T frames exactly step_hours apart (+-5 min), all from one storm. The frames do not have to be
adjacent rows: the repo samples storms hourly or 3-hourly, so with step_hours=3 an hourly storm contributes
every third row. A window that would need a frame missing from the file (a gap, e.g. a NaN crop dropped by
excluded_crops.txt) is skipped and counted in `n_skipped`.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hurricane_intensity.dataset import SPENT_2025, STORM_ID

TOLERANCE_S = 5 * 60


def rotate_window(frames, angle):
    """Rotate every frame of a [T, 1, H, W] tensor by the same `angle` (radians), bilinear.

    A proper rotation (determinant +1), never a reflection, so the storm's spin direction is kept. Pixels
    rotated in from outside the frame take the nearest edge value (padding_mode="border", as resample() in
    dataset.py), so no out-of-range value is introduced and [0, 1] inputs stay in [0, 1].
    """
    c, s = math.cos(angle), math.sin(angle)
    theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=frames.dtype).expand(frames.shape[0], 2, 3)
    grid = F.affine_grid(theta, list(frames.shape), align_corners=False)
    return F.grid_sample(frames, grid, mode="bilinear", padding_mode="border", align_corners=False)


def delta_24h(intensities, step_hours):
    """Intensity 24 h after the first frame minus the first frame (the dI24 label), in the input's units.

    `intensities` is [T] or [B, T]; returns a scalar tensor or [B]. 24 must be a whole number of steps
    that fits in the window, else ValueError.
    """
    k = 24 / step_hours
    if k != int(k):
        raise ValueError(f"24 h is not a whole number of {step_hours} h steps")
    k = int(k)
    if k >= intensities.shape[-1]:
        raise ValueError(f"window of {intensities.shape[-1]} frames does not reach 24 h at {step_hours} h steps")
    return intensities[..., k] - intensities[..., 0]


class StormWindowDataset(Dataset):
    def __init__(self, data_dir, storm_ids, T=9, step_hours=3, stride=1, augment=False,
                 tb_min=180.0, tb_max=320.0, generator=None):
        """Index every valid (storm_id, start) window. stride steps over start rows of each storm's file.

        n_skipped counts start rows whose window would end inside the storm's time range (start time +
        (T-1)*step_hours <= its last frame, within tolerance) but is missing at least one frame; start rows
        too close to the end of the storm are not windows at all and are not counted. generator (optional)
        seeds the rotation angle, for reproducible tests.
        """
        ids = list(storm_ids)
        for storm_id in ids:
            if not isinstance(storm_id, str) or not STORM_ID.fullmatch(storm_id):
                raise ValueError(f"malformed storm ID {storm_id!r}: expected two capital letters and six digits")
            if storm_id in SPENT_2025:
                raise ValueError(f"{storm_id} is a spent 2025 test storm and is never loaded")
        if len(set(ids)) != len(ids):
            raise ValueError("storm_ids lists a storm more than once")
        if T < 1 or stride < 1 or step_hours <= 0:
            raise ValueError("T and stride must be >= 1 and step_hours > 0")
        self.data_dir = Path(data_dir)
        self.storm_ids = ids
        self.T, self.step_hours, self.stride, self.augment = T, step_hours, stride, augment
        self.tb_min, self.tb_max = float(tb_min), float(tb_max)
        self.generator = generator
        self._arrays = {}

        step_s = step_hours * 3600
        self.meta = {}
        self.windows = []  # (storm_id, start row, frame rows [T])
        self.n_skipped = 0
        self.skipped_per_storm = {}
        for storm_id in ids:
            df = pd.read_csv(self.data_dir / f"{storm_id}.csv")
            times = pd.to_datetime(df["time"], utc=True)
            secs = (times - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
            if len(secs) and np.any(np.diff(secs) <= 0):
                raise ValueError(f"{storm_id}.csv: times are not strictly increasing")
            n_frames = np.load(self.data_dir / f"{storm_id}.npy", mmap_mode="r").shape[0]
            if n_frames != len(df):
                raise ValueError(f"{storm_id}: {n_frames} frames in .npy but {len(df)} rows in .csv")
            self.meta[storm_id] = {
                "time": times.dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
                "intensity": df["intensity_kt"].to_numpy(np.float32),
                "is_fix": df["is_fix"].to_numpy().astype(bool),
            }
            skipped = 0
            offsets = np.arange(T) * step_s
            for start in range(0, len(secs), stride):
                targets = secs[start] + offsets
                if targets[-1] > secs[-1] + TOLERANCE_S:
                    continue
                # Nearest frame to each target time (off-grid rows, e.g. a landfall record, can sit nearby).
                hi = np.minimum(np.searchsorted(secs, targets), len(secs) - 1)
                lo = np.maximum(hi - 1, 0)
                rows = np.where(np.abs(secs[lo] - targets) <= np.abs(secs[hi] - targets), lo, hi)
                if (np.abs(secs[rows] - targets) <= TOLERANCE_S).all():
                    self.windows.append((storm_id, start, rows))
                else:
                    skipped += 1
            self.skipped_per_storm[storm_id] = skipped
            self.n_skipped += skipped

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        # Workers re-open the memmaps themselves; a pickled memmap would be copied whole.
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def _array(self, storm_id):
        arr = self._arrays.get(storm_id)
        if arr is None:
            arr = self._arrays[storm_id] = np.load(self.data_dir / f"{storm_id}.npy", mmap_mode="r")
        return arr

    def __getitem__(self, idx):
        storm_id, start, rows = self.windows[idx]
        bt = np.asarray(self._array(storm_id)[rows], dtype=np.float32)  # reads only these T frames
        bt = (np.clip(bt, self.tb_min, self.tb_max) - self.tb_min) / (self.tb_max - self.tb_min)
        frames = torch.from_numpy(bt).unsqueeze(1)  # [T, 1, H, W]
        if self.augment:
            angle = 2 * math.pi * torch.rand(1, generator=self.generator).item()
            frames = rotate_window(frames, angle)
        m = self.meta[storm_id]
        return {
            "frames": frames,
            "intensities": torch.from_numpy(m["intensity"][rows].copy()),
            "is_fix": torch.from_numpy(m["is_fix"][rows].copy()),
            "storm_id": storm_id,
            "start_time": str(m["time"][start]),
        }


def make_loaders(data_dir, train_ids, val_ids, test_ids, batch_size=16, num_workers=4, **dataset_kwargs):
    """(train, val, test) DataLoaders, split by storm. Train is shuffled and augmented; val and test are
    neither. All pin memory. On Windows, call this under `if __name__ == "__main__":` (spawned workers)."""
    sets = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sorted(sets[a] & sets[b])
        if shared:
            raise ValueError(f"{a} and {b} share storms {shared}: splits must be by storm")
    if "augment" in dataset_kwargs:
        raise ValueError("make_loaders sets augment itself (train only); do not pass it")
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    train = DataLoader(StormWindowDataset(data_dir, train_ids, augment=True, **dataset_kwargs),
                       shuffle=True, **common)
    val = DataLoader(StormWindowDataset(data_dir, val_ids, augment=False, **dataset_kwargs),
                     shuffle=False, **common)
    test = DataLoader(StormWindowDataset(data_dir, test_ids, augment=False, **dataset_kwargs),
                      shuffle=False, **common)
    return train, val, test
