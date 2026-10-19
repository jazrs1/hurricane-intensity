import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F

BT_MIN = 180.0
BT_MAX = 320.0
SIZE = 224

# The six 2025 Atlantic test storms, spent by the final test (logs/decisions.md, 2026-10-16): TCDataset never
# loads them. Kept in code, never read from a file, the environment or the cwd (check_nan.py builds TCDataset
# from an empty temporary cwd). Other 2025 storms (AL012025, the East Pacific H-EP set) load as usual.
SPENT_2025 = frozenset({"AL042025", "AL052025", "AL072025", "AL082025", "AL092025", "AL132025"})
# A storm ID is exactly two capital letters and six ASCII digits, with no normalisation.
STORM_ID = re.compile(r"[A-Z]{2}[0-9]{6}")


def crop_storm(processed_root, crop_path):
    """The storm directory crop_path resolves to under processed_root, upper-cased; None if it is outside it."""
    return resolved_crop_storm(Path(processed_root).resolve(), crop_path)


def resolved_crop_storm(root, crop_path):
    """crop_storm for a processed_root that is already resolved."""
    try:
        parts = (root / crop_path).resolve().relative_to(root).parts
    except ValueError:
        return None
    return parts[0].rstrip(". ").upper() if len(parts) > 1 else None

# Circular mask: everything outside the circle inscribed in the 224 x 224 frame is set to 300 K, which reads
# as clear tropical ocean (0 would be 180 K, the coldest cloud top the scale allows).
MASK_FILL = (300.0 - BT_MIN) / (BT_MAX - BT_MIN)
_c = torch.arange(SIZE, dtype=torch.float32) + 0.5 - SIZE / 2
OUTSIDE_CIRCLE = (_c[None, :] ** 2 + _c[:, None] ** 2) > (SIZE / 2) ** 2


def resample(bt, angle=0.0):
    """Native crop (1, 1, H, W) -> (1, 1, 224, 224) in one bilinear resampling, rotated by `angle` radians.

    Rotation is applied in the sampling grid's normalized coordinates, which span the crop's square ground box,
    so it is a rotation on the ground even though native pixels are not square. It is a proper rotation
    (determinant +1), never a reflection. The grid samples the same points as F.interpolate(bilinear,
    align_corners=False) at angle 0 but is not bit-identical (float rounding, <= 0.002 K over every train,
    val and H6 frame), so angle 0 goes through F.interpolate itself and matches the unmasked resize exactly.
    """
    if angle == 0.0:
        return F.interpolate(bt, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    theta = torch.tensor([[[c, -s, 0.0], [s, c, 0.0]]], dtype=bt.dtype)
    grid = F.affine_grid(theta, (1, 1, SIZE, SIZE), align_corners=False)
    return F.grid_sample(bt, grid, mode="bilinear", padding_mode="border", align_corners=False)


class TCDataset(Dataset):
    def __init__(self, processed_root, storm_ids, circular_mask=False, rotate=False, generator=None):
        """circular_mask: fill outside the inscribed circle with MASK_FILL. rotate: rotate each sample by a
        uniform random angle (training augmentation; requires circular_mask, so no frame corner sampled from
        outside the crop is ever visible). With both off, preprocessing is exactly as before.

        Raises ValueError, before anything is read, for a storm ID that is malformed or spent (SPENT_2025),
        and after reading a manifest whose rows belong to another storm or point outside its directory.
        The block is path-based: it does not detect spent crop content copied or hard-linked under another
        storm's directory. Keeping spent crops out of other storms' directories is a process rule."""
        ids = list(storm_ids)
        for storm_id in ids:
            if not isinstance(storm_id, str) or not STORM_ID.fullmatch(storm_id):
                raise ValueError(f"malformed storm ID {storm_id!r}: expected two capital letters and six digits")
            if storm_id in SPENT_2025:
                raise ValueError(f"{storm_id} is a spent 2025 test storm and is never loaded")
        if rotate and not circular_mask:
            raise ValueError("rotate requires circular_mask")
        self.circular_mask = circular_mask
        self.rotate = rotate
        self.generator = generator
        self.processed_root = processed_root
        self.storm_ids = ids
        self._resolved_root = (str(processed_root), Path(processed_root).resolve())
        self._crop_storms = {}
        frames = []
        for storm_id in ids:
            manifest_path = Path(processed_root) / storm_id / "manifest.csv"
            df = pd.read_csv(manifest_path)
            if "atcf_id" not in df or (df["atcf_id"].astype(str) != storm_id).any():
                raise ValueError(f"{manifest_path}: atcf_id missing or not {storm_id} in some rows")
            stray = [p for p in df["crop_path"].dropna() if self._crop_storm(p) != storm_id]
            if stray:
                raise ValueError(f"{manifest_path}: crop_path outside {storm_id}/ in {len(stray)} rows, e.g. {stray[0]}")
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        df = df[df["crop_path"].notna()]

        df = pd.concat(frames, ignore_index=True)
        df = df[df["crop_path"].notna()]

        excluded_file = Path("excluded_crops.txt")
        if excluded_file.exists():
            excluded = set(excluded_file.read_text().split())
            df = df[~df["crop_path"].isin(excluded)]
        print(f"after exclusion: {len(df)}")


        exists = df["crop_path"].apply(lambda p: (Path(processed_root) / p).exists())
        self.df = df[exists].reset_index(drop=True)
    def __len__(self):
        return len(self.df)

    def _crop_storm(self, crop_path):
        """crop_storm under self.processed_root, memoised per (processed_root, crop_path) so each crop is resolved
        once. Only the resolution is cached: __getitem__ still checks the result on every item, whatever df holds."""
        root = str(self.processed_root)
        if self._resolved_root[0] != root:
            self._resolved_root = (root, Path(root).resolve())
        key = (root, crop_path)
        if key not in self._crop_storms:
            self._crop_storms[key] = resolved_crop_storm(self._resolved_root[1], crop_path)
        return self._crop_storms[key]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        storm = self._crop_storm(row["crop_path"])
        if storm is None or storm in SPENT_2025:
            raise ValueError(f"{row['crop_path']} is a spent 2025 crop or outside {self.processed_root}")
        crop_path = Path(self.processed_root) / row["crop_path"]
        data = np.load(crop_path)
        bt = data["bt"]
        bt = np.clip(bt, BT_MIN, BT_MAX)
        bt = (bt - BT_MIN) / (BT_MAX - BT_MIN)
        bt = torch.from_numpy(bt).float()
        bt = bt.unsqueeze(0).unsqueeze(0)
        if self.circular_mask:
            angle = 0.0
            if self.rotate:
                angle = 2 * np.pi * torch.rand(1, generator=self.generator).item()
            bt = resample(bt, angle)
            bt[0, 0][OUTSIDE_CIRCLE] = MASK_FILL
        else:
            bt = F.interpolate(bt, size=(224, 224), mode = "bilinear", align_corners=False)
        bt = bt.squeeze(0).repeat(3, 1, 1)
        label = torch.tensor(row["wind_kt"] / 185.0, dtype=torch.float32)
        return bt, label 

        
        

if __name__ == "__main__":
    ds = TCDataset("data/processed", ["AL142018", "AL142024", "AL052019", "AL132020", "AL192020", "AL062020", "AL092021", "AL142021", "AL102023", "AL162023", "AL012024", "AL042024"])
    img, label = ds[91]
    print(label.item() * 185.0) 
