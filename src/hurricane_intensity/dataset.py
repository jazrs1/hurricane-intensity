import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F

BT_MIN = 180.0
BT_MAX = 320.0
SIZE = 224

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
        outside the crop is ever visible). With both off, preprocessing is exactly as before."""
        if rotate and not circular_mask:
            raise ValueError("rotate requires circular_mask")
        self.circular_mask = circular_mask
        self.rotate = rotate
        self.generator = generator
        self.processed_root = processed_root
        self.storm_ids = storm_ids
        frames = []
        for storm_id in storm_ids:
            manifest_path = Path(processed_root) / storm_id / "manifest.csv"
            df = pd.read_csv(manifest_path)
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
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
