import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F

BT_MIN = 180.0
BT_MAX = 320.0


class TCDataset(Dataset):
    def __init__(self, processed_root, storm_ids):
        self.processed_root = processed_root
        self.storm_ids = storm_ids
        frames = []
        for storm_id in storm_ids:
            manifest_path = Path(processed_root) / storm_id / "manifest.csv"
            df = pd.read_csv(manifest_path)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
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
        bt = F.interpolate(bt, size=(224, 224), mode = "bilinear", align_corners=False)
        bt = bt.squeeze(0).repeat(3, 1, 1)
        label = torch.tensor(row["wind_kt"] / 185.0, dtype=torch.float32)
        return bt, label 

        
        

if __name__ == "__main__":
    ds = TCDataset("data/processed", ["AL142018", "AL142024", "AL052019", "AL132020", "AL192020", "AL062020", "AL092021", "AL142021", "AL102023", "AL162023", "AL012024", "AL042024"])
    img, label = ds[91]
    print(label.item() * 185.0) 
