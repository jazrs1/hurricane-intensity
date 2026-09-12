import torch
from pathlib import Path
from hurricane_intensity.dataset import TCDataset

ALL_STORMS = ["AL142018", "AL142024", "AL052019", "AL062020", "AL092021",
              "AL142021", "AL102023", "AL012024", "AL042024",
              "AL192020", "AL162023", "AL132020",
              "AL092022", "AL172022", "AL032021"]
Path("excluded_crops.txt").unlink(missing_ok=True)
ds = TCDataset("data/processed", ALL_STORMS)

bad = []
for i in range(len(ds)):
    img, label = ds[i]
    if torch.isnan(img).any() or torch.isnan(label).any():
        bad.append(ds.df.iloc[i]["crop_path"])

Path("excluded_crops.txt").write_text("\n".join(bad))
print(f"{len(bad)} of {len(ds)} excluded")