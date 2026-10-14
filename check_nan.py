import torch
from pathlib import Path
from hurricane_intensity.dataset import TCDataset

# Every storm with crops on disk, whichever list it is in, so regenerating
# the exclusion file can't drop a storm that was added after this was written.
ALL_STORMS = sorted(p.parent.name for p in Path("data/processed").glob("*/manifest.csv"))
Path("excluded_crops.txt").unlink(missing_ok=True)
ds = TCDataset("data/processed", ALL_STORMS)

bad = []
for i in range(len(ds)):
    img, label = ds[i]
    if torch.isnan(img).any() or torch.isnan(label).any():
        bad.append(ds.df.iloc[i]["crop_path"])

Path("excluded_crops.txt").write_text("\n".join(bad))
print(f"{len(bad)} of {len(ds)} excluded")
