import numpy as np, pandas as pd, torch
from pathlib import Path
edges = [34, 64, 83, 96, 113, 137]
for e in edges:
    rt = (torch.tensor(e / 185.0, dtype=torch.float32).numpy() * 185.0)
    print(e, repr(rt), rt.dtype, "below edge" if rt < e else "ok")
# every distinct held-out label: does the float32 round trip change its half-open band?
root = Path("data/processed")
ex = set(Path("excluded_crops.txt").read_text().split())
def band(w):
    return int(np.searchsorted(edges, w, side="right"))
for name, ids in [("orig3", ["AL092022", "AL172022", "AL032021"]), ("all18", [d.name for d in root.iterdir() if (d / "manifest.csv").exists()])]:
    df = pd.concat([pd.read_csv(root / s / "manifest.csv") for s in ids], ignore_index=True)
    df = df[df.crop_path.notna() & ~df.crop_path.isin(ex)]
    w = df.wind_kt.to_numpy()
    rt = torch.tensor(w / 185.0, dtype=torch.float32).numpy() * 185.0
    moved = np.array([band(a) != band(b) for a, b in zip(w, rt)])
    print(name, "frames:", len(w), "| band changes under float32 round trip:", int(moved.sum()), "| exactly on an edge:", int(np.isin(w, edges).sum()))
    if moved.any():
        print(df[moved][["atcf_id", "sample_time", "wind_kt"]].assign(roundtrip=rt[moved]).to_string())
