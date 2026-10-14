"""Read-only: do the crops on disk match what was recorded around the 9.00 kt run?"""
import datetime as dt
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(sys.argv[1])
ROOT = REPO / "data" / "processed"
TRAIN = ["AL142018", "AL142024", "AL052019", "AL062020", "AL092021", "AL142021", "AL102023", "AL012024", "AL042024"]
VAL = ["AL192020", "AL162023", "AL132020"]
HOLD = ["AL092022", "AL172022", "AL032021"]
NEW = ["AL022024", "AL132023", "AL182021"]
CKPT_TIME = dt.datetime.fromtimestamp((REPO / "best_model.pt").stat().st_mtime)
print("best_model.pt last written:", CKPT_TIME)

print("\n== 1. every crop and manifest vs the checkpoint time ==")
rows = []
for group, ids in (("train", TRAIN), ("val", VAL), ("holdout", HOLD), ("added later", NEW)):
    for s in ids:
        man = pd.read_csv(ROOT / s / "manifest.csv")
        have = man[man.crop_path.notna()]
        files = sorted((ROOT / s / "crops").glob("*.npz"))
        mt = np.array([f.stat().st_mtime for f in files])
        on_disk = {f"{s}\\crops\\{f.name}" for f in files}
        listed = set(have.crop_path)
        # shape recorded in the manifest when the crop was cut vs the array in the file now
        shape_ok, keys, dtypes = 0, set(), set()
        order = []
        for r in have.itertuples():
            with np.load(ROOT / r.crop_path) as z:
                keys.add(tuple(sorted(z.files)))
                bt = z["bt"]
                dtypes.add(str(bt.dtype))
                shape_ok += f"{bt.shape[0]}x{bt.shape[1]}" == r.bt_shape
            order.append((ROOT / r.crop_path).stat().st_mtime)
        rows.append(dict(
            group=group, storm=s, crops=len(files),
            newest_crop=dt.datetime.fromtimestamp(mt.max()).strftime("%m-%d %H:%M"),
            manifest=dt.datetime.fromtimestamp((ROOT / s / "manifest.csv").stat().st_mtime).strftime("%m-%d %H:%M"),
            after_ckpt=int((mt > CKPT_TIME.timestamp()).sum()),
            files_eq_manifest=on_disk == listed,
            shape_eq_manifest=f"{shape_ok}/{len(have)}",
            written_in_manifest_order=bool(np.all(np.diff(order) >= 0)),
            manifest_after_last_crop_s=round((ROOT / s / "manifest.csv").stat().st_mtime - mt.max(), 1),
            npz_keys=";".join(",".join(k) for k in keys), dtype=";".join(dtypes),
        ))
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 2. held-out: manifest columns recorded in the committed predictions CSV vs manifests on disk ==")
committed = pd.read_csv(io.StringIO(subprocess.run(
    ["git", "show", "ed0401c:results/heldout_predictions.csv"], capture_output=True, text=True, check=True).stdout))
now = pd.concat([pd.read_csv(ROOT / s / "manifest.csv") for s in HOLD], ignore_index=True)
now = now[now.crop_path.notna()].reset_index(drop=True)
cols = [c for c in now.columns if c in committed.columns]
print("rows committed / on disk:", len(committed), len(now), "| shared columns:", cols)
for c in cols:
    a, b = committed[c], now[c]
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        same = bool(np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True))
    else:
        same = bool(((a.astype(str) == b.astype(str)) | (a.isna() & b.isna())).all())
    print(f"  {c:16s} identical: {same}")

print("\n== 3. held-out pixels: site/frames/ian/*.png (exported from the model input) vs the crops on disk now ==")
sys.path.insert(0, str(REPO / "src"))
import os
os.chdir(REPO)
from hurricane_intensity.dataset import TCDataset
ds = TCDataset(str(ROOT), ["AL092022"])
t = pd.to_datetime(ds.df.sample_time, utc=True).to_numpy()
order = np.argsort(t, kind="stable")
pngs = sorted((REPO / "site" / "frames" / "ian").glob("*.png"))
png_time = dt.datetime.fromtimestamp(max(p.stat().st_mtime for p in pngs))
maxdiff, nexact = 0, 0
for k, p in zip(order, pngs):
    img, _ = ds[int(k)]
    gray = np.clip(img[0].numpy() * 255.0, 0, 255).astype(np.uint8)
    d = np.abs(gray.astype(int) - np.array(Image.open(p)).astype(int)).max()
    maxdiff = max(maxdiff, int(d)); nexact += d == 0
print(f"PNG frames {len(pngs)} (written {png_time:%Y-%m-%d %H:%M}), dataset Ian frames {len(order)}, pixel-identical {nexact}, max abs diff {maxdiff} grey levels")

print("\n== 4. training storm: Grad-CAM arrays committed for Michael vs recomputed from the crops on disk ==")
sys.path.insert(0, str(REPO / "scripts"))
import torch
import gradcam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = gradcam.load_model(device)
dsm = TCDataset(str(ROOT), ["AL142018"])
for f in sorted((REPO / "results" / "gradcam").glob("AL142018_*.npy")):
    stamp = f.stem.split("_")[1]
    ts = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}"
    idx = gradcam.resolve_frame(dsm, "AL142018", ts)
    cam = gradcam.compute_gradcam(model, dsm[idx][0], device)
    old = np.load(f)
    print(f"  {f.name}: saved {dt.datetime.fromtimestamp(f.stat().st_mtime):%Y-%m-%d}, shapes {old.shape} {np.asarray(cam).shape}, max abs diff {np.abs(old - np.asarray(cam)).max():.2e}")

print("\n== 5. facts logged in decisions.md when Michael was built ==")
m = pd.read_csv(ROOT / "AL142018" / "manifest.csv")
print(f"Michael: {len(m)} samples, {int((~m.is_interpolated.astype(bool)).sum())} fixes, {int(m.is_interpolated.astype(bool).sum())} interpolated, "
      f"{int(m.crop_path.isna().sum())} skips, median |offset| {m.offset_min.abs().median():.2f} min  (logged: 122 / 22 / 100 / 0 / ~0.6)")
print("  crops of 274x309:", int((m.bt_shape == "274x309").sum()), "| first shapes:", list(m.bt_shape.head(3)))

print("\n== 6. same crop geometry across build dates? pixel size per degree of half-width implied by shape vs latitude ==")
allm = pd.concat([pd.read_csv(ROOT / s / "manifest.csv").assign(storm=s, grp=g)
                  for g, ids in (("Aug 18-19 (Michael, Milton)", TRAIN[:2]), ("Aug 24-25 (other 13)", TRAIN[2:] + VAL + HOLD), ("Sep 13 (Beryl, Lee, Sam)", NEW))
                  for s in ids], ignore_index=True)
allm = allm[allm.crop_path.notna()]
hw = allm.bt_shape.str.split("x", expand=True).astype(int)
allm["rows"], allm["cols"] = hw[0], hw[1]
allm["latbin"] = (allm.lat.abs() // 5 * 5).astype(int)
print(allm.groupby(["latbin", "grp"]).agg(n=("rows", "size"), rows_med=("rows", "median"), cols_med=("cols", "median"),
                                           offset_abs_med=("offset_min", lambda s: s.abs().median())).unstack("grp").to_string())
print("\nsource_key product/platform per group:")
allm["prod"] = allm.source_key.str.extract(r"^(ABI-[^/]+)/")[0] + " " + allm.source_key.str.extract(r"_(G\d\d)_")[0] + " C" + allm.source_key.str.extract(r"C(\d\d)_G")[0]
print(allm.groupby(["grp", "prod"]).size().to_string())
