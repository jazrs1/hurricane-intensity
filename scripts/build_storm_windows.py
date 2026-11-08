#!/usr/bin/env python
"""Pack each storm's per-frame crops into one array for StormWindowDataset.

    .venv\\Scripts\\python.exe scripts\\build_storm_windows.py                 # every storm under data/processed
    .venv\\Scripts\\python.exe scripts\\build_storm_windows.py --ids AL142018  # selected storms

Reads data/processed/<ID>/manifest.csv and the .npz crops it lists (the build_dataset.py output) and
writes, per storm, to --out_dir (default data/storm_windows/, gitignored like the rest of data/):

    <ID>.npy  float32 [N, 224, 224], brightness temperature in kelvin (not clipped or scaled), sorted by time
    <ID>.csv  time (ISO UTC, the manifest's sample_time), intensity_kt (manifest wind_kt: HURDAT2 linearly
              interpolated to that time), is_fix (1 only for a non-interpolated row at 00/06/12/18 UTC),
              lat, lon (interpolated best-track center; lon wrapped to [-180, 180)), status (HURDAT2 status of
              the most recent best-track record at or before the frame: the manifest leaves status empty on
              interpolated rows, so it is carried forward from the last fix, before any frame is dropped)

Conventions kept from TCDataset (src/hurricane_intensity/dataset.py):
  * rows with no crop, and crops listed in excluded_crops.txt or configs/analog_excluded_crops.txt (frames
    with NaN pixels), are dropped, which
    leaves gaps in the time series; StormWindowDataset skips any window across one.
  * native crops vary in size (e.g. 268x312), so each is resized to 224x224 with the same
    F.interpolate(bilinear, align_corners=False) call TCDataset uses at load time.
  * malformed IDs (e.g. BOGUS99) are skipped and the spent 2025 test storms (SPENT_2025) are never read.

HURDAT2 also has off-synoptic records (landfall, peak intensity) that are real but not 00/06/12/18 UTC;
the spec defines is_fix as a synoptic best-track time, so those rows get is_fix = 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hurricane_intensity.config import PROCESSED_DIR, PROJECT_ROOT
from hurricane_intensity.dataset import SIZE, SPENT_2025, STORM_ID

DEFAULT_OUT = PROJECT_ROOT / "data" / "storm_windows"
# excluded_crops.txt is pinned by the frozen v2 recipe (run_v2_recipe.py guard 4), so NaN frames found in the
# analog storms by check_nan.py are listed separately in configs/analog_excluded_crops.txt; both are applied.
EXCLUDED = (PROJECT_ROOT / "excluded_crops.txt", PROJECT_ROOT / "configs" / "analog_excluded_crops.txt")


def convert_storm(storm_id, processed_root, out_dir, excluded):
    """Write <out_dir>/<storm_id>.npy and .csv; return the frame count."""
    if storm_id in SPENT_2025:
        raise ValueError(f"{storm_id} is a spent 2025 test storm and is never loaded")
    df = pd.read_csv(Path(processed_root) / storm_id / "manifest.csv")
    if (df["atcf_id"].astype(str) != storm_id).any():
        raise ValueError(f"{storm_id}: manifest rows belong to another storm")
    df = df.assign(time=pd.to_datetime(df["sample_time"], utc=True)).sort_values("time")
    status = df["status"].where(df["status"].notna() & (df["status"].astype(str).str.strip() != ""))
    if pd.isna(status.iloc[0]):
        raise ValueError(f"{storm_id}: first manifest row has no status (expected a best-track fix)")
    df = df.assign(status=status.ffill())
    df = df[df["crop_path"].notna() & ~df["crop_path"].isin(excluded)]
    # As TCDataset does, a row whose crop file is gone is dropped (e.g. empty off-disk crops removed by hand).
    df = df[[(Path(processed_root) / str(p).replace("\\", "/")).is_file() for p in df["crop_path"]]]
    if df["time"].duplicated().any():
        raise ValueError(f"{storm_id}: duplicate sample_time rows")

    frames = np.empty((len(df), SIZE, SIZE), dtype=np.float32)
    for i, rel in enumerate(df["crop_path"]):
        rel = Path(str(rel).replace("\\", "/"))
        if rel.parts[0].upper() != storm_id:
            raise ValueError(f"{storm_id}: crop_path {rel} is outside the storm's directory")
        bt = np.load(Path(processed_root) / rel)["bt"].astype(np.float32)
        if np.isnan(bt).any():
            raise ValueError(f"{rel} has NaN pixels but is not in excluded_crops.txt")
        t = torch.from_numpy(bt)[None, None]
        frames[i] = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()

    t = df["time"]
    synoptic = (t.dt.hour % 6 == 0) & (t.dt.minute == 0) & (t.dt.second == 0)
    is_fix = (~df["is_interpolated"].astype(bool)) & synoptic
    out = pd.DataFrame({
        "time": t.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "intensity_kt": df["wind_kt"].astype(float),
        "is_fix": is_fix.astype(int),
        "lat": df["lat"].astype(float).round(4),
        "lon": ((df["lon"].astype(float) + 180.0) % 360.0 - 180.0).round(4),
        "status": df["status"].astype(str),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{storm_id}.npy", frames)
    out.to_csv(out_dir / f"{storm_id}.csv", index=False)
    return len(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--processed_root", type=Path, default=PROCESSED_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ids", nargs="*", help="storm IDs (default: every storm directory with a manifest)")
    args = p.parse_args(argv)

    ids = args.ids or sorted(d.name for d in args.processed_root.iterdir() if (d / "manifest.csv").is_file())
    excluded = {line for f in EXCLUDED if f.exists() for line in f.read_text().split()}
    total = 0
    for storm_id in ids:
        if not STORM_ID.fullmatch(storm_id):
            print(f"skip {storm_id}: not an ATCF storm ID")
            continue
        if storm_id in SPENT_2025:
            print(f"skip {storm_id}: spent 2025 test storm")
            continue
        n = convert_storm(storm_id, args.processed_root, args.out_dir, excluded)
        total += n
        print(f"{storm_id}: {n} frames")
    print(f"wrote {total} frames to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
