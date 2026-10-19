import os
import sys
import tempfile
import torch
from pathlib import Path
from hurricane_intensity.dataset import TCDataset

EXCLUDED = Path("excluded_crops.txt")
PROCESSED = Path("data/processed")
# Spent 2025 Atlantic test storms: never loaded; their existing entries are carried forward unchanged.
# Other 2025 storms (East Pacific, the future H-EP set) are scanned like any other storm.
SPENT_2025 = {"AL042025", "AL052025", "AL072025", "AL082025", "AL092025", "AL132025"}


def storm_of(crop_path):
    return crop_path.replace("\\", "/").split("/")[0].strip().upper()


def scan(storms):
    """(crop_path of every frame with a NaN pixel or label, frames scanned), ignoring current exclusions."""
    root = str(PROCESSED.resolve())
    # TCDataset drops the frames listed in ./excluded_crops.txt; build it from an empty directory instead of
    # deleting the file, so every frame is scanned and the file survives any exception.
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as empty:
        os.chdir(empty)
        try:
            ds = TCDataset(root, storms)
        finally:
            os.chdir(cwd)
    bad = []
    for i in range(len(ds)):
        img, label = ds[i]
        if torch.isnan(img).any() or torch.isnan(label).any():
            bad.append(ds.df.iloc[i]["crop_path"])
    return bad, len(ds)


def main():
    # Every storm with crops on disk, whichever list it is in, so regenerating
    # the exclusion file can't drop a storm that was added after this was written.
    all_storms = sorted(p.parent.name for p in PROCESSED.glob("*/manifest.csv"))
    old = [line for line in EXCLUDED.read_text().splitlines() if line.strip()] if EXCLUDED.exists() else []

    found, n = scan([s for s in all_storms if s.strip().upper() not in SPENT_2025])
    kept = [line for line in old if storm_of(line) in SPENT_2025]
    # Same order as before: storm by storm (sorted), frames in manifest order.
    storms = sorted({storm_of(p) for p in found + kept})
    bad = [p for s in storms for p in kept + found if storm_of(p) == s]

    removed = [line for line in old if line not in bad]
    if removed:
        print(f"refusing to drop {len(removed)} existing entries; {EXCLUDED} is unchanged:", file=sys.stderr)
        for line in removed:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)

    fd, tmp = tempfile.mkstemp(prefix=EXCLUDED.name + ".", suffix=".tmp", dir=EXCLUDED.resolve().parent)
    os.close(fd)
    try:
        Path(tmp).write_text("\n".join(bad))
        os.replace(tmp, EXCLUDED)
    finally:
        Path(tmp).unlink(missing_ok=True)
    print(f"{len(bad)} of {n} excluded")
    added = [p for p in bad if p not in old]
    print(f"{len(added)} added" + "".join(f"\n  {p}" for p in added))


if __name__ == "__main__":
    main()
