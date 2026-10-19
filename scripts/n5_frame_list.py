"""N5 atlantic-arm frame list (reports/v2_survey/n5/design.md §2 Marco rule, §12.5).

    python scripts/n5_frame_list.py [--processed-root data/processed] [--out results/n5_atlantic/frame_list_atlantic.txt]

Every training frame of storms_train_n5_atlantic.txt (load_splits with storms_val_expanded.txt; TCDataset, so the
frames in ./excluded_crops.txt are already gone), in TCDataset order, minus the Marco AL142020 frames whose 650 km
box overlaps a validation, H9, 2025-test or H-EP storm's box at the same time (scripts/concurrency.py, decided A4).
Marco's positions are its built manifest's own (sample_time, lat, lon); the other storms' come from the pinned
Atlantic HURDAT2, interpolated between fixes. H-EP storms are East Pacific, so they are not in that file; they are
checked by year instead, and none shares a year with Marco. The rule must drop exactly the design's 4 frames
(2020-08-25 09Z, 12Z, 15Z, 18Z), or nothing is written.

Writes one crop_path per line, LF, UTF-8, with a final newline. An existing file with other bytes is never
overwritten. Prints the frame count and the SHA-256 of the bytes written (ARM_TRAIN_FRAMES, FRAME_LIST_SHA256).
Reads data only; downloads nothing.
"""
import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from hurricane_intensity.dataset import TCDataset
from hurricane_intensity.splits import load_splits, load_storm_list

ROOT = Path(__file__).resolve().parents[1]
TRAIN_LIST, VAL_LIST = "storms_train_n5_atlantic.txt", "storms_val_expanded.txt"
PROTECTED_LISTS = ("storms_val_expanded.txt", "storms_holdout_h9.txt", "storms_test_2025.txt", "storms_holdout_hep.txt")
MARCO = "AL142020"
DESIGN_DROPS = ["2020-08-25T09Z", "2020-08-25T12Z", "2020-08-25T15Z", "2020-08-25T18Z"]
ATLANTIC_PIN = "ee35819cba2b61ab0c2175601084def682813727ba209c2dab7d3af71d0bcb5c"

_spec = importlib.util.spec_from_file_location("concurrency", ROOT / "scripts" / "concurrency.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def build(processed_root, hurdat2_path):
    """(frame list, dropped Marco rows as a DataFrame)."""
    if hashlib.sha256(Path(hurdat2_path).read_bytes()).hexdigest() != ATLANTIC_PIN:
        raise SystemExit(f"{hurdat2_path} is not the pinned Atlantic HURDAT2 (ee35819c...)")
    train, _, _ = load_splits(train_list=TRAIN_LIST, val_list=VAL_LIST)
    df = TCDataset(str(processed_root), train).df
    protected = sorted({s for name in PROTECTED_LISTS for s in load_storm_list(ROOT / name)})
    other_basin = [s for s in protected if not s.startswith("AL")]
    same_year = [s for s in other_basin if s[4:] == MARCO[4:]]
    if same_year:
        raise SystemExit(f"{same_year} share Marco's year but are not in the Atlantic file; can't check them")
    marco = df[df["atcf_id"] == MARCO]
    track = pd.DataFrame({"time": pd.to_datetime(marco["sample_time"], utc=True).to_numpy(),
                          "lat": marco["lat"].to_numpy(float), "lon": marco["lon"].to_numpy(float)})
    hit = pd.Series(False, index=marco.index)
    for s in (s for s in protected if s.startswith("AL")):
        hit |= C.concurrent(track, C.track(s, hurdat2_path)[0])
    dropped = marco[hit.to_numpy()]
    got = [f"{t:%Y-%m-%dT%HZ}" for t in pd.to_datetime(dropped["sample_time"], utc=True)]
    if got != DESIGN_DROPS:
        raise SystemExit(f"the Marco rule drops {got}, not the design's {DESIGN_DROPS}; nothing written")
    frames = df.loc[~df.index.isin(dropped.index), "crop_path"].tolist()
    if len(set(frames)) != len(frames):
        raise SystemExit("duplicate crop paths in the training frames; nothing written")
    return frames, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--processed-root", default="data/processed")
    ap.add_argument("--hurdat2", default="data/interim/hurdat2_atlantic.txt")
    ap.add_argument("--out", default="results/n5_atlantic/frame_list_atlantic.txt")
    args = ap.parse_args(argv)
    frames, dropped = build(args.processed_root, args.hurdat2)
    data = ("\n".join(frames) + "\n").encode("utf-8")
    out = Path(args.out)
    if out.exists() and out.read_bytes() != data:
        raise SystemExit(f"{out} exists with other bytes; not overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"dropped (Marco rule): {dropped['crop_path'].tolist()}")
    print(f"frames: {len(frames)} from {len({p.split(chr(92))[0].split('/')[0] for p in frames})} storms")
    print(f"sha256: {hashlib.sha256(data).hexdigest()}")
    return frames


if __name__ == "__main__":
    sys.exit(main() and None)
