"""Export frame PNGs + metadata for the Hurricane Ian scrubber on the project site.

Writes site/frames/ian/NNN.png (160x160 grayscale, chronological) and
site/frames/ian.json, pairing each frame with its held-out prediction.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# TCDataset reads excluded_crops.txt via a CWD-relative path, so run from the
# repo root to get the same exclusions evaluate.py saw. Our own paths are all
# absolute and __file__-derived regardless.
os.chdir(REPO_ROOT)

from hurricane_intensity.dataset import TCDataset  # noqa: E402

HOLDOUT_STORMS = ["AL092022", "AL172022", "AL032021"]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
PREDICTIONS_CSV = REPO_ROOT / "results" / "heldout_predictions.csv"
IAN_ATCF_ID = "AL092022"

OUT_DIR = REPO_ROOT / "site" / "frames"
FRAME_DIR = OUT_DIR / "ian"
JSON_PATH = OUT_DIR / "ian.json"
FRAME_SIZE = 160


def main():
    ds = TCDataset(str(PROCESSED_ROOT), HOLDOUT_STORMS)

    # Rows of the CSV are in dataset order, so the k-th Ian row of ds.df and the
    # k-th Ian row of the CSV describe the same frame.
    ds_ian_positions = np.flatnonzero(
        (ds.df["atcf_id"] == IAN_ATCF_ID).to_numpy()
    )
    preds = pd.read_csv(PREDICTIONS_CSV)
    csv_ian = preds[preds["atcf_id"] == IAN_ATCF_ID].reset_index(drop=True)

    assert len(ds_ian_positions) == len(csv_ian), (
        f"Ian frame mismatch: dataset has {len(ds_ian_positions)}, "
        f"CSV has {len(csv_ian)}"
    )

    ds_times = pd.to_datetime(
        ds.df.iloc[ds_ian_positions]["sample_time"], utc=True
    ).to_numpy()
    csv_times = pd.to_datetime(csv_ian["sample_time"], utc=True).to_numpy()
    assert (ds_times == csv_times).all(), "dataset/CSV row order diverged for Ian"

    # Chronological order drives both the file numbering and the JSON indices.
    order = np.argsort(csv_times, kind="stable")
    assert len(np.unique(csv_times)) == len(csv_times), "duplicate sample_time in Ian frames"

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(len(order) - 1)))

    records = []
    total_bytes = 0
    for index, k in enumerate(order):
        image, _ = ds[int(ds_ian_positions[k])]
        # Channel 0 of the post-transform tensor is the normalized BT field in [0, 1].
        gray = np.clip(image[0].numpy() * 255.0, 0, 255).astype(np.uint8)
        png = Image.fromarray(gray, mode="L").resize(
            (FRAME_SIZE, FRAME_SIZE), Image.LANCZOS
        )
        name = f"{index:0{width}d}.png"
        png.save(FRAME_DIR / name, optimize=True)
        total_bytes += (FRAME_DIR / name).stat().st_size

        row = csv_ian.iloc[int(k)]
        records.append({
            "index": index,
            "time": pd.Timestamp(csv_times[k]).isoformat(),
            "truth": round(float(row["truth"]), 1),
            "pred": round(float(row["pred"]), 1),
            "is_interpolated": bool(row["is_interpolated"]),
        })

    payload = {"storm": str(csv_ian["storm_name"].iloc[0]), "frames": records}
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    total_bytes += JSON_PATH.stat().st_size

    truths = [r["truth"] for r in records]
    preds_kt = [r["pred"] for r in records]
    print(f"frames:      {len(records)}  -> {FRAME_DIR}")
    print(f"truth wind:  {min(truths):.1f} - {max(truths):.1f} kt")
    print(f"pred wind:   {min(preds_kt):.1f} - {max(preds_kt):.1f} kt")
    print(f"output size: {total_bytes / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
