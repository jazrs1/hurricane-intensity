"""Export held-out C13 sample crops for the personal site's interactive picker.

For each --scene (a held-out storm and a UTC timestamp), writes a ~150px WebP
of the model's input frame to <out-dir>/<ATCF>_<YYYYMMDDTHHMMZ>.webp, and
records the HURDAT2 label, the trained model's prediction and the error in
<out-dir>/samples.json. With --gradcam it also writes <name>_gradcam.webp, the
Grad-CAM overlay from scripts/gradcam.py on the same frame, aligned pixel for
pixel with the crop. Nothing is written unless every scene passes its checks.

Every scene must be held out: its storm is in storms_holdout.txt and not in
storms_train.txt (training and validation storms), and its GOES file was
not used for any training/validation crop. The model must also reproduce
results/heldout_predictions.csv for it.

Usage:
    python scripts/export_samples.py --fixes-only \\
        --scene AL092022 2022-09-28T12:00 --scene AL172022 2022-11-09T18:00
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# TCDataset reads excluded_crops.txt via a CWD-relative path (see the same
# note in scripts/export_scrubber.py), so run from the repo root.
os.chdir(REPO_ROOT)

PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
PREDICTIONS_CSV = REPO_ROOT / "results" / "heldout_predictions.csv"
HOLDOUT_LIST = REPO_ROOT / "storms_holdout.txt"
TRAIN_LIST = REPO_ROOT / "storms_train.txt"

WEBP_QUALITY = 90
SAMPLE_SIZE = 150
GRADCAM_SUFFIX = "_gradcam"
# best_model.pt on CPU vs the GPU run that wrote heldout_predictions.csv
# differs by float noise only; anything larger means the checkpoint changed.
PRED_CHECK_TOLERANCE_KT = 0.1
MAJOR_HURRICANE_KT = 96.0


def training_split_storms(load_storm_list) -> set[str]:
    """Every storm a checkpoint can have seen: storms_train.txt, the list
    train.py reads, which names the training and the validation storms (VAL
    drove checkpoint selection)."""
    return set(load_storm_list(TRAIN_LIST))


def training_scan_keys(training: set[str]) -> set[str]:
    """Every GOES file any training/validation crop was cut from. A held-out
    scene sharing one would mean the network saw part of that scan."""
    keys = set()
    for storm in sorted(training):
        manifest = PROCESSED_ROOT / storm / "manifest.csv"
        if not manifest.exists():
            raise SystemExit(f"missing {manifest}; can't verify held-out scenes against it")
        keys.update(pd.read_csv(manifest, usecols=["source_key"])["source_key"].dropna())
    return keys


def export_samples(args) -> None:
    import torch

    import gradcam  # load_model, resolve_frame, load_storm_list
    from hurricane_intensity.dataset import TCDataset

    holdout = set(gradcam.load_storm_list(HOLDOUT_LIST))
    training = training_split_storms(gradcam.load_storm_list)
    # Explicit raises rather than `assert`, which python -O strips.
    if holdout & training:
        raise SystemExit(f"storms_holdout.txt overlaps the training split: {sorted(holdout & training)}")

    scenes = [(storm.strip().upper(), ts) for storm, ts in args.scene]
    for storm, ts in scenes:
        if storm in training:
            raise SystemExit(f"refusing {storm} {ts}: {storm} is in the training split")
        if storm not in holdout:
            raise SystemExit(f"refusing {storm} {ts}: {storm} is not in storms_holdout.txt")
    training_keys = training_scan_keys(training)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gradcam.load_model(device)

    reference = pd.read_csv(PREDICTIONS_CSV)
    reference["time"] = pd.to_datetime(reference["sample_time"], utc=True)

    datasets = {}
    records = []
    thumbnails = []  # (filename, image), written only after every scene passes
    for storm, ts in scenes:
        if storm not in datasets:
            datasets[storm] = TCDataset(str(PROCESSED_ROOT), [storm])
        ds = datasets[storm]
        idx = gradcam.resolve_frame(ds, storm, ts)
        row = ds.df.iloc[idx]
        time = pd.Timestamp(row["sample_time"]).tz_convert("UTC")

        if args.fixes_only and bool(row["is_interpolated"]):
            raise SystemExit(f"refusing {storm} {ts}: not a HURDAT2 fix (its truth is interpolated between fixes)")
        if row["source_key"] in training_keys:
            raise SystemExit(f"refusing {storm} {ts}: its scan {row['source_key']} was used in training")

        image, _ = ds[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).item() * 185.0
        truth = float(row["wind_kt"])

        ref = reference[(reference["atcf_id"] == storm) & (reference["time"] == time)]
        if len(ref) != 1:
            raise SystemExit(f"{storm} {ts}: expected one row in {PREDICTIONS_CSV.name}, found {len(ref)}")
        if abs(float(ref["pred"].iloc[0]) - pred) > PRED_CHECK_TOLERANCE_KT:
            raise SystemExit(
                f"{storm} {ts}: model predicts {pred:.2f} kt but {PREDICTIONS_CSV.name} has "
                f"{float(ref['pred'].iloc[0]):.2f} kt -- best_model.pt no longer matches the reported results"
            )

        # Channel 0 of the model input is the normalized BT field -- the same
        # pixels export_scrubber.py renders.
        gray = np.clip(image[0].numpy() * 255.0, 0, 255).astype(np.uint8)
        name = f"{storm}_{time:%Y%m%dT%H%MZ}.webp"
        thumbnails.append((name, Image.fromarray(gray).resize((args.size, args.size), Image.LANCZOS)))
        if args.gradcam:
            # gradcam.py's own attribution and overlay on the same 224x224 input, then
            # the same resize as the crop, so the two line up pixel for pixel.
            cam = gradcam.compute_gradcam(model, image, device)
            overlay = gradcam.render_threshold_image(gray, cam)
            thumbnails.append((f"{Path(name).stem}{GRADCAM_SUFFIX}.webp",
                               overlay.resize((args.size, args.size), Image.LANCZOS)))

        interpolated = bool(row["is_interpolated"])
        if interpolated:
            print(
                f"WARNING: {storm} {ts} is between HURDAT2 fixes; its truth is interpolated, "
                "not a best-track value (fixes are 00/06/12/18Z plus landfalls)",
                file=sys.stderr,
            )
        record = {
            "filename": name,
            "storm": str(row["storm_name"]).title(),
            "fix_time_utc": f"{time:%Y-%m-%dT%H:%MZ}",
            "truth": round(truth, 1),
            "prediction": round(pred, 1),
            "error": round(pred - truth, 1),
        }
        if interpolated:
            record["truth_is_interpolated"] = True
        records.append(record)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, thumbnail in thumbnails:
        thumbnail.save(out_dir / name, quality=WEBP_QUALITY, method=6)
    json_path = out_dir / "samples.json"
    json_path.write_text(json.dumps(records, indent=2) + "\n")

    all_abs = reference["abs_error"].to_numpy()
    print(f"\n{'storm':<10}{'time (UTC)':<18}{'truth':>7}{'pred':>7}{'error':>8}  |error| pctile in held-out")
    for r in records:
        pct = (all_abs < abs(r["error"])).mean() * 100
        print(f"{r['storm']:<10}{r['fix_time_utc']:<18}{r['truth']:>7.1f}{r['prediction']:>7.1f}{r['error']:>+8.1f}  {pct:5.1f}")
    if not any(r["truth"] >= MAJOR_HURRICANE_KT and r["prediction"] < r["truth"] for r in records):
        print(f"note: no scene is a major hurricane (>= {MAJOR_HURRICANE_KT:g} kt) that the model underestimates")
    if args.gradcam:
        print(
            "\nGrad-CAM (scripts/gradcam.py): gradient of the predicted wind speed with respect to "
            "ResNet-18 layer4 activations (7x7, 512 channels), bilinear-upsampled to 224 px, "
            f"Gaussian-blurred (sigma {gradcam.CAM_BLUR_SIGMA:g} px) and normalized to 0-1 per frame; "
            f"overlay = gradcam.render_threshold_image (plasma wash, off below {gradcam.THRESHOLD:g})"
        )
    print(f"\nwrote {len(records)} crops{' + Grad-CAM overlays' if args.gradcam else ''} and {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", nargs=2, action="append", required=True, metavar=("STORM", "TIME"),
                        help="held-out ATCF id and UTC timestamp; repeat per scene")
    parser.add_argument("--fixes-only", action="store_true",
                        help="refuse any scene whose truth is interpolated between HURDAT2 fixes")
    parser.add_argument("--gradcam", action="store_true",
                        help=f"also write <name>{GRADCAM_SUFFIX}.webp, gradcam.py's layer4 overlay on the same crop")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "site" / "samples"))
    export_samples(parser.parse_args())


if __name__ == "__main__":
    main()
