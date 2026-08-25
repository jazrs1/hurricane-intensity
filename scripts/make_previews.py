#!/usr/bin/env python
"""Render saved crops as sequential PNGs for visual QC.

Walks a storm's manifest in time order, and for each successfully matched
crop writes a PNG with:
  - the brightness-temperature array, IR convention (cold cloud tops = white)
  - a red crosshair at the array's center pixel (the crop is built centered
    on the HURDAT2 storm center, so if the eye visibly drifts off that
    crosshair across frames, that's a real bug in the crop/projection code,
    not just JPEG noise)
  - a text overlay with sample time, wind (kt), and whether that sample was
    an exact HURDAT2 fix or interpolated

Usage:
    python scripts/make_previews.py --atcf-id AL142018
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from hurricane_intensity import config

# Typical IR brightness-temperature display range in Kelvin. Values are
# clipped to this range before stretching to 0-255; adjust if a storm's
# convective tops blow past 180K (rare) or its background is unusually warm.
BT_MIN_K = 180.0
BT_MAX_K = 300.0


def bt_to_image(bt: np.ndarray) -> Image.Image:
    clipped = np.clip(bt, BT_MIN_K, BT_MAX_K)
    stretched = (clipped - BT_MIN_K) / (BT_MAX_K - BT_MIN_K)  # 0..1, warm->1
    inverted = 1.0 - stretched  # cold (convective tops) -> bright, IR convention
    img_u8 = (inverted * 255).astype(np.uint8)
    return Image.fromarray(img_u8, mode="L").convert("RGB")


def draw_overlay(img: Image.Image, row: pd.Series) -> Image.Image:
    draw = ImageDraw.Draw(img)
    h, w = img.height, img.width
    cx, cy = w // 2, h // 2
    size = 8
    draw.line([(cx - size, cy), (cx + size, cy)], fill=(255, 0, 0), width=1)
    draw.line([(cx, cy - size), (cx, cy + size)], fill=(255, 0, 0), width=1)

    tag = "fix" if not row["is_interpolated"] else "interp"
    text = f"{row['sample_time']}  {row['wind_kt']:.0f}kt  [{tag}]"
    draw.rectangle([(0, 0), (w, 14)], fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 0))
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atcf-id", required=True)
    p.add_argument(
        "--out-root",
        default=str(config.PROCESSED_DIR),
        help=f"Root that contains <atcf-id>/manifest.csv. Default: {config.PROCESSED_DIR}",
    )
    p.add_argument(
        "--preview-dir",
        default=None,
        help="Defaults to <out-root>/<atcf-id>/previews",
    )
    args = p.parse_args()

    atcf_id = args.atcf_id.strip().upper()
    storm_dir = Path(args.out_root) / atcf_id
    manifest_path = storm_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(
            f"No manifest at {manifest_path} -- run build_dataset.py for "
            f"{atcf_id} first."
        )

    preview_dir = Path(args.preview_dir) if args.preview_dir else storm_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, parse_dates=["sample_time"])
    matched = manifest[manifest["match_status"] == "ok"].sort_values("sample_time")

    if matched.empty:
        raise SystemExit(f"No matched crops in {manifest_path} to preview.")

    for i, (_, row) in enumerate(matched.iterrows(), start=1):
        # crop_path in the manifest is stored relative to out_root
        crop_path = Path(args.out_root) / row["crop_path"]
        with np.load(crop_path) as npz:
            bt = npz["bt"]

        img = bt_to_image(bt)
        img = draw_overlay(img, row)
        out_path = preview_dir / f"{i:04d}.png"
        img.save(out_path)

    print(f"Wrote {len(matched)} preview PNGs to {preview_dir}")


if __name__ == "__main__":
    main()
