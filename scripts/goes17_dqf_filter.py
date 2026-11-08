#!/usr/bin/env python
"""GOES-17 loop-heat-pipe (cooling) defect filter for the analog storms -> configs/analog_excluded_crops.txt.

    .venv\\Scripts\\python.exe scripts\\goes17_dqf_filter.py            # report and append new entries
    .venv\\Scripts\\python.exe scripts\\goes17_dqf_filter.py --dry_run  # report only

Implements the recommendation in reports/v2_survey/survey.md section 4, from the dqf array saved in every crop:
a GOES-17 frame (source key _G17_) is excluded if any pixel with finite BT has DQF 4
(focal_plane_temperature_threshold_exceeded_qf), or any pixel has DQF 1 or 2. GOES-17 CMI files carry DQF 4
only from 2019-04-08, so a GOES-17 frame before that date is instead excluded by the survey's fallback windows:
within 01-29..05-09 or 08-03..11-11 (+-50 days of an equinox) AND 04:00-14:00 UTC.

Only GOES-17 frames are tested (other satellites have no LHP defect). Entries already in excluded_crops.txt or
configs/analog_excluded_crops.txt are not added again; excluded_crops.txt itself is never written (it is
pinned by the frozen v2 recipe). Rebuild the affected storms afterwards with
scripts/build_storm_windows.py --ids <printed IDs>.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from hurricane_intensity.config import PROCESSED_DIR, PROJECT_ROOT

STORMS = PROJECT_ROOT / "configs" / "analog_storms.csv"
ANALOG_EXCLUDED = PROJECT_ROOT / "configs" / "analog_excluded_crops.txt"
FROZEN_EXCLUDED = PROJECT_ROOT / "excluded_crops.txt"
DQF4_FROM = pd.Timestamp("2019-04-08", tz="UTC")
SEASONS = (("01-29", "05-09"), ("08-03", "11-11"))


def in_fallback_window(t):
    md = f"{t.month:02d}-{t.day:02d}"
    return any(a <= md <= b for a, b in SEASONS) and 4 <= t.hour < 14


def reason(path, t):
    """Why this GOES-17 frame is excluded, or None."""
    if t < DQF4_FROM:
        return "pre-2019-04-08 fallback window" if in_fallback_window(t) else None
    z = np.load(path)
    dqf, bt = z["dqf"], z["bt"]
    if ((dqf == 4) & np.isfinite(bt)).any():
        return "DQF 4"
    if np.isin(dqf, (1, 2)).any():
        return "DQF 1/2"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    listed = set()
    for f in (FROZEN_EXCLUDED, ANALOG_EXCLUDED):
        if f.exists():
            listed |= set(f.read_text().split())
    storms = pd.read_csv(STORMS).query("usable == 1")["atcf_id"]
    new, reasons, g17_frames, per_storm = [], Counter(), 0, Counter()
    for s in storms:
        m = pd.read_csv(PROCESSED_DIR / s / "manifest.csv")
        m = m[(m["match_status"] == "ok") & m["source_key"].astype(str).str.contains("_G17_")]
        for rel, t in zip(m["crop_path"], pd.to_datetime(m["sample_time"], utc=True)):
            path = PROCESSED_DIR / rel.replace("\\", "/")
            if rel in listed or not path.is_file():
                continue
            g17_frames += 1
            why = reason(path, t)
            if why:
                new.append(rel)
                reasons[why] += 1
                per_storm[s] += 1
    print(f"GOES-17 frames tested (not already excluded): {g17_frames}; newly excluded: {len(new)} {dict(reasons)}")
    for s, n in sorted(per_storm.items()):
        print(f"  {s}: {n}")
    if new and not args.dry_run:
        with open(ANALOG_EXCLUDED, "a", newline="\n") as f:
            f.write("".join(f"{rel}\n" for rel in new))
        print(f"appended {len(new)} entries to {ANALOG_EXCLUDED}")
    print("affected storms:", " ".join(sorted(per_storm)))


if __name__ == "__main__":
    main()
