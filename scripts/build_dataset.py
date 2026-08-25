#!/usr/bin/env python
"""Build a matched GOES-16 / HURDAT2 dataset for one storm.

Usage:
    python scripts/build_dataset.py --atcf-id AL142018

Adding a second storm is just a different --atcf-id, e.g.:
    python scripts/build_dataset.py --atcf-id AL092021   # Ida, 2021

Find ATCF ids in the HURDAT2 file itself (data/interim/hurdat2_atlantic.txt
after the first run), or on the NHC best-track site.
"""

from __future__ import annotations

import argparse
import logging

from hurricane_intensity import config, pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--atcf-id",
        required=True,
        help="ATCF storm id, e.g. AL142018 for Hurricane Michael (2018).",
    )
    p.add_argument(
        "--sample-interval",
        default=config.DEFAULT_SAMPLE_INTERVAL,
        help=(
            "Pandas frequency string for the sample grid, e.g. '1h', '15min', "
            f"'6h'. Default: {config.DEFAULT_SAMPLE_INTERVAL}. Exact HURDAT2 "
            "fix times are always included in addition to this grid."
        ),
    )
    p.add_argument(
        "--box-km",
        type=float,
        default=config.DEFAULT_BOX_KM,
        help=f"Crop box side length in km. Default: {config.DEFAULT_BOX_KM}.",
    )
    p.add_argument(
        "--tolerance-min",
        type=float,
        default=config.DEFAULT_TOLERANCE_MIN,
        help=(
            "Max minutes between a sample time and the nearest ABI scan for "
            f"it to count as a match. Default: {config.DEFAULT_TOLERANCE_MIN}."
        ),
    )
    p.add_argument(
        "--out-root",
        default=str(config.PROCESSED_DIR),
        help=f"Root output directory. Default: {config.PROCESSED_DIR}",
    )
    p.add_argument(
        "--hurdat2-url",
        default=None,
        help=(
            "Explicit HURDAT2 source URL. By default (omitted) the current "
            "Atlantic HURDAT2 link is auto-resolved by scraping NHC's data "
            f"page, falling back to {config.HURDAT2_DEFAULT_URL} if that "
            "fails. Pass this to force a specific file and skip resolution."
        ),
    )
    p.add_argument(
        "--force-hurdat2-refresh",
        action="store_true",
        help="Re-download HURDAT2 even if a cached copy exists.",
    )
    p.add_argument(
        "--include-statuses",
        default=",".join(config.DEFAULT_INCLUDE_STATUSES),
        help=(
            "Comma-separated HURDAT2 status codes to keep, e.g. "
            "'LO,TD,SD,TS,SS,HU'. Default excludes EX (extratropical) so "
            "the track is trimmed to the tropical/subtropical lifetime."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    include_statuses = tuple(
        s.strip().upper() for s in args.include_statuses.split(",") if s.strip()
    )

    pipeline.run_pipeline(
        atcf_id=args.atcf_id,
        sample_interval=args.sample_interval,
        box_km=args.box_km,
        tolerance_min=args.tolerance_min,
        out_root=args.out_root,
        hurdat2_url=args.hurdat2_url,
        force_hurdat2_refresh=args.force_hurdat2_refresh,
        include_statuses=include_statuses,
    )


if __name__ == "__main__":
    main()
