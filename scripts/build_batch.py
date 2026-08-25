#!/usr/bin/env python
"""Run the build_dataset pipeline over a batch of storms, sequentially.

Usage:
    python scripts/build_batch.py --ids-file 2021_storms.txt
    python scripts/build_batch.py AL142018 AL092021 --skip-existing

IDs can be passed directly as arguments, or one per line in a text file via
--ids-file (blank lines and lines starting with '#' are ignored). If both
are given, the file's IDs are appended to the ones on the command line.

Each storm is built by calling hurricane_intensity.pipeline.run_pipeline()
directly -- same code path as build_dataset.py, just looped. If a storm
fails, the error is logged and the run moves on to the next storm; a
success/failure summary is printed at the end, plus combined matched/
skipped/offset totals across every storm that has a manifest (newly built
or, with --skip-existing, already present).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hurricane_intensity import config, pipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "atcf_ids",
        nargs="*",
        help="ATCF storm ids to build, e.g. AL142018 AL092021.",
    )
    p.add_argument(
        "--ids-file",
        default=None,
        help="Text file with one ATCF id per line ('#' comments allowed).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a storm if it already has a non-empty manifest.csv under "
            "--out-root/<ATCF_ID>/ instead of rebuilding it."
        ),
    )
    # Pass-through pipeline options, same as build_dataset.py -- applied
    # uniformly to every storm in the batch.
    p.add_argument("--sample-interval", default=config.DEFAULT_SAMPLE_INTERVAL)
    p.add_argument("--box-km", type=float, default=config.DEFAULT_BOX_KM)
    p.add_argument("--tolerance-min", type=float, default=config.DEFAULT_TOLERANCE_MIN)
    p.add_argument("--out-root", default=str(config.PROCESSED_DIR))
    p.add_argument("--hurdat2-url", default=None)
    p.add_argument("--force-hurdat2-refresh", action="store_true")
    p.add_argument(
        "--include-statuses", default=",".join(config.DEFAULT_INCLUDE_STATUSES)
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args()


def load_ids(atcf_ids: list[str], ids_file: str | None) -> list[str]:
    ids = list(atcf_ids)
    if ids_file:
        for line in Path(ids_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    if not ids:
        raise SystemExit("No ATCF ids given -- pass them as arguments or via --ids-file.")
    # Dedupe while preserving order, in case an id shows up on the command
    # line and in the file.
    seen: set[str] = set()
    deduped = []
    for atcf_id in ids:
        atcf_id = atcf_id.strip().upper()
        if atcf_id not in seen:
            seen.add(atcf_id)
            deduped.append(atcf_id)
    return deduped


def manifest_stats(manifest_path: Path) -> tuple[int, int, list[float]]:
    """Read a manifest.csv back and return (n_total, n_matched, offsets_min)."""
    manifest = pd.read_csv(manifest_path)
    matched = manifest[manifest["match_status"] == "ok"]
    return len(manifest), len(matched), matched["offset_min"].astype(float).tolist()


@dataclass
class BatchResult:
    atcf_id: str
    status: str  # "built", "skipped-existing", "failed"
    detail: str = ""
    n_total: int = 0
    n_matched: int = 0
    offsets_min: list[float] | None = None


def print_combined_totals(results: list[BatchResult]) -> None:
    with_data = [r for r in results if r.status in ("built", "skipped-existing")]
    n_total = sum(r.n_total for r in with_data)
    n_matched = sum(r.n_matched for r in with_data)
    all_offsets = [o for r in with_data for o in (r.offsets_min or [])]

    print("\n=== combined totals ===")
    print(f"  storms with data:       {len(with_data)}")
    print(f"  sample times attempted: {n_total}")
    print(f"  matched:                {n_matched}")
    print(f"  skipped (no scan in tolerance): {n_total - n_matched}")
    if all_offsets:
        arr = np.array(all_offsets)
        print(
            "  time offset (min, scan - target): "
            f"min={arr.min():+.2f} max={arr.max():+.2f} "
            f"mean={arr.mean():+.2f} median={np.median(arr):+.2f}"
        )
    else:
        print("  time offset: n/a (no matches)")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    atcf_ids = load_ids(args.atcf_ids, args.ids_file)
    include_statuses = tuple(
        s.strip().upper() for s in args.include_statuses.split(",") if s.strip()
    )
    out_root = Path(args.out_root)

    results: list[BatchResult] = []

    for atcf_id in atcf_ids:
        existing_manifest = out_root / atcf_id / "manifest.csv"
        if args.skip_existing and existing_manifest.exists():
            n_total, n_matched, offsets = manifest_stats(existing_manifest)
            if n_total > 0:
                print(f"\n=== {atcf_id}: skipping (existing manifest found) ===")
                print(f"  manifest: {existing_manifest}")
                results.append(
                    BatchResult(
                        atcf_id,
                        "skipped-existing",
                        n_total=n_total,
                        n_matched=n_matched,
                        offsets_min=offsets,
                    )
                )
                continue

        try:
            manifest_path = pipeline.run_pipeline(
                atcf_id=atcf_id,
                sample_interval=args.sample_interval,
                box_km=args.box_km,
                tolerance_min=args.tolerance_min,
                out_root=args.out_root,
                hurdat2_url=args.hurdat2_url,
                force_hurdat2_refresh=args.force_hurdat2_refresh,
                include_statuses=include_statuses,
            )
        except Exception as exc:
            logger.exception("%s: pipeline failed -- continuing with next storm", atcf_id)
            results.append(BatchResult(atcf_id, "failed", detail=str(exc)))
            continue

        n_total, n_matched, offsets = manifest_stats(manifest_path)
        results.append(
            BatchResult(
                atcf_id, "built", n_total=n_total, n_matched=n_matched, offsets_min=offsets
            )
        )

    print("\n=== batch summary ===")
    for r in results:
        if r.status == "failed":
            print(f"  {r.atcf_id}: FAILED -- {r.detail}")
        else:
            print(f"  {r.atcf_id}: {r.status} ({r.n_matched}/{r.n_total} matched)")

    succeeded = [r.atcf_id for r in results if r.status != "failed"]
    failed = [r.atcf_id for r in results if r.status == "failed"]
    print(f"\n  succeeded: {len(succeeded)} -- {', '.join(succeeded) if succeeded else 'none'}")
    print(f"  failed:    {len(failed)} -- {', '.join(failed) if failed else 'none'}")

    print_combined_totals(results)


if __name__ == "__main__":
    main()
