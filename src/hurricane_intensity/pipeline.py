"""Orchestrates one storm end-to-end: HURDAT2 fixes -> sample grid ->
nearest-scan matching -> download/crop/delete -> saved crops + manifest.

This is the only module that ties the others together; hurdat2.py,
goes_index.py, projection.py, and crop.py all work standalone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config, crop, goes_index, hurdat2

logger = logging.getLogger(__name__)


def build_sample_grid(fixes: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Build the set of sample times for a storm: a regular grid at
    `interval` plus every exact HURDAT2 fix time, unioned together, with
    lat/lon/wind linearly interpolated between the surrounding fixes.

    Columns: time, lat, lon, wind_kt, is_interpolated, status
    (status is the HURDAT2 code for exact fixes, NaN for interpolated ones).

    Note: longitude is interpolated linearly, which would break across the
    antimeridian (+-180). Not a concern for Atlantic-basin storms, but
    worth knowing if this is ever pointed at a Pacific storm that crosses it.
    """
    fixes = fixes.sort_values("time").reset_index(drop=True)
    first, last = fixes["time"].iloc[0], fixes["time"].iloc[-1]

    base_grid = pd.date_range(start=first, end=last, freq=interval, tz="UTC")
    exact_times = pd.DatetimeIndex(fixes["time"])
    all_times = pd.DatetimeIndex(sorted(set(base_grid) | set(exact_times)))

    t_fix = (exact_times - first).total_seconds().to_numpy()
    t_all = (all_times - first).total_seconds().to_numpy()

    lat = np.interp(t_all, t_fix, fixes["lat"].to_numpy())
    lon = np.interp(t_all, t_fix, fixes["lon"].to_numpy())
    wind_kt = np.interp(t_all, t_fix, fixes["wind_kt"].to_numpy())

    exact_set = set(exact_times)
    is_interpolated = np.array([t not in exact_set for t in all_times])

    grid = pd.DataFrame(
        {
            "time": all_times,
            "lat": lat,
            "lon": lon,
            "wind_kt": wind_kt,
            "is_interpolated": is_interpolated,
        }
    )
    grid = grid.merge(fixes[["time", "status"]], on="time", how="left")
    return grid


@dataclass
class StormSummary:
    atcf_id: str
    storm_name: str
    n_total: int
    n_matched: int
    n_skipped: int
    offsets_min: list[float]

    def print_report(self) -> None:
        print(f"\n=== {self.atcf_id} ({self.storm_name}) ===")
        print(f"  sample times attempted: {self.n_total}")
        print(f"  matched:                {self.n_matched}")
        print(f"  skipped (no scan in tolerance): {self.n_skipped}")
        if self.offsets_min:
            arr = np.array(self.offsets_min)
            print(
                "  time offset (min, scan - target): "
                f"min={arr.min():+.2f} max={arr.max():+.2f} "
                f"mean={arr.mean():+.2f} median={np.median(arr):+.2f}"
            )
        else:
            print("  time offset: n/a (no matches)")


def run_pipeline(
    atcf_id: str,
    sample_interval: str = config.DEFAULT_SAMPLE_INTERVAL,
    box_km: float = config.DEFAULT_BOX_KM,
    tolerance_min: float = config.DEFAULT_TOLERANCE_MIN,
    out_root: Path = config.PROCESSED_DIR,
    hurdat2_url: str | None = None,
    force_hurdat2_refresh: bool = False,
    include_statuses: tuple[str, ...] = config.DEFAULT_INCLUDE_STATUSES,
) -> Path:
    """Run the full pipeline for one storm. Returns the manifest CSV path."""
    atcf_id = atcf_id.strip().upper()
    out_dir = Path(out_root) / atcf_id
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    hurdat2_path = hurdat2.download_hurdat2(url=hurdat2_url, force=force_hurdat2_refresh)
    storm_name, fixes = hurdat2.parse_storm(atcf_id, hurdat2_path, include_statuses)

    grid = build_sample_grid(fixes, sample_interval)
    logger.info(
        "%s (%s): %d sample times (%d exact fixes, %d interpolated) at interval=%s",
        atcf_id,
        storm_name,
        len(grid),
        (~grid["is_interpolated"]).sum(),
        grid["is_interpolated"].sum(),
        sample_interval,
    )

    s3_client = goes_index.make_anonymous_s3_client()

    rows: list[dict] = []
    offsets: list[float] = []
    n_matched = 0

    for row in tqdm(grid.itertuples(index=False), total=len(grid), desc=atcf_id):
        sample_time = row.time.to_pydatetime()
        record = {
            "atcf_id": atcf_id,
            "storm_name": storm_name,
            "sample_time": sample_time.isoformat(),
            "lat": row.lat,
            "lon": row.lon,
            "wind_kt": row.wind_kt,
            "is_interpolated": bool(row.is_interpolated),
            "status": row.status if isinstance(row.status, str) else "",
            "match_status": "missing_scan",
            "scan_time": "",
            "offset_min": "",
            "source_key": "",
            "crop_path": "",
            "bt_shape": "",
        }

        match = goes_index.find_nearest_scan(
            s3_client, sample_time, tolerance_min=tolerance_min
        )
        if match is None:
            rows.append(record)
            continue

        try:
            result = crop.process_one_scan(
                s3_client, match.key, row.lat, row.lon, box_km
            )
        except Exception:
            logger.exception(
                "Failed to crop scan %s for sample %s -- skipping",
                match.key,
                sample_time.isoformat(),
            )
            rows.append(record)
            continue

        crop_filename = f"{atcf_id}_{sample_time:%Y%m%dT%H%MZ}.npz"
        crop_path = crops_dir / crop_filename
        np.savez(
            crop_path,
            bt=result.bt,
            dqf=result.dqf,
            wind_kt=np.float32(row.wind_kt),
            lat=np.float32(row.lat),
            lon=np.float32(row.lon),
            sample_time_unix=np.int64(sample_time.timestamp()),
            scan_time_unix=np.int64(match.scan_time.timestamp()),
            offset_min=np.float32(match.offset_min),
            is_interpolated=np.bool_(row.is_interpolated),
            atcf_id=np.str_(atcf_id),
            storm_name=np.str_(storm_name),
            status=np.str_(record["status"]),
        )

        record.update(
            match_status="ok",
            scan_time=match.scan_time.isoformat(),
            offset_min=round(match.offset_min, 3),
            source_key=match.key,
            crop_path=str(crop_path.relative_to(out_root)),
            bt_shape=f"{result.bt.shape[0]}x{result.bt.shape[1]}",
        )
        rows.append(record)
        offsets.append(match.offset_min)
        n_matched += 1

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = StormSummary(
        atcf_id=atcf_id,
        storm_name=storm_name,
        n_total=len(grid),
        n_matched=n_matched,
        n_skipped=len(grid) - n_matched,
        offsets_min=offsets,
    )
    summary.print_report()
    print(f"  manifest: {manifest_path}")
    print(f"  crops:    {crops_dir}")

    return manifest_path
