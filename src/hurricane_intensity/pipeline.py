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


def lon_diff(a, b) -> np.ndarray:
    """a - b in degrees of longitude, the short way round, in [-180, 180) (B-v2-1).

    Where a - b is already in range it is returned exactly, so tracks that never
    cross 180 deg (every Atlantic storm) get bit-identical results. lon_diff(x, 0)
    re-wraps a longitude into [-180, 180) (180.0E becomes -180.0).
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.where((d >= -180.0) & (d < 180.0), d, (d + 180.0) % 360.0 - 180.0)


def build_sample_grid(fixes: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Build the set of sample times for a storm: a regular grid at
    `interval` plus every exact HURDAT2 fix time, unioned together, with
    lat/lon/wind linearly interpolated between the surrounding fixes.

    Columns: time, lat, lon, wind_kt, is_interpolated, status
    (status is the HURDAT2 code for exact fixes, NaN for interpolated ones).

    Longitude is interpolated the short way round, linearly in time, and
    re-wrapped to [-180, 180) (John, 2026-10-20): each fix is offset by the
    multiple of 360 deg that unwraps the track, so a storm crossing 180 deg is
    interpolated across the seam, not the long way round the globe.
    """
    fixes = fixes.sort_values("time").reset_index(drop=True)
    first, last = fixes["time"].iloc[0], fixes["time"].iloc[-1]

    base_grid = pd.date_range(start=first, end=last, freq=interval, tz="UTC")
    exact_times = pd.DatetimeIndex(fixes["time"])
    all_times = pd.DatetimeIndex(sorted(set(base_grid) | set(exact_times)))

    t_fix = (exact_times - first).total_seconds().to_numpy()
    t_all = (all_times - first).total_seconds().to_numpy()

    lat = np.interp(t_all, t_fix, fixes["lat"].to_numpy())
    lon_fix = fixes["lon"].to_numpy(dtype=float)
    # 0 for each step that doesn't cross 180 deg, +-360 for one that does. Fixes with no offset are used as they
    # are (not + 0.0, which would turn a -0.0 fix into +0.0), so an uncrossed track is bit-identical.
    step = 360.0 * np.round((lon_diff(lon_fix[1:], lon_fix[:-1]) - (lon_fix[1:] - lon_fix[:-1])) / 360.0)
    offset = np.concatenate([[0.0], np.cumsum(step)])
    lon = lon_diff(np.interp(t_all, t_fix, np.where(offset != 0.0, lon_fix + offset, lon_fix)), 0.0)
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
    bucket: str = config.GOES_BUCKET,
    hurdat2_path: Path | None = None,
) -> Path:
    """Run the full pipeline for one storm. Returns the manifest CSV path.

    `bucket` selects the satellite (noaa-goes16, noaa-goes19, ...); it is
    used both to find the nearest scan and to download it.

    `hurdat2_path` reads a pinned basin file (e.g. the East Pacific one from
    hurdat2.fetch_ep_hurdat2) instead of the Atlantic cache: it is checked by
    filename and SHA-256, and download_hurdat2 is never called. It cannot be
    combined with hurdat2_url or force_hurdat2_refresh.
    """
    if hurdat2_path is not None and (hurdat2_url is not None or force_hurdat2_refresh):
        raise ValueError("hurdat2_path cannot be combined with hurdat2_url or force_hurdat2_refresh")
    if hurdat2_path is not None:
        hurdat2_path = hurdat2.verify_pinned_hurdat2(hurdat2_path)
    atcf_id = atcf_id.strip().upper()
    sealed = None
    if atcf_id in hurdat2.sealed_ids():
        # H-EP gate (d): the sealed build is the one call path that reads a sealed storm. It checks the
        # output root, the ID and the pinned file before anything is written.
        sealed = hurdat2._parse_sealed_hep_storm(
            atcf_id, hurdat2_path if hurdat2_path is not None else config.HURDAT2_CACHE_PATH,
            include_statuses, out_root)
    out_dir = Path(out_root) / atcf_id
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    if sealed is not None:
        storm_name, fixes = sealed
    else:
        if hurdat2_path is None:
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
            s3_client, sample_time, tolerance_min=tolerance_min, bucket=bucket
        )
        if match is None:
            rows.append(record)
            continue

        try:
            result = crop.process_one_scan(
                s3_client, match.key, row.lat, row.lon, box_km, bucket=bucket
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
            lon=np.float32(lon_diff(np.float32(row.lon), 0.0)),  # re-wrapped after the cast: 179.99999999 -> -180
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
