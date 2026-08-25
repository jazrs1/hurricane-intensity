"""Download one ABI scan, crop it, and delete it. Never accumulate raw files.

This is the disk-safety-critical module. The pattern for every sample is
strictly: download -> open -> extract crop into memory -> close -> delete.
The raw .nc file at config.SCRATCH_NC_PATH is reused and overwritten for
every sample, and is deleted (successfully, verified) before the next
download starts -- including on error paths.

Windows-specific trap this guards against: xarray/netCDF4 can leave the
underlying file handle open longer than you'd expect (lazy loading, HDF5
internal caching), and Windows -- unlike Linux -- refuses to delete a file
that still has an open handle. `safe_unlink` forces garbage collection and
retries the delete rather than assuming `.close()` was enough.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from . import config, projection

logger = logging.getLogger(__name__)


class RawFileStillLockedError(RuntimeError):
    pass


def safe_unlink(path: Path, retries: int = 5, delay_s: float = 0.3) -> None:
    """Delete a file, retrying through Windows file-lock delays.

    Forces a GC pass before each attempt: a lingering reference to a closed
    xarray Dataset (or its underlying h5py/netCDF4 file object) is enough to
    keep the Windows-level handle open even after `.close()` returns.
    """
    path = Path(path)
    if not path.exists():
        return
    last_err: Exception | None = None
    for attempt in range(retries):
        gc.collect()
        try:
            path.unlink()
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay_s)
    raise RawFileStillLockedError(
        f"Could not delete {path} after {retries} attempts -- "
        f"a file handle is still open somewhere. Last error: {last_err}"
    )


def download_scan(s3_client, key: str, dest_path: Path = config.SCRATCH_NC_PATH) -> Path:
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(config.GOES_BUCKET, key, str(dest_path))
    return dest_path


@dataclass
class Crop:
    bt: np.ndarray  # float32, (H, W), brightness temperature in Kelvin
    dqf: np.ndarray  # uint8/float, (H, W), ABI data quality flag (0 = good)


def extract_crop(
    nc_path: Path, center_lat: float, center_lon: float, box_km: float
) -> Crop:
    """Open the downloaded scan, crop it, and force everything into a plain
    in-memory numpy array before the file is closed.
    """
    nc_path = Path(nc_path)
    with xr.open_dataset(nc_path) as ds:
        crs, sat_height_m = projection.get_geostationary_crs(ds)

        bt_full = ds[config.ABI_VARIABLE]
        bt_crop = projection.crop_box(
            bt_full, center_lat, center_lon, box_km, crs, sat_height_m
        )
        bt = np.asarray(bt_crop.values, dtype=np.float32)  # forces load into memory

        dqf_full = ds["DQF"]
        dqf_crop = projection.crop_box(
            dqf_full, center_lat, center_lon, box_km, crs, sat_height_m
        )
        dqf = np.asarray(dqf_crop.values)

    return Crop(bt=bt, dqf=dqf)


def process_one_scan(
    s3_client,
    key: str,
    center_lat: float,
    center_lon: float,
    box_km: float,
    scratch_path: Path = config.SCRATCH_NC_PATH,
) -> Crop:
    """Download -> crop -> delete, for one ABI scan. Guarantees the raw file
    is removed even if cropping raises.
    """
    download_scan(s3_client, key, scratch_path)
    try:
        return extract_crop(scratch_path, center_lat, center_lon, box_km)
    finally:
        safe_unlink(scratch_path)
