"""List and time-match GOES-16 ABI scans on the noaa-goes16 S3 bucket.

All access is anonymous (unsigned) -- the bucket is public AWS Open Data,
no credentials needed or wanted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from . import config

# e.g. OR_ABI-L2-CMIPF-M3C13_G16_s20182800000377_e20182800011156_c20182800011224.nc
# The mode digit after "M" varies (M3 pre-Apr-2019, M6 after) so it's a
# wildcard here rather than hardcoded, to keep this generic across storms.
_KEY_RE = re.compile(
    r"OR_ABI-L2-CMIPF-M\dC(?P<band>\d{2})_G16_s(?P<start>\d{13})\d_e\d{14}_c\d{14}\.nc$"
)


@dataclass(frozen=True)
class ScanMatch:
    scan_time: datetime  # UTC
    key: str
    offset_min: float  # scan_time - target_time, in minutes (signed)


def make_anonymous_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _parse_scan_start(key: str) -> datetime:
    m = _KEY_RE.search(key)
    if not m:
        raise ValueError(f"Key does not match expected ABI filename pattern: {key}")
    # "start" is YYYYDDDHHMMSS (13 digits); the file also has a trailing
    # tenths-of-a-second digit we don't need.
    return datetime.strptime(m.group("start"), "%Y%j%H%M%S").replace(
        tzinfo=timezone.utc
    )


def _list_keys_for_hour(
    s3_client, bucket: str, product: str, band: int, dt_hour: datetime
) -> list[tuple[datetime, str]]:
    prefix = f"{product}/{dt_hour.year:04d}/{dt_hour.timetuple().tm_yday:03d}/{dt_hour.hour:02d}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            m = _KEY_RE.search(key)
            if m and int(m.group("band")) == band:
                out.append((_parse_scan_start(key), key))
    return out


def find_nearest_scan(
    s3_client,
    target_time: datetime,
    tolerance_min: float = config.DEFAULT_TOLERANCE_MIN,
    bucket: str = config.GOES_BUCKET,
    product: str = config.ABI_PRODUCT,
    band: int = config.ABI_BAND,
) -> ScanMatch | None:
    """Find the ABI scan closest in time to target_time.

    Lists the target hour plus the hours immediately before/after, so a
    target near an hour boundary still finds its nearest scan. Returns None
    if nothing is found within tolerance_min minutes.
    """
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)

    candidates: list[tuple[datetime, str]] = []
    for delta_hours in (-1, 0, 1):
        dt_hour = target_time + timedelta(hours=delta_hours)
        candidates.extend(_list_keys_for_hour(s3_client, bucket, product, band, dt_hour))

    if not candidates:
        return None

    best_time, best_key = min(
        candidates, key=lambda tk: abs((tk[0] - target_time).total_seconds())
    )
    offset_min = (best_time - target_time).total_seconds() / 60.0

    if abs(offset_min) > tolerance_min:
        return None

    return ScanMatch(scan_time=best_time, key=best_key, offset_min=offset_min)
