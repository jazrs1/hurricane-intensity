"""Download and parse the Atlantic HURDAT2 best-track file.

HURDAT2 is a plain-text file, not a regular CSV: it's a flat sequence of
storms, each a header line followed by N fixed-format data lines. See
https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atlantic.pdf for the
official field spec. Example (Michael, 2018):

    AL142018,            MICHAEL,     38,
    20181006, 1800,  , LO, 17.8N,  86.6W,  25, 1006, ...
    20181007, 0000,  , LO, 18.1N,  86.9W,  25, 1004, ...
    ...

Header fields: ATCF id, name, number of data lines that follow.
Data fields (first 8, which is all we need): date, time (UTC), record
identifier (blank or a landfall/intensity-peak flag), status, lat, lon,
max sustained wind (kt), min central pressure (mb).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)

# HURDAT2 uses these sentinels for "not reported".
MISSING_WIND = -99
MISSING_PRESSURE = -999

# Matches the Atlantic HURDAT2 download link on NHC's data page, e.g.
#   href="/data/hurdat/hurdat2-1851-2025-02272026.txt"
# Basin-specific files (Pacific: "hurdat2-nepac-...", "hurdat2-cpac-...")
# have a basin tag right after "hurdat2-" instead of a 4-digit year, so this
# pattern naturally excludes them without needing to scope to a section of
# the page.
_ATLANTIC_HURDAT2_LINK_RE = re.compile(
    r'href="(/data/hurdat/hurdat2-\d{4}-\d{4}-\d+\.txt)"'
)


def resolve_hurdat2_url(
    fallback_url: str = config.HURDAT2_DEFAULT_URL,
    data_page_url: str = config.NHC_DATA_PAGE_URL,
) -> str:
    """Scrape the current Atlantic HURDAT2 download link off NHC's data page.

    NHC republishes HURDAT2 under a new dated filename periodically, so a
    hardcoded URL goes stale. This fetches the live page and pulls the link
    out of the "Best Track Data (HURDAT2)" section. If anything goes wrong
    -- network error, NHC restructures the page, the regex stops matching --
    this logs a warning and falls back to `fallback_url` rather than raising,
    so a broken scrape never blocks the pipeline outright.
    """
    try:
        resp = requests.get(data_page_url, timeout=30)
        resp.raise_for_status()
        match = _ATLANTIC_HURDAT2_LINK_RE.search(resp.text)
        if not match:
            raise ValueError(
                "No Atlantic HURDAT2 link found on the NHC data page "
                "(page layout may have changed)"
            )
        resolved = urljoin(data_page_url, match.group(1))
        logger.info("Resolved current Atlantic HURDAT2 URL: %s", resolved)
        return resolved
    except Exception as e:
        logger.warning(
            "Could not resolve current HURDAT2 URL from %s (%s); "
            "falling back to hardcoded URL %s",
            data_page_url,
            e,
            fallback_url,
        )
        return fallback_url


def _source_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".source.json")


def download_hurdat2(
    url: str | None = None,
    cache_path: Path = config.HURDAT2_CACHE_PATH,
    force: bool = False,
) -> Path:
    """Download the HURDAT2 text file once and cache it locally.

    If `url` is None (the default), the current Atlantic HURDAT2 URL is
    auto-resolved by scraping NHC's data page (resolve_hurdat2_url), with a
    hardcoded fallback if that fails. Pass `url` explicitly (e.g. via
    --hurdat2-url) to skip resolution entirely and force a specific source.

    Returns the cached path. Does nothing network-wise if the cache already
    exists, unless force=True -- note this also skips URL resolution on a
    cache hit; pass force=True (or delete the cache file) to re-resolve.
    """
    cache_path = Path(cache_path)
    meta_path = _source_meta_path(cache_path)

    if cache_path.exists() and not force:
        used_url = "unknown (cached before source tracking existed)"
        if meta_path.exists():
            used_url = json.loads(meta_path.read_text(encoding="utf-8"))["url"]
        print(f"HURDAT2: using cached file at {cache_path} (source: {used_url})")
        logger.info("Using cached HURDAT2 file at %s", cache_path)
        return cache_path

    resolved_url = url if url is not None else resolve_hurdat2_url()
    print(f"HURDAT2: downloading from {resolved_url}")
    logger.info("Downloading HURDAT2 best track from %s", resolved_url)
    resp = requests.get(resolved_url, timeout=60)
    resp.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(resp.text, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "url": resolved_url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    logger.info("Cached HURDAT2 file to %s (%d bytes)", cache_path, len(resp.text))
    return cache_path


def _parse_lat(raw: str) -> float:
    raw = raw.strip()
    sign = 1.0 if raw.endswith("N") else -1.0
    return sign * float(raw[:-1])


def _parse_lon(raw: str) -> float:
    raw = raw.strip()
    sign = 1.0 if raw.endswith("E") else -1.0
    return sign * float(raw[:-1])


def parse_storm(
    atcf_id: str,
    hurdat2_path: Path,
    include_statuses: tuple[str, ...] = config.DEFAULT_INCLUDE_STATUSES,
) -> tuple[str, pd.DataFrame]:
    """Parse a single storm's fixes out of the full HURDAT2 file.

    Parameters
    ----------
    atcf_id:
        e.g. "AL142018" for Hurricane Michael (2018).
    hurdat2_path:
        Path to the full Atlantic HURDAT2 text file (from download_hurdat2).
    include_statuses:
        Which best-track "status" codes to keep. Defaults to the
        tropical/subtropical lifetime (excludes extratropical remnant
        stages) -- see config.DEFAULT_INCLUDE_STATUSES.

    Returns
    -------
    (storm_name, fixes) where fixes is a DataFrame with columns:
        time (UTC, tz-aware pandas Timestamp), lat, lon, wind_kt, status
    sorted by time, with missing-wind rows dropped (logged as a warning).
    """
    atcf_id = atcf_id.strip().upper()
    lines = Path(hurdat2_path).read_text(encoding="utf-8").splitlines()

    storm_name: str | None = None
    n_expected = 0
    rows: list[dict] = []
    n_dropped_missing_wind = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("AL") or line.startswith("EP") or line.startswith("CP"):
            fields = [f.strip() for f in line.split(",")]
            header_id, header_name, header_n = fields[0], fields[1], fields[2]
            n = int(header_n)
            if header_id == atcf_id:
                storm_name = header_name
                n_expected = n
                for j in range(i + 1, i + 1 + n):
                    dfields = [f.strip() for f in lines[j].split(",")]
                    date_s, time_s, _record_id, status = dfields[0:4]
                    lat = _parse_lat(dfields[4])
                    lon = _parse_lon(dfields[5])
                    wind_kt = int(dfields[6])

                    if status not in include_statuses:
                        continue
                    if wind_kt == MISSING_WIND:
                        n_dropped_missing_wind += 1
                        continue

                    ts = pd.to_datetime(
                        f"{date_s} {time_s}", format="%Y%m%d %H%M", utc=True
                    )
                    rows.append(
                        {
                            "time": ts,
                            "lat": lat,
                            "lon": lon,
                            "wind_kt": float(wind_kt),
                            "status": status,
                        }
                    )
                i += n + 1
                break
            i += n + 1
        else:
            i += 1

    if storm_name is None:
        raise ValueError(f"ATCF id {atcf_id!r} not found in {hurdat2_path}")

    if n_dropped_missing_wind:
        logger.warning(
            "%s (%s): dropped %d/%d fixes with missing wind (-99)",
            atcf_id,
            storm_name,
            n_dropped_missing_wind,
            n_expected,
        )

    fixes = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    logger.info(
        "%s (%s): parsed %d usable fixes spanning %s to %s",
        atcf_id,
        storm_name,
        len(fixes),
        fixes["time"].min(),
        fixes["time"].max(),
    )
    return storm_name, fixes
