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

import hashlib
import json
import logging
import os
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_atlantic_cache(path: Path) -> bool:
    """True if `path` is the protected Atlantic cache (same path, or a link to the same file)."""
    path, atlantic = Path(path), Path(config.HURDAT2_CACHE_PATH)
    if path.resolve() == atlantic.resolve():
        return True
    return path.exists() and atlantic.exists() and os.path.samefile(path, atlantic)


def _write_new(path: Path, data: bytes) -> None:
    """Write `data` to a fresh file: a stale file (or a link) at `path` is unlinked, never written through."""
    path.unlink(missing_ok=True)
    with open(path, "xb") as f:
        f.write(data)


def fetch_pinned_hurdat2(url: str, sha256: str, cache_path: Path) -> Path:
    """Return a verified local copy of one pinned HURDAT2 file (B-v2-2).

    For the basin files (config.HURDAT2_EP_*), not the Atlantic one: there
    is no URL resolution and no force/refresh. The cache file must carry the
    pinned NHC filename and may never be the Atlantic cache. A cached copy is
    re-hashed and refused on mismatch, never overwritten. A fresh download is
    hashed before anything is written, so a wrong file never reaches disk;
    the SHA-256 is recorded in the .source.json next to it.
    """
    cache_path = Path(cache_path)
    meta_path = _source_meta_path(cache_path)
    partial = cache_path.with_name(cache_path.name + ".partial")
    if any(_is_atlantic_cache(p) for p in (cache_path, meta_path, partial)):
        raise ValueError(f"{cache_path}: refusing to use the protected Atlantic HURDAT2 cache")
    if cache_path.name != url.rsplit("/", 1)[-1]:
        raise ValueError(f"{cache_path}: the cache filename must be the pinned file's name ({url})")

    if cache_path.exists():
        got = _sha256_file(cache_path)
        if got != sha256:
            raise ValueError(f"{cache_path} has SHA-256 {got}, pinned {sha256}; refusing (not overwritten)")
        print(f"HURDAT2: using pinned cached file at {cache_path} (SHA-256 {got})")
        return cache_path

    print(f"HURDAT2: downloading pinned file {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    got = hashlib.sha256(resp.content).hexdigest()
    if got != sha256:
        raise ValueError(f"{url} has SHA-256 {got}, pinned {sha256}; refusing (nothing written)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new(partial, resp.content)  # raw bytes: the pin is of the bytes NHC serves
    os.replace(partial, cache_path)
    meta = {"url": url, "sha256": got, "fetched_at": datetime.now(timezone.utc).isoformat()}
    _write_new(meta_path, json.dumps(meta).encode("utf-8"))
    logger.info("Cached pinned HURDAT2 file to %s (%d bytes)", cache_path, len(resp.content))
    return cache_path


def fetch_ep_hurdat2() -> Path:
    """The pinned East Pacific (nepac) HURDAT2, fetched once into its own cache."""
    return fetch_pinned_hurdat2(config.HURDAT2_EP_URL, config.HURDAT2_EP_SHA256, config.HURDAT2_EP_CACHE_PATH)


def verify_pinned_hurdat2(path: Path) -> Path:
    """Check an explicitly given HURDAT2 path (--hurdat2-path) before it is read.

    Only pinned basin files are accepted, by filename and SHA-256. The
    Atlantic cache is refused here: Atlantic builds use the default path.
    """
    pins = {Path(config.HURDAT2_EP_CACHE_PATH).name: config.HURDAT2_EP_SHA256}
    path = Path(path)
    if _is_atlantic_cache(path):
        raise ValueError(f"{path}: the Atlantic cache is not a --hurdat2-path file; omit --hurdat2-path")
    if path.name not in pins:
        raise ValueError(f"{path}: not a pinned HURDAT2 file (pinned: {sorted(pins)})")
    if not path.is_file():
        raise FileNotFoundError(f"{path}: fetch it first with scripts/fetch_hurdat2_ep.py")
    got = _sha256_file(path)
    if got != pins[path.name]:
        raise ValueError(f"{path} has SHA-256 {got}, pinned {pins[path.name]}; refusing")
    return path


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

    Refuses the sealed H-EP storms and their concurrency exclusions, from any
    file (refuse_sealed). The sealed build reads them only through
    run_pipeline -> _parse_sealed_hep_storm.
    """
    refuse_sealed(atcf_id)
    return _parse_storm(atcf_id, hurdat2_path, include_statuses)


def sealed_ids() -> frozenset[str]:
    """The H-EP storms (config.HEP_LIST, read now) plus their concurrency exclusions.

    Fails closed: a missing or empty list raises rather than sealing nothing.
    """
    from .splits import load_storm_list

    hep = load_storm_list(config.HEP_LIST)  # FileNotFoundError if missing
    if not hep:
        raise ValueError(f"{config.HEP_LIST} names no storm; refusing to read without the sealed set")
    return frozenset(hep) | frozenset(config.HEP_CONCURRENT_IDS)


def refuse_sealed(atcf_id: str) -> None:
    """Raise if `atcf_id` is sealed (H-EP gate (d)). The message carries the ID only."""
    if atcf_id.strip().upper() in sealed_ids():
        raise SealedStormError(
            f"{atcf_id.strip().upper()} is sealed (H-EP set or its concurrency exclusions); "
            "it is read only by the sealed H-EP build (run_pipeline into config.HEP_SEALED_DIR)"
        )


class SealedStormError(ValueError):
    pass


def _parse_sealed_hep_storm(
    atcf_id: str,
    hurdat2_path: Path,
    include_statuses: tuple[str, ...],
    out_root: Path,
) -> tuple[str, pd.DataFrame]:
    """The one exception to refuse_sealed: the sealed H-EP build (protocol §5-6).

    Its only caller is pipeline.run_pipeline (pinned by a test). It re-checks
    everything itself rather than trusting the caller: the output root is
    config.HEP_SEALED_DIR, the file is the pinned East Pacific HURDAT2, and the
    ID is one of the H-EP storms (the concurrency exclusions are never read,
    not even here).
    """
    from .splits import load_storm_list

    atcf_id = atcf_id.strip().upper()
    if Path(out_root).resolve() != Path(config.HEP_SEALED_DIR).resolve():
        raise SealedStormError(f"{atcf_id}: the sealed read writes only to {config.HEP_SEALED_DIR}")
    if atcf_id not in load_storm_list(config.HEP_LIST):
        raise SealedStormError(f"{atcf_id} is not an H-EP storm; the sealed read is for those only")
    verify_pinned_hurdat2(hurdat2_path)
    return _parse_storm(atcf_id, hurdat2_path, include_statuses)


def read_positions(atcf_id: str, hurdat2_path: Path) -> pd.DataFrame:
    """Positions only, for E3's concurrency gate (E3 design §5, §9).

    Returns a DataFrame with exactly the columns time, lat, lon: every fix,
    no status trim, sorted by time. Wind and status are parsed (the format is
    checked) but never returned, and nothing is logged or printed. Errors carry
    the ID and a line number, never a name, wind or status. Sealed IDs are
    readable here: loading the excluded storms' tracks is this function's
    purpose. The file must be the pinned East Pacific HURDAT2.
    """
    atcf_id = atcf_id.strip().upper()
    lines = Path(verify_pinned_hurdat2(hurdat2_path)).read_text(encoding="utf-8").splitlines()
    rows: list[tuple] = []
    found = False
    i = 0
    while i < len(lines) and not found:
        fields = [f.strip() for f in lines[i].split(",")]
        if lines[i][:2] in ("AL", "EP", "CP"):
            try:
                n = int(fields[2])
            except (IndexError, ValueError):
                raise ValueError(f"malformed header line {i + 1} while reading {atcf_id}") from None
            if fields[0] == atcf_id:
                found = True
                for j in range(i + 1, i + 1 + n):
                    try:
                        d = [f.strip() for f in lines[j].split(",")]
                        int(d[6]), d[3]  # wind and status: parsed for format, never kept
                        rows.append((
                            pd.to_datetime(f"{d[0]} {d[1]}", format="%Y%m%d %H%M", utc=True),
                            _parse_lat(d[4]),
                            _parse_lon(d[5]),
                        ))
                    except (IndexError, ValueError):
                        raise ValueError(f"malformed fix line {j + 1} for {atcf_id}") from None
            i += n + 1
        else:
            i += 1
    if not found:
        raise ValueError(f"ATCF id {atcf_id!r} not found in {Path(hurdat2_path).name}")
    return pd.DataFrame(rows, columns=["time", "lat", "lon"]).sort_values("time").reset_index(drop=True)


def _parse_storm(
    atcf_id: str,
    hurdat2_path: Path,
    include_statuses: tuple[str, ...] = config.DEFAULT_INCLUDE_STATUSES,
) -> tuple[str, pd.DataFrame]:
    """parse_storm without the sealed-ID refusal. Call parse_storm instead."""
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
