#!/usr/bin/env python
"""List the analog-forecasting storms and probe GOES band 13 coverage; writes configs/analog_storms.csv.

    .venv\\Scripts\\python.exe scripts\\analog_storms.py

Storms: every Atlantic (AL) and Northeast Pacific (EP) storm of 2017-2025 in HURDAT2 with at least one
record of status TS or HU at >= 35 kt. Central Pacific (CP) IDs in the nepac file are not included.
Never listed: the six spent 2025 test storms (SPENT_2025), malformed IDs, and the sealed H-EP storms and
their concurrency exclusions (hurdat2.sealed_ids()). Sealed storms are skipped on their header line, so none
of their records is ever parsed; only their count is printed.

Satellite per storm, by the storm's first fix (reports/v2_survey/survey.md, satellite timeline):
    AL: noaa-goes16, or noaa-goes19 from 2025-04-07
    EP: noaa-goes17 from 2019-02-12 18Z, noaa-goes18 from 2023-01-04 18Z; before 2019-02-12 GOES-West had no
        ABI, so noaa-goes16 (the fallback) if its full disk covers the storm.
Coverage probe: band 13 full-disk keys are listed (no download) for up to PROBES sample fixes spread over the
storm's pipeline lifetime (statuses config.DEFAULT_INCLUDE_STATUSES), and the viewing zenith angle (VZA) of
every fix is computed from the satellite's nominal sub-point longitude (spherical Earth). A storm is usable
if at least half its probed hours have a scan and at least half its fixes have VZA <= VZA_MAX.
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hurricane_intensity import config, goes_index, hurdat2
from hurricane_intensity.dataset import SPENT_2025, STORM_ID

OUT = config.PROJECT_ROOT / "configs" / "analog_storms.csv"
YEARS = range(2017, 2026)
PROBES = 6
VZA_MAX = 70.0
G19_EAST = datetime(2025, 4, 7, tzinfo=timezone.utc)
G17_WEST = datetime(2019, 2, 12, 18, tzinfo=timezone.utc)
G18_WEST = datetime(2023, 1, 4, 18, tzinfo=timezone.utc)


def subpoint_lon(bucket, when):
    """Nominal sub-satellite longitude. GOES-16 sat at its 89.5W checkout slot until 2017-11-30 (moved to
    75.2W by 2017-12-11); the probe uses 89.5W for any 2017 date before 2017-12-01."""
    if bucket == "noaa-goes16":
        return -89.5 if when < datetime(2017, 12, 1, tzinfo=timezone.utc) else -75.2
    return {"noaa-goes17": -137.2, "noaa-goes18": -137.0, "noaa-goes19": -75.2}[bucket]


def vza_deg(lat, lon, sat_lon):
    """Viewing zenith angle (deg) of a point from a geostationary satellite, spherical Earth."""
    re_, h = 6371.0, 35786.0 + 6371.0
    lat, dlon = np.radians(lat), np.radians(np.asarray(lon) - sat_lon)
    cos_c = np.cos(lat) * np.cos(dlon)  # central angle between point and sub-point
    c = np.arccos(np.clip(cos_c, -1, 1))
    slant = np.sqrt(re_ ** 2 + h ** 2 - 2 * re_ * h * cos_c)
    return np.degrees(np.arcsin(np.clip(h * np.sin(c) / slant, -1, 1)))


def candidate_ids(path, prefix, sealed):
    """(IDs in YEARS with this basin prefix, number of sealed IDs skipped), from header lines only."""
    ids, n_sealed = [], 0
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        parts = [p.strip() for p in lines[i].split(",")]
        n = int(parts[2])
        atcf = parts[0].upper()
        if atcf.startswith(prefix) and int(atcf[-4:]) in YEARS:
            if atcf in sealed:
                n_sealed += 1
            elif STORM_ID.fullmatch(atcf) and atcf not in SPENT_2025:
                ids.append(atcf)
        i += n + 1  # skip the data lines without parsing them
    return ids, n_sealed


def bucket_for(basin, first):
    if basin == "AL":
        return "noaa-goes19" if first >= G19_EAST else "noaa-goes16"
    if first >= G18_WEST:
        return "noaa-goes18"
    if first >= G17_WEST:
        return "noaa-goes17"
    return "noaa-goes16"


def probe(row, s3):
    times = row.pop("_times")
    idx = np.unique(np.linspace(0, len(times) - 1, min(PROBES, len(times))).round().astype(int))
    hits = 0
    for i in idx:
        t = times[i].to_pydatetime().replace(minute=0, second=0)
        if goes_index._list_keys_for_hour(s3, row["bucket"], config.ABI_PRODUCT, config.ABI_BAND, t):
            hits += 1
    row["probe_hits"] = f"{hits}/{len(idx)}"
    reasons = []
    if hits * 2 < len(idx):
        reasons.append("no band 13 scans for most of the lifetime")
    if row["frac_vza_ok"] < 0.5:
        reasons.append(f"VZA > {VZA_MAX:g} deg for most fixes")
    row["usable"] = int(not reasons)
    row["reason"] = "; ".join(reasons)
    return row


def main():
    sealed = hurdat2.sealed_ids()
    ep_path = hurdat2.verify_pinned_hurdat2(config.HURDAT2_EP_CACHE_PATH)
    sources = [("AL", config.HURDAT2_CACHE_PATH), ("EP", ep_path)]
    rows = []
    for basin, path in sources:
        ids, n_sealed = candidate_ids(path, basin, sealed)
        print(f"{basin}: {len(ids)} candidate IDs 2017-2025; {n_sealed} sealed IDs skipped unread")
        for atcf in ids:
            try:
                name, all_fixes = hurdat2.parse_storm(atcf, path, include_statuses=("TS", "HU"))
            except KeyError:  # no TS/HU record at all: parse_storm cannot build an empty frame
                continue
            if all_fixes.empty or all_fixes["wind_kt"].max() < 35:
                continue
            _, fixes = hurdat2.parse_storm(atcf, path)  # pipeline lifetime
            first = fixes["time"].iloc[0].to_pydatetime()
            bucket = bucket_for(basin, first)
            sat = np.array([subpoint_lon(bucket, t.to_pydatetime()) for t in fixes["time"]])
            vza = vza_deg(fixes["lat"].to_numpy(), hurdat2_lon(fixes["lon"].to_numpy(), sat), sat)
            rows.append({
                "atcf_id": atcf, "basin": basin, "year": int(atcf[-4:]), "name": name,
                "first_fix": fixes["time"].iloc[0].strftime("%Y-%m-%dT%H:%MZ"),
                "last_fix": fixes["time"].iloc[-1].strftime("%Y-%m-%dT%H:%MZ"),
                "bucket": bucket, "median_vza": round(float(np.median(vza)), 1),
                "frac_vza_ok": round(float((vza <= VZA_MAX).mean()), 3), "_times": list(fixes["time"]),
            })
    s3 = goes_index.make_anonymous_s3_client()
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(lambda r: probe(r, s3), rows))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    usable = [r for r in rows if r["usable"]]
    for basin in ("AL", "EP"):
        print(f"{basin}: {sum(r['basin'] == basin for r in rows)} storms >= 35 kt TS/HU, "
              f"{sum(r['basin'] == basin for r in usable)} usable")
    for r in rows:
        if not r["usable"]:
            print(f"  drop {r['atcf_id']} {r['name']} ({r['bucket']}, probes {r['probe_hits']}, "
                  f"median VZA {r['median_vza']}): {r['reason']}")
    print(f"wrote {OUT}")


def hurdat2_lon(lon, sat_lon):
    """Longitude wrapped to within 180 deg of the satellite (tracks crossing 180 stay continuous)."""
    return sat_lon + (np.asarray(lon) - sat_lon + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    main()
