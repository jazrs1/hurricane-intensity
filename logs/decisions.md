# Decisions log

## 2026-08-18 — Data pipeline (Michael, AL142018)

**Data product**: GOES-16 `ABI-L2-CMIPF` (full-disk Cloud & Moisture Imagery),
band 13 (10.3 um clean IR). Chosen over raw `ABI-L1b-Rad` because CMIP is
already calibrated to brightness temperature (Kelvin) in-file — no manual
Planck-function inversion needed, fewer places for a calibration bug.

**Track sampling**: sample grid = regular grid at `--sample-interval`
(default `1h`) unioned with every exact HURDAT2 fix time. Between fixes,
both position (lat/lon) and wind (kt) are **linearly interpolated** —
explicitly not held constant, to avoid step artifacts in the wind label.
Every saved sample carries `is_interpolated: bool` (false only for exact
HURDAT2 fix timestamps) in both the manifest and the per-sample .npz.
Rationale for interval default: 15-min native ABI cadence produces
near-duplicate frames that inflate the dataset without adding information;
hourly gives ~120 samples for Michael's ~5-day tropical lifetime.

**Storm lifetime cutoff**: fixes filtered to HURDAT2 status codes
`LO,TD,SD,TS,SS,HU` (excludes `EX`/extratropical). This reproduces
"roughly Oct 6-11" for Michael without hardcoding dates, and generalizes to
other storms' post-tropical remnant tails automatically.

**Scan matching tolerance**: nearest ABI full-disk scan (~15 min cadence)
must be within ±10 minutes of the sample time or the sample is skipped and
logged (`match_status=missing_scan`), pipeline continues. Actual signed
offset (`scan_time - target_time`, minutes) is recorded per matched sample;
an end-of-run summary reports match/skip counts and the offset distribution.

**Crop box**: 600 km square in real distance, computed via local flat-Earth
lat/lon delta at the storm's latitude (not a fixed pixel box) — see
`projection._box_corners_latlon` docstring for the accuracy tradeoff. Corners
are reprojected into the ABI fixed-grid (scan-angle) coordinates via pyproj
using each file's own `goes_imager_projection` metadata, not assumed to be a
plain lat/lon grid.

**Disk usage**: one reusable scratch path (`data/raw/_scratch.nc`),
download -> crop into memory -> delete before the next file, enforced even
on exceptions (`crop.process_one_scan`). `safe_unlink` retries through
Windows file-lock delays via `gc.collect()` + retry rather than trusting
`.close()` alone.
