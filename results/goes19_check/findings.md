# GOES-19 readability check (one sample scan)

A check that the pipeline's modules can read GOES-19 ABI data, ahead of
the 2025 final test (design in `logs/decisions.md`, 2026-10-15). Full
attribute dumps of both scans are in `goes19_vs_goes16_scan_check.json`.

## What was done

- `goes_index.find_nearest_scan(bucket="noaa-goes19")` for 2025-06-15
  12:00 UTC, full-disk CMIP, band 13, 10 min tolerance. This is outside
  all six test storms' lifetimes (the earliest, Dexter, is August).
- `crop.download_scan(bucket="noaa-goes19")`, then `crop.extract_crop` at
  an arbitrary point (20 N, 60 W, 600 km box).
- The same steps for a GOES-16 reference scan at 2024-06-15 12:00 UTC,
  and the two dumps were compared field by field.
- Files went to the session scratch directory, not `data/raw/`, and both
  were deleted afterwards.
- No test-storm frame was built and no model was run.

Scans used:
- GOES-19: `ABI-L2-CMIPF/2025/166/12/OR_ABI-L2-CMIPF-M6C13_G19_s20251661200206_e20251661209525_c20251661209576.nc`
- GOES-16: `ABI-L2-CMIPF/2024/167/12/OR_ABI-L2-CMIPF-M6C13_G16_s20241671200212_e20241671209532_c20241671210010.nc`

## Result: readable; the only data difference is the instrument

| | GOES-16 (2024) | GOES-19 (2025) |
|---|---|---|
| bucket | noaa-goes16 | noaa-goes19 |
| file naming | `OR_ABI-L2-CMIPF-M6C13_G16_s..._e..._c....nc` | same, `G19` |
| key prefix | `ABI-L2-CMIPF/YYYY/DDD/HH/` | same |
| scan mode | Mode 6, full disk, 10 min | same |
| instrument / platform ID | FM1 / G16 | **FM4 / G19** |
| production site | NSOF | RBU |
| `goes_imager_projection` | lon_0 -75.0, h 35786023 m, GRS80, sweep x | identical |
| grid | 5424 x 5424, 5.6e-5 rad | identical |
| `CMI` | int16, scale 0.061453, offset 89.62 K, valid 0-4095, fill -1, units K | identical |
| `DQF` attributes | same flag definitions | identical (only the per-scan percentages differ) |
| Planck coefficients (fk1, fk2, bc1) | 10803.30, 1392.74, 0.07550 | **10860.40, 1395.19, 0.07481** |
| crop at 20 N 60 W | 273 x 311, float32, no NaN, DQF all 0 | 273 x 311, float32, no NaN, DQF all 0 |

- File naming: `goes_index._KEY_RE` already treats the platform as a
  wildcard (`G\d{2}`), and the GOES-19 key matched it unchanged.
- Projection: the metadata is identical, and `projection.py` builds the
  CRS from each file's own metadata, so nothing changes.
- Band 13 calibration: `CMI` is already brightness temperature in kelvin,
  packed identically. It is computed with each instrument's own Planck
  coefficients, and those differ: FM4's band-13 spectral response is not
  FM1's. The data are correctly calibrated BT on both satellites, but the
  same scene can read slightly differently in C13 between them. This
  check does not measure the size of that (the two crops are different
  days and scenes, so their BT values are not comparable). It is part of
  what the 2025 test measures, and it is not corrected.
- Production site (RBU vs NSOF) is file provenance; it does not change
  the data format.
- GOES-16 in 2025: the `noaa-goes16` bucket has band-13 full-disk scans
  on 2025-04-01 but none on 2025-09-01. The six storms (August to
  October) can only be built from GOES-19, so the test is necessarily
  cross-satellite.

## Gap: the pipeline does not pass the bucket through

`find_nearest_scan` and `download_scan` take `bucket`, but
`pipeline.run_pipeline` passes it to neither, and
`scripts/build_dataset.py` has no flag for it. Today a build silently
uses `noaa-goes16`, which for these storms would find no scans at all.
The design lists the small plumbing change needed (a `--bucket` flag,
defaulting to `noaa-goes16` so existing builds are unchanged) as a step
before any 2025 build. It is not implemented here.
