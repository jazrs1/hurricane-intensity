"""Central constants and paths for the data pipeline.

Everything here is a default that can be overridden via CLI flags in
scripts/build_dataset.py and scripts/make_previews.py -- nothing downstream
should hardcode these values a second time.
"""

from __future__ import annotations

from pathlib import Path

# --- Project layout -------------------------------------------------------
# src/hurricane_intensity/config.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Single reusable scratch path. Every downloaded ABI scan lands here,
# gets cropped, and is deleted before the next one is downloaded -- see
# crop.py. Never accumulates raw files.
SCRATCH_NC_PATH = RAW_DIR / "_scratch.nc"

# --- HURDAT2 ----------------------------------------------------------------
# NHC republishes this file under a new dated filename periodically. By
# default hurdat2.download_hurdat2() scrapes the current link from
# NHC_DATA_PAGE_URL instead of trusting a hardcoded name; this is only the
# fallback used if that scrape fails (network error, NHC changes their page
# layout, etc). Update it opportunistically when you notice it's stale, but
# it is not load-bearing under normal operation.
HURDAT2_DEFAULT_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"
NHC_DATA_PAGE_URL = "https://www.nhc.noaa.gov/data/"
HURDAT2_CACHE_PATH = INTERIM_DIR / "hurdat2_atlantic.txt"

# East Pacific (nepac) HURDAT2 (B-v2-2). Unlike the Atlantic file this is
# never auto-resolved: one exact NHC file, pinned by filename and SHA-256 (the
# file H-EP's protocol pins), cached under its own NHC filename so it can
# never collide with HURDAT2_CACHE_PATH. See hurdat2.fetch_pinned_hurdat2.
HURDAT2_EP_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2025-092926.txt"
HURDAT2_EP_SHA256 = "caece626f7038db64a803a73f5985b563693f0130f3ff16b4b0a0a93c7aa0507"
HURDAT2_EP_CACHE_PATH = INTERIM_DIR / "hurdat2-nepac-1949-2025-092926.txt"

# H-EP sealing in the reader (HEP.md gate (d), B-v2-2b). hurdat2.parse_storm
# refuses the H-EP storms (read from HEP_LIST at call time, never copied here)
# and their concurrency exclusions, except on the sealed build's call path:
# run_pipeline into HEP_SEALED_DIR (H-EP protocol §5). The concurrency IDs are
# Octave EP152025 and Raymond EP172025, concurrent with an H-EP storm under the
# 650 km box rule; tests/test_list_invariants.py keeps them out of training.
HEP_LIST = PROJECT_ROOT / "storms_holdout_hep.txt"
HEP_CONCURRENT_IDS = ("EP152025", "EP172025")
HEP_SEALED_DIR = DATA_DIR / "hep_sealed"

# Best-track "status" codes to keep. This excludes EX (extratropical) and
# other post-tropical remnant stages, so a storm's track is automatically
# trimmed to its tropical/subtropical lifetime without hardcoding dates --
# for Michael this naturally reproduces "roughly Oct 6-11" (track continues,
# as EX, through Oct 15 in the raw file).
DEFAULT_INCLUDE_STATUSES = ("LO", "TD", "SD", "TS", "SS", "HU")

# --- GOES-16 ABI on AWS Open Data -------------------------------------------
GOES_BUCKET = "noaa-goes16"
ABI_PRODUCT = "ABI-L2-CMIPF"  # full-disk, Cloud & Moisture Imagery -> pre-calibrated BT
ABI_BAND = 13  # 10.3 um "clean" longwave IR
ABI_VARIABLE = "CMI"  # brightness temperature in Kelvin, already calibrated

# --- Crop parameters ---------------------------------------------------------
DEFAULT_BOX_KM = 600.0
DEFAULT_SAMPLE_INTERVAL = "1h"
DEFAULT_TOLERANCE_MIN = 10.0
