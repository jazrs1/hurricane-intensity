"""Export three co-registered images of one GOES scene for the site's band switcher.

Writes <out-dir>/c02.webp, c09.webp and c13.webp -- all the same size, the
same lat/lon extent, north up, with identically feathered transparent edges --
plus hero.json.

Scan listing, download, decoding, projection and cropping go through
src/hurricane_intensity (goes_index, crop, projection). This script adds
matching the three channels to one scan, resampling onto one lat/lon grid,
and rendering.

Usage:
    python scripts/export_hero.py
    python scripts/export_hero.py --half-width 3.5 --c02-white-point 0.8
    python scripts/export_hero.py --half-width 1.9 --alpha-mode radial --radial-radius 0.5 --luminance-weight 1.5
    python scripts/export_hero.py --channels 2 --half-width 4 5 6 --c02-white-point 0.27 --feather-width 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import metpy
import numpy as np
import pyproj
import xarray as xr
from metpy.plots import ctables
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hurricane_intensity import config, crop, goes_index, projection  # noqa: E402
from make_previews import BT_MAX_K, BT_MIN_K, bt_to_image  # noqa: E402

CHANNELS = (2, 9, 13)
SECTOR = "F"  # full disk; goes_index matches filenames by sector
KM_PER_DEG = 111.32  # the flat-Earth constant projection._box_corners_latlon uses

# Scan angle kept beyond the lat/lon box when cropping, so the outermost
# sample points still land on a source pixel. 2 x 56 urad = two 2 km pixels.
CROP_PAD_RAD = 2 * 56e-6

# Automatic supersampling: enough sample points per output pixel that each
# 0.5 km C02 pixel is hit ~3 times across, capped to bound the work.
C02_PIXEL_KM = 0.5
SAMPLES_PER_C02_PIXEL = 3
MAX_SUPERSAMPLE = 8
ROWS_PER_CHUNK = 25  # output rows per projection batch, to bound memory

# C13 and C09 colors come from MetPy's colortable registry (metpy.plots.ctables),
# which ships GEMPAK's image lookup tables (github.com/Unidata/gempak,
# gempak/tables/luts). A GEMPAK table holds one color per 8-bit image count,
# 0-255. MetPy's copies drop GEMPAK's leading all-black entries, so for some
# tables MetPy index 0 sits at a nonzero count (checked entry for entry against
# the GEMPAK files). Counts convert to brightness temperature by the McIDAS/GINI
# calibration those images used:
#   T = (660 - count) / 2 K for count < 176, T = 418 - count K otherwise.
# Under that mapping ir_bd's steps land on the published Dvorak BD thresholds.
# Both tables are applied at their native counts, so a color means the same
# brightness temperature on every scene.
C13_TABLE = "ir_bd"  # GEMPAK bd.tbl: the Dvorak BD curve, in gray steps
C09_TABLE = "WVCIMSS"  # GEMPAK WVCIMSS.tbl
TABLE_COUNT_OFFSETS = {C13_TABLE: 58, C09_TABLE: 110}  # image count of MetPy index 0
STEP_JUMP = 30  # summed RGB change between adjacent entries that counts as a visible step

# --c13-style rbtop: NOAA OSPO's McIDAS RBTOP enhancement, applied unmodified at its native brightness counts
# (the same count <-> K calibration as above). Rows exactly as published in the EU TABLE text file: count min,
# count max, then blue, green and red, each as min and max, interpolated linearly across the row.
RBTOP_SOURCE = "https://www.ospo.noaa.gov/data/FAQ/text/RBTOP.txt"
RBTOP_ROWS = (
    (0, 60, 0, 120, 0, 120, 0, 120),
    (61, 90, 3, 90, 3, 90, 3, 90),
    (91, 100, 91, 100, 91, 100, 91, 100),
    (101, 144, 103, 248, 103, 248, 103, 248),
    (145, 154, 250, 255, 250, 0, 250, 191),
    (155, 170, 255, 255, 0, 0, 191, 0),
    (171, 190, 242, 0, 12, 255, 0, 0),
    (191, 200, 0, 0, 255, 255, 25, 255),
    (201, 210, 0, 0, 229, 0, 255, 255),
    (211, 220, 0, 0, 0, 0, 229, 0),
    (221, 245, 20, 255, 20, 255, 20, 255),
    (246, 255, 255, 255, 255, 255, 255, 255),
)

# --c13-style custom: a CUSTOM colorization, not a published enhancement. The published tables put their colors at
# fixed temperatures, and on a storm as cold as Melissa (core 193-200 K) they paint the whole central overcast in one
# or two flat tones. This one places its steps at percentiles of the scene's own C13 brightness temperatures, so each
# step holds a chosen share of the pixels. Warm to cold: grays, green, yellow, orange, red, then black to white for
# the coldest tops. Each step ramps between two colors, so texture inside a step survives. Because the steps follow
# this scene's histogram, a color does not mean the same temperature on another scene; hero.json records the exact
# breakpoints in K.
CUSTOM_STEPS = (  # (name, color at the step's warm edge, color at its cold edge)
    ("gray", (40, 40, 40), (205, 205, 205)),
    ("green", (180, 228, 160), (24, 128, 44)),
    ("yellow", (255, 246, 150), (224, 188, 0)),
    ("orange", (255, 196, 120), (220, 100, 0)),
    ("red", (255, 110, 110), (150, 0, 0)),
    ("black-white", (0, 0, 0), (255, 255, 255)),
)
CUSTOM_PERCENTILES = (80.0, 62.0, 45.0, 27.0, 10.0, 1.0)  # step boundaries, warm to cold; the last is where tops reach white
CUSTOM_WARM_K = 300.0  # warm (dark) end of the gray ramp; anything warmer stays at the darkest gray

UNITS = {2: "reflectance", 9: "K", 13: "K"}
WEBP_QUALITY = 90
EDGE_RING_PX = 20  # border width for the per-export edge statistics

# Edge feathering: alpha rises from the border to fully opaque over the feather
# width, as the product of one ramp per axis. Each edge gets a straight soft
# inset; only the corner squares (width x width) fade on both axes, so the
# corners keep far more of the scene than a circular vignette would.
FEATHER_SHAPES = {
    "linear": lambda t: t,
    "smoothstep": lambda t: t * t * (3.0 - 2.0 * t),
    "cosine": lambda t: 0.5 - 0.5 * np.cos(np.pi * t),
}

# Radial alpha (--alpha-mode radial): one mask, computed from C02's brightness and
# applied unchanged to all three bands, so the outline is pixel-identical and only
# the colormap changes on a band switch. C02 is reflectance, so bright there means
# cloud; C13's BD table and C09's WVCIMSS don't track cloud with brightness. The
# mask fades with distance from the centre, pushed outward where C02 is bright and
# inward where it is dark, so ocean dissolves early and cloud persists. Luminance is
# blurred first so texture doesn't speckle the edge. A second, independent fade
# (--outer-fade-start/--outer-fade-end) takes alpha to 0 by r = 1 at the latest, so
# no square edge can show; starting it well inside r = 1 lets the brightest cloud
# taper off rather than stop at a disc.
ALPHA_MODES = ("square", "radial")
MASK_CHANNEL = 2  # the radial mask comes from this channel's rendered image
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])  # Rec. 709 luma of the rendered RGB
LUMA_BLUR = 0.012  # Gaussian blur radius for luminance, as a fraction of the image side


@dataclass
class BandCrop:
    channel: int
    values: np.ndarray  # float32 CMI crop: reflectance factor (C02) or brightness temperature (K)
    x: np.ndarray  # scan-angle radians of each column, file order
    y: np.ndarray  # scan-angle radians of each row, file order (north first)
    crs_wkt: str
    sat_height_m: float
    platform_id: str
    scan_start: str  # the file's time_coverage_start
    lat: float  # center and half-width this crop was cut for, for cache reuse
    lon: float
    half_width: float

    @property
    def crs(self) -> pyproj.CRS:
        return pyproj.CRS.from_wkt(self.crs_wkt)


def match_scan(s3_client, bucket, product, target, channels=CHANNELS):
    """List the target hour for each requested channel and pick the scan start
    nearest `target` that they all share. The channels are separate files
    whose end/creation stamps differ, but one scan has one start time."""
    hour = target.replace(minute=0, second=0, microsecond=0)
    print(f"listing s3://{bucket}/{product}/{hour:%Y/%j/%H}/  ({target:%Y-%m-%d} is day {target:%j})")
    by_channel = {}
    for ch in channels:
        found = goes_index._list_keys_for_hour(s3_client, bucket, product, ch, hour, SECTOR)
        by_channel[ch] = dict(found)
        print(f"  C{ch:02d}: {len(found)} files")

    starts = sorted(set.intersection(*(set(d) for d in by_channel.values())))
    if not starts:
        raise SystemExit(f"no scan start in that hour has all of {', '.join(f'C{ch:02d}' for ch in channels)}")
    start = min(starts, key=lambda t: abs((t - target).total_seconds()))
    print(f"  scan starts with all requested channels: {', '.join(f'{t:%H:%M:%S}' for t in starts)}")
    print(f"  nearest to {target:%H:%M:%S}Z: {start:%H:%M:%S}Z ({(start - target).total_seconds() / 60:+.2f} min)")
    keys = {ch: by_channel[ch][start] for ch in channels}
    for ch, key in keys.items():
        print(f"    C{ch:02d}  {key.rsplit('/', 1)[-1]}")
    return start, keys


def crop_km_for_box(lat, lon, half_width, crs, sat_height_m):
    """box_km for projection.crop_box whose selection contains the whole
    lat/lon box plus CROP_PAD_RAD.

    crop_box selects the scan-angle bounding box of a km box's four corners.
    A degree box's edges are curves in scan angle, so trace them rather than
    assume the corners bound them, then grow box_km until the selection fits.
    """
    t = np.linspace(-half_width, half_width, 201)
    edge_lat = np.concatenate([np.full_like(t, lat + half_width), np.full_like(t, lat - half_width), lat + t, lat + t])
    edge_lon = np.concatenate([lon + t, lon + t, np.full_like(t, lon - half_width), np.full_like(t, lon + half_width)])
    ex, ey = projection.latlon_to_abi_xy(edge_lat, edge_lon, crs, sat_height_m)

    box_km = 2 * half_width * KM_PER_DEG
    for _ in range(100):
        corners = [
            projection.latlon_to_abi_xy(clat, clon, crs, sat_height_m)
            for clat, clon in projection._box_corners_latlon(lat, lon, box_km)
        ]
        cx = [c[0] for c in corners]
        cy = [c[1] for c in corners]
        if (
            min(cx) <= ex.min() - CROP_PAD_RAD and max(cx) >= ex.max() + CROP_PAD_RAD
            and min(cy) <= ey.min() - CROP_PAD_RAD and max(cy) >= ey.max() + CROP_PAD_RAD
        ):
            return box_km
        box_km *= 1.05
    raise ValueError(f"could not size a crop around the {half_width:g} deg box")


def _read_cache(path: Path) -> BandCrop:
    with np.load(path) as z:
        return BandCrop(
            channel=int(z["channel"]), values=z["values"], x=z["x"], y=z["y"],
            crs_wkt=str(z["crs_wkt"]), sat_height_m=float(z["sat_height_m"]),
            platform_id=str(z["platform_id"]), scan_start=str(z["scan_start"]),
            lat=float(z["lat"]), lon=float(z["lon"]), half_width=float(z["half_width"]),
        )


def _write_cache(path: Path, band: BandCrop) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, channel=band.channel, values=band.values, x=band.x, y=band.y,
        crs_wkt=np.str_(band.crs_wkt), sat_height_m=band.sat_height_m,
        platform_id=np.str_(band.platform_id), scan_start=np.str_(band.scan_start),
        lat=band.lat, lon=band.lon, half_width=band.half_width,
    )


def load_crop(s3_client, bucket, key, channel, lat, lon, half_width, cache_dir: Path | None) -> BandCrop:
    """Download -> crop -> delete for one channel file, the same pattern as
    crop.process_one_scan but keeping the crop's scan-angle coordinates and
    projection, which crop.extract_crop drops and resampling needs.

    The crop (not the raw file) is cached, so re-rendering at the same or a
    smaller half-width, or with a different stretch, skips the ~300 MB C02
    download.
    """
    name = key.rsplit("/", 1)[-1]
    cache_path = cache_dir / f"{Path(name).stem}.npz" if cache_dir else None
    if cache_path and cache_path.exists():
        cached = _read_cache(cache_path)
        if np.isclose(cached.lat, lat) and np.isclose(cached.lon, lon) and cached.half_width >= half_width:
            print(f"  C{channel:02d}: cached crop (cut for half-width {cached.half_width:g} deg)")
            return cached

    print(f"  C{channel:02d}: downloading {name}")
    crop.download_scan(s3_client, key, config.SCRATCH_NC_PATH, bucket=bucket)
    try:
        with xr.open_dataset(config.SCRATCH_NC_PATH) as ds:
            crs, sat_height_m = projection.get_geostationary_crs(ds)
            box_km = crop_km_for_box(lat, lon, half_width, crs, sat_height_m)
            da = projection.crop_box(ds[config.ABI_VARIABLE], lat, lon, box_km, crs, sat_height_m)
            band = BandCrop(
                channel=channel,
                values=np.array(da.values, dtype=np.float32),
                x=np.array(da["x"].values, dtype=np.float64),
                y=np.array(da["y"].values, dtype=np.float64),
                crs_wkt=crs.to_wkt(),
                sat_height_m=sat_height_m,
                platform_id=str(ds.attrs["platform_ID"]),
                scan_start=str(ds.attrs["time_coverage_start"]),
                lat=lat, lon=lon, half_width=half_width,
            )
            del da
    finally:
        crop.safe_unlink(config.SCRATCH_NC_PATH)

    if cache_path:
        _write_cache(cache_path, band)
    return band


def supersample_factor(pixel_deg: float, lat: float) -> int:
    """Sample points per output-pixel side, from the narrower (E-W) side."""
    pixel_km = pixel_deg * KM_PER_DEG * math.cos(math.radians(lat))
    wanted = math.ceil(SAMPLES_PER_C02_PIXEL * pixel_km / C02_PIXEL_KM)
    return int(min(MAX_SUPERSAMPLE, max(2, wanted)))


def resample_to_latlon(crops: dict[int, BandCrop], lat, lon, half_width, size, ss) -> dict[int, np.ndarray]:
    """Area-averaged resample of every channel onto one size x size lat/lon
    grid spanning lat/lon +- half_width, row 0 north, column 0 west.

    Each output pixel gets ss x ss evenly spaced sample points. Those points
    are projected once into ABI scan angle; each channel then looks up the
    source pixel each point falls in (by that file's own x/y coordinates) and
    averages. All channels share the same points, which is what aligns them.
    """
    ref = next(iter(crops.values()))
    crs, sat_height_m = ref.crs, ref.sat_height_m
    pixel_deg = 2 * half_width / size
    # Sample positions in output-pixel units: ss points inside each pixel.
    offsets = (np.arange(size)[:, None] + (np.arange(ss) + 0.5) / ss).ravel()
    sample_lon = lon - half_width + offsets * pixel_deg
    sample_lat = lat + half_width - offsets * pixel_deg

    out = {ch: np.empty((size, size), dtype=np.float64) for ch in crops}
    for r0 in range(0, size, ROWS_PER_CHUNK):
        r1 = min(size, r0 + ROWS_PER_CHUNK)
        lon2d, lat2d = np.meshgrid(sample_lon, sample_lat[r0 * ss:r1 * ss])
        x, y = projection.latlon_to_abi_xy(lat2d, lon2d, crs, sat_height_m)
        for ch, band in crops.items():
            n_rows, n_cols = band.values.shape
            dx = (band.x[-1] - band.x[0]) / (n_cols - 1)  # signed: ABI x runs W->E, y N->S
            dy = (band.y[-1] - band.y[0]) / (n_rows - 1)
            col = np.floor((x - band.x[0]) / dx + 0.5).astype(np.intp)
            row = np.floor((y - band.y[0]) / dy + 0.5).astype(np.intp)
            if col.min() < 0 or col.max() >= n_cols or row.min() < 0 or row.max() >= n_rows:
                raise ValueError(f"C{ch:02d} crop does not cover the output grid")
            samples = band.values[row, col]
            out[ch][r0:r1] = samples.reshape(r1 - r0, ss, size, ss).mean(axis=(1, 3))
    return out


def print_stats(fields: dict[int, np.ndarray], render_ranges: dict[int, tuple[float, float]]) -> None:
    print("\nchannel data on the output grid (exactly what the renderer receives):")
    print(f"  {'':4}{'units':<13}{'min':>9}{'max':>9}{'mean':>9}   {'render range':<14}{'below / above range'}")
    for ch, f in fields.items():
        lo, hi = render_ranges[ch]
        print(
            f"  C{ch:02d} {UNITS[ch]:<13}{f.min():>9.3f}{f.max():>9.3f}{f.mean():>9.3f}   "
            f"{f'{lo:g}-{hi:g}':<14}{(f < lo).mean() * 100:5.2f}% / {(f > hi).mean() * 100:5.2f}%"
        )


def render_c02(reflectance: np.ndarray, white_point: float, exponent: float) -> Image.Image:
    """Grayscale: divide by the white point, then a power curve (exponent
    0.5 is a square root). A low sun leaves the scene far below reflectance
    1, so the white point is what lets the brightest tops reach white;
    reflectance at or above it clips. The exponent shapes the curve between
    black and white -- smaller lifts darks but flattens the tops."""
    stretched = np.clip(reflectance / white_point, 0.0, 1.0) ** exponent
    return Image.fromarray(np.clip(stretched * 255.0, 0, 255).astype(np.uint8))


def brit_count(bt_k: np.ndarray) -> np.ndarray:
    """8-bit image count for a brightness temperature: the inverse of the
    McIDAS/GINI calibration, rounded to the nearest count."""
    counts = np.where(bt_k > 242.0, 660.0 - 2.0 * bt_k, 418.0 - bt_k)
    return np.clip(np.rint(counts), 0, 255).astype(np.intp)


def brit_temperature(count) -> np.ndarray:
    """Brightness temperature (K) of an 8-bit image count."""
    count = np.asarray(count, dtype=np.float64)
    return np.where(count < 176, (660.0 - count) / 2.0, 418.0 - count)


def table_rgb(name: str) -> np.ndarray:
    """A registered MetPy colortable as (N, 3) uint8 RGB."""
    colors = np.asarray(ctables.registry.get_colortable(name).colors)[:, :3]
    return np.round(colors * 255).astype(np.uint8)


def count_lut(name: str) -> np.ndarray:
    """The 256-entry GEMPAK lookup table behind a MetPy table: MetPy's
    entries at their image counts, black before them as in GEMPAK."""
    rgb = table_rgb(name)
    offset = TABLE_COUNT_OFFSETS[name]
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[offset:offset + len(rgb)] = rgb
    return lut


def color_runs(rgb: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) entry of each run of a table, split where adjacent entries
    differ by more than STEP_JUMP -- the table's visible steps."""
    jumps = np.abs(np.diff(rgb.astype(int), axis=0)).sum(axis=1) > STEP_JUMP
    starts = [0] + [int(i) + 1 for i in np.flatnonzero(jumps)]
    ends = [s - 1 for s in starts[1:]] + [len(rgb) - 1]
    return list(zip(starts, ends))


def _hex(rgb) -> str:
    return "%02x%02x%02x" % tuple(int(v) for v in rgb)


def print_count_table(channel: str, name: str, bt: np.ndarray) -> None:
    """A table's visible steps as image counts, K and degC, with the share of
    this scene's pixels in each."""
    lut = count_lut(name)
    counts = brit_count(bt)
    print(
        f"\n{channel} colortable '{name}': MetPy {metpy.__version__} metpy.plots.ctables registry "
        f"= GEMPAK gempak/tables/luts/{name}.tbl (MetPy index 0 = image count {TABLE_COUNT_OFFSETS[name]}); "
        "counts -> K by the McIDAS/GINI calibration"
    )
    print(f"  {'counts':<9}{'K':>13}{'degC':>17}   {'colors':<15}{'share':>8}")
    for first, last in color_runs(lut):
        k_warm, k_cold = float(brit_temperature(first)), float(brit_temperature(last))
        share = ((counts >= first) & (counts <= last)).mean() * 100
        print(
            f"  {f'{first}-{last}':<9}{f'{k_warm:.1f}-{k_cold:.1f}':>13}"
            f"{f'{k_warm - 273.15:+.1f} to {k_cold - 273.15:+.1f}':>17}   "
            f"{_hex(lut[first])}->{_hex(lut[last])} {share:7.2f}%"
        )


def rbtop_lut() -> np.ndarray:
    """RBTOP as a 256-entry lookup table indexed by image count (RGB)."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c0, c1, b0, b1, g0, g1, r0, r1 in RBTOP_ROWS:
        for c in range(c0, c1 + 1):
            t = (c - c0) / (c1 - c0) if c1 > c0 else 0.0
            lut[c] = [round(r0 + t * (r1 - r0)), round(g0 + t * (g1 - g0)), round(b0 + t * (b1 - b0))]
    return lut


def print_rbtop_table(bt: np.ndarray) -> None:
    """RBTOP's published rows as image counts, K and degC, with the share of this scene's pixels in each."""
    lut, counts = rbtop_lut(), brit_count(bt)
    print(f"\nC13 colortable RBTOP: NOAA OSPO McIDAS enhancement, unmodified ({RBTOP_SOURCE}); "
          "counts -> K by the McIDAS/GINI calibration")
    print(f"  {'counts':<9}{'K':>13}{'degC':>17}   {'colors':<15}{'share':>8}")
    for c0, c1, *_ in RBTOP_ROWS:
        k_warm, k_cold = float(brit_temperature(c0)), float(brit_temperature(c1))
        share = ((counts >= c0) & (counts <= c1)).mean() * 100
        print(f"  {f'{c0}-{c1}':<9}{f'{k_warm:.1f}-{k_cold:.1f}':>13}"
              f"{f'{k_warm - 273.15:+.1f} to {k_cold - 273.15:+.1f}':>17}   {_hex(lut[c0])}->{_hex(lut[c1])} {share:7.2f}%")


def custom_breaks(bt: np.ndarray, percentiles, breaks_k) -> list[float]:
    """Step boundaries in K, warm to cold: explicit --c13-custom-breaks if given, else the scene's percentiles."""
    if breaks_k:
        return [float(b) for b in breaks_k]
    return [float(np.percentile(bt, p)) for p in percentiles]


def custom_steps(breaks: list[float]) -> list[tuple[str, float, float, tuple, tuple, np.ndarray]]:
    """(name, warm K, cold K, warm color, cold color) per step, warm to cold."""
    edges = [CUSTOM_WARM_K] + list(breaks)
    return [(name, edges[i], edges[i + 1], c0, c1) for i, (name, c0, c1) in enumerate(CUSTOM_STEPS)]


def custom_masks(bt: np.ndarray, steps) -> list[np.ndarray]:
    """Pixels in each step. The first step takes everything at or above its cold edge, the last everything below
    its warm edge (clamping to white past the last break), the rest [cold, warm)."""
    masks = []
    for i, (_, warm, cold, _, _) in enumerate(steps):
        if i == 0:
            masks.append(bt >= cold)
        elif i == len(steps) - 1:
            masks.append(bt < warm)
        else:
            masks.append((bt < warm) & (bt >= cold))
    return masks


def render_c13_custom(bt: np.ndarray, breaks: list[float]) -> Image.Image:
    steps = custom_steps(breaks)
    rgb = np.zeros(bt.shape + (3,), dtype=np.float64)
    for (_, warm, cold, c0, c1), sel in zip(steps, custom_masks(bt, steps)):
        t = np.clip((warm - bt[sel]) / (warm - cold), 0.0, 1.0)
        rgb[sel] = np.asarray(c0, dtype=np.float64) + t[:, None] * (np.asarray(c1, dtype=np.float64) - np.asarray(c0))
    return Image.fromarray(np.round(rgb).astype(np.uint8))


def print_custom_table(bt: np.ndarray, breaks: list[float], percentiles) -> list[dict]:
    """The custom steps as K ranges with the share of this scene's pixels in each; returned for hero.json."""
    steps = custom_steps(breaks)
    how = f"percentiles {', '.join(f'{p:g}' for p in percentiles)} of this scene" if percentiles else "explicit --c13-custom-breaks"
    print(f"\nC13 CUSTOM colorization (not a published enhancement): steps at {how}")
    print(f"  {'step':<13}{'K (warm -> cold)':>18}{'degC':>17}   {'colors':<15}{'share':>8}")
    out = []
    for (name, warm, cold, c0, c1), sel in zip(steps, custom_masks(bt, steps)):
        share = float(sel.mean() * 100)
        warm_label = f">= {cold:.1f}" if name == steps[0][0] else f"{warm:.1f}-{cold:.1f}"
        print(f"  {name:<13}{warm_label:>18}{f'{warm - 273.15:+.1f} to {cold - 273.15:+.1f}':>17}   "
              f"{_hex(c0)}->{_hex(c1)} {share:7.2f}%")
        out.append({"step": name, "K_warm": round(warm, 2), "K_cold": round(cold, 2),
                    "rgb_warm": f"#{_hex(c0)}", "rgb_cold": f"#{_hex(c1)}", "pixel_share_pct": round(share, 2)})
    return out


def render_c13(bt: np.ndarray, style: str, breaks: list[float] | None = None) -> Image.Image:
    """"ir_bd": the Dvorak BD curve at its native image counts. "rbtop": NOAA OSPO's RBTOP table, unmodified, at
    its native counts. "custom": the histogram-mapped custom colorization. "gray": the repo's IR plotting
    convention (make_previews.bt_to_image), cold tops white."""
    if style == "gray":
        return bt_to_image(bt)
    if style == "rbtop":
        return Image.fromarray(rbtop_lut()[brit_count(bt)])
    if style == "custom":
        return render_c13_custom(bt, breaks)
    return Image.fromarray(count_lut(C13_TABLE)[brit_count(bt)])


def render_c09(bt: np.ndarray) -> Image.Image:
    """WVCIMSS at its native image counts."""
    return Image.fromarray(count_lut(C09_TABLE)[brit_count(bt)])


def feather_mask(size: int, width_px: float, shape: str) -> Image.Image | None:
    """size x size alpha mask, 0 at the border rising to 255 over width_px from
    every edge (None when width_px is 0). Distances are to pixel centers, so the
    outermost pixels are nearly, not exactly, transparent."""
    if width_px <= 0:
        return None
    centers = np.arange(size) + 0.5
    t = np.clip(np.minimum(centers, size - centers) / width_px, 0.0, 1.0)
    ramp = FEATHER_SHAPES[shape](t)
    return Image.fromarray(np.round(np.outer(ramp, ramp) * 255.0).astype(np.uint8))


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def radius_grid(size: int) -> np.ndarray:
    """Distance of each pixel center from the image center, in half-sides
    (1 at the edge midpoints, sqrt 2 at the corners)."""
    c = (np.arange(size) + 0.5 - size / 2) / (size / 2)
    return np.hypot(c[None, :], c[:, None])


def luminance(image: Image.Image, blur_px: float) -> np.ndarray:
    """Rec. 709 luma of the rendered image, 0-1, Gaussian-blurred by blur_px."""
    luma = (np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0) @ LUMA_WEIGHTS
    if blur_px > 0:
        blurred = Image.fromarray(np.round(luma * 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur_px))
        luma = np.asarray(blurred, dtype=np.float64) / 255.0
    return luma


def radial_alpha(image: Image.Image, radius: float, falloff: float, luminance_weight: float,
                 outer_start: float, outer_end: float) -> np.ndarray:
    """Alpha 0-1 from one image's brightness. Each pixel fades over
    [radius, radius + falloff] of a luminance-shifted distance,
    r - luminance_weight * falloff * (luma - 0.5): at weight 1 the darkest and
    brightest pixels fade half a falloff width earlier and later than mid-grey;
    weight 0 is a plain radial fade. The result is then multiplied by a smooth
    fade to 0 across [outer_start, outer_end] of r."""
    size = image.size[0]
    r = radius_grid(size)
    shifted = r - luminance_weight * falloff * (luminance(image, LUMA_BLUR * size) - 0.5)
    alpha = 1.0 - smoothstep(radius, radius + falloff, shifted)
    return alpha * (1.0 - smoothstep(outer_start, outer_end, r))


def alpha_summary(alpha: np.ndarray) -> str:
    """How much of an image survives an alpha mask."""
    r = radius_grid(alpha.shape[0]).ravel()
    order = np.argsort(r)
    cumulative = np.cumsum(alpha.ravel()[order])
    r90 = r[order][np.searchsorted(cumulative, 0.9 * cumulative[-1])]
    return (
        f"mean alpha {alpha.mean() * 100:.1f}%, alpha >= 0.5 on {(alpha >= 0.5).mean() * 100:.1f}% of pixels, "
        f"90% of the alpha within r = {r90:.2f}"
    )


def edge_stats(image: Image.Image, ring_px: int = EDGE_RING_PX) -> str:
    """Darkest pixel, mean, and mean of the outer ring_px border of a rendered
    image, 0-255 (color images via PIL's L conversion): a quick read on whether
    dark sea is in the frame and reaches its edge."""
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    ring = np.ones(gray.shape, dtype=bool)
    ring[ring_px:-ring_px, ring_px:-ring_px] = False
    return f"darkest pixel {int(gray.min())}, mean {gray.mean():.1f}, outer {ring_px}px ring mean {gray[ring].mean():.1f}"


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _six_descending(text: str) -> tuple[float, ...]:
    """Six comma-separated numbers, strictly decreasing (warm to cold)."""
    try:
        values = tuple(float(v) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected six comma-separated numbers") from exc
    if len(values) != len(CUSTOM_STEPS) or any(b >= a for a, b in zip(values, values[1:])):
        raise argparse.ArgumentTypeError(f"expected {len(CUSTOM_STEPS)} strictly decreasing values, warm to cold")
    return values


def _present(by_band: dict, channels) -> dict:
    """The entries of a {"C09": ..., "C13": ...} dict for the exported channels only."""
    return {k: v for k, v in by_band.items() if int(k[1:]) in channels}


def export_half_width(args, crops: dict[int, BandCrop], channels, half_width: float, out_dir: Path, suffix: str) -> None:
    """Resample, render and write one export at one half-width, from crops cut at least that wide."""
    if suffix:
        print(f"\n===== half-width {half_width:g} deg")
    pixel_deg = 2 * half_width / args.size
    ss = args.supersample or supersample_factor(pixel_deg, args.lat)
    print(
        f"\nresampling onto a {args.size}x{args.size} lat/lon grid: {pixel_deg:.5f} deg/px "
        f"(~{pixel_deg * KM_PER_DEG:.3f} km N-S, {pixel_deg * KM_PER_DEG * math.cos(math.radians(args.lat)):.3f} km E-W), "
        f"{ss}x{ss} samples per pixel"
    )
    fields = resample_to_latlon(crops, args.lat, args.lon, half_width, args.size, ss)
    for ch, field in fields.items():
        n_missing = int(np.isnan(field).sum())
        if n_missing:
            raise SystemExit(f"C{ch:02d} has {n_missing} missing output pixels")
    count_range = (float(brit_temperature(255)), float(brit_temperature(0)))
    breaks = None
    if 13 in fields and args.c13_style == "custom":
        breaks = custom_breaks(fields[13], args.c13_custom_percentiles, args.c13_custom_breaks)
        if breaks[0] >= CUSTOM_WARM_K or any(b >= a for a, b in zip(breaks, breaks[1:])):
            raise SystemExit(f"custom breakpoints must fall strictly below {CUSTOM_WARM_K:g} K and decrease: {breaks}")
    c13_range = {"gray": (BT_MIN_K, BT_MAX_K), "custom": (breaks[-1], CUSTOM_WARM_K) if breaks else None}.get(args.c13_style, count_range)
    print_stats(fields, {2: (0.0, args.c02_white_point), 9: count_range, 13: c13_range})
    if 9 in fields:
        print_count_table("C09", C09_TABLE, fields[9])
    custom_step_report = None
    if 13 in fields:
        if args.c13_style == C13_TABLE:
            print_count_table("C13", C13_TABLE, fields[13])
        elif args.c13_style == "rbtop":
            print_rbtop_table(fields[13])
        elif args.c13_style == "custom":
            custom_percentiles = None if args.c13_custom_breaks else args.c13_custom_percentiles
            custom_step_report = print_custom_table(fields[13], breaks, custom_percentiles)

    renderers = {
        2: lambda: render_c02(fields[2], args.c02_white_point, args.c02_exponent),
        9: lambda: render_c09(fields[9]),
        13: lambda: render_c13(fields[13], args.c13_style, breaks),
    }
    images = {ch: renderers[ch]() for ch in channels}
    if 2 in images:
        at_white = (np.asarray(images[2]) == 255).mean() * 100
        print(
            f"\nC02 pixels at 255 after the stretch: {at_white:.2f}% "
            f"(white point {args.c02_white_point:g}, exponent {args.c02_exponent:g})"
        )
    feather_px = args.feather_width * args.size
    if args.alpha_mode == "square":
        # One mask for all bands, so the band switch never reveals a different edge.
        mask = feather_mask(args.size, feather_px, args.feather_shape)
        masks = {ch: mask for ch in images}
        if mask is not None:
            print(f"edge feather: {feather_px:.0f} px ({args.feather_width:g} of the side), {args.feather_shape}")
    else:
        # One mask from C02's brightness, applied unchanged to every band: identical outline, only the colormap changes.
        alpha = radial_alpha(images[MASK_CHANNEL], args.radial_radius, args.radial_falloff, args.luminance_weight,
                             args.outer_fade_start, args.outer_fade_end)
        mask = Image.fromarray(np.round(alpha * 255.0).astype(np.uint8))
        masks = {ch: mask for ch in images}
        print(
            f"radial alpha from C{MASK_CHANNEL:02d} brightness, shared by all bands: radius {args.radial_radius:g}, "
            f"falloff {args.radial_falloff:g}, luminance weight {args.luminance_weight:g}, "
            f"outer fade {args.outer_fade_start:g}-{args.outer_fade_end:g} (half-sides, 1 = edge midpoint)"
        )
        print(f"  {alpha_summary(alpha)}")
    print()
    for ch, image in images.items():
        # Edge statistics on the rendered pixels, before any alpha.
        print(f"C{ch:02d}{suffix}: {edge_stats(image)}")
        mask = masks[ch]
        if mask is not None:
            image = image.convert("RGBA")
            image.putalpha(mask)
        path = out_dir / f"c{ch:02d}{suffix}.webp"
        image.save(path, quality=WEBP_QUALITY, method=6)
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")

    band = crops[channels[0]]
    meta = {
        "satellite": f"GOES-{int(band.platform_id[1:])}",
        "product": args.product,
        "scan_start_utc": band.scan_start,
        "channels": [f"C{ch:02d}" for ch in channels],
        "center_lat": args.lat,
        "center_lon": args.lon,
        "half_width_deg": half_width,
        "output_size_px": [args.size, args.size],
        "pixel_size_deg": round(pixel_deg, 6),
        "colortables": _present({
            "C09": C09_TABLE,
            "C13": {
                C13_TABLE: C13_TABLE,
                "rbtop": "RBTOP (NOAA OSPO McIDAS enhancement, unmodified)",
                "custom": "CUSTOM colorization -- not a published enhancement (see c13_custom)",
                "gray": "gray (make_previews.bt_to_image)",
            }[args.c13_style],
        }, channels),
        "colortable_source": _present({
            "C09": "MetPy metpy.plots.ctables registry (GEMPAK gempak/tables/luts)",
            "C13": {
                C13_TABLE: "MetPy metpy.plots.ctables registry (GEMPAK gempak/tables/luts)",
                "rbtop": RBTOP_SOURCE,
                "custom": "scripts/export_hero.py CUSTOM_STEPS; breakpoints from this scene's C13 histogram",
                "gray": "scripts/make_previews.py bt_to_image",
            }[args.c13_style],
        }, channels),
        "colortable_mapping": _present({
            "C09": "native GEMPAK image counts; K = (660 - count) / 2 for count < 176, else 418 - count",
            "C13": {
                C13_TABLE: "native GEMPAK image counts; K = (660 - count) / 2 for count < 176, else 418 - count",
                "rbtop": "native McIDAS brightness counts; K = (660 - count) / 2 for count < 176, else 418 - count",
                "custom": "brightness temperature in K, per c13_custom.steps",
                "gray": f"linear {BT_MIN_K:g}-{BT_MAX_K:g} K, cold tops white",
            }[args.c13_style],
        }, channels),
        "c13_custom": (
            {
                "type": "custom colorization -- NOT a published enhancement",
                "method": "color steps placed at percentiles of this scene's own C13 brightness temperatures "
                          "(histogram-mapped), each step a linear ramp between two colors; a color does not "
                          "correspond to the same temperature on other scenes",
                "percentiles": None if args.c13_custom_breaks else list(args.c13_custom_percentiles),
                "breakpoints_K": [round(b, 2) for b in breaks],
                "warm_end_K": CUSTOM_WARM_K,
                "steps": custom_step_report,
            }
            if args.c13_style == "custom" and 13 in channels else None
        ),
        "metpy_version": metpy.__version__,
        "alpha_mode": args.alpha_mode,
        "edge_feather": (
            {"width_px": round(feather_px, 1), "shape": args.feather_shape}
            if args.alpha_mode == "square" and feather_px > 0 else None
        ),
        "radial_alpha": (
            {
                "radius": args.radial_radius, "falloff": args.radial_falloff,
                "luminance_weight": args.luminance_weight,
                "luminance": f"Rec. 709 luma of the rendered C{MASK_CHANNEL:02d} image; one mask shared by all bands",
                "luminance_blur_px": round(LUMA_BLUR * args.size, 1),
                "outer_fade_r": [args.outer_fade_start, args.outer_fade_end],
            }
            if args.alpha_mode == "radial" else None
        ),
    }
    json_path = out_dir / f"hero{suffix}.json"
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", default="noaa-goes19")
    parser.add_argument("--product", default="ABI-L2-CMIPF", help="full-disk CMIP product")
    parser.add_argument("--date", default="2025-10-26", help="UTC date, YYYY-MM-DD")
    parser.add_argument("--time", default="21:30", help="UTC time HH:MM; the nearest scan in that hour is used")
    parser.add_argument("--lat", type=float, default=16.39, help="center latitude, degrees north")
    parser.add_argument("--lon", type=float, default=-77.10, help="center longitude, degrees east")
    parser.add_argument("--half-width", type=_positive_float, nargs="+", default=[2.5],
                        help="box half-width in degrees of lat and lon; give several (e.g. 4 5 6) to export each from "
                             "one crop, with the half-width in every filename (c02_hw4.0.webp, hero_hw4.0.json)")
    parser.add_argument("--channels", type=int, nargs="+", choices=CHANNELS, default=list(CHANNELS),
                        help="bands to export, e.g. 2 for C02 only (the others are then never downloaded)")
    parser.add_argument("--size", type=int, default=900, help="output width and height in pixels")
    parser.add_argument("--c02-exponent", type=_positive_float, default=0.5,
                        help="C02 stretch exponent (0.5 = sqrt; smaller lifts dark areas more)")
    parser.add_argument("--c02-white-point", type=_positive_float, default=0.72,
                        help="C02 reflectance rendered as white, applied before the exponent; "
                             "raise it if the tops clip")
    parser.add_argument("--c13-style", choices=(C13_TABLE, "rbtop", "custom", "gray"), default=C13_TABLE,
                        help="ir_bd: the Dvorak BD curve (MetPy/GEMPAK colortable); rbtop: NOAA OSPO's published RBTOP "
                             "table, unmodified; custom: a histogram-mapped custom colorization (not a published "
                             "enhancement; see --c13-custom-*); gray: cold tops white")
    parser.add_argument("--c13-custom-percentiles", type=_six_descending, default=CUSTOM_PERCENTILES,
                        help="custom: step boundaries as percentiles of the scene's C13 brightness temperatures, warm "
                             "to cold: gray|green, green|yellow, yellow|orange, orange|red, red|black, and where the "
                             "coldest tops reach white (default 80,62,45,27,10,1)")
    parser.add_argument("--c13-custom-breaks", type=_six_descending, default=None,
                        help="custom: the same six boundaries as explicit brightness temperatures in K, warm to cold; "
                             "overrides --c13-custom-percentiles")
    parser.add_argument("--feather-width", type=float, default=0.08,
                        help="width of the transparent edge fade, as a fraction of the image side (0 disables)")
    parser.add_argument("--feather-shape", choices=tuple(FEATHER_SHAPES), default="smoothstep",
                        help="falloff from the transparent border to the opaque interior")
    parser.add_argument("--alpha-mode", choices=ALPHA_MODES, default="square",
                        help="square: feathered edges (--feather-width/--feather-shape); "
                             "radial: fade from the centre weighted by C02 brightness, one mask for all bands, "
                             "no visible frame")
    parser.add_argument("--radial-radius", type=_positive_float, default=0.5,
                        help="radial: distance from the centre where mid-grey starts to fade, "
                             "in half-sides (1 = edge midpoint)")
    parser.add_argument("--radial-falloff", type=_positive_float, default=0.35,
                        help="radial: width of the fade, in half-sides")
    parser.add_argument("--luminance-weight", type=float, default=1.5,
                        help="radial: how far brightness moves a pixel's fade, in falloff widths across the "
                             "full dark-to-bright range (0 = plain radial fade)")
    parser.add_argument("--outer-fade-start", type=float, default=0.6,
                        help="radial: distance where the outer fade begins, in half-sides; lower lets the "
                             "brightest cloud taper off instead of ending at a disc")
    parser.add_argument("--outer-fade-end", type=float, default=1.0,
                        help="radial: distance where the outer fade reaches 0 (at most 1, so the square edge never shows)")
    parser.add_argument("--supersample", type=int, default=0,
                        help="sample points per output-pixel side (0 = choose from C02 resolution)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "site" / "hero"))
    parser.add_argument("--cache-dir", default=str(config.INTERIM_DIR / "hero_crops"))
    parser.add_argument("--no-cache", action="store_true", help="always download, and don't save crops")
    args = parser.parse_args()
    if not args.product.endswith("CMIPF"):
        parser.error("only full-disk (...CMIPF) products are supported")
    if not 0.0 <= args.feather_width <= 0.5:
        parser.error("--feather-width must be between 0 and 0.5")
    if args.luminance_weight < 0:
        parser.error("--luminance-weight must be >= 0")
    if not 0.0 <= args.outer_fade_start < args.outer_fade_end <= 1.0:
        parser.error("need 0 <= --outer-fade-start < --outer-fade-end <= 1")

    channels = tuple(ch for ch in CHANNELS if ch in args.channels)
    if args.alpha_mode == "radial" and MASK_CHANNEL not in channels:
        parser.error(f"--alpha-mode radial derives its mask from C{MASK_CHANNEL:02d}; include {MASK_CHANNEL} in --channels")
    half_widths = sorted(set(args.half_width))

    target = datetime.strptime(f"{args.date} {args.time}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    s3_client = goes_index.make_anonymous_s3_client()
    _, keys = match_scan(s3_client, args.bucket, args.product, target, channels)

    # Crop once, for the widest box: a crop cut for a wider box around the same center covers every
    # narrower one, so every half-width comes from the same granule and at most one download.
    widest = half_widths[-1]
    print(f"\ncropping (once, for the widest half-width: {widest:g} deg)")
    cache_dir = None if args.no_cache else Path(args.cache_dir)
    crops = {
        ch: load_crop(s3_client, args.bucket, keys[ch], ch, args.lat, args.lon, widest, cache_dir)
        for ch in channels
    }
    for ch, band in crops.items():
        print(f"  C{ch:02d}: time_coverage_start {band.scan_start}, crop {band.values.shape[0]}x{band.values.shape[1]} px")
    if len({band.scan_start for band in crops.values()}) != 1:
        raise SystemExit("the files do not share a scan start time")
    if len({(band.crs_wkt, band.sat_height_m, band.platform_id) for band in crops.values()}) != 1:
        raise SystemExit("the files do not share a satellite and projection")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for half_width in half_widths:
        suffix = f"_hw{half_width:.1f}" if len(half_widths) > 1 else ""
        export_half_width(args, crops, channels, half_width, out_dir, suffix)


if __name__ == "__main__":
    main()
