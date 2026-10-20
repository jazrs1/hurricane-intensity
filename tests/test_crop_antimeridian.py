"""F-35: the crop box near 180 deg, for centres 0.5 to 6 deg either side of it (gate before H-EP gate 7).

projection.crop_box takes the ABI x/y bounding box of the four corners of a nominal box_km x box_km box: centre
+- (box_km / 2) / 111.32 deg of latitude and +- (box_km / 2) / (111.32 cos lat) deg of longitude (its docstring's
flat-earth definition). Near 180 deg one or two of those corners lie across the seam. A crop_box that clips them
at 180, or otherwise mishandles them, loses part of the box on that side.

tests/test_antimeridian.py's crop pins check only that the point is at least 100 px inside every edge, and a box
clipped at 180 passes that at a centre of 177.9 deg (hc-adversary item 21b). Here every edge of the crop is checked
against the nominal box, computed in the test with its own pyproj transformer. Each of the crop's four edges must
lie within one pixel of the nominal box's edge on that side, and the four points box_km / 2 west, east, north and
south of the centre must lie inside the crop. The nominal box is symmetric in lat/lon about the centre, so this is
the "symmetric margins" check. Its pixel margins are not symmetric: the projection skews them by up to ~70 km at
30N on the centre row, for any longitude, so pixel counts either side of the centre are not compared.

Synthetic GOES-18 fixed grid (lon_0 = -137), zero data: crop_box only uses the x/y coordinates, and there is no
scan to read. No downloads, no data/, no network, no GPU.
"""
import functools

import numpy as np
import pyproj
import pytest
import xarray as xr

from hurricane_intensity import projection

H = 35786023.0
CRS = pyproj.CRS.from_proj4(f"+proj=geos +h={H} +a=6378137.0 +b=6356752.31414 +lon_0=-137.0 +sweep=x +units=m +no_defs")
PIXEL_RAD = 56e-6  # ABI 2 km band 13 pixel
BOX_KM = 600.0
KM_PER_DEG = 111.32
FWD = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)


@functools.lru_cache(maxsize=1)
def synthetic_disk():
    x = np.arange(-0.151844, 0.151844 + 1e-9, PIXEL_RAD)
    return xr.DataArray(np.zeros((x.size, x.size), np.float32), coords={"y": x[::-1], "x": x}, dims=("y", "x"))


def wrap(lon):
    return (lon + 180.0) % 360.0 - 180.0


def xy(lat, lon):
    x, y = FWD.transform(wrap(lon), lat)
    return x / H, y / H


def nominal_box(lat, lon):
    """Corners and edge midpoints of the nominal box, longitudes wrapped, from the box's own definition."""
    dlat = BOX_KM / 2 / KM_PER_DEG
    dlon = BOX_KM / 2 / (KM_PER_DEG * np.cos(np.radians(lat)))
    corners = [(lat + sy * dlat, wrap(lon + sx * dlon)) for sy in (1, -1) for sx in (-1, 1)]
    mids = {"west": (lat, wrap(lon - dlon)), "east": (lat, wrap(lon + dlon)),
            "north": (lat + dlat, wrap(lon)), "south": (lat - dlat, wrap(lon))}
    return corners, mids


def check_full_box(c, lat, lon):
    """Problems with crop c against the nominal box at (lat, lon); empty if it is the full box."""
    corners, mids = nominal_box(lat, lon)
    cx, cy = zip(*(xy(*p) for p in corners))
    want = {"west": min(cx), "east": max(cx), "south": min(cy), "north": max(cy)}
    if c.x.size == 0 or c.y.size == 0:
        return [f"empty crop {c.shape}"]
    xs, ys = np.sort(c.x.values), np.sort(c.y.values)
    got = {"west": xs[0], "east": xs[-1], "south": ys[0], "north": ys[-1]}
    bad = [f"{side} edge {(got[side] - want[side]) / PIXEL_RAD:+.1f} px from the nominal box"
           for side in want if abs(got[side] - want[side]) > PIXEL_RAD]
    for side, p in mids.items():
        px, py = xy(*p)
        if not (xs[0] - PIXEL_RAD <= px <= xs[-1] + PIXEL_RAD and ys[0] - PIXEL_RAD <= py <= ys[-1] + PIXEL_RAD):
            bad.append(f"the point {BOX_KM / 2:.0f} km {side} of the centre ({p[0]:.2f}, {p[1]:.2f}) is outside the crop")
    return bad


LATS = [10.0, 18.0, 30.0]
OFFSETS = [0.5, 1.0, 2.0, 2.1, 3.0, 4.5, 6.0]  # deg from 180; 2.1 is 177.9E / 177.9W (item 21b)
CASES = [(lat, side * (180.0 - off)) for lat in LATS for off in OFFSETS for side in (1, -1)]


@pytest.mark.parametrize("lat, lon", CASES, ids=[f"{lat:.0f}N-{lon:+.1f}" for lat, lon in CASES])
def test_crop_box_is_full_nominal_box_near_180(lat, lon):
    c = projection.crop_box(synthetic_disk(), lat, lon, BOX_KM, CRS, H)
    bad = check_full_box(c, lat, lon)
    assert not bad, f"centre ({lat}, {lon}), crop {c.shape}: " + "; ".join(bad)


@pytest.mark.parametrize("lat, lon", CASES, ids=[f"{lat:.0f}N-{lon:+.1f}" for lat, lon in CASES])
def test_crop_box_same_for_both_spellings_near_180(lat, lon):
    """lon and lon -+ 360 name the same centre, so they must give the same crop."""
    alt = lon - 360.0 if lon > 0 else lon + 360.0
    a = projection.crop_box(synthetic_disk(), lat, lon, BOX_KM, CRS, H)
    b = projection.crop_box(synthetic_disk(), lat, alt, BOX_KM, CRS, H)
    assert a.shape == b.shape and np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y), \
        f"centre ({lat}, {lon}) gives {a.shape}, ({lat}, {alt}) gives {b.shape}"


def test_nominal_box_straddles_180_for_most_cases():
    """Sanity check on the inputs: most cases really do put a corner across 180 deg: 26 of 42 (the half-width is 2.74 deg
    at 10N and 3.11 deg at 30N), and the 6 deg cases do not, as controls."""
    straddle = [abs(lon) + BOX_KM / 2 / (KM_PER_DEG * np.cos(np.radians(lat))) > 180.0 for lat, lon in CASES]
    assert sum(straddle) == 26  # 0.5-2.1 deg at every latitude (24), and 3 deg at 30N (2)
    assert not any(s for s, (lat, lon) in zip(straddle, CASES) if abs(180.0 - abs(lon)) >= 4.5)
