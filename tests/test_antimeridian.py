"""B-v2-1: tracks that cross the antimeridian (+-180 deg). Failing tests first; John writes the fix.

The tests marked xfail(strict=True, raises=AssertionError) fail today because longitude is interpolated or
differenced linearly, so a track from 179.0E to 179.0W passes through 0 deg instead of 180 deg. A fix flips
them to XPASS, which strict turns into an error until the mark is removed. The unmarked tests pass today and
pin what must not change: HURDAT2 E/W parsing, an East Pacific track that never reaches 180 deg, and the crop
geometry, which is already wrap-safe given a correct centre (PROJ normalises longitude).

Expected positions are computed here, independently of the code under test, by unwrapped-linear arithmetic:
take the short way round (dlon in [-180, 180)), interpolate linearly in time, compare modulo 360. That is the
pipeline's own linear rule for any track that does not cross 180 deg, so the non-crossing pin and the crossing
expectations agree. The position tests compare longitudes as wrapped differences. The convention for the
values themselves is John's (2026-10-20, shortly before 00:22 EDT): "interpolation short way round, linear in time; saved
longitudes re-wrapped to [-180, 180)". test_grid_lon_in_range pins it, so a fix at 180.0E is saved as -180.0.

Synthetic fixes only: no downloads, no data/ reads or writes, no network, no GPU.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr

from hurricane_intensity import hurdat2, pipeline, projection

_spec = importlib.util.spec_from_file_location(
    "concurrency", Path(__file__).resolve().parents[1] / "scripts" / "concurrency.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

TOL_DEG = 1e-6
XFAIL = dict(strict=True, raises=AssertionError)


def wrapped_diff(a, b):
    """a - b in degrees of longitude, the short way round, in [-180, 180)."""
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


def expected_lon(fix_times, fix_lons, times):
    """Unwrapped-linear longitude at `times`, computed without np.interp: per segment, the short way round."""
    fix_times = pd.DatetimeIndex(fix_times)
    out = []
    for t in pd.DatetimeIndex(times):
        k = max(i for i in range(len(fix_times)) if fix_times[i] <= t)
        if k == len(fix_times) - 1:
            out.append(fix_lons[k])
            continue
        f = (t - fix_times[k]) / (fix_times[k + 1] - fix_times[k])
        d = float(wrapped_diff(fix_lons[k + 1], fix_lons[k]))
        out.append(fix_lons[k] + f * d)
    return np.array(out)


def fixes(start, lats, lons, winds=None, step="6h"):
    times = pd.date_range(start, periods=len(lats), freq=step, tz="UTC")
    winds = winds if winds is not None else [65.0] * len(lats)
    return pd.DataFrame({"time": times, "lat": lats, "lon": lons, "wind_kt": winds, "status": ["HU"] * len(lats)})


def lon_mismatch(got, want):
    err = np.abs(wrapped_diff(got, want))
    i = int(np.argmax(err))
    return err.max(), f"row {i}: got lon {got[i]:.4f}, expected {want[i]:.4f} (wrapped error {err[i]:.4f} deg)"


# Westward (E->W in the sense of the storm's motion, Dora's direction): from the western hemisphere's 178.5W
# across 180 into the eastern hemisphere. Eastward: the reverse.
WESTWARD = fixes("2023-08-12 00:00", [17.0, 17.2, 17.4, 17.6], [-178.5, -179.6, 179.3, 178.2],
                 [70.0, 75.0, 85.0, 90.0])
EASTWARD = fixes("2023-09-01 00:00", [20.0, 20.4, 20.8, 21.2], [178.4, 179.5, -179.4, -178.3],
                 [60.0, 55.0, 50.0, 40.0])


# --- pass today: what the fix must not change ------------------------------------------------------------

def test_hurdat2_lon_sign_parsing(tmp_path):
    """HURDAT2 E/W suffixes are parsed with the right sign, including both spellings of 180 deg. Not a wrap
    site: hurdat2._parse_lon is correct and stays as it is."""
    text = "\n".join([
        "EP992023,            SYNTH,      5,",
        "20230812, 0000,  , HU, 17.0N, 178.5W,  80,  980,",
        "20230812, 0600,  , HU, 17.1N, 180.0W,  80,  980,",
        "20230812, 1200,  , HU, 17.2N, 179.3E,  80,  980,",
        "20230812, 1800,  , HU, 17.3N, 180.0E,  80,  980,",
        "20230813, 0000,  , HU, 17.4N, 168.0E,  75,  985,",
        "",
    ])
    p = tmp_path / "hurdat2_synthetic.txt"
    p.write_text(text, encoding="utf-8")
    name, fx = hurdat2.parse_storm("EP992023", p)
    assert name == "SYNTH"
    assert fx["lon"].tolist() == [-178.5, -180.0, 179.3, 180.0, 168.0]
    assert fx["lat"].tolist() == [17.0, 17.1, 17.2, 17.3, 17.4]


def test_noncrossing_east_pacific_track_unchanged():
    """A realistic East Pacific track entirely east of 180 deg (Dora-like, 2023-08-01..03, 101W to 115W):
    the sample grid is pinned exactly as the pipeline builds it today."""
    lats = [14.8, 14.9, 15.0, 15.0, 14.9, 14.8, 14.6, 14.5, 14.3]
    lons = [-101.2, -102.6, -104.1, -105.7, -107.4, -109.1, -110.9, -112.8, -114.7]
    winds = [35.0, 45.0, 55.0, 65.0, 80.0, 95.0, 110.0, 115.0, 115.0]
    fx = fixes("2023-08-01 00:00", lats, lons, winds)
    grid = pipeline.build_sample_grid(fx, "1h")

    assert len(grid) == 49
    assert grid["time"].iloc[0] == pd.Timestamp("2023-08-01 00:00", tz="UTC")
    assert grid["time"].iloc[-1] == pd.Timestamp("2023-08-03 00:00", tz="UTC")
    assert grid["is_interpolated"].tolist() == [h % 6 != 0 for h in range(49)]
    assert grid["status"].isna().tolist() == [h % 6 != 0 for h in range(49)]
    # Independent linear arithmetic: hour h lies in segment h // 6 at fraction (h % 6) / 6.
    for col, vals in (("lat", lats), ("lon", lons), ("wind_kt", winds)):
        want = [vals[h // 6] + (h % 6) / 6 * (vals[h // 6 + 1] - vals[h // 6]) if h < 48 else vals[-1]
                for h in range(49)]
        np.testing.assert_allclose(grid[col].to_numpy(), want, rtol=0, atol=1e-9, err_msg=col)
    # A few literal values, so the pin does not rest on the formula alone.
    assert grid["lon"].iloc[3] == pytest.approx(-101.9, abs=1e-9)
    assert grid["lon"].iloc[27] == pytest.approx(-108.25, abs=1e-9)
    assert grid["lat"].iloc[45] == pytest.approx(14.4, abs=1e-9)


GOES18_H = 35786023.0
GOES18_CRS = pyproj.CRS.from_proj4(
    f"+proj=geos +h={GOES18_H} +a=6378137.0 +b=6356752.31414 +lon_0=-137.0 +sweep=x +units=m +no_defs")
PIXEL_RAD = 56e-6  # ABI 2 km band 13 pixel


def synthetic_disk():
    """A full-disk-sized GOES-18 fixed grid (x ascending, y descending) with zero data. No scan is needed:
    crop_box only uses the x/y coordinates."""
    x = np.arange(-0.151844, 0.151844 + 1e-9, PIXEL_RAD)
    return xr.DataArray(np.zeros((x.size, x.size), np.float32), coords={"y": x[::-1], "x": x}, dims=("y", "x"))


def abi_xy(lat, lon):
    """Independent of projection.latlon_to_abi_xy: our own transformer, longitude wrapped to [-180, 180)."""
    t = pyproj.Transformer.from_crs("EPSG:4326", GOES18_CRS, always_xy=True)
    x, y = t.transform(float(wrapped_diff(lon, 0.0)), lat)
    return x / GOES18_H, y / GOES18_H


MARGIN_PX = 100  # a 600 km box is ~260-310 px; its centre sits ~4 px off the middle away from nadir


def centred(c, lat, lon):
    """The crop is full size and the true centre lies inside it, at least MARGIN_PX from every edge."""
    if min(c.shape) < 2 * MARGIN_PX:
        return False
    x, y = abi_xy(lat, lon)
    xs, ys = np.sort(c.x.values), np.sort(c.y.values)
    return xs[MARGIN_PX] < x < xs[-1 - MARGIN_PX] and ys[MARGIN_PX] < y < ys[-1 - MARGIN_PX]


@pytest.mark.parametrize("lon_a, lon_b", [(179.8, -180.2), (180.0, -180.0), (-179.8, 180.2), (-179.95, 180.05)])
def test_crop_box_geometry_wrap_safe_near_180(lon_a, lon_b):
    """crop_box (projection.py, used by crop.extract_crop) is already wrap-safe: two spellings of the same
    meridian give the same crop, and it is full size with the point well inside it.
    Tested on the geometry function with a synthetic GOES-18 grid because there is no scan to read."""
    da = synthetic_disk()
    a = projection.crop_box(da, 18.0, lon_a, 600.0, GOES18_CRS, GOES18_H)
    b = projection.crop_box(da, 18.0, lon_b, 600.0, GOES18_CRS, GOES18_H)
    assert a.shape == b.shape and np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert centred(a, 18.0, lon_a)


@pytest.mark.parametrize("lon_a, lon_b", [(178.0, -182.0), (-178.0, 182.0)])
def test_crop_box_straddles_180_centre_not_near_it(lon_a, lon_b):
    """Only the box crosses 180 deg: the centre is 2 deg from it, inside the 600 km box's half-width at 18N
    (300 / (111.32 cos 18N) = 2.83 deg). Full size, point inside, same crop for both spellings."""
    half = 300.0 / (111.32 * np.cos(np.radians(18.0)))
    assert 2.0 < half
    da = synthetic_disk()
    a = projection.crop_box(da, 18.0, lon_a, 600.0, GOES18_CRS, GOES18_H)
    b = projection.crop_box(da, 18.0, lon_b, 600.0, GOES18_CRS, GOES18_H)
    assert a.shape == b.shape and np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert centred(a, 18.0, lon_a)


# A track through the prime meridian (eastern Atlantic, e.g. a storm heading for Iberia) is not a crossing:
# it must stay exactly linear. This catches a fix that interpolates in [0, 360) and maps back, which only
# moves the seam from 180 deg to 0 deg.
GREENWICH = fixes("2023-10-01 00:00", [40.0, 41.0, 42.0], [-1.5, -0.3, 0.9])


def linear_hourly(vals):
    """Hour h of fixes 6 h apart: segment h // 6 at fraction (h % 6) / 6, plain linear arithmetic."""
    n = 6 * (len(vals) - 1)
    return np.array([vals[h // 6] + (h % 6) / 6 * (vals[h // 6 + 1] - vals[h // 6]) if h < n else vals[-1]
                     for h in range(n + 1)])


def test_greenwich_grid_unchanged():
    grid = pipeline.build_sample_grid(GREENWICH, "1h")
    assert len(grid) == 13
    np.testing.assert_allclose(grid["lon"].to_numpy(), linear_hourly([-1.5, -0.3, 0.9]), rtol=0, atol=1e-9)
    assert grid["lon"].iloc[3] == pytest.approx(-0.9, abs=1e-9)
    assert grid["lon"].iloc[9] == pytest.approx(0.3, abs=1e-9)


def test_greenwich_position_at_unchanged():
    times = pd.date_range("2023-10-01 00:00", "2023-10-01 12:00", freq="1h", tz="UTC")
    _, lon = C.position_at(GREENWICH, times)
    np.testing.assert_allclose(lon, linear_hourly([-1.5, -0.3, 0.9]), rtol=0, atol=1e-9)


def test_greenwich_boxes_overlap():
    assert bool(C.boxes_overlap(40.0, -0.1, 40.0, 0.1))
    assert bool(C.boxes_overlap(40.0, 0.1, 40.0, -0.1))


@pytest.mark.parametrize("track", ["westward", "eastward"])
def test_crossing_grid_lat_wind_rows_unchanged(track):
    """On a crossing track only lon is wrong today: lat and wind are linear, and the rows and is_interpolated
    are right. Pinned here so a fix can't disturb them; lon stays in the xfail tests."""
    fx = WESTWARD if track == "westward" else EASTWARD
    grid = pipeline.build_sample_grid(fx, "1h")
    assert len(grid) == 19
    assert int(grid["is_interpolated"].sum()) == 15
    assert grid["is_interpolated"].tolist() == [h % 6 != 0 for h in range(19)]
    assert grid["time"].tolist() == list(pd.date_range(fx["time"].iloc[0], periods=19, freq="1h"))
    for col in ("lat", "wind_kt"):
        np.testing.assert_allclose(grid[col].to_numpy(), linear_hourly(fx[col].tolist()), rtol=0, atol=1e-9,
                                   err_msg=col)


# --- xfail today: the wrap sites --------------------------------------------------------------------------

@pytest.mark.xfail(**XFAIL, reason="B-v2-1 antimeridian: build_sample_grid interpolates lon linearly (westward)")
def test_grid_crossing_westward():
    grid = pipeline.build_sample_grid(WESTWARD, "1h")
    want = expected_lon(WESTWARD["time"], WESTWARD["lon"].tolist(), grid["time"])
    err, msg = lon_mismatch(grid["lon"].to_numpy(), want)
    assert err < TOL_DEG, msg


@pytest.mark.xfail(**XFAIL, reason="B-v2-1 antimeridian: build_sample_grid interpolates lon linearly (eastward)")
def test_grid_crossing_eastward():
    grid = pipeline.build_sample_grid(EASTWARD, "1h")
    want = expected_lon(EASTWARD["time"], EASTWARD["lon"].tolist(), grid["time"])
    err, msg = lon_mismatch(grid["lon"].to_numpy(), want)
    assert err < TOL_DEG, msg


# A fix exactly at 180 deg, spelt either way (HURDAT2 180.0W -> -180.0, 180.0E -> +180.0), in both directions.
# Each case straddles the seam on one side of the 180 fix, so each fails today; the test checks the midpoints
# on both sides.
SEAM_CASES = {
    "westward-180W": [-179.0, -180.0, 179.0],
    "westward-180E": [-179.0, 180.0, 179.0],
    "eastward-180W": [179.0, -180.0, -179.0],
    "eastward-180E": [179.0, 180.0, -179.0],
}


@pytest.mark.xfail(**XFAIL, reason="B-v2-1 antimeridian: interpolation between fixes on either side of a fix at 180")
@pytest.mark.parametrize("case", list(SEAM_CASES))
def test_grid_fix_exactly_at_180(case):
    lons = SEAM_CASES[case]
    fx = fixes("2023-08-12 00:00", [17.0, 17.2, 17.4], lons)
    grid = pipeline.build_sample_grid(fx, "3h")  # rows at 0, 3 (midpoint), 6 (the 180 fix), 9 (midpoint), 12 h
    assert len(grid) == 5
    want = expected_lon(fx["time"], lons, grid["time"])
    err, msg = lon_mismatch(grid["lon"].to_numpy(), want)
    assert err < TOL_DEG, msg


RANGE_XFAIL = pytest.mark.xfail(**XFAIL, reason="B-v2-1 antimeridian: a fix at 180.0E is saved as +180.0, outside [-180, 180)")


@pytest.mark.parametrize("track", [
    "westward", "eastward", "westward-180W", "eastward-180W",
    pytest.param("westward-180E", marks=RANGE_XFAIL), pytest.param("eastward-180E", marks=RANGE_XFAIL)])
def test_grid_lon_in_range(track):
    """John's convention: every saved longitude lies in [-180, 180). Today linear interpolation between
    in-range fixes stays in range, so this passes except where a fix is +180.0 itself; a fix that unwraps
    without re-wrapping (values like -180.7) fails it."""
    if track in ("westward", "eastward"):
        fx = WESTWARD if track == "westward" else EASTWARD
        grid = pipeline.build_sample_grid(fx, "1h")
    else:
        fx = fixes("2023-08-12 00:00", [17.0, 17.2, 17.4], SEAM_CASES[track])
        grid = pipeline.build_sample_grid(fx, "1h")
    lon = grid["lon"].to_numpy()
    bad = lon[(lon < -180.0) | (lon >= 180.0)]
    assert bad.size == 0, f"{bad.size} lon(s) outside [-180, 180): {bad[:5].tolist()}"


@pytest.mark.xfail(**XFAIL, reason="B-v2-1 antimeridian: crop centre of an interpolated frame between 179E and 179W")
@pytest.mark.parametrize("track", ["westward", "eastward"])
def test_crop_centred_on_interpolated_crossing_frame(track):
    """End to end from fixes to crop: the pipeline passes the grid's lon to crop.process_one_scan, which crops
    with projection.crop_box. The interpolated frames between the fixes either side of 180 deg must be cropped
    full size with the true position well inside the crop (the same check as the geometry test above)."""
    fx = WESTWARD if track == "westward" else EASTWARD
    grid = pipeline.build_sample_grid(fx, "1h")
    want = expected_lon(fx["time"], fx["lon"].tolist(), grid["time"])
    da = synthetic_disk()
    bad = []
    for i in range(7, 12):  # the interpolated frames of the segment that crosses 180 deg (fixes at 6 and 12 h)
        c = projection.crop_box(da, grid["lat"].iloc[i], grid["lon"].iloc[i], 600.0, GOES18_CRS, GOES18_H)
        if not centred(c, grid["lat"].iloc[i], want[i]):
            bad.append(f"row {i}: lon {grid['lon'].iloc[i]:.3f} (expected {want[i]:.3f}), crop shape {c.shape}")
    assert not bad, "; ".join(bad)


@pytest.mark.xfail(**XFAIL, reason="B-v2-1 antimeridian: concurrency.position_at interpolates lon linearly")
@pytest.mark.parametrize("track", ["westward", "eastward"])
def test_concurrency_position_at_crossing(track):
    # Both directions: a fix that wraps only one way passes one of these and fails the other.
    fx = WESTWARD if track == "westward" else EASTWARD
    times = pd.date_range(fx["time"].iloc[0], periods=19, freq="1h")
    _, lon = C.position_at(fx, times)
    err, msg = lon_mismatch(lon, expected_lon(fx["time"], fx["lon"].tolist(), times))
    assert err < TOL_DEG, msg


@pytest.mark.xfail(**XFAIL, reason="B-v2-1 antimeridian: concurrency.position_at either side of a fix at 180")
@pytest.mark.parametrize("case", list(SEAM_CASES))
def test_concurrency_position_at_fix_exactly_at_180(case):
    lons = SEAM_CASES[case]
    fx = fixes("2023-08-12 00:00", [17.0, 17.2, 17.4], lons)
    times = pd.date_range("2023-08-12 00:00", "2023-08-12 12:00", freq="3h", tz="UTC")
    _, lon = C.position_at(fx, times)
    err, msg = lon_mismatch(lon, expected_lon(fx["time"], lons, times))
    assert err < TOL_DEG, msg


@pytest.mark.xfail(**XFAIL, reason="B-v2-1 antimeridian: concurrency.boxes_overlap differences lon without wrapping")
@pytest.mark.parametrize("lon1, lon2", [(179.9, -179.9), (-179.9, 179.9)])
def test_concurrency_boxes_overlap_across_180(lon1, lon2):
    # Two centres 0.2 deg of longitude apart (about 21 km at 18N) on either side of 180 deg, so the boxes overlap.
    # Both argument orders: a fix that wraps only one sign of the difference fails one of them.
    dx_km = 0.2 * C.KM_PER_DEG * np.cos(np.radians(18.0))
    assert dx_km < 25.0
    assert bool(C.boxes_overlap(18.0, lon1, 18.0, lon2)), f"{lon1} and {lon2} read as ~360 deg apart"


@pytest.mark.xfail(**XFAIL, reason="B-v2-1 antimeridian: boxes_overlap, centres on opposite sides of 180, not near it")
@pytest.mark.parametrize("lon1, lon2", [(178.5, -178.5), (-178.5, 178.5)])
def test_concurrency_boxes_overlap_straddling_pair(lon1, lon2):
    # 3 deg of longitude apart at 18N: 3 * 111.32 * cos 18N = 317.6 km < 650 km, so the boxes overlap.
    dx_km = 3.0 * C.KM_PER_DEG * np.cos(np.radians(18.0))
    assert 300.0 < dx_km < 330.0
    assert bool(C.boxes_overlap(18.0, lon1, 18.0, lon2)), f"{lon1} and {lon2} ({dx_km:.0f} km apart) do not overlap"


@pytest.mark.parametrize("lon1, lon2", [(175.0, -175.0), (-175.0, 175.0)])
def test_concurrency_boxes_far_pair_across_180(lon1, lon2):
    # 10 deg apart at 18N: 1059 km > 650 km, so the boxes must not overlap. Passes today; a fix must keep it.
    assert 1000.0 < 10.0 * C.KM_PER_DEG * np.cos(np.radians(18.0))
    assert not bool(C.boxes_overlap(18.0, lon1, 18.0, lon2))
