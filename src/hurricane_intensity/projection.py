"""ABI fixed-grid geostationary projection handling.

The GOES ABI "fixed grid" is NOT a lat/lon grid. Each file's x/y coordinate
arrays are scan angles (radians) in a geostationary projection, defined by
metadata stored in the file's `goes_imager_projection` variable. Treating
x/y as if they were lon/lat is a common and serious bug -- this module
exists specifically to avoid it, by building the real projection from each
file's own metadata and using pyproj to transform between it and lat/lon.

Reference: "GOES-R Series Product Definition and Users' Guide", section on
the ABI fixed grid format.
"""

from __future__ import annotations

import math

import pyproj
import xarray as xr


def get_geostationary_crs(ds: xr.Dataset) -> tuple[pyproj.CRS, float]:
    """Build the pyproj CRS for an ABI file's own fixed-grid projection.

    Returns (crs, satellite_height_m). The height is needed separately
    because ABI x/y coordinates are stored as scan angles in radians
    (projected meters / satellite height), not meters.
    """
    proj_var = ds["goes_imager_projection"].attrs
    h = float(proj_var["perspective_point_height"])
    a = float(proj_var["semi_major_axis"])
    b = float(proj_var["semi_minor_axis"])
    lon_0 = float(proj_var["longitude_of_projection_origin"])
    sweep = proj_var["sweep_angle_axis"]

    proj4 = (
        f"+proj=geos +h={h} +a={a} +b={b} "
        f"+lon_0={lon_0} +sweep={sweep} +units=m +no_defs"
    )
    crs = pyproj.CRS.from_proj4(proj4)
    return crs, h


def latlon_to_abi_xy(
    lat: float, lon: float, crs: pyproj.CRS, sat_height_m: float
) -> tuple[float, float]:
    """Convert a lat/lon point to ABI fixed-grid (x, y) scan-angle radians."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_m, y_m = transformer.transform(lon, lat)
    return x_m / sat_height_m, y_m / sat_height_m


def _box_corners_latlon(
    center_lat: float, center_lon: float, box_km: float
) -> list[tuple[float, float]]:
    """Four corners of a box_km x box_km box centered on (lat, lon).

    Uses a local flat-Earth approximation (constant km/degree latitude,
    km/degree longitude scaled by cos(latitude) at the center). This is not
    a full geodesic square, but the error over a 600 km box at tropical/
    subtropical latitudes is a few km at most -- well under one ABI pixel
    (~2 km) of consequence for a crop box, so it's not worth the extra
    complexity of a true geodesic calculation here.
    """
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))

    half = box_km / 2.0
    dlat = half / km_per_deg_lat
    dlon = half / km_per_deg_lon

    return [
        (center_lat + dlat, center_lon - dlon),
        (center_lat + dlat, center_lon + dlon),
        (center_lat - dlat, center_lon - dlon),
        (center_lat - dlat, center_lon + dlon),
    ]


def _ordered_slice(coord: xr.DataArray, lo: float, hi: float) -> slice:
    """xarray label-slicing requires slice bounds in the coordinate's own
    order. ABI x is ascending (W->E) and y is descending (N->S) by
    convention, but we check the actual values rather than assume it.
    """
    values = coord.values
    if values[0] <= values[-1]:
        return slice(lo, hi)
    return slice(hi, lo)


def crop_box(
    da: xr.DataArray,
    center_lat: float,
    center_lon: float,
    box_km: float,
    crs: pyproj.CRS,
    sat_height_m: float,
) -> xr.DataArray:
    """Crop a box_km x box_km box centered on (lat, lon) out of an ABI
    DataArray, correctly accounting for the fixed-grid projection.
    """
    corners = _box_corners_latlon(center_lat, center_lon, box_km)
    xs, ys = [], []
    for clat, clon in corners:
        x, y = latlon_to_abi_xy(clat, clon, crs, sat_height_m)
        xs.append(x)
        ys.append(y)

    x_slice = _ordered_slice(da["x"], min(xs), max(xs))
    y_slice = _ordered_slice(da["y"], min(ys), max(ys))
    return da.sel(x=x_slice, y=y_slice)
