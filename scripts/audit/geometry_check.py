"""Read-only: does every crop's recorded shape match what the committed crop code produces for its lat/lon?

No download needed: the GOES-16 full-disk 2 km fixed grid is fixed (5424 x 5424, 56 microradian spacing),
so projection.crop_box's output shape is a pure function of (lat, lon, box_km).
"""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "src"))
from hurricane_intensity import config, projection  # the working-tree code, uncommitted edits included

ROOT = REPO / "data" / "processed"
H, A, B, LON0 = 35786023.0, 6378137.0, 6356752.31414, -75.0
crs = pyproj.CRS.from_proj4(f"+proj=geos +h={H} +a={A} +b={B} +lon_0={LON0} +sweep=x +units=m +no_defs")
N, STEP = 5424, 56e-6
x = -0.151844 + STEP * np.arange(N)   # ascending, W -> E
y = 0.151844 - STEP * np.arange(N)    # descending, N -> S


def expected_shape(lat, lon, box_km):
    xs, ys = zip(*(projection.latlon_to_abi_xy(clat, clon, crs, H) for clat, clon in projection._box_corners_latlon(lat, lon, box_km)))
    cols = int(((x >= min(xs)) & (x <= max(xs))).sum())
    rows = int(((y >= min(ys)) & (y <= max(ys))).sum())
    return rows, cols


groups = {"Aug 18-19": ["AL142018", "AL142024"],
          "Aug 24": ["AL052019", "AL062020", "AL092021", "AL142021", "AL102023", "AL012024", "AL042024", "AL192020", "AL162023", "AL132020"],
          "Aug 25 (held-out)": ["AL092022", "AL172022", "AL032021"],
          "Sep 13 (Beryl, Lee, Sam)": ["AL022024", "AL132023", "AL182021"]}
print(f"box_km in config: {config.DEFAULT_BOX_KM}")
out = []
for g, ids in groups.items():
    for s in ids:
        m = pd.read_csv(ROOT / s / "manifest.csv")
        m = m[m.crop_path.notna()]
        exp = np.array([expected_shape(r.lat, r.lon, config.DEFAULT_BOX_KM) for r in m.itertuples()])
        got = m.bt_shape.str.split("x", expand=True).astype(int).to_numpy()
        d = got - exp
        out.append(dict(built=g, storm=s, name=m.storm_name.iloc[0], frames=len(m),
                        exact=int((d == 0).all(axis=1).sum()), within_1px=int((np.abs(d) <= 1).all(axis=1).sum()),
                        max_abs_diff=int(np.abs(d).max()), lon_range=f"{m.lon.min():.0f}..{m.lon.max():.0f}", lat_range=f"{m.lat.min():.0f}..{m.lat.max():.0f}"))
pd.set_option("display.width", 250)
print(pd.DataFrame(out).to_string(index=False))

print("\nheld-out 'lat' column, committed predictions CSV vs manifest on disk:")
committed = pd.read_csv(io.StringIO(subprocess.run(["git", "show", "ed0401c:results/heldout_predictions.csv"], capture_output=True, text=True, check=True).stdout))
now = pd.concat([pd.read_csv(ROOT / s / "manifest.csv") for s in groups["Aug 25 (held-out)"]], ignore_index=True)
now = now[now.crop_path.notna()].reset_index(drop=True)
d = (committed.lat - now.lat).abs()
print(f"  rows differing: {int((d > 0).sum())}, max abs difference: {d.max():.3e} degrees")
