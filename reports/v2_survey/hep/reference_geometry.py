"""H-EP protocol draft: reference geometry and frame estimates, computed before any East Pacific download.

Read-only: reads data/processed/<id>/manifest.csv (lat/lon/source_key only; no crop is opened), the pinned Atlantic
HURDAT2 and the scratch nepac HURDAT2. No model, no network. Writes three CSVs next to this file:
  vza_reference.csv   per-set VZA distribution (training / validation / H9 frames, from their manifests)
  hep_estimates.csv   per H-EP storm: hourly-grid frames, >96 / >=137 counts, estimated VZA from GOES-West and -East
  hep_concurrency.csv H-EP storms vs every 2023-2025 nepac storm and 2023-2025 Atlantic storm within 1000 km

    .venv\\Scripts\\python.exe -I reports/v2_survey/hep/reference_geometry.py <repo_root> <nepac_path>
"""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo, nepac = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo / "src"))
from hurricane_intensity import hurdat2, pipeline  # noqa: E402
from hurricane_intensity.splits import load_storm_list  # noqa: E402

OUT = Path(__file__).resolve().parent
ATL = repo / "data/interim/hurdat2_atlantic.txt"
assert hashlib.sha256(ATL.read_bytes()).hexdigest().startswith("ee35819c")
assert hashlib.sha256(nepac.read_bytes()).hexdigest().startswith("caece626")
R, H = 6378.137, 35786.0
LON0 = {"G16": -75.0, "G19": -75.0, "G18": -137.0, "G17": -137.2}  # lon_0 as in the files read (survey §3, 10-15)
HEP = ["EP052025", "EP062025", "EP112025", "EP162025", "EP112023", "EP182023"]


def vza(lat, lon, sublon):
    lat, lon = np.radians(lat), np.radians(np.asarray(lon) - sublon)
    g = np.arccos(np.cos(lat) * np.cos(lon))
    return np.degrees(np.arctan2(np.sin(g), np.cos(g) - R / (R + H)))


def q(v):
    v = np.asarray(v)
    return {"n": len(v), "min": v.min(), "p5": np.percentile(v, 5), "p25": np.percentile(v, 25),
            "median": np.median(v), "p75": np.percentile(v, 75), "p95": np.percentile(v, 95), "max": v.max(),
            "share_gt_35deg": float((v > 35).mean()), "share_gt_45deg": float((v > 45).mean())}


val = load_storm_list(repo / "storms_val_expanded.txt")
train = [s for s in load_storm_list(repo / "storms_train_expanded.txt") if s not in val]
sets = {"training (storms_train_expanded minus val)": train, "validation (storms_val_expanded)": val,
        "H9": load_storm_list(repo / "storms_holdout_h9.txt")}
excluded = set((repo / "excluded_crops.txt").read_text(encoding="utf-8").split())
rows = []
for name, ids in sets.items():
    v_all, v_hi = [], []
    for s in ids:
        m = pd.read_csv(repo / "data/processed" / s / "manifest.csv")
        m = m[(m.match_status == "ok") & ~m.crop_path.isin(excluded)]
        plat = m.source_key.str.extract(r"_(G\d\d)_s")[0]
        assert plat.isin(["G16"]).all(), (s, plat.unique())
        v = vza(m.lat.to_numpy(), m.lon.to_numpy(), LON0["G16"])
        v_all.append(v)
        v_hi.append(v[m.wind_kt.to_numpy() > 96])
    rows.append({"set": name, "frames": "all", **q(np.concatenate(v_all))})
    rows.append({"set": name, "frames": ">96 kt", **q(np.concatenate(v_hi))})

# E2a's training set (item 8, finding 8b): expanded training + the 31 PROPOSED N5 storms. The new storms are not
# built, so their frames are ESTIMATED from the HURDAT2 3-hourly grid (positions only); recomputed from manifests
# immediately before the pass.
sys.path.insert(0, str(repo / "reports/v2_survey/n5"))
N5_NEW = ["AL072021", "AL272020", "AL282020", "AL082018", "AL162018", "AL032018", "AL062024", "AL102019", "AL102024",
          "AL132024", "AL172020", "AL052024", "AL152023", "AL082020", "AL092020", "AL132018", "AL142023", "AL152022",
          "AL052021", "AL052022", "AL132022", "AL162022", "AL162024", "AL022018", "AL022019", "AL052023", "AL082021",
          "AL092018", "AL142020", "AL162020", "AL252020"]
v_all, v_hi = [], []
for s in train:
    m = pd.read_csv(repo / "data/processed" / s / "manifest.csv")
    m = m[(m.match_status == "ok") & ~m.crop_path.isin(excluded)]
    v = vza(m.lat.to_numpy(), m.lon.to_numpy(), LON0["G16"])
    v_all.append(v)
    v_hi.append(v[m.wind_kt.to_numpy() > 96])
for s in N5_NEW:
    _, fx = hurdat2.parse_storm(s, ATL)
    g = pipeline.build_sample_grid(fx, "3h")
    v = vza(g.lat.to_numpy(), g.lon.to_numpy(), LON0["G16"])
    v_all.append(v)
    v_hi.append(v[g.wind_kt.to_numpy() > 96])
name = "E2a training (expanded + 31 N5 storms; new storms estimated from HURDAT2 3h grid)"
rows.append({"set": name, "frames": "all", **q(np.concatenate(v_all))})
rows.append({"set": name, "frames": ">96 kt", **q(np.concatenate(v_hi))})
ref = pd.DataFrame(rows)

est = []
grids = {}
for s in HEP:
    name, fx = hurdat2.parse_storm(s, nepac)
    g = pipeline.build_sample_grid(fx, "1h")
    grids[s] = g
    w, e = vza(g.lat, g.lon, LON0["G18"]), vza(g.lat, g.lon, LON0["G19"] if s.endswith("2025") else LON0["G16"])
    hi = g.wind_kt > 96
    est.append({"atcf_id": s, "name": name.title(), "start": f"{g.time.min():%Y-%m-%d %H}",
                "end": f"{g.time.max():%Y-%m-%d %H}", "frames_1h": len(g), "gt96": int(hi.sum()),
                "ge137": int((g.wind_kt >= 137).sum()), "peak_kt": float(g.wind_kt.max()),
                "lon_min": float(g.lon.min()), "lon_max": float(g.lon.max()),
                "vza_west_median": float(np.median(w)), "vza_west_max": float(w.max()),
                "vza_west_median_gt96": float(np.median(w[hi])) if hi.any() else np.nan,
                "vza_east_median": float(np.median(e)), "vza_east_median_gt96": float(np.median(e[hi])) if hi.any() else np.nan})
est = pd.DataFrame(est)

# concurrency (F-10 style: 650 km box margin) against every 2023-2025 storm in both files
def lines_ids(path):
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln[:2] in ("AL", "EP", "CP") and ln[2:4].isdigit():
            sid = ln.split(",")[0].strip()
            if 2023 <= int(sid[4:]) <= 2025:
                out.append(sid)
    return out


conc = []
for path in (nepac, ATL):
    for o in lines_ids(path):
        if o in HEP:
            continue
        try:
            _, fo = hurdat2.parse_storm(o, path)
        except Exception:
            continue
        go = pipeline.build_sample_grid(fo, "1h")[["time", "lat", "lon"]]
        for s in HEP:
            m = grids[s][["time", "lat", "lon"]].merge(go, on="time", suffixes=("", "_o"))
            if m.empty:
                continue
            dy = (m.lat_o - m.lat) * 111.32
            dx = (m.lon_o - m.lon) * 111.32 * np.cos(np.radians((m.lat + m.lat_o) / 2))
            dist = np.hypot(dx, dy)
            if (dist < 1000).any():
                conc.append({"hep": s, "other": o, "min_km_flat": round(float(dist.min()), 1),
                             "hours_box650": int(((dx.abs() < 650) & (dy.abs() < 650)).sum())})
conc = pd.DataFrame(conc, columns=["hep", "other", "min_km_flat", "hours_box650"])

kw = dict(index=False, lineterminator="\n", float_format="%.3f")
ref.to_csv(OUT / "vza_reference.csv", **kw)
est.to_csv(OUT / "hep_estimates.csv", **kw)
conc.to_csv(OUT / "hep_concurrency.csv", **kw)
pd.set_option("display.width", 250)
print(ref.round(2).to_string(index=False))
print(est.round(1).to_string(index=False))
print(conc.to_string(index=False))
