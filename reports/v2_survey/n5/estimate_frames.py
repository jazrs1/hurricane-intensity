"""N5-D: estimate frames for the PROPOSED Atlantic-only arm from HURDAT2, before any build.

Read-only. Uses the pipeline's own grid code (hurdat2.parse_storm + pipeline.build_sample_grid) on the
pinned Atlantic cache, so the estimate is the sample-time grid the build would attempt. No network, no
downloads; nothing is written outside this directory.

Also runs the N4 concurrency rule (Marco/Laura, review_b1.md I-1) for every candidate against every
validation, H9, 2025-test and H-EP storm, on the hourly grid.

    .venv\\Scripts\\python.exe -I reports/v2_survey/n5/estimate_frames.py <repo_root> <nepac_path>

Writes reports/v2_survey/n5/candidate_frames.csv and reports/v2_survey/n5/concurrency.csv.
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
PIN = "ee35819cba2b61ab0c2175601084def682813727ba209c2dab7d3af71d0bcb5c"
NEPAC_SHA = "caece626f7038db64a803a73f5985b563693f0130f3ff16b4b0a0a93c7aa0507"
assert hashlib.sha256(ATL.read_bytes()).hexdigest() == PIN
assert hashlib.sha256(nepac.read_bytes()).hexdigest() == NEPAC_SHA

# PROPOSED: survey §1 hurricanes (>= 64 kt) not in any list, minus AL022022 (aliased to EP042022),
# AL192019 Rebekah (70 kt only while extratropical) and AL182019 Pablo (GOES-16 edge). Coordinator 21:01/21:09.
CANDIDATES = [
    "AL072021", "AL272020", "AL282020",  # unused majors: Grace, Epsilon, Zeta
    "AL082018", "AL162018", "AL032018", "AL062024", "AL102019", "AL102024", "AL132024", "AL172020",
    "AL052024", "AL152023", "AL082020", "AL092020", "AL132018", "AL142023", "AL152022", "AL052021",
    "AL052022", "AL132022", "AL162022", "AL162024", "AL022018", "AL022019", "AL052023", "AL082021",
    "AL092018", "AL142020", "AL162020", "AL252020",
]
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"),
         (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, float("inf"), "Cat 5")]

listed = set()
for p in repo.glob("storms_*.txt"):
    listed |= set(load_storm_list(p))
assert not set(CANDIDATES) & listed, "candidate already in a list"
assert len(set(CANDIDATES)) == len(CANDIDATES) == 31

rows = []
for sid in CANDIDATES:
    name, fixes = hurdat2.parse_storm(sid, ATL)
    g = pipeline.build_sample_grid(fixes, "3h")
    r = {"atcf_id": sid, "name": name.title(), "start": f"{g.time.min():%Y-%m-%d}", "end": f"{g.time.max():%Y-%m-%d}",
         "peak_kt_tropical": int(fixes.wind_kt.max()), "frames_3h": len(g)}
    for lo, hi, b in BANDS:
        r[b] = int(((g.wind_kt >= lo) & (g.wind_kt < hi)).sum())
    r["ge96"], r["ge113"], r["ge137"] = (int((g.wind_kt >= t).sum()) for t in (96, 113, 137))
    rows.append(r)
cand = pd.DataFrame(rows)


# --- N4 concurrency rule, on the hourly grid of both storms -----------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def boxes_overlap(lat1, lon1, lat2, lon2, box_km=600.0):
    # flat-earth test as in review_b1.md: |dx|, |dy| < box at the mean latitude
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * np.cos(np.radians((lat1 + lat2) / 2))
    return (np.abs(dx) < box_km) & (np.abs(dy) < box_km)


protected = {}
for f in ("storms_val_expanded.txt", "storms_holdout_h9.txt", "storms_test_2025.txt"):
    for s in load_storm_list(repo / f):
        protected[s] = (f, ATL)
for s in ("EP052025", "EP062025", "EP112025", "EP162025", "EP112023", "EP182023"):  # PROPOSED H-EP
    protected[s] = ("H-EP (proposed)", nepac)

hourly = {}
for s, (_, path) in protected.items():
    _, fx = hurdat2.parse_storm(s, path)
    hourly[s] = pipeline.build_sample_grid(fx, "1h")[["time", "lat", "lon"]]

conc = []
for sid in CANDIDATES:
    _, fx = hurdat2.parse_storm(sid, ATL)
    h = pipeline.build_sample_grid(fx, "1h")[["time", "lat", "lon"]]
    g3 = set(pipeline.build_sample_grid(fx, "3h").time)
    for s, (role, _) in protected.items():
        m = h.merge(hourly[s], on="time", suffixes=("", "_o"))
        if m.empty:
            continue
        d = haversine_km(m.lat, m.lon, m.lat_o, m.lon_o)
        ov = boxes_overlap(m.lat, m.lon, m.lat_o, m.lon_o)
        ov650 = boxes_overlap(m.lat, m.lon, m.lat_o, m.lon_o, box_km=650.0)  # N4 rule (A4): 50 km margin
        within700 = d <= 700.0
        if (d < 1000).any():
            conc.append({
                "candidate": sid, "other": s, "other_role": role, "min_km": round(float(d.min()), 1),
                "hours_le700km": int(within700.sum()), "hours_box_overlap": int(ov.sum()),
                "frames3h_le700km": int(m.time[within700].isin(g3).sum()),
                "frames3h_box_overlap": int(m.time[ov].isin(g3).sum()),
                "hours_box650": int(ov650.sum()), "frames3h_box650": int(m.time[ov650].isin(g3).sum()),
                "box650_frames3h": ";".join(f"{t:%Y-%m-%dT%H}" for t in m.time[ov650] if t in g3),
                "box_overlap_times": ";".join(f"{t:%Y-%m-%dT%H}" for t in m.time[ov]),
            })
conc = pd.DataFrame(conc, columns=["candidate", "other", "other_role", "min_km", "hours_le700km", "hours_box_overlap",
                                   "frames3h_le700km", "frames3h_box_overlap", "hours_box650",
                                   "frames3h_box650", "box650_frames3h", "box_overlap_times"])

cand.to_csv(OUT / "candidate_frames.csv", index=False, lineterminator="\n")
conc.to_csv(OUT / "concurrency.csv", index=False, lineterminator="\n")
print(cand.to_string(index=False))
tot = cand.drop(columns=["atcf_id", "name", "start", "end", "peak_kt_tropical"]).sum()
print("\nTOTAL", tot.to_dict())
print("\nCONCURRENCY (< 1000 km at the same hour)\n", conc.drop(columns=["box_overlap_times", "box650_frames3h"]).to_string(index=False))
for r in conc.itertuples():
    print(r.candidate, r.other, "box600:", r.box_overlap_times, "| N4 box650 3h frames:", r.box650_frames3h)
