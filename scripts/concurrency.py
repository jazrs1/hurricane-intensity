"""Storm concurrency: do two storms' crop boxes overlap at the same time? (N4 rule, F-10)

A training storm whose crop box overlaps a validation, held-out or test storm's box at the same time can put
that storm's outer cloud into training frames (review_b1.md I-1, Marco AL142020 vs Laura AL132020). The N4 rule
(reports/v2_survey/n5/design.md section 2, decided A4) drops such training frames, with a 50 km margin on the
600 km box: flat-earth at the mean latitude, |dx| < 650 km and |dy| < 650 km between centres.

Pure functions on (time, lat, lon) tables, plus track() to read a storm from a HURDAT2 file the way the
pipeline builds its sample grid. Nothing here downloads or writes anything.
"""
import numpy as np
import pandas as pd

KM_PER_DEG = 111.32
BOX_KM = 650.0  # 600 km crop box + 50 km margin (N4, A4)


def boxes_overlap(lat1, lon1, lat2, lon2, box_km=BOX_KM):
    """Flat-earth box test at the mean latitude: |dx| < box_km and |dy| < box_km (element-wise)."""
    lat1, lon1, lat2, lon2 = (np.asarray(v, float) for v in (lat1, lon1, lat2, lon2))
    dy = (lat2 - lat1) * KM_PER_DEG
    dx = (lon2 - lon1) * KM_PER_DEG * np.cos(np.radians((lat1 + lat2) / 2))
    return (np.abs(dx) < box_km) & (np.abs(dy) < box_km)


def position_at(fixes, times):
    """The other storm's (lat, lon) at `times`, linearly interpolated between its fixes as the pipeline does.
    NaN outside the first..last fix, so a storm that does not exist yet (or any more) never overlaps."""
    fixes = fixes.sort_values("time")
    t_fix = (pd.DatetimeIndex(fixes["time"]) - pd.Timestamp(0, tz="UTC")).total_seconds().to_numpy()
    t = (pd.DatetimeIndex(times) - pd.Timestamp(0, tz="UTC")).total_seconds().to_numpy()
    inside = (t >= t_fix[0]) & (t <= t_fix[-1])
    lat = np.where(inside, np.interp(t, t_fix, fixes["lat"].to_numpy()), np.nan)
    lon = np.where(inside, np.interp(t, t_fix, fixes["lon"].to_numpy()), np.nan)
    return lat, lon


def concurrent(track_df, other_fixes, box_km=BOX_KM):
    """Boolean array over the rows of track_df (time, lat, lon): True where the other storm exists at that time
    and the two boxes overlap."""
    lat, lon = position_at(other_fixes, track_df["time"])
    ok = ~np.isnan(lat)
    hit = np.zeros(len(track_df), bool)
    hit[ok] = boxes_overlap(track_df["lat"].to_numpy()[ok], track_df["lon"].to_numpy()[ok], lat[ok], lon[ok], box_km)
    return hit


def track(storm_id, hurdat2_path, interval="1h"):
    """(fixes, sample grid) for a storm, from a HURDAT2 file, with the pipeline's own parser and grid."""
    from hurricane_intensity import hurdat2, pipeline
    _, fixes = hurdat2.parse_storm(storm_id, hurdat2_path)
    return fixes, pipeline.build_sample_grid(fixes, interval)
