"""E3 East Pacific training-storm selection (reports/v2_survey/e3/design.md §5, §5a, §6 VZA row, Disclosure).

    python scripts/select_e3_ep.py --nepac data/interim/hurdat2-nepac-1949-2025-092926.txt \
        --atlantic-hurdat2 data/interim/hurdat2_atlantic.txt --train-list storms_train_expanded.txt \
        --val-list storms_val_expanded.txt --control-list storms_train_control2.txt \
        --processed-root data/processed --excluded-crops excluded_crops.txt --satellite noaa-goes18 \
        --out storms_train_e3.txt --record reports/v2_survey/e3/selection_record.json

Every argument is required; nothing defaults to a real data file. Reads only; downloads nothing; never writes
any input (the Atlantic HURDAT2 cache included). The two outputs are written only after the concurrency hard gate
passes on every kept frame, and an existing output with other bytes is never overwritten.

Pipeline per EP candidate (each step cites the design row it implements):
  EP headers of the pinned nepac file, years of §5 narrowed by §5a -> skip the sealed IDs (H-EP set and its
  concurrency exclusions, hurdat2.sealed_ids) before any read -> hurdat2.parse_storm with the default statuses
  -> 00/06/12/18Z fixes only (F-43) -> NaN fixes dropped (F-44) -> pipeline.build_sample_grid at the Atlantic
  expansion's interval -> frames past the VZA cap dropped -> frames before the satellite's operational start
  dropped -> frames concurrent with an excluded storm dropped (scripts/concurrency.py) -> floors -> SHA-256
  order and cap -> the gate re-run on every kept frame -> write.

F-43 (binding): no fix or frame time, from either set, is printed, logged, written or put in an error. The
frame-level drops are therefore not persisted; the EP crop build (§9 step 3) re-derives them with the same
functions here (clean_fixes, ep_frames, gate_mask). The excluded storms are read as (time, lat, lon) only,
through hurdat2.read_positions; nothing about them (ID, name, wind, position, count) reaches an output.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hurricane_intensity import config, hurdat2, pipeline
from hurricane_intensity.splits import load_storm_list

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("concurrency", ROOT / "scripts" / "concurrency.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

# §5 years row: the Atlantic train list's span; §5a narrows it to the satellite's operational years.
BASE_YEARS = (2018, 2024)
# §5a table: nominal sub-points and GOES-West operational years. GOES-16 is John's fallback only (§5a); the
# script never picks a satellite itself, so --satellite is required.
SATELLITES = {
    "noaa-goes18": {"sub_lon": -137.2, "years": (2023, 2024), "operational_from": "2023-01-04"},
    "noaa-goes16": {"sub_lon": -75.2, "years": (2018, 2024), "operational_from": None},
}
ATLANTIC_SUB_LON = -75.2  # §5 VZA row: the cap is GOES-16's VZA over the Atlantic train + val fixes
# §5 order-and-cap row: E3's train size = F-31's atlantic arm (4044) minus the expanded list's frames (2081).
F31_ATLANTIC_FRAMES, EXPANDED_TRAIN_FRAMES = 4044, 2081
EP_TARGET_FRAMES = F31_ATLANTIC_FRAMES - EXPANDED_TRAIN_FRAMES  # 1963
SYNOPTIC_HOURS = (0, 6, 12, 18)  # F-43
EARTH_RADIUS_KM, GEO_HEIGHT_KM = 6378.137, 35786.0  # spherical Earth, geostationary altitude (§5 VZA row)


# --- fixes ---------------------------------------------------------------------------------------------------

def synoptic_only(fixes):
    """F-43: keep the 00/06/12/18Z fixes (minute and second zero); every other fix is dropped."""
    t = pd.DatetimeIndex(fixes["time"])
    keep = np.isin(t.hour, SYNOPTIC_HOURS) & (t.minute == 0) & (t.second == 0)
    return fixes[keep].reset_index(drop=True)


def drop_nan_fixes(fixes):
    """F-44: drop fixes with a NaN lat or lon before anything unwraps longitude. One NaN step would make
    pipeline.unwrap_lon's cumulative offset NaN for every later fix, and concurrency.position_at would then
    report no position (so no overlap) for the rest of the track."""
    ok = np.isfinite(fixes["lat"].to_numpy(float)) & np.isfinite(fixes["lon"].to_numpy(float))
    return fixes[ok].reset_index(drop=True)


def clean_fixes(fixes):
    """F-43 then F-44, sorted by time. Applied to every track this script reads."""
    return drop_nan_fixes(synoptic_only(fixes.sort_values("time").reset_index(drop=True)))


def vza_deg(lat, lon, sub_lon):
    """Viewing zenith angle (deg) of a geostationary satellite at `sub_lon` over (lat, lon), spherical Earth.
    Same formula for the Atlantic cap and the EP frames (§5 VZA row). Below the horizon gives > 90."""
    lat = np.radians(np.asarray(lat, float))
    dlon = np.radians(pipeline.lon_diff(np.asarray(lon, float), sub_lon))
    cos_g = np.cos(lat) * np.cos(dlon)
    sin_g = np.sqrt(np.clip(1.0 - cos_g ** 2, 0.0, None))
    return np.degrees(np.arctan2(sin_g, cos_g - EARTH_RADIUS_KM / (EARTH_RADIUS_KM + GEO_HEIGHT_KM)))


@contextlib.contextmanager
def quiet_reader():
    """hurdat2.parse_storm logs each storm's first and last fix time at INFO (F-43): silence it during reads."""
    log = logging.getLogger(hurdat2.__name__)
    was = log.disabled
    log.disabled = True
    try:
        yield
    finally:
        log.disabled = was


def headers(path):
    """[(storm ID, 1-based header line)] for every storm block of a HURDAT2 file, in file order."""
    out, lines, i = [], Path(path).read_text(encoding="utf-8").splitlines(), 0
    while i < len(lines):
        if lines[i][:2] in ("AL", "EP", "CP"):
            fields = [f.strip() for f in lines[i].split(",")]
            try:
                n = int(fields[2])
            except (IndexError, ValueError):
                raise ValueError(f"malformed header line {i + 1}") from None
            out.append((fields[0].upper(), i + 1))
            i += n + 1
        else:
            i += 1
    return out


def read_track(storm_id, path, line_no):
    """A non-sealed storm's status-filtered fixes (hurdat2.parse_storm, default statuses, §5 status row), then
    clean_fixes. None if no fix survives. Errors carry the ID and header line only (F-43)."""
    try:
        with quiet_reader():
            _, fixes = hurdat2.parse_storm(storm_id, path)
    except hurdat2.SealedStormError:
        raise
    except KeyError:  # no fix with a kept status: parse_storm's empty frame has no columns
        return None
    except (IndexError, ValueError):
        raise ValueError(f"{storm_id}: malformed storm block (header line {line_no})") from None
    fixes = clean_fixes(fixes)
    return fixes if len(fixes) else None


def excluded_tracks(nepac):
    """§5 concurrency gate: every sealed storm (H-EP set + concurrency exclusions, hurdat2.sealed_ids) as
    (time, lat, lon) only, via hurdat2.read_positions, then clean_fixes. Every fix, no status trim (Disclosure,
    binding note 2), synoptic only (F-43). Fails closed if one is missing or has no usable fix. Errors name
    neither the storm nor a time: only its index and the reader's line number."""
    tracks = []
    for k, sid in enumerate(sorted(hurdat2.sealed_ids()), 1):
        try:
            pos = hurdat2.read_positions(sid, nepac)
        except (ValueError, FileNotFoundError) as e:
            raise ValueError(f"excluded storm {k}: {str(e).replace(sid, '<excluded>')}") from None
        pos = clean_fixes(pos[["time", "lat", "lon"]])
        if not len(pos):
            raise ValueError(f"excluded storm {k}: no usable synoptic fix; the gate cannot run")
        tracks.append(pos)
    return tracks


# --- the Atlantic reference (§5 VZA, intensity and frame floors; sampling row) ---------------------------------

def grid_interval(manifest):
    """The sample-grid step of a built storm: the most common gap between its consecutive sample times."""
    t = pd.DatetimeIndex(pd.to_datetime(manifest["sample_time"], utc=True)).sort_values().unique()
    gaps = pd.Series(t[1:] - t[:-1])
    step = gaps.mode().min()
    hours = step / pd.Timedelta("1h")
    return f"{int(hours)}h" if hours == int(hours) else f"{int(step / pd.Timedelta('1min'))}min"


def atlantic_reference(train_list, val_list, control_list, atlantic_hurdat2, processed_root, excluded_crops):
    """The comparator's training/validation facts the §5 rules are defined from. Reads only."""
    val = load_storm_list(val_list)
    train = [s for s in load_storm_list(train_list) if s not in val]  # §5: the train list minus its val storms
    added = [s for s in train if s not in load_storm_list(control_list)]
    excluded = set(Path(excluded_crops).read_text().split())
    peaks, frames, intervals, vza_max = {}, {}, {}, -np.inf
    for sid in train + val:
        with quiet_reader():
            _, fixes = hurdat2.parse_storm(sid, atlantic_hurdat2)
        fixes = clean_fixes(fixes)
        vza_max = max(vza_max, float(np.max(vza_deg(fixes["lat"], fixes["lon"], ATLANTIC_SUB_LON))))
        if sid in val:
            continue
        peaks[sid] = float(fixes["wind_kt"].max())
        m = pd.read_csv(Path(processed_root) / sid / "manifest.csv")
        frames[sid] = int((m["crop_path"].notna() & ~m["crop_path"].isin(excluded)).sum())
        intervals[sid] = grid_interval(m)
    added_steps = {intervals[s] for s in added}
    if len(added_steps) != 1:  # §5 sampling row: "3-hourly per the list header" (the expanded arm's new storms)
        raise ValueError(f"the expanded list's added storms disagree on the sample interval: {sorted(added_steps)}")
    ref = {"vza_cap_deg": vza_max, "min_peak_kt": min(peaks.values()), "min_frames": min(frames.values()),
           "interval": added_steps.pop(), "atlantic_train_frames": sum(frames.values()),
           "atlantic_train_storms": len(train), "atlantic_val_storms": len(val),
           "atlantic_intervals": intervals, "atlantic_peaks_kt": peaks, "atlantic_frames": frames}
    # design "Open": every Atlantic train storm meets its own floors (by construction).
    assert all(peaks[s] >= ref["min_peak_kt"] and frames[s] >= ref["min_frames"] for s in train)
    return ref


# --- the EP frames ----------------------------------------------------------------------------------------------

def gate_mask(frames, tracks):
    """§5 hard gate: True where a frame's 650 km box overlaps an excluded storm's box at the same time
    (concurrency.concurrent, positions interpolated as the pipeline does)."""
    hit = np.zeros(len(frames), bool)
    for tr in tracks:
        hit |= C.concurrent(frames[["time", "lat", "lon"]], tr)
    return hit


def ep_frames(fixes, interval, sat, vza_cap, tracks):
    """(kept frames, counts) for one EP candidate. Frames are pipeline.build_sample_grid's at the Atlantic
    expansion's interval (§5 sampling row). The VZA cap (§5 VZA row) is applied to every frame: on an exact-fix
    frame that is the design's fix drop; interpolated frames get the same test, so no frame past the cap is kept
    and none is interpolated across a dropped stretch. Concurrent frames are dropped before counting (§5)."""
    grid = pipeline.build_sample_grid(fixes, interval)
    n_grid = len(grid)
    over = vza_deg(grid["lat"], grid["lon"], sat["sub_lon"]) > vza_cap
    grid = grid[~over]
    if sat["operational_from"]:  # §5a: GOES-West from this date
        grid = grid[pd.DatetimeIndex(grid["time"]) >= pd.Timestamp(sat["operational_from"], tz="UTC")]
    n_pre_gate = len(grid)
    grid = grid[~gate_mask(grid, tracks)].reset_index(drop=True)
    counts = {"frames_grid": n_grid, "frames_dropped_vza": int(over.sum()),
              "frames_dropped_concurrent": n_pre_gate - len(grid), "frames_kept": len(grid)}
    return grid, counts


def verify_no_overlap(kept, tracks):
    """§5 hard gate, after selection: re-run the box test on every kept frame; True if none overlaps."""
    for frames in kept.values():
        for tr in tracks:
            if C.concurrent(frames[["time", "lat", "lon"]], tr).any():
                return False
    return True


def storm_hash(sid):
    return hashlib.sha256(sid.encode("ascii")).hexdigest()


def select(nepac, satellite, ref):
    """§5 selection. Returns (selected IDs in hash order, per-candidate rows, kept frames, excluded tracks)."""
    sat = SATELLITES[satellite]
    y0, y1 = max(BASE_YEARS[0], sat["years"][0]), min(BASE_YEARS[1], sat["years"][1])
    sealed = hurdat2.sealed_ids()
    tracks = excluded_tracks(nepac)
    rows, kept = [], {}
    for sid, line_no in headers(nepac):
        if not sid.startswith("EP") or not y0 <= int(sid[-4:]) <= y1 or sid in sealed:
            continue  # §5 basin and years rows; sealed IDs are never read here (gate (d)) and never named
        row = {"id": sid, "sha256": storm_hash(sid)}
        fixes = read_track(sid, nepac, line_no)
        if fixes is None:
            rows.append({**row, "eligible": False, "reason": "no usable fix"})
            continue
        frames, counts = ep_frames(fixes, ref["interval"], sat, ref["vza_cap_deg"], tracks)
        row.update(counts, peak_kt=float(fixes["wind_kt"].max()))
        reasons = []
        if row["peak_kt"] < ref["min_peak_kt"]:
            reasons.append("peak below floor")
        if counts["frames_kept"] < ref["min_frames"]:
            reasons.append("frames below floor")
        row.update(eligible=not reasons, reason="; ".join(reasons) or None)
        rows.append(row)
        if not reasons:
            kept[sid] = frames
    order = sorted(kept, key=storm_hash)  # §5 order row: SHA-256 of the ID string
    selected, total = [], 0
    for sid in order:  # whole storms until >= the target; all of them if short (§5)
        if total >= EP_TARGET_FRAMES:
            break
        selected.append(sid)
        total += len(kept[sid])
    return selected, rows, {s: kept[s] for s in selected}, tracks


def vza_tertiles(kept, satellite):
    """§6 VZA row: tertile edges of E3's EP training fixes' VZA from the §5a satellite. The fixes are the kept
    exact-fix frames of the selected storms; numpy's default (linear) quantile at 1/3 and 2/3."""
    v = [vza_deg(f.loc[~f["is_interpolated"], "lat"], f.loc[~f["is_interpolated"], "lon"],
                 SATELLITES[satellite]["sub_lon"]) for f in kept.values()]
    v = np.concatenate(v) if v else np.array([])
    return [float(x) for x in np.quantile(v, [1 / 3, 2 / 3])] if len(v) else None


# --- output ---------------------------------------------------------------------------------------------------

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, text):
    """LF, UTF-8. An existing file with the same bytes is left alone; one with other bytes is refused."""
    data = text.encode("utf-8")
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"{path} exists with other bytes; not overwritten")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(data)


def check_outputs(outs, inputs):
    """Refuse an output that is an input, the Atlantic cache, the H-EP list, or under the sealed root."""
    protected = [Path(p).resolve() for p in inputs] + [Path(config.HURDAT2_CACHE_PATH).resolve(),
                                                       Path(config.HEP_LIST).resolve()]
    sealed_root = Path(config.HEP_SEALED_DIR).resolve()
    seen = set()
    for out in outs:
        r = Path(out).resolve()
        if r in protected or r in seen or r == sealed_root or sealed_root in r.parents:
            raise SystemExit(f"{out}: refusing to write an input, a protected file or the same output twice")
        seen.add(r)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--nepac", "--atlantic-hurdat2", "--train-list", "--val-list", "--control-list",
                 "--processed-root", "--excluded-crops", "--out", "--record"):
        p.add_argument(flag, required=True)
    p.add_argument("--satellite", required=True, choices=sorted(SATELLITES))
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    inputs = [a.nepac, a.atlantic_hurdat2, a.train_list, a.val_list, a.control_list, a.excluded_crops]
    check_outputs([a.out, a.record], inputs + [a.processed_root])
    hurdat2.verify_pinned_hurdat2(a.nepac)  # §5 source row: the pinned nepac file only
    ref = atlantic_reference(a.train_list, a.val_list, a.control_list, a.atlantic_hurdat2, a.processed_root,
                             a.excluded_crops)
    selected, rows, kept, tracks = select(a.nepac, a.satellite, ref)
    if not verify_no_overlap(kept, tracks):
        raise SystemExit("hard gate: a kept EP frame overlaps an excluded storm's box; nothing written")
    n_ep = sum(len(f) for f in kept.values())

    train_name, val_name = Path(a.train_list).name, Path(a.val_list).name
    lines = [f"# E3 train list (reports/v2_survey/e3/design.md sections 1, 5): {train_name} plus the East",
             f"# Pacific storms selected by scripts/select_e3_ep.py. Use with --val-list {val_name}.",
             f"# EP frames: the build's frames minus the section 5 VZA cap and concurrency gate ({n_ep} frames).",
             *load_storm_list(a.train_list),
             f"# East Pacific ({a.satellite}, design sections 5, 5a), SHA-256 order",
             *selected]
    record = {
        "design": "reports/v2_survey/e3/design.md §5, §5a, §6 (VZA row)",
        "rule_inputs": {
            "satellite": a.satellite, "sub_lon_deg": SATELLITES[a.satellite]["sub_lon"],
            "years": list(SATELLITES[a.satellite]["years"]), "base_years": list(BASE_YEARS),
            "statuses": list(config.DEFAULT_INCLUDE_STATUSES), "synoptic_hours_utc": list(SYNOPTIC_HOURS),
            "box_km": C.BOX_KM, "target_ep_frames": EP_TARGET_FRAMES,
            "scan_tolerance_min_not_used_in_selection": config.DEFAULT_TOLERANCE_MIN,
            "vza_formula": "spherical Earth, R=6378.137 km, h=35786 km", "atlantic_sub_lon_deg": ATLANTIC_SUB_LON,
            **{k: v for k, v in ref.items()},
            "input_sha256": {k: sha256_file(getattr(a, k)) for k in
                             ("nepac", "atlantic_hurdat2", "train_list", "val_list", "control_list", "excluded_crops")},
        },
        "candidates": rows,
        "hash_order": sorted([r["id"] for r in rows if r["eligible"]], key=storm_hash),
        "selected": selected,
        "ep_frames": n_ep,
        "vza_tertile_edges_deg": vza_tertiles(kept, a.satellite),
        "hard_gate": "passed: no kept frame overlaps",
    }
    write_new(a.out, "\n".join(lines) + "\n")
    write_new(a.record, json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"candidates: {len(rows)}  eligible: {len(record['hash_order'])}  selected: {len(selected)}  "
          f"EP frames: {n_ep} (target {EP_TARGET_FRAMES})")
    print(f"wrote {a.out} ({sha256_file(a.out)}) and {a.record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
