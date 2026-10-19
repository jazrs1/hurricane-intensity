"""The N4 / F-10 concurrency helper (scripts/concurrency.py): synthetic cases, and the real Marco/Laura pair.

The real-data tests read the protected Atlantic HURDAT2 cache in the main checkout, after checking its pinned
hash. They download nothing and write nothing.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hurricane_intensity.splits import load_storm_list

_spec = importlib.util.spec_from_file_location(
    "concurrency", Path(__file__).resolve().parents[1] / "scripts" / "concurrency.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

ATLANTIC_PIN = "ee35819cba2b61ab0c2175601084def682813727ba209c2dab7d3af71d0bcb5c"


def km_east(lat, km):
    return km / (C.KM_PER_DEG * np.cos(np.radians(lat)))


def frame(times, lat, lon):
    return pd.DataFrame({"time": pd.DatetimeIndex(times, tz="UTC"), "lat": lat, "lon": lon})


def test_box_rule_and_margin():
    lat = 20.0
    for dx, at_650, at_600 in ((620, True, False), (649, True, False), (651, False, False), (590, True, True)):
        lon2 = -60 + km_east(lat, dx)
        assert bool(C.boxes_overlap(lat, -60, lat, lon2)) is at_650, dx
        assert bool(C.boxes_overlap(lat, -60, lat, lon2, 600.0)) is at_600, dx
    # Both axes must be inside: 620 km east and 700 km north is outside, though each box edge is close.
    assert not C.boxes_overlap(lat, -60, lat + 700 / C.KM_PER_DEG, -60 + km_east(lat, 620))
    # A diagonal of 849 km (600 km on each axis) still overlaps at 650: the dropped 700 km centre rule missed this.
    assert C.boxes_overlap(lat, -60, lat + 600 / C.KM_PER_DEG, -60 + km_east(lat + 300 / C.KM_PER_DEG, 600))


def test_other_storm_only_counts_while_it_exists():
    times = pd.date_range("2020-08-25T00", periods=6, freq="h", tz="UTC")
    track = frame(times, [20.0] * 6, [-60.0] * 6)
    other = frame(["2020-08-25T02", "2020-08-25T04"], [20.0, 20.0], [-61.0, -61.0])
    assert C.concurrent(track, other).tolist() == [False, False, True, True, True, False]


def test_interpolates_the_other_storm_between_fixes():
    times = pd.date_range("2020-08-25T00", periods=3, freq="3h", tz="UTC")
    track = frame(times, [20.0] * 3, [-60.0] * 3)
    far, near = -60 - km_east(20.0, 2000), -60 - km_east(20.0, 100)
    other = frame(["2020-08-25T00", "2020-08-25T06"], [20.0, 20.0], [far, near])
    # At 03Z the other storm is halfway (about 1050 km west): outside. At 06Z it is 100 km west: inside.
    assert C.concurrent(track, other).tolist() == [False, False, True]


@pytest.fixture(scope="module")
def atlantic(main_root):
    path = main_root / "data" / "interim" / "hurdat2_atlantic.txt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ATLANTIC_PIN, f"{path} is not the pinned cache"
    return path


def test_marco_laura_reproduces_the_design_estimate(atlantic):
    """design.md section 2 / concurrency.csv: 12 hours under the 650 km box (8 under plain 600 km overlap), and on
    the 3-hourly grid exactly 2020-08-25 09Z, 12Z, 15Z, 18Z."""
    _, marco_1h = C.track("AL142020", atlantic, "1h")
    laura, _ = C.track("AL132020", atlantic)
    on_hour = marco_1h["time"].dt.minute.eq(0).to_numpy()
    assert int((C.concurrent(marco_1h, laura) & on_hour).sum()) == 12
    assert int((C.concurrent(marco_1h, laura, 600.0) & on_hour).sum()) == 8
    _, marco_3h = C.track("AL142020", atlantic, "3h")
    hit = marco_3h["time"][C.concurrent(marco_3h, laura)]
    assert [f"{t:%Y-%m-%dT%HZ}" for t in hit] == ["2020-08-25T09Z", "2020-08-25T12Z", "2020-08-25T15Z",
                                                  "2020-08-25T18Z"]


def test_n5_training_storms_overlap_no_protected_storm_except_marco_laura(checkout_root, atlantic):
    """F-10 run against the first v2 train list: every N5 training storm (hourly grid) against every validation,
    H9 and 2025-test storm. Marco/Laura is the only pair, and its frames are dropped by the frame list (PR B).
    H-EP and the 2025 East Pacific storms are not in the Atlantic file; the design's estimate (concurrency.csv,
    with the nepac file) found no N5 candidate overlapping them."""
    protected = set()
    for name in ("storms_val_expanded.txt", "storms_holdout_h9.txt", "storms_test_2025.txt"):
        protected |= set(load_storm_list(checkout_root / name))
    val = set(load_storm_list(checkout_root / "storms_val_expanded.txt"))
    train = [s for s in load_storm_list(checkout_root / "storms_train_n5_atlantic.txt") if s not in val]
    fixes = {s: C.track(s, atlantic)[0] for s in sorted(protected)}
    pairs = []
    for s in train:
        _, grid = C.track(s, atlantic, "1h")
        pairs += [(s, p) for p in sorted(protected) if C.concurrent(grid, fixes[p]).any()]
    assert pairs == [("AL142020", "AL132020")]
