"""StormWindowDataset on synthetic storms (no real data needed).

Three hourly storms of 16x16 frames; each frame's intensity encodes (storm, hour) as 1000*s + hour, so a
returned window says exactly which frames it holds:
    AL011999  40 frames, hours 0..39, no gaps             T=9, 3 h: starts 0..15  -> 16 windows, 0 skipped
    AL021999  30 frames, hours 0..29, frame 10 +4 min     starts 0..5             ->  6 windows, 0 skipped
    AL031999  hours 0..19 then 25..59 (a 6 h gap 19->25)  starts 0..19 all cross the gap (20 skipped);
                                                           starts 25..35          -> 11 windows
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from hurricane_intensity.storm_windows import (
    StormWindowDataset, delta_24h, make_loaders, make_loaders_from_split, rotate_window)

H = W = 16
T0 = pd.Timestamp("1999-08-01T00:00:00Z")
STORMS = {
    "AL011999": list(range(40)),
    "AL021999": list(range(30)),
    "AL031999": list(range(20)) + list(range(25, 60)),
}
IDS = list(STORMS)
EXPECTED = {"AL011999": (16, 0), "AL021999": (6, 0), "AL031999": (11, 20)}


OCEAN = (25.0, -60.0)  # open Atlantic
LAND = (30.0, -97.7)   # Austin, Texas


def write_storm(d, storm_id, hours, frames=None, minute_offsets=None, land_hours=(), status=None,
                not_fix_hours=()):
    """Synthetic storm files; every frame at OCEAN with status HU unless land_hours / status say otherwise."""
    if frames is None:
        frames = np.full((len(hours), H, W), 250.0)
    times = [T0 + pd.Timedelta(hours=h) for h in hours]
    if minute_offsets:
        times = [t + pd.Timedelta(minutes=minute_offsets.get(i, 0)) for i, t in enumerate(times)]
    s = IDS.index(storm_id) + 1 if storm_id in IDS else 9
    np.save(d / f"{storm_id}.npy", frames.astype(np.float32))
    pd.DataFrame({
        "time": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times],
        "intensity_kt": [1000.0 * s + h for h in hours],
        "is_fix": [int(h % 6 == 0 and h not in not_fix_hours) for h in hours],
        "lat": [(LAND if h in land_hours else OCEAN)[0] for h in hours],
        "lon": [(LAND if h in land_hours else OCEAN)[1] for h in hours],
        "status": [(status or {}).get(h, "HU") for h in hours],
    }).to_csv(d / f"{storm_id}.csv", index=False)


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    for storm_id, hours in STORMS.items():
        frames = rng.uniform(150.0, 350.0, size=(len(hours), H, W))  # beyond [180, 320] on both sides
        write_storm(tmp_path, storm_id, hours, frames,
                    minute_offsets={10: 4} if storm_id == "AL021999" else None)
    return tmp_path


def test_window_and_skip_counts(data_dir):
    for storm_id, (n, skipped) in EXPECTED.items():
        ds = StormWindowDataset(data_dir, [storm_id])
        assert (len(ds), ds.n_skipped) == (n, skipped), storm_id
    ds = StormWindowDataset(data_dir, IDS)
    assert len(ds) == 33 and ds.n_skipped == 20
    assert ds.skipped_per_storm == {k: v[1] for k, v in EXPECTED.items()}


def test_stride_and_step(data_dir):
    assert len(StormWindowDataset(data_dir, ["AL011999"], stride=2)) == 8  # starts 0, 2, ..., 14
    # 6 h steps, T=5 (still 24 h): of AL031999's starts 0..19 only 1, 7, 13, 19 step over hours 20..24
    # (16 skipped); starts 25..35 -> 11
    ds = StormWindowDataset(data_dir, ["AL031999"], T=5, step_hours=6)
    assert sorted(w[1] for w in ds.windows)[:4] == [1, 7, 13, 19]
    assert len(ds) == 15 and ds.n_skipped == 16


def test_tolerance(tmp_path):
    frames = np.full((12, H, W), 250.0)
    write_storm(tmp_path, "AL091999", list(range(12)), frames, minute_offsets={3: 6})
    ds = StormWindowDataset(tmp_path, ["AL091999"], T=3, step_hours=3, exclude_land=False)
    # starts 0..5 end by hour 11; the hour-3 frame is 6 min late, so starts 0 and 3 are skipped
    assert sorted(w[1] for w in ds.windows) == [1, 2, 4, 5] and ds.n_skipped == 2


def test_no_window_crosses_a_gap_or_storm(data_dir):
    ds = StormWindowDataset(data_dir, IDS)
    for i in range(len(ds)):
        item = ds[i]
        kt = item["intensities"].numpy()
        storm, hour = np.divmod(kt, 1000.0)
        assert (storm == IDS.index(item["storm_id"]) + 1).all()
        assert np.allclose(np.diff(hour), 3.0)
        assert not ((hour >= 20) & (hour <= 24) & (item["storm_id"] == "AL031999")).any()
        assert item["start_time"] == (T0 + pd.Timedelta(hours=float(hour[0]))).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert item["is_fix"].dtype == torch.bool
        assert item["is_fix"].tolist() == [h % 6 == 0 for h in hour]


def test_loader_shapes_and_normalization(data_dir):
    train, val, test = make_loaders(data_dir, ["AL011999"], ["AL021999"], ["AL031999"],
                                    batch_size=4, num_workers=0)
    assert train.dataset.augment and not val.dataset.augment and not test.dataset.augment
    batch = next(iter(val))
    assert batch["frames"].shape == (4, 9, 1, H, W) and batch["frames"].dtype == torch.float32
    assert batch["intensities"].shape == (4, 9) and batch["intensities"].dtype == torch.float32
    assert batch["is_fix"].shape == (4, 9)
    for loader in (train, val, test):
        for batch in loader:
            f = batch["frames"]
            assert f.min() >= 0.0 and f.max() <= 1.0
    f = next(iter(val))["frames"]
    assert f.min() == 0.0 and f.max() == 1.0  # clipping reached at both ends


def test_make_loaders_rejects_shared_storms(data_dir):
    with pytest.raises(ValueError, match="share storms"):
        make_loaders(data_dir, ["AL011999", "AL021999"], ["AL021999"], ["AL031999"], num_workers=0)


def test_same_rotation_across_window(tmp_path):
    # frame k = base + 2k K, all inside [180, 320]: a shared rotation (border padding) keeps frame k - frame 0
    # equal to the constant 2k/140 at every pixel; a different angle per frame would not.
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:H, 0:W]
    base = 200.0 + 3.0 * xx + 1.0 * yy + rng.uniform(0, 5, size=(H, W))
    frames = np.stack([base + 2.0 * k for k in range(12)])
    write_storm(tmp_path, "AL091999", list(range(12)), frames)
    plain = StormWindowDataset(tmp_path, ["AL091999"], T=4, step_hours=3, exclude_land=False)[0]["frames"]
    gen = torch.Generator().manual_seed(0)
    aug = StormWindowDataset(tmp_path, ["AL091999"], T=4, step_hours=3, augment=True,
                             generator=gen, exclude_land=False)
    for i in range(5):
        f = aug[0]["frames"]
        assert f.shape == (4, 1, H, W)
        assert not torch.allclose(f, plain, atol=1e-3)
        for k in range(1, 4):
            assert torch.allclose(f[k] - f[0], torch.full_like(f[0], 6.0 * k / 140.0), atol=1e-5)


def test_rotation_is_not_a_reflection():
    x = torch.arange(H * W, dtype=torch.float32).reshape(1, 1, H, W)
    r = rotate_window(x, math.pi / 2)
    assert any(torch.allclose(r, torch.rot90(x, k, dims=(2, 3)), atol=1e-3) for k in (1, 3))
    assert not torch.allclose(r, torch.flip(x, dims=(3,)), atol=1e-3)


def test_delta_24h():
    kt = torch.tensor([[30.0, 35, 40, 45, 50, 55, 60, 65, 70], [100.0, 95, 90, 85, 80, 75, 70, 65, 60]])
    assert torch.equal(delta_24h(kt, 3), torch.tensor([40.0, -40.0]))
    assert delta_24h(kt[0], 3).item() == 40.0
    assert torch.equal(delta_24h(kt[:, ::2], 6), torch.tensor([40.0, -40.0]))
    with pytest.raises(ValueError):
        delta_24h(kt[:, :8], 3)  # 24 h needs index 8
    with pytest.raises(ValueError):
        delta_24h(kt, 5)


def test_rejects_spent_and_malformed_ids(data_dir):
    with pytest.raises(ValueError, match="spent"):
        StormWindowDataset(data_dir, ["AL042025"])
    with pytest.raises(ValueError, match="malformed"):
        StormWindowDataset(data_dir, ["BOGUS99"])


def starts(ds):
    return [w[1] for w in ds.windows]


def test_land_filter(tmp_path):
    # hours 0..29, T=9 at 3 h: starts 0..5. Land at hour 5 (a start), 26 (24 h after start 2) and 13
    # (mid-window only, which the filter ignores).
    write_storm(tmp_path, "AL091999", list(range(30)), land_hours={5, 13, 26})
    ds = StormWindowDataset(tmp_path, ["AL091999"])
    assert starts(ds) == [0, 1, 3, 4] and ds.n_valid == 6 and ds.n_land == 2
    assert len(StormWindowDataset(tmp_path, ["AL091999"], exclude_land=False)) == 6


def test_status_filter(tmp_path):
    # start statuses: TD at 0 and 1, LO at 4, SS at 5 (not TS/HU) -> starts 2 and 3 kept; TS kept like HU
    write_storm(tmp_path, "AL091999", list(range(30)), status={0: "TD", 1: "TD", 2: "TS", 4: "LO", 5: "SS"})
    ds = StormWindowDataset(tmp_path, ["AL091999"])
    assert starts(ds) == [2, 3] and ds.n_status == 4
    assert len(StormWindowDataset(tmp_path, ["AL091999"], tropical_only=False)) == 6


def test_filters_count_independently(tmp_path):
    write_storm(tmp_path, "AL091999", list(range(30)), land_hours={0}, status={0: "TD"})
    ds = StormWindowDataset(tmp_path, ["AL091999"])
    assert (ds.n_land, ds.n_status, len(ds)) == (1, 1, 5)


def test_eval_only(tmp_path):
    # hours 0..47, starts 0..23. Fixes at 00/06/12/18 except hour 24, and an off-synoptic is_fix row
    # cannot appear (the converter only flags synoptic fixes). Eligible starts: 6, 12, 18 (0 -> 24 is no fix).
    write_storm(tmp_path, "AL091999", list(range(48)), not_fix_hours={24})
    ds = StormWindowDataset(tmp_path, ["AL091999"])
    assert len(ds) == 24 and ds.n_eval == 3
    assert [s for s, e in zip(starts(ds), ds.eval_eligible) if e] == [6, 12, 18]
    assert ds[6]["eval_eligible"] and not ds[0]["eval_eligible"]
    only = StormWindowDataset(tmp_path, ["AL091999"], eval_only=True)
    assert starts(only) == [6, 12, 18] and all(only.eval_eligible) and only.n_eval == 3


def test_filters_need_the_24h_frame(tmp_path):
    write_storm(tmp_path, "AL091999", list(range(30)))
    with pytest.raises(ValueError, match="24 h"):
        StormWindowDataset(tmp_path, ["AL091999"], T=5)
    assert len(StormWindowDataset(tmp_path, ["AL091999"], T=5, exclude_land=False)) == 18


def test_old_csv_without_filter_columns(tmp_path):
    write_storm(tmp_path, "AL091999", list(range(30)))
    p = tmp_path / "AL091999.csv"
    pd.read_csv(p).drop(columns=["lat", "lon", "status"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="rebuild"):
        StormWindowDataset(tmp_path, ["AL091999"])
    assert len(StormWindowDataset(tmp_path, ["AL091999"], exclude_land=False, tropical_only=False)) == 6


def test_make_loaders_from_split(data_dir, tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["AL011999"], "val": ["AL021999"], "test": ["AL031999"], "note": "x"}))
    train, val, test = make_loaders_from_split(split, data_dir, batch_size=4, num_workers=0)
    assert [len(dl.dataset) for dl in (train, val, test)] == [16, 6, 11]
    assert train.dataset.augment and not test.dataset.augment


def test_history_windows(tmp_path):
    # hours 0..39, frame value = 200 + hour K. history=2, T=9, 3 h: anchors need t-6 >= 0 and t+24 <= 39 -> 6..15.
    # Land at hour 0 (only ever a history frame) does not drop anything: filters apply at t and t+24 h.
    hours = list(range(40))
    frames = np.stack([np.full((H, W), 200.0 + h) for h in hours])
    write_storm(tmp_path, "AL091999", hours, frames, land_hours={0}, status={7: "TD"})
    ds = StormWindowDataset(tmp_path, ["AL091999"], history=2, load_positions=(0, 1, 2))
    assert ds.anchor == 2 and starts(ds) == [6, 8, 9, 10, 11, 12, 13, 14, 15]  # anchor 7 is TD
    item = ds[0]
    assert item["frames"].shape == (3, 1, H, W) and item["intensities"].shape == (11,)
    hour_of = item["frames"][:, 0, 0, 0] * 140.0 + 180.0 - 200.0
    assert torch.allclose(hour_of, torch.tensor([0.0, 3.0, 6.0]), atol=1e-3)
    assert item["anchor_time"].endswith("T06:00:00Z") and item["start_time"].endswith("T00:00:00Z")
    assert item["intensities"][ds.anchor].item() % 1000 == 6
    # eval flag at the anchor: t = 6 and t + 24 = 30 are fixes; t = 9 is not
    assert ds.eval_eligible[0] and not ds.eval_eligible[2]
    # history=0 as before: starts 0..15 minus 0 (land at its start) and 7 (TD)
    assert len(StormWindowDataset(tmp_path, ["AL091999"])) == 14


def test_history_requires_its_frames(tmp_path):
    # gap at hour 4: anchors whose t-6 / t-3 land on hour 4 (t = 7 and 10) are skipped like any gap
    hours = [h for h in range(40) if h != 4]
    write_storm(tmp_path, "AL091999", hours)
    ds = StormWindowDataset(tmp_path, ["AL091999"], history=2)
    anchor_hours = [int(ds.meta["AL091999"]["intensity"][w[1]]) % 1000 for w in ds.windows]
    assert anchor_hours == [6] + list(range(8, 10)) + list(range(11, 16))  # 7 and 10 need hour 4
    assert ds.n_skipped == 2  # anchors 7 and 10; hour 4 has no row, so it is never an anchor
    with pytest.raises(ValueError, match="load_positions"):
        StormWindowDataset(tmp_path, ["AL091999"], history=2, load_positions=(11,))
