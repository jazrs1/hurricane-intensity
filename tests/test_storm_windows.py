"""StormWindowDataset on synthetic storms (no real data needed).

Three hourly storms of 16x16 frames; each frame's intensity encodes (storm, hour) as 1000*s + hour, so a
returned window says exactly which frames it holds:
    AL011999  40 frames, hours 0..39, no gaps             T=9, 3 h: starts 0..15  -> 16 windows, 0 skipped
    AL021999  30 frames, hours 0..29, frame 10 +4 min     starts 0..5             ->  6 windows, 0 skipped
    AL031999  hours 0..19 then 25..59 (a 6 h gap 19->25)  starts 0..19 all cross the gap (20 skipped);
                                                           starts 25..35          -> 11 windows
"""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from hurricane_intensity.storm_windows import StormWindowDataset, delta_24h, make_loaders, rotate_window

H = W = 16
T0 = pd.Timestamp("1999-08-01T00:00:00Z")
STORMS = {
    "AL011999": list(range(40)),
    "AL021999": list(range(30)),
    "AL031999": list(range(20)) + list(range(25, 60)),
}
IDS = list(STORMS)
EXPECTED = {"AL011999": (16, 0), "AL021999": (6, 0), "AL031999": (11, 20)}


def write_storm(d, storm_id, hours, frames, minute_offsets=None):
    times = [T0 + pd.Timedelta(hours=h) for h in hours]
    if minute_offsets:
        times = [t + pd.Timedelta(minutes=minute_offsets.get(i, 0)) for i, t in enumerate(times)]
    s = IDS.index(storm_id) + 1 if storm_id in IDS else 9
    np.save(d / f"{storm_id}.npy", frames.astype(np.float32))
    pd.DataFrame({
        "time": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times],
        "intensity_kt": [1000.0 * s + h for h in hours],
        "is_fix": [int(h % 6 == 0) for h in hours],
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
    ds = StormWindowDataset(tmp_path, ["AL091999"], T=3, step_hours=3)
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
    plain = StormWindowDataset(tmp_path, ["AL091999"], T=4, step_hours=3)[0]["frames"]
    gen = torch.Generator().manual_seed(0)
    aug = StormWindowDataset(tmp_path, ["AL091999"], T=4, step_hours=3, augment=True, generator=gen)
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
