"""check_nan.py regenerates excluded_crops.txt without ever losing it (hc-adversary item 5, finding 15).

Every test runs in a temporary directory with fixture storms. None touches the real data/ or the real
excluded_crops.txt.
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from hurricane_intensity import dataset

OLD_SCRIPT_COMMIT = "2e55087"  # check_nan.py before this change, for the output comparison
SPENT_2025 = ["AL042025", "AL052025", "AL072025", "AL082025", "AL092025", "AL132025"]


@pytest.fixture
def check_nan(checkout_root):
    spec = importlib.util.spec_from_file_location("check_nan", checkout_root / "check_nan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is behind the __main__ guard, so this runs nothing
    return module


def write_storm(root, storm_id, nan_hours=(), hours=3):
    crops = root / storm_id / "crops"
    crops.mkdir(parents=True)
    rows = []
    for hour in range(hours):
        name = f"{storm_id}_20200101T{hour:02d}00Z.npz"
        bt = np.full((20, 22), 250.0, dtype=np.float32)
        if hour in nan_hours:
            bt[3, 4] = np.nan
        np.savez(crops / name, bt=bt)
        rows.append({"atcf_id": storm_id, "wind_kt": 50.0, "crop_path": os.path.join(storm_id, "crops", name)})
    pd.DataFrame(rows).to_csv(root / storm_id / "manifest.csv", index=False)
    return [r["crop_path"] for r in rows]


def entry(storm_id, hour):
    return os.path.join(storm_id, "crops", f"{storm_id}_20200101T{hour:02d}00Z.npz")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fixture checkout: NaN frames in an old storm and in EP052025, none in the clean storm."""
    monkeypatch.chdir(tmp_path)
    processed = tmp_path / "data" / "processed"
    write_storm(processed, "AL142018", nan_hours=(0, 2))
    write_storm(processed, "AL012020")
    write_storm(processed, "EP052025", nan_hours=(1,))
    return tmp_path


def test_importing_runs_nothing(repo, check_nan):
    assert not (repo / "excluded_crops.txt").exists()


def test_normal_run(repo, check_nan, capsys):
    check_nan.main()
    assert (repo / "excluded_crops.txt").read_text().splitlines() == [
        entry("AL142018", 0), entry("AL142018", 2), entry("EP052025", 1)]
    assert "3 of 9 excluded" in capsys.readouterr().out


def test_ep_2025_storm_is_scanned(repo, check_nan):
    check_nan.main()
    assert entry("EP052025", 1) in (repo / "excluded_crops.txt").read_text().splitlines()


def test_spent_2025_storms_never_loaded(repo, check_nan, monkeypatch):
    processed = repo / "data" / "processed"
    # One spent storm on disk in lower case: NTFS would serve it as the real storm.
    for storm_id in SPENT_2025[:-1] + [SPENT_2025[-1].lower()]:
        write_storm(processed, storm_id, nan_hours=(0,))
    # Existing entry for a spent storm: carried forward unchanged, never re-derived.
    spent_entry = entry("AL052025", 1)
    (repo / "excluded_crops.txt").write_text(spent_entry)

    built, loaded = [], []
    real_init, real_load = dataset.TCDataset.__init__, dataset.np.load
    monkeypatch.setattr(dataset.TCDataset, "__init__",
                        lambda self, root, ids, *a, **k: built.extend(ids) or real_init(self, root, ids, *a, **k))
    monkeypatch.setattr(dataset.np, "load", lambda p, *a, **k: loaded.append(str(p)) or real_load(p, *a, **k))
    check_nan.main()

    spent = {s.upper() for s in SPENT_2025}
    assert built and not [s for s in built if s.strip().upper() in spent]
    assert loaded and not [p for p in loaded if any(s in p.upper() for s in spent)]
    lines = (repo / "excluded_crops.txt").read_text().splitlines()
    assert spent_entry in lines
    assert not [p for p in lines if p != spent_entry and any(s in p.upper() for s in spent)]


def test_exception_mid_scan_leaves_file_identical(repo, check_nan, monkeypatch):
    existing = repo / "excluded_crops.txt"
    existing.write_bytes(b"AL142018\\crops\\AL142018_20200101T0000Z.npz\r\nAL142018\\crops\\x.npz")
    before = existing.read_bytes()
    calls = []

    def boom(self, idx):
        calls.append(idx)
        if len(calls) == 4:
            raise RuntimeError("simulated failure mid-scan")
        return real_getitem(self, idx)

    real_getitem = dataset.TCDataset.__getitem__
    monkeypatch.setattr(dataset.TCDataset, "__getitem__", boom)
    with pytest.raises(RuntimeError, match="mid-scan"):
        check_nan.main()
    assert existing.read_bytes() == before
    assert sorted(p.name for p in repo.iterdir()) == ["data", "excluded_crops.txt"]


def test_exception_on_write_leaves_file_identical(repo, check_nan, monkeypatch):
    existing = repo / "excluded_crops.txt"
    existing.write_text(entry("AL142018", 0))
    before = existing.read_bytes()

    def fail(*a, **k):
        raise OSError("simulated failure in os.replace")

    monkeypatch.setattr(check_nan.os, "replace", fail)
    with pytest.raises(OSError, match="os.replace"):
        check_nan.main()
    assert existing.read_bytes() == before
    assert sorted(p.name for p in repo.iterdir()) == ["data", "excluded_crops.txt"]


def test_removal_is_refused(repo, check_nan, capsys):
    existing = repo / "excluded_crops.txt"
    stale = os.path.join("AL012020", "crops", "AL012020_20200101T0100Z.npz")  # clean frame: a scan drops it
    existing.write_text("\n".join([entry("AL142018", 0), stale]))
    before = existing.read_bytes()
    with pytest.raises(SystemExit) as exc:
        check_nan.main()
    assert exc.value.code not in (0, None)
    assert existing.read_bytes() == before
    assert stale in capsys.readouterr().err


def test_added_entries_are_summarised(repo, check_nan, capsys):
    (repo / "excluded_crops.txt").write_text(entry("AL142018", 0))
    check_nan.main()
    out = capsys.readouterr().out
    assert "2 added" in out and entry("AL142018", 2) in out and entry("EP052025", 1) in out


@pytest.mark.parametrize("existing", [None, "same"], ids=["no-file", "rerun"])
def test_output_matches_old_script(repo, checkout_root, existing, tmp_path_factory):
    """The old script (before this change) and the new one write the same bytes on the same fixture."""
    old_src = subprocess.run(["git", "show", f"{OLD_SCRIPT_COMMIT}:check_nan.py"], cwd=checkout_root,
                             capture_output=True, text=True, check=True).stdout
    old_script = tmp_path_factory.mktemp("old") / "check_nan_old.py"
    old_script.write_text(old_src)
    env = {**os.environ, "PYTHONPATH": str(checkout_root / "src")}

    def run(script):
        r = subprocess.run([sys.executable, str(script)], cwd=repo, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines(), (repo / "excluded_crops.txt").read_bytes()

    if existing:
        run(old_script)
    old_out, old_bytes = run(old_script)
    if existing:
        run(old_script)
    else:
        (repo / "excluded_crops.txt").unlink()
    new_out, new_bytes = run(checkout_root / "check_nan.py")
    assert new_bytes == old_bytes
    assert new_out[:len(old_out)] == old_out
