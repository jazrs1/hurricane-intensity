"""check_nan.py regenerates excluded_crops.txt without ever losing it (hc-adversary items 5 and 9, close-out 8b).

Every test runs in a temporary directory with fixture storms. None touches the real data/ or the real
excluded_crops.txt.
"""
import importlib.util
import os
import pathlib
import shutil
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


def write_storm(root, storm_id, nan_hours=(), hours=3, atcf_id=None):
    crops = root / storm_id / "crops"
    crops.mkdir(parents=True)
    rows = []
    for hour in range(hours):
        name = f"{storm_id}_20200101T{hour:02d}00Z.npz"
        bt = np.full((20, 22), 250.0, dtype=np.float32)
        if hour in nan_hours:
            bt[3, 4] = np.nan
        np.savez(crops / name, bt=bt)
        rows.append({"atcf_id": atcf_id or storm_id, "wind_kt": 50.0,
                     "crop_path": os.path.join(storm_id, "crops", name)})
    pd.DataFrame(rows).to_csv(root / storm_id / "manifest.csv", index=False)
    return [r["crop_path"] for r in rows]


def entry(storm_id, hour):
    return os.path.join(storm_id, "crops", f"{storm_id}_20200101T{hour:02d}00Z.npz")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fixture checkout (its own git repo, so a missing excluded_crops.txt is untracked): NaN frames in an old
    storm and in EP052025, none in the clean storm."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))  # git never looks above the fixture
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    processed = tmp_path / "data" / "processed"
    write_storm(processed, "AL142018", nan_hours=(0, 2))
    write_storm(processed, "AL012020")
    write_storm(processed, "EP052025", nan_hours=(1,))
    return tmp_path


@pytest.fixture
def loads(monkeypatch):
    """Every crop np.load'ed through hurricane_intensity.dataset."""
    loaded = []
    real = dataset.np.load
    monkeypatch.setattr(dataset.np, "load", lambda p, *a, **k: loaded.append(str(p)) or real(p, *a, **k))
    return loaded


def spent_in(paths):
    return [p for p in paths if any(s in p.upper() for s in SPENT_2025)]


def assert_refused(check_nan, repo, capsys, reason, argv=()):
    existing = repo / "excluded_crops.txt"
    before = existing.read_bytes() if existing.exists() else None
    with pytest.raises(SystemExit) as exc:
        check_nan.main(list(argv))
    assert exc.value.code not in (0, None)
    assert (existing.read_bytes() if existing.exists() else None) == before
    assert not list(repo.glob("excluded_crops.txt.*.tmp"))
    err = capsys.readouterr().err
    assert reason in err, err
    return err


def test_importing_runs_nothing(repo, check_nan):
    assert not (repo / "excluded_crops.txt").exists()


def test_normal_run(repo, check_nan, capsys):
    check_nan.main([])
    assert (repo / "excluded_crops.txt").read_text().splitlines() == [
        entry("AL142018", 0), entry("AL142018", 2), entry("EP052025", 1)]
    assert "3 of 9 excluded" in capsys.readouterr().out


def test_ep_2025_storm_is_scanned(repo, check_nan):
    check_nan.main([])
    assert entry("EP052025", 1) in (repo / "excluded_crops.txt").read_text().splitlines()


def test_spent_2025_storms_never_loaded(repo, check_nan, monkeypatch, loads):
    processed = repo / "data" / "processed"
    for storm_id in SPENT_2025:
        write_storm(processed, storm_id, nan_hours=(0,))
    # Existing entry for a spent storm: carried forward unchanged, never re-derived.
    spent_entry = entry("AL052025", 1)
    (repo / "excluded_crops.txt").write_text(spent_entry)

    built = []
    real_init = dataset.TCDataset.__init__
    monkeypatch.setattr(dataset.TCDataset, "__init__",
                        lambda self, root, ids, *a, **k: built.extend(ids) or real_init(self, root, ids, *a, **k))
    check_nan.main([])

    assert built and not spent_in(built)
    assert loads and not spent_in(loads)
    lines = (repo / "excluded_crops.txt").read_text().splitlines()
    assert spent_entry in lines
    assert spent_in(lines) == [spent_entry]


# --- aliases of a storm directory (item 9, finding 1) -----------------------------------------------------

@pytest.mark.parametrize("name", ["al132025", "BOGUS99", "AL14201", "AL1420188"])
def test_malformed_dir_refused(repo, check_nan, capsys, loads, name):
    # A lowercase spent dir would load the real storm on NTFS; any non-ATCF name is refused outright.
    write_storm(repo / "data" / "processed", name, nan_hours=(0,))
    assert_refused(check_nan, repo, capsys, "not named like an ATCF ID")
    assert loads == []


@pytest.mark.parametrize("source", ["AL042025", "AL142018"], ids=["spent", "legal"])
def test_copied_manifest_refused(repo, check_nan, capsys, loads, source):
    processed = repo / "data" / "processed"
    if source == "AL042025":
        write_storm(processed, source, nan_hours=(1,))
    (processed / "ZZ042025").mkdir()
    shutil.copy(processed / source / "manifest.csv", processed / "ZZ042025" / "manifest.csv")
    assert_refused(check_nan, repo, capsys, "belong to another storm")
    assert loads == []


def test_manifest_with_other_atcf_id_refused(repo, check_nan, capsys, loads):
    write_storm(repo / "data" / "processed", "ZZ012020", nan_hours=(0,), atcf_id="AL042025")
    assert_refused(check_nan, repo, capsys, "belong to another storm")
    assert loads == []


@pytest.mark.skipif(sys.platform != "win32", reason="directory junctions are Windows-only")
def test_junction_refused(repo, check_nan, capsys, loads):
    import _winapi
    processed = repo / "data" / "processed"
    write_storm(processed, "AL042025", nan_hours=(1,))
    _winapi.CreateJunction(str(processed / "AL042025"), str(processed / "XX042025"))
    assert (processed / "XX042025" / "manifest.csv").exists()
    assert_refused(check_nan, repo, capsys, "links or junctions")
    assert loads == []


def test_stray_dataset_row_refused_before_any_load(repo, check_nan, capsys, monkeypatch):
    """Belt and braces: a row in ds.df outside the scanned storms is refused before __getitem__ runs."""
    write_storm(repo / "data" / "processed", "AL042025", nan_hours=(1,))
    real_init = dataset.TCDataset.__init__

    def init(self, root, ids, *a, **k):
        real_init(self, root, ids, *a, **k)
        extra = {"atcf_id": "AL042025", "wind_kt": 50.0, "crop_path": entry("AL042025", 1)}
        self.df = pd.concat([self.df, pd.DataFrame([extra])], ignore_index=True)

    got = []
    monkeypatch.setattr(dataset.TCDataset, "__init__", init)
    monkeypatch.setattr(dataset.TCDataset, "__getitem__", lambda self, i: got.append(i))
    assert_refused(check_nan, repo, capsys, "dataset rows outside the scanned storms")
    assert got == []


# --- failures leave the file as it was --------------------------------------------------------------------

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
        check_nan.main([])
    assert existing.read_bytes() == before
    assert sorted(p.name for p in repo.iterdir() if p.name != ".git") == ["data", "excluded_crops.txt"]


def test_exception_on_write_leaves_file_identical(repo, check_nan, monkeypatch):
    existing = repo / "excluded_crops.txt"
    existing.write_text(entry("AL142018", 0))
    before = existing.read_bytes()

    def fail(*a, **k):
        raise OSError("simulated failure in os.replace")

    monkeypatch.setattr(check_nan.os, "replace", fail)
    with pytest.raises(OSError, match="os.replace"):
        check_nan.main([])
    assert existing.read_bytes() == before
    assert sorted(p.name for p in repo.iterdir() if p.name != ".git") == ["data", "excluded_crops.txt"]


def test_failed_cleanup_does_not_mask_the_error(repo, check_nan, monkeypatch, capsys):
    """Item 9, finding 7: the temp file can't be removed either; the original error is still the one raised."""
    existing = repo / "excluded_crops.txt"
    existing.write_text(entry("AL142018", 0))
    before = existing.read_bytes()

    def fail(*a, **k):
        raise OSError("simulated failure in os.replace")

    real_unlink = pathlib.Path.unlink

    def locked(self, *a, **k):
        if self.name.endswith(".tmp"):
            raise PermissionError("simulated lock on the temp file")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(check_nan.os, "replace", fail)
    monkeypatch.setattr(pathlib.Path, "unlink", locked)
    with pytest.raises(OSError, match="os.replace") as exc:
        check_nan.main([])
    assert any("simulated lock" in note for note in getattr(exc.value, "__notes__", []))
    assert "simulated lock" in capsys.readouterr().err
    assert existing.read_bytes() == before


# --- a missing excluded_crops.txt (F-16): only "untracked inside a repo" (rc 1) may start from empty -------

def git_rc(repo):
    return subprocess.run(["git", "ls-files", "--error-unmatch", "--", "excluded_crops.txt"],
                          cwd=repo, capture_output=True).returncode


def test_missing_untracked_file_starts_empty(repo, check_nan):
    assert git_rc(repo) == 1
    check_nan.main([])
    assert (repo / "excluded_crops.txt").read_text().splitlines() == [
        entry("AL142018", 0), entry("AL142018", 2), entry("EP052025", 1)]


def test_missing_but_tracked_file_refused(repo, check_nan, capsys):
    existing = repo / "excluded_crops.txt"
    existing.write_text(entry("AL142018", 0))
    subprocess.run(["git", "add", "excluded_crops.txt"], cwd=repo, check=True, capture_output=True)
    existing.unlink()
    assert git_rc(repo) == 0
    assert_refused(check_nan, repo, capsys, "missing but tracked")
    assert not existing.exists()


def test_missing_file_outside_a_repo_refused(repo, check_nan, capsys):
    shutil.rmtree(repo / ".git")
    assert git_rc(repo) == 128
    assert_refused(check_nan, repo, capsys, "exited 128")
    assert not (repo / "excluded_crops.txt").exists()


@pytest.mark.parametrize("rc", [2, 129, -1])
def test_missing_file_with_other_git_rc_refused(repo, check_nan, capsys, monkeypatch, rc):
    monkeypatch.setattr(check_nan.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, rc, b"", b""))
    assert_refused(check_nan, repo, capsys, f"exited {rc}")


def test_missing_file_without_git_refused(repo, check_nan, capsys, monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError(2, "The system cannot find the file specified", "git")

    monkeypatch.setattr(check_nan.subprocess, "run", no_git)
    assert_refused(check_nan, repo, capsys, "git could not be run")


# --- existing entries -------------------------------------------------------------------------------------

def test_removal_is_refused(repo, check_nan, capsys):
    stale = os.path.join("AL012020", "crops", "AL012020_20200101T0100Z.npz")  # clean frame: a scan drops it
    (repo / "excluded_crops.txt").write_text("\n".join([entry("AL142018", 0), stale]))
    err = assert_refused(check_nan, repo, capsys, "would drop 1 existing entries")
    assert stale in err


def test_trailing_whitespace_is_not_a_removal(repo, check_nan):
    (repo / "excluded_crops.txt").write_text(entry("AL142018", 0) + "  \n" + entry("EP052025", 1) + "\t")
    check_nan.main([])
    assert (repo / "excluded_crops.txt").read_text().splitlines() == [
        entry("AL142018", 0), entry("AL142018", 2), entry("EP052025", 1)]


def test_duplicate_entries_are_reported(repo, check_nan, capsys):
    (repo / "excluded_crops.txt").write_text("\n".join([entry("AL142018", 0)] * 2))
    check_nan.main([])
    out = capsys.readouterr().out
    assert "notice: 1 duplicate entries" in out and entry("AL142018", 0) in out
    assert (repo / "excluded_crops.txt").read_text().splitlines().count(entry("AL142018", 0)) == 1


def test_added_entries_are_summarised(repo, check_nan, capsys):
    (repo / "excluded_crops.txt").write_text(entry("AL142018", 0))
    check_nan.main([])
    out = capsys.readouterr().out
    assert "2 added" in out and entry("AL142018", 2) in out and entry("EP052025", 1) in out


def test_entries_of_storm_under_no_root_carried_with_warning(repo, check_nan, capsys):
    """F-17: a storm with no directory under any known root (mistyped or removed) keeps its entries, with a warning."""
    absent = [entry("EP112025", 0), entry("AL992019", 2)]
    (repo / "excluded_crops.txt").write_text("\n".join(absent))
    check_nan.main([])
    lines = (repo / "excluded_crops.txt").read_text().splitlines()
    assert set(absent) <= set(lines)
    err = capsys.readouterr().err
    assert "warning: 2 entries carried forward for storms with no directory under" in err
    assert all(a in err for a in absent)


def test_bogus_entry_under_other_root_refused(repo, check_nan, capsys):
    """F-17 (hc-adversary 9b): an entry for a clean crop under the other root is re-derived there, so it is refused
    as a removal instead of being carried forward unchecked."""
    write_storm(repo / "data" / "hep_sealed", "EP112025", nan_hours=(2,))
    bogus = entry("EP112025", 0)  # a clean frame
    (repo / "excluded_crops.txt").write_text("\n".join([entry("AL142018", 0), bogus]))
    err = assert_refused(check_nan, repo, capsys, "would drop 1 existing entries")
    assert bogus in err


def test_entries_under_other_root_are_rederived(repo, check_nan, loads):
    write_storm(repo / "data" / "hep_sealed", "EP112025", nan_hours=(2,))
    (repo / "excluded_crops.txt").write_text(entry("EP112025", 2))
    check_nan.main([])  # scans data/processed, and EP112025 under data/hep_sealed
    assert any("hep_sealed" in p and "EP112025" in p for p in loads)
    assert entry("EP112025", 2) in (repo / "excluded_crops.txt").read_text().splitlines()


def test_storm_under_both_roots_refused(repo, check_nan, capsys):
    write_storm(repo / "data" / "hep_sealed", "AL012020")
    assert_refused(check_nan, repo, capsys, "more than one root")


# --- F-19 -----------------------------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="directory junctions are Windows-only")
@pytest.mark.parametrize("argv", [[], ["--processed-root", "data/hep_sealed"]], ids=["default", "hep-root"])
def test_known_root_junction_refused(repo, check_nan, capsys, loads, argv):
    """data/hep_sealed as a junction to data/processed would un-seal H-EP: refused before anything loads."""
    import _winapi
    _winapi.CreateJunction(str(repo / "data" / "processed"), str(repo / "data" / "hep_sealed"))
    assert (repo / "data" / "hep_sealed" / "AL142018" / "manifest.csv").exists()
    assert_refused(check_nan, repo, capsys, "roots that are links or junctions", argv)
    assert loads == []


@pytest.mark.parametrize("root", ["processed", "hep_sealed"])
def test_listed_storm_without_manifest_refused(repo, check_nan, capsys, loads, root):
    """A listed storm whose directory has no manifest.csv is a refusal with a message, not a FileNotFoundError."""
    (repo / "data" / root / "EP112025").mkdir(parents=True)
    (repo / "excluded_crops.txt").write_text(entry("EP112025", 0))
    err = assert_refused(check_nan, repo, capsys, "no manifest.csv")
    assert "EP112025" in err
    assert loads == []


@pytest.mark.parametrize("order", [("processed", "hep_sealed"), ("hep_sealed", "processed")])
def test_two_roots_carry_each_other_forward(repo, check_nan, order):
    write_storm(repo / "data" / "hep_sealed", "EP112025", nan_hours=(2,))
    for root in order + order:
        check_nan.main(["--processed-root", f"data/{root}"])
    assert (repo / "excluded_crops.txt").read_text().splitlines() == [
        entry("AL142018", 0), entry("AL142018", 2), entry("EP052025", 1), entry("EP112025", 2)]


def test_removal_refused_under_second_root(repo, check_nan, capsys):
    write_storm(repo / "data" / "hep_sealed", "EP112025", nan_hours=(2,))
    stale = entry("EP112025", 0)  # clean frame of a scanned storm
    (repo / "excluded_crops.txt").write_text("\n".join([entry("AL142018", 0), stale]))
    err = assert_refused(check_nan, repo, capsys, "would drop 1 existing entries", ["--processed-root", "data/hep_sealed"])
    assert stale in err


@pytest.mark.parametrize("existing", [None, "same"], ids=["no-file", "rerun"])
@pytest.mark.slow
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
