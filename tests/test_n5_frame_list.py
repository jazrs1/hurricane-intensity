"""The committed N5 frame list is reproducible (scripts/n5_frame_list.py on the main checkout's built data, with this
checkout's excluded_crops.txt), and the Marco rule drops exactly the design's 4 frames or writes nothing.

Reads data/processed and the pinned Atlantic HURDAT2 in the main checkout; writes only under tmp_path.
"""
import hashlib
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("n5_frame_list", CHECKOUT / "scripts" / "n5_frame_list.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
_spec = importlib.util.spec_from_file_location("analyze_n5_fl", CHECKOUT / "scripts" / "analyze_n5.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def args(main_root, out):
    return ["--processed-root", str(main_root / "data" / "processed"),
            "--hurdat2", str(main_root / "data" / "interim" / "hurdat2_atlantic.txt"), "--out", str(out)]


@pytest.mark.slow
def test_frame_list_reproduces_the_committed_file(main_root, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(CHECKOUT)  # TCDataset reads ./excluded_crops.txt; load_splits reads the lists
    out = tmp_path / "frame_list_atlantic.txt"
    frames = F.main(args(main_root, out))
    committed = (CHECKOUT / "results" / "n5_atlantic" / "frame_list_atlantic.txt").read_bytes()
    assert out.read_bytes() == committed
    assert hashlib.sha256(committed).hexdigest() == A.FRAME_LIST_SHA256 and len(frames) == A.ARM_TRAIN_FRAMES
    printed = capsys.readouterr().out
    for t in ("0900Z", "1200Z", "1500Z", "1800Z"):
        assert f"AL142020_20200825T{t}.npz" in printed and f"AL142020_20200825T{t}.npz" not in "\n".join(frames)
    assert {p.split("\\")[0] for p in frames} == set(A.ARM_IDS)


@pytest.mark.slow
def test_a_different_marco_drop_writes_nothing(main_root, tmp_path, monkeypatch):
    monkeypatch.chdir(CHECKOUT)
    monkeypatch.setattr(F, "DESIGN_DROPS", F.DESIGN_DROPS[:3])
    out = tmp_path / "frame_list_atlantic.txt"
    with pytest.raises(SystemExit, match="not the design's"):
        F.main(args(main_root, out))
    assert not out.exists()


@pytest.mark.slow
def test_never_overwrites_other_bytes(main_root, tmp_path, monkeypatch):
    monkeypatch.chdir(CHECKOUT)
    out = tmp_path / "frame_list_atlantic.txt"
    out.write_text("something else\n")
    with pytest.raises(SystemExit, match="exists with other bytes"):
        F.main(args(main_root, out))
    assert out.read_text() == "something else\n"


def test_frame_list_bytes_survive_checkout():
    """core.autocrlf=true would give the main checkout a CRLF copy whose SHA-256 is not FRAME_LIST_SHA256 (run_n5
    guard 2 hashes the working file); .gitattributes marks it -text."""
    r = subprocess.run(["git", "check-attr", "text", "--", "results/n5_atlantic/frame_list_atlantic.txt"],
                       cwd=CHECKOUT, capture_output=True, text=True, check=True)
    assert r.stdout.strip().endswith("text: unset"), r.stdout


def test_refuses_an_unpinned_hurdat2(main_root, tmp_path, monkeypatch):
    monkeypatch.chdir(CHECKOUT)
    bad = tmp_path / "hurdat2.txt"
    bad.write_bytes((main_root / "data" / "interim" / "hurdat2_atlantic.txt").read_bytes() + b"x")
    a = args(main_root, tmp_path / "out.txt")
    a[a.index("--hurdat2") + 1] = str(bad)
    with pytest.raises(SystemExit, match="not the pinned Atlantic HURDAT2"):
        F.main(a)


def test_refuses_an_other_basin_storm_from_marcos_year(main_root, tmp_path, monkeypatch):
    """An East Pacific protected storm from 2020 could share a time with Marco but isn't in the Atlantic file."""
    for name in (F.TRAIN_LIST, F.VAL_LIST, *F.PROTECTED_LISTS):
        shutil.copy(CHECKOUT / name, tmp_path / name)
    with open(tmp_path / "storms_holdout_hep.txt", "a", encoding="utf-8") as f:
        f.write("\nEP132020\n")
    monkeypatch.setattr(F, "ROOT", tmp_path)
    monkeypatch.chdir(CHECKOUT)
    with pytest.raises(SystemExit, match=r"\['EP132020'\] share Marco's year"):
        F.main(args(main_root, tmp_path / "out.txt"))


@pytest.mark.slow
def test_exclusions_keep_the_pinned_counts(main_root, monkeypatch):
    """The regenerated excluded_crops.txt added entries for the 31 new storms only: the control, validation and H9
    frame sets the analysis pins are unchanged."""
    from hurricane_intensity.dataset import TCDataset
    from hurricane_intensity.splits import load_splits
    monkeypatch.chdir(CHECKOUT)
    processed = str(main_root / "data" / "processed")
    ctl, val, _ = load_splits(train_list="storms_train_expanded.txt", val_list="storms_val_expanded.txt")
    assert len(TCDataset(processed, ctl).df) == A.CONTROL_TRAIN_FRAMES
    assert len(TCDataset(processed, val).df) == A.VAL_FRAMES
    h9 = TCDataset(processed, list(A.H9)).df
    assert h9["atcf_id"].value_counts().to_dict() == A.H9
    assert A.h9_pin(h9.assign(truth=h9["wind_kt"]).sort_values("crop_path")) == A.H9_PIN
    entries = (CHECKOUT / "excluded_crops.txt").read_text().splitlines()
    old = subprocess.run(["git", "show", "e8e4fc1:excluded_crops.txt"], cwd=CHECKOUT, capture_output=True,
                         check=True).stdout.decode().splitlines()
    assert [e for e in entries if e in old] == old  # every earlier entry kept, in order
    assert {e.split("\\")[0] for e in entries if e not in old} <= set(A.NEW_IDS)
