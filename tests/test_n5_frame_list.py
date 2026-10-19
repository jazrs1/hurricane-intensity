"""The committed N5 frame list is reproducible (scripts/n5_frame_list.py on the main checkout's built data, with this
checkout's excluded_crops.txt), and the Marco rule drops exactly the design's 4 frames or writes nothing.

Reads data/processed and the pinned Atlantic HURDAT2 in the main checkout; writes only under tmp_path.
"""
import hashlib
import importlib.util
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


def test_a_different_marco_drop_writes_nothing(main_root, tmp_path, monkeypatch):
    monkeypatch.chdir(CHECKOUT)
    monkeypatch.setattr(F, "DESIGN_DROPS", F.DESIGN_DROPS[:3])
    out = tmp_path / "frame_list_atlantic.txt"
    with pytest.raises(SystemExit, match="not the design's"):
        F.main(args(main_root, out))
    assert not out.exists()


def test_never_overwrites_other_bytes(main_root, tmp_path, monkeypatch):
    monkeypatch.chdir(CHECKOUT)
    out = tmp_path / "frame_list_atlantic.txt"
    out.write_text("something else\n")
    with pytest.raises(SystemExit, match="exists with other bytes"):
        F.main(args(main_root, out))
    assert out.read_text() == "something else\n"
