"""F-6: check.py --e2e never leaves an hc_e2e_* temp dir behind, and never passes silently while leaking.

The real check_e2e runs with the network, S3 and pipeline stubbed: the stub pipeline writes one small real .npz
crop and its manifest into the case's temp dir, and the model is one random ResNet-18 written to a fake main
checkout. Temp dirs go to a private tempfile.tempdir, so the hc_e2e_* count is exact. No network, no real S3,
no real models.
"""
import functools
import hashlib
import importlib.util
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("check_e2e_mod", Path(__file__).resolve().parents[1] / "scripts" / "check.py")
CHECK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CHECK)

STORM, TIME, LABEL = "AL142018", "2018-10-09T12:00:00Z", 90.0
CASE = {"name": "e2e stub", "storm": STORM, "bucket": "stub-bucket", "time": TIME, "hurdat2": None}
CROP = f"{STORM}/crops/{STORM}_20181009T1200Z.npz"


@pytest.fixture(scope="module")
def member(tmp_path_factory):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    path = tmp_path_factory.mktemp("member") / "seed0.pt"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture
def env(tmp_path, monkeypatch, member):
    """A fake main checkout, a private temp dir for hc_e2e_*, and the pipeline/S3/HURDAT2 stubs."""
    from hurricane_intensity import config, crop, hurdat2, pipeline
    main = (tmp_path / "main").resolve()
    (main / "data" / "interim").mkdir(parents=True)
    cache = main / "data" / "interim" / "hurdat2_atlantic.txt"
    cache.write_text("stub cache\n")
    (main / "models").mkdir()
    shutil.copy(member, main / "models" / "seed0.pt")
    digest = hashlib.sha256((main / "models" / "seed0.pt").read_bytes()).hexdigest()
    (main / CHECK.MEMBERS_SHA).parent.mkdir(parents=True)
    (main / CHECK.MEMBERS_SHA).write_text(f"{digest}  models/seed0.pt\n" * 2)  # two members, so a spread prints
    tmpdir = tmp_path / "temp"
    tmpdir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmpdir))
    monkeypatch.chdir(tmp_path)  # TCDataset reads ./excluded_crops.txt
    monkeypatch.setattr(config, "PROJECT_ROOT", main)
    monkeypatch.setattr(config, "HURDAT2_CACHE_PATH", cache)
    monkeypatch.setattr(CHECK, "E2E_TOLERANCE_KT", 1e9)  # a random model is far from the label; not tested here

    def no_network(*a, **k):
        raise AssertionError("the e2e test reached the network")
    monkeypatch.setattr(hurdat2, "download_hurdat2", lambda *a, **k: cache)
    monkeypatch.setattr(crop, "process_one_scan", lambda s3_client, key, *a, scratch_path=None, **k: None)

    def stub_pipeline(atcf_id, out_root, bucket):
        crop.process_one_scan(None, "stub/key.nc")  # check_e2e's wrapper records the key
        rng = np.random.default_rng(0)
        bt = rng.uniform(200.0, 290.0, size=(30, 34)).astype(np.float32)
        path = Path(out_root) / CROP
        path.parent.mkdir(parents=True)
        np.savez(path, bt=bt, dqf=np.zeros_like(bt))
        manifest = Path(out_root) / STORM / "manifest.csv"
        pd.DataFrame([{"atcf_id": STORM, "storm_name": "MICHAEL", "sample_time": TIME, "wind_kt": LABEL,
                       "is_interpolated": False, "status": "HU", "match_status": "ok", "scan_time": TIME,
                       "offset_min": 0.0, "source_key": "stub/key.nc", "crop_path": str(Path(CROP))}]).to_csv(
            manifest, index=False)
        return manifest
    monkeypatch.setattr(pipeline, "run_pipeline", stub_pipeline)
    fixes = pd.DataFrame({"time": [pd.Timestamp(TIME)], "lat": [25.0], "lon": [-86.0], "wind_kt": [LABEL],
                          "status": ["HU"]})
    monkeypatch.setattr(hurdat2, "parse_storm", lambda sid, path: ("MICHAEL", fixes))
    import requests
    monkeypatch.setattr(requests, "get", no_network)
    return main, tmpdir


def leftovers(tmpdir):
    return sorted(p.name for p in Path(tmpdir).glob("hc_e2e_*"))


def test_success_leaves_no_temp_dir(env):
    main, tmpdir = env
    msg = CHECK.check_e2e(main, CASE)
    assert msg.startswith(f"{STORM} {TIME}: ensemble") and leftovers(tmpdir) == []


@pytest.mark.parametrize("where", ["after the crop is read", "while the crop is open"])
def test_exception_partway_leaves_no_temp_dir_and_propagates(env, monkeypatch, where):
    main, tmpdir = env
    if where == "after the crop is read":
        import hurricane_intensity.dataset as dataset

        def planted(*a, **k):
            raise RuntimeError("planted failure in TCDataset")
        monkeypatch.setattr(dataset, "TCDataset", planted)
    else:
        real = np.count_nonzero

        def count(a, *args, **kw):
            if getattr(a, "shape", None) == (30, 34):
                raise RuntimeError("planted failure while the crop is open")  # inside the np.load block
            return real(a, *args, **kw)
        monkeypatch.setattr(np, "count_nonzero", count)
    with pytest.raises(RuntimeError, match="planted failure") as exc:
        CHECK.check_e2e(main, CASE)
    assert not getattr(exc.value, "__notes__", None)
    assert leftovers(tmpdir) == []


@pytest.mark.parametrize("lock", ["skipped", "raises"])
def test_a_temp_dir_that_cannot_be_removed_fails_the_case_with_its_path(env, monkeypatch, lock):
    main, tmpdir = env
    real_rmtree = shutil.rmtree

    def locked(path, *a, **k):
        if lock == "raises":
            raise PermissionError(32, "The process cannot access the file", str(Path(path) / CROP))
    monkeypatch.setattr(CHECK.shutil, "rmtree", locked)
    try:
        with pytest.raises(AssertionError, match="temp dir left behind: .*hc_e2e_") as exc:
            CHECK.check_e2e(main, CASE)
        (left,) = leftovers(tmpdir)
        assert left in str(exc.value)
    finally:
        for p in Path(tmpdir).glob("hc_e2e_*"):
            real_rmtree(p)


def test_cleanup_failure_does_not_mask_the_case_error(env, monkeypatch, capsys):
    main, tmpdir = env
    import hurricane_intensity.dataset as dataset

    def planted(*a, **k):
        raise RuntimeError("planted failure in TCDataset")
    monkeypatch.setattr(dataset, "TCDataset", planted)
    real_rmtree = shutil.rmtree
    monkeypatch.setattr(CHECK.shutil, "rmtree", lambda path, *a, **k: None)
    try:
        with pytest.raises(RuntimeError, match="planted failure") as exc:
            CHECK.check_e2e(main, CASE)
        notes = getattr(exc.value, "__notes__", [])
        assert any("temp dir left behind" in n and "hc_e2e_" in n for n in notes), notes
        assert "ALSO: temp dir left behind" in capsys.readouterr().out
    finally:
        for p in Path(tmpdir).glob("hc_e2e_*"):
            real_rmtree(p)
