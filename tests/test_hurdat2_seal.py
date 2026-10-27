"""HEP.md gate (d), B-v2-2b: the HURDAT2 reader refuses sealed storms; the positions-only reader for E3's gate.

Sealed = the IDs in storms_holdout_hep.txt (read by reference) plus config.HEP_CONCURRENT_IDS. Fixtures only:
the nepac file here is synthetic. Each sealed ID gets a made-up track with a sentinel name, wind and status, so
the tests can show none of them reaches an output, a log line or an error. No real H-EP label data is used.
The Atlantic cache's hash (a temp stand-in and the real protected file, read only) is checked around every test.
"""
import hashlib
import importlib.util
import inspect
import logging
import re
from pathlib import Path

import pandas as pd
import pytest

from hurricane_intensity import config, goes_index, hurdat2, pipeline
from hurricane_intensity.splits import load_storm_list

ROOT = Path(__file__).resolve().parents[1]
HEP = load_storm_list(ROOT / "storms_holdout_hep.txt")
CONCURRENT = list(config.HEP_CONCURRENT_IDS)
SEALED = HEP + CONCURRENT
OPEN_EP = "EP012099"
SENTINEL_NAME = "SENTINELNAME"
SENTINEL_WIND = "173"
ATLANTIC_FIXTURE = "AL012099,              TESTC,      1,\n20990701, 0000,  , TS, 25.0N,  80.0W,  40, 1000,\n"
PAD = ", 1000,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -999"


def block(atcf_id, name, rows):
    out = [f"{atcf_id}, {name:>18},     {len(rows)},"]
    out += [f"{d}, {t},  , {s}, {lat}, {lon}, {w:>4}{PAD}" for d, t, s, lat, lon, w in rows]
    return out


TRACK = [("20990801", "0000", "LO", "15.0N", "105.0W", "25"),
         ("20990801", "0600", "HU", "15.5N", "106.0W", SENTINEL_WIND),
         ("20990801", "1200", "EX", "16.0N", "107.0W", " -99"),
         ("20990801", "1800", "XY", "16.5N", "108.0W", "45")]
NEPAC = "\r\n".join(
    [ln for sid in SEALED for ln in block(sid, SENTINEL_NAME, TRACK)]
    + block(OPEN_EP, "TESTA", [("20990901", "0000", "TS", "18.0N", "110.0W", "40"),
                               ("20990901", "0600", "HU", "18.5N", "111.0W", "70")])) + "\r\n"
NEPAC_BYTES = NEPAC.encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, main_root):
    interim = tmp_path / "data" / "interim"
    interim.mkdir(parents=True)
    atlantic = interim / "hurdat2_atlantic.txt"
    atlantic.write_text(ATLANTIC_FIXTURE, encoding="utf-8")
    ep = interim / "hurdat2-nepac-1949-2025-092926.txt"
    ep.write_bytes(NEPAC_BYTES)
    monkeypatch.setattr(config, "HURDAT2_CACHE_PATH", atlantic)
    monkeypatch.setattr(config, "HURDAT2_EP_SHA256", hashlib.sha256(NEPAC_BYTES).hexdigest())
    monkeypatch.setattr(config, "HURDAT2_EP_CACHE_PATH", ep)
    monkeypatch.setattr(config, "HEP_SEALED_DIR", tmp_path / "data" / "hep_sealed")
    monkeypatch.setattr(goes_index, "make_anonymous_s3_client", lambda: None)
    monkeypatch.setattr(goes_index, "find_nearest_scan", lambda *a, **k: None)  # every sample is a missing scan

    def no_download(*a, **k):
        raise AssertionError("download_hurdat2 was called")
    monkeypatch.setattr(hurdat2, "download_hurdat2", no_download)
    import requests
    monkeypatch.setattr(requests, "get", no_download)
    protected = main_root / "data" / "interim" / "hurdat2_atlantic.txt"
    before = (sha(atlantic), sha(protected))
    assert before[1].startswith("ee35819cba2b61ab")
    yield ep
    assert (sha(atlantic), sha(protected)) == before, "an Atlantic HURDAT2 cache changed"


def load_concurrency():
    spec = importlib.util.spec_from_file_location("concurrency_under_test", ROOT / "scripts" / "concurrency.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the sealed set ---------------------------------------------------------------------------------------------

def test_sealed_ids_are_the_hep_list_plus_the_concurrency_exclusions():
    assert len(HEP) == 6 and set(CONCURRENT) == {"EP152025", "EP172025"}
    assert hurdat2.sealed_ids() == frozenset(SEALED)
    assert config.HEP_LIST == config.PROJECT_ROOT / "storms_holdout_hep.txt"


@pytest.mark.parametrize("content", [None, "# no storms\n"], ids=["missing", "empty"])
def test_the_seal_fails_closed_without_the_list(tmp_path, monkeypatch, env, content):
    lst = tmp_path / "storms_holdout_hep.txt"
    if content is not None:
        lst.write_text(content)
    monkeypatch.setattr(config, "HEP_LIST", lst)
    with pytest.raises((FileNotFoundError, ValueError)):
        hurdat2.parse_storm(OPEN_EP, env)


# --- refusal on every reading path --------------------------------------------------------------------------------

@pytest.mark.parametrize("sid", SEALED)
def test_parse_storm_refuses_each_sealed_id(env, sid):
    for spelled in (sid, f" {sid.lower()} "):
        with pytest.raises(hurdat2.SealedStormError, match=f"{sid} is sealed"):
            hurdat2.parse_storm(spelled, env)


@pytest.mark.parametrize("sid", SEALED)
def test_run_pipeline_outside_the_sealed_root_refuses_each_sealed_id(tmp_path, env, sid):
    out = tmp_path / "out"
    with pytest.raises(hurdat2.SealedStormError, match="sealed read writes only to"):
        pipeline.run_pipeline(sid, sample_interval="6h", out_root=out, hurdat2_path=env)
    assert not out.exists()


@pytest.mark.parametrize("sid", SEALED)
def test_concurrency_track_refuses_each_sealed_id(env, sid):
    with pytest.raises(hurdat2.SealedStormError):
        load_concurrency().track(sid, env)


# --- the one exception: the sealed build ------------------------------------------------------------------------

@pytest.mark.parametrize("sid", HEP)
def test_the_sealed_build_reads_each_hep_storm_into_the_sealed_root(env, sid):
    manifest = pipeline.run_pipeline(sid, sample_interval="6h", out_root=config.HEP_SEALED_DIR, hurdat2_path=env)
    df = pd.read_csv(manifest)
    assert Path(manifest).parent == config.HEP_SEALED_DIR / sid
    assert list(df["lat"]) == [15.0, 15.5] and set(df["atcf_id"]) == {sid}  # LO and HU kept, EX and XY trimmed


@pytest.mark.parametrize("sid", CONCURRENT)
def test_the_sealed_build_refuses_the_concurrency_exclusions(env, sid):
    with pytest.raises(hurdat2.SealedStormError, match="not an H-EP storm"):
        pipeline.run_pipeline(sid, out_root=config.HEP_SEALED_DIR, hurdat2_path=env)
    assert not config.HEP_SEALED_DIR.exists()


def test_the_sealed_build_refuses_the_atlantic_file():
    with pytest.raises(ValueError, match="Atlantic cache"):
        pipeline.run_pipeline(HEP[0], out_root=config.HEP_SEALED_DIR)
    assert not config.HEP_SEALED_DIR.exists()


def test_the_sealed_build_refuses_an_unpinned_file(env):
    env.write_bytes(NEPAC_BYTES + b"\r\n")
    with pytest.raises(ValueError, match="pinned"):
        pipeline.run_pipeline(HEP[0], out_root=config.HEP_SEALED_DIR, hurdat2_path=env)
    assert not config.HEP_SEALED_DIR.exists()


def test_the_sealed_read_rechecks_its_root_itself(tmp_path, env):
    with pytest.raises(hurdat2.SealedStormError, match="sealed read writes only to"):
        hurdat2._parse_sealed_hep_storm(HEP[0], env, config.DEFAULT_INCLUDE_STATUSES, tmp_path / "elsewhere")


def test_the_sealed_read_has_exactly_one_caller_and_no_flag():
    calls = []
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "_parse_sealed_hep_storm(" in line and not line.lstrip().startswith("def "):
                calls.append((path.relative_to(ROOT).as_posix(), n))
    assert [p for p, _ in calls] == ["src/hurricane_intensity/pipeline.py"], calls
    assert list(inspect.signature(hurdat2.parse_storm).parameters) == ["atcf_id", "hurdat2_path", "include_statuses"]
    run_params = set(inspect.signature(pipeline.run_pipeline).parameters)
    assert not {p for p in run_params if re.search("seal|hep|allow|unsafe|override", p)}, run_params


# --- open storms are unchanged ----------------------------------------------------------------------------------

def test_an_open_ep_storm_and_the_atlantic_file_parse_as_before(tmp_path, env):
    assert hurdat2.parse_storm(OPEN_EP, env)[1].equals(hurdat2._parse_storm(OPEN_EP, env)[1])
    name, fixes = hurdat2.parse_storm("AL012099", config.HURDAT2_CACHE_PATH)
    assert name == "TESTC" and list(fixes["wind_kt"]) == [40.0]
    manifest = pipeline.run_pipeline(OPEN_EP, sample_interval="6h", out_root=tmp_path / "out", hurdat2_path=env)
    assert list(pd.read_csv(manifest)["wind_kt"]) == [40.0, 70.0]


# --- the positions-only reader -------------------------------------------------------------------------------------

@pytest.mark.parametrize("sid", SEALED + [OPEN_EP])
def test_positions_are_time_lat_lon_only_for_every_fix(env, sid, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    df = hurdat2.read_positions(sid, env)
    assert list(df.columns) == ["time", "lat", "lon"]
    if sid != OPEN_EP:  # all four fixes: no status trim, and the -99 wind row is kept
        assert list(df["lat"]) == [15.0, 15.5, 16.0, 16.5] and list(df["lon"]) == [-105.0, -106.0, -107.0, -108.0]
        assert str(df["time"].iloc[-1]) == "2099-08-01 18:00:00+00:00"
    text = df.to_csv() + caplog.text + "".join(capsys.readouterr())
    assert caplog.records == [] and SENTINEL_NAME not in text and SENTINEL_WIND not in text
    for status in ("HU", "EX", "XY"):
        assert status not in text


@pytest.mark.parametrize("bad", ["wind", "lat"])
def test_position_errors_carry_no_name_wind_or_status(env, monkeypatch, bad):
    sid = HEP[0]
    text = env.read_bytes().decode()
    if bad == "wind":  # a malformed wind token that int() would echo
        text = text.replace(f" {SENTINEL_WIND},", " 1x3,", 1)
    else:
        text = text.replace("15.5N", "1x.5N", 1)
    env.write_bytes(text.encode())
    monkeypatch.setattr(config, "HURDAT2_EP_SHA256", sha(env))
    with pytest.raises(ValueError) as e:
        hurdat2.read_positions(sid, env)
    msg = str(e.value)
    assert msg.startswith("malformed fix line") and sid in msg and e.value.__suppress_context__
    for token in (SENTINEL_NAME, SENTINEL_WIND, "1x3", "1x.5N", "HU"):
        assert token not in msg


def test_positions_unknown_id_error_names_only_the_id(env):
    with pytest.raises(ValueError) as e:
        hurdat2.read_positions("EP992099", env)
    assert "EP992099" in str(e.value) and SENTINEL_NAME not in str(e.value)


@pytest.mark.parametrize("which", ["atlantic", "unpinned"])
def test_positions_read_only_the_pinned_ep_file(tmp_path, env, which):
    if which == "atlantic":
        path = config.HURDAT2_CACHE_PATH
    else:
        path = tmp_path / "hurdat2-nepac-copy.txt"
        path.write_bytes(NEPAC_BYTES)
    with pytest.raises(ValueError):
        hurdat2.read_positions(HEP[0], path)
