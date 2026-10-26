"""B-v2-2: the East Pacific (nepac) HURDAT2 cache.

Every test runs against a fixture nepac file served by a fake requests.get, so
nothing reaches the network, and config's Atlantic cache path is pointed at a
temp sentinel. An autouse fixture hashes both that sentinel and the real
protected Atlantic cache in the main checkout (read only) before and after
every test, so every new code path is checked against both.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from hurricane_intensity import config, goes_index, hurdat2, pipeline

ROOT = Path(__file__).resolve().parents[1]
ATLANTIC_PIN = "ee35819cba2b61ab"  # prefix of the protected cache's SHA-256 (decisions.md, 2025 test)
EP_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2025-092926.txt"

NEPAC = (
    "EP012099,              TESTA,      3,\r\n"
    "20990801, 0000,  , TS, 15.0N, 105.0W,  40, 1000,   40,   30,    0,   30,    0,    0,    0,    0,"
    "    0,    0,    0,    0, -999\r\n"
    "20990801, 0600,  , HU, 15.5N, 106.0W,  70,  985,   60,   50,   30,   50,   20,   20,   10,   20,"
    "    0,    0,    0,    0, -999\r\n"
    "20990801, 1200,  , HU, 16.0N, 107.0W, 100,  960,   80,   70,   50,   70,   30,   30,   20,   30,"
    "   15,   15,   10,   15, -999\r\n"
    "CP012099,              TESTB,      2,\r\n"
    "20990901, 0000,  , TS, 18.0N, 160.0W,  45,  998,    0,    0,    0,    0,    0,    0,    0,    0,"
    "    0,    0,    0,    0, -999\r\n"
    "20990901, 0600,  , TS, 18.5N, 161.0W,  50,  996,    0,    0,    0,    0,    0,    0,    0,    0,"
    "    0,    0,    0,    0, -999\r\n"
).encode()
NEPAC_SHA = hashlib.sha256(NEPAC).hexdigest()
ATLANTIC_FIXTURE = "AL012099,              TESTC,      1,\n20990701, 0000,  , TS, 25.0N,  80.0W,  40, 1000,\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeResponse:
    def __init__(self, content):
        self.content, self.text = content, content.decode()

    def raise_for_status(self):
        pass


@pytest.fixture
def served():
    """URL -> bytes the fake network serves; every URL fetched is recorded in served['calls']."""
    return {"calls": []}


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, served, main_root):
    interim = tmp_path / "data" / "interim"
    interim.mkdir(parents=True)
    atlantic = interim / "hurdat2_atlantic.txt"
    atlantic.write_text(ATLANTIC_FIXTURE, encoding="utf-8")
    monkeypatch.setattr(config, "HURDAT2_CACHE_PATH", atlantic)
    monkeypatch.setattr(config, "HURDAT2_EP_URL", EP_URL)
    monkeypatch.setattr(config, "HURDAT2_EP_SHA256", NEPAC_SHA)
    monkeypatch.setattr(config, "HURDAT2_EP_CACHE_PATH", interim / "hurdat2-nepac-1949-2025-092926.txt")
    served[EP_URL] = NEPAC

    def fake_get(url, timeout=None):
        served["calls"].append(url)
        if url not in served:
            raise AssertionError(f"unexpected network request: {url}")
        return FakeResponse(served[url])

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    protected = main_root / "data" / "interim" / "hurdat2_atlantic.txt"
    before = (sha(atlantic), sha(protected))
    assert before[1].startswith(ATLANTIC_PIN)
    yield interim
    assert (sha(atlantic), sha(protected)) == before, "an Atlantic HURDAT2 cache changed"


# --- the EP cache path ----------------------------------------------------------------------------

def test_real_ep_cache_path_is_separate_from_the_atlantic_cache():
    spec = importlib.util.spec_from_file_location("config_fresh", config.__file__)  # its own values, unpatched
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    ep, atl = Path(c.HURDAT2_EP_CACHE_PATH), Path(c.HURDAT2_CACHE_PATH)
    assert ep != atl and ep.name != atl.name and "nepac" in ep.name and atl.name == "hurdat2_atlantic.txt"
    assert ep.name == c.HURDAT2_EP_URL.rsplit("/", 1)[-1]  # pinned by filename
    assert ep.parent == atl.parent == c.INTERIM_DIR
    assert hurdat2._source_meta_path(ep) != hurdat2._source_meta_path(atl)
    assert c.HURDAT2_EP_SHA256 == "caece626f7038db64a803a73f5985b563693f0130f3ff16b4b0a0a93c7aa0507"


# --- fetch_pinned_hurdat2 / fetch_ep_hurdat2 ---------------------------------------------------------

def test_fetch_downloads_hashes_and_records_the_pin(env, served):
    path = hurdat2.fetch_ep_hurdat2()
    assert path == config.HURDAT2_EP_CACHE_PATH and path.read_bytes() == NEPAC
    meta = json.loads(hurdat2._source_meta_path(path).read_text(encoding="utf-8"))
    assert meta["url"] == EP_URL and meta["sha256"] == NEPAC_SHA
    assert served["calls"] == [EP_URL]
    assert sorted(p.name for p in env.iterdir()) == sorted(
        ["hurdat2_atlantic.txt", path.name, path.name + ".source.json"])


def test_fetch_reuses_a_verified_cache_without_the_network(served):
    hurdat2.fetch_ep_hurdat2()
    assert hurdat2.fetch_ep_hurdat2().read_bytes() == NEPAC
    assert served["calls"] == [EP_URL]


def test_fetch_refuses_a_download_with_the_wrong_hash_and_writes_nothing(env, served):
    served[EP_URL] = NEPAC.replace(b"TESTA", b"TESTX")
    with pytest.raises(ValueError, match="pinned .*nothing written"):
        hurdat2.fetch_ep_hurdat2()
    assert [p.name for p in env.iterdir()] == ["hurdat2_atlantic.txt"]


def test_fetch_refuses_a_cached_copy_with_the_wrong_hash_and_leaves_it(served):
    path = config.HURDAT2_EP_CACHE_PATH
    path.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="not overwritten"):
        hurdat2.fetch_ep_hurdat2()
    assert path.read_bytes() == b"tampered\n" and served["calls"] == []


def test_fetch_refuses_the_atlantic_cache_as_its_target(served):
    with pytest.raises(ValueError, match="protected Atlantic"):
        hurdat2.fetch_pinned_hurdat2(EP_URL, NEPAC_SHA, config.HURDAT2_CACHE_PATH)
    assert served["calls"] == []


def test_fetch_refuses_a_hard_link_to_the_atlantic_cache(env, served):
    link = env / "hurdat2-nepac-1949-2025-092926.txt"
    os.link(config.HURDAT2_CACHE_PATH, link)
    with pytest.raises(ValueError, match="protected Atlantic"):
        hurdat2.fetch_ep_hurdat2()
    assert served["calls"] == []


def test_fetch_refuses_a_cache_not_named_as_the_pinned_file(env, served):
    with pytest.raises(ValueError, match="pinned file's name"):
        hurdat2.fetch_pinned_hurdat2(EP_URL, NEPAC_SHA, env / "hurdat2_ep.txt")
    assert served["calls"] == []


def test_fetch_has_no_force_or_resolve_parameter():
    import inspect
    assert list(inspect.signature(hurdat2.fetch_pinned_hurdat2).parameters) == ["url", "sha256", "cache_path"]
    assert list(inspect.signature(hurdat2.fetch_ep_hurdat2).parameters) == []


# --- verify_pinned_hurdat2 (the --hurdat2-path gate) ------------------------------------------------

def test_verify_accepts_the_pinned_file():
    path = hurdat2.fetch_ep_hurdat2()
    assert hurdat2.verify_pinned_hurdat2(path) == path


@pytest.mark.parametrize("case", ["wrong hash", "unpinned name", "atlantic", "missing"])
def test_verify_refuses(env, case):
    path = config.HURDAT2_EP_CACHE_PATH
    if case == "wrong hash":
        path.write_bytes(NEPAC + b"\r\n")
        match = "pinned .*refusing"
    elif case == "unpinned name":
        path = env / "hurdat2-nepac-1949-2025-092326.txt"
        path.write_bytes(NEPAC)
        match = "not a pinned"
    elif case == "atlantic":
        path = config.HURDAT2_CACHE_PATH
        match = "Atlantic cache"
    else:
        match = "fetch it first"
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        hurdat2.verify_pinned_hurdat2(path)


# --- run_pipeline -----------------------------------------------------------------------------------

@pytest.fixture
def no_scans(monkeypatch):
    """Every sample time is a missing scan: no S3, no crop, no imagery."""
    monkeypatch.setattr(goes_index, "make_anonymous_s3_client", lambda: None)
    monkeypatch.setattr(goes_index, "find_nearest_scan", lambda *a, **k: None)
    calls = []

    def no_download(url=None, force=False, **kw):
        calls.append((url, force))
        return config.HURDAT2_CACHE_PATH
    monkeypatch.setattr(hurdat2, "download_hurdat2", no_download)
    return calls


def test_run_pipeline_reads_the_ep_file_and_never_calls_download_hurdat2(tmp_path, no_scans):
    path = hurdat2.fetch_ep_hurdat2()
    manifest = pipeline.run_pipeline("EP012099", sample_interval="6h", out_root=tmp_path / "out",
                                     hurdat2_path=path)
    df = pd.read_csv(manifest)
    assert no_scans == []
    assert list(df["wind_kt"]) == [40.0, 70.0, 100.0] and list(df["lon"]) == [-105.0, -106.0, -107.0]
    assert set(df["atcf_id"]) == {"EP012099"} and set(df["match_status"]) == {"missing_scan"}


@pytest.mark.parametrize("extra", [dict(hurdat2_url=EP_URL), dict(force_hurdat2_refresh=True)])
def test_run_pipeline_refuses_ep_path_with_url_or_force(tmp_path, no_scans, served, extra):
    path = hurdat2.fetch_ep_hurdat2()
    with pytest.raises(ValueError, match="cannot be combined"):
        pipeline.run_pipeline("EP012099", out_root=tmp_path / "out", hurdat2_path=path, **extra)
    assert no_scans == [] and served["calls"] == [EP_URL] and not (tmp_path / "out").exists()


def test_run_pipeline_refuses_a_wrong_ep_hash_before_any_output(tmp_path, no_scans):
    config.HURDAT2_EP_CACHE_PATH.write_bytes(NEPAC.replace(b"100,", b"105,"))
    with pytest.raises(ValueError, match="refusing"):
        pipeline.run_pipeline("EP012099", out_root=tmp_path / "out", hurdat2_path=config.HURDAT2_EP_CACHE_PATH)
    assert no_scans == [] and not (tmp_path / "out").exists()


# --- the Atlantic behaviour is unchanged --------------------------------------------------------------

@pytest.mark.parametrize("kw,expected", [({}, (None, False)),
                                         (dict(hurdat2_url="https://x/hurdat2-1851-2099-010199.txt",
                                               force_hurdat2_refresh=True),
                                          ("https://x/hurdat2-1851-2099-010199.txt", True))])
def test_run_pipeline_without_ep_path_uses_download_hurdat2_as_before(tmp_path, no_scans, kw, expected):
    manifest = pipeline.run_pipeline("AL012099", out_root=tmp_path / "out", **kw)
    assert no_scans == [expected]
    assert list(pd.read_csv(manifest)["atcf_id"]) == ["AL012099"]


def test_resolver_still_picks_the_atlantic_link_over_nepac(served):
    page = ('<a href="/data/hurdat/hurdat2-nepac-1949-2025-092926.txt">x</a>'
            '<a href="/data/hurdat/hurdat2-1851-2025-092326.txt">y</a>')
    served[config.NHC_DATA_PAGE_URL] = page.encode()
    assert hurdat2.resolve_hurdat2_url() == "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt"


def test_download_hurdat2_default_cache_is_still_the_atlantic_path():
    import inspect
    default = inspect.signature(hurdat2.download_hurdat2).parameters["cache_path"].default
    assert Path(default).name == "hurdat2_atlantic.txt"


# --- build_dataset.py --------------------------------------------------------------------------------

def load_build_dataset():
    spec = importlib.util.spec_from_file_location("build_dataset", ROOT / "scripts" / "build_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("extra", [["--force-hurdat2-refresh"], ["--hurdat2-url", EP_URL]])
def test_build_dataset_refuses_hurdat2_path_with_url_or_force(monkeypatch, extra):
    bd = load_build_dataset()
    monkeypatch.setattr(sys, "argv", ["build_dataset.py", "--atcf-id", "EP012099", "--hurdat2-path", "x", *extra])
    with pytest.raises(SystemExit) as e:
        bd.parse_args()
    assert e.value.code == 2


@pytest.mark.parametrize("argv,expected", [(["--hurdat2-path", "p.txt"], "p.txt"), ([], None)])
def test_build_dataset_passes_hurdat2_path_through(monkeypatch, argv, expected):
    bd = load_build_dataset()
    got = {}
    monkeypatch.setattr(bd.pipeline, "run_pipeline", lambda **kw: got.update(kw))
    monkeypatch.setattr(sys, "argv", ["build_dataset.py", "--atcf-id", "EP012099", *argv])
    bd.main()
    assert got["hurdat2_path"] == expected
    assert got["hurdat2_url"] is None and got["force_hurdat2_refresh"] is False
