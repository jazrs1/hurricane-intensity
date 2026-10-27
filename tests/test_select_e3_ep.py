"""scripts/select_e3_ep.py: the E3 East Pacific selection rule (e3/design.md §5, §5a, §6 VZA row) on fixtures.

Synthetic only: a made-up nepac file (pinned in-test by monkeypatching its SHA-256 and path), a made-up H-EP list
(config.HEP_LIST points at a temp file of fake IDs, so the real list is never read), a made-up Atlantic HURDAT2,
lists and manifests. The Atlantic cache's hash (the temp stand-in and the real protected file, read only) is checked
around every test, as tests/test_hurdat2_seal.py does. Nothing is downloaded.
"""
import hashlib
import importlib.util
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hurricane_intensity import config, goes_index, hurdat2

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("select_e3_ep", ROOT / "scripts" / "select_e3_ep.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

EXPANDED_TRAIN_FRAMES = S.EXPANDED_TRAIN_FRAMES  # before the env fixture patches it
FAKE_HEP = "EP902023"  # stand-in for an H-EP storm; concurrent with EP042023
EXCLUDED = [FAKE_HEP, *config.HEP_CONCURRENT_IDS]
SENTINEL = "SENTINELNAME"
PAD = ", 1000,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -999"
# Any time-like string from the fixtures: dates, HHMM fields, ISO times.
TIME_PATTERNS = [r"20[0-9]{2}-[01][0-9]-[0-3][0-9]", r"20(23|25|22|99)0[78][0-3][0-9]", r"T[0-2][0-9]:?[0-5][0-9]",
                 r"[0-2][0-9]:[0-5][0-9]", r"\b(0330|2130|0000|0600|1200|1800)\b"]


def block(atcf_id, name, rows):
    out = [f"{atcf_id}, {name:>18},     {len(rows)},"]
    out += [f"{d}, {t},  , {s}, {lat}, {lon}, {w:>4}{PAD}" for d, t, s, lat, lon, w in rows]
    return out


def track(day0, ndays, lat, lon0, lon1, wind, status="TS", month="08", year="2023"):
    """Synoptic fixes over ndays from day0, lon linear lon0 -> lon1 (deg W, positive numbers)."""
    hours = [(d, h) for d in range(day0, day0 + ndays) for h in (0, 6, 12, 18)]
    lons = np.linspace(lon0, lon1, len(hours))
    return [(f"{year}{month}{d:02d}", f"{h:02d}00", status, f"{lat:.1f}N", f"{lo:.1f}W", str(wind))
            for (d, h), lo in zip(hours, lons)]


EP01 = track(1, 3, 15.0, 110.0, 110.0, 60)
EP01.insert(1, ("20230801", "0330", "HU", "15.0N", "110.5W", "140"))  # off-synoptic, and the would-be peak
EP02 = track(1, 3, 12.0, 100.0, 104.0, 30)  # peak below the Atlantic floor
EP03 = track(1, 3, 15.0, 100.0, 80.0, 60)  # runs past the VZA cap (GOES-18 at 137.2 W)
EP04 = track(1, 3, 15.0, 120.0, 120.0, 70)  # concurrent with FAKE_HEP on 08-02 00..18Z
HEP = ([("20230801", "1800", "XX", "15.0N", "NaNW", "999")]  # NaN lon before the overlap (F-44)
       + track(2, 1, 15.0, 122.0, 122.0, 999, status="XX")
       + [("20230802", "2130", "XX", "15.0N", "121.0W", "999")])  # off-synoptic end: kept (F-43 option 2)
FAR = track(1, 2, 15.0, 130.0, 131.0, 999, status="XX", year="2025")
NEPAC = "\r\n".join(
    block("EP012023", "TESTA", EP01) + block("EP022023", "TESTB", EP02) + block("EP032023", "TESTC", EP03)
    + block("EP042023", "TESTD", EP04) + block(FAKE_HEP, SENTINEL, HEP)
    + block("EP052022", "TESTE", track(1, 3, 15.0, 110.0, 110.0, 80, year="2022"))
    + block(config.HEP_CONCURRENT_IDS[0], SENTINEL, FAR) + block(config.HEP_CONCURRENT_IDS[1], SENTINEL, FAR)) + "\r\n"
NEPAC_BYTES = NEPAC.encode()

# Atlantic: AL012099 control train (1h build), AL022099 added train (3h build), AL032099 validation.
ATLANTIC = "\n".join(
    block("AL012099", "TESTF", track(1, 2, 25.0, 60.0, 70.0, 50, month="07", year="2099"))
    + block("AL022099", "TESTG", track(5, 2, 25.0, 60.0, 65.0, 45, month="07", year="2099"))
    + block("AL032099", "TESTH", track(9, 2, 40.0, 50.0, 50.0, 90, month="07", year="2099"))) + "\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(path, sid, start, freq, n):
    t = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    crops = [f"{sid}\\crops\\{sid}_{x:%Y%m%dT%H%MZ}.npz" for x in t]
    path.mkdir(parents=True)
    pd.DataFrame({"atcf_id": sid, "sample_time": [x.isoformat() for x in t], "crop_path": crops}).to_csv(
        path / "manifest.csv", index=False)
    return crops


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, main_root):
    interim = tmp_path / "data" / "interim"
    interim.mkdir(parents=True)
    atlantic = interim / "hurdat2_atlantic.txt"
    atlantic.write_text(ATLANTIC, encoding="utf-8")
    ep = interim / "hurdat2-nepac-1949-2025-092926.txt"
    ep.write_bytes(NEPAC_BYTES)
    hep_list = tmp_path / "hep_list.txt"
    hep_list.write_text(FAKE_HEP + "\n")
    monkeypatch.setattr(config, "HURDAT2_CACHE_PATH", atlantic)
    monkeypatch.setattr(config, "HURDAT2_EP_SHA256", hashlib.sha256(NEPAC_BYTES).hexdigest())
    monkeypatch.setattr(config, "HURDAT2_EP_CACHE_PATH", ep)
    monkeypatch.setattr(config, "HEP_LIST", hep_list)
    monkeypatch.setattr(config, "HEP_SEALED_DIR", tmp_path / "data" / "hep_sealed")
    monkeypatch.setattr(S, "EXPANDED_TRAIN_FRAMES", 5 + 7)  # the fixture's train frames (real: 2081)
    monkeypatch.setattr(goes_index, "make_anonymous_s3_client", lambda: None)

    def no_download(*a, **k):
        raise AssertionError("a download was attempted")
    monkeypatch.setattr(hurdat2, "download_hurdat2", no_download)
    import requests
    monkeypatch.setattr(requests, "get", no_download)

    lists = tmp_path / "lists"
    lists.mkdir()
    (lists / "train.txt").write_text("# train\nAL012099\nAL022099\n# validation\nAL032099\n")
    (lists / "val.txt").write_text("AL032099\n")
    (lists / "control.txt").write_text("AL012099\nAL032099\n")
    processed = tmp_path / "processed"
    manifest(processed / "AL012099", "AL012099", "2099-07-01", "1h", 5)  # the minimum: 5 hourly frames, 4 h
    c2 = manifest(processed / "AL022099", "AL022099", "2099-07-05", "3h", 8)
    manifest(processed / "AL032099", "AL032099", "2099-07-09", "1h", 5)
    (lists / "excluded.txt").write_text(c2[0] + "\n")
    out = tmp_path / "out"
    argv = ["--nepac", str(ep), "--atlantic-hurdat2", str(atlantic), "--train-list", str(lists / "train.txt"),
            "--val-list", str(lists / "val.txt"), "--control-list", str(lists / "control.txt"),
            "--processed-root", str(processed), "--excluded-crops", str(lists / "excluded.txt"),
            "--satellite", "noaa-goes18", "--out", str(out / "storms_train_e3.txt"),
            "--record", str(out / "record.json")]
    protected = main_root / "data" / "interim" / "hurdat2_atlantic.txt"
    before = (sha(atlantic), sha(protected))
    assert before[1].startswith("ee35819cba2b61ab")
    yield {"argv": argv, "out": out, "nepac": ep}
    assert (sha(atlantic), sha(protected)) == before, "an Atlantic HURDAT2 cache changed"


def run(env):
    assert S.main(env["argv"]) == 0
    return ((env["out"] / "storms_train_e3.txt").read_text(encoding="utf-8"),
            json.loads((env["out"] / "record.json").read_text(encoding="utf-8")))


def frames(times, lat, lon):
    return pd.DataFrame({"time": pd.DatetimeIndex(times, tz="UTC"), "lat": lat, "lon": lon})


def no_time(text):
    return not any(re.search(p, text) for p in TIME_PATTERNS)


# --- F-43: synoptic fixes only, no time anywhere ---------------------------------------------------------------

def test_off_synoptic_fixes_are_dropped():
    f = frames(["2023-08-01 00:00", "2023-08-01 03:30", "2023-08-01 06:00", "2023-08-01 06:30",
                "2023-08-01 09:00", "2023-08-01 12:00:30", "2023-08-01 18:00"], 15.0, -110.0)
    kept = S.synoptic_only(f)
    assert [t.strftime("%H:%M:%S") for t in kept["time"]] == ["00:00:00", "06:00:00", "18:00:00"]


def test_candidate_reads_drop_off_synoptic_fixes_and_excluded_reads_keep_them(env):
    fixes = S.read_track("EP012023", env["nepac"], 1)
    assert len(fixes) == 12 and fixes["wind_kt"].max() == 60.0  # the 03:30 fix (140 kt) is gone
    tr = S.excluded_tracks(env["nepac"])
    assert all(list(t.columns) == ["time", "lat", "lon"] for t in tr)
    assert sum((pd.DatetimeIndex(t["time"]).minute != 0).sum() for t in tr) == 1  # the 21:30 fix is kept
    assert sum(len(t) for t in tr) == 5 + 8 + 8  # FAKE_HEP: 4 synoptic + 21:30 (NaN fix dropped); FAR x2


def test_a_frame_overlapping_only_an_off_synoptic_end_stretch_is_dropped(env):
    tr = S.excluded_tracks(env["nepac"])
    cand = frames(["2023-08-02 21:00", "2023-08-02 22:00"], 15.0, -120.0)  # after 18Z; 21Z is before 21:30
    assert list(S.gate_mask(cand, tr)) == [True, False]
    synoptic = [S.synoptic_only(t) for t in tr]  # what the shortened track would have missed
    assert not S.gate_mask(cand, synoptic).any()


def test_no_candidate_or_excluded_storm_time_or_field_in_outputs_or_logs(env, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    out, rec = run(env)
    printed = capsys.readouterr()
    rec_text = (env["out"] / "record.json").read_text(encoding="utf-8")
    for text in (out, rec_text, printed.out, printed.err, caplog.text):
        assert no_time(text), text
        for sid in EXCLUDED:
            assert sid not in text
        assert SENTINEL not in text and "999" not in text
    assert not logging.getLogger(hurdat2.__name__).disabled  # the reader's logger is restored


def test_errors_carry_id_and_line_only(env, monkeypatch, tmp_path):
    bad = NEPAC.replace("20230801, 0600,", "2023O801, 0600,", 1)  # first EP012023 synoptic fix after 0000
    assert bad != NEPAC
    p = tmp_path / "bad" / env["nepac"].name
    p.parent.mkdir()
    p.write_bytes(bad.encode())
    with pytest.raises(ValueError) as e:
        S.read_track("EP012023", p, 1)
    assert str(e.value) == "EP012023: malformed storm block (header line 1)"


def test_excluded_errors_name_no_storm(env, monkeypatch, tmp_path):
    bad = NEPAC.replace("20230802, 0600,  , XX", "2023080X, 0600,  , XX", 1)
    p = tmp_path / "bad" / env["nepac"].name
    p.parent.mkdir()
    p.write_bytes(bad.encode())
    monkeypatch.setattr(config, "HURDAT2_EP_SHA256", sha(p))
    with pytest.raises(ValueError) as e:
        S.excluded_tracks(p)
    msg = str(e.value)
    assert re.fullmatch(r"excluded storm \d: malformed fix line \d+ for <excluded>", msg), msg
    assert all(sid not in msg for sid in EXCLUDED) and no_time(msg)


# --- F-44: a NaN longitude never poisons the unwrap --------------------------------------------------------------

def test_nan_fixes_are_dropped():
    f = frames(["2023-08-01 00:00", "2023-08-01 06:00", "2023-08-01 12:00"], [15.0, np.nan, 15.0],
               [-110.0, -111.0, np.nan])
    assert len(S.drop_nan_fixes(f)) == 1


def test_a_nan_longitude_before_an_overlap_does_not_hide_it():
    other = frames(["2023-08-01 18:00", "2023-08-02 00:00", "2023-08-02 06:00"], 15.0, [np.nan, -122.0, -122.0])
    cand = frames(["2023-08-02 03:00"], 15.0, -120.0)
    assert not S.C.concurrent(cand, other).any()  # unguarded: the NaN step makes every later offset NaN
    assert S.gate_mask(cand, [S.clean_fixes(other)]).all()


# --- the concurrency hard gate (§5) -------------------------------------------------------------------------------

def test_box_test_drops_overlap_keeps_651_km_and_other_times():
    other = frames(["2023-08-02 00:00", "2023-08-02 18:00"], 15.0, -120.0)
    cand = frames(["2023-08-02 06:00", "2023-08-02 06:00", "2023-08-02 06:00", "2023-08-03 06:00"],
                  [15.0, 15.0 + 649 / S.C.KM_PER_DEG, 15.0 + 651 / S.C.KM_PER_DEG, 15.0], -120.0)
    assert list(S.gate_mask(cand, [other])) == [True, True, False, False]


def test_concurrent_frames_are_dropped_before_counting(env):
    _, rec = run(env)
    rows = {r["id"]: r for r in rec["candidates"]}
    assert rows["EP042023"]["frames_dropped_concurrent"] == 8  # 08-02 00..21Z at 3h (21:30 fix kept); NaN fix dropped
    assert rows["EP042023"]["frames_kept"] == rows["EP042023"]["frames_grid"] - 8
    assert rows["EP012023"]["frames_dropped_concurrent"] == 0


def test_the_hard_gate_refuses_to_write_on_an_injected_overlap(env, monkeypatch):
    monkeypatch.setattr(S, "gate_mask", lambda f, tracks: np.zeros(len(f), bool))  # first pass keeps everything
    with pytest.raises(SystemExit, match="hard gate"):
        S.main(env["argv"])
    assert not (env["out"] / "storms_train_e3.txt").exists() and not (env["out"] / "record.json").exists()


def test_the_gate_fails_closed_without_an_excluded_storm(env, monkeypatch, tmp_path):
    lst = tmp_path / "hep2.txt"
    lst.write_text(FAKE_HEP + "\nEP992023\n")
    monkeypatch.setattr(config, "HEP_LIST", lst)
    with pytest.raises(ValueError, match="excluded storm"):
        S.excluded_tracks(env["nepac"])


# --- selection (§5, §5a, §6) --------------------------------------------------------------------------------------

def test_selection_rule(env):
    out, rec = run(env)
    ri = rec["rule_inputs"]
    assert ri["interval"] == "3h" and ri["min_frames"] == 5 and ri["min_peak_kt"] == 45.0
    assert (ri["min_frames_storm"], ri["min_frames_storm_interval"], ri["ep_min_frames"]) == ("AL012099", "1h", 2)
    assert rec["ep_frames_note"] == "ep_frames = grid frames, pre-build"
    assert ri["atlantic_intervals"] == {"AL012099": "1h", "AL022099": "3h"}
    assert ri["vza_cap_deg"] == pytest.approx(float(np.max(S.vza_deg(40.0, -50.0, -75.2))))
    ids = [r["id"] for r in rec["candidates"]]
    assert sorted(ids) == ["EP012023", "EP022023", "EP032023", "EP042023"]  # sealed and 2022 never candidates
    rows = {r["id"]: r for r in rec["candidates"]}
    assert rows["EP022023"]["eligible"] is False and rows["EP022023"]["reason"] == "peak below floor"
    assert rows["EP012023"]["peak_kt"] == 60.0 and rows["EP012023"]["frames_grid"] == 23
    assert rows["EP032023"]["frames_dropped_vza"] > 0
    for r in rows.values():
        assert r["frames_grid"] == (r["frames_dropped_vza"] + r["frames_dropped_pre_operational"]
                                    + r["frames_dropped_concurrent"] + r["frames_kept"])
    eligible = [i for i in ids if rows[i]["eligible"]]
    assert rec["hash_order"] == sorted(eligible, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    assert rec["selected"] == rec["hash_order"]  # 1963 not reached: all are used
    assert rec["ep_frames"] == sum(rows[i]["frames_kept"] for i in rec["selected"])
    lo, hi = rec["vza_tertile_edges_deg"]
    assert 0 < lo <= hi < ri["vza_cap_deg"]
    body = [ln for ln in out.splitlines() if not ln.startswith("#")]
    assert body == ["AL012099", "AL022099", "AL032099", *rec["selected"]] and out.endswith("\n") and "\r" not in out


def test_the_frame_floor_is_duration_matched():
    assert S.ep_frame_floor(25, "1h", "3h") == 9  # AL012024: 24 h at 1h -> 9 frames at 3h
    assert S.ep_frame_floor(26, "1h", "3h") == 9 and S.ep_frame_floor(28, "1h", "3h") == 10
    assert S.ep_frame_floor(7, "3h", "3h") == 7 and S.ep_frame_floor(1, "1h", "3h") == 1


def test_the_real_manifests_give_a_floor_of_9_and_2081_frames(main_root):
    """Read-only, on the main checkout's built manifests: the inputs the real run will use."""
    from hurricane_intensity.splits import load_storm_list
    val = load_storm_list(ROOT / "storms_val_expanded.txt")
    train = [s for s in load_storm_list(ROOT / "storms_train_expanded.txt") if s not in val]
    excluded = set((ROOT / "excluded_crops.txt").read_text().split())
    m = {s: pd.read_csv(main_root / "data" / "processed" / s / "manifest.csv") for s in train}
    frames = {s: int((d["crop_path"].notna() & ~d["crop_path"].isin(excluded)).sum()) for s, d in m.items()}
    assert sum(frames.values()) == EXPANDED_TRAIN_FRAMES == 2081  # the unpatched constant
    n_min = min(frames.values())
    (storm,) = [s for s in train if frames[s] == n_min]
    assert (storm, n_min, S.grid_interval(m[storm])) == ("AL012024", 25, "1h")
    assert S.ep_frame_floor(n_min, S.grid_interval(m[storm]), "3h") == 9


def test_a_train_frame_total_other_than_2081_is_refused(env, monkeypatch):
    monkeypatch.setattr(S, "EXPANDED_TRAIN_FRAMES", 5 + 7 + 1)
    with pytest.raises(ValueError, match="sum to 12, not 13"):
        S.main(env["argv"])
    assert not env["out"].exists()


def test_the_cap_adds_whole_storms_in_hash_order(env, monkeypatch):
    monkeypatch.setattr(S, "EP_TARGET_FRAMES", 1)
    _, rec = run(env)
    assert rec["selected"] == rec["hash_order"][:1] and len(rec["hash_order"]) > 1


def test_vza_formula():
    assert S.vza_deg(0.0, -137.2, -137.2) == pytest.approx(0.0, abs=1e-9)
    v = S.vza_deg(15.0, [-137.2, -120.0, -100.0, -80.0], -137.2)
    assert np.all(np.diff(v) > 0) and v[-1] < 90


def test_every_argument_is_required_with_no_default():
    actions = [a for a in S.build_parser()._actions if a.option_strings != ["-h", "--help"]]
    assert len(actions) == 10 and all(a.required and a.default is None for a in actions)
    with pytest.raises(SystemExit):
        S.build_parser().parse_args([])


def test_existing_output_with_other_bytes_is_not_overwritten(env):
    env["out"].mkdir()
    (env["out"] / "storms_train_e3.txt").write_text("other\n")
    with pytest.raises(SystemExit, match="other bytes"):
        S.main(env["argv"])
    assert (env["out"] / "storms_train_e3.txt").read_text() == "other\n"


def test_outputs_may_not_be_inputs_or_protected(env):
    argv = list(env["argv"])
    argv[argv.index("--out") + 1] = str(config.HURDAT2_CACHE_PATH)
    with pytest.raises(SystemExit, match="refusing"):
        S.main(argv)
