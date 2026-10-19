"""Tests for analyze_n5.py (draft r2) on synthetic inputs only. No real predictions, logs, models or git are used.

    .venv\\Scripts\\python.exe -I -B -m pytest -q tests/test_analyze_n5.py -p no:cacheprovider

Each test builds a synthetic run tree (manifests, train/eval logs, fake checkpoint bytes, H9 CSVs) under
<tempdir>/hc_n5_synthetic/<pid>_<random>/<test>/ in the layout run_n5.py will write, and deletes it afterwards. The
per-process folder keeps concurrent runs (two checkouts' check.py, say) out of each other's trees.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "analyze_n5", Path(__file__).resolve().parents[1] / "scripts" / "analyze_n5.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)  # main() is behind the __main__ guard

SYN_ROOT = Path(tempfile.gettempdir()) / "hc_n5_synthetic"
SYN_ROOT.mkdir(exist_ok=True)
SYN = Path(tempfile.mkdtemp(prefix=f"{os.getpid()}_", dir=SYN_ROOT))  # this process's own folder
ARM_FRAMES = 4072
FAKE_PROT = {"best_model.pt": "a" * 64, "data/interim/hurdat2_atlantic.txt": "b" * 64}
FRAME_SHA = "f" * 64
COMMIT = "c0ffee" * 6 + "abcd"


def base_frames():
    rng = np.random.default_rng(7)
    rows = []
    for sid, n in A.H9.items():
        walk = np.clip(40 + np.cumsum(rng.normal(1.0, 4, n)), 20, 160)
        for i, w in enumerate(walk):
            rows.append({"atcf_id": sid, "storm_name": f"S{sid[2:4]}", "sample_time": f"2022-01-01T{i:05d}",
                         "wind_kt": round(float(w), 3), "is_interpolated": bool(i % 3), "status": "HU",
                         "crop_path": f"{sid}\\crops\\{sid}_{i:04d}.npz"})
    d = pd.DataFrame(rows)
    d["base_err"] = rng.normal(0, 11.5, len(d))
    return d


BASE = base_frames()
PIN = A.h9_pin(BASE.assign(truth=BASE.wind_kt).sort_values("crop_path"))
DESIGN = A.design_constants(h9_pin=PIN, frames={"control": A.CONTROL_TRAIN_FRAMES, "atlantic": ARM_FRAMES},
                            frame_list_sha={"control": None, "atlantic": FRAME_SHA}, protected=FAKE_PROT)


def fake_git(dirty=False, differs=False):
    def git(args, check=True):
        if args[0] == "status":
            return SimpleNamespace(stdout=b" M scripts/analyze_n5.py\n" if dirty else b"", returncode=0)
        if args[0] == "diff":
            return SimpleNamespace(stdout=b"", returncode=1 if differs else 0)
        if args[0] == "show":
            return SimpleNamespace(stdout=b"blob bytes\n", returncode=0)
        if args[0] == "rev-parse":
            return SimpleNamespace(stdout=b"1234abcd\n", returncode=0)
        raise AssertionError(args)
    return git


def safe(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(root, rel, data):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        p.write_bytes(data)
    else:
        p.write_text(data, encoding="utf-8", newline="\n")
    return rel


def eval_log(ckpt, csv_rel, d):
    """evaluate.py's stdout lines that the analysis links to the CSV (evaluate.py:23-24, 50-51, last line)."""
    return (f"checkpoint: {ckpt}\nheld-out storms: {list(A.H9)}\nheld-out samples: {len(d)}\n"
            f"overall MAE:      {d['abs_error'].mean():.2f} kt\n\nwrote {csv_rel}  ({len(d)} rows)\n")


def make_preds(arm, s, scale, hi_shift, seed_sd, bias_sd):
    rng = np.random.default_rng([s, 0 if arm == "control" else 1])
    d = BASE.copy()
    d["truth"] = d["wind_kt"]
    hi = d["truth"] > 96
    d["pred"] = d["truth"] + d["base_err"] * scale * (1 + rng.normal(0, seed_sd)) + \
        np.where(hi, hi_shift + rng.normal(0, bias_sd), 0.0)
    d["signed_error"] = d["pred"] - d["truth"]
    d["abs_error"] = d["signed_error"].abs()
    return d.drop(columns=["base_err"]).sample(frac=1, random_state=s)


def build(name, arm_scale=1.0, arm_hi=-7.0, ctl_hi=-7.0, seed_sd=0.03, bias_sd=0.6, val_min_frac=0.55,
          csv_mutate=None, log_mutate=None, eval_mutate=None, man_mutate=None, native_differs=False):
    root = SYN / safe(name)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    for arm in A.ARMS:
        epochs, frames = DESIGN["epochs"][arm], DESIGN["frames"][arm]
        ids = DESIGN["ids"][arm]
        for s in A.SEEDS:
            rng = np.random.default_rng([s, 7, 0 if arm == "control" else 1])
            m = val_min_frac * epochs
            val = {e: round(8.0 + 0.02 * (e - m) ** 2 + rng.normal(0, 0.05), 6) for e in range(1, epochs + 1)}
            ckpt = f"models/n5_atlantic/{arm}/seed{s}.pt"
            lines = [f"RUN arm={arm} seed={s} order={2 * s + (arm == 'atlantic')} list={A.TRAIN_LISTS[arm]}",
                     "START 2026-10-19T09:00:00-04:00 | 2026/10/19 09:00:00.000, 39, 2535 MHz, 60.60 W",
                     "device: cuda", f"train storms: {ids}", f"val storms:   {A.VAL_IDS}",
                     f"seed: {s}  checkpoint: {ckpt}  loss: l1  circular_mask: False  rotate: False"]
            if arm == "atlantic":
                lines.append(f"frame list: results/n5_atlantic/frame_list_atlantic.txt: {frames} frames from 50 storms")
            lines.append(f"train samples: {frames}  val samples: {A.VAL_FRAMES}")
            best = np.inf
            for e in range(1, epochs + 1):
                lines.append(f"epoch {e}: train {10 - e * 0.2:.2f} kt  val {val[e]:.2f} kt")
                if val[e] < best:
                    best = val[e]
                    lines.append("  saved")
            lines.append("END 2026-10-19T09:02:00-04:00 | 2026/10/19 09:02:00.000, 44, 2910 MHz, 71.65 W")
            txt = "\n".join(lines) + "\n"
            if log_mutate:
                txt = log_mutate(arm, s, txt)
            train_log = write(root, f"logs/n5_atlantic/{arm}_seed{s}.log", txt)
            write(root, ckpt, f"native {arm} {s}".encode())
            man = {"arm": arm, "seed": s, "order": 2 * s + (arm == "atlantic"), "commit": COMMIT,
                   "train_list": A.TRAIN_LISTS[arm], "frame_list_sha256": FRAME_SHA if arm == "atlantic" else None,
                   "train_log": train_log, "native_checkpoint": ckpt, "protected_before": dict(FAKE_PROT),
                   "protected_after": dict(FAKE_PROT)}
            scale = arm_scale if arm == "atlantic" else 1.0
            hi = arm_hi if arm == "atlantic" else ctl_hi
            preds = make_preds(arm, s, scale, hi, seed_sd, bias_sd)
            if csv_mutate:
                preds = csv_mutate(arm, s, preds)
            if arm == "control":
                for e in range(1, epochs + 1):
                    write(root, f"models/n5_atlantic/control/seed{s}_epochs/epoch_{e:02d}.pt", f"ep {s} {e}".encode())
                sel = min(A.CONTROL_GRID, key=lambda e: (val[e], e))
                man["val_mae_revalidated"] = {str(e): val[e] for e in range(1, epochs + 1)}
                man["tie_rule"] = None
                man["off_grid_primary"] = False
                man["odd_grid_near_ties"] = A.odd_grid_near_ties(val)
                man["selected_epoch"] = sel
                man["selected_checkpoint"] = f"models/n5_atlantic/control/seed{s}_epochs/epoch_{sel:02d}.pt"
                native_epoch = min(val, key=lambda e: (val[e], e))
                if native_epoch == sel and not native_differs:  # odd native pick: same weights, same predictions
                    nat = preds
                else:  # native selection on an even epoch: a different checkpoint, slightly different predictions
                    nat = make_preds(arm, s, 1.02, ctl_hi, seed_sd, bias_sd)
                man["native_h9_csv"] = f"results/n5_atlantic/control_seed{s}_native_h9.csv"
                (root / man["native_h9_csv"]).parent.mkdir(parents=True, exist_ok=True)
                nat.to_csv(root / man["native_h9_csv"], index=False, lineterminator="\n")
                man["native_eval_log"] = write(root, f"logs/n5_atlantic/control_seed{s}_native_eval_h9.log",
                                               eval_log(ckpt, man["native_h9_csv"], nat))
            else:
                man["selected_epoch"] = max(e for e in val if val[e] == min(val.values()))
                man["selected_checkpoint"] = ckpt
            man["h9_csv"] = f"results/n5_atlantic/{arm}_seed{s}_h9.csv"
            (root / man["h9_csv"]).parent.mkdir(parents=True, exist_ok=True)
            preds.to_csv(root / man["h9_csv"], index=False, lineterminator="\n")
            ev = eval_log(man["selected_checkpoint"], man["h9_csv"], preds)
            if eval_mutate:
                ev = eval_mutate(arm, s, ev)
            man["eval_log"] = write(root, f"logs/n5_atlantic/{arm}_seed{s}_eval_h9.log", ev)
            for k in ("train_log", "native_checkpoint", "selected_checkpoint", "h9_csv", "eval_log",
                      "native_h9_csv", "native_eval_log"):
                if k in man:
                    man[f"{k}_sha256"] = sha(root / man[k])
            if man_mutate:
                man = man_mutate(arm, s, man, root)
            write(root, f"logs/n5_atlantic/runs/{arm}_seed{s}.json", json.dumps(man, indent=1))
    return root


def analyse(root, out="out", git=None, frozen_dir=None):
    return A.run(root / "logs/n5_atlantic/runs", root / out, frozen_dir=frozen_dir, root=root, design=DESIGN,
                 git=git or fake_git())


@pytest.fixture(autouse=True)
def cleanup(request):
    yield
    shutil.rmtree(SYN / safe(request.node.name), ignore_errors=True)


@pytest.fixture(autouse=True, scope="module")
def remove_process_folder():
    yield
    shutil.rmtree(SYN, ignore_errors=True)
    try:
        SYN_ROOT.rmdir()  # only when no other process still has a folder there
    except OSError:
        pass


def prim(res, name):
    return res["primary"].set_index("metric").loc[name]


# ---- outcomes ------------------------------------------------------------------------------------------------
def test_planted_mae_gain_detected(request):
    res = analyse(build(request.node.name, arm_scale=0.9))
    p = prim(res, "overall MAE")
    assert p.p_signflip == pytest.approx(2 / 1024) and p.verdict == "BETTER" and -1.05 < p.mean_diff < -0.75
    assert p["ci97.5_hi"] < 0 and p.alpha == 0.025
    assert res["joint"].startswith("overall MAE better; the other primary")
    report = (SYN / safe(request.node.name) / "out" / "report.txt").read_text(encoding="utf-8")
    assert "H3 MAE" in report and "added6 MAE" in report and "97.5% CI" in report and "Sam (AL182021)" in report


def test_planted_trade_off(request):
    res = analyse(build(request.node.name, arm_scale=0.7, arm_hi=-11.0))  # MAE better, bias ~4 kt more negative
    b = prim(res, ">96 bias")
    assert b.branch.startswith("both negative") and b.mean_diff == pytest.approx(-4.0, abs=0.6)
    assert b.verdict == "WORSE"
    assert res["joint"].startswith("TRADE-OFF: overall MAE better, >96 kt bias worse")


def test_planted_reverse_trade_off(request):
    res = analyse(build(request.node.name, arm_scale=1.3, arm_hi=-3.0))
    assert prim(res, ">96 bias").verdict == "BETTER"
    assert res["joint"].startswith("REVERSE TRADE-OFF")


def test_sign_crossing_uses_absolute_bias(request):
    res = analyse(build(request.node.name, arm_hi=+2.0, ctl_hi=-7.0))
    b = prim(res, ">96 bias")
    assert b.branch.startswith("sign crossing") and b.mean_diff < 0 and b.direction == "better"


def test_significant_but_below_sesoi(request):
    res = analyse(build(request.node.name, arm_scale=0.96, seed_sd=0.003, bias_sd=0.05))  # ~ -0.35 kt, tiny noise
    p = prim(res, "overall MAE")
    assert p.p_signflip < 0.025 and p.verdict.startswith("SIGNIFICANT BUT BELOW SESOI (better")


def test_informative_null(request):
    res = analyse(build(request.node.name, seed_sd=0.02, bias_sd=0.3))
    assert prim(res, "overall MAE").verdict.startswith("NULL (informative")
    assert prim(res, ">96 bias").verdict.startswith("NULL (informative")
    assert res["joint"] == "NO SIGNIFICANT DIFFERENCE on either primary"


def test_inconclusive_null(request):
    res = analyse(build(request.node.name, seed_sd=0.15, bias_sd=4.0))
    assert prim(res, "overall MAE").verdict.startswith("NULL (inconclusive")
    assert prim(res, ">96 bias").verdict.startswith("NULL (inconclusive")


def test_budget_limited_flag(request):
    res = analyse(build(request.node.name, val_min_frac=1.2))  # validation still falling: last epochs selected
    assert "BUDGET-LIMITED" in res["joint"]
    pr = res["per_run"]
    assert pr[pr.arm == "atlantic"].undertrain_flag.sum() >= 5


def test_sensitivity_and_frozen_reported(request):
    root = build(request.node.name, arm_scale=0.9)
    fz = root / "frozen"
    fz.mkdir()
    for s in A.SEEDS:
        make_preds("control", s, 1.0, -7.0, 0.03, 0.6).to_csv(fz / f"expanded_seed{s}_h9.csv", index=False)
    res = analyse(root, frozen_dir=fz)
    assert len(res["sensitivity"]) == 2 and res["frozen"] is not None and "p_unpaired_exact" in res["frozen"]
    nat = res["sensitivity"].set_index("metric").loc["overall MAE", "control_mean"]  # item 7 A2: NATIVE control
    assert res["frozen"].set_index("metric").loc["overall MAE", "retrained_native_control_mean"] == pytest.approx(nat)
    assert (root / "out" / "frozen_vs_control.csv").exists()


def test_per_storm_rule(request):
    root = build(request.node.name)
    analyse(root)
    ps = pd.read_csv(root / "out" / "per_storm.csv", index_col=0)
    for s in A.H9:
        n = ps.loc[f"{s} >96 n", "control"]
        assert np.isnan(ps.loc[f"{s} >96 bias", "control"]) == (n < 10)


def test_deterministic_outputs(request):
    root = build(request.node.name, arm_scale=0.95)
    analyse(root, "out1")
    analyse(root, "out2")
    for f in A.OUTPUTS:
        if (root / "out1" / f).exists():
            b1 = (root / "out1" / f).read_bytes()
            assert b1 == (root / "out2" / f).read_bytes(), f
            assert b"\r" not in b1


def test_refuses_overwrite(request):
    root = build(request.node.name)
    analyse(root)
    with pytest.raises(A.DesignError, match="overwrite"):
        analyse(root)


# ---- the adversary's attack (item 5, finding 1) and linkage ----------------------------------------------------
def test_adversary_attack_same_csvs_and_ep_training_storms_refused(request):
    ep_ids = [f"EP{i:02d}2023" for i in range(1, 32)]

    def man(arm, s, m, root):
        if arm == "atlantic":  # point the arm at the control's CSV, recomputing hashes so only the content is wrong
            m["h9_csv"] = f"results/n5_atlantic/control_seed{s}_h9.csv"
            m["h9_csv_sha256"] = sha(root / m["h9_csv"])
        return m

    def log(arm, s, t):
        return t.replace(str(A.ARM_IDS), str(A.CONTROL_IDS + ep_ids)) if arm == "atlantic" else t

    root = build(request.node.name, man_mutate=man, log_mutate=log)
    with pytest.raises(A.DesignError, match="not the design's"):  # r3: refused at the pinned path already
        analyse(root)


def test_adversary_attack_ep_training_storms_alone_refused(request):
    ep_ids = [f"EP{i:02d}2023" for i in range(1, 32)]
    log = lambda arm, s, t: t.replace(str(A.ARM_IDS), str(A.CONTROL_IDS + ep_ids)) if arm == "atlantic" else t  # noqa
    with pytest.raises(A.DesignError, match="training storm set"):
        analyse(build(request.node.name, log_mutate=log))


def test_attack_csv_contents_swapped_between_arms_refused(request):
    """Item 7 A1: each seed's control and atlantic CSV contents swapped, manifests rehashed consistently."""
    root = build(request.node.name, arm_scale=0.9)
    for s in A.SEEDS:
        c = root / f"results/n5_atlantic/control_seed{s}_h9.csv"
        a = root / f"results/n5_atlantic/atlantic_seed{s}_h9.csv"
        cb, ab = c.read_bytes(), a.read_bytes()
        c.write_bytes(ab)
        a.write_bytes(cb)
        for arm, f in (("control", c), ("atlantic", a)):
            mp = root / f"logs/n5_atlantic/runs/{arm}_seed{s}.json"
            man = json.loads(mp.read_text(encoding="utf-8"))
            man["h9_csv_sha256"] = sha(f)
            mp.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(A.DesignError, match="not from the same evaluation"):
        analyse(root)


def test_attack_atlantic_csv_points_at_control_native_refused(request):
    def man(arm, s, m, root):
        if arm == "atlantic":
            m["h9_csv"] = f"results/n5_atlantic/control_seed{s}_native_h9.csv"
            m["h9_csv_sha256"] = sha(root / m["h9_csv"])
        return m
    with pytest.raises(A.DesignError, match="not the design's"):
        analyse(build(request.node.name, man_mutate=man))


def test_attack_eval_log_at_arbitrary_path_refused(request):
    def man(arm, s, m, root):
        if (arm, s) == ("atlantic", 2):
            src = root / m["eval_log"]
            dst = root / "logs/old_run/atlantic_seed2_eval_h9.log"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            m["eval_log"] = "logs/old_run/atlantic_seed2_eval_h9.log"
            m["eval_log_sha256"] = sha(dst)
        return m
    with pytest.raises(A.DesignError, match="not the design's"):
        analyse(build(request.node.name, man_mutate=man))


def test_native_vectors_in_identity_check(request):
    def man(arm, s, m, root):  # an atlantic CSV whose content equals a control native CSV, log made consistent
        if (arm, s) == ("atlantic", 4):
            src = root / "results/n5_atlantic/control_seed4_native_h9.csv"
            (root / m["h9_csv"]).write_bytes(src.read_bytes())
            m["h9_csv_sha256"] = sha(root / m["h9_csv"])
            d = pd.read_csv(root / m["h9_csv"])
            (root / m["eval_log"]).write_text(eval_log(m["selected_checkpoint"], m["h9_csv"], d),
                                              encoding="utf-8", newline="\n")
            m["eval_log_sha256"] = sha(root / m["eval_log"])
        return m
    with pytest.raises(A.DesignError, match="identical prediction vectors"):
        analyse(build(request.node.name, man_mutate=man))


def test_identical_prediction_vectors_refused(request):
    def man(arm, s, m, root):  # control content copied into the atlantic CSV at its pinned path, log consistent
        if arm == "atlantic":
            (root / m["h9_csv"]).write_bytes((root / f"results/n5_atlantic/control_seed{s}_h9.csv").read_bytes())
            m["h9_csv_sha256"] = sha(root / m["h9_csv"])
            d = pd.read_csv(root / m["h9_csv"])
            (root / m["eval_log"]).write_text(eval_log(m["selected_checkpoint"], m["h9_csv"], d),
                                              encoding="utf-8", newline="\n")
            m["eval_log_sha256"] = sha(root / m["eval_log"])
        return m
    with pytest.raises(A.DesignError, match="identical prediction vectors"):
        analyse(build(request.node.name, man_mutate=man))


def test_checkpoint_path_in_train_log_checked(request):
    log = lambda arm, s, t: t.replace(f"checkpoint: models/n5_atlantic/{arm}/seed{s}.pt",  # noqa: E731
                                      "checkpoint: best_model.pt") if s == 4 else t
    with pytest.raises(A.DesignError, match="checkpoint path"):
        analyse(build(request.node.name, log_mutate=log))


def test_eval_log_checkpoint_checked(request):
    ev = lambda arm, s, t: t.replace("seed3.pt", "seed2.pt") if (arm, s) == ("atlantic", 3) else t  # noqa: E731
    with pytest.raises(A.DesignError, match="evaluated checkpoint"):
        analyse(build(request.node.name, eval_mutate=ev))


def test_file_changed_after_manifest_refused(request):
    root = build(request.node.name)
    p = root / "results/n5_atlantic/atlantic_seed6_h9.csv"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(A.DesignError, match="SHA-256 differs"):
        analyse(root)


def test_frame_list_sha_checked(request):
    def man(arm, s, m, root):
        if arm == "atlantic" and s == 0:
            m["frame_list_sha256"] = "e" * 64
        return m
    with pytest.raises(A.DesignError, match="frame list SHA-256"):
        analyse(build(request.node.name, man_mutate=man))


def test_mixed_commits_refused(request):
    def man(arm, s, m, root):
        if s == 7:
            m["commit"] = "deadbeef"
        return m
    with pytest.raises(A.DesignError, match="more than one commit"):
        analyse(build(request.node.name, man_mutate=man))


def test_protected_hash_change_refused(request):
    def man(arm, s, m, root):
        if (arm, s) == ("control", 1):
            m["protected_after"]["best_model.pt"] = "0" * 64
        return m
    with pytest.raises(A.DesignError, match="protected_after"):
        analyse(build(request.node.name, man_mutate=man))


def test_control_grid_selection_checked(request):
    def man(arm, s, m, root):
        if (arm, s) == ("control", 5):
            m["selected_epoch"] = 25
            m["selected_checkpoint"] = "models/n5_atlantic/control/seed5_epochs/epoch_25.pt"
            m["selected_checkpoint_sha256"] = sha(root / m["selected_checkpoint"])
        return m
    with pytest.raises(A.DesignError, match="odd-epoch grid|evaluated checkpoint"):
        analyse(build(request.node.name, man_mutate=man))


def test_grid_values_must_match_log(request):
    def man(arm, s, m, root):
        if (arm, s) == ("control", 2):
            m["val_mae_revalidated"]["13"] = m["val_mae_revalidated"]["13"] + 0.5
        return m
    with pytest.raises(A.DesignError, match="inconsistent with the train log"):
        analyse(build(request.node.name, man_mutate=man))


def test_revalidation_must_reproduce_native_pick(request):
    def man(arm, s, m, root):  # values still round to the log, but the unrounded minimum moves off train.py's pick
        if (arm, s) == ("control", 6):
            r = m["val_mae_revalidated"]
            nat = min(r, key=lambda e: (r[e], int(e)))
            other = next(e for e in r if e != nat)
            r[other] = r[nat] - 1e-3  # moves the minimum, outside the tie window; the rounding check may trip first
        return m
    with pytest.raises(A.DesignError, match="does not reproduce training|inconsistent with the train log"):
        analyse(build(request.node.name, man_mutate=man))


def test_revalidation_pick_check_alone(request):
    """Re-validated values all round to the log, but their unrounded minimum is not train.py's 'saved' pick:
    only the end-to-end pick check catches it (the 2-dp agreement check alone would pass)."""
    def man(arm, s, m, root):
        if (arm, s) == ("control", 0):
            r = m["val_mae_revalidated"]
            nat = min(r, key=lambda e: (r[e], int(e)))
            e2 = "1" if nat != "1" else "2"
            r[e2] = round(r[nat] - 0.001, 6)  # now the minimum, while 'saved' still marks nat
            p = root / m["train_log"]
            t = p.read_text(encoding="utf-8")
            t = re.sub(rf"^epoch {e2}: (train \S+ kt)\s+val \S+ kt", rf"epoch {e2}: \1  val {r[e2]:.2f} kt", t,
                       flags=re.M)
            p.write_text(t, encoding="utf-8", newline="\n")
            m["train_log_sha256"] = sha(p)
        return m
    with pytest.raises(A.DesignError, match="does not reproduce training"):
        analyse(build(request.node.name, man_mutate=man))


# ---- 2026-10-19 amendment (a): the tie rule ---------------------------------------------------------------------
def plant_tie(delta, record=True, select_native=True):
    """Control seed 0: an epoch e2 re-validates `delta` kt below train.py's pick (its log line rewritten so the
    rounding still holds). With select_native, the manifest selects and evaluates train.py's own checkpoint."""
    def man(arm, s, m, root):
        if (arm, s) != ("control", 0):
            return m
        r = m["val_mae_revalidated"]
        nat = min(r, key=lambda e: (r[e], int(e)))
        e2 = "1" if nat != "1" else "2"
        r[e2] = r[nat] - delta
        p = root / m["train_log"]
        t = re.sub(rf"^epoch {e2}: (train \S+ kt)\s+val \S+ kt", rf"epoch {e2}: \1  val {r[e2]:.2f} kt",
                   p.read_text(encoding="utf-8"), flags=re.M)
        p.write_text(t, encoding="utf-8", newline="\n")
        m["train_log_sha256"] = sha(p)
        m["odd_grid_near_ties"] = A.odd_grid_near_ties({int(k): v for k, v in r.items()})
        if record:
            m["tie_rule"] = {"revalidated_argmin": int(e2), "native_epoch": int(nat), "delta_kt": delta}
            m["off_grid_primary"] = int(nat) % 2 == 0
        if select_native == "epoch_only":  # train.py's epoch, but the grid pick's checkpoint file stays
            m["selected_epoch"] = int(nat)
        elif select_native:
            m["selected_epoch"], m["selected_checkpoint"] = int(nat), m["native_checkpoint"]
            shutil.copy(root / m["native_h9_csv"], root / m["h9_csv"])  # same checkpoint, same predictions
            write(root, m["eval_log"], eval_log(m["native_checkpoint"], m["h9_csv"], pd.read_csv(root / m["h9_csv"])))
            for k in ("selected_checkpoint", "h9_csv", "eval_log"):
                m[f"{k}_sha256"] = sha(root / m[k])
        return m
    return man


def test_tie_rule_accepts_train_py_checkpoint_inside_the_window(request):
    res = analyse(build(request.node.name, man_mutate=plant_tie(5e-5)))
    row = res["per_run"].set_index(["arm", "seed"]).loc[("control", 0)]
    assert row.selected_epoch == row.native_selected_epoch
    assert "control seeds [0]" in (SYN / safe(request.node.name) / "out" / "report.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("kw, msg", [
    (dict(delta=2e-4), "does not reproduce training"),  # outside the window: refused as before
    (dict(delta=5e-5, record=False), "tie_rule None does not match"),  # fired but not recorded
    (dict(delta=5e-5, select_native=False), "train.py's own saved checkpoint"),  # fired, but the grid pick evaluated
    (dict(delta=5e-5, select_native="epoch_only"), "train.py's own saved checkpoint"),  # right epoch, other file
], ids=["outside_window", "not_recorded", "grid_pick_evaluated", "other_checkpoint_file"])
def test_tie_rule_refusals(request, kw, msg):
    with pytest.raises(A.DesignError, match=msg):
        analyse(build(request.node.name, man_mutate=plant_tie(**kw)))


def test_tie_rule_recorded_when_it_did_not_fire_or_field_missing(request):
    def recorded(arm, s, m, root):
        if (arm, s) == ("control", 4):
            m["tie_rule"] = {"revalidated_argmin": 3, "native_epoch": 5, "delta_kt": 0.0}
        return m

    def missing(arm, s, m, root):
        if (arm, s) == ("control", 4):
            del m["tie_rule"]
        return m
    with pytest.raises(A.DesignError, match="does not match the re-validated values"):
        analyse(build(request.node.name, man_mutate=recorded))
    with pytest.raises(A.DesignError, match="no tie_rule field"):
        analyse(build(request.node.name, man_mutate=missing))


@pytest.mark.parametrize("plant", ["no_tie_marked", "tie_unmarked"])
def test_off_grid_primary_verified(request, plant):
    """John, 2026-10-19, decision 1: an even-native tie's primary is marked off-grid; the mark must be exact."""
    if plant == "no_tie_marked":
        def man(arm, s, m, root):
            if (arm, s) == ("control", 1):
                m["off_grid_primary"] = True
            return m
    else:
        tie = plant_tie(5e-5)

        def man(arm, s, m, root):
            m = tie(arm, s, m, root)
            if (arm, s) == ("control", 0):
                m["off_grid_primary"] = not m["off_grid_primary"]
            return m
    with pytest.raises(A.DesignError, match="off_grid_primary .* is wrong"):
        analyse(build(request.node.name, man_mutate=man))


def test_odd_grid_near_ties_verified_and_reported(request):
    def wrong(arm, s, m, root):
        if (arm, s) == ("control", 3):
            m["odd_grid_near_ties"] = [25]
        return m

    with pytest.raises(A.DesignError, match="odd_grid_near_ties .* does not match"):
        analyse(build(request.node.name, man_mutate=wrong))
    reval = {e: 8.2 + 0.01 * e for e in range(1, 26)}
    assert A.odd_grid_near_ties({**reval, 13: 8.00105, 15: 8.0010}) == [13]  # against the odd-grid minimum (15)
    assert A.odd_grid_near_ties({**reval, 14: 8.00105, 13: 8.0010, 15: 8.00109}) == [15]  # never the even epoch
    assert A.odd_grid_near_ties(reval) == []


def test_control_selection_values():
    reval = {e: 8.2 + 0.01 * e for e in range(1, 26)}
    assert A.control_selection({**reval, 14: 7.99}, 14) == (1, None)  # no tie: odd-grid first minimum
    assert A.control_selection({**reval, 13: 8.00105, 15: 8.0010}, 15) == (15, None)
    sel, tie = A.control_selection({**reval, 13: 8.00105, 15: 8.0010}, 13)  # train.py saved 13: it is evaluated
    assert (sel, tie["revalidated_argmin"], tie["native_epoch"]) == (13, 15, 13)
    assert tie["delta_kt"] == pytest.approx(5e-5)
    with pytest.raises(A.DesignError, match="does not reproduce training"):
        A.control_selection({**reval, 13: 8.0012, 15: 8.0010}, 13)  # |d| = 2e-4
    assert A.TIE_KT == 1e-4


def test_self_check_dirty_or_different(request):
    root = build(request.node.name)
    with pytest.raises(A.DesignError, match="uncommitted"):
        analyse(root, git=fake_git(dirty=True))
    with pytest.raises(A.DesignError, match="differs from the launch commit"):
        analyse(root, "out_b", git=fake_git(differs=True))
    res = analyse(root, "out_c")
    assert res["blob_sha"] == hashlib.sha256(b"blob bytes\n").hexdigest()


# ---- inputs ----------------------------------------------------------------------------------------------------
def _swap(sid_new):
    def m(arm, s, d):
        if (arm, s) == ("atlantic", 3):
            d = d.copy()
            d.loc[d.index[0], "atcf_id"] = sid_new
        return d
    return m


@pytest.mark.parametrize("sid,msg", [("AL082025", "2025-or-later"), ("AL012026", "2025-or-later"),
                                     ("AL132020", "outside H9")])
def test_refuses_bad_storm_ids(request, sid, msg):
    with pytest.raises(A.DesignError, match=msg):
        analyse(build(request.node.name, csv_mutate=_swap(sid)))


def test_refuses_missing_frame(request):
    with pytest.raises(A.DesignError, match="frame counts"):
        analyse(build(request.node.name, csv_mutate=lambda a, s, d: d.iloc[1:] if (a, s) == ("control", 5) else d))


def test_refuses_changed_label_via_pin(request):
    def m(arm, s, d):
        if (arm, s) == ("atlantic", 9):
            d = d.copy()
            i = d.index[0]
            d.loc[i, ["wind_kt", "truth"]] = d.loc[i, "truth"] + 5
            d.loc[i, "signed_error"] = d.loc[i, "pred"] - d.loc[i, "truth"]
            d.loc[i, "abs_error"] = abs(d.loc[i, "signed_error"])
        return d
    with pytest.raises(A.DesignError, match="pinned set"):
        analyse(build(request.node.name, csv_mutate=m))


@pytest.mark.parametrize("mut,msg", [
    (lambda a, s, t: t.replace("'AL202020', ", "") if s == 2 else t, "validation storms"),
    (lambda a, s, t: t.replace("epoch 13:", "epoch 99:") if a == "atlantic" else t, "epochs"),
    (lambda a, s, t: t.replace("'AL142018'", "'AL092022'") if a == "control" else t, "training storm set"),
    (lambda a, s, t: t.replace("rotate: False", "rotate: True") if s == 1 else t, "loss/mask/rotation"),
    (lambda a, s, t: t.replace("END ", "TRAIN_FAILED ") if s == 8 else t, "TRAIN_FAILED"),
])
def test_refuses_train_log_off_design(request, mut, msg):
    with pytest.raises(A.DesignError, match=msg):
        analyse(build(request.node.name, log_mutate=mut))


@pytest.mark.parametrize("enc,msg", [("utf-16", "UTF-16"), ("utf-8-sig", "BOM")])
def test_refuses_log_encoding(request, enc, msg):
    root = build(request.node.name)
    p = root / "logs/n5_atlantic/atlantic_seed0.log"
    p.write_bytes(p.read_text(encoding="utf-8").encode(enc))
    man = json.loads((root / "logs/n5_atlantic/runs/atlantic_seed0.json").read_text(encoding="utf-8"))
    man["train_log_sha256"] = sha(p)
    (root / "logs/n5_atlantic/runs/atlantic_seed0.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(A.DesignError, match=msg):
        analyse(root)


def test_refuses_unfinished_design():
    with pytest.raises(A.DesignError, match="not final"):
        A.design_constants(frames={"control": A.CONTROL_TRAIN_FRAMES, "atlantic": None})
    with pytest.raises(A.DesignError, match="not final"):
        A.design_constants(frame_list_sha={"control": None, "atlantic": None})


def test_cli_refuses_protected_or_foreign_out_dir():
    for out in ("results/storms/n5", "models/n5", "results/somewhere_else"):
        with pytest.raises(SystemExit):
            A.main(["--manifest-dir", "x", "--out-dir", out])


def test_budget_rule_and_signflip_values():
    assert A.epochs_for(4072) == (13, 128) and A.epochs_for(2081) == (25, 66) and A.epochs_for(1307) == (40, 41)
    assert A.epochs_for(4033)[0] == 13 and A.epochs_for(4096)[0] == 13
    assert A.signflip_p(np.full(10, -1.0)) == pytest.approx(2 / 1024)
    assert A.signflip_p(np.array([1, -1] * 5, float)) == 1.0
    with pytest.raises(A.DesignError):
        A.signflip_p(np.ones(9))


# ---- item 7b: odd native epochs -------------------------------------------------------------------------------
def test_odd_native_epoch_passes(request):
    root = build(request.node.name, arm_scale=0.9)
    res = analyse(root)
    ctl = res["per_run"][res["per_run"].arm == "control"]
    same = ctl[ctl.selected_epoch == ctl.native_selected_epoch]
    assert len(same) >= 1 and (same.native_selected_epoch % 2 == 1).all()  # the builder does model odd natives
    assert prim(res, "overall MAE").verdict == "BETTER"


def test_odd_native_epoch_with_differing_vectors_refused(request):
    with pytest.raises(A.DesignError, match="must be identical"):
        analyse(build(request.node.name, native_differs=True))
