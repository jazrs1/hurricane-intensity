"""Tests for analyze_f31.py (launch PR A draft) on synthetic inputs only. No real predictions, logs or models are used.

    .venv\\Scripts\\python.exe -I -B -m pytest -q tests/test_analyze_f31.py -p no:cacheprovider

Each full test builds a synthetic run tree (manifests, phase-1 records, train/eval logs, fake checkpoint bytes, H9
CSVs for the slot epochs only) under <tempdir>/hc_f31_synthetic/<pid>_<random>/ in the layout run_f31.py will write,
and deletes it afterwards. Two files are copied in read-only from this checkout: N5's report.txt (stored -text, so its
pinned SHA-256 holds) and the frozen ensemble's metrics CSV. git is faked, except that `show` of
scripts/analyze_n5.py returns the real committed blob, so the N5 pin is checked against real bytes.
"""
import hashlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_f31", REPO / "scripts" / "analyze_f31.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)  # main() is behind the __main__ guard
N5 = A.N5

SYN_ROOT = Path(tempfile.gettempdir()) / "hc_f31_synthetic"
SYN_ROOT.mkdir(exist_ok=True)
SYN = Path(tempfile.mkdtemp(prefix=f"{os.getpid()}_", dir=SYN_ROOT))
FAKE_PROT = {"best_model.pt": "a" * 64, "data/interim/hurdat2_atlantic.txt": "b" * 64}
FRAME_SHA = "f" * 64
COMMIT = "c0ffee" * 6 + "abcd"


@lru_cache(maxsize=None)
def n5_blob():
    out = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{A.N5_PATH}"], capture_output=True)
    if out.returncode != 0:
        pytest.skip("needs the repo's git to read the committed analyze_n5.py blob")
    return out.stdout


def fake_git(dirty=None, differs=None, n5_bytes=None, untracked=()):
    def git(args, check=True):
        if args[0] == "ls-files":
            return SimpleNamespace(stdout=b"", returncode=1 if args[-1] in untracked else 0)
        if args[0] == "status":
            return SimpleNamespace(stdout=b" M x\n" if args[-1] == dirty else b"", returncode=0)
        if args[0] == "diff":
            return SimpleNamespace(stdout=b"", returncode=1 if args[-1] == differs else 0)
        if args[0] == "show":
            if args[1].endswith(A.N5_PATH):
                return SimpleNamespace(stdout=n5_blob() if n5_bytes is None else n5_bytes, returncode=0)
            return SimpleNamespace(stdout=b"blob bytes\n", returncode=0)
        if args[0] == "rev-parse":
            return SimpleNamespace(stdout=b"1234abcd\n", returncode=0)
        raise AssertionError(args)
    return git


def base_frames():
    rng = np.random.default_rng(7)
    rows = []
    for sid, n in N5.H9.items():
        walk = np.clip(40 + np.cumsum(rng.normal(1.0, 4, n)), 20, 160)
        for i, w in enumerate(walk):
            rows.append({"atcf_id": sid, "storm_name": f"S{sid[2:4]}", "sample_time": f"2022-01-01T{i:05d}",
                         "wind_kt": round(float(w), 3), "is_interpolated": bool(i % 3), "status": "HU",
                         "crop_path": f"{sid}\\crops\\{sid}_{i:04d}.npz"})
    d = pd.DataFrame(rows)
    d["base_err"] = rng.normal(0, 11.5, len(d))
    return d


BASE = base_frames()
PIN = N5.h9_pin(BASE.assign(truth=BASE.wind_kt).sort_values("crop_path"))
DESIGN = A.design_constants(h9_pin=PIN, frame_list_sha={"control": None, "atlantic": FRAME_SHA}, protected=FAKE_PROT,
                            ci_grid=101)


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
    return (f"checkpoint: {ckpt}\nheld-out storms: {list(N5.H9)}\nheld-out samples: {len(d)}\n"
            f"overall MAE:      {d['abs_error'].mean():.2f} kt\n\nwrote {csv_rel}  ({len(d)} rows)\n")


def val_curve(kind, arm, s):
    """Validation MAE by epoch (unrounded, as re-validation returns it)."""
    rng = np.random.default_rng([s, 11, 0 if arm == "control" else 1])
    e_max = A.EPOCHS[arm][4]
    e = np.arange(1, e_max + 1)
    if kind == "improving":  # still falling at 4x: picks late at every budget
        v = 8.0 + 6.0 * np.exp(-e / (0.12 * e_max)) - 0.6 * e / e_max + rng.normal(0, 0.01, e_max)
    elif kind == "plateau":  # flat after about 1x: picks spread over the plateau
        v = 8.0 + 6.0 * np.exp(-e / (0.06 * e_max)) + rng.normal(0, 0.15, e_max)
    else:
        raise ValueError(kind)
    return {int(i): float(round(x, 9)) for i, x in zip(e, v)}


def preds_for(arm, s, e, gain, hi_shift=-7.0):
    """H9 predictions of epoch e: error scale falls by `gain` (a fraction) from the 1x budget to the 4x budget."""
    rng = np.random.default_rng([s, e, 0 if arm == "control" else 1])
    e1, e4 = A.EPOCHS[arm][1], A.EPOCHS[arm][4]
    frac = min(1.0, max(0.0, (e - e1) / (e4 - e1)))
    scale = (1 + rng.normal(0, 0.03)) * (1 - gain.get(arm, 0.0) * frac)
    d = BASE.copy()
    d["truth"] = d["wind_kt"]
    hi = d["truth"] > 96
    d["pred"] = d["truth"] + d["base_err"] * scale + np.where(hi, hi_shift + rng.normal(0, 0.5), 0.0)
    d["signed_error"] = d["pred"] - d["truth"]
    d["abs_error"] = d["signed_error"].abs()
    return d.drop(columns=["base_err"]).sample(frac=1, random_state=s)


def train_log(arm, s, val, saved):
    ckpt = A.pinned_paths(arm, s)["native_checkpoint"]
    lines = [f"RUN arm={arm} seed={s} list={N5.TRAIN_LISTS[arm]} val=storms_val_expanded.txt epochs={A.EPOCHS[arm][4]}",
             "START 2026-10-21T09:00:00-04:00 | 2026/10/21 09:00:00.000, 39, 2535 MHz, 60.60 W",
             "device: cuda", f"train storms: {DESIGN['ids'][arm]}", f"val storms:   {N5.VAL_IDS}",
             f"seed: {s}  checkpoint: {ckpt}  loss: l1  circular_mask: False  rotate: False"]
    if arm == "atlantic":
        lines.append(f"frame list: results/n5_atlantic/frame_list_atlantic.txt: {A.FRAMES[arm]} frames from 50 storms")
    lines.append(f"train samples: {A.FRAMES[arm]}  val samples: {N5.VAL_FRAMES}")
    for e in sorted(val):
        lines.append(f"epoch {e}: train {10 - e * 0.05:.2f} kt  val {val[e]:.2f} kt")
        if e in saved:
            lines.append("  saved")
    lines.append("END 2026-10-21T09:10:00-04:00 | 2026/10/21 09:10:00.000, 44, 2910 MHz, 71.65 W")
    return "\n".join(lines) + "\n"


def running_min(val):
    best, saved = np.inf, []
    for e in sorted(val):
        if val[e] < best:
            best = val[e]
            saved.append(e)
    return saved


def build(name, kind="plateau", gain=None, val_fn=None, reval_fn=None, man_mutate=None, after=None):
    """The full run_f31.py layout for 40 runs. after(root) may mutate files once everything is written."""
    gain = gain or {}
    root = SYN / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    for rel in (A.N5_REPORT, A.FROZEN_METRICS):
        write(root, rel, (REPO / rel).read_bytes())
    order = 0
    for s in A.SEEDS:
        for arm in (A.ARMS if s % 2 == 0 else A.ARMS[::-1]):
            order += 1
            val = val_fn(arm, s) if val_fn else val_curve(kind, arm, s)
            saved = running_min(val)  # train.py's 'saved' lines come from its own (training-time) values
            reval = reval_fn(arm, s, val) if reval_fn else val  # what the driver's re-validation recorded
            slots, ties, near = A.selections(arm, reval, saved)
            pins = A.pinned_paths(arm, s)
            write(root, pins["train_log"], train_log(arm, s, val, saved))
            write(root, pins["native_checkpoint"], f"native {arm} {s}".encode())
            kept = sorted(set(slots.values()))
            for e in kept:
                write(root, A.epoch_paths(arm, s, e)["checkpoint"], f"ep {arm} {s} {e}".encode())
            files = {pins["train_log"]: sha(root / pins["train_log"]),
                     pins["native_checkpoint"]: sha(root / pins["native_checkpoint"]),
                     **{A.epoch_paths(arm, s, e)["checkpoint"]: sha(root / A.epoch_paths(arm, s, e)["checkpoint"])
                        for e in kept}}
            rec = {"arm": arm, "seed": s, "order": order, "commit": COMMIT, "train_list": N5.TRAIN_LISTS[arm],
                   "frame_list_sha256": FRAME_SHA if arm == "atlantic" else None, "train_py_sha256": A.TRAIN_PY_SHA256,
                   "native_same_weights": True, **pins, "val_mae_revalidated": {str(e): v for e, v in reval.items()},
                   "slots": slots, "tie_rule": ties, "near_ties": near, "files_sha256": files,
                   "deleted_epoch_sha256": {str(e): hashlib.sha256(f"gone {arm} {s} {e}".encode()).hexdigest()
                                            for e in val if e not in kept},
                   "protected_before": dict(FAKE_PROT), "protected_after": dict(FAKE_PROT)}
            evals = {}
            for e in kept:
                p = A.epoch_paths(arm, s, e)
                d = preds_for(arm, s, e, gain)
                (root / p["csv"]).parent.mkdir(parents=True, exist_ok=True)
                d.to_csv(root / p["csv"], index=False, lineterminator="\n")
                write(root, p["eval_log"], eval_log(p["checkpoint"], p["csv"], d))
                evals[str(e)] = {"csv_sha256": sha(root / p["csv"]), "eval_log_sha256": sha(root / p["eval_log"])}
            man = dict(rec, evals=evals)
            if man_mutate:
                man = man_mutate(arm, s, man, root)
            write(root, f"logs/f31/runs/{arm}_seed{s}.train.json", json.dumps(rec, indent=1))
            write(root, f"logs/f31/runs/{arm}_seed{s}.json", json.dumps(man, indent=1))
    if after:
        after(root)
    return root


@lru_cache(maxsize=None)
def cached(kind, gain_items=()):
    return build(f"base_{kind}_{len(gain_items)}", kind=kind, gain=dict(gain_items))


def copy_of(name, kind="plateau", gain_items=()):
    src = cached(kind, gain_items)
    dst = SYN / name
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    return dst


def analyse(root, out="results/f31/analysis", git=None, deviations=None, design=None):
    return A.run(root / "logs/f31/runs", root / out, root=root, design=design or DESIGN, git=git or fake_git(),
                 deviations=deviations)


def rewrite_manifest(root, arm, s, fn):
    p = root / f"logs/f31/runs/{arm}_seed{s}.json"
    man = json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps(fn(man), indent=1), encoding="utf-8", newline="\n")


def teardown_module():
    shutil.rmtree(SYN, ignore_errors=True)


# ---- budgets, selection rules, flags -----------------------------------------------------------------------
def test_budget_table():
    assert A.EPOCHS == {"control": {1: 25, 2: 50, 4: 99}, "atlantic": {1: 13, 2: 26, 4: 52}}
    assert A.BATCHES == {"control": 66, "atlantic": 127}
    assert {a: A.EPOCHS[a][4] * A.BATCHES[a] for a in A.ARMS} == {"control": 6534, "atlantic": 6604}
    assert A.SEEDS == tuple(range(10, 30))


def test_native_pick_is_trains_running_minimum_at_each_budget():
    val = {1: 9.0, 2: 8.0, 3: 8.5, 4: 7.0, 5: 7.5, 6: 6.9}
    saved = running_min(val)
    assert saved == [1, 2, 4, 6]
    assert A.native_pick(val, saved, 3) == (2, None)
    assert A.native_pick(val, saved, 5) == (4, None)
    assert A.native_pick(val, saved, 6) == (6, None)


def test_tie_rule_fires_below_tie_kt_and_refuses_above():
    val = {1: 9.0, 2: 8.0, 3: 8.0 - 0.5e-4}  # re-validation puts epoch 3 lower by < TIE_KT; train.py saved epoch 2
    pick, tie = A.native_pick(val, [1, 2], 3)
    assert pick == 2 and tie["revalidated_argmin"] == 3 and tie["saved_epoch"] == 2
    with pytest.raises(A.DesignError, match="does not reproduce training"):
        A.native_pick({1: 9.0, 2: 8.0, 3: 7.9}, [1, 2], 3)


def test_matched_pick_odd_grid_and_near_ties():
    val = {1: 9, 2: 5, 3: 7, 4: 6, 5: 7 + 0.5e-4, 6: 8, 7: 8}
    assert A.matched_pick(val, 7) == (3, [5])
    assert A.matched_pick(val, 4) == (3, [])


def test_smoothed_pick_hand_computed():
    val = {1: 10.0, 2: 9.0, 3: 5.0, 4: 9.0, 5: 8.0, 6: 8.0, 7: 8.1}
    m = A.smoothed_values(val, 7)
    assert m[1] == 9.0 and m[2] == 9.0  # [1..3] -> 9; [1..4] -> median(10,9,5,9) = 9
    assert m[3] == 9.0 and m[4] == 8.0 and m[5] == 8.0  # [1..5] -> 9; [2..6] -> 8; [3..7] -> 8
    assert m[6] == pytest.approx(8.05) and m[7] == 8.0  # [4..7] -> median(9, 8, 8, 8.1); [5..7] -> 8
    pick, near = A.smoothed_pick(val, 7)
    assert pick == 4 and near == [5, 7]  # first minimum kept; equal medians recorded as near ties
    assert A.native_pick(val, running_min(val), 7)[0] == 3  # the single-epoch dip native picks


def test_flag_thresholds():
    assert not A.flagged(84, 99) and A.flagged(85, 99)
    assert not A.flagged(42, 50) and A.flagged(43, 50)
    assert not A.flagged(11, 13) and A.flagged(12, 13)
    assert not A.flagged(44, 52) and A.flagged(45, 52)


def test_val_stability_hand_computed():
    val = {e: float(e) for e in range(1, 21)}
    st = A.val_stability(val, 1, 20)
    assert st["val_at_pick"] == 1 and st["val_neighbours_min"] == 2 and st["val_neighbours_max"] == 3
    assert st["val_sd_last15"] == pytest.approx(np.std([18.0, 19.0, 20.0], ddof=1))  # epochs > 17: 18, 19, 20
    st = A.val_stability(val, 10, 20)
    assert (st["val_neighbours_min"], st["val_neighbours_max"]) == (8, 12)


# ---- statistics ----------------------------------------------------------------------------------------------
def test_signflip_n20_matches_brute_force_and_floor():
    d = np.full(20, -0.5)
    assert N5.signflip_p(d, n=20) == pytest.approx(2 / 2 ** 20)
    rng = np.random.default_rng(3)
    x = rng.normal(0.2, 1, 6)
    brute = np.mean([abs(np.dot(sg, x)) >= abs(x.sum()) - 1e-12 for sg in itertools.product((1, -1), repeat=6)])
    assert N5.signflip_p(x, n=6) == pytest.approx(brute)


def test_zero_seeds_set_the_p_floor():
    d = np.zeros(20)
    d[:6] = -2.0  # six non-zero seeds, all better: p = 2^(1-6) = 0.03125, not < 0.025
    r = A.paired_test("x", d, 0.025, -1, 0.9, "b", 101)
    assert r["m_nonzero"] == 6 and r["p_floor"] == pytest.approx(2 ** -5)
    assert r["p_signflip"] == pytest.approx(2 ** -5) and not r["significant"]
    assert "significance unreachable" in r["verdict"]
    d[6] = -2.0  # seven: 2^(1-7) = 0.0156 < 0.025
    r = A.paired_test("x", d, 0.025, -1, 0.9, "b", 101)
    assert r["significant"] and "unreachable" not in r["verdict"]


def test_zero_p_equals_p_on_nonzero_seeds():
    rng = np.random.default_rng(5)
    nz = rng.normal(-0.6, 1.0, 9)
    d = np.concatenate([nz, np.zeros(11)])
    assert N5.signflip_p(d, n=20) == pytest.approx(N5.signflip_p(nz, n=9))


def test_bias_branch():
    d, sign, br = A.bias_contrast([-5, -6], [-8, -7])
    assert sign == +1 and list(d) == [3, 1] and "both negative" in br
    d, sign, br = A.bias_contrast([3, -1], [-8, -7])  # new mean +1: a sign crossing
    assert sign == -1 and list(d) == [-5, -6] and "sign crossing" in br


# ---- full runs -------------------------------------------------------------------------------------------------
def test_planted_gain_in_atlantic_only():
    root = copy_of("gain", "improving", (("atlantic", 0.12),))
    out = analyse(root)
    p1, p2 = out["primary"]
    assert p2["verdict"].startswith("BETTER") and p2["significant"] and p2["mean_diff"] < -0.9
    assert not p1["significant"] and p1["m_nonzero"] == 20
    assert out["zeros"] == {"control": 0, "atlantic": 0}
    rep = (root / "results/f31/analysis/report.txt").read_text(encoding="utf-8")
    n5 = (REPO / A.N5_REPORT).read_text(encoding="utf-8").splitlines()
    for ln in (n5[3], n5[4], next(x for x in n5 if x.startswith("JOINT VERDICT:"))):
        assert f"    {ln}" in rep  # N5's verdict, verbatim, beside S1/S2
    assert "N5's verdict stands" in rep and "NO atlantic - control test at 1x or 2x" in rep
    # no atlantic - control test at 1x or 2x anywhere in the tests' outputs
    tests = pd.concat([pd.read_csv(root / f"results/f31/analysis/{f}") for f in
                       ("primary_tests.csv", "sensitivity.csv", "secondary.csv")])
    between = tests[tests.test.str.contains("atlantic - control")]
    assert len(between) == 3 and all(("4x" in t) for t in between.test)
    for f in A.OUTPUTS:
        assert b"\r" not in (root / "results/f31/analysis" / f).read_bytes()


def test_improving_curves_fire_budget_limited_at_4x():
    root = copy_of("limited", "improving", (("atlantic", 0.12),))
    out = analyse(root)
    assert out["qualifiers"][("control", 4)] and out["qualifiers"][("atlantic", 4)]
    assert "BUDGET-LIMITED at 4x" in out["primary"][0]["verdict"]
    assert "-> BUDGET-LIMITED" in "\n".join(out["report"])


def test_plateau_null_and_deterministic_outputs():
    root = copy_of("null", "plateau")
    out = analyse(root)
    assert all(not r["significant"] for r in out["primary"])
    assert not out["qualifiers"][("control", 4)]
    first = {f: (root / "results/f31/analysis" / f).read_bytes() for f in A.OUTPUTS}
    analyse(root, out="results/f31/analysis2")
    for f in A.OUTPUTS:
        assert (root / "results/f31/analysis2" / f).read_bytes() == first[f]


def test_refuses_overwrite():
    root = copy_of("overwrite")
    analyse(root)
    with pytest.raises(A.DesignError, match="refusing to overwrite"):
        analyse(root)


def test_slots_read_on_h9_are_exactly_the_slots():
    root = copy_of("slots")
    out = analyse(root)
    sel = out["selection"]
    assert set(sel[sel.arm == "control"].slot) == set(A.SLOTS["control"])
    assert set(sel[sel.arm == "atlantic"].slot) == set(A.SLOTS["atlantic"])
    n_csv = len(list((root / "results/f31").glob("*_h9.csv")))
    assert n_csv == sel.groupby(["arm", "seed"]).epoch.nunique().sum() <= 180
    for c in ("val_at_pick", "val_neighbours_min", "val_neighbours_max", "val_sd_last15"):
        assert sel[c].notna().all()
    assert "H9 SPLIT" in "\n".join(out["report"])


# ---- refusals ------------------------------------------------------------------------------------------------
def first_nonslot(root, arm="control", s=10):
    man = json.loads((root / f"logs/f31/runs/{arm}_seed{s}.json").read_text(encoding="utf-8"))
    return next(e for e in range(1, A.EPOCHS[arm][4] + 1) if e not in man["slots"].values()), man


def test_refuses_h9_csv_for_a_nonslot_epoch():
    root = copy_of("nonslot")
    e, _ = first_nonslot(root)
    p = A.epoch_paths("control", 10, e)
    d = preds_for("control", 10, e, {})
    d.to_csv(root / p["csv"], index=False, lineterminator="\n")
    with pytest.raises(A.DesignError, match="non-slot epoch"):
        analyse(root)


def test_refuses_slots_not_the_validation_selection():
    root = copy_of("badslots")
    rewrite_manifest(root, "control", 10, lambda m: {**m, "slots": {**m["slots"], "native_4x": 1}})
    with pytest.raises(A.DesignError, match="validation-only selections"):
        analyse(root)


def test_refuses_other_train_py():
    root = copy_of("trainpy")
    rewrite_manifest(root, "atlantic", 11, lambda m: {**m, "train_py_sha256": "0" * 64})
    with pytest.raises(A.DesignError, match="train.py blob"):
        analyse(root)


def test_refuses_kept_nonslot_file_and_leftover_epoch_files():
    root = copy_of("kept")
    e, man = first_nonslot(root)
    rel = A.epoch_paths("control", 10, e)["checkpoint"]
    write(root, rel, b"left over")
    with pytest.raises(A.DesignError, match="non-slot epoch files still on disk"):
        analyse(root)
    rewrite_manifest(root, "control", 10, lambda m: {**m, "files_sha256": {**m["files_sha256"], rel: sha(root / rel)}})
    with pytest.raises(A.DesignError, match="files_sha256 must name exactly"):
        analyse(root)


def test_refuses_deleted_hashes_incomplete():
    root = copy_of("deleted")
    def drop_one(m):
        k = sorted(m["deleted_epoch_sha256"])[0]
        return {**m, "deleted_epoch_sha256": {x: v for x, v in m["deleted_epoch_sha256"].items() if x != k}}
    rewrite_manifest(root, "atlantic", 12, drop_one)
    with pytest.raises(A.DesignError, match="deleted_epoch_sha256"):
        analyse(root)


def dip(arm, s):
    """A plateau curve with a planted global minimum at epoch 5 (7.004) and epoch 6 just above it (7.00401): train.py
    saves epoch 5 and not 6, and both log as 7.00."""
    v = val_curve("plateau", arm, s)
    if (arm, s) == ("control", 14):
        v[5], v[6] = 7.004, 7.00401
    return v


def test_refuses_revalidation_not_reproducing_training():
    def reval_fn(arm, s, v):
        return {**v, 6: 7.0025} if (arm, s) == ("control", 14) else v  # 0.0015 below train.py's pick: > TIE_KT
    with pytest.raises(A.DesignError, match="does not reproduce training"):
        build("reval", val_fn=dip, reval_fn=reval_fn)  # the builder's own selection refuses, as the driver would
    root = build("reval2", val_fn=dip)
    rewrite_manifest(root, "control", 14, lambda m: {**m, "val_mae_revalidated": {**m["val_mae_revalidated"],
                                                                                  "6": 7.0025}})
    with pytest.raises(A.DesignError, match="does not reproduce training"):
        analyse(root)


def test_tie_rule_fires_and_trains_pick_is_evaluated():
    def reval_fn(arm, s, v):
        return {**v, 6: 7.004 - 0.5e-4} if (arm, s) == ("control", 14) else v  # below train.py's pick by < TIE_KT
    root = build("tie", val_fn=dip, reval_fn=reval_fn)
    man = json.loads((root / "logs/f31/runs/control_seed14.json").read_text(encoding="utf-8"))
    assert man["slots"]["native_1x"] == man["slots"]["native_4x"] == 5  # train.py's pick, not the re-validated 6
    assert all(man["tie_rule"][b]["revalidated_argmin"] == 6 for b in ("1x", "2x", "4x"))
    out = analyse(root)
    assert "control seed 14" in "\n".join(out["report"])
    rewrite_manifest(root, "control", 14, lambda m: {**m, "tie_rule": {"1x": None, "2x": None, "4x": None}})
    with pytest.raises(A.DesignError, match="tie_rule"):
        analyse(root, out="results/f31/analysis_b")


def test_refuses_csv_swapped_against_its_eval_log():
    root = copy_of("swap")
    man = json.loads((root / "logs/f31/runs/control_seed16.json").read_text(encoding="utf-8"))
    e = sorted(set(man["slots"].values()))[0]
    p = A.epoch_paths("control", 16, e)
    other = A.epoch_paths("atlantic", 16, json.loads((root / "logs/f31/runs/atlantic_seed16.json")
                                                     .read_text(encoding="utf-8"))["slots"]["native_1x"])
    shutil.copy(root / other["csv"], root / p["csv"])
    rewrite_manifest(root, "control", 16, lambda m: {**m, "evals": {**m["evals"], str(e): {
        **m["evals"][str(e)], "csv_sha256": sha(root / p["csv"])}}})
    with pytest.raises(A.DesignError, match="not from the same evaluation|differs from the manifest|identical"):
        analyse(root)


def test_refuses_identical_vectors_across_runs():
    root = copy_of("ident")
    a = json.loads((root / "logs/f31/runs/control_seed17.json").read_text(encoding="utf-8"))
    b = json.loads((root / "logs/f31/runs/control_seed18.json").read_text(encoding="utf-8"))
    ea, eb = a["slots"]["native_1x"], b["slots"]["native_1x"]
    pa, pb = A.epoch_paths("control", 17, ea), A.epoch_paths("control", 18, eb)
    shutil.copy(root / pa["csv"], root / pb["csv"])
    d = pd.read_csv(root / pb["csv"])
    write(root, pb["eval_log"], eval_log(pb["checkpoint"], pb["csv"], d))
    rewrite_manifest(root, "control", 18, lambda m: {**m, "evals": {**m["evals"], str(eb): {
        "csv_sha256": sha(root / pb["csv"]), "eval_log_sha256": sha(root / pb["eval_log"])}}})
    with pytest.raises(A.DesignError, match="identical prediction vectors"):
        analyse(root)


@pytest.mark.parametrize("bad_id", ["AL052025", "EP112023", "AL192020"])
def test_refuses_spent_sealed_and_validation_ids(bad_id):
    root = copy_of(f"id_{bad_id}")
    man = json.loads((root / "logs/f31/runs/atlantic_seed19.json").read_text(encoding="utf-8"))
    e = man["slots"]["native_4x"]
    p = A.epoch_paths("atlantic", 19, e)
    d = pd.read_csv(root / p["csv"])
    d.loc[0, "atcf_id"] = bad_id
    d.to_csv(root / p["csv"], index=False, lineterminator="\n")
    rewrite_manifest(root, "atlantic", 19, lambda m: {**m, "evals": {**m["evals"], str(e): {
        **m["evals"][str(e)], "csv_sha256": sha(root / p["csv"])}}})
    with pytest.raises(A.DesignError, match="2025-or-later|outside H9"):
        analyse(root)


def test_refuses_altered_n5_report_and_frozen_metrics():
    root = copy_of("n5rep")
    p = root / A.N5_REPORT
    p.write_bytes(p.read_bytes().replace(b"NULL", b"BETTER", 1))
    with pytest.raises(A.DesignError, match="pinned report"):
        analyse(root)
    root = copy_of("frozen")
    p = root / A.FROZEN_METRICS
    p.write_text(p.read_text(encoding="utf-8").replace("8.570941059931508", "8.4"), encoding="utf-8")
    with pytest.raises(A.DesignError, match="not the committed"):
        analyse(root)


def test_refuses_unpinned_analyze_n5_and_dirty_self():
    root = copy_of("selfcheck")
    with pytest.raises(A.DesignError, match="not N5's pinned blob"):
        analyse(root, git=fake_git(n5_bytes=b"something else\n"))
    with pytest.raises(A.DesignError, match="uncommitted changes"):
        analyse(root, git=fake_git(dirty=A.SELF_PATH))
    with pytest.raises(A.DesignError, match="differs from the launch commit"):
        analyse(root, git=fake_git(differs=A.N5_PATH))


def test_refuses_mixed_commits_and_protected_change():
    root = copy_of("commits")
    rewrite_manifest(root, "atlantic", 20, lambda m: {**m, "commit": "f" * 40})
    with pytest.raises(A.DesignError, match="more than one commit"):
        analyse(root)
    root = copy_of("protected")
    rewrite_manifest(root, "control", 20, lambda m: {**m, "protected_after": {**m["protected_after"],
                                                                              "best_model.pt": "0" * 64}})
    with pytest.raises(A.DesignError, match="protected hashes differ"):
        analyse(root)


def test_refuses_utf16_train_log():
    root = copy_of("utf16")
    p = root / A.pinned_paths("control", 21)["train_log"]
    p.write_bytes(p.read_text(encoding="utf-8").encode("utf-16"))
    with pytest.raises(A.DesignError, match="UTF-16"):
        analyse(root)


# ---- rule (A) --------------------------------------------------------------------------------------------------
def test_deviation_drops_only_a_failing_run():
    root = copy_of("dev", "improving", (("atlantic", 0.12),))
    dev = root / A.DEVIATIONS_PATH
    write(root, A.DEVIATIONS_PATH, json.dumps({"dropped_runs": [{"arm": "control", "seed": 22, "reason": "x"}]}))
    with pytest.raises(A.DesignError, match="still verifies"):
        analyse(root, deviations=dev)
    (root / A.pinned_paths("control", 22)["native_checkpoint"]).write_bytes(b"corrupted")
    out = analyse(root, deviations=dev)
    p1, p2 = out["primary"]
    assert p1["n"] == 19 and p2["n"] == 20 and p1["verdict"].startswith("NON-CONFIRMATORY")
    s1 = out["secondary"][0]
    assert s1["n"] == 19 and s1["verdict"].startswith("NON-CONFIRMATORY")
    assert "DEVIATION (rule (A))" in "\n".join(out["report"])


def test_deviation_refuses_a_run_with_no_record():
    root = copy_of("devmissing")
    (root / "logs/f31/runs/atlantic_seed23.train.json").unlink()
    write(root, A.DEVIATIONS_PATH, json.dumps({"dropped_runs": [{"arm": "atlantic", "seed": 23, "reason": "x"}]}))
    with pytest.raises(A.DesignError, match="has no training record"):
        analyse(root, deviations=root / A.DEVIATIONS_PATH)


def test_deviation_file_must_be_committed():
    root = copy_of("devgit")
    write(root, A.DEVIATIONS_PATH, json.dumps({"dropped_runs": [{"arm": "control", "seed": 22, "reason": "x"}]}))
    with pytest.raises(A.DesignError, match="must be committed"):
        analyse(root, deviations=root / A.DEVIATIONS_PATH, git=fake_git(untracked=(A.DEVIATIONS_PATH,)))


def test_main_out_dir_must_be_under_results_f31(tmp_path):
    with pytest.raises(SystemExit):
        A.main(["--manifest-dir", str(tmp_path), "--out-dir", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        A.main(["--manifest-dir", str(tmp_path), "--out-dir", str(REPO / "results/n5_atlantic/again")])
