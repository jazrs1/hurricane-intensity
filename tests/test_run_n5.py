"""Tests for scripts/run_n5.py on a synthetic repo. Nothing real is trained, evaluated, downloaded or built.

train.py and evaluate.py are never executed: every test hands launch() a fake runner that writes train.py's and
evaluate.py's stdout lines and output files, and an autouse fixture makes any real subprocess call fail. The
synthetic repo has a copy of scripts/analyze_n5.py with the two post-build constants (and the H9 pin and the
protected hashes, which point at synthetic files) filled in, so the real guards and the analysis's own linkage
checks run on the driver's output.
"""
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

CHECKOUT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load("run_n5", CHECKOUT / "scripts" / "run_n5.py")
A_REAL = load("analyze_n5_real", CHECKOUT / "scripts" / "analyze_n5.py")
CHECK = load("check_for_run_n5", CHECKOUT / "scripts" / "check.py")
ARM_FRAMES = 4072
COMMIT = "c0ffee" * 6 + "abcd"


@pytest.fixture(autouse=True)
def no_real_subprocess(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError(f"a real subprocess was started: {a[:1]}")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)


# ---- the synthetic repo -------------------------------------------------------------------------------------
def base_frames():
    rng = np.random.default_rng(7)
    rows = []
    for sid, n in A_REAL.H9.items():
        walk = np.clip(40 + np.cumsum(rng.normal(1.0, 4, n)), 20, 160)
        for i, w in enumerate(walk):
            rows.append({"atcf_id": sid, "storm_name": f"S{sid[2:4]}", "sample_time": f"2022-01-01T{i:05d}",
                         "wind_kt": round(float(w), 3), "is_interpolated": bool(i % 3), "status": "HU",
                         "crop_path": f"{sid}\\crops\\{sid}_{i:04d}.npz"})
    return pd.DataFrame(rows)


BASE = base_frames()
PIN = A_REAL.h9_pin(BASE.assign(truth=BASE.wind_kt).sort_values("crop_path"))


def val_curve(arm, seed, shift=None):
    """Unrounded validation MAE per epoch; odd seeds of the control pick an odd native epoch, even seeds an even one."""
    epochs = 25 if arm == "control" else 13
    rng = np.random.default_rng([seed, 7, 0 if arm == "control" else 1])
    m = (13 if seed % 2 else 14) if arm == "control" else 8
    return {e: round(8.0 + 0.02 * (e - m) ** 2 + rng.normal(0, 0.003), 6) for e in range(1, epochs + 1)}


def fake_repo(tmp_path, filled=True, frame_list_ok=True, protected_drift=False):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / ".git").mkdir()  # the main checkout has a .git directory; a worktree has a .git file
    prot = {}
    for i, rel in enumerate(CHECK.PROTECTED_PATHS):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"protected {i}".encode())
        prot[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    block = "\n".join(f"{h}  {p}" for p, h in prot.items())
    (root / "COORDINATOR.md").write_text(f"{CHECK.HASH_HEADING}\n\n```sha256\n{block}\n```\n", encoding="utf-8")
    fl = root / R.FRAME_LIST
    fl.parent.mkdir(parents=True)
    fl.write_text("frame list\n", encoding="utf-8")
    src = (CHECKOUT / "scripts" / "analyze_n5.py").read_text(encoding="utf-8")
    if filled:
        sha = hashlib.sha256(fl.read_bytes()).hexdigest() if frame_list_ok else "0" * 64
        src = src.replace("ARM_TRAIN_FRAMES = None", f"ARM_TRAIN_FRAMES = {ARM_FRAMES}", 1)
        src = src.replace("FRAME_LIST_SHA256 = None", f'FRAME_LIST_SHA256 = "{sha}"', 1)
    src = src.replace(A_REAL.H9_PIN, PIN, 1)
    for rel, h in A_REAL.PROTECTED.items():
        new = "e" * 64 if protected_drift and rel == "best_model.pt" else prot[rel]
        src = re.sub(rf'("{re.escape(rel)}": )"{h}"', rf'\g<1>"{new}"', src, count=1)
    (root / "scripts" / "analyze_n5.py").write_text(src, encoding="utf-8")
    return root


def fake_git(dirty=False, differs=False, fetch_rc=0, ancestor_rc=0):
    def git(args, check=True):
        if args[0] == "fetch":
            assert args == ["fetch", "origin"]
            return SimpleNamespace(stdout=b"", returncode=fetch_rc)
        if args[0] == "merge-base":
            assert args == ["merge-base", "--is-ancestor", "HEAD", "origin/phase2-experiments"]
            return SimpleNamespace(stdout=b"", returncode=ancestor_rc)
        if args[0] == "status":
            return SimpleNamespace(stdout=b" M scripts/train.py\n" if dirty else b"", returncode=0)
        if args[0] == "diff":
            return SimpleNamespace(stdout=b"", returncode=1 if differs else 0)
        if args[0] == "show":
            return SimpleNamespace(stdout=b"blob bytes\n", returncode=0)
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=COMMIT.encode() + b"\n", returncode=0)
        if args[0] == "rev-parse":
            return SimpleNamespace(stdout=b"1234abcd\n", returncode=0)
        raise AssertionError(args)
    return git


class FakeRunner:
    """Stands in for subprocess.run on train.py and evaluate.py: writes their stdout lines and output files."""

    def __init__(self, root, train_rc=0, on_train=None, fail=None):
        self.root, self.train_rc, self.on_train, self.fail, self.calls = root, train_rc, on_train, fail, []

    def __call__(self, cmd, stdout, stderr, cwd, env):
        assert Path(cwd) == self.root and env["PYTHONUTF8"] == "1" and stderr is subprocess.STDOUT
        self.calls.append(cmd)
        args = dict(zip(cmd[3::2], cmd[4::2])) if cmd[2] == "scripts/train.py" else dict(zip(cmd[2::2], cmd[3::2]))
        rc = self.train_rc if cmd[2] == "scripts/train.py" else 0
        if cmd[2] == "scripts/train.py" and self.fail == (self.arm_of(args), int(args["--seed"])):
            rc = 1
        out = (self.train(args, rc) if cmd[2] == "scripts/train.py" else self.evaluate(args)) if cmd[1] != "x" else ""
        stdout.write(out.encode("utf-8"))
        return SimpleNamespace(returncode=rc)

    @staticmethod
    def arm_of(a):
        return "atlantic" if a["--train-list"] == "storms_train_n5_atlantic.txt" else "control"

    def train(self, a, rc):
        assert a["--val-list"] == "storms_val_expanded.txt" and a["--loss"] == "l1"
        arm = self.arm_of(a)
        seed, epochs = int(a["--seed"]), int(a["--epochs"])
        ids = A_REAL.ARM_IDS if arm == "atlantic" else A_REAL.CONTROL_IDS
        frames = ARM_FRAMES if arm == "atlantic" else A_REAL.CONTROL_TRAIN_FRAMES
        lines = ["device: cpu", f"train storms: {ids}", f"val storms:   {A_REAL.VAL_IDS}",
                 f"seed: {seed}  checkpoint: {a['--out']}  loss: l1  circular_mask: False  rotate: False"]
        if arm == "atlantic":
            lines.append(f"frame list: {a['--frame-list']}: {frames} frames from 50 storms")
        lines.append(f"train samples: {frames}  val samples: {A_REAL.VAL_FRAMES}")
        if self.on_train:
            self.on_train(arm, seed)
        if rc:
            return "\n".join(lines + ["Traceback (most recent call last):"]) + "\n"
        val, best = val_curve(arm, seed), np.inf
        for e in range(1, epochs + 1):
            weights = f"weights {arm} {seed} {e}".encode()
            lines.append(f"epoch {e}: train {10 - e * 0.2:.2f} kt  val {val[e]:.2f} kt")
            if "--epoch-dir" in a:
                (self.root / a["--epoch-dir"]).mkdir(parents=True, exist_ok=True)
                (self.root / a["--epoch-dir"] / f"epoch_{e:02d}.pt").write_bytes(weights)
            if val[e] < best:
                best = val[e]
                (self.root / a["--out"]).write_bytes(weights)
                lines.append("  saved")
        return "\n".join(lines) + "\n"

    def evaluate(self, a):
        assert a["--holdout-list"] == "storms_holdout_h9.txt"
        weights = (self.root / a["--checkpoint"]).read_bytes()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(weights).digest()[:8], "little"))
        d = BASE.copy()
        d["truth"] = d["wind_kt"]
        d["pred"] = d["truth"] + rng.normal(0, 11.5, len(d)) - np.where(d["truth"] > 96, 7.0, 0.0)
        d["signed_error"] = d["pred"] - d["truth"]
        d["abs_error"] = d["signed_error"].abs()
        (self.root / a["--out"]).parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(self.root / a["--out"], index=False, lineterminator="\n")
        return (f"checkpoint: {a['--checkpoint']}\nheld-out storms: {list(A_REAL.H9)}\nheld-out samples: {len(d)}\n"
                f"overall MAE:      {d['abs_error'].mean():.2f} kt\n\nwrote {a['--out']}  ({len(d)} rows)\n")


def reval_from_curve(epoch_dir, epochs, train_list):
    assert train_list == "storms_train_expanded.txt" and epochs == 25
    seed = int(re.search(r"seed(\d+)_epochs", Path(epoch_dir).as_posix()).group(1))
    assert all((Path(epoch_dir) / f"epoch_{e:02d}.pt").exists() for e in range(1, 26))
    return val_curve("control", seed)


def go(root, git=None, runner=None, reval=reval_from_curve, main_root=None, cuda=True):
    runner = runner or FakeRunner(root)
    return R.launch(root, git=git or fake_git(), main_root=main_root or root, check=CHECK, runner=runner,
                    revalidate_fn=reval, snapshot=lambda: "gpu 40 C", cuda_ok=lambda: cuda), runner


def is_train(cmd):
    return cmd[2] == "scripts/train.py"


# ---- commands and order -------------------------------------------------------------------------------------
def test_commands_are_the_design_argument_lists():
    tl = A_REAL.TRAIN_LISTS
    ep = {"control": 25, "atlantic": 13}
    assert R.train_cmd("PY", "control", 3, tl, ep) == [
        "PY", "-u", "scripts/train.py", "--train-list", "storms_train_expanded.txt", "--val-list",
        "storms_val_expanded.txt", "--epochs", "25", "--seed", "3", "--loss", "l1", "--out",
        "models/n5_atlantic/control/seed3.pt", "--epoch-dir", "models/n5_atlantic/control/seed3_epochs"]
    assert R.train_cmd("PY", "atlantic", 3, tl, ep) == [
        "PY", "-u", "scripts/train.py", "--train-list", "storms_train_n5_atlantic.txt", "--val-list",
        "storms_val_expanded.txt", "--epochs", "13", "--seed", "3", "--loss", "l1", "--frame-list",
        "results/n5_atlantic/frame_list_atlantic.txt", "--out", "models/n5_atlantic/atlantic/seed3.pt"]
    assert R.eval_cmd("PY", "models/n5_atlantic/control/seed3_epochs/epoch_07.pt", "storms_train_expanded.txt",
                      "results/n5_atlantic/control_seed3_h9.csv") == [
        "PY", "scripts/evaluate.py", "--checkpoint", "models/n5_atlantic/control/seed3_epochs/epoch_07.pt",
        "--train-list", "storms_train_expanded.txt", "--holdout-list", "storms_holdout_h9.txt", "--out",
        "results/n5_atlantic/control_seed3_h9.csv"]
    for arm in ("control", "atlantic"):  # every path the driver writes is the analysis's pinned path
        pinned = A_REAL.pinned_paths(arm, 3)
        assert {k: v for k, v in R.run_paths(arm, 3).items() if k in pinned} == {
            k: v for k, v in pinned.items() if k != "selected_checkpoint"}


def test_schedule_interleaves_seeds_and_alternates_arm_order():
    order = R.schedule(range(10))
    assert [k for k, _, _ in order] == list(range(1, 21))
    assert [(a[0], s) for _, a, s in order[:6]] == [("c", 0), ("a", 0), ("a", 1), ("c", 1), ("c", 2), ("a", 2)]
    assert sorted((a, s) for _, a, s in order) == sorted((a, s) for a in ("control", "atlantic") for s in range(10))


def test_output_paths_are_confined():
    R.check_output_paths(R.planned_outputs(range(10)))
    for bad in ("models/storms/expanded/seed0.pt", "best_model.pt", "results/storms/x.csv", "site/x",
                "results/heldout_predictions.csv", "models/other/seed0.pt"):
        with pytest.raises(R.LaunchError, match="refusing output paths"):
            R.check_output_paths([bad])


# ---- guards -------------------------------------------------------------------------------------------------
def test_the_committed_analysis_refuses_today(tmp_path):
    """PR A's own state: the constants are None, so launch can't start before PR B."""
    with pytest.raises(R.LaunchError, match="guard 2: .*not filled in"):
        go(fake_repo(tmp_path, filled=False))


def test_real_checkout_constants_are_not_filled():
    assert A_REAL.ARM_TRAIN_FRAMES is None and A_REAL.FRAME_LIST_SHA256 is None
    with pytest.raises(A_REAL.DesignError, match="not final"):
        A_REAL.design_constants()


@pytest.mark.parametrize("git, match", [(fake_git(dirty=True), "guard 1: the tracked tree"),
                                        (fake_git(differs=True), "guard 1: scripts/analyze_n5.py differs")],
                         ids=["dirty", "differs"])
def test_refuses_an_uncommitted_tree_or_analysis(tmp_path, git, match):
    with pytest.raises(R.LaunchError, match=match):
        go(fake_repo(tmp_path), git=git)


def test_refuses_a_frame_list_with_another_sha(tmp_path):
    with pytest.raises(R.LaunchError, match="guard 2: .*frame_list_atlantic.txt"):
        go(fake_repo(tmp_path, frame_list_ok=False))


def test_refuses_an_analysis_protected_copy_that_drifted(tmp_path):
    with pytest.raises(R.LaunchError, match="guard 3: analyze_n5.PROTECTED differs"):
        go(fake_repo(tmp_path, protected_drift=True))


def test_refuses_a_changed_protected_file(tmp_path):
    root = fake_repo(tmp_path)
    (root / "best_model.pt").write_bytes(b"changed")
    with pytest.raises(R.LaunchError, match=r"guard 3: protected files differ: \['best_model.pt'\]"):
        go(root)


def test_refuses_any_existing_output(tmp_path):
    root = fake_repo(tmp_path)
    (root / "logs/n5_atlantic/runs").mkdir(parents=True)
    (root / "logs/n5_atlantic/runs/atlantic_seed7.json").write_text("{}")
    with pytest.raises(R.LaunchError, match="guard 4: .*atlantic_seed7.json"):
        go(root)


# ---- the runs -----------------------------------------------------------------------------------------------
def test_launch_end_to_end_and_the_analysis_accepts_it(tmp_path, capsys):
    root = fake_repo(tmp_path)
    mans, runner = go(root)
    assert f"analysis blob sha256 {hashlib.sha256(b'blob bytes' + bytes([10])).hexdigest()}" in capsys.readouterr().out
    assert [(m["order"], m["arm"], m["seed"]) for m in mans] == R.schedule(range(10))
    scripts = [c[2] if c[1] == "-u" else c[1] for c in runner.calls]
    assert set(scripts) == {"scripts/train.py", "scripts/evaluate.py"} and len(runner.calls) == 50
    # condition (b): all 20 trainings, in the schedule's order, before the first H9 evaluation
    assert [is_train(c) for c in runner.calls] == [True] * 20 + [False] * 30
    assert [(FakeRunner.arm_of(dict(zip(c[3::2], c[4::2]))), int(c[c.index("--seed") + 1])) for c in runner.calls[:20]] \
        == [(a, s) for _, a, s in R.schedule(range(10))]
    assert all(m["tie_rule"] is None for m in mans if m["arm"] == "control")
    for m in mans:
        log = (root / m["train_log"]).read_bytes()
        assert log.startswith(b"RUN arm=") and b"\r" not in log and not log.startswith(b"\xef\xbb\xbf")
        assert re.search(rb"^START .* \| gpu 40 C$", log, re.M) and re.search(rb"^END ", log, re.M)
        assert m["commit"] == COMMIT and m["protected_before"] == m["protected_after"]
    for s in range(10):
        epochs = root / f"models/n5_atlantic/control/seed{s}_epochs"
        assert sorted(p.name for p in epochs.iterdir()) == [f"epoch_{e:02d}.pt" for e in range(1, 26, 2)]
        man = json.loads((root / f"logs/n5_atlantic/runs/control_seed{s}.json").read_text(encoding="utf-8"))
        curve = val_curve("control", s)
        assert man["selected_epoch"] == min(range(1, 26, 2), key=lambda e: (curve[e], e))
        assert man["selected_epoch"] % 2 == 1
    # The full pre-registered analysis runs on the driver's output, from the synthetic repo's analyze_n5.
    A = load("analyze_n5_fake", root / "scripts" / "analyze_n5.py")
    res = A.run(root / "logs/n5_atlantic/runs", root / "results/n5_atlantic/analysis", root=root, git=fake_git())
    assert len(res["per_run"]) == 20 and res["joint"]
    natives = res["per_run"][res["per_run"].arm == "control"].native_selected_epoch
    assert set(natives % 2) == {0, 1}  # both the odd-native (identical vectors) and the even-native case ran


def test_revalidation_that_misses_train_py_pick_stops(tmp_path, monkeypatch):
    """Epochs 14 and 16 both log 8.00; train.py saved 14. Re-validated values that still round to the log but put
    the minimum on 16 mean the re-validation path does not reproduce training."""
    root = fake_repo(tmp_path)
    flat = {e: 8.0 + 0.05 * abs(e - 15) for e in range(1, 26)}
    flat.update({14: 7.9990, 16: 8.0010})
    monkeypatch.setitem(globals(), "val_curve", lambda arm, seed: dict(flat) if arm == "control" else
                        {e: 8.0 + 0.02 * (e - 8) ** 2 for e in range(1, 14)})

    def moved(epoch_dir, epochs, train_list):
        return {**flat, 14: 8.0012}  # 14 still rounds to 8.00, but 16 (8.0010) is now the minimum

    with pytest.raises(R.LaunchError, match="re-validation minimum is not train.py's native selection"):
        go(root, reval=moved)  # |d| = 2e-4 kt: outside the tie window
    assert len(list((root / "models/n5_atlantic/control/seed0_epochs").iterdir())) == 25  # nothing deleted
    assert not (root / "logs/n5_atlantic/runs/control_seed0.train.json").exists()


def test_revalidation_that_does_not_round_to_the_log_stops(tmp_path):
    def drift(epoch_dir, epochs, train_list):
        return {e: v + 0.02 for e, v in reval_from_curve(epoch_dir, epochs, train_list).items()}

    with pytest.raises(R.LaunchError, match="do not round to the logged ones"):
        go(fake_repo(tmp_path), reval=drift)


def test_training_failure_is_logged_and_stops(tmp_path):
    root = fake_repo(tmp_path)
    runner = FakeRunner(root, train_rc=1)
    with pytest.raises(R.LaunchError, match="training failed: logs/n5_atlantic/control_seed0.log"):
        go(root, runner=runner)
    log = (root / "logs/n5_atlantic/control_seed0.log").read_text(encoding="utf-8")
    assert log.startswith("RUN arm=control seed=0 order=1 ") and "\nTRAIN_FAILED rc=1 " in log and "\nEND " not in log
    assert len(runner.calls) == 1


def test_protected_file_changed_during_a_run_stops(tmp_path):
    root = fake_repo(tmp_path)

    def tamper(arm, seed):
        if (arm, seed) == ("atlantic", 0):
            (root / "results/heldout_predictions.csv").write_bytes(b"overwritten")

    with pytest.raises(R.LaunchError, match=r"protected hashes changed during atlantic seed 0: "
                                            r"\['results/heldout_predictions.csv'\]"):
        go(root, runner=FakeRunner(root, on_train=tamper))
    assert not (root / "logs/n5_atlantic/runs/atlantic_seed0.train.json").exists()  # no record: the run is unfinished


def test_driver_refuses_an_output_that_appears_mid_launch(tmp_path):
    root = fake_repo(tmp_path)

    def plant(arm, seed):
        if (arm, seed) == ("control", 0):
            (root / "results/n5_atlantic").mkdir(parents=True, exist_ok=True)
            (root / "results/n5_atlantic/atlantic_seed0_h9.csv").write_text("planted")

    with pytest.raises(R.LaunchError, match="refusing to overwrite .*atlantic_seed0_h9.csv"):
        go(root, runner=FakeRunner(root, on_train=plant))


# ---- 2026-10-19 amendment, guards B and C --------------------------------------------------------------------
@pytest.mark.parametrize("git, match", [(fake_git(fetch_rc=1), "guard 5: git fetch origin failed"),
                                        (fake_git(ancestor_rc=1), "guard 5: HEAD is not an ancestor of "
                                                                  "origin/phase2-experiments")],
                         ids=["fetch_fails", "not_ancestor"])
def test_refuses_a_commit_not_on_the_remote_branch(tmp_path, git, match):
    with pytest.raises(R.LaunchError, match=match):
        go(fake_repo(tmp_path), git=git)


def test_refuses_a_worktree_or_another_checkout(tmp_path):
    root = fake_repo(tmp_path / "a")
    (root / ".git").rmdir()
    (root / ".git").write_text("gitdir: C:/elsewhere/.git/worktrees/wt\n")  # what a worktree has
    with pytest.raises(R.LaunchError, match="guard 6: .* is not the main checkout"):
        go(root)
    other = fake_repo(tmp_path / "b")  # same protected bytes, so guards 1-5 pass; but it isn't the main checkout
    with pytest.raises(R.LaunchError, match="guard 6: .* is not the main checkout"):
        go(other, main_root=fake_repo(tmp_path / "c"))


def test_refuses_without_cuda(tmp_path):
    with pytest.raises(R.LaunchError, match=r"guard 6: torch.cuda.is_available\(\) is False"):
        go(fake_repo(tmp_path), cuda=False)


# ---- 2026-10-19 amendment, condition (b): the phase split and the abort procedure ------------------------------
def train_calls(runner):
    return [(FakeRunner.arm_of(dict(zip(c[3::2], c[4::2]))), int(c[c.index("--seed") + 1])) for c in runner.calls
            if is_train(c)]


@pytest.mark.parametrize("fail", [("atlantic", 1), ("control", 9)], ids=["order3", "order20"])
def test_phase2_refuses_while_any_training_run_is_unfinished(tmp_path, fail):
    root = fake_repo(tmp_path)
    ctx = R.preflight(root, fake_git(), root, CHECK, lambda: True)
    runner = FakeRunner(root, fail=fail)
    with pytest.raises(R.LaunchError, match="training failed"):
        R.train_phase(ctx, root, root, runner, reval_from_curve, lambda: "gpu 40 C")
    k = {(a, s): k for k, a, s in R.schedule(range(10))}[fail]
    with pytest.raises(R.LaunchError, match=r"phase 2: H9 evaluation refuses to start until all 20 training runs have "
                                            rf"finished; unfinished: \[\({k}, '{fail[0]}', {fail[1]}\)"):
        R.evaluate_phase(ctx, root, root, runner)
    assert all(is_train(c) for c in runner.calls) and R.h9_outputs(root, range(10)) == []


def test_training_never_follows_an_h9_output(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed"):
        go(root, runner=FakeRunner(root, fail=("atlantic", 1)))
    R.abort_run("atlantic", 1, "planted failure", root)
    (root / "results/n5_atlantic/control_seed0_h9.csv").write_text("looked")
    with pytest.raises(R.LaunchError, match=r"guard 4: H9 outputs exist while runs remain to be trained"):
        go(root)


def test_an_h9_output_planted_mid_phase1_stops_training(tmp_path):
    root = fake_repo(tmp_path)

    def plant(arm, seed):
        if (arm, seed) == ("atlantic", 0):  # order 2; order 1 (control seed 0) has its record by now
            (root / "results/n5_atlantic/control_seed0_h9.csv").write_text("looked")

    runner = FakeRunner(root, on_train=plant)
    with pytest.raises(R.LaunchError, match="phase 1: H9 outputs exist"):
        go(root, runner=runner)
    assert train_calls(runner) == [("control", 0), ("atlantic", 0)]


def test_abort_then_restart_keeps_seed_and_order_and_records_the_attempt(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed: logs/n5_atlantic/atlantic_seed1.log"):
        go(root, runner=FakeRunner(root, fail=("atlantic", 1)))
    with pytest.raises(R.LaunchError, match="guard 4: unfinished runs have files on disk: .*atlantic_seed1.log.*abort-run"):
        go(root)
    failed = (root / "logs/n5_atlantic/atlantic_seed1.log").read_bytes()
    dest = R.abort_run("atlantic", 1, "planted training failure", root)
    assert dest == "logs/n5_atlantic/aborted/atlantic_seed1_attempt1"
    ab = json.loads((root / dest / "abort.json").read_text(encoding="utf-8"))
    assert (ab["arm"], ab["seed"], ab["order"], ab["attempt"]) == ("atlantic", 1, 3, 1)
    assert ab["moved"] == {"logs/n5_atlantic/atlantic_seed1.log": {
        "to": f"{dest}/logs/n5_atlantic/atlantic_seed1.log", "sha256": hashlib.sha256(failed).hexdigest()}}
    assert (root / dest / "logs/n5_atlantic/atlantic_seed1.log").read_bytes() == failed  # kept, unchanged
    mans, runner = go(root)
    assert train_calls(runner) == [(a, s) for k, a, s in R.schedule(range(10)) if k >= 3]  # resumes at order 3
    m = next(m for m in mans if (m["arm"], m["seed"]) == ("atlantic", 1))
    assert (m["order"], m["seed"], m["aborted_attempts"]) == (3, 1, [f"{dest}/abort.json"])
    assert (root / m["train_log"]).read_bytes().startswith(b"RUN arm=atlantic seed=1 order=3 ")
    A = load("analyze_n5_fake_abort", root / "scripts" / "analyze_n5.py")
    assert len(A.run(root / "logs/n5_atlantic/runs", root / "results/n5_atlantic/analysis", root=root,
                     git=fake_git())["per_run"]) == 20


def test_abort_run_refusals(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="no files to move"):
        R.abort_run("control", 0, "nothing there", root)
    with pytest.raises(R.LaunchError, match="training failed"):
        go(root, runner=FakeRunner(root, fail=("atlantic", 1)))
    with pytest.raises(R.LaunchError, match="has a training record; finished runs are not aborted"):
        R.abort_run("control", 0, "finished", root)
    (root / "results/n5_atlantic/control_seed0_h9.csv").write_text("looked")
    with pytest.raises(R.LaunchError, match="no run may be aborted after an H9 evaluation"):
        R.abort_run("atlantic", 1, "after a look", root)
    assert (root / "logs/n5_atlantic/atlantic_seed1.log").exists() and not (root / R.ABORTED).exists()


# ---- 2026-10-19 amendment, condition (a): the tie rule ---------------------------------------------------------
@pytest.mark.parametrize("native, other", [(14, 16), (13, 15)], ids=["even_native", "odd_native"])
def test_tie_rule_evaluates_the_checkpoint_train_py_saved(tmp_path, monkeypatch, native, other):
    """train.py saved `native` (logged 8.00); re-validation puts `other` (also 8.00) 5e-5 kt below it, inside the
    tie window. For every seed the evaluated checkpoint is train.py's own seed<s>.pt. In the odd case two odd epochs
    tie (13, 15): the matched grid's first minimum alone would have taken 15; the tie rule takes 13."""
    root = fake_repo(tmp_path)
    flat = {e: 8.2 + 0.01 * e for e in range(1, 26)}
    flat.update({native: 7.9990, other: 8.0010})
    monkeypatch.setitem(globals(), "val_curve", lambda arm, seed: dict(flat) if arm == "control" else
                        {e: 8.0 + 0.02 * (e - 8) ** 2 + 0.001 * seed for e in range(1, 14)})
    mans, runner = go(root, reval=lambda epoch_dir, epochs, train_list: {**flat, native: 8.00105})
    ctl = [m for m in mans if m["arm"] == "control"]
    for m in ctl:
        assert (m["selected_epoch"], m["selected_checkpoint"]) == (native, m["native_checkpoint"])
        assert m["tie_rule"] == {"revalidated_argmin": other, "native_epoch": native,
                                 "delta_kt": pytest.approx(5e-5)}
    primary_evals = [c[c.index("--checkpoint") + 1] for c in runner.calls if c[-1].endswith(("control_seed0_h9.csv",))]
    assert primary_evals == ["models/n5_atlantic/control/seed0.pt"]
    A = load("analyze_n5_fake_tie", root / "scripts" / "analyze_n5.py")
    A.run(root / "logs/n5_atlantic/runs", root / "results/n5_atlantic/analysis", root=root, git=fake_git())
    report = (root / "results/n5_atlantic/analysis/report.txt").read_text(encoding="utf-8")
    assert f"control seeds {list(range(10))}" in report


def test_odd_grid_near_tie_is_recorded_and_the_first_minimum_stands(tmp_path, monkeypatch):
    """No disagreement with train.py (its pick, 14, is the re-validated argmin), but odd epochs 13 and 15 are within
    TIE_KT of each other. train.py never chose between them, so the pre-registered first minimum over the odd grid
    (15) stands, and the near tie is recorded for the reader."""
    root = fake_repo(tmp_path)
    flat = {e: 8.2 + 0.01 * e for e in range(1, 26)}
    flat.update({14: 7.9900, 13: 8.00105, 15: 8.0010})
    monkeypatch.setitem(globals(), "val_curve", lambda arm, seed: dict(flat) if arm == "control" else
                        {e: 8.0 + 0.02 * (e - 8) ** 2 + 0.001 * seed for e in range(1, 14)})
    mans, _ = go(root, reval=lambda epoch_dir, epochs, train_list: dict(flat))
    for m in (m for m in mans if m["arm"] == "control"):
        assert (m["selected_epoch"], m["tie_rule"], m["odd_grid_near_ties"]) == (15, None, [13])


# ---- spec stages and the protected copy ---------------------------------------------------------------------
def test_build_spec_and_plan_run_nothing(capsys, monkeypatch):
    monkeypatch.chdir(CHECKOUT)
    R.main(["build-spec"])
    out = capsys.readouterr().out
    assert "['PY', 'scripts/build_batch.py', '--ids-file', 'storms_n5_new.txt', '--sample-interval', '3h']" in out
    assert "never --force-hurdat2-refresh or --hurdat2-url" in out
    R.main(["plan"])
    out = capsys.readouterr().out
    assert out.count("order ") == 40 and "<epochs_for(ARM_TRAIN_FRAMES)>" in out
    assert out.index("PHASE 1") < out.index("PHASE 2") and "eval:" not in out[:out.index("PHASE 2")]
    for bad in (["abort-run"], ["abort-run", "atlantic", "1"], ["plan", "control", "0"]):
        with pytest.raises(SystemExit):
            R.main(bad)


def test_analysis_protected_copy_equals_the_coordinator_block(main_root):
    """hc-adversary item 13 (a): analyze_n5.PROTECTED is a second copy of the protected block; it must equal the
    source of truth, the main checkout's COORDINATOR.md."""
    assert dict(CHECK.load_expected(main_root / "COORDINATOR.md")) == A_REAL.PROTECTED
