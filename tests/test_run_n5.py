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
import os
import re
import subprocess
import sys
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


def fake_git(root=None, dirty=False, differs_at=(), fetch_rc=0, head=COMMIT, first_parent=(COMMIT,), pushed=True,
             untracked=(), committed_differs=(), extra_changed=()):
    """git for the synthetic repo. The 20 training records count as committed (at HEAD) unless named in
    `untracked` or `committed_differs`."""
    def git(args, check=True):
        if args[0] == "fetch":
            assert args == ["fetch", "origin"]
            return SimpleNamespace(stdout=b"", returncode=fetch_rc)
        if args[:2] == ["rev-list", "--first-parent"]:
            assert args == ["rev-list", "--first-parent", "origin/phase2-experiments"]
            return SimpleNamespace(stdout="\n".join(first_parent).encode() + b"\n", returncode=0)
        if args[0] == "merge-base":
            assert args == ["merge-base", "--is-ancestor", "HEAD", "origin/phase2-experiments"]
            return SimpleNamespace(stdout=b"", returncode=0 if pushed else 1)
        if args[0] == "status":
            return SimpleNamespace(stdout=b" M scripts/train.py\n" if dirty else b"", returncode=0)
        if args[:2] == ["diff", "--quiet"]:
            return SimpleNamespace(stdout=b"", returncode=1 if args[2] in differs_at else 0)
        if args[:4] == ["diff", "--no-renames", "-z", "--name-only"]:
            names = [R.run_paths(a, s)["record"] for _, a, s in R.schedule(range(10))] if args[4] != head else []
            return SimpleNamespace(stdout="\0".join(names + list(extra_changed)).encode() + b"\0", returncode=0)
        if args[0] == "ls-files":
            return SimpleNamespace(stdout=b"", returncode=1 if args[-1] in untracked else 0)
        if args[0] == "show" and args[1].endswith(".train.json"):
            rel = args[1].split(":", 1)[1]
            data = (root / rel).read_bytes() if (root / rel).exists() else b""
            return SimpleNamespace(stdout=data + (b" " if rel in committed_differs else b""), returncode=0)
        if args[0] == "show":
            return SimpleNamespace(stdout=b"blob bytes\n", returncode=0)
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=head.encode() + b"\n", returncode=0)
        if args[0] == "rev-parse":
            return SimpleNamespace(stdout=b"1234abcd\n", returncode=0)
        raise AssertionError(args)
    return git


class FakeRunner:
    """Stands in for subprocess.run on train.py and evaluate.py: writes their stdout lines and output files."""

    def __init__(self, root, train_rc=0, on_train=None, fail=None):
        self.root, self.train_rc, self.on_train, self.fail, self.calls = root, train_rc, on_train, fail, []
        self.eval_fail = set()  # --out paths whose evaluate.py "crashes" (rc 1, nothing written)

    def __call__(self, cmd, stdout, stderr, cwd, env):
        assert Path(cwd) == self.root and env["PYTHONUTF8"] == "1" and stderr is subprocess.STDOUT
        self.calls.append(cmd)
        args = dict(zip(cmd[3::2], cmd[4::2])) if cmd[2] == "scripts/train.py" else dict(zip(cmd[2::2], cmd[3::2]))
        rc = self.train_rc if cmd[2] == "scripts/train.py" else 0
        if cmd[2] != "scripts/train.py" and args["--out"] in self.eval_fail:
            stdout.write(b"Traceback (most recent call last):\n")
            return SimpleNamespace(returncode=1)
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


def same_bytes(a, b):
    return Path(a).read_bytes() == Path(b).read_bytes()


def pkg_in_src(root):
    return str(Path(root) / "src" / "hurricane_intensity" / "__init__.py")


def train(root, git=None, runner=None, reval=reval_from_curve, main_root=None, cuda=True, pkg=pkg_in_src,
          same=same_bytes):
    """Phase 1 (launch). Returns the runner."""
    runner = runner or FakeRunner(root)
    R.launch(root, git=git or fake_git(root), main_root=main_root or root, check=CHECK, runner=runner,
             revalidate_fn=reval, snapshot=lambda: "gpu 40 C", cuda_ok=lambda: cuda, child_import=pkg,
             same_weights_fn=same)
    return runner


def ev(root, git=None, runner=None, pkg=pkg_in_src):
    """Phase 2 (evaluate). Returns (manifests, runner)."""
    runner = runner or FakeRunner(root)
    return R.evaluate(root, git=git or fake_git(root), main_root=root, check=CHECK, runner=runner,
                      cuda_ok=lambda: True, child_import=pkg), runner


def go(root, git=None, runner=None, **kw):
    """Both phases with one runner, the records counting as committed at HEAD. Returns (manifests, runner)."""
    runner = train(root, git=git, runner=runner, **kw)
    return ev(root, git=git, runner=runner)[0], runner


def is_train(cmd):
    return cmd[2] == "scripts/train.py"


def train_calls(runner):
    return [(FakeRunner.arm_of(dict(zip(c[3::2], c[4::2]))), int(c[c.index("--seed") + 1])) for c in runner.calls
            if is_train(c)]


def analyse(root, name):
    A = load(name, root / "scripts" / "analyze_n5.py")
    A.run(root / "logs/n5_atlantic/runs", root / "results/n5_atlantic/analysis", root=root, git=fake_git(root))
    return A, (root / "results/n5_atlantic/analysis/report.txt").read_text(encoding="utf-8")


ORDER = {(a, s): k for k, a, s in R.schedule(range(10))}


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
        train(fake_repo(tmp_path, filled=False))


def test_real_checkout_constants_are_not_filled():
    assert A_REAL.ARM_TRAIN_FRAMES is None and A_REAL.FRAME_LIST_SHA256 is None
    with pytest.raises(A_REAL.DesignError, match="not final"):
        A_REAL.design_constants()


@pytest.mark.parametrize("kw, match", [(dict(dirty=True), "guard 1: the tracked tree"),
                                       (dict(differs_at=("HEAD",)), "guard 1: scripts/analyze_n5.py differs")],
                         ids=["dirty", "differs"])
def test_refuses_an_uncommitted_tree_or_analysis(tmp_path, kw, match):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match=match):
        train(root, git=fake_git(root, **kw))
    with pytest.raises(R.LaunchError, match=match):
        ev(root, git=fake_git(root, **kw))


def test_refuses_a_frame_list_with_another_sha(tmp_path):
    with pytest.raises(R.LaunchError, match="guard 2: .*frame_list_atlantic.txt"):
        train(fake_repo(tmp_path, frame_list_ok=False))


def test_refuses_an_analysis_protected_copy_that_drifted(tmp_path):
    with pytest.raises(R.LaunchError, match="guard 3: analyze_n5.PROTECTED differs"):
        train(fake_repo(tmp_path, protected_drift=True))


def test_refuses_a_changed_protected_file(tmp_path):
    root = fake_repo(tmp_path)
    (root / "best_model.pt").write_bytes(b"changed")
    with pytest.raises(R.LaunchError, match=r"guard 3: protected files differ: \['best_model.pt'\]"):
        train(root)


def test_refuses_any_existing_output(tmp_path):
    root = fake_repo(tmp_path)
    (root / "logs/n5_atlantic/runs").mkdir(parents=True)
    (root / "logs/n5_atlantic/runs/atlantic_seed7.json").write_text("{}")
    with pytest.raises(R.LaunchError, match="guard 4: .*atlantic_seed7.json"):
        train(root)


@pytest.mark.parametrize("kw, match", [(dict(fetch_rc=1), "guard 5: git fetch origin failed"),
                                       (dict(first_parent=("e" * 40,)), "guard 5: .* is not on the first-parent chain")],
                         ids=["fetch_fails", "not_first_parent"])
def test_refuses_a_commit_not_on_the_first_parent_chain(tmp_path, kw, match):
    """hc-adversary item 16, finding 2: an ancestor is not enough; a PR branch's own commits are ancestors too."""
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match=match):
        train(root, git=fake_git(root, **kw))
    assert not (root / "logs/n5_atlantic/control_seed0.log").exists()


def test_refuses_a_worktree_or_another_checkout(tmp_path):
    root = fake_repo(tmp_path / "a")
    (root / ".git").rmdir()
    (root / ".git").write_text("gitdir: C:/elsewhere/.git/worktrees/wt\n")  # what a worktree has
    with pytest.raises(R.LaunchError, match="guard 6: .* is not the main checkout"):
        train(root)
    other = fake_repo(tmp_path / "b")  # same protected bytes, so guards 1-3 pass; but it isn't the main checkout
    with pytest.raises(R.LaunchError, match="guard 6: .* is not the main checkout"):
        train(other, main_root=fake_repo(tmp_path / "c"))


def test_refuses_a_child_that_imports_the_package_from_elsewhere(tmp_path):
    root = fake_repo(tmp_path)
    for pkg in (lambda r: str(tmp_path / "other" / "src" / "hurricane_intensity" / "__init__.py"), lambda r: ""):
        with pytest.raises(R.LaunchError, match="guard 6: a child python imports hurricane_intensity from"):
            train(root, pkg=pkg)
        with pytest.raises(R.LaunchError, match="guard 6: a child python imports hurricane_intensity from"):
            ev(root, pkg=pkg)


def test_refuses_without_cuda(tmp_path):
    with pytest.raises(R.LaunchError, match=r"guard 6: torch.cuda.is_available\(\) is False"):
        train(fake_repo(tmp_path), cuda=False)


def test_a_held_lock_refuses_every_writing_stage(tmp_path):
    root = fake_repo(tmp_path)
    (root / "logs/n5_atlantic").mkdir(parents=True)
    (root / R.LOCK).write_text("pid 1\n")
    for stage in (lambda: train(root), lambda: ev(root), lambda: R.abort_run("control", 0, "x", root),
                  lambda: R.abort_all("x", root), lambda: R.abort_eval("control", 0, "x", root)):
        with pytest.raises(R.LaunchError, match="run_n5.lock exists"):
            stage()
    assert (root / R.LOCK).read_text() == "pid 1\n"  # a refused stage never removes another's lock


# ---- the runs -----------------------------------------------------------------------------------------------
def test_launch_end_to_end_and_the_analysis_accepts_it(tmp_path, capsys):
    root = fake_repo(tmp_path)
    runner = train(root)
    out = capsys.readouterr().out
    assert f"HEAD {COMMIT}; analysis blob sha256 {hashlib.sha256(b'blob bytes' + bytes([10])).hexdigest()}" in out
    assert "PHASE 1 DONE" in out and all(R.run_paths(a, s)["record"] in out for _, a, s in R.schedule(range(10)))
    assert all(is_train(c) for c in runner.calls) and len(runner.calls) == 20  # launch stops after phase 1
    assert not (root / R.LOCK).exists()
    mans, runner = ev(root, runner=runner)
    assert [(m["order"], m["arm"], m["seed"]) for m in mans] == R.schedule(range(10))
    assert [is_train(c) for c in runner.calls] == [True] * 20 + [False] * 30
    assert train_calls(runner) == [(a, s) for _, a, s in R.schedule(range(10))]
    for m in mans:
        log = (root / m["train_log"]).read_bytes()
        assert log.startswith(b"RUN arm=") and b"\r" not in log and not log.startswith(b"\xef\xbb\xbf")
        assert re.search(rb"^START .* \| gpu 40 C$", log, re.M) and re.search(rb"^END ", log, re.M)
        assert m["commit"] == COMMIT and m["protected_before"] == m["protected_after"]
        assert m["aborted_attempts"] == [] and m["eval_attempts"] == []
        assert m.get("off_grid_primary", False) is False
    assert all(m["tie_rule"] is None for m in mans if m["arm"] == "control")
    for s in range(10):
        epochs = root / f"models/n5_atlantic/control/seed{s}_epochs"
        assert sorted(p.name for p in epochs.iterdir()) == [f"epoch_{e:02d}.pt" for e in range(1, 26, 2)]
        man = json.loads((root / f"logs/n5_atlantic/runs/control_seed{s}.json").read_text(encoding="utf-8"))
        curve = val_curve("control", s)
        assert man["selected_epoch"] == min(range(1, 26, 2), key=lambda e: (curve[e], e))
        assert man["selected_epoch"] % 2 == 1
    # The full pre-registered analysis runs on the driver's output, from the synthetic repo's analyze_n5.
    A = load("analyze_n5_fake", root / "scripts" / "analyze_n5.py")
    res = A.run(root / "logs/n5_atlantic/runs", root / "results/n5_atlantic/analysis", root=root, git=fake_git(root))
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
        train(root, reval=moved)  # |d| = 2e-4 kt: outside the tie window
    assert len(list((root / "models/n5_atlantic/control/seed0_epochs").iterdir())) == 25  # nothing deleted
    assert not (root / "logs/n5_atlantic/runs/control_seed0.train.json").exists()


def test_revalidation_that_does_not_round_to_the_log_stops(tmp_path):
    def drift(epoch_dir, epochs, train_list):
        return {e: v + 0.02 for e, v in reval_from_curve(epoch_dir, epochs, train_list).items()}

    with pytest.raises(R.LaunchError, match="do not round to the logged ones"):
        train(fake_repo(tmp_path), reval=drift)


def test_native_checkpoint_must_hold_the_parsed_epoch_weights(tmp_path):
    """Item 16, finding 9: seed<s>.pt must equal epoch_<native>.pt before anything is deleted."""
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match=r"seed0.pt does not hold epoch_\d\d.pt's weights"):
        train(root, same=lambda a, b: False)
    assert len(list((root / "models/n5_atlantic/control/seed0_epochs").iterdir())) == 25


@pytest.mark.skipif(not hasattr(__import__("_winapi"), "CreateJunction") if sys.platform == "win32" else True,
                    reason="directory junctions are a Windows feature")
def test_epoch_dir_resolving_outside_the_repo_is_refused(tmp_path):
    import _winapi
    root = fake_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    def junction(arm, seed):
        if (arm, seed) == ("control", 0):
            _winapi.CreateJunction(str(outside), str(root / "models/n5_atlantic/control/seed0_epochs"))

    with pytest.raises(R.LaunchError, match="seed0_epochs resolves to .*outside"):
        train(root, runner=FakeRunner(root, on_train=junction))
    assert len(list(outside.iterdir())) == 25  # nothing deleted through the junction


def test_training_failure_is_logged_and_stops(tmp_path):
    root = fake_repo(tmp_path)
    runner = FakeRunner(root, train_rc=1)
    with pytest.raises(R.LaunchError, match="training failed: logs/n5_atlantic/control_seed0.log"):
        train(root, runner=runner)
    log = (root / "logs/n5_atlantic/control_seed0.log").read_text(encoding="utf-8")
    assert log.startswith("RUN arm=control seed=0 order=1 ") and "\nTRAIN_FAILED rc=1 " in log and "\nEND " not in log
    assert len(runner.calls) == 1 and not (root / R.LOCK).exists()


def test_protected_file_changed_during_a_run_stops(tmp_path):
    root = fake_repo(tmp_path)

    def tamper(arm, seed):
        if (arm, seed) == ("atlantic", 0):
            (root / "results/heldout_predictions.csv").write_bytes(b"overwritten")

    with pytest.raises(R.LaunchError, match=r"protected hashes changed during atlantic seed 0: "
                                            r"\['results/heldout_predictions.csv'\]"):
        train(root, runner=FakeRunner(root, on_train=tamper))
    assert not (root / "logs/n5_atlantic/runs/atlantic_seed0.train.json").exists()  # no record: the run is unfinished


def test_driver_refuses_an_output_that_appears_mid_launch(tmp_path):
    root = fake_repo(tmp_path)

    def plant(arm, seed):
        if (arm, seed) == ("control", 0):
            (root / "results/n5_atlantic").mkdir(parents=True, exist_ok=True)
            (root / "results/n5_atlantic/atlantic_seed0_h9.csv").write_text("planted")

    with pytest.raises(R.LaunchError, match="refusing to overwrite .*atlantic_seed0_h9.csv"):
        train(root, runner=FakeRunner(root, on_train=plant))


# ---- condition (b): the phase split ---------------------------------------------------------------------------
@pytest.mark.parametrize("fail", [("atlantic", 1), ("control", 9)], ids=["order3", "order20"])
def test_phase2_refuses_while_any_training_run_is_unfinished(tmp_path, fail):
    root = fake_repo(tmp_path)
    runner = FakeRunner(root, fail=fail)
    with pytest.raises(R.LaunchError, match="training failed"):
        train(root, runner=runner)
    with pytest.raises(R.LaunchError, match=r"phase 2: H9 evaluation refuses to start until all 20 training runs have "
                                            rf"finished; unfinished: \[\({ORDER[fail]}, '{fail[0]}', {fail[1]}\)"):
        ev(root, runner=runner)
    assert all(is_train(c) for c in runner.calls) and R.h9_outputs(root, range(10)) == []


@pytest.mark.parametrize("kw, match", [
    (dict(untracked=("logs/n5_atlantic/runs/atlantic_seed4.train.json",)), "guard 5e: .*atlantic_seed4.train.json is not tracked"),
    (dict(committed_differs=("logs/n5_atlantic/runs/control_seed2.train.json",)),
     "guard 5e: .*control_seed2.train.json differs from its committed blob"),
    (dict(head="d" * 40, first_parent=(COMMIT,), pushed=False), "guard 5e: HEAD is not pushed"),
    (dict(head="d" * 40, first_parent=("d" * 40,)), f"guard 5: {COMMIT} is not on the first-parent chain"),
    (dict(head="d" * 40, first_parent=(COMMIT,), extra_changed=("scripts/train.py",)),
     r"guard 5e: HEAD differs from the launch commit .* in more than the 20 records and the decisions docs: "
     r"\['scripts/train.py'\]"),
], ids=["untracked", "differs", "not_pushed", "launch_not_first_parent", "code_changed"])
def test_phase2_requires_the_committed_records_as_evidence(tmp_path, kw, match):
    """Item 16, finding 3: the 20 records committed and pushed before phase 2 witness that all trainings finished
    before any H9 look. HEAD may be a later commit, but only by those records."""
    root = fake_repo(tmp_path)
    runner = train(root)
    with pytest.raises(R.LaunchError, match=match):
        ev(root, git=fake_git(root, **kw), runner=runner)
    assert all(is_train(c) for c in runner.calls)
    mans, runner = ev(root, git=fake_git(root, head="d" * 40, first_parent=(COMMIT, "d" * 40)), runner=runner)
    assert len(mans) == 20 and {m["commit"] for m in mans} == {COMMIT}  # records one commit later: accepted


@pytest.mark.parametrize("edit, match", [
    (lambda r: r.pop("tie_rule"), "no tie_rule field"),
    (lambda r: r.update(commit="f" * 40), "runs come from more than one commit"),
    (lambda r: r.update(analysis_blob_sha256="0" * 64), "the records' analysis blob is not HEAD's"),
    (lambda r: r.update(selected_epoch=r["selected_epoch"] + 2), "selected checkpoint is not the validation minimum"),
    (lambda r: r.update(odd_grid_near_ties=[25]), "odd_grid_near_ties .* does not match"),
], ids=["no_tie_rule", "two_commits", "other_blob", "wrong_selection", "near_ties"])
def test_phase2_entry_applies_the_training_side_checks_before_any_h9(tmp_path, edit, match):
    """Item 16, finding 1(c): a record the analysis would refuse is refused before the first H9 evaluation."""
    root = fake_repo(tmp_path)
    runner = train(root)
    p = root / "logs/n5_atlantic/runs/control_seed2.train.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    edit(rec)
    p.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(R.LaunchError, match=match):
        ev(root, runner=runner)
    assert all(is_train(c) for c in runner.calls) and R.h9_outputs(root, range(10)) == []


def test_phase2_entry_runs_verify_self_at_the_launch_commit(tmp_path):
    root = fake_repo(tmp_path)
    runner = train(root)
    with pytest.raises(R.LaunchError, match="phase 2 entry, before any H9 evaluation: .*differs from the launch commit"):
        ev(root, git=fake_git(root, differs_at=(COMMIT,)), runner=runner)
    assert all(is_train(c) for c in runner.calls)


def test_relaunch_at_another_commit_is_refused_until_abort_all(tmp_path):
    """Item 16, finding 1(a)/(b): records from commit c0ffee... and HEAD at a fix commit. launch refuses; abort-all
    (before any H9 output) moves every run aside, finished ones included, and all 20 restart in the same order."""
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed"):
        train(root, runner=FakeRunner(root, fail=("atlantic", 1)))
    R.abort_run("atlantic", 1, "crash; fixed in a follow-up PR", root)
    fix = fake_git(root, head="f" * 40, first_parent=(COMMIT, "f" * 40))
    runner = FakeRunner(root)
    with pytest.raises(R.LaunchError, match="guard 4: runs recorded at another commit .* abort-all"):
        train(root, git=fix, runner=runner)
    assert runner.calls == []
    moved = R.abort_all("relaunch from the fix commit", root)
    assert sorted(Path(m).name for m in moved) == ["atlantic_seed0_attempt1", "control_seed0_attempt1"]  # finished
    mans, runner = go(root, git=fix)
    assert train_calls(runner) == [(a, s) for _, a, s in R.schedule(range(10))]
    assert {m["commit"] for m in mans} == {"f" * 40}
    m = next(m for m in mans if (m["arm"], m["seed"]) == ("atlantic", 1))
    assert m["order"] == 3 and len(m["aborted_attempts"]) == 1


def test_abort_all_refuses_after_an_h9_output(tmp_path):
    root = fake_repo(tmp_path)
    train(root)
    (root / "results/n5_atlantic/control_seed0_h9.csv").write_text("looked")
    with pytest.raises(R.LaunchError, match="abort-all: H9 outputs exist"):
        R.abort_all("after a look", root)
    assert (root / "logs/n5_atlantic/runs/control_seed0.train.json").exists()


def test_training_never_follows_an_h9_output(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed"):
        train(root, runner=FakeRunner(root, fail=("atlantic", 1)))
    R.abort_run("atlantic", 1, "planted failure", root)
    (root / "results/n5_atlantic/control_seed0_h9.csv").write_text("looked")
    with pytest.raises(R.LaunchError, match=r"guard 4: H9 outputs exist while runs remain to be trained"):
        train(root)


def test_an_h9_output_planted_mid_phase1_stops_training(tmp_path):
    root = fake_repo(tmp_path)

    def plant(arm, seed):
        if (arm, seed) == ("atlantic", 0):  # order 2; order 1 (control seed 0) has its record by now
            (root / "results/n5_atlantic/control_seed0_h9.csv").write_text("looked")

    runner = FakeRunner(root, on_train=plant)
    with pytest.raises(R.LaunchError, match="phase 1: H9 outputs exist"):
        train(root, runner=runner)
    assert train_calls(runner) == [("control", 0), ("atlantic", 0)]


# ---- condition (b): aborts ---------------------------------------------------------------------------------------
def test_abort_then_restart_keeps_seed_and_order_and_records_the_attempt(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed: logs/n5_atlantic/atlantic_seed1.log"):
        train(root, runner=FakeRunner(root, fail=("atlantic", 1)))
    with pytest.raises(R.LaunchError, match="guard 4: unfinished runs have files on disk: .*atlantic_seed1.log.*abort-run"):
        train(root)
    failed = (root / "logs/n5_atlantic/atlantic_seed1.log").read_bytes()
    dest = R.abort_run("atlantic", 1, "planted training failure", root)
    assert dest == "logs/n5_atlantic/aborted/atlantic_seed1_attempt1"
    ab = json.loads((root / dest / "abort.json").read_text(encoding="utf-8"))
    assert (ab["arm"], ab["seed"], ab["order"], ab["attempt"], ab["status"]) == ("atlantic", 1, 3, 1, "complete")
    assert ab["moved"] == {"logs/n5_atlantic/atlantic_seed1.log": {
        "to": f"{dest}/logs/n5_atlantic/atlantic_seed1.log", "sha256": hashlib.sha256(failed).hexdigest()}}
    assert (root / dest / "logs/n5_atlantic/atlantic_seed1.log").read_bytes() == failed  # kept, unchanged
    mans, runner = go(root)
    assert train_calls(runner) == [(a, s) for k, a, s in R.schedule(range(10)) if k >= 3]  # resumes at order 3
    m = next(m for m in mans if (m["arm"], m["seed"]) == ("atlantic", 1))
    assert (m["order"], m["seed"], m["aborted_attempts"]) == (3, 1, [f"{dest}/abort.json"])
    assert (root / m["train_log"]).read_bytes().startswith(b"RUN arm=atlantic seed=1 order=3 ")
    A, report = analyse(root, "analyze_n5_fake_abort")
    assert "Aborted training attempts (restarted with the same seed and order): {'atlantic seed 1': [" in report
    (root / dest / "abort.json").unlink()  # the analysis verifies every recorded attempt
    with pytest.raises(A.DesignError, match="aborted_attempts .*abort.json is missing or not complete"):
        A.run(root / "logs/n5_atlantic/runs", root / "results/n5_atlantic/analysis2", root=root, git=fake_git(root))


def test_abort_run_refusals(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="no files to move"):
        R.abort_run("control", 0, "nothing there", root)
    with pytest.raises(R.LaunchError, match="training failed"):
        train(root, runner=FakeRunner(root, fail=("atlantic", 1)))
    with pytest.raises(R.LaunchError, match="has a training record that verifies; finished runs are not aborted"):
        R.abort_run("control", 0, "finished", root)
    (root / "results/n5_atlantic/control_seed0_h9.csv").write_text("looked")
    with pytest.raises(R.LaunchError, match="no run may be aborted after an H9 evaluation"):
        R.abort_run("atlantic", 1, "after a look", root)
    assert (root / "logs/n5_atlantic/atlantic_seed1.log").exists() and not (root / R.ABORTED).exists()
    assert not (root / R.LOCK).exists()


def test_a_record_that_fails_verification_can_be_aborted_and_retrained(tmp_path):
    """Item 16, finding 4(a): a kept file touched after training. Phase 2 refuses; abort-run takes the run (and its
    record); relaunch retrains it alone; phase 2 then runs."""
    root = fake_repo(tmp_path)
    runner = train(root)
    (root / "models/n5_atlantic/control/seed4_epochs/epoch_13.pt").write_bytes(b"swapped after training")
    with pytest.raises(R.LaunchError, match=r"phase 2: training records that do not verify: \[\('control', 4\)\]"):
        ev(root, runner=runner)
    with pytest.raises(R.LaunchError, match=r"guard 4: training records that do not verify: \[\('control', 4\)\]"):
        train(root)
    dest = R.abort_run("control", 4, "kept epoch file changed", root)
    moved = json.loads((root / dest / "abort.json").read_text(encoding="utf-8"))["moved"]
    assert "logs/n5_atlantic/runs/control_seed4.train.json" in moved and len(moved) == 2 + 13 + 1
    runner = train(root)
    assert train_calls(runner) == [("control", 4)]
    mans, _ = ev(root, runner=runner)
    assert len(mans) == 20


def test_a_crashed_evaluation_is_moved_aside_and_the_same_checkpoint_re_evaluated(tmp_path):
    """Item 16, finding 4(b)."""
    root = fake_repo(tmp_path)
    runner = train(root)
    with pytest.raises(R.LaunchError, match="abort-eval: atlantic seed 2 is trained"):
        R.abort_eval("atlantic", 2, "nothing to move", root)
    (root / "results/n5_atlantic").mkdir(exist_ok=True)
    (root / "results/n5_atlantic/atlantic_seed2_h9.csv").write_text("left by a crashed evaluation")
    (root / "logs/n5_atlantic/atlantic_seed2_eval_h9.log").write_text("checkpoint: x\noverall MAE: 9.99 kt\n")
    with pytest.raises(R.LaunchError, match=r"phase 2: H9 outputs without a manifest for \[\('atlantic', 2\)\]; .*abort-eval"):
        ev(root, runner=runner)
    with pytest.raises(R.LaunchError, match="abort-run: H9 outputs exist"):
        R.abort_run("atlantic", 2, "retrain after a look", root)
    dest = R.abort_eval("atlantic", 2, "evaluation crashed", root)
    assert dest == "logs/n5_atlantic/aborted/atlantic_seed2_eval1"
    ab = json.loads((root / dest / "abort.json").read_text(encoding="utf-8"))
    assert sorted(ab["moved"]) == ["logs/n5_atlantic/atlantic_seed2_eval_h9.log", "results/n5_atlantic/atlantic_seed2_h9.csv"]
    mans, runner = ev(root, runner=runner)
    m = next(m for m in mans if (m["arm"], m["seed"]) == ("atlantic", 2))
    assert m["eval_attempts"] == [f"{dest}/abort.json"]
    assert m["selected_checkpoint"] == "models/n5_atlantic/atlantic/seed2.pt"
    _, report = analyse(root, "analyze_n5_fake_eval")
    assert "aborted H9 evaluation attempts (same checkpoint re-evaluated): {'atlantic seed 2': [" in report


def test_an_interrupted_abort_is_pending_blocks_launch_and_resumes(tmp_path, monkeypatch):
    """Item 16, finding 5: abort.json (pending, every file hashed) is written before any move."""
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed"):
        train(root, runner=FakeRunner(root, fail=("atlantic", 1)))
    (root / "models/n5_atlantic/atlantic/seed1.pt").write_bytes(b"partial checkpoint")
    real, calls = os.rename, []

    def flaky(a, b):
        calls.append(a)
        if len(calls) == 2:
            raise PermissionError(32, "in use", str(a))
        return real(a, b)

    monkeypatch.setattr(R.os, "rename", flaky)
    with pytest.raises(PermissionError):
        R.abort_run("atlantic", 1, "crash", root)
    monkeypatch.setattr(R.os, "rename", real)
    ab = json.loads((root / "logs/n5_atlantic/aborted/atlantic_seed1_attempt1/abort.json").read_text(encoding="utf-8"))
    assert ab["status"] == "pending" and len(ab["moved"]) == 2 and not (root / R.LOCK).exists()
    with pytest.raises(R.LaunchError, match="guard 4: unfinished aborts .*atlantic_seed1_attempt1"):
        train(root)
    assert R.abort_run("atlantic", 1, "crash", root) == "logs/n5_atlantic/aborted/atlantic_seed1_attempt1"
    ab = json.loads((root / "logs/n5_atlantic/aborted/atlantic_seed1_attempt1/abort.json").read_text(encoding="utf-8"))
    assert ab["status"] == "complete" and all((root / v["to"]).is_file() for v in ab["moved"].values())
    assert len(go(root)[0]) == 20


# ---- item 16b: a look survives abort-eval; pending aborts; 5e paths -----------------------------------------------
def test_abort_eval_cannot_launder_an_h9_look(tmp_path):
    """hc-adversary item 16b, finding 1, its exact sequence: control seed 0's native evaluation crashes after its
    selected evaluation logged an MAE. abort-eval empties the pinned H9 paths, yet abort-all, abort-run and launch
    stay refused: the eval abort and the phase-2 marker are the look."""
    root = fake_repo(tmp_path)
    runner = train(root)
    runner.eval_fail = {"results/n5_atlantic/control_seed0_native_h9.csv"}
    with pytest.raises(R.LaunchError, match="evaluation failed: logs/n5_atlantic/control_seed0_native_eval_h9.log"):
        ev(root, runner=runner)
    assert b"overall MAE:" in (root / "logs/n5_atlantic/control_seed0_eval_h9.log").read_bytes()  # the look
    marker = json.loads((root / R.PHASE2_MARKER).read_text(encoding="utf-8"))
    assert (marker["head"], marker["launch_commit"]) == (COMMIT, COMMIT)
    R.abort_eval("control", 0, "native evaluation crashed", root)
    assert R.h9_outputs(root, range(10)) == []  # the pinned paths are empty again...
    for stage, match in ((lambda: R.abort_all("start over", root), "abort-all: H9 outputs exist"),
                         (lambda: R.abort_run("atlantic", 3, "retrain", root), "abort-run: H9 outputs exist"),
                         (lambda: train(root), "guard 4: logs/n5_atlantic/phase2_started.json exists")):
        with pytest.raises(R.LaunchError, match=match):  # ...but the look is not forgotten
            stage()
    runner.eval_fail = set()
    mans, runner = ev(root, runner=runner)
    m = mans[0]
    assert (m["arm"], m["seed"], m["eval_attempts"]) == ("control", 0, ["logs/n5_atlantic/aborted/control_seed0_eval1/abort.json"])
    assert len(mans) == 20 and (root / R.PHASE2_MARKER).exists()


def test_an_eval_abort_during_phase1_blocks_further_training(tmp_path):
    """Item 16b, finding 1, phase-1 variant: an H9 output on a trained run, moved aside by abort-eval, still counts."""
    root = fake_repo(tmp_path)

    def plant(arm, seed):
        if (arm, seed) == ("atlantic", 0):
            (root / "results/n5_atlantic/control_seed0_h9.csv").write_text("looked")

    with pytest.raises(R.LaunchError, match="phase 1: H9 outputs exist"):
        train(root, runner=FakeRunner(root, on_train=plant))
    R.abort_eval("control", 0, "stray H9 output", root)
    assert R.h9_outputs(root, range(10)) == []
    runner = FakeRunner(root)
    with pytest.raises(R.LaunchError, match=r"guard 4: H9 outputs exist while runs remain to be trained: "
                                            r"\['logs/n5_atlantic/aborted/control_seed0_eval1/abort.json'\]"):
        train(root, runner=runner)
    assert runner.calls == []
    with pytest.raises(R.LaunchError, match="abort-all: H9 outputs exist"):
        R.abort_all("start over", root)


def test_the_phase2_marker_alone_is_a_look(tmp_path):
    """Pinned H9 outputs deleted by hand (no abort-eval, no trace in aborted/): the marker still blocks every abort
    and launch."""
    root = fake_repo(tmp_path)
    runner = train(root)
    runner.eval_fail = {"results/n5_atlantic/control_seed0_native_h9.csv"}
    with pytest.raises(R.LaunchError, match="evaluation failed"):
        ev(root, runner=runner)
    for rel in R.h9_outputs(root, range(10)):
        (root / rel).unlink()
    assert R.looked(root, range(10)) == [R.PHASE2_MARKER]
    for stage, match in ((lambda: R.abort_all("start over", root), "abort-all: H9 outputs exist .*phase2_started"),
                         (lambda: R.abort_run("atlantic", 3, "retrain", root), "abort-run: H9 outputs exist .*phase2_started")):
        with pytest.raises(R.LaunchError, match=match):
            stage()


def test_an_eval_abort_appearing_mid_phase1_stops_training(tmp_path):
    """The per-run phase-1 check counts every kind of look, not only the pinned H9 paths."""
    root = fake_repo(tmp_path)

    def plant(arm, seed):
        if (arm, seed) == ("atlantic", 0):  # order 2: an H9 evaluation of control seed 0, aborted, leaves this
            d = root / "logs/n5_atlantic/aborted/control_seed0_eval1"
            d.mkdir(parents=True)
            (d / "abort.json").write_text(json.dumps({"status": "complete", "moved": {}}), encoding="utf-8")

    runner = FakeRunner(root, on_train=plant)
    with pytest.raises(R.LaunchError, match="phase 1: H9 outputs exist .*control_seed0_eval1/abort.json"):
        train(root, runner=runner)
    assert train_calls(runner) == [("control", 0), ("atlantic", 0)]


def test_a_pending_eval_abort_blocks_evaluate_and_resumes(tmp_path, monkeypatch):
    """Item 16b, finding 2: evaluate refuses an unfinished abort before any H9 step; re-running abort-eval
    finishes it."""
    root = fake_repo(tmp_path)
    runner = train(root)
    runner.eval_fail = {"results/n5_atlantic/control_seed0_native_h9.csv"}
    with pytest.raises(R.LaunchError, match="evaluation failed"):
        ev(root, runner=runner)
    runner.eval_fail, calls_before = set(), len(runner.calls)
    real, n = R.write_json_replace, []

    def flaky(path, obj):
        n.append(path)
        if len(n) == 2:  # the "complete" write
            raise OSError("disk full")
        return real(path, obj)

    monkeypatch.setattr(R, "write_json_replace", flaky)
    with pytest.raises(OSError):
        R.abort_eval("control", 0, "native evaluation crashed", root)
    monkeypatch.setattr(R, "write_json_replace", real)
    with pytest.raises(R.LaunchError, match="guard 4: unfinished aborts .*control_seed0_eval1"):
        ev(root, runner=runner)
    assert len(runner.calls) == calls_before
    R.abort_eval("control", 0, "native evaluation crashed", root)  # resumes the pending abort
    mans, _ = ev(root, runner=runner)
    assert mans[0]["eval_attempts"] == ["logs/n5_atlantic/aborted/control_seed0_eval1/abort.json"]


def test_evaluate_one_checks_its_eval_attempts_before_the_h9_step(tmp_path, monkeypatch):
    """Item 16b, finding 2, second half: even past the stage guard, no H9 step runs for a run whose recorded eval
    attempt is incomplete."""
    root = fake_repo(tmp_path)
    runner = train(root)
    d = root / "logs/n5_atlantic/aborted/control_seed0_eval1"
    d.mkdir(parents=True)
    (d / "abort.json").write_text(json.dumps({"status": "pending", "moved": {}}), encoding="utf-8")
    monkeypatch.setattr(R, "pending_aborts", lambda root: [])  # as if the stage guard had missed it
    with pytest.raises(R.LaunchError, match="before the H9 evaluation of control seed 0: .*control_seed0_eval1"):
        ev(root, runner=runner)
    assert all(is_train(c) for c in runner.calls)


@pytest.mark.parametrize("extra, ok", [
    (("logs/decisions.md",), True),
    (("reports/decisions/N5.md", "logs/decisions.md"), True),
    (("logs/decisions.md", "scripts/train.py"), False),
    (("reports/decisions/sub/x.md",), False),
    (("scripts/a b.py",), False),
], ids=["decisions_md", "decision_records", "code_with_decisions", "nested_doc", "path_with_space"])
def test_phase2_allows_decisions_docs_between_l_and_head(tmp_path, extra, ok):
    """Item 16b, finding 3: `git diff --no-renames -z` (the fake git asserts those flags); docs may change between L
    and HEAD, code may not, and a path with a space is one path."""
    root = fake_repo(tmp_path)
    runner = train(root)
    git = fake_git(root, head="d" * 40, first_parent=(COMMIT, "d" * 40), extra_changed=extra)
    if ok:
        assert len(ev(root, git=git, runner=runner)[0]) == 20
    else:
        bad = [p for p in extra if not p.startswith(("logs/decisions.md", "reports/decisions/N5"))]
        with pytest.raises(R.LaunchError, match=re.escape(f"records and the decisions docs: {bad}")):
            ev(root, git=git, runner=runner)


def test_abort_eval_takes_a_manifest_that_load_run_refuses(tmp_path):
    """Item 16b, finding 4: an H9 CSV changed after its manifest. evaluate refuses with the remedy; abort-eval moves
    the outputs and the manifest; evaluate re-evaluates the same recorded checkpoint, that run only."""
    root = fake_repo(tmp_path)
    mans, runner = go(root)
    with pytest.raises(R.LaunchError, match="abort-eval: atlantic seed 3 is evaluated"):
        R.abort_eval("atlantic", 3, "valid manifest", root)
    (root / "results/n5_atlantic/atlantic_seed3_h9.csv").write_bytes(b"damaged")
    with pytest.raises(R.LaunchError, match="phase 2: .*atlantic_seed3.json: h9_csv .* abort-eval atlantic 3"):
        ev(root, runner=runner)
    dest = R.abort_eval("atlantic", 3, "H9 CSV damaged after its manifest", root)
    assert "logs/n5_atlantic/runs/atlantic_seed3.json" in json.loads((root / dest / "abort.json").read_text(
        encoding="utf-8"))["moved"]
    n = len(runner.calls)
    mans, runner = ev(root, runner=runner)
    assert [c[c.index("--checkpoint") + 1] for c in runner.calls[n:]] == ["models/n5_atlantic/atlantic/seed3.pt"]
    m = next(m for m in mans if (m["arm"], m["seed"]) == ("atlantic", 3))
    assert m["eval_attempts"] == [f"{dest}/abort.json"]


# ---- condition (a): the tie rule ---------------------------------------------------------------------------------
@pytest.mark.parametrize("native, other", [(14, 16), (13, 15), (14, 13), (13, 14)],
                         ids=["even_even", "odd_odd", "even_native_odd_argmin", "odd_native_even_argmin"])
def test_tie_rule_evaluates_the_checkpoint_train_py_saved(tmp_path, monkeypatch, native, other):
    """train.py saved `native` (logged 8.00); re-validation puts `other` (also 8.00) 5e-5 kt below it, inside the
    tie window. For every seed the evaluated checkpoint is train.py's own seed<s>.pt. odd_odd: the matched grid's
    first minimum alone would have taken 15; the tie rule takes 13. even_native_odd_argmin: the off-grid primary."""
    root = fake_repo(tmp_path)
    flat = {e: 8.2 + 0.01 * e for e in range(1, 26)}
    flat.update({native: 7.9990, other: 8.0010})
    monkeypatch.setitem(globals(), "val_curve", lambda arm, seed: dict(flat) if arm == "control" else
                        {e: 8.0 + 0.02 * (e - 8) ** 2 + 0.001 * seed for e in range(1, 14)})
    reval = {**flat, native: 8.00105}
    mans, runner = go(root, reval=lambda epoch_dir, epochs, train_list: dict(reval))
    ctl = [m for m in mans if m["arm"] == "control"]
    for m in ctl:
        assert (m["selected_epoch"], m["selected_checkpoint"]) == (native, m["native_checkpoint"])
        assert m["tie_rule"] == {"revalidated_argmin": other, "native_epoch": native,
                                 "delta_kt": pytest.approx(5e-5)}
        assert m["odd_grid_near_ties"] == A_REAL.odd_grid_near_ties(reval)  # against the odd-grid minimum
        assert m["off_grid_primary"] is (native % 2 == 0)  # John, 2026-10-19, decision 1
    assert ctl[0]["odd_grid_near_ties"] == ([13] if (native, other) == (13, 15) else [])  # odd-grid minimum is 15
    primary_evals = [c[c.index("--checkpoint") + 1] for c in runner.calls if c[-1].endswith("control_seed0_h9.csv")]
    assert primary_evals == ["models/n5_atlantic/control/seed0.pt"]
    _, report = analyse(root, "analyze_n5_fake_tie")
    assert f"control seeds {list(range(10))}; off-grid primary" in report
    assert (f"off-grid primary (even epoch, the same checkpoint as the sensitivity): {list(range(10))}" in report) \
        == (native % 2 == 0)


def test_odd_grid_near_tie_is_recorded_and_the_first_minimum_stands(tmp_path, monkeypatch):
    """No disagreement with train.py (its pick, 14, is the re-validated argmin), but odd epochs 13 and 15 are within
    TIE_KT of each other. train.py never chose between them, so the pre-registered first minimum over the odd grid
    (15) stands, and the near tie is recorded, verified and reported."""
    root = fake_repo(tmp_path)
    flat = {e: 8.2 + 0.01 * e for e in range(1, 26)}
    flat.update({14: 7.9900, 13: 8.00105, 15: 8.0010})
    monkeypatch.setitem(globals(), "val_curve", lambda arm, seed: dict(flat) if arm == "control" else
                        {e: 8.0 + 0.02 * (e - 8) ** 2 + 0.001 * seed for e in range(1, 14)})
    mans, _ = go(root, reval=lambda epoch_dir, epochs, train_list: dict(flat))
    for m in (m for m in mans if m["arm"] == "control"):
        assert (m["selected_epoch"], m["tie_rule"], m["odd_grid_near_ties"]) == (15, None, [13])
    _, report = analyse(root, "analyze_n5_fake_near")
    assert "Odd-grid near ties (within 0.0001 kt of the odd-grid minimum; the first minimum is kept): {0: [13]," in report


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
    assert out.index("PHASE 1") < out.index("BETWEEN THE PHASES") < out.index("PHASE 2")
    assert "eval:" not in out[:out.index("PHASE 2")]
    for bad in (["abort-run"], ["abort-run", "atlantic", "1"], ["plan", "control", "0"], ["abort-all"],
                ["evaluate", "--reason", "x"], ["abort-eval", "--reason", "x"], ["abort-all", "control", "0", "--reason", "x"]):
        with pytest.raises(SystemExit):
            R.main(bad)


def test_analysis_protected_copy_equals_the_coordinator_block(main_root):
    """hc-adversary item 13 (a): analyze_n5.PROTECTED is a second copy of the protected block; it must equal the
    source of truth, the main checkout's COORDINATOR.md."""
    assert dict(CHECK.load_expected(main_root / "COORDINATOR.md")) == A_REAL.PROTECTED
