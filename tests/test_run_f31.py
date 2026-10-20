"""Tests for scripts/run_f31.py on a synthetic repo. Nothing real is trained, evaluated, downloaded or built.

    .venv\\Scripts\\python.exe -I -B -m pytest -q tests/test_run_f31.py -p no:cacheprovider

train.py and evaluate.py are never executed: every test hands the driver a fake runner that writes train.py's and
evaluate.py's stdout lines and output files, and an autouse fixture makes any real subprocess call fail. The synthetic
repo has copies of scripts/analyze_f31.py (unchanged) and scripts/analyze_n5.py (with the H9 pin, the frame-list
SHA-256 and the protected hashes pointing at synthetic files), so the real guards and the analysis's own checks run on
the driver's output. git is faked; `show` of scripts/run_n5.py, scripts/analyze_n5.py and scripts/train.py returns
their real committed blobs (read once at import), so the three pins are checked against real bytes.

HC_RUN_F31_UNDER_TEST=<path> loads another copy of the driver (reports/v2_survey/f31/run_f31_breakage.py uses it to
show that every guard fails a test when disabled). It is not a PYTHON* variable, so -I does not drop it.

Windows note: the abort tests move files under paths like <base>/repo/logs/f31/aborted/<arm>_seed<s>_attempt1/models/
f31/<arm>/seed<s>_epochs/epoch_NN.pt. Run pytest with a base folder shallower than about 120 characters (for example
--basetemp C:/t/f31), or those moves can exceed MAX_PATH (260) and fail with FileNotFoundError.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
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


R = load("run_f31", Path(os.environ.get("HC_RUN_F31_UNDER_TEST") or CHECKOUT / "scripts" / "run_f31.py"))
A_REAL = load("analyze_f31_real", CHECKOUT / "scripts" / "analyze_f31.py")
N5_REAL = A_REAL.N5
CHECK = load("check_for_run_f31", CHECKOUT / "scripts" / "check.py")
COMMIT = "c0ffee" * 6 + "abcd"
SEEDS = tuple(range(10, 30))
REAL_BLOBS = {rel: subprocess.run(["git", "-C", str(CHECKOUT), "show", f"HEAD:{rel}"], capture_output=True).stdout
              for rel in (R.RUN_N5, R.ANALYZE_N5, R.TRAIN_PY)}


@pytest.fixture(autouse=True)
def default_git_is_fake(monkeypatch):
    """The abort stages resolve git themselves (real_git(root)); in these tests that is the fake git."""
    monkeypatch.setattr(R, "real_git", lambda root: fake_git(root))


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
    for sid, n in N5_REAL.H9.items():
        walk = np.clip(40 + np.cumsum(rng.normal(1.0, 4, n)), 20, 160)
        for i, w in enumerate(walk):
            rows.append({"atcf_id": sid, "storm_name": f"S{sid[2:4]}", "sample_time": f"2022-01-01T{i:05d}",
                         "wind_kt": round(float(w), 3), "is_interpolated": bool(i % 3), "status": "HU",
                         "crop_path": f"{sid}\\crops\\{sid}_{i:04d}.npz"})
    return pd.DataFrame(rows)


BASE = base_frames()
PIN = N5_REAL.h9_pin(BASE.assign(truth=BASE.wind_kt).sort_values("crop_path"))


def val_curve(arm, seed):
    """Unrounded validation MAE per epoch: falls, then a noisy plateau, so picks spread over the run."""
    e_max = A_REAL.EPOCHS[arm][4]
    rng = np.random.default_rng([seed, 7, 0 if arm == "control" else 1])
    return {e: round(float(8.0 + 6.0 * np.exp(-e / (0.08 * e_max)) + rng.normal(0, 0.08)), 6)  # Python floats, as
            for e in range(1, e_max + 1)}  # revalidate returns: np.round differs from "%.2f" at x.xx5


def fake_repo(tmp_path, frame_list_ok=True, protected_drift=False):
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
    src = (CHECKOUT / R.ANALYZE_N5).read_text(encoding="utf-8")
    sha = hashlib.sha256(fl.read_bytes()).hexdigest() if frame_list_ok else "0" * 64
    src, k = re.subn(r"^FRAME_LIST_SHA256 = \S+", f"FRAME_LIST_SHA256 = {sha!r}", src, count=1, flags=re.M)
    assert k == 1
    src = src.replace(N5_REAL.H9_PIN, PIN, 1)
    for rel, h in N5_REAL.PROTECTED.items():
        new = "e" * 64 if protected_drift and rel == "best_model.pt" else prot[rel]
        src = re.sub(rf'("{re.escape(rel)}": )"{h}"', rf'\g<1>"{new}"', src, count=1)
    (root / R.ANALYZE_N5).write_text(src, encoding="utf-8")
    shutil.copy(CHECKOUT / R.ANALYSIS, root / R.ANALYSIS)
    for rel in (A_REAL.N5_REPORT, A_REAL.FROZEN_METRICS):  # read by the analysis in the end-to-end tests
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(CHECKOUT / rel, root / rel)
    return root


def fake_git(root=None, dirty=False, dirty_paths=(), differs=(), fetch_rc=0, head=COMMIT, first_parent=(COMMIT,),
             pushed=True, untracked=(), committed_differs=(), extra_changed=(), blobs=None, branch_files=None,
             unreadable=(), history=(), log_rc=0):
    """git for the synthetic repo. The 40 training records and the deviations file count as committed at HEAD (their
    bytes on disk) unless named in `untracked` or `committed_differs`. differs: (rev, path) pairs or paths."""
    blobs = {**REAL_BLOBS, **(blobs or {})}

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
        if args[:3] == ["status", "--porcelain", "--untracked-files=no"]:
            return SimpleNamespace(stdout=b" M scripts/train.py\n" if dirty else b"", returncode=0)
        if args[:2] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout=b" M x\n" if args[-1] in dirty_paths else b"", returncode=0)
        if args[:2] == ["diff", "--quiet"]:
            bad = (args[2], args[-1]) in differs or args[-1] in differs
            return SimpleNamespace(stdout=b"", returncode=1 if bad else 0)
        if args[:4] == ["diff", "--no-renames", "-z", "--name-only"]:
            names = [R.run_paths(A_REAL, a, s)["record"] for _, a, s in R.schedule(A_REAL)] if args[4] != head else []
            return SimpleNamespace(stdout="\0".join(names + list(extra_changed)).encode() + b"\0", returncode=0)
        if args[:2] == ["log", "--all"]:  # every path ever added under logs/f31 and results/f31 (default: none)
            assert args == ["log", "--all", "--diff-filter=A", "--name-only", "-z", "--format=", "--", "logs/f31",
                            "results/f31"]
            return SimpleNamespace(stdout="\0".join(history).encode() + (b"\0" if history else b""),
                                   returncode=log_rc)
        if args[:4] == ["ls-tree", "-r", "-z", "--name-only"]:  # the branch trees looked() reads (default: empty)
            ref = args[4]
            assert ref in ("origin/phase2-experiments", "phase2-experiments")
            if ref in unreadable:
                return SimpleNamespace(stdout=b"", returncode=128)
            return SimpleNamespace(stdout="\0".join((branch_files or {}).get(ref, ())).encode(), returncode=0)
        if args[0] == "ls-files":
            return SimpleNamespace(stdout=b"", returncode=1 if args[-1] in untracked else 0)
        if args[0] == "show":
            rel = args[1].split(":", 1)[1]
            if rel.endswith((".train.json", "deviations.json")):
                data = (root / rel).read_bytes() if (root / rel).exists() else b""
                return SimpleNamespace(stdout=data + (b" " if rel in committed_differs else b""), returncode=0)
            return SimpleNamespace(stdout=blobs.get(rel, b"blob bytes\n"), returncode=0)
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=head.encode() + b"\n", returncode=0)
        if args[0] == "rev-parse":
            return SimpleNamespace(stdout=b"1234abcd\n", returncode=0)
        raise AssertionError(args)
    return git


class FakeRunner:
    """Stands in for subprocess.run on train.py and evaluate.py: writes their stdout lines and output files."""

    def __init__(self, root, train_rc=0, on_train=None, fail=None, curve=None):
        self.root, self.train_rc, self.on_train, self.fail, self.calls = root, train_rc, on_train, fail, []
        self.curve = curve or val_curve
        self.eval_fail = set()  # --out paths whose evaluate.py "crashes" (rc 1, nothing written)
        self.eval_bad_log = set()  # --out paths whose stdout logs an MAE that is not the CSV's

    def __call__(self, cmd, stdout, stderr, cwd, env):
        assert Path(cwd) == self.root and env["PYTHONUTF8"] == "1" and stderr is subprocess.STDOUT
        self.calls.append(cmd)
        is_train = cmd[2] == "scripts/train.py"
        args = dict(zip(cmd[3::2], cmd[4::2])) if is_train else dict(zip(cmd[2::2], cmd[3::2]))
        rc = self.train_rc if is_train else 0
        if not is_train and args["--out"] in self.eval_fail:
            stdout.write(b"Traceback (most recent call last):\n")
            return SimpleNamespace(returncode=1)
        if is_train and self.fail == (self.arm_of(args), int(args["--seed"])):
            rc = 1
        stdout.write((self.train(args, rc) if is_train else self.evaluate(args)).encode("utf-8"))
        return SimpleNamespace(returncode=rc)

    @staticmethod
    def arm_of(a):
        return "atlantic" if a["--train-list"] == "storms_train_n5_atlantic.txt" else "control"

    def train(self, a, rc):
        assert a["--val-list"] == "storms_val_expanded.txt" and a["--loss"] == "l1" and "--epoch-dir" in a
        arm = self.arm_of(a)
        seed, epochs = int(a["--seed"]), int(a["--epochs"])
        ids = N5_REAL.ARM_IDS if arm == "atlantic" else N5_REAL.CONTROL_IDS
        frames = A_REAL.FRAMES[arm]
        lines = ["device: cpu", f"train storms: {ids}", f"val storms:   {N5_REAL.VAL_IDS}",
                 f"seed: {seed}  checkpoint: {a['--out']}  loss: l1  circular_mask: False  rotate: False"]
        if arm == "atlantic":
            lines.append(f"frame list: {a['--frame-list']}: {frames} frames from 50 storms")
        lines.append(f"train samples: {frames}  val samples: {N5_REAL.VAL_FRAMES}")
        if self.on_train:
            self.on_train(arm, seed)
        if rc:
            return "\n".join(lines + ["Traceback (most recent call last):"]) + "\n"
        val, best = self.curve(arm, seed), np.inf
        (self.root / a["--epoch-dir"]).mkdir(parents=True, exist_ok=True)
        for e in range(1, epochs + 1):
            weights = f"weights {arm} {seed} {e}".encode()
            lines.append(f"epoch {e}: train {10 - e * 0.05:.2f} kt  val {val[e]:.2f} kt")
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
        mae = 99.99 if a["--out"] in self.eval_bad_log else d["abs_error"].mean()
        return (f"checkpoint: {a['--checkpoint']}\nheld-out storms: {list(N5_REAL.H9)}\nheld-out samples: {len(d)}\n"
                f"overall MAE:      {mae:.2f} kt\n\nwrote {a['--out']}  ({len(d)} rows)\n")


def reval_from_curve(epoch_dir, epochs, train_list):
    m = re.search(r"f31/(control|atlantic)/seed(\d+)_epochs", Path(epoch_dir).as_posix())
    arm, seed = m.group(1), int(m.group(2))
    assert train_list == N5_REAL.TRAIN_LISTS[arm] and epochs == A_REAL.EPOCHS[arm][4]
    assert all((Path(epoch_dir) / f"epoch_{e:02d}.pt").exists() for e in range(1, epochs + 1))
    return val_curve(arm, seed)


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


def ev(root, git=None, runner=None, pkg=pkg_in_src, deviations=None, cuda=True):
    """Phase 2 (evaluate). Returns (manifests, runner)."""
    runner = runner or FakeRunner(root)
    return R.evaluate(root, deviations=deviations, git=git or fake_git(root), main_root=root, check=CHECK,
                      runner=runner, cuda_ok=lambda: cuda, child_import=pkg), runner


def is_train(cmd):
    return cmd[2] == "scripts/train.py"


def train_calls(runner):
    return [(FakeRunner.arm_of(dict(zip(c[3::2], c[4::2]))), int(c[c.index("--seed") + 1])) for c in runner.calls
            if is_train(c)]


def analysis_of(root, name):
    return load(name, root / R.ANALYSIS)


def analyse(root, name, deviations=None, out="results/f31/analysis"):
    A = analysis_of(root, name)
    design = A.design_constants(ci_grid=101, power_reps=32)
    res = A.run(root / "logs/f31/runs", root / out, root=root, design=design, git=fake_git(root),
                deviations=deviations)
    return A, res, (root / out / "report.txt").read_text(encoding="utf-8")


def record(root, arm, seed):
    return json.loads((root / f"logs/f31/runs/{arm}_seed{seed}.train.json").read_text(encoding="utf-8"))


def touch_kept(path, data):
    """Bit rot or a deliberate edit of a read-only kept file: a plain write is refused first."""
    with pytest.raises(PermissionError):
        path.write_bytes(data)
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    path.write_bytes(data)


@pytest.fixture(scope="module")
def trained_base(tmp_path_factory):
    """One phase-1-complete synthetic repo, built once; tests copy it."""
    root = fake_repo(tmp_path_factory.mktemp("f31base"))
    train(root)
    return root


@pytest.fixture
def trained(tmp_path, trained_base):
    dst = tmp_path / "repo"
    shutil.copytree(trained_base, dst)
    return dst


ORDER = {(a, s): k for k, a, s in R.schedule(A_REAL)}
RECORD_KEYS = {"arm", "seed", "order", "commit", "train_list", "frame_list_sha256", "train_py_sha256",
               "native_same_weights", "train_log", "native_checkpoint", "epoch_dir", "val_mae_revalidated", "slots",
               "tie_rule", "near_ties", "files_sha256", "deleted_epoch_sha256", "protected_before", "protected_after",
               "aborted_attempts", "eval_attempts"}


# ---- commands, order, paths -----------------------------------------------------------------------------------
def test_commands_are_the_design_argument_lists():
    assert R.train_cmd("PY", A_REAL, "control", 13) == [
        "PY", "-u", "scripts/train.py", "--train-list", "storms_train_expanded.txt", "--val-list",
        "storms_val_expanded.txt", "--epochs", "99", "--seed", "13", "--loss", "l1", "--out",
        "models/f31/control/seed13.pt", "--epoch-dir", "models/f31/control/seed13_epochs"]
    assert R.train_cmd("PY", A_REAL, "atlantic", 13) == [
        "PY", "-u", "scripts/train.py", "--train-list", "storms_train_n5_atlantic.txt", "--val-list",
        "storms_val_expanded.txt", "--epochs", "52", "--seed", "13", "--loss", "l1", "--frame-list",
        "results/n5_atlantic/frame_list_atlantic.txt", "--out", "models/f31/atlantic/seed13.pt", "--epoch-dir",
        "models/f31/atlantic/seed13_epochs"]
    ep = A_REAL.epoch_paths("control", 13, 7)
    assert R.eval_cmd("PY", ep["checkpoint"], "storms_train_expanded.txt", ep["csv"]) == [
        "PY", "scripts/evaluate.py", "--checkpoint", "models/f31/control/seed13_epochs/epoch_07.pt", "--train-list",
        "storms_train_expanded.txt", "--holdout-list", "storms_holdout_h9.txt", "--out",
        "results/f31/control_seed13_e07_h9.csv"]
    for arm in A_REAL.ARMS:  # every path the driver writes is the analysis's pinned path
        assert {k: v for k, v in R.run_paths(A_REAL, arm, 13).items() if k in A_REAL.pinned_paths(arm, 13)} == \
            A_REAL.pinned_paths(arm, 13)


def test_schedule_is_the_analysis_order():
    order = R.schedule(A_REAL)
    assert [k for k, _, _ in order] == list(range(1, 41))
    assert [(a[0], s) for _, a, s in order[:6]] == [("c", 10), ("a", 10), ("a", 11), ("c", 11), ("c", 12), ("a", 12)]
    assert all(k == A_REAL.expected_order(a, s) for k, a, s in order)


def test_output_paths_are_confined():
    R.check_output_paths(R.planned_outputs(A_REAL))
    for bad in ("models/storms/expanded/seed0.pt", "best_model.pt", "results/storms/x.csv", "site/x",
                "results/heldout_predictions.csv", "models/other/seed0.pt", "models/n5_atlantic/control/seed0.pt",
                "results/n5_atlantic/x.csv", "logs/n5_atlantic/x.log", "results/ensemble/x", "results/final_2025/x",
                "data/processed/x", "data/interim/hurdat2_atlantic.txt", "results/hep/x", "models/baseline_9kt.pt"):
        with pytest.raises(R.LaunchError, match="refusing output paths"):
            R.check_output_paths([bad])
    for rel in CHECK.PROTECTED_PATHS:  # none of the 14 protected files could ever be an output
        with pytest.raises(R.LaunchError, match="refusing output paths"):
            R.check_output_paths([rel])


# ---- guards -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw, match", [
    (dict(dirty=True), "guard 1: the tracked tree"),
    (dict(differs=("scripts/analyze_f31.py",)), "guard 1: scripts/analyze_f31.py differs"),
    (dict(differs=("scripts/analyze_n5.py",)), "guard 1: scripts/analyze_n5.py differs"),
    (dict(differs=("scripts/run_n5.py",)), "guard 1: scripts/run_n5.py differs"),
    (dict(differs=("scripts/train.py",)), "guard 1: scripts/train.py differs"),
    (dict(blobs={"scripts/run_n5.py": b"edited\n"}), "guard 1: HEAD's scripts/run_n5.py is not the pinned blob"),
    (dict(blobs={"scripts/analyze_n5.py": b"edited\n"}), "guard 1: HEAD's scripts/analyze_n5.py is not"),
    (dict(blobs={"scripts/train.py": b"lr = schedule(epochs)\n"}), "guard 2: HEAD's scripts/train.py is not the pinned"),
], ids=["dirty", "analysis", "analyze_n5", "run_n5", "train_py", "run_n5_blob", "analyze_n5_blob", "train_py_blob"])
def test_refuses_an_uncommitted_or_unpinned_tree(tmp_path, kw, match):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match=match):
        train(root, git=fake_git(root, **kw))
    with pytest.raises(R.LaunchError, match=match):
        ev(root, git=fake_git(root, **kw))
    assert not (root / "logs/f31/control_seed10.log").exists()


def test_refuses_a_frame_list_with_another_sha(tmp_path):
    with pytest.raises(R.LaunchError, match="guard 2: .*frame_list_atlantic.txt"):
        train(fake_repo(tmp_path, frame_list_ok=False))


def test_refuses_an_analysis_protected_copy_that_drifted(tmp_path):
    with pytest.raises(R.LaunchError, match="guard 3: analyze_f31's PROTECTED differs"):
        train(fake_repo(tmp_path, protected_drift=True))


def test_refuses_a_changed_protected_file(tmp_path):
    root = fake_repo(tmp_path)
    (root / "best_model.pt").write_bytes(b"changed")
    with pytest.raises(R.LaunchError, match=r"guard 3: protected files differ: \['best_model.pt'\]"):
        train(root)
    with pytest.raises(R.LaunchError, match=r"guard 3: protected files differ: \['best_model.pt'\]"):
        ev(root)


def test_refuses_any_existing_output(tmp_path):
    root = fake_repo(tmp_path)
    (root / "logs/f31/runs").mkdir(parents=True)
    (root / "logs/f31/runs/atlantic_seed17.json").write_text("{}")
    with pytest.raises(R.LaunchError, match=r"guard 4: unfinished runs have files on disk: \[\('atlantic', 17\)\]"):
        train(root)


@pytest.mark.parametrize("kw, match", [(dict(fetch_rc=1), "guard 5: git fetch origin failed"),
                                       (dict(first_parent=("e" * 40,)), "guard 5: .* is not on the first-parent chain")],
                         ids=["fetch_fails", "not_first_parent"])
def test_refuses_a_commit_not_on_the_first_parent_chain(tmp_path, kw, match):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match=match):
        train(root, git=fake_git(root, **kw))
    assert not (root / "logs/f31/control_seed10.log").exists()


def test_refuses_a_worktree_or_another_checkout(tmp_path):
    root = fake_repo(tmp_path / "a")
    (root / ".git").rmdir()
    (root / ".git").write_text("gitdir: C:/elsewhere/.git/worktrees/wt\n")
    with pytest.raises(R.LaunchError, match="guard 6: .* is not the main checkout"):
        train(root)
    other = fake_repo(tmp_path / "b")
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
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match=r"guard 6: torch.cuda.is_available\(\) is False"):
        train(root, cuda=False)
    with pytest.raises(R.LaunchError, match=r"guard 6: torch.cuda.is_available\(\) is False"):
        ev(root, cuda=False)


def test_a_held_lock_refuses_every_writing_stage(tmp_path):
    root = fake_repo(tmp_path)
    (root / "logs/f31").mkdir(parents=True)
    (root / R.LOCK).write_text("pid 1\n")
    for stage in (lambda: train(root), lambda: ev(root), lambda: R.abort_run("control", 10, "x", root),
                  lambda: R.abort_all("x", root), lambda: R.abort_eval("control", 10, "x", root)):
        with pytest.raises(R.LaunchError, match="run_f31.lock exists"):
            stage()
    assert (root / R.LOCK).read_text() == "pid 1\n"


# ---- the runs -----------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_end_to_end_records_manifests_and_the_analysis_accepts_them(tmp_path, capsys):
    root = fake_repo(tmp_path)
    runner = train(root)
    out = capsys.readouterr().out
    assert "PHASE 1 DONE" in out and all(R.run_paths(A_REAL, a, s)["record"] in out for _, a, s in R.schedule(A_REAL))
    assert all(is_train(c) for c in runner.calls) and len(runner.calls) == 40 and not (root / R.LOCK).exists()
    for _, arm, s in R.schedule(A_REAL):
        rec = record(root, arm, s)
        assert set(rec) == RECORD_KEYS  # exactly the analysis's contract
        assert rec["order"] == A_REAL.expected_order(arm, s) and rec["train_py_sha256"] == A_REAL.TRAIN_PY_SHA256
        assert set(rec["near_ties"]) == ({"matched_4x", "smoothed_4x"} if arm == "control" else {"smoothed_4x"})
        assert set(rec["tie_rule"]) == {"1x", "2x", "4x"} and set(rec["slots"]) == set(A_REAL.SLOTS[arm])
        kept = sorted(set(rec["slots"].values()))
        on_disk = sorted(int(f.name[6:8]) for f in (root / rec["epoch_dir"]).iterdir())
        assert on_disk == kept  # only the slot epochs' files remain
        assert sorted(map(int, rec["deleted_epoch_sha256"])) == [e for e in range(1, A_REAL.EPOCHS[arm][4] + 1)
                                                                 if e not in kept]
        assert rec["deleted_epoch_sha256"][str(min(set(range(1, 99)) - set(kept)))] == hashlib.sha256(
            f"weights {arm} {s} {min(set(range(1, 99)) - set(kept))}".encode()).hexdigest()
        for f in rec["files_sha256"]:
            assert not os.access(root / f, os.W_OK), f
        log = (root / rec["train_log"]).read_bytes()
        assert log.startswith(f"RUN arm={arm} seed={s} order={rec['order']} ".encode()) and b"\r" not in log
        assert re.search(rb"^END ", log, re.M) and b"TRAIN_FAILED" not in log
    mans, runner = ev(root, runner=runner)
    evals = [c for c in runner.calls if not is_train(c)]
    assert all("_epochs/epoch_" in c[c.index("--checkpoint") + 1] for c in evals)  # never seed<s>.pt
    assert len(evals) == sum(len(set(m["slots"].values())) for m in mans) <= 180
    for m in mans:
        rec = record(root, m["arm"], m["seed"])
        assert set(m) - set(rec) == {"evals", "protected_after_eval"} and {k: m[k] for k in rec} == rec
        assert m["protected_after_eval"] == rec["protected_after"] and m["eval_attempts"] == []
    _, res, report = analyse(root, "analyze_f31_e2e")
    assert len(res["primary"]) == 2 and "PRIMARIES" in report and len(res["selection"]) == 20 * 5 + 20 * 4


def dip_curve(arm, seed):
    """Control seed 10: epochs 30 and 31 both log 7.00, and train.py saves 30 (7.004 < 7.00401) as the run's minimum."""
    v = val_curve(arm, seed)
    if (arm, seed) == ("control", 10):
        v.update({30: 7.004, 31: 7.00401})
    return v


def dip_reval(shift):
    """Re-validation that reproduces training except that control seed 10's epoch 31 comes back `shift` below 30."""
    def reval(epoch_dir, epochs, train_list):
        m = re.search(r"f31/(control|atlantic)/seed(\d+)_epochs", Path(epoch_dir).as_posix())
        v = dip_curve(m.group(1), int(m.group(2)))
        if (m.group(1), m.group(2)) == ("control", "10"):
            v[31] = 7.004 - shift
        return v
    return reval


def test_revalidation_that_misses_train_py_pick_stops_and_deletes_nothing(tmp_path):
    root = fake_repo(tmp_path)  # 31 at 7.002: still logs 7.00, but 0.002 kt below train.py's pick, outside TIE_KT
    with pytest.raises(R.LaunchError, match="control seed 10: .*does not reproduce training.*nothing deleted"):
        train(root, runner=FakeRunner(root, curve=dip_curve), reval=dip_reval(0.002))
    assert len(list((root / "models/f31/control/seed10_epochs").iterdir())) == 99
    assert not (root / "logs/f31/runs/control_seed10.train.json").exists()


@pytest.mark.slow
def test_revalidation_below_tie_kt_keeps_train_py_pick(tmp_path):
    root = fake_repo(tmp_path)
    train(root, runner=FakeRunner(root, curve=dip_curve), reval=dip_reval(0.5e-4))
    rec = record(root, "control", 10)
    assert rec["slots"]["native_2x"] == rec["slots"]["native_4x"] == 30  # train.py's pick, not the re-validated 31
    assert rec["tie_rule"]["4x"]["revalidated_argmin"] == 31 and rec["tie_rule"]["2x"]["saved_epoch"] == 30
    assert rec["tie_rule"]["1x"] is None
    mans, _ = ev(root)
    assert next(m for m in mans if (m["arm"], m["seed"]) == ("control", 10))["evals"].get("30")


def test_revalidation_that_does_not_round_to_the_log_stops(tmp_path):
    def drift(epoch_dir, epochs, train_list):
        return {e: v + 0.02 for e, v in reval_from_curve(epoch_dir, epochs, train_list).items()}
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="do not round to the logged ones"):
        train(root, reval=drift)
    assert len(list((root / "models/f31/control/seed10_epochs").iterdir())) == 99


def test_native_checkpoint_must_hold_the_native_4x_epoch_weights(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match=r"seed10.pt does not hold .*epoch_\d\d.pt's weights"):
        train(root, same=lambda a, b: False)
    assert len(list((root / "models/f31/control/seed10_epochs").iterdir())) == 99
    assert not (root / "logs/f31/runs/control_seed10.train.json").exists()


def test_a_missing_epoch_file_stops_before_any_delete(tmp_path):
    root = fake_repo(tmp_path)
    runner = FakeRunner(root)
    orig = runner.train

    def train_then_lose(a, rc):
        out = orig(a, rc)
        (root / a["--epoch-dir"] / "epoch_40.pt").unlink()
        return out

    def never(epoch_dir, epochs, train_list):
        raise AssertionError("re-validation must not run")

    runner.train = train_then_lose
    with pytest.raises(R.LaunchError, match="epoch files missing: .*epoch_40.pt"):
        train(root, runner=runner, reval=never)
    assert len(list((root / "models/f31/control/seed10_epochs").iterdir())) == 98


@pytest.mark.skipif(not hasattr(__import__("_winapi"), "CreateJunction") if sys.platform == "win32" else True,
                    reason="directory junctions are a Windows feature")
def test_epoch_dir_resolving_outside_the_repo_is_refused(tmp_path):
    import _winapi
    root = fake_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    def junction(arm, seed):
        if (arm, seed) == ("control", 10):
            (root / "models/f31/control").mkdir(parents=True, exist_ok=True)
            _winapi.CreateJunction(str(outside), str(root / "models/f31/control/seed10_epochs"))

    with pytest.raises(R.LaunchError, match="seed10_epochs resolves to .*outside"):
        train(root, runner=FakeRunner(root, on_train=junction))
    assert len(list(outside.iterdir())) == 99  # nothing deleted through the junction


def test_training_failure_is_logged_and_stops(tmp_path):
    root = fake_repo(tmp_path)
    runner = FakeRunner(root, train_rc=1)
    with pytest.raises(R.LaunchError, match="training failed: logs/f31/control_seed10.log"):
        train(root, runner=runner)
    log = (root / "logs/f31/control_seed10.log").read_text(encoding="utf-8")
    assert log.startswith("RUN arm=control seed=10 order=1 ") and "\nTRAIN_FAILED rc=1 " in log and "\nEND " not in log
    assert len(runner.calls) == 1 and not (root / R.LOCK).exists()


def test_protected_file_changed_during_a_run_stops(tmp_path):
    root = fake_repo(tmp_path)

    def tamper(arm, seed):
        if (arm, seed) == ("atlantic", 10):
            (root / "results/heldout_predictions.csv").write_bytes(b"overwritten")

    with pytest.raises(R.LaunchError, match=r"protected hashes changed during atlantic seed 10: "
                                            r"\['results/heldout_predictions.csv'\]"):
        train(root, runner=FakeRunner(root, on_train=tamper))
    assert not (root / "logs/f31/runs/atlantic_seed10.train.json").exists()


def test_protected_file_changed_during_revalidation_stops(tmp_path):
    root = fake_repo(tmp_path)

    def tamper(epoch_dir, epochs, train_list):
        if "atlantic/seed10_" in Path(epoch_dir).as_posix():
            (root / "best_model.pt").write_bytes(b"overwritten")
        return reval_from_curve(epoch_dir, epochs, train_list)

    with pytest.raises(R.LaunchError, match=r"protected hashes changed during atlantic seed 10 \(re-validation or "
                                            r"deletion\): \['best_model.pt'\]"):
        train(root, reval=tamper)
    assert not (root / "logs/f31/runs/atlantic_seed10.train.json").exists()


def test_protected_drift_before_a_run_stops_it_unstarted(tmp_path, monkeypatch):
    root = fake_repo(tmp_path)
    real, n = R.protected_now, []

    def drift_second_call(main_root, paths):  # call 1 is guard 3; call 2 is the first run's before-run check
        n.append(1)
        got = real(main_root, paths)
        return {**got, "best_model.pt": "0" * 64} if len(n) == 2 else got

    monkeypatch.setattr(R, "protected_now", drift_second_call)
    runner = FakeRunner(root)
    with pytest.raises(R.LaunchError, match="protected hashes differ before control seed 10"):
        train(root, runner=runner)
    assert runner.calls == [] and not (root / "logs/f31/control_seed10.log").exists()


def test_driver_refuses_an_output_that_appears_mid_launch(tmp_path):
    root = fake_repo(tmp_path)

    def plant(arm, seed):
        if (arm, seed) == ("control", 10):
            (root / "models/f31/atlantic/seed10_epochs").mkdir(parents=True)

    with pytest.raises(R.LaunchError, match="refusing to overwrite .*atlantic/seed10_epochs"):
        train(root, runner=FakeRunner(root, on_train=plant))


# ---- the phase split ---------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("fail", [("atlantic", 11), ("control", 29)], ids=["order3", "order40"])
def test_phase2_refuses_while_any_training_run_is_unfinished(tmp_path, fail):
    root = fake_repo(tmp_path)
    runner = FakeRunner(root, fail=fail)
    with pytest.raises(R.LaunchError, match="training failed"):
        train(root, runner=runner)
    with pytest.raises(R.LaunchError, match=r"phase 2: H9 evaluation refuses to start until all 40 training runs have "
                                            rf"finished; unfinished: \[\({ORDER[fail]}, '{fail[0]}', {fail[1]}\)"):
        ev(root, runner=runner)
    assert all(is_train(c) for c in runner.calls) and R.h9_outputs(root, A_REAL) == []


@pytest.mark.slow
@pytest.mark.parametrize("kw, match", [
    (dict(untracked=("logs/f31/runs/atlantic_seed14.train.json",)), "guard 5e: .*atlantic_seed14.train.json is not tracked"),
    (dict(committed_differs=("logs/f31/runs/control_seed12.train.json",)),
     "guard 5e: .*control_seed12.train.json differs from its committed blob"),
    (dict(head="d" * 40, first_parent=(COMMIT,), pushed=False), "guard 5e: HEAD is not pushed"),
    (dict(head="d" * 40, first_parent=("d" * 40,)), f"guard 5: {COMMIT} is not on the first-parent chain"),
    (dict(head="d" * 40, first_parent=(COMMIT,), extra_changed=("scripts/train.py",)),
     r"guard 5e: HEAD differs from the launch commit .* in more than the 40 records, the decisions docs and the "
     r"deviations file: \['scripts/train.py'\]"),
], ids=["untracked", "differs", "not_pushed", "launch_not_first_parent", "code_changed"])
def test_phase2_requires_the_committed_records_as_evidence(trained, kw, match):
    with pytest.raises(R.LaunchError, match=match):
        ev(trained, git=fake_git(trained, **kw))
    assert R.h9_outputs(trained, A_REAL) == [] and not (trained / R.PHASE2_MARKER).exists()


@pytest.mark.slow
def test_phase2_accepts_records_one_commit_later_with_docs(trained):
    git = fake_git(trained, head="d" * 40, first_parent=(COMMIT, "d" * 40),
                   extra_changed=("logs/decisions.md", "reports/decisions/F31.md"))
    mans, _ = ev(trained, git=git)
    assert len(mans) == 40 and {m["commit"] for m in mans} == {COMMIT}


@pytest.mark.slow
@pytest.mark.parametrize("edit, match", [
    (lambda r: r.update(slots={**r["slots"], "native_4x": 1}), "validation-only selections"),
    (lambda r: r.update(train_py_sha256="0" * 64), "train.py blob"),
    (lambda r: r.update(near_ties={**r["near_ties"], "smoothed_4x": [1]}), "near_ties"),
    (lambda r: r.update(order=40), "schedule"),
    (lambda r: r.update(commit="f" * 40), "more than one commit"),
    (lambda r: r.update(deleted_epoch_sha256={}), "deleted_epoch_sha256"),
], ids=["slots", "train_py", "near_ties", "order", "two_commits", "deleted"])
def test_phase2_entry_applies_the_training_side_checks_before_any_h9(trained, edit, match):
    p = trained / "logs/f31/runs/control_seed12.train.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    edit(rec)
    p.write_text(json.dumps(rec), encoding="utf-8")
    runner = FakeRunner(trained)
    with pytest.raises(R.LaunchError, match=match):
        ev(trained, runner=runner)
    assert runner.calls == [] and R.h9_outputs(trained, A_REAL) == []


@pytest.mark.slow
@pytest.mark.parametrize("kw", [dict(differs=((COMMIT, "scripts/analyze_f31.py"),)),
                                dict(blobs={f"scripts/train.py": REAL_BLOBS["scripts/train.py"]},
                                     differs=((COMMIT, "scripts/analyze_n5.py"),))], ids=["analysis", "analyze_n5"])
def test_phase2_entry_runs_verify_self_at_the_launch_commit(trained, kw):
    runner = FakeRunner(trained)
    with pytest.raises(R.LaunchError, match="phase 2 entry, before this invocation's first H9 evaluation: .*differs from the launch commit"):
        ev(trained, git=fake_git(trained, **kw), runner=runner)
    assert runner.calls == []


@pytest.mark.slow
def test_relaunch_at_another_commit_is_refused_until_abort_all(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed"):
        train(root, runner=FakeRunner(root, fail=("atlantic", 11)))
    R.abort_run("atlantic", 11, "crash; fixed in a follow-up PR", root)
    fix = fake_git(root, head="f" * 40, first_parent=(COMMIT, "f" * 40))
    runner = FakeRunner(root)
    with pytest.raises(R.LaunchError, match="guard 4: runs recorded at another commit .* abort-all"):
        train(root, git=fix, runner=runner)
    assert runner.calls == []
    moved = R.abort_all("relaunch from the fix commit", root)
    assert sorted(Path(m).name for m in moved) == ["atlantic_seed10_attempt1", "control_seed10_attempt1"]
    runner = train(root, git=fix)
    assert train_calls(runner) == [(a, s) for _, a, s in R.schedule(A_REAL)]
    assert {record(root, a, s)["commit"] for _, a, s in R.schedule(A_REAL)} == {"f" * 40}
    assert len(record(root, "atlantic", 11)["aborted_attempts"]) == 1


@pytest.mark.slow
def test_abort_all_refuses_after_an_h9_output(trained):
    (trained / "results/f31/control_seed10_e05_h9.csv").write_text("looked")
    with pytest.raises(R.LaunchError, match="abort-all: H9 has been looked at"):
        R.abort_all("after a look", trained)
    assert (trained / "logs/f31/runs/control_seed10.train.json").exists()


def test_training_never_follows_an_h9_output(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed"):
        train(root, runner=FakeRunner(root, fail=("atlantic", 11)))
    R.abort_run("atlantic", 11, "planted failure", root)
    (root / "results/f31/control_seed10_e05_h9.csv").write_text("looked")
    with pytest.raises(R.LaunchError, match=r"guard 4: H9 has been looked at while runs remain to be trained"):
        train(root)


def test_an_h9_output_planted_mid_phase1_stops_training(tmp_path):
    root = fake_repo(tmp_path)

    def plant(arm, seed):
        if (arm, seed) == ("atlantic", 10):  # order 2; order 1 (control seed 10) has its record by now
            (root / "results/f31/control_seed10_e05_h9.csv").write_text("looked")

    runner = FakeRunner(root, on_train=plant)
    with pytest.raises(R.LaunchError, match="phase 1: H9 has been looked at"):
        train(root, runner=runner)
    assert train_calls(runner) == [("control", 10), ("atlantic", 10)]


# ---- aborts -------------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_abort_then_restart_keeps_seed_and_order_and_records_the_attempt(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed: logs/f31/atlantic_seed11.log"):
        train(root, runner=FakeRunner(root, fail=("atlantic", 11)))
    with pytest.raises(R.LaunchError, match=r"guard 4: unfinished runs have files on disk: \[\('atlantic', 11\)\].*abort-run"):
        train(root)
    failed = (root / "logs/f31/atlantic_seed11.log").read_bytes()
    dest = R.abort_run("atlantic", 11, "planted training failure", root)
    assert dest == "logs/f31/aborted/atlantic_seed11_attempt1"
    ab = json.loads((root / dest / "abort.json").read_text(encoding="utf-8"))
    assert (ab["arm"], ab["seed"], ab["order"], ab["attempt"], ab["status"]) == ("atlantic", 11, 3, 1, "complete")
    assert ab["moved"]["logs/f31/atlantic_seed11.log"] == {
        "to": f"{dest}/logs/f31/atlantic_seed11.log", "sha256": hashlib.sha256(failed).hexdigest()}
    runner = train(root)
    assert train_calls(runner) == [(a, s) for k, a, s in R.schedule(A_REAL) if k >= 3]
    rec = record(root, "atlantic", 11)
    assert (rec["order"], rec["aborted_attempts"]) == (3, [f"{dest}/abort.json"])
    assert (root / rec["train_log"]).read_bytes().startswith(b"RUN arm=atlantic seed=11 order=3 ")


def test_abort_run_refusals(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="no files to move"):
        R.abort_run("control", 10, "nothing there", root)
    with pytest.raises(R.LaunchError, match="unknown run"):
        R.abort_run("control", 3, "an N5 seed", root)
    with pytest.raises(R.LaunchError, match="training failed"):
        train(root, runner=FakeRunner(root, fail=("atlantic", 11)))
    with pytest.raises(R.LaunchError, match="has a training record that verifies; finished runs are not aborted"):
        R.abort_run("control", 10, "finished", root)
    (root / "results/f31/control_seed10_e05_h9.csv").write_text("looked")
    with pytest.raises(R.LaunchError, match="no run may be aborted after an H9 evaluation"):
        R.abort_run("atlantic", 11, "after a look", root)
    assert (root / "logs/f31/atlantic_seed11.log").exists() and not (root / R.ABORTED).exists()


@pytest.mark.slow
def test_a_record_that_fails_verification_can_be_aborted_and_retrained(trained):
    rec = record(trained, "control", 14)
    kept = next(f for f in rec["files_sha256"] if "_epochs/" in f)
    touch_kept(trained / kept, b"swapped after training")
    with pytest.raises(R.LaunchError, match=r"phase 2: training records that do not verify: \[\('control', 14\)\]"):
        ev(trained)
    with pytest.raises(R.LaunchError, match=r"guard 4: training records that do not verify: \[\('control', 14\)\]"):
        train(trained)
    dest = R.abort_run("control", 14, "kept epoch file changed", trained)
    moved = json.loads((trained / dest / "abort.json").read_text(encoding="utf-8"))["moved"]
    assert "logs/f31/runs/control_seed14.train.json" in moved and len(moved) == len(rec["files_sha256"]) + 1
    runner = train(trained)
    assert train_calls(runner) == [("control", 14)]
    assert len(ev(trained, runner=runner)[0]) == 40


@pytest.mark.slow
def test_kept_files_are_read_only_and_abort_all_still_moves_them(trained):
    rec = record(trained, "control", 12)
    moved = R.abort_all("read-only files", trained)
    assert len(moved) == 40
    dest = trained / "logs/f31/aborted/control_seed12_attempt1"
    for f in rec["files_sha256"]:
        assert not (trained / f).exists() and (dest / f).is_file() and not os.access(dest / f, os.W_OK)


@pytest.mark.slow
def test_after_a_look_a_bad_record_asks_for_the_bytes_back(trained):
    runner = FakeRunner(trained)
    rec = record(trained, "atlantic", 12)
    runner.eval_fail = {A_REAL.epoch_paths("atlantic", 12, rec["slots"]["native_1x"])["csv"]}
    with pytest.raises(R.LaunchError, match="evaluation failed"):
        ev(trained, runner=runner)
    bad = record(trained, "control", 18)
    kept = trained / next(f for f in bad["files_sha256"] if "_epochs/" in f)
    good = kept.read_bytes()
    touch_kept(kept, b"bit rot")
    for stage, match in ((lambda: ev(trained, runner=runner), r"do not verify: \[\('control', 18\)\]; H9 has been "
                                                               r"looked at, so it can't be retrained: restore the exact"),
                         (lambda: train(trained), "phase2_started.json exists")):
        with pytest.raises(R.LaunchError, match=match) as e:
            stage()
        assert "abort-run" not in str(e.value)
    with pytest.raises(R.LaunchError, match="abort-run: H9 has been looked at"):
        R.abort_run("control", 18, "bit rot", trained)
    kept.write_bytes(good)
    runner.eval_fail = set()
    R.abort_eval("atlantic", 12, "evaluation crashed", trained)
    assert len(ev(trained, runner=runner)[0]) == 40


@pytest.mark.slow
def test_a_crashed_evaluation_is_moved_aside_and_the_same_slots_re_evaluated(trained):
    with pytest.raises(R.LaunchError, match="abort-eval: atlantic seed 12 is trained"):
        R.abort_eval("atlantic", 12, "nothing to move", trained)
    rec = record(trained, "atlantic", 12)
    first = min(set(rec["slots"].values()))
    ep = A_REAL.epoch_paths("atlantic", 12, first)
    (trained / ep["csv"]).write_text("left by a crashed evaluation")
    (trained / ep["eval_log"]).write_text("checkpoint: x\noverall MAE: 9.99 kt\n")
    with pytest.raises(R.LaunchError, match=r"phase 2: H9 outputs without a manifest for \[\('atlantic', 12\)\]; .*abort-eval"):
        ev(trained)
    with pytest.raises(R.LaunchError, match="abort-run: H9 has been looked at"):
        R.abort_run("atlantic", 12, "retrain after a look", trained)
    dest = R.abort_eval("atlantic", 12, "evaluation crashed", trained)
    assert dest == "logs/f31/aborted/atlantic_seed12_eval1"
    ab = json.loads((trained / dest / "abort.json").read_text(encoding="utf-8"))
    assert sorted(ab["moved"]) == sorted([ep["eval_log"], ep["csv"]])
    mans, runner = ev(trained)
    m = next(m for m in mans if (m["arm"], m["seed"]) == ("atlantic", 12))
    assert m["eval_attempts"] == [f"{dest}/abort.json"]
    cps = [c[c.index("--checkpoint") + 1] for c in runner.calls if "atlantic_seed12_e" in c[-1]]
    assert cps == [A_REAL.epoch_paths("atlantic", 12, e)["checkpoint"] for e in sorted(set(rec["slots"].values()))]
    _, _, report = analyse(trained, "analyze_f31_eval_abort")
    assert "aborted H9 evaluation attempts: {'atlantic seed 12': [" in report


def test_an_interrupted_abort_is_pending_blocks_launch_and_resumes(tmp_path, monkeypatch):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed"):
        train(root, runner=FakeRunner(root, fail=("atlantic", 11)))
    (root / "models/f31/atlantic/seed11.pt").write_bytes(b"partial checkpoint")
    real, calls = os.rename, []

    def flaky(a, b):
        calls.append(a)
        if len(calls) == 2:
            raise PermissionError(32, "in use", str(a))
        return real(a, b)

    monkeypatch.setattr(R.os, "rename", flaky)
    with pytest.raises(PermissionError):
        R.abort_run("atlantic", 11, "crash", root)
    monkeypatch.setattr(R.os, "rename", real)
    ab = json.loads((root / "logs/f31/aborted/atlantic_seed11_attempt1/abort.json").read_text(encoding="utf-8"))
    assert ab["status"] == "pending" and not (root / R.LOCK).exists()
    with pytest.raises(R.LaunchError, match="guard 4: unfinished aborts .*atlantic_seed11_attempt1"):
        train(root)
    assert R.abort_run("atlantic", 11, "crash", root) == "logs/f31/aborted/atlantic_seed11_attempt1"
    ab = json.loads((root / "logs/f31/aborted/atlantic_seed11_attempt1/abort.json").read_text(encoding="utf-8"))
    assert ab["status"] == "complete" and all((root / v["to"]).is_file() for v in ab["moved"].values())


@pytest.mark.slow
def test_abort_eval_cannot_launder_an_h9_look(trained):
    runner = FakeRunner(trained)
    rec = record(trained, "control", 10)
    epochs = sorted(set(rec["slots"].values()))
    runner.eval_fail = {A_REAL.epoch_paths("control", 10, epochs[-1])["csv"]}
    with pytest.raises(R.LaunchError, match="evaluation failed"):
        ev(trained, runner=runner)
    marker = json.loads((trained / R.PHASE2_MARKER).read_text(encoding="utf-8"))
    assert (marker["head"], marker["launch_commit"]) == (COMMIT, COMMIT)
    R.abort_eval("control", 10, "evaluation crashed", trained)
    assert R.h9_outputs(trained, A_REAL) == []
    for stage, match in ((lambda: R.abort_all("start over", trained), "abort-all: H9 has been looked at"),
                         (lambda: R.abort_run("atlantic", 13, "retrain", trained), "abort-run: H9 has been looked at"),
                         (lambda: train(trained), "guard 4: logs/f31/phase2_started.json exists")):
        with pytest.raises(R.LaunchError, match=match):
            stage()
    runner.eval_fail = set()
    mans, _ = ev(trained, runner=runner)
    assert mans[0]["eval_attempts"] == ["logs/f31/aborted/control_seed10_eval1/abort.json"] and len(mans) == 40


@pytest.mark.slow
def test_the_phase2_marker_alone_is_a_look(trained):
    runner = FakeRunner(trained)
    rec = record(trained, "control", 10)
    runner.eval_fail = {A_REAL.epoch_paths("control", 10, min(set(rec["slots"].values())))["csv"]}
    with pytest.raises(R.LaunchError, match="evaluation failed"):
        ev(trained, runner=runner)
    for rel in R.h9_outputs(trained, A_REAL):
        (trained / rel).unlink()
    assert R.looked(trained, A_REAL) == [R.PHASE2_MARKER]
    with pytest.raises(R.LaunchError, match="abort-all: H9 has been looked at .*phase2_started"):
        R.abort_all("start over", trained)


@pytest.mark.slow
def test_a_pending_eval_abort_blocks_evaluate_and_resumes(trained, monkeypatch):
    runner = FakeRunner(trained)
    rec = record(trained, "control", 10)
    runner.eval_fail = {A_REAL.epoch_paths("control", 10, max(set(rec["slots"].values())))["csv"]}
    with pytest.raises(R.LaunchError, match="evaluation failed"):
        ev(trained, runner=runner)
    runner.eval_fail, calls_before = set(), len(runner.calls)
    real, n = R.write_json_replace, []

    def flaky(path, obj):
        n.append(path)
        if len(n) == 2:
            raise OSError("disk full")
        return real(path, obj)

    monkeypatch.setattr(R, "write_json_replace", flaky)
    with pytest.raises(OSError):
        R.abort_eval("control", 10, "evaluation crashed", trained)
    monkeypatch.setattr(R, "write_json_replace", real)
    with pytest.raises(R.LaunchError, match="guard 4: unfinished aborts .*control_seed10_eval1"):
        ev(trained, runner=runner)
    assert len(runner.calls) == calls_before
    R.abort_eval("control", 10, "evaluation crashed", trained)
    mans, _ = ev(trained, runner=runner)
    assert mans[0]["eval_attempts"] == ["logs/f31/aborted/control_seed10_eval1/abort.json"]


@pytest.mark.slow
def test_evaluate_one_checks_its_eval_attempts_before_the_h9_step(trained, monkeypatch):
    d = trained / "logs/f31/aborted/control_seed10_eval1"
    d.mkdir(parents=True)
    (d / "abort.json").write_text(json.dumps({"status": "pending", "moved": {}}), encoding="utf-8")
    monkeypatch.setattr(R, "pending_aborts", lambda root: [])
    runner = FakeRunner(trained)
    with pytest.raises(R.LaunchError, match="before the H9 evaluation of control seed 10: .*control_seed10_eval1"):
        ev(trained, runner=runner)
    assert runner.calls == []


@pytest.mark.slow
def test_a_manifest_the_analysis_refuses_stops_phase2_at_that_run(trained):
    """The driver applies analyze_f31.load_run to each manifest right after writing it: an eval log whose MAE is not
    its CSV's (log and CSV not from one evaluation) stops phase 2 at that run, with abort-eval as the remedy."""
    runner = FakeRunner(trained)
    rec = record(trained, "atlantic", 11)  # order 3
    runner.eval_bad_log = {A_REAL.epoch_paths("atlantic", 11, rec["slots"]["native_4x"])["csv"]}
    with pytest.raises(R.LaunchError, match="after the H9 evaluation of atlantic seed 11: .*not from the same evaluation"):
        ev(trained, runner=runner)
    assert not (trained / "logs/f31/runs/control_seed11.json").exists()  # order 4 never started
    runner.eval_bad_log = set()
    R.abort_eval("atlantic", 11, "eval log disagreed with its CSV", trained)
    assert len(ev(trained, runner=runner)[0]) == 40


@pytest.mark.slow
def test_protected_file_changed_during_an_h9_step_stops_without_a_manifest(trained):
    runner = FakeRunner(trained)
    rec = record(trained, "atlantic", 10)  # order 2
    target = A_REAL.epoch_paths("atlantic", 10, min(set(rec["slots"].values())))["csv"]
    orig = runner.evaluate

    def evaluate_and_tamper(a):
        out = orig(a)
        if a["--out"] == target:
            (trained / "models/baseline_9kt.pt").write_bytes(b"overwritten")
        return out

    runner.evaluate = evaluate_and_tamper
    with pytest.raises(R.LaunchError, match=r"protected hashes changed during the H9 evaluation of atlantic seed 10 at "
                                            r"epoch \d+: \['models/baseline_9kt.pt'\]; no manifest written"):
        ev(trained, runner=runner)
    assert not (trained / "logs/f31/runs/atlantic_seed10.json").exists()
    assert (trained / "logs/f31/runs/control_seed10.json").exists()  # order 1 was finished before


@pytest.mark.slow
def test_protected_drift_before_an_h9_step_stops_it_unrun(trained, monkeypatch):
    real, n = R.protected_changed, []

    def drift_first_call(main_root, ctx):  # the first call in phase 2 is the before-step check of order 1's first epoch
        n.append(1)
        return ["best_model.pt"] if len(n) == 1 else real(main_root, ctx)

    monkeypatch.setattr(R, "protected_changed", drift_first_call)
    runner = FakeRunner(trained)
    with pytest.raises(R.LaunchError, match=r"protected hashes differ before the H9 evaluation of control seed 10 at "
                                            r"epoch \d+: \['best_model.pt'\]"):
        ev(trained, runner=runner)
    assert runner.calls == []


def test_a_tampered_run_n5_is_refused_before_import(tmp_path):
    d = tmp_path / "scripts"
    d.mkdir()
    shutil.copy(Path(R.__file__), d / "run_f31.py")
    marker = tmp_path / "executed"
    good = (CHECKOUT / "scripts" / "run_n5.py").read_bytes()
    (d / "run_n5.py").write_bytes(good + f"\nopen({str(marker)!r}, 'w').close()\n".encode())
    with pytest.raises(SystemExit, match="not the pinned run_n5.py blob .*refused before import"):  # its own LaunchError
        load("run_f31_tampered", d / "run_f31.py")
    assert not marker.exists()  # never executed
    (d / "run_n5.py").write_bytes(good.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))  # a CRLF working copy
    assert load("run_f31_crlf", d / "run_f31.py").RN5.UPSTREAM == "origin/phase2-experiments"  # CRLF copy loads


def test_run_n5_runs_from_the_hashed_bytes_not_a_file_swapped_after_the_check(tmp_path, monkeypatch):
    """The file is swapped for a tampered one right after it was read and hashed (a concurrent writer): the code that
    runs is still the hashed bytes, so the tampered file never executes."""
    path = tmp_path / "run_n5.py"
    path.write_bytes((CHECKOUT / "scripts" / "run_n5.py").read_bytes())
    marker = tmp_path / "executed"
    tampered = path.read_bytes() + f"\nopen({str(marker)!r}, 'w').close()\n".encode()
    real = R.hashlib.sha256

    def hash_then_swap(data=b""):
        path.write_bytes(tampered)  # the swap, after the read whose bytes are being hashed
        return real(data)

    monkeypatch.setattr(R.hashlib, "sha256", hash_then_swap)
    module = R.load_pinned_run_n5(path)
    monkeypatch.setattr(R.hashlib, "sha256", real)
    assert path.read_bytes() == tampered and not marker.exists()  # swapped on disk, yet never executed
    assert module.UPSTREAM == "origin/phase2-experiments" and module.__file__ == str(path)


@pytest.mark.slow
def test_abort_eval_takes_a_manifest_that_load_run_refuses(trained):
    mans, runner = ev(trained)
    with pytest.raises(R.LaunchError, match="abort-eval: atlantic seed 13 is evaluated"):
        R.abort_eval("atlantic", 13, "valid manifest", trained)
    m = next(m for m in mans if (m["arm"], m["seed"]) == ("atlantic", 13))
    csv = A_REAL.epoch_paths("atlantic", 13, int(min(m["evals"], key=int)))["csv"]
    (trained / csv).write_bytes(b"damaged")
    with pytest.raises(R.LaunchError, match="phase 2: .*atlantic_seed13.json: .* abort-eval atlantic 13"):
        ev(trained, runner=runner)
    dest = R.abort_eval("atlantic", 13, "H9 CSV damaged after its manifest", trained)
    assert "logs/f31/runs/atlantic_seed13.json" in json.loads((trained / dest / "abort.json").read_text(
        encoding="utf-8"))["moved"]
    n = len(runner.calls)
    mans, runner = ev(trained, runner=runner)
    assert [c[c.index("--checkpoint") + 1] for c in runner.calls[n:]] == [
        A_REAL.epoch_paths("atlantic", 13, int(e))["checkpoint"] for e in sorted(m["evals"], key=int)]


# ---- rule (A) in phase 2 ---------------------------------------------------------------------------------------------
def write_deviations(root, rows):
    (root / A_REAL.DEVIATIONS_PATH).write_text(json.dumps({"dropped_runs": rows}), encoding="utf-8")
    return A_REAL.DEVIATIONS_PATH


@pytest.mark.slow
def test_rule_a_completes_when_a_record_breaks_mid_phase2(trained):
    runner = FakeRunner(trained)
    rec = record(trained, "atlantic", 13)  # order 7
    runner.eval_fail = {A_REAL.epoch_paths("atlantic", 13, min(set(rec["slots"].values())))["csv"]}
    with pytest.raises(R.LaunchError, match="evaluation failed"):
        ev(trained, runner=runner)
    assert (trained / "logs/f31/runs/control_seed10.json").exists()
    touch_kept(trained / "models/f31/control/seed10.pt", b"bit rot")
    runner.eval_fail = set()
    with pytest.raises(R.LaunchError, match=r"do not verify: \[\('control', 10\)\]; H9 has been looked at"):
        ev(trained, runner=runner)
    R.abort_eval("atlantic", 13, "evaluation crashed", trained)
    dev = write_deviations(trained, [{"arm": "control", "seed": 10, "reason": "seed10.pt changed after H9; bytes lost"}])
    n = len(runner.calls)
    mans, runner = ev(trained, runner=runner, deviations=dev)
    assert len(mans) == 39 and ("control", 10) not in {(m["arm"], m["seed"]) for m in mans}
    assert not any("control/seed10" in c[c.index("--checkpoint") + 1] for c in runner.calls[n:])
    _, res, report = analyse(trained, "analyze_f31_rule_a", deviations=trained / dev)
    assert res["primary"][0]["n"] == 19 and res["primary"][1]["n"] == 20
    assert res["primary"][0]["verdict"].startswith("NON-CONFIRMATORY") and "DEVIATION (rule (A)" in report


@pytest.mark.slow
def test_rule_a_is_refused_before_any_look(trained):
    touch_kept(trained / "models/f31/control/seed10.pt", b"bit rot")
    dev = write_deviations(trained, [{"arm": "control", "seed": 10, "reason": "seed10.pt changed"}])
    assert R.looked(trained, A_REAL) == []
    runner = FakeRunner(trained)
    with pytest.raises(R.LaunchError, match="no H9 look yet: abort-run and retrain instead"):
        ev(trained, runner=runner, deviations=dev)
    assert runner.calls == [] and not (trained / R.PHASE2_MARKER).exists()


@pytest.mark.slow
@pytest.mark.parametrize("case", ["healthy", "untracked", "differs", "other_path", "restorable"])
def test_evaluate_refuses_a_deviation_rule_a_does_not_cover(trained, case):
    runner = FakeRunner(trained)
    rec = record(trained, "control", 10)
    runner.eval_fail = {A_REAL.epoch_paths("control", 10, min(set(rec["slots"].values())))["csv"]}
    with pytest.raises(R.LaunchError, match="evaluation failed"):
        ev(trained, runner=runner)
    calls = len(runner.calls)
    if case == "restorable":  # a stray non-slot epoch file: deleting it restores the run (analyze_f31 item 23)
        e = next(e for e in range(1, 99) if e not in rec["slots"].values())
        (trained / A_REAL.epoch_paths("control", 12, e)["checkpoint"]).write_bytes(b"stray")
        dev = write_deviations(trained, [{"arm": "control", "seed": 12, "reason": "x"}])
    else:
        touch_kept(trained / "models/f31/control/seed10.pt", b"bit rot")
        dev = write_deviations(trained, [{"arm": "control", "seed": 13 if case == "healthy" else 10, "reason": "x"}])
    git = fake_git(trained, untracked=(dev,) if case == "untracked" else (),
                   committed_differs=(dev,) if case == "differs" else ())
    if case == "other_path":
        (trained / "deviations.json").write_text((trained / dev).read_text(encoding="utf-8"), encoding="utf-8")
        dev = "deviations.json"
    match = {"healthy": "kept files can't be restored", "restorable": "kept files can't be restored",
             "untracked": "must be committed", "differs": "must be committed, and identical to its blob at HEAD",
             "other_path": "--deviations must be logs/f31/deviations.json"}[case]
    with pytest.raises(R.LaunchError, match=match):
        ev(trained, git=git, runner=runner, deviations=dev)
    assert len(runner.calls) == calls


# ---- plan and the CLI -----------------------------------------------------------------------------------------------
def test_plan_runs_nothing_and_the_cli_refuses_bad_arguments(capsys, monkeypatch):
    monkeypatch.chdir(CHECKOUT)
    R.main(["plan"])
    out = capsys.readouterr().out
    assert out.count("order ") == 80 and "'--epochs', '99'" in out and "'--epochs', '52'" in out
    assert out.index("PHASE 1") < out.index("BETWEEN THE PHASES") < out.index("PHASE 2")
    assert "eval:" not in out[:out.index("PHASE 2")] and "epoch_NN.pt" in out and "_eNN_h9.csv" in out
    for bad in (["abort-run"], ["abort-run", "atlantic", "11"], ["plan", "control", "10"], ["abort-all"],
                ["evaluate", "--reason", "x"], ["abort-eval", "--reason", "x"], ["abort-all", "control", "10", "--reason", "x"],
                ["launch", "--deviations", "logs/f31/deviations.json"], ["build-spec"]):
        with pytest.raises(SystemExit):
            R.main(bad)


def test_pins_match_the_committed_blobs():
    assert hashlib.sha256(REAL_BLOBS[R.RUN_N5]).hexdigest() == R.RUN_N5_BLOB_SHA256
    assert hashlib.sha256(REAL_BLOBS[R.ANALYZE_N5]).hexdigest() == A_REAL.N5_BLOB_SHA256
    assert hashlib.sha256(REAL_BLOBS[R.TRAIN_PY]).hexdigest() == A_REAL.TRAIN_PY_SHA256


# ---- look evidence committed on the branch (tightening after PR #25) ---------------------------------------------
RESULTS_ON_BRANCH = {"origin/phase2-experiments": [
    "logs/decisions.md", "logs/f31/runs/control_seed10.train.json", "logs/f31/runs/control_seed10.json",
    "logs/f31/phase2_started.json", "results/f31/control_seed10_e25_h9.csv"]}


@pytest.mark.slow
def test_a_merged_results_pr_is_a_look_even_with_a_clean_disk(trained, monkeypatch):
    """Route (ii) after the results PR: `git checkout --detach L` removes the marker, CSVs and manifests from the
    disk, but the branch still carries them. abort-run, abort-all and launch all refuse."""
    git = fake_git(trained, branch_files=RESULTS_ON_BRANCH)
    monkeypatch.setattr(R, "real_git", lambda root: git)
    assert R.h9_outputs(trained, A_REAL) == [] and not (trained / R.PHASE2_MARKER).exists()  # the disk is clean
    seen = R.looked(trained, A_REAL, git)
    assert seen == ["origin/phase2-experiments:logs/f31/phase2_started.json",
                    "origin/phase2-experiments:logs/f31/runs/control_seed10.json",
                    "origin/phase2-experiments:results/f31/control_seed10_e25_h9.csv"]  # records and docs aren't looks
    touch_kept(trained / next(f for f in record(trained, "control", 14)["files_sha256"] if "_epochs/" in f), b"x")
    with pytest.raises(R.LaunchError, match="abort-run: H9 has been looked at .*origin/phase2-experiments:"):
        R.abort_run("control", 14, "retrain after the results PR", trained)
    with pytest.raises(R.LaunchError, match="abort-all: H9 has been looked at .*origin/phase2-experiments:"):
        R.abort_all("start over", trained)
    runner = FakeRunner(trained)
    with pytest.raises(R.LaunchError, match="guard 4: training records that do not verify: .*restore the exact bytes"):
        train(trained, git=git, runner=runner)
    assert runner.calls == []
    local = fake_git(trained, branch_files={"phase2-experiments": RESULTS_ON_BRANCH["origin/phase2-experiments"]})
    assert R.looked(trained, A_REAL, local)  # the local ref alone counts too


def test_unreadable_branch_refs_refuse(tmp_path, monkeypatch):
    root = fake_repo(tmp_path)
    git = fake_git(root, unreadable=("origin/phase2-experiments", "phase2-experiments"))
    monkeypatch.setattr(R, "real_git", lambda r: git)
    with pytest.raises(R.LaunchError, match="can't confirm no look"):
        R.looked(root, A_REAL, git)
    for stage in (lambda: R.abort_run("control", 10, "x", root), lambda: R.abort_all("x", root),
                  lambda: train(root, git=git)):
        with pytest.raises(R.LaunchError, match="can't confirm no look"):
            stage()
    one = fake_git(root, unreadable=("phase2-experiments",))
    assert R.looked(root, A_REAL, one) == []  # one readable ref is enough


@pytest.mark.slow
def test_route_ii_before_any_look_still_retrains_one_run(trained, monkeypatch):
    """Before any look, with the 40 records committed (only .train.json and docs on the branch), route (ii)'s abort-run
    and launch retrain the one bad run."""
    records = [R.run_paths(A_REAL, a, s)["record"] for _, a, s in R.schedule(A_REAL)] + ["logs/decisions.md"]
    git = fake_git(trained, branch_files={"origin/phase2-experiments": records, "phase2-experiments": records})
    monkeypatch.setattr(R, "real_git", lambda root: git)
    assert R.looked(trained, A_REAL, git) == []
    touch_kept(trained / next(f for f in record(trained, "control", 14)["files_sha256"] if "_epochs/" in f), b"x")
    R.abort_run("control", 14, "kept epoch file changed", trained)
    runner = train(trained, git=git)
    assert train_calls(runner) == [("control", 14)]


def test_the_looked_stage_prints_the_evidence_and_writes_nothing(tmp_path, capsys):
    root = fake_repo(tmp_path)
    def files():  # Python's own bytecode cache (check.py runs pytest without -B) is not a driver write
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if "__pycache__" not in p.parts)

    before = files()
    assert R.print_looked(root, fake_git(root)) == []
    assert "H9 LOOK EVIDENCE: none" in capsys.readouterr().out
    found = R.print_looked(root, fake_git(root, branch_files=RESULTS_ON_BRANCH))
    assert len(found) == 3 and "origin/phase2-experiments:logs/f31/phase2_started.json" in capsys.readouterr().out
    assert files() == before
    with pytest.raises(SystemExit):
        R.main(["looked", "control", "10"])


# ---- look evidence anywhere in history (hc-adversary item 27, cases A-E) -------------------------------------------
RESULTS = ["logs/f31/phase2_started.json", "results/f31/control_seed10_e25_h9.csv", "logs/f31/runs/control_seed10.json"]
RECORDS_ONLY = ["logs/f31/runs/control_seed10.train.json", "logs/f31/runs/atlantic_seed10.train.json"]


@pytest.mark.parametrize("case, kw, look", [
    ("A_merged", dict(branch_files={"origin/phase2-experiments": RESULTS, "phase2-experiments": RESULTS},
                      history=RECORDS_ONLY + RESULTS), True),
    ("B_unmerged_branch", dict(history=RECORDS_ONLY + RESULTS), True),  # tips carry nothing; a pushed PR branch does
    ("C_added_then_removed", dict(history=RECORDS_ONLY + RESULTS[:1]), True),  # tips carry nothing any more
    ("D_B_with_failed_fetch", dict(history=RECORDS_ONLY + RESULTS, fetch_rc=1), True),
    ("E_no_look", dict(branch_files={"origin/phase2-experiments": RECORDS_ONLY}, history=RECORDS_ONLY), False),
], ids=["A", "B", "C", "D", "E"])
def test_a_look_anywhere_in_history_counts(tmp_path, monkeypatch, case, kw, look):
    root = fake_repo(tmp_path)
    git = fake_git(root, **kw)
    monkeypatch.setattr(R, "real_git", lambda r: git)
    seen = R.looked(root, A_REAL, git)
    assert bool(seen) is look, seen
    if case != "A_merged" and look:  # the history read alone finds it
        assert seen and all(x.startswith("history:") for x in seen)
    if look:
        with pytest.raises(R.LaunchError, match="abort-run: H9 has been looked at .*history:"):
            R.abort_run("control", 10, "route (ii) after a look", root)
        with pytest.raises(R.LaunchError, match="abort-all: H9 has been looked at"):
            R.abort_all("start over", root)


def test_a_failing_history_read_refuses(tmp_path, monkeypatch):
    root = fake_repo(tmp_path)
    git = fake_git(root, log_rc=128)
    monkeypatch.setattr(R, "real_git", lambda r: git)
    for stage in (lambda: R.looked(root, A_REAL, git), lambda: R.abort_run("control", 10, "x", root),
                  lambda: R.abort_all("x", root), lambda: train(root, git=git)):
        with pytest.raises(R.LaunchError, match="can't confirm no look: `git log --all`"):
            stage()
