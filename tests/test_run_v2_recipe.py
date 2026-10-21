"""Tests for scripts/run_v2_recipe.py on a synthetic repo. Nothing real is trained, evaluated, downloaded or built.

    .venv\\Scripts\\python.exe -I -B -m pytest -q tests/test_run_v2_recipe.py -p no:cacheprovider

train.py is never executed: every test hands the driver a fake runner that writes train.py's stdout lines and output
files, and an autouse fixture makes any other subprocess call fail. git is faked; `show` of scripts/train.py,
scripts/run_n5.py, scripts/analyze_n5.py and excluded_crops.txt returns their real committed blobs (read once at
import), so the pins are checked against real bytes. The protected hashes are synthetic: the driver's keyword hook
`expected` replaces analyze_n5.PROTECTED (main() exposes no hook). The other hooks stand in for the GPU parts: CUDA,
the child import path, nvidia-smi, re-validation and the tensor-equal check.

HC_RUN_V2_UNDER_TEST=<path> loads another copy of the driver (used to show that each guard fails a test when disabled).
It is not a PYTHON* variable, so -I does not drop it.
"""
import ast
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
import pytest

CHECKOUT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER = Path(os.environ.get("HC_RUN_V2_UNDER_TEST") or CHECKOUT / "scripts" / "run_v2_recipe.py")
R = load("run_v2_recipe", DRIVER)
CHECK = load("check_for_run_v2", CHECKOUT / "scripts" / "check.py")
COMMIT = "c0ffee" * 6 + "abcd"
SEEDS = tuple(range(30, 40))
PY = sys.executable


def real_git(*args):
    """Real git, at import only (before the autouse fixture replaces subprocess)."""
    return subprocess.run(["git", "-C", str(CHECKOUT), *args], capture_output=True)


REAL_BLOBS = {rel: real_git("show", f"HEAD:{rel}").stdout
              for rel in (R.TRAIN_PY, R.RUN_N5, R.ANALYZE_N5, R.EXCLUDED_CROPS)}
REAL_IDS = {rel: real_git("rev-parse", f"HEAD:{rel}").stdout.decode().strip()
            for rel in (R.EXCLUDED_CROPS, R.TRAIN_LIST, R.VAL_LIST)}
COMMITTED_KINDS = ("logs/v2_recipe/seed30.log", "logs/v2_recipe/runs/seed30.train.json",
                   "logs/v2_recipe/aborted/seed30_attempt1/abort.json",
                   "logs/v2_recipe/aborted/seed30_attempt1/logs/v2_recipe/seed30.log",
                   "logs/v2_recipe/aborted/freeze_1/members_sha256.txt", "results/v2_recipe/members_sha256.txt",
                   "reports/decisions/V2-freeze.md")
REAL = {  # every real-git answer the pin and attribute tests need, read once here
    "ls-tree": {ref: real_git("ls-tree", ref, "src/hurricane_intensity/").stdout.decode().splitlines()
                for ref in ("fc71b98", "HEAD")},
    "rev-parse": {(ref, rel): real_git("rev-parse", f"{ref}:{rel}").stdout.decode().strip()
                  for ref in ("fc71b98", "HEAD") for rel in (R.EXCLUDED_CROPS, R.TRAIN_LIST, R.VAL_LIST)},
    "check-attr": real_git("check-attr", "text", "--", *COMMITTED_KINDS).stdout.decode().splitlines(),
    "ignored": {rel: real_git("check-ignore", "-q", rel).returncode for rel in (R.LOCK, "logs/v2_recipe/seed30.log")},
}


@pytest.fixture(autouse=True)
def no_real_subprocess(monkeypatch):
    """Any subprocess the driver starts, directly or through run_n5's code, fails the test unless it goes through the
    runner the test hands it."""
    started = []

    def refuse(*a, **k):
        started.append(a[:1])
        raise AssertionError(f"a real subprocess was started: {a[:1]}")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    yield started


# ---- the synthetic repo ---------------------------------------------------------------------------------------
def fake_repo(tmp_path):
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
    (root / "COORDINATOR.md").write_text(f"# c\n\n{CHECK.HASH_HEADING}\n\n```sha256\n{block}\n```\n", encoding="utf-8")
    for rel in (R.TRAIN_LIST, R.VAL_LIST) + R.FORBIDDEN_LISTS:  # the real lists, comments and all
        shutil.copy(CHECKOUT / rel, root / rel)
    (root / "prot.json").write_text(json.dumps(prot), encoding="utf-8")
    pkg = root / R.SRC_DIR  # the pinned package's files (guard 2 checks nothing else is in the folder)
    (pkg / "__pycache__").mkdir(parents=True)
    for name in R.SRC_BLOBS:
        (pkg / name).write_text("# synthetic\n", encoding="utf-8")
    return root


def prot_of(root):
    return json.loads((root / "prot.json").read_text(encoding="utf-8"))


def fake_git(root=None, dirty=False, differs=(), fetch_rc=0, head=COMMIT, first_parent=(COMMIT,), blobs=None,
             ids=None, src=None, tracked=(), on_origin=(), decisions=None, origin_tree=(), tree_rc=0, changed=()):
    """git for the synthetic repo. blobs/ids override `show`/`rev-parse` of HEAD:<path>; src overrides the
    src/hurricane_intensity listing; decisions maps a ref to its logs/decisions.md text."""
    blobs = {**REAL_BLOBS, **(blobs or {})}
    ids = {**REAL_IDS, **(ids or {})}
    src = R.SRC_BLOBS if src is None else src
    decisions = {"HEAD": "# decisions\n", "origin/phase2-experiments": "# decisions\n", **(decisions or {})}

    def git(args, check=True):
        if args == ["fetch", "origin"]:
            return SimpleNamespace(stdout=b"", returncode=fetch_rc)
        if args == ["rev-list", "--first-parent", "origin/phase2-experiments"]:
            return SimpleNamespace(stdout="\n".join(first_parent).encode() + b"\n", returncode=0)
        if args == ["status", "--porcelain", "--untracked-files=no"]:
            return SimpleNamespace(stdout=b" M scripts/train.py\n" if dirty else b"", returncode=0)
        if args[:4] == ["diff", "--quiet", "HEAD", "--"]:
            return SimpleNamespace(stdout=b"", returncode=1 if args[4] in differs else 0)
        if args[:4] == ["diff", "--no-renames", "-z", "--name-only"]:
            return SimpleNamespace(stdout="\0".join(changed).encode() + b"\0", returncode=0)
        if args == ["ls-tree", "HEAD", "src/hurricane_intensity/"]:
            out = "".join(f"100644 blob {b}\tsrc/hurricane_intensity/{n}\n" for n, b in src.items())
            return SimpleNamespace(stdout=out.encode(), returncode=0)
        if args[:5] == ["ls-tree", "-r", "-z", "--name-only", "origin/phase2-experiments"]:
            assert args[5] == "--" and tuple(args[6:]) == R.LATER_OUTPUTS
            return SimpleNamespace(stdout="\0".join(origin_tree).encode(), returncode=tree_rc)
        if args[:2] == ["ls-files", "--error-unmatch"]:
            return SimpleNamespace(stdout=b"", returncode=0 if args[2] in tracked else 1)
        if args[:2] == ["cat-file", "-e"]:
            return SimpleNamespace(stdout=b"", returncode=0 if args[2].split(":", 1)[1] in on_origin else 128)
        if args[0] == "show":
            ref, rel = args[1].split(":", 1)
            if rel == "logs/decisions.md":
                return SimpleNamespace(stdout=decisions[ref].encode(), returncode=0)
            assert ref == "HEAD", args
            return SimpleNamespace(stdout=blobs[rel], returncode=0)
        if args == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=head.encode() + b"\n", returncode=0)
        if args[0] == "rev-parse" and args[1].startswith("HEAD:"):
            return SimpleNamespace(stdout=ids[args[1][5:]].encode() + b"\n", returncode=0)
        raise AssertionError(args)
    return git


def val_curve(seed):
    """Unrounded validation MAE per epoch: falls, then a noisy plateau, so the picks spread over the run."""
    rng = np.random.default_rng([seed, 31])
    return {e: round(float(8.0 + 6.0 * np.exp(-e / 8.0) + rng.normal(0, 0.08)), 6) for e in range(1, R.EPOCHS + 1)}


class FakeRunner:
    """Stands in for subprocess.run on train.py: writes its stdout lines (CRLF, as on Windows) and output files."""

    def __init__(self, root, rc=0, curve=val_curve, lines=None, on_train=None):
        self.root, self.rc, self.curve, self.lines, self.on_train, self.calls = root, rc, curve, lines, on_train, []

    def __call__(self, cmd, stdout, stderr, cwd, env):
        assert Path(cwd) == self.root and env["PYTHONUTF8"] == "1" and stderr is subprocess.STDOUT
        self.calls.append(list(cmd))
        a = dict(zip(cmd[3::2], cmd[4::2]))
        seed = int(a["--seed"])
        head = ["device: cuda", f"train storms: {R.CONTROL_IDS}", f"val storms:   {R.VAL_IDS}",
                f"seed: {seed}  checkpoint: {a['--out']}  loss: l1  circular_mask: False  rotate: False",
                "after exclusion: 2081", "after exclusion: 940", "train samples: 2081  val samples: 940"]
        lines = self.lines(head) if self.lines else head
        if self.on_train:
            self.on_train(seed)
        if self.rc:
            stdout.write(("\r\n".join(lines + ["Traceback (most recent call last):"]) + "\r\n").encode())
            return SimpleNamespace(returncode=self.rc)
        val, best = self.curve(seed), np.inf
        (self.root / a["--epoch-dir"]).mkdir(parents=True, exist_ok=True)
        for e in range(1, int(a["--epochs"]) + 1):
            weights = f"weights {seed} {e}".encode()
            lines.append(f"epoch {e}: train {10 - e * 0.05:.2f} kt  val {val[e]:.2f} kt")
            (self.root / a["--epoch-dir"] / f"epoch_{e:02d}.pt").write_bytes(weights)
            if val[e] < best:
                best = val[e]
                (self.root / a["--out"]).write_bytes(weights)
                lines.append("  saved")
        stdout.write(("\r\n".join(lines) + "\r\n").encode())
        return SimpleNamespace(returncode=0)


def reval_from(curve=val_curve):
    def reval(epoch_dir, epochs, train_list):
        seed = int(re.search(r"seed(\d+)_epochs$", Path(epoch_dir).as_posix()).group(1))
        assert train_list == R.TRAIN_LIST and epochs == R.EPOCHS
        assert all((Path(epoch_dir) / f"epoch_{e:02d}.pt").exists() for e in range(1, epochs + 1))
        return curve(seed)
    return reval


def same_bytes(a, b):
    return Path(a).read_bytes() == Path(b).read_bytes()


def pkg_in_src(root):
    return str(Path(root) / "src" / "hurricane_intensity" / "__init__.py")


def train(root, git=None, runner=None, reval=None, cuda=True, pkg=pkg_in_src, weights=same_bytes, expected=None):
    runner = runner or FakeRunner(root)
    R.launch(root, git=git or fake_git(root), expected=expected or prot_of(root), runner=runner,
             revalidate_fn=reval or reval_from(), snapshot=lambda: "gpu", cuda_ok=lambda: cuda, child_import=pkg,
             same_weights_fn=weights)
    return runner


def record(root, seed):
    return json.loads((root / R.run_paths(seed)["record"]).read_text(encoding="utf-8"))


def rewrite_record(root, seed, **fields):
    p = root / R.run_paths(seed)["record"]
    rec = dict(record(root, seed), **fields)
    p.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")


@pytest.fixture(scope="module")
def trained_base(tmp_path_factory):
    """The 10 runs, trained once per module; tests copy it. Its setup takes about 2 s, so every test that uses it
    (through `trained` or `frozen`) is marked slow."""
    root = fake_repo(tmp_path_factory.mktemp("base"))
    train(root)
    return root


@pytest.fixture
def trained(tmp_path, trained_base):
    root = tmp_path / "repo"
    shutil.copytree(trained_base, root)
    for f in root.rglob("*"):
        if f.is_file():
            os.chmod(f, stat.S_IREAD | stat.S_IWRITE)  # copytree keeps read-only bits; make kept copies writable
    for s in SEEDS:
        for k in ("train_log", "native_checkpoint"):
            os.chmod(root / R.run_paths(s)[k], stat.S_IREAD)
    return root


# ---- the design, the pins and the source -------------------------------------------------------------------------
def test_commands_are_the_design_argument_lists():
    assert R.SEEDS == SEEDS and [R.order_of(s) for s in SEEDS] == list(range(1, 11))
    assert R.train_cmd("PY", 30) == [
        "PY", "-u", "scripts/train.py", "--train-list", "storms_train_expanded.txt", "--val-list",
        "storms_val_expanded.txt", "--epochs", "99", "--seed", "30", "--loss", "l1", "--out",
        "models/v2_recipe/seed30.pt", "--epoch-dir", "models/v2_recipe/seed30_epochs"]
    assert R.epochs_for(2081, 4 * 1640) == (99, 66) and R.CONTROL_TRAIN_FRAMES == 2081 and R.VAL_FRAMES == 940
    assert R.member_paths() == [f"models/v2_recipe/seed{s}.pt" for s in SEEDS]


def test_pins_match_the_committed_blobs():
    """The pins are F-31's: train.py, run_n5.py and analyze_n5.py at HEAD, and the src, excluded_crops and list
    blobs at fc71b98, which HEAD still holds."""
    for rel, pin in ((R.TRAIN_PY, R.TRAIN_PY_SHA256), (R.RUN_N5, R.RUN_N5_SHA256), (R.ANALYZE_N5, R.ANALYZE_N5_SHA256),
                     (R.EXCLUDED_CROPS, R.EXCLUDED_CROPS_SHA256)):
        assert hashlib.sha256(REAL_BLOBS[rel]).hexdigest() == pin, rel
    for ref in ("fc71b98", "HEAD"):
        out = REAL["ls-tree"][ref]
        assert {Path(ln.split("\t")[1]).name: ln.split()[2] for ln in out} == R.SRC_BLOBS, ref
        for rel, pin in ((R.EXCLUDED_CROPS, R.EXCLUDED_CROPS_BLOB), (R.TRAIN_LIST, R.TRAIN_LIST_BLOB),
                         (R.VAL_LIST, R.VAL_LIST_BLOB)):
            assert REAL["rev-parse"][(ref, rel)] == pin, (ref, rel)


FORBIDDEN_NAMES = ("eval_cmd", "evaluate_step", "evaluate_phase", "HOLDOUT_LIST", "evaluate.py", "ensemble_eval.py",
                   "storms_holdout_")


def test_driver_source_names_no_evaluation_path():
    """Design §3's name check. The one exception: guard 5's FORBIDDEN_LISTS line names the held-out lists it reads
    for their IDs (and the behavioural test shows no command carries them)."""
    lines = DRIVER.read_text(encoding="utf-8").splitlines()
    hits = [(i, n) for i, ln in enumerate(lines, 1) for n in FORBIDDEN_NAMES if n in ln]
    hits += [(i, "h9_") for i, ln in enumerate(lines, 1) if re.search(r"h9_", ln, flags=re.I)]
    exempt = [i for i, ln in enumerate(lines, 1) if ln.startswith("FORBIDDEN_LISTS = (")]
    assert len(exempt) == 1
    assert [h for h in hits if h[0] not in exempt] == []
    assert {n for i, n in hits if i in exempt} == {"storms_holdout_"}


def test_driver_imports_only_the_two_pinned_modules():
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods == {"argparse", "hashlib", "json", "os", "re", "stat", "subprocess", "sys", "types", "pathlib"}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "load_pinned"]
    assert sorted(ast.literal_eval(c.args[1].right) for c in calls) == ["analyze_n5.py", "run_n5.py"]
    assert not any(isinstance(n, ast.Name) and n.id in ("importlib", "__import__") for n in ast.walk(tree))


def test_a_tampered_module_is_refused_before_it_runs(tmp_path):
    bad = tmp_path / "run_n5.py"
    bad.write_bytes(REAL_BLOBS[R.RUN_N5] + b"\nraise RuntimeError('executed')\n")
    with pytest.raises(R.LaunchError, match="refused before import"):
        R.load_pinned("x", bad, R.RUN_N5_SHA256)


def test_a_pinned_module_runs_from_the_hashed_bytes_with_file_set_first(tmp_path, monkeypatch):
    f = tmp_path / "m.py"
    f.write_bytes(b"SEEN = __file__\r\nX = 1\r\n")  # CRLF working copy, hashed as LF
    pin = hashlib.sha256(b"SEEN = __file__\nX = 1\n").hexdigest()
    real_read = Path.read_bytes
    reads = []

    def counting(self):
        reads.append(self)
        return real_read(self)
    with monkeypatch.context() as mp:
        mp.setattr(Path, "read_bytes", counting)
        m = R.load_pinned("m", f, pin)
    assert m.SEEN == str(f) and m.X == 1 and reads == [f]


def test_coordinator_parser_agrees_with_check(tmp_path):
    root = fake_repo(tmp_path)
    assert R.coordinator_block(root / "COORDINATOR.md") == dict(CHECK.load_expected(root / "COORDINATOR.md"))
    assert set(R.PROTECTED) == set(CHECK.PROTECTED_PATHS)


# ---- the behavioural no-evaluation test (design §3) --------------------------------------------------------------
@pytest.mark.slow
def test_launch_starts_only_train_py_with_the_design_lists_in_seed_order(no_real_subprocess, tmp_path):
    """The 10-seed launch hands the runner exactly the ten §6 commands, in seed order, and nothing else; with the
    tripwire on, no other subprocess is started (git, CUDA, the child import and nvidia-smi are hooks here)."""
    root = fake_repo(tmp_path)
    runner = train(root)
    assert runner.calls == [R.train_cmd(PY, s) for s in SEEDS]
    assert no_real_subprocess == []
    for s in SEEDS:
        assert not any(x in " ".join(runner.calls[s - 30]) for x in R.FORBIDDEN_LISTS + R.SEALED_ROOTS)


@pytest.mark.slow
def test_records_pass_verify_and_freeze_writes_the_members_file(trained, capsys):
    git = fake_git(trained)
    rows = R.verify(trained, git=git, expected=prot_of(trained))
    assert [r["seed"] for r in rows] == list(SEEDS)
    for s in SEEDS:
        rec = record(trained, s)
        assert rec["native_same_weights"] is True and rec["order"] == s - 29 and rec["epochs"] == 99
        assert set(rec["files_sha256"]) == {f"logs/v2_recipe/seed{s}.log", f"models/v2_recipe/seed{s}.pt"}
        assert sorted(map(int, rec["deleted_epoch_sha256"])) == list(range(1, 100))
        assert not (trained / R.run_paths(s)["epoch_dir"]).exists()
        assert rec["under_trained"] is (rec["native_epoch"] >= 85)
        log = (trained / R.run_paths(s)["train_log"]).read_bytes()
        assert log.startswith(f"RUN v2_recipe seed={s} order={s - 29} ".encode()) and b"\r\n" in log
    R.freeze(trained, git=git, expected=prot_of(trained))
    raw = (trained / R.MEMBERS).read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    # ensemble_eval.py --expect-sha's comparison (ensemble_eval.py:42-47), with --checkpoints as design §4 passes them
    want = [f"{hashlib.sha256((trained / c).read_bytes()).hexdigest()} *{Path(c).as_posix()}" for c in R.member_paths()]
    assert [ln for ln in raw.decode().split("\n") if ln.strip()] == want
    assert "never this file as an output path" in capsys.readouterr().out


# ---- launch guards -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw, match", [
    (dict(dirty=True), "guard 1: the tracked tree"),
    (dict(differs=(R.SELF,)), "guard 1: scripts/run_v2_recipe.py differs"),
    (dict(differs=(R.TRAIN_PY,)), "guard 1: scripts/train.py differs"),
    (dict(blobs={R.TRAIN_PY: b"other\n"}), "guard 2: HEAD's scripts/train.py"),
    (dict(blobs={R.RUN_N5: b"other\n"}), "guard 2: HEAD's scripts/run_n5.py"),
    (dict(blobs={R.ANALYZE_N5: b"other\n"}), "guard 2: HEAD's scripts/analyze_n5.py"),
    (dict(src={**R.SRC_BLOBS, "dataset.py": "0" * 40}), r"guard 2: .*differs: \['dataset.py'\]"),
    (dict(src={**R.SRC_BLOBS, "extra.py": "0" * 40}), r"differs: \['extra.py'\]"),
    (dict(src={k: v for k, v in R.SRC_BLOBS.items() if k != "splits.py"}), r"differs: \['splits.py'\]"),
    (dict(ids={R.EXCLUDED_CROPS: "0" * 40}), "guard 4: HEAD's excluded_crops.txt"),
    (dict(blobs={R.EXCLUDED_CROPS: b"other\n"}), "guard 4: HEAD's excluded_crops.txt"),
    (dict(ids={R.TRAIN_LIST: "0" * 40}), "guard 4: HEAD's storms_train_expanded.txt"),
    (dict(ids={R.VAL_LIST: "0" * 40}), "guard 4: HEAD's storms_val_expanded.txt"),
    (dict(fetch_rc=1), "guard 1: git fetch origin failed"),
    (dict(first_parent=("1" * 40,)), "guard 1: .* is not on the first-parent chain"),
    (dict(tracked=(R.MEMBERS,)), "guard 8: v2 is frozen"),
    (dict(on_origin=(R.MEMBERS,)), "guard 8: v2 is frozen"),
])
def test_launch_guards_refuse_before_any_run(tmp_path, kw, match):
    root = fake_repo(tmp_path)
    runner = FakeRunner(root)
    with pytest.raises(R.LaunchError, match=match):
        train(root, git=fake_git(root, **kw), runner=runner)
    assert runner.calls == [] and not (root / "models/v2_recipe").exists()


def test_guard_3_refuses_a_block_mismatch_or_a_changed_protected_file(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="guard 3: analyze_n5.PROTECTED differs"):
        train(root, expected={**prot_of(root), "best_model.pt": "e" * 64})
    (root / "best_model.pt").write_bytes(b"changed")
    with pytest.raises(R.LaunchError, match=r"guard 3: protected files differ: \['best_model.pt'\]"):
        train(root)


def test_guard_3_uses_analyze_n5s_protected_by_default(tmp_path):
    root = fake_repo(tmp_path)  # synthetic hashes, so the real analyze_n5.PROTECTED can't match the block
    with pytest.raises(R.LaunchError, match="guard 3"):
        R.launch(root, git=fake_git(root), runner=FakeRunner(root), revalidate_fn=reval_from(),
                 snapshot=lambda: "", cuda_ok=lambda: True, child_import=pkg_in_src, same_weights_fn=same_bytes)


def test_guard_5_refuses_lists_that_meet_a_held_out_set(tmp_path):
    root = fake_repo(tmp_path)
    with open(root / "storms_holdout_hep.txt", "a", encoding="utf-8") as f:
        f.write(f"{R.CONTROL_IDS[0]}\n")
    with pytest.raises(R.LaunchError, match=f"guard 5: .*{R.CONTROL_IDS[0]}.*storms_holdout_hep.txt"):
        train(root)
    root2 = fake_repo(tmp_path / "b")
    (root2 / R.VAL_LIST).write_text("\n".join(R.VAL_IDS[:-1] + ["AL012026"]) + "\n", encoding="utf-8")
    with pytest.raises(R.LaunchError, match="guard 5: the train or validation list is not"):
        train(root2)
    root3 = fake_repo(tmp_path / "c")
    (root3 / "storms_test_2025.txt").write_text("# none\n", encoding="utf-8")
    with pytest.raises(R.LaunchError, match="guard 5: storms_test_2025.txt names no storm"):
        train(root3)


def test_guard_5_refuses_a_2026_id(tmp_path, monkeypatch):
    root = fake_repo(tmp_path)
    ids = R.CONTROL_IDS + ["AL012026"]
    monkeypatch.setattr(R, "CONTROL_IDS", sorted(ids))
    (root / R.TRAIN_LIST).write_text("\n".join(ids) + "\n", encoding="utf-8")
    with pytest.raises(R.LaunchError, match=r"guard 5: .*2026 storms \['AL012026'\]"):
        train(root)


def test_guard_5_reads_list_ids_not_comments(tmp_path):
    root = fake_repo(tmp_path)
    (root / R.VAL_LIST).write_text("# AL092022 is H9, mentioned in a comment\n" + "\n".join(R.VAL_IDS) + "\n",
                                   encoding="utf-8")
    assert R.list_ids(root, R.VAL_LIST, 5) == R.VAL_IDS


@pytest.mark.parametrize("cmd_edit, match", [
    (lambda c: c + ["--rotate"], "not design §6's"),
    (lambda c: c[:5] + ["storms_train_n5_atlantic.txt"] + c[6:], "not design §6's"),
])
def test_guard_7_refuses_any_other_command(cmd_edit, match):
    with pytest.raises(R.LaunchError, match=match):
        R.check_command(cmd_edit(R.train_cmd(PY, 30)), PY, 30, set())


def test_guard_7_refuses_held_out_sealed_or_2026_arguments(monkeypatch):
    for bad in ("models/v2_recipe/AL092022.pt", "data/hep_sealed/x", "models/AL052026.pt", "storms_holdout_hep.txt"):
        cmd = R.train_cmd(PY, 30)
        cmd[-1] = bad
        monkeypatch.setattr(R, "train_cmd", lambda py, seed, c=cmd: c)
        with pytest.raises(R.LaunchError, match="guard 7: seed 30: argument"):
            R.check_command(cmd, PY, 30, {"AL092022"})


def test_guard_6_confines_outputs():
    R.check_output_paths(R.planned_outputs())
    for bad in ("models/storms/expanded/seed0.pt", "results/v2_recipe/other.txt", "results/hep/x",
                "logs/v2_recipe/../../best_model.pt", "data/processed/x"):
        with pytest.raises(R.LaunchError, match="guard 6"):
            R.check_output_paths([bad])


def test_guard_9_refuses_a_worktree(tmp_path):
    root = fake_repo(tmp_path)
    (root / ".git").rmdir()
    (root / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    with pytest.raises(R.LaunchError, match="guard 9: .* is not the main checkout"):
        train(root)


def test_guard_9_refuses_a_child_that_imports_the_package_from_elsewhere(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="guard 9: a child python imports"):
        train(root, pkg=lambda r: str(tmp_path / "other" / "src" / "hurricane_intensity" / "__init__.py"))
    with pytest.raises(R.LaunchError, match=r"guard 9: .*\(nowhere\)"):
        train(root, pkg=lambda r: "")


def test_guard_9_refuses_without_cuda(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="guard 9: torch.cuda.is_available"):
        train(root, cuda=False)


def test_a_held_lock_refuses_every_writing_stage(tmp_path):
    root = fake_repo(tmp_path)
    (root / "logs/v2_recipe").mkdir(parents=True)
    (root / R.LOCK).write_text("pid 1\n", encoding="utf-8")
    git, prot = fake_git(root), prot_of(root)
    for call in (lambda: train(root), lambda: R.freeze(root, git=git, expected=prot),
                 lambda: R.abort_run(30, "x", root, git=git, expected=prot), lambda: R.unfreeze("x", root, git=git)):
        with pytest.raises(R.LaunchError, match="run_v2.lock exists"):
            call()


# ---- per-run checks ----------------------------------------------------------------------------------------------
def only_seed_30(root, **kw):
    """Train with a runner that stops the launch after seed 30 (its record written), via the next run's guard."""
    runner = FakeRunner(root, **kw)
    with pytest.raises(R.LaunchError):
        train(root, runner=runner, git=fake_git(root), weights=lambda a, b: same_bytes(a, b) and "seed30" in str(a))
    return runner


def test_training_failure_is_logged_and_stops(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="training failed: logs/v2_recipe/seed30.log"):
        train(root, runner=FakeRunner(root, rc=1))
    assert b"TRAIN_FAILED rc=1" in (root / "logs/v2_recipe/seed30.log").read_bytes()
    assert not (root / R.run_paths(30)["record"]).exists()


@pytest.mark.parametrize("edit, match", [
    (lambda h: [ln.replace("2081  val", "2080  val") for ln in h], "train/val frame counts"),
    (lambda h: h + ["frame list: results/n5_atlantic/frame_list_atlantic.txt: 4044 frames from 50 storms"],
     "a frame list was used"),
    (lambda h: [ln.replace("rotate: False", "rotate: True") for ln in h], "loss/mask/rotation"),
    (lambda h: [ln.replace("models/v2_recipe/seed30.pt", "models/other.pt") for ln in h], "checkpoint path"),
    (lambda h: [ln for ln in h if not ln.startswith("val storms")], "validation storms"),
])
def test_a_train_log_off_the_design_stops_with_nothing_deleted(tmp_path, edit, match):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match=f"seed 30: .*{match}"):
        train(root, runner=FakeRunner(root, lines=edit))
    assert len(list((root / R.run_paths(30)["epoch_dir"]).iterdir())) == 99
    assert not (root / R.run_paths(30)["record"]).exists()


def test_revalidation_that_does_not_round_to_the_log_stops(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="do not round to the logged ones; nothing deleted"):
        train(root, reval=reval_from(lambda s: {e: v + 0.02 for e, v in val_curve(s).items()}))
    assert len(list((root / R.run_paths(30)["epoch_dir"]).iterdir())) == 99


def near_pick(seed):
    """train.py's curve with one later epoch e2 just above the pick (0.0004 kt): not saved, same rounded value."""
    v = val_curve(seed)
    pick = min(v, key=lambda e: (v[e], e))
    e2 = 99 if pick != 99 else 98
    return {**v, e2: v[pick] + 0.0004}, pick, e2


def dip(shift):
    """Re-validation that puts e2 `shift` kt below train.py's pick: a first minimum train.py did not save."""
    def curve(seed):
        v, pick, e2 = near_pick(seed)
        return {**v, e2: v[pick] - shift}
    return curve


def test_revalidation_that_misses_train_py_pick_stops(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="not train.py's pick.*nothing deleted"):
        train(root, runner=FakeRunner(root, curve=lambda s: near_pick(s)[0]), reval=reval_from(dip(0.003)))
    assert len(list((root / R.run_paths(30)["epoch_dir"]).iterdir())) == 99


def test_revalidation_below_tie_kt_keeps_train_py_pick(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="seed 31: .*not tensor-equal"):
        train(root, runner=FakeRunner(root, curve=lambda s: near_pick(s)[0]), reval=reval_from(dip(R.TIE_KT / 2)),
              weights=lambda a, b: same_bytes(a, b) and "seed30" in str(a))
    rec = record(root, 30)
    _, pick, e2 = near_pick(30)
    assert rec["native_epoch"] == pick and rec["tie_rule"]["saved_epoch"] == pick
    assert rec["tie_rule"]["revalidated_argmin"] == e2 and abs(rec["tie_rule"]["delta_kt"]) < R.TIE_KT


def test_the_native_checkpoint_must_be_tensor_equal_to_the_pick(tmp_path):
    root = fake_repo(tmp_path)
    with pytest.raises(R.LaunchError, match="is not tensor-equal to"):
        train(root, weights=lambda a, b: False)
    assert len(list((root / R.run_paths(30)["epoch_dir"]).iterdir())) == 99


def test_a_missing_or_extra_epoch_file_stops_before_any_delete(tmp_path):
    root = fake_repo(tmp_path)
    runner = FakeRunner(root)

    def call(cmd, **k):
        r = runner(cmd, **k)
        (root / R.epoch_file(30, 50)).unlink()
        return r
    with pytest.raises(R.LaunchError, match="epoch files missing"):
        train(root, runner=call)
    assert len(list((root / R.run_paths(30)["epoch_dir"]).iterdir())) == 98
    root2 = fake_repo(tmp_path / "b")
    runner2 = FakeRunner(root2)

    def call2(cmd, **k):
        r = runner2(cmd, **k)
        (root2 / R.run_paths(30)["epoch_dir"] / "notes.txt").write_text("x", encoding="utf-8")
        return r
    with pytest.raises(R.LaunchError, match="holds files other than"):
        train(root2, runner=call2)
    assert len(list((root2 / R.run_paths(30)["epoch_dir"]).iterdir())) == 100


def test_protected_file_changed_during_a_run_stops(tmp_path):
    root = fake_repo(tmp_path)
    runner = FakeRunner(root, on_train=lambda s: (root / "best_model.pt").write_bytes(b"x"))
    with pytest.raises(R.LaunchError, match=r"guard 3: protected hashes changed during seed 30: \['best_model.pt'\]"):
        train(root, runner=runner)
    assert len(list((root / R.run_paths(30)["epoch_dir"]).iterdir())) == 99  # stopped before re-validation


def test_protected_drift_between_runs_stops_the_next_run_unstarted(tmp_path, monkeypatch):
    root = fake_repo(tmp_path)
    real = R.write_json_new

    def write_then_drift(path, obj):
        real(path, obj)
        (root / "best_model.pt").write_bytes(b"x")  # after seed 30's record, before seed 31
    monkeypatch.setattr(R, "write_json_new", write_then_drift)
    runner = FakeRunner(root)
    with pytest.raises(R.LaunchError, match="guard 3: protected hashes differ before seed 31; not started"):
        train(root, runner=runner)
    assert runner.calls == [R.train_cmd(PY, 30)] and not (root / R.run_paths(31)["train_log"]).exists()


@pytest.mark.slow
def test_a_finished_run_is_skipped_and_partial_ones_refuse_until_aborted(tmp_path):
    root = fake_repo(tmp_path)
    only_seed_30(root)
    assert record(root, 30)["seed"] == 30 and (root / R.run_paths(31)["train_log"]).exists()
    with pytest.raises(R.LaunchError, match=r"guard 6: unfinished runs have files on disk: seeds \[31\]"):
        train(root)
    git, prot = fake_git(root), prot_of(root)
    with pytest.raises(R.LaunchError, match="abort-run: seed 30 has a record that passes"):
        R.abort_run(30, "x", root, git=git, expected=prot)
    dest = R.abort_run(31, "same_weights hook refused", root, git=git, expected=prot)
    assert dest == "logs/v2_recipe/aborted/seed31_attempt1"
    info = json.loads((root / dest / "abort.json").read_text(encoding="utf-8"))
    assert info["status"] == "complete" and info["seed"] == 31 and info["order"] == 2
    runner = train(root)
    assert runner.calls == [R.train_cmd(PY, s) for s in SEEDS[1:]]
    assert record(root, 31)["aborted_attempts"] == [f"{dest}/abort.json"]
    R.verify(root, git=git, expected=prot)


@pytest.mark.slow
def test_a_record_that_fails_a_check_refuses_launch_and_can_be_aborted(trained):
    rewrite_record(trained, 33, native_epoch=1)
    with pytest.raises(R.LaunchError, match=r"guard 6: training records that fail a check.*seeds \[33\]"):
        train(trained)
    R.abort_run(33, "bad record", trained, git=fake_git(trained), expected=prot_of(trained))
    runner = train(trained)
    assert runner.calls == [R.train_cmd(PY, 33)]


@pytest.mark.slow
def test_records_made_at_another_commit_refuse_launch(trained):
    with pytest.raises(R.LaunchError, match=r"or are stale .*seeds \[30, 31"):
        train(trained, git=fake_git(trained, head="1" * 40, first_parent=("1" * 40,)))


def test_an_interrupted_abort_blocks_launch_and_resumes(tmp_path, monkeypatch):
    root = fake_repo(tmp_path)
    only_seed_30(root)
    real = os.rename
    calls = []

    def flaky(a, b):
        calls.append(a)
        if len(calls) == 2:
            raise OSError("interrupted")
        return real(a, b)
    monkeypatch.setattr(os, "rename", flaky)
    with pytest.raises(OSError):
        R.abort_run(31, "x", root, git=fake_git(root), expected=prot_of(root))
    monkeypatch.setattr(os, "rename", real)
    with pytest.raises(R.LaunchError, match="guard 6: unfinished aborts"):
        train(root)
    dest = R.abort_run(31, "x", root, git=fake_git(root), expected=prot_of(root))
    assert dest.endswith("seed31_attempt1") and not (root / "logs/v2_recipe/aborted/seed31_attempt2").exists()
    assert json.loads((root / dest / "abort.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_abort_run_refusals(tmp_path):
    root = fake_repo(tmp_path)
    git, prot = fake_git(root), prot_of(root)
    with pytest.raises(R.LaunchError, match="unknown seed 29"):
        R.abort_run(29, "x", root, git=git, expected=prot)
    with pytest.raises(R.LaunchError, match="no --reason"):
        R.abort_run(30, "", root, git=git, expected=prot)
    with pytest.raises(R.LaunchError, match="no files to move"):
        R.abort_run(30, "x", root, git=git, expected=prot)
    with pytest.raises(R.LaunchError, match="abort-run: v2 is frozen"):
        R.abort_run(30, "x", root, git=fake_git(root, tracked=(R.MEMBERS,)), expected=prot)


# ---- verify: the training-only checks ----------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("fields, match", [
    (dict(order=3), "order=3"),
    (dict(epochs=98), "epochs=98"),
    (dict(frame_list_sha256="0" * 64), "frame_list_sha256"),
    (dict(train_py_sha256="0" * 64), "train_py_sha256"),
    (dict(native_same_weights=False), "native_same_weights"),
    (dict(val_list="storms_val.txt"), "val_list"),
    (dict(native_epoch=1), "native_epoch 1"),
    (dict(tie_rule={"revalidated_argmin": 1, "saved_epoch": 2, "delta_kt": 0.0}), "tie_rule"),
    (dict(under_trained="x"), "under_trained must be"),
    (dict(aborted_attempts=["logs/v2_recipe/aborted/seed30_attempt1/abort.json"]), "aborted_attempts"),
    (dict(deleted_epoch_sha256={"1": "0" * 64}), "deleted_epoch_sha256 must hash all 99"),
    (dict(files_sha256={"logs/v2_recipe/seed30.log": "0" * 64}), "files_sha256 must name exactly"),
    (dict(protected_after={}), "protected_after"),
    (dict(val_mae_revalidated={"1": 9.0}), "re-validated values"),
])
def test_verify_refuses_a_record_off_the_design(trained, fields, match):
    rewrite_record(trained, 30, **fields)
    with pytest.raises(R.LaunchError, match=f"verify: logs/v2_recipe/runs/seed30.train.json: .*{re.escape(match)}"):
        R.verify(trained, git=fake_git(trained), expected=prot_of(trained))


@pytest.mark.slow
def test_verify_refuses_a_changed_member_or_a_left_over_epoch_file(trained):
    p = trained / R.run_paths(31)["native_checkpoint"]
    os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
    p.write_bytes(b"changed")
    with pytest.raises(R.LaunchError, match="models/v2_recipe/seed31.pt is missing or changed"):
        R.verify(trained, git=fake_git(trained), expected=prot_of(trained))


@pytest.mark.slow
def test_verify_refuses_a_left_over_epoch_file(trained):
    (trained / R.run_paths(32)["epoch_dir"]).mkdir(parents=True)
    (trained / R.epoch_file(32, 7)).write_bytes(b"x")
    with pytest.raises(R.LaunchError, match="seed32.train.json: epoch files still on disk"):
        R.verify(trained, git=fake_git(trained), expected=prot_of(trained))


@pytest.mark.slow
def test_verify_refuses_an_unlisted_abort_directory(trained):
    d = trained / "logs/v2_recipe/aborted/seed34_attempt1"
    d.mkdir(parents=True)
    (d / "abort.json").write_text('{"status": "complete"}\n', encoding="utf-8")
    with pytest.raises(R.LaunchError, match="seed34.train.json: aborted_attempts"):
        R.verify(trained, git=fake_git(trained), expected=prot_of(trained))


@pytest.mark.slow
def test_verify_accepts_a_freeze_branch_head_but_not_other_changes(trained):
    git = fake_git(trained, head="2" * 40, changed=("logs/v2_recipe/runs/seed30.train.json", R.MEMBERS,
                                                    "logs/decisions.md", "reports/decisions/V2-freeze.md"))
    R.verify(trained, git=git, expected=prot_of(trained))
    git = fake_git(trained, head="2" * 40, changed=("src/hurricane_intensity/dataset.py",))
    with pytest.raises(R.LaunchError, match="more than the decisions docs and the v2 outputs"):
        R.verify(trained, git=git, expected=prot_of(trained))


@pytest.mark.slow
def test_verify_refuses_records_from_two_commits(trained):
    rewrite_record(trained, 39, commit="1" * 40)
    with pytest.raises(R.LaunchError, match="more than one launch commit"):
        R.verify(trained, git=fake_git(trained), expected=prot_of(trained))


def test_verify_refuses_an_unfinished_set(tmp_path):
    root = fake_repo(tmp_path)
    only_seed_30(root)
    with pytest.raises(R.LaunchError, match="not all 10 runs have finished"):
        R.verify(root, git=fake_git(root), expected=prot_of(root))


# ---- freeze and unfreeze -----------------------------------------------------------------------------------------
@pytest.mark.slow
def test_freeze_refuses_twice_and_launch_refuses_after_it(trained):
    git, prot = fake_git(trained), prot_of(trained)
    R.freeze(trained, git=git, expected=prot)
    with pytest.raises(R.LaunchError, match="guard 8: already frozen"):
        R.freeze(trained, git=git, expected=prot)
    with pytest.raises(R.LaunchError, match="guard 8: v2 is frozen"):
        train(trained)


@pytest.mark.slow
def test_freeze_refuses_a_member_changed_since_its_record(trained):
    p = trained / R.run_paths(35)["native_checkpoint"]
    os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
    p.write_bytes(b"changed")
    with pytest.raises(R.LaunchError, match="seed35.pt is missing or changed"):
        R.freeze(trained, git=fake_git(trained), expected=prot_of(trained))
    assert not (trained / R.MEMBERS).exists()


@pytest.fixture
def frozen(trained):
    R.freeze(trained, git=fake_git(trained), expected=prot_of(trained))
    return trained


@pytest.mark.slow
def test_unfreeze_moves_the_members_file_aside_and_launch_reopens(frozen):
    sha = hashlib.sha256((frozen / R.MEMBERS).read_bytes()).hexdigest()
    dest = R.unfreeze("defect in review", frozen, git=fake_git(frozen))
    assert dest == "logs/v2_recipe/aborted/freeze_1" and not (frozen / R.MEMBERS).exists()
    info = json.loads((frozen / dest / "abort.json").read_text(encoding="utf-8"))
    assert info["status"] == "complete" and info["members_sha256"] == sha and info["reason"] == "defect in review"
    assert hashlib.sha256((frozen / dest / "members_sha256.txt").read_bytes()).hexdigest() == sha
    runner = train(frozen)
    assert runner.calls == []  # every run already trained; nothing retrained
    R.freeze(frozen, git=fake_git(frozen), expected=prot_of(frozen))
    assert R.unfreeze("again", frozen, git=fake_git(frozen)).endswith("freeze_2")


ENTRY = ("# decisions\n\n2026-12-14, 10:05. v2 Atlantic ensemble frozen (launch commit L, 10 members). The\n"
         "members file's blob SHA-256 is ...\n")  # decisions.md's own entry form


@pytest.mark.slow
@pytest.mark.parametrize("kw, match", [
    (dict(fetch_rc=1), "git fetch origin failed"),
    (dict(on_origin=(R.MEMBERS,)), "the freeze PR has merged"),
    (dict(decisions={"origin/phase2-experiments": ENTRY}), "origin/phase2-experiments's logs/decisions.md already"),
    (dict(decisions={"HEAD": ENTRY, "origin/phase2-experiments": ENTRY}), "already holds .*the freeze PR has merged"),
    (dict(origin_tree=("results/hep/x.csv",)), "H-EP or clean-test output exists"),
    (dict(tree_rc=128), "can't read origin/phase2-experiments's tree"),
])
def test_unfreeze_refusals(frozen, kw, match):
    with pytest.raises(R.LaunchError, match=match):
        R.unfreeze("x", frozen, git=fake_git(frozen, **kw))
    assert (frozen / R.MEMBERS).exists()


@pytest.mark.slow
def test_unfreeze_refuses_later_outputs_on_disk_and_a_quoted_phrase_is_not_the_entry(frozen):
    git = fake_git(frozen, decisions={"origin/phase2-experiments":
                                      "2026-12-01, 09:00. Plan. The freeze PR adds an entry, \"v2 Atlantic ensemble\n"
                                      "frozen\", later; a body line naming v2 Atlantic ensemble frozen is not it.\n"})
    (frozen / "results/final_2026").mkdir(parents=True)
    with pytest.raises(R.LaunchError, match="H-EP or clean-test output exists: \\['results/final_2026'\\]"):
        R.unfreeze("x", frozen, git=git)
    (frozen / "results/final_2026").rmdir()
    R.unfreeze("x", frozen, git=git)  # the title in a body line, or under another entry's title, is not the entry
    with pytest.raises(R.LaunchError, match="nothing to withdraw"):
        R.unfreeze("x", frozen, git=git)


# ---- the selection copies agree with analyze_f31 (design §6) -----------------------------------------------------
ANALYZE_F31_SHA256 = "feee9bafabee141dd393a86c3c469b4cafd17c160c8f97076cab55da0f8016e1"


@pytest.fixture(scope="module")
def analyze_f31():
    """Imported in this test only, after hashing it and analyze_n5 (which it executes on import)."""
    for rel, pin in (("scripts/analyze_f31.py", ANALYZE_F31_SHA256), (R.ANALYZE_N5, R.ANALYZE_N5_SHA256)):
        assert hashlib.sha256((CHECKOUT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == pin, rel
    return load("analyze_f31_for_v2_test", CHECKOUT / "scripts" / "analyze_f31.py")


def f31_saved(seed):
    lines = (CHECKOUT / f"logs/f31/control_seed{seed}.log").read_text(encoding="utf-8").splitlines()
    return [int(re.match(r"epoch (\d+):", lines[i - 1]).group(1)) for i, ln in enumerate(lines)
            if ln.strip() == "saved" and i > 0]


def test_selection_agrees_with_analyze_f31_on_the_20_control_records(analyze_f31):
    A = analyze_f31
    for seed in range(10, 30):
        rec = json.loads((CHECKOUT / f"logs/f31/runs/control_seed{seed}.train.json").read_text(encoding="utf-8"))
        reval = {int(k): float(v) for k, v in rec["val_mae_revalidated"].items()}
        saved = f31_saved(seed)
        assert R.last_saved(saved, 99) == A.last_saved(saved, 99)
        assert R.native_pick(reval, saved, 99) == A.native_pick(reval, saved, 99) == (rec["slots"]["native_4x"], None)
        assert R.flagged(rec["slots"]["native_4x"], 99) == A.flagged(rec["slots"]["native_4x"], 99)
    assert all(R.flagged(e, 99) == A.flagged(e, 99) for e in range(1, 100))


def table(argmin_shift):
    """A synthetic 99-epoch table: train.py saved epochs 1, 10, 40; the re-validated first minimum is epoch 20,
    below epoch 40 by argmin_shift."""
    reval = {e: 10.0 - 0.01 * min(e, 40) for e in range(1, 100)}
    reval[20] = reval[40] - argmin_shift
    return reval, [1, 10, 40]


def test_selection_agrees_on_the_synthetic_tie_and_refusal_tables(analyze_f31):
    A = analyze_f31
    reval, saved = table(R.TIE_KT / 2)  # below TIE_KT: train.py's pick stands, with the tie record
    got, want = R.native_pick(reval, saved, 99), A.native_pick(reval, saved, 99)
    assert got == want and got[0] == 40 and got[1]["revalidated_argmin"] == 20 and got[1]["saved_epoch"] == 40
    for shift in (1.01 * R.TIE_KT, 2 * R.TIE_KT, 0.5):  # TIE_KT or more (1.01: clear of float rounding): both refuse
        reval, saved = table(shift)
        with pytest.raises(R.DesignError, match="not train.py's pick"):
            R.native_pick(reval, saved, 99)
        with pytest.raises(A.DesignError, match="not train.py's pick"):
            A.native_pick(reval, saved, 99)
    with pytest.raises(R.DesignError, match="no 'saved' line"):
        R.last_saved([], 99)


# ---- line endings and the CLI ------------------------------------------------------------------------------------
def test_every_committed_kind_is_stored_byte_for_byte():
    """design §5: `git check-attr text` is unset for an example of each committed kind; the lock is ignored."""
    assert REAL["check-attr"] == [f"{p}: text: unset" for p in COMMITTED_KINDS]
    assert REAL["ignored"] == {R.LOCK: 0, "logs/v2_recipe/seed30.log": 1}


def test_plan_runs_nothing_and_the_cli_refuses_bad_arguments(capsys, monkeypatch, no_real_subprocess):
    monkeypatch.chdir(os.getcwd())  # main() changes into ROOT; restored afterwards
    monkeypatch.setattr(sys, "path", list(sys.path))
    for stage in ("launch", "verify", "freeze", "abort_run", "unfreeze"):
        monkeypatch.setattr(R, stage, lambda *a, **k: pytest.fail("a writing stage was called"))
    R.main(["plan"])
    out = capsys.readouterr().out
    assert out.count("train: ['PY', '-u', 'scripts/train.py'") == 10 and no_real_subprocess == []
    for argv in (["launch", "30"], ["abort-run"], ["abort-run", "30"], ["unfreeze"], ["verify", "--reason", "x"],
                 ["evaluate"]):
        with pytest.raises(SystemExit):
            R.main(argv)


# ---- item 34: D1 (a record verifies at its own commit), D2, C1, C2, F1 -------------------------------------------
OTHER = "0ther" * 8


def docs_head(root, **kw):
    """HEAD = OTHER, a later commit on the chain that differs from the records' commit only in the decisions docs."""
    return fake_git(root, head=OTHER, first_parent=(OTHER, COMMIT), changed=("logs/decisions.md",), **kw)


@pytest.mark.slow
def test_d1_at_a_docs_only_head_abort_run_refuses_a_verified_seed_and_launch_skips_it(trained):
    git, prot = docs_head(trained), prot_of(trained)
    with pytest.raises(R.LaunchError, match="abort-run: seed 33 has a record that passes"):
        R.abort_run(33, "re-roll", trained, git=git, expected=prot)  # hc-adversary's probe 1: re-roll at head=OTHER
    runner = train(trained, git=git)
    assert runner.calls == []
    assert R.run_state(trained, 33, git, prot, {OTHER, COMMIT}) == "trained"


@pytest.mark.slow
@pytest.mark.parametrize("drift", ["src/hurricane_intensity/dataset.py", "scripts/train.py", "storms_val_expanded.txt"])
def test_d1_code_drift_makes_every_record_stale(trained, drift):
    git, prot = fake_git(trained, head=OTHER, first_parent=(OTHER, COMMIT), changed=(drift,)), prot_of(trained)
    assert R.run_state(trained, 30, git, prot, {OTHER, COMMIT}) == "stale"
    with pytest.raises(R.LaunchError, match=r"or are stale .*seeds \[30, 31, 32"):
        train(trained, git=git)
    assert R.abort_run(30, "stale", trained, git=git, expected=prot).endswith("seed30_attempt1")


@pytest.mark.slow
def test_d1_a_record_whose_commit_is_off_the_chain_is_stale(trained):
    git, prot = fake_git(trained, head=OTHER, first_parent=(OTHER,), changed=("logs/decisions.md",)), prot_of(trained)
    assert R.run_state(trained, 30, git, prot, {OTHER}) == "stale"
    R.abort_run(30, "off the chain", trained, git=git, expected=prot)


@pytest.mark.slow
def test_d1_no_new_run_is_trained_at_another_commit_than_the_verified_records(trained):
    R.abort_run(36, "bad", trained, git=fake_git(trained, head=OTHER, first_parent=(OTHER, COMMIT),
                                                 changed=("scripts/train.py",)), expected=prot_of(trained))
    with pytest.raises(R.LaunchError, match=f"guard 6: verified records come from launch commit .*relaunch at {COMMIT}"):
        train(trained, git=docs_head(trained))
    runner = train(trained)  # at the records' own commit
    assert runner.calls == [R.train_cmd(PY, 36)]


@pytest.mark.slow
def test_d1_abort_run_refuses_when_the_chain_cannot_be_read(trained):
    with pytest.raises(R.LaunchError, match="git fetch origin failed"):
        R.abort_run(30, "x", trained, git=fake_git(trained, fetch_rc=1), expected=prot_of(trained))


@pytest.mark.slow
def test_d2_an_entry_on_head_only_does_not_block_unfreeze(frozen):
    assert R.unfreeze("x", frozen, git=fake_git(frozen, decisions={"HEAD": ENTRY})).endswith("freeze_1")


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["verify", "freeze"])
@pytest.mark.parametrize("kw, match", [
    (dict(blobs={R.TRAIN_PY: b"other"}), "guard 2: HEAD's scripts/train.py"),
    (dict(src={**R.SRC_BLOBS, "dataset.py": "0" * 40}), r"guard 2: .*differs: \['dataset.py'\]"),
    (dict(first_parent=("9" * 40,)), "is not on the first-parent chain"),
])
def test_c1_verify_and_freeze_apply_the_code_guards_and_the_chain(trained, stage, kw, match):
    with pytest.raises(R.LaunchError, match=match):
        getattr(R, stage)(trained, git=fake_git(trained, **kw), expected=prot_of(trained))
    assert not (trained / R.MEMBERS).exists()


def direct_ctx(root):
    return dict(commit=COMMIT, expected=prot_of(root), forbidden=set(), git=fake_git(root), chain={COMMIT})


def direct_hooks(root):
    return dict(runner=FakeRunner(root), revalidate_fn=reval_from(), snapshot=lambda: "gpu", same_weights_fn=same_bytes)


def test_c1_train_one_refuses_to_overwrite(tmp_path):
    root = fake_repo(tmp_path)
    (root / "models/v2_recipe").mkdir(parents=True)
    (root / R.run_paths(30)["native_checkpoint"]).write_bytes(b"old")
    hooks = direct_hooks(root)
    with pytest.raises(R.LaunchError, match="refusing to overwrite"):
        R.train_one(30, direct_ctx(root), root, hooks)
    assert hooks["runner"].calls == [] and (root / R.run_paths(30)["native_checkpoint"]).read_bytes() == b"old"


@pytest.mark.skipif(not hasattr(__import__("_winapi"), "CreateJunction") if sys.platform == "win32" else True,
                    reason="needs a Windows junction")
def test_c1_an_epoch_dir_resolving_outside_the_repo_is_refused(tmp_path):
    import _winapi
    root = fake_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    hooks = direct_hooks(root)
    inner = hooks["runner"]

    def runner(cmd, **k):  # train.py's --epoch-dir turns out to be a junction to a folder outside the repo
        _winapi.CreateJunction(str(outside), str(root / R.run_paths(30)["epoch_dir"]))
        return inner(cmd, **k)
    hooks["runner"] = runner
    with pytest.raises(R.LaunchError, match="resolves to .*outside .*nothing deleted"):
        R.train_one(30, direct_ctx(root), root, hooks)
    assert len(list(outside.iterdir())) == 99


@pytest.mark.slow
def test_c1_kept_files_and_the_members_file_are_read_only(trained):
    for s in SEEDS:
        for k in ("train_log", "native_checkpoint"):
            assert not os.stat(trained / R.run_paths(s)[k]).st_mode & stat.S_IWRITE, (s, k)
    R.freeze(trained, git=fake_git(trained), expected=prot_of(trained))
    assert not os.stat(trained / R.MEMBERS).st_mode & stat.S_IWRITE


def test_c1_train_one_makes_its_kept_files_read_only(tmp_path):
    root = fake_repo(tmp_path)
    R.train_one(30, direct_ctx(root), root, direct_hooks(root))
    for k in ("train_log", "native_checkpoint"):
        assert not os.stat(root / R.run_paths(30)[k]).st_mode & stat.S_IWRITE, k


@pytest.mark.parametrize("entry, is_dir", [("extra.py", False), ("old_dataset.pyc", False), ("sub", True)])
def test_f1_an_untracked_or_ignored_entry_in_the_package_folder_refuses(tmp_path, entry, is_dir):
    root = fake_repo(tmp_path)
    p = root / R.SRC_DIR / entry
    p.mkdir() if is_dir else p.write_text("x", encoding="utf-8")
    with pytest.raises(R.LaunchError, match=f"guard 2: .*untracked or ignored entries \\['{re.escape(entry)}'\\]"):
        train(root)


@pytest.mark.slow
def test_c2_unfreeze_retrain_refreeze_on_the_freeze_branch(frozen):
    """design §4's path, by the module docstring's steps: B is the freeze branch, L = COMMIT."""
    prot, B = prot_of(frozen), "b" * 40
    outputs = tuple(v for s in SEEDS for k, v in R.run_paths(s).items() if k in ("train_log", "record"))
    on_b = dict(head=B, first_parent=(COMMIT,), changed=outputs + (R.MEMBERS, "logs/decisions.md"),
                decisions={"HEAD": ENTRY})
    # 1. on B: unfreeze (the entry is only on B's HEAD, the members file only tracked on B)
    R.unfreeze("seed 34's log was found damaged", frozen, git=fake_git(frozen, **on_b))
    # the defect: seed 34's record no longer verifies
    rewrite_record(frozen, 34, native_epoch=1)
    on_b2 = dict(on_b, changed=on_b["changed"] + ("logs/v2_recipe/aborted/freeze_1/abort.json",))
    with pytest.raises(R.LaunchError, match="guard 1: .* is not on the first-parent chain"):
        train(frozen, git=fake_git(frozen, **on_b2))  # a launch on B itself refuses
    with pytest.raises(R.LaunchError, match="abort-run: seed 33 has a record that passes"):
        R.abort_run(33, "x", frozen, git=fake_git(frozen, **on_b2), expected=prot)
    # 3-4. reset --mixed L: HEAD = L, the records stay on disk untracked; 5. abort-run the failed seed at L
    R.abort_run(34, "record fails check_training", frozen, git=fake_git(frozen), expected=prot)
    # 6. launch at L retrains seed 34 alone; 7. verify and freeze
    runner = train(frozen)
    assert runner.calls == [R.train_cmd(PY, 34)]
    assert record(frozen, 34)["aborted_attempts"] == ["logs/v2_recipe/aborted/seed34_attempt1/abort.json"]
    R.verify(frozen, git=fake_git(frozen), expected=prot)
    R.freeze(frozen, git=fake_git(frozen), expected=prot)
    # 8-9. back on B: the records still verify there
    R.verify(frozen, git=fake_git(frozen, **on_b2), expected=prot)
