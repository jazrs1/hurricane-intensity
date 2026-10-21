"""v2 Atlantic recipe driver: the 10 training runs, their training-only checks and the freeze
(reports/v2_survey/v2_recipe/design.md, approved by John; §4-§6).

    python scripts/run_v2_recipe.py plan                          # print every command; runs nothing, writes nothing
    python scripts/run_v2_recipe.py launch                        # guards 1-9, then train seeds 30-39 in order; resumable
    python scripts/run_v2_recipe.py verify                        # every training-only check on the 10 records; writes nothing
    python scripts/run_v2_recipe.py freeze                        # write results/v2_recipe/members_sha256.txt
    python scripts/run_v2_recipe.py abort-run SEED --reason TEXT  # move an unfinished or unverifiable run aside
    python scripts/run_v2_recipe.py unfreeze --reason TEXT        # withdraw the members file, only before the freeze PR merges

The recipe is the frozen ensemble's configuration at F-31's 4x budget: storms_train_expanded.txt (2081 frames after
excluded_crops.txt), storms_val_expanded.txt (940), 99 epochs = analyze_n5.epochs_for(2081, 4 x 1640) (66 batches,
6534 steps), native selection (train.py's own --out, the first strict minimum of validation MAE), seeds 30-39.
Nothing is evaluated on any held-out set: the driver starts no subprocess but train.py, with exactly the design §6
argument list. The members file is never handed to any tool as an output path (design §4).

Imports (design §6): only scripts/run_n5.py and scripts/analyze_n5.py, each hashed (CRLF read as LF) against its pin
before it is executed, then compiled and executed from exactly the hashed bytes, with __file__ set first (run_n5
derives its ROOT from __file__) and no .pyc. run_f31 and analyze_f31 are not imported; their lock, record, abort and
log-parser helpers are rewritten here on the v2 paths, and the selection functions last_saved, native_pick and
flagged are copied from analyze_f31.py (a test checks that the copies agree with the originals).

launch refuses before the first run unless (design §6):
  1. the tracked tree is clean; this driver, run_n5.py, analyze_n5.py and train.py equal their HEAD blobs; after
     `git fetch origin`, HEAD is on origin/phase2-experiments' first-parent chain (so it is pushed);
  2. HEAD's train.py, run_n5.py and analyze_n5.py are the pinned blobs, and src/hurricane_intensity/ at HEAD holds
     exactly fc71b98's nine .py blobs;
  3. the 14 protected files match the COORDINATOR.md block (and analyze_n5.PROTECTED); checked again before and
     after every run;
  4. HEAD's excluded_crops.txt, storms_train_expanded.txt and storms_val_expanded.txt are fc71b98's blobs, and
     epochs_for(2081, 6560) == (99, 66);
  5. the train and validation lists are analyze_n5's CONTROL_IDS and VAL_IDS, share no ID with H9, the 2025 test or
     H-EP, and name no 2026 storm;
  6. every output path is under models/v2_recipe/ or logs/v2_recipe/, or is the members file; no unfinished run,
     unverifiable record or pending abort is on disk (a seed<s>.pt exists only with a record that verifies);
  7. each command is exactly design §6's train.py list for its seed, with no held-out, sealed or 2026 argument;
  8. no members file exists: on disk, tracked, or on origin/phase2-experiments;
  9. the root is the main checkout (a .git directory, not a worktree's .git file); a child python imports
     hurricane_intensity from <root>/src; CUDA is available.

Per run: train to 99 epochs with --epoch-dir; check the train log; re-validate every epoch on validation only
(run_n5.revalidate); the native pick must be the re-validated first minimum, or within TIE_KT of it (tie recorded);
seed<s>.pt must be tensor-equal to the pick's epoch file; every epoch file is hashed into deleted_epoch_sha256 and
deleted; the record logs/v2_recipe/runs/seed<s>.train.json is written, and the train log and seed<s>.pt become
read-only. One lock file (logs/v2_recipe/run_v2.lock) serialises every stage that writes.

Aborts move files unchanged to logs/v2_recipe/aborted/seed<s>_attempt<k>/ or aborted/freeze_<n>/, after writing
abort.json there with every file's SHA-256 and status "pending"; the status becomes "complete" once all are moved, and
a pending abort is finished by running the same command again (every other writing stage refuses while one is
pending).
"""
import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/run_v2_recipe.py"
RUN_N5 = "scripts/run_n5.py"
RUN_N5_SHA256 = "e8f13e06b80c29101ef1a4efaeac7aaa17e5922f0a8bebdc26a19f270d6d3ed7"  # blob 2ebd8cea (F-31's pin)
ANALYZE_N5 = "scripts/analyze_n5.py"
ANALYZE_N5_SHA256 = "0130641583cf09c8abb5791b9ea9467fd6410b953bc8a57c3a2ae68fb3bb1288"  # blob 679e2169
TRAIN_PY = "scripts/train.py"
TRAIN_PY_SHA256 = "afceb4b19ea7d8c732c6f26b7d52e4b3b6d313d58153e2d0518887a3a6db231f"  # blob 500bbc1e (F-31's pin)
SRC_DIR = "src/hurricane_intensity"
SRC_BLOBS = {  # fc71b98 (F-31's launch commit): git ls-tree fc71b98 src/hurricane_intensity/
    "__init__.py": "7a6fe6399245c5e368ffd7f648d8aff589527f21",
    "config.py": "305d3c4947141fe97fcb3634b3b22d614894a889",
    "crop.py": "101be8fcc4be79ec9d9523cf2d8107f7eb1075a7",
    "dataset.py": "458ed363d48b50b672b69792827cb5b82c6a316b",
    "goes_index.py": "bba3b98f65838b68028bf545d833e6e70fd78815",
    "hurdat2.py": "11dee801471a6ed5b86a6872bbdf840f6e447dfa",
    "pipeline.py": "58746c606bd3b3beb3ad5c6d5d271ee4abd11305",
    "projection.py": "29cc9b6c08ecd93d58c773165b606b8688fb5b6d",
    "splits.py": "355f5f63e37b3f31c504c71dce7e745560c7a0f9",
}
EXCLUDED_CROPS = "excluded_crops.txt"
EXCLUDED_CROPS_BLOB = "cc9dfcff4036fcbefe5d2ec1a6488fa2feef8f5f"
EXCLUDED_CROPS_SHA256 = "102009773f843410149f8f8cadeb1b3c981252fcb8d7c53cc8d43e018df00cea"
TRAIN_LIST = "storms_train_expanded.txt"
TRAIN_LIST_BLOB = "12a20a1499cafa50a753a6f5f7da3f4439c8fa8f"  # = fc71b98's
VAL_LIST = "storms_val_expanded.txt"
VAL_LIST_BLOB = "ead8004f39712f43f9f787e1851de6092c35abab"  # = fc71b98's
# Guard 5 only: read for their storm IDs (the I5 forbidden sets), never passed to any command (design §3 name check).
FORBIDDEN_LISTS = ("storms_holdout_h9.txt", "storms_test_2025.txt", "storms_holdout_hep.txt")
SEALED_ROOTS = ("data/hep_sealed", "data/2026_sealed")
SEEDS = tuple(range(30, 40))
EPOCHS, BATCHES, TARGET_STEPS = 99, 66, 4 * 1640  # design §1: epochs_for(2081, 6560) == (99, 66)
UPSTREAM = "origin/phase2-experiments"
ABORTED = "logs/v2_recipe/aborted"
LOCK = "logs/v2_recipe/run_v2.lock"
MEMBERS = "results/v2_recipe/members_sha256.txt"
OUTPUT_ROOTS = ("models/v2_recipe/", "logs/v2_recipe/")
DECISIONS = "logs/decisions.md"
FROZEN_ENTRY = re.compile(r"^#+ .*v2 Atlantic ensemble frozen", re.M)  # the freeze entry's heading (design §4)
LATER_OUTPUTS = ("results/hep", "data/hep_sealed", "results/final_2026", "data/2026_sealed")  # H-EP or clean test
COORDINATOR = "COORDINATOR.md"
HASH_HEADING = "## Protected hashes (check.py source of truth)"
TRAIN_KEYS = ("train_log", "native_checkpoint", "epoch_dir", "record")
HEX64 = re.compile(r"[0-9a-f]{64}")
ID_2026 = re.compile(r"[A-Z]{2}\d{2}2026")


class LaunchError(SystemExit):
    pass


def load_pinned(name, path, sha):
    """Hash BEFORE executing: the file's bytes, with CRLF read as LF (the working copy under core.autocrlf), must be
    the pin; anything else is refused unexecuted. The module is compiled and executed from exactly the hashed bytes
    (no re-read, no .pyc), with __file__ set first."""
    data = Path(path).read_bytes()
    got = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    if got != sha:
        raise LaunchError(f"{path} hashes to {got[:12]}..., not the pinned {sha[:12]}...; refused before import")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)  # both have a __main__ guard; executing runs nothing
    return module


RN5 = load_pinned("run_n5_for_v2", Path(__file__).resolve().parent / "run_n5.py", RUN_N5_SHA256)
AN5 = load_pinned("analyze_n5_for_v2", Path(__file__).resolve().parent / "analyze_n5.py", ANALYZE_N5_SHA256)
sha256_file, protected_now, same_weights = RN5.sha256_file, RN5.protected_now, RN5.same_weights
write_json_new, write_json_replace, run_step, child_env = (RN5.write_json_new, RN5.write_json_replace, RN5.run_step,
                                                           RN5.child_env)
write_line, now_iso, nvidia_snapshot, cuda_available = RN5.write_line, RN5.now_iso, RN5.nvidia_snapshot, RN5.cuda_available
child_package_path, real_git, revalidate, doc_path = (RN5.child_package_path, RN5.real_git, RN5.revalidate,
                                                      RN5.doc_path)
epochs_for, TIE_KT, UNDERTRAIN_FRACTION = AN5.epochs_for, AN5.TIE_KT, AN5.UNDERTRAIN_FRACTION
read_text_strict, ids_in, PROTECTED, DesignError = AN5.read_text_strict, AN5.ids_in, AN5.PROTECTED, AN5.DesignError
VAL_IDS, VAL_FRAMES, CONTROL_IDS, CONTROL_TRAIN_FRAMES = AN5.VAL_IDS, AN5.VAL_FRAMES, AN5.CONTROL_IDS, AN5.CONTROL_TRAIN_FRAMES


# ---- selection (validation only), copied from analyze_f31.py:153-211 -----------------------------------------------
def last_saved(saved, e_k):
    """train.py's own pick at budget E_k: its last 'saved' line at an epoch <= E_k (running minimum, unrounded)."""
    s = [e for e in saved if e <= e_k]
    if not s:
        raise DesignError(f"no 'saved' line at or before epoch {e_k}")
    return max(s)


def native_pick(reval, saved, e_k):
    """(epoch, tie record or None). The first re-validated minimum over 1..E_k must be train.py's pick at E_k. When
    it is not, but the two differ by less than TIE_KT, train.py's pick stands (N5 amendment (a)); otherwise refuse."""
    argmin = min(range(1, e_k + 1), key=lambda e: (reval[e], e))
    pick = last_saved(saved, e_k)
    if argmin == pick:
        return pick, None
    delta = reval[pick] - reval[argmin]
    if not abs(delta) < TIE_KT:
        raise DesignError(f"re-validated minimum over epochs 1-{e_k} is epoch {argmin}, not train.py's pick {pick}: "
                          "the re-validation path does not reproduce training")
    return pick, {"revalidated_argmin": argmin, "saved_epoch": pick, "delta_kt": delta}


def flagged(epoch, e_k):
    """Under-training flag (N5's rule): the selected epoch lies in the last 15% of the budget."""
    return epoch > UNDERTRAIN_FRACTION * e_k


# ---- paths and commands ------------------------------------------------------------------------------------------
def order_of(seed):
    return SEEDS.index(seed) + 1  # 30 -> 1, ..., 39 -> 10


def run_paths(seed):
    return {"train_log": f"logs/v2_recipe/seed{seed}.log", "native_checkpoint": f"models/v2_recipe/seed{seed}.pt",
            "epoch_dir": f"models/v2_recipe/seed{seed}_epochs", "record": f"logs/v2_recipe/runs/seed{seed}.train.json"}


def epoch_file(seed, e):
    return f"models/v2_recipe/seed{seed}_epochs/epoch_{e:02d}.pt"


def member_paths():
    return [run_paths(s)["native_checkpoint"] for s in SEEDS]


def train_cmd(py, seed):
    """Design §6, exactly: no --frame-list, --circular-mask or --rotate."""
    p = run_paths(seed)
    return [py, "-u", TRAIN_PY, "--train-list", TRAIN_LIST, "--val-list", VAL_LIST, "--epochs", str(EPOCHS),
            "--seed", str(seed), "--loss", "l1", "--out", p["native_checkpoint"], "--epoch-dir", p["epoch_dir"]]


def check_command(cmd, py, seed, forbidden_ids):
    """Guard 7: the command is design §6's list for this seed, and no argument names a held-out or sealed set."""
    if cmd != train_cmd(py, seed):
        raise LaunchError(f"guard 7: seed {seed}: refusing a command that is not design §6's train.py list: {cmd}")
    for arg in cmd[1:]:
        named = sorted(set(ids_in(arg)) & forbidden_ids) + ID_2026.findall(arg)
        if named or any(r in arg.replace("\\", "/") for r in SEALED_ROOTS) or arg in FORBIDDEN_LISTS:
            raise LaunchError(f"guard 7: seed {seed}: argument {arg!r} names a held-out, sealed or 2026 set")


def check_output_paths(paths):
    """Guard 6: only the v2_recipe roots and the members file are written (design §5)."""
    bad = [p for p in paths if not (p.startswith(OUTPUT_ROOTS) or p == MEMBERS) or ".." in Path(p).parts]
    if bad:
        raise LaunchError(f"guard 6: refusing output paths outside {OUTPUT_ROOTS} and {MEMBERS}: {bad}")


def planned_outputs():
    out = [v for s in SEEDS for v in run_paths(s).values()]
    out += [epoch_file(s, e) for s in SEEDS for e in range(1, EPOCHS + 1)]
    return out + [ABORTED + "/", LOCK, MEMBERS]


# ---- the train log and the record -------------------------------------------------------------------------------
def log_header(seed):
    return f"RUN v2_recipe seed={seed} order={order_of(seed)} list={TRAIN_LIST} val={VAL_LIST} epochs={EPOCHS}"


def parse_train_log(path, seed):
    """The v2 train log (this driver's header, train.py's stdout, this driver's END footer). Returns
    ([(epoch, train, val)], [saved epochs])."""
    txt = read_text_strict(path)
    lines = txt.splitlines()
    ep = [(int(e), float(tr), float(va)) for e, tr, va in
          re.findall(r"^epoch (\d+): train ([\d.]+) kt\s+val ([\d.]+) kt", txt, flags=re.M)]
    saved = [int(m.group(1)) for i, ln in enumerate(lines) if ln.strip() == "saved" and i > 0
             for m in [re.match(r"epoch (\d+):", lines[i - 1])] if m]
    m_seed = re.search(r"^seed: (\d+)\s+checkpoint: (\S+)", txt, flags=re.M)
    m_val = re.search(r"^val storms:\s+(\[.*\])", txt, flags=re.M)
    m_tr = re.search(r"^train storms: (\[.*\])", txt, flags=re.M)
    m_n = re.findall(r"^train samples: (\d+)\s+val samples: (\d+)", txt, flags=re.M)
    flags = re.search(r"loss: (\w+)\s+circular_mask: (\w+)\s+rotate: (\w+)", txt)
    problems = []
    if not lines or lines[0] != log_header(seed) or len(lines) < 2 or not lines[1].startswith("START "):
        problems.append("driver header")
    if not m_seed or int(m_seed.group(1)) != seed:
        problems.append("seed")
    if not m_seed or Path(m_seed.group(2)).as_posix() != run_paths(seed)["native_checkpoint"]:
        problems.append("checkpoint path")
    if not m_val or ids_in(m_val.group(1)) != VAL_IDS:
        problems.append("validation storms")
    if not m_tr or sorted(ids_in(m_tr.group(1))) != CONTROL_IDS:
        problems.append("training storm set")
    if not m_n or any((int(a), int(b)) != (CONTROL_TRAIN_FRAMES, VAL_FRAMES) for a, b in m_n):
        problems.append(f"train/val frame counts (need 'train samples: {CONTROL_TRAIN_FRAMES}  val samples: "
                        f"{VAL_FRAMES}')")
    if not flags or flags.groups() != ("l1", "False", "False"):
        problems.append("loss/mask/rotation")
    if re.search(r"^frame list:", txt, flags=re.M):
        problems.append("a frame list was used")
    if [e for e, _, _ in ep] != list(range(1, EPOCHS + 1)):
        problems.append(f"epochs ({len(ep)} logged, 1-{EPOCHS} expected in order)")
    if not saved:
        problems.append("no 'saved' line")
    nonblank = [ln for ln in lines if ln.strip()]
    if "TRAIN_FAILED" in txt or not nonblank or not nonblank[-1].startswith("END "):
        problems.append("TRAIN_FAILED or no END footer")
    if problems:
        raise DesignError(f"{path}: does not match the design: {', '.join(problems)}")
    return ep, saved


def abort_files(root, seed, kind="attempt"):
    """abort.json paths of one seed's training aborts (or, seed None, the freeze aborts), in attempt order."""
    pat, sep = (f"seed{seed}_attempt*", "attempt") if seed is not None else ("freeze_*", "freeze_")
    found = Path(root).glob(f"{ABORTED}/{pat}/abort.json")
    return sorted(found, key=lambda f: int(f.parent.name.rsplit(sep, 1)[1]))


def aborted_attempts(root, seed):
    return [f.relative_to(root).as_posix() for f in abort_files(root, seed)]


def pending_aborts(root):
    return sorted(f.relative_to(root).as_posix() for f in Path(root).glob(f"{ABORTED}/*/abort.json")
                  if json.loads(f.read_text(encoding="utf-8")).get("status") != "complete")


def check_training(rec, seed, root, commit, expected):
    """Every training-only check of design §6 on one record. Raises DesignError; returns a summary row."""
    root, where = Path(root), run_paths(seed)["record"]
    want = {"arm": "control", "seed": seed, "order": order_of(seed), "train_list": TRAIN_LIST, "val_list": VAL_LIST,
            "frame_list_sha256": None, "epochs": EPOCHS, "commit": commit, "train_py_sha256": TRAIN_PY_SHA256,
            "native_same_weights": True, **{k: v for k, v in run_paths(seed).items() if k != "record"}}
    bad = sorted(k for k, v in want.items() if rec.get(k, "<missing>") != v)
    if bad:
        raise DesignError(f"{where}: {', '.join(f'{k}={rec.get(k)!r} (want {want[k]!r})' for k in bad)}")
    for when in ("protected_before", "protected_after"):
        if rec.get(when) != expected:
            raise DesignError(f"{where}: {when} protected hashes differ from the block")
    listed = rec.get("aborted_attempts")
    if listed != aborted_attempts(root, seed):
        raise DesignError(f"{where}: aborted_attempts {listed} are not the abort directories on disk "
                          f"{aborted_attempts(root, seed)}")
    for rel in listed:
        if json.loads(read_text_strict(root / rel)).get("status") != "complete":
            raise DesignError(f"{where}: aborted attempt {rel} is not complete")
    ep, saved = parse_train_log(root / run_paths(seed)["train_log"], seed)
    reval = {int(k): float(v) for k, v in (rec.get("val_mae_revalidated") or {}).items()}
    logged = {e: va for e, _, va in ep}
    if sorted(reval) != list(range(1, EPOCHS + 1)) or any(abs(round(v, 2) - logged[e]) > 0.0051
                                                          for e, v in reval.items()):
        raise DesignError(f"{where}: re-validated values missing or inconsistent with the train log")
    try:
        pick, tie = native_pick(reval, saved, EPOCHS)
    except DesignError as e:
        raise DesignError(f"{where}: {e}") from None
    if rec.get("native_epoch") != pick or rec.get("tie_rule") != tie:
        raise DesignError(f"{where}: native_epoch {rec.get('native_epoch')} / tie_rule {rec.get('tie_rule')} are not "
                          f"the re-validated {pick} / {tie}")
    if rec.get("under_trained") is not flagged(pick, EPOCHS):
        raise DesignError(f"{where}: under_trained must be {flagged(pick, EPOCHS)}")
    p = run_paths(seed)
    files = rec.get("files_sha256") or {}
    if set(files) != {p["train_log"], p["native_checkpoint"]}:
        raise DesignError(f"{where}: files_sha256 must name exactly the train log and seed{seed}.pt")
    for rel, h in files.items():
        if not (root / rel).is_file() or sha256_file(root / rel) != h:
            raise DesignError(f"{where}: {rel} is missing or changed since training")
    deleted = rec.get("deleted_epoch_sha256") or {}
    if set(deleted) != {str(e) for e in range(1, EPOCHS + 1)} or not all(
            isinstance(h, str) and HEX64.fullmatch(h) for h in deleted.values()):
        raise DesignError(f"{where}: deleted_epoch_sha256 must hash all {EPOCHS} epoch files")
    left = [epoch_file(seed, e) for e in range(1, EPOCHS + 1) if (root / epoch_file(seed, e)).exists()]
    if left:
        raise DesignError(f"{where}: epoch files still on disk: {left[:3]}")
    return {"seed": seed, "order": order_of(seed), "native_epoch": pick, "val_at_pick": reval[pick],
            "under_trained": flagged(pick, EPOCHS), "tie_rule": tie}


def load_record(root, seed):
    rel = run_paths(seed)["record"]
    try:
        return json.loads(read_text_strict(Path(root) / rel))
    except ValueError as e:
        raise DesignError(f"{rel} is not JSON ({e})") from None


def run_state(root, seed, commit, expected):
    """fresh (nothing on disk), partial (files but no record), bad_record (a record that fails a check) or trained."""
    p = run_paths(seed)
    if not (Path(root) / p["record"]).exists():
        return "partial" if any((Path(root) / v).exists() for v in p.values()) else "fresh"
    try:
        check_training(load_record(root, seed), seed, root, commit, expected)
    except DesignError:
        return "bad_record"
    return "trained"


class Lock:
    """One run_v2_recipe stage that writes at a time. A stale lock (a killed process) is removed by hand, after
    checking that none is running."""

    def __init__(self, root):
        self.path = Path(root) / LOCK

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise LaunchError(f"{LOCK} exists: another run_v2_recipe stage is running (or one was killed; remove the "
                              "file by hand only after checking that none is running)") from None
        os.write(fd, f"pid {os.getpid()} {now_iso()}\n".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink()


# ---- guards ------------------------------------------------------------------------------------------------------
def coordinator_block(path):
    """(path -> sha256) from the first ```sha256 block under HASH_HEADING, read as check.load_expected reads it."""
    path = Path(path)
    if not path.is_file():
        raise LaunchError(f"guard 3: {path} is missing: the source of truth for the protected hashes")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == HASH_HEADING)
        fence = next(i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("```"))
        end = next(i for i in range(fence + 1, len(lines)) if lines[i].strip() == "```")
    except StopIteration:
        raise LaunchError(f"guard 3: {path} has no ```sha256 block under {HASH_HEADING!r}") from None
    if lines[fence].strip() != "```sha256":
        raise LaunchError(f"guard 3: {path} has no ```sha256 block under {HASH_HEADING!r}")
    out = {}
    for ln in lines[fence + 1:end]:
        parts = ln.split()
        if not parts:
            continue
        if len(parts) != 2 or not HEX64.fullmatch(parts[0]):
            raise LaunchError(f"guard 3: {path}: malformed hash line {ln!r}")
        out[parts[1].lstrip("*").replace("\\", "/")] = parts[0]
    return out


def git_sha(git, spec):
    return hashlib.sha256(git(["show", spec]).stdout).hexdigest()


def git_blob(git, spec):
    r = git(["rev-parse", spec], check=False)
    return r.stdout.decode().strip() if r.returncode == 0 else None


def first_parent(git, commit, guard):
    """After a fetch, commit is on origin/phase2-experiments' first-parent chain (a merge commit of a reviewed PR)."""
    if git(["fetch", "origin"], check=False).returncode != 0:
        raise LaunchError(f"guard {guard}: git fetch origin failed; can't check {commit} against the remote")
    chain = git(["rev-list", "--first-parent", UPSTREAM], check=False)
    if chain.returncode != 0 or commit not in chain.stdout.decode().split():
        raise LaunchError(f"guard {guard}: {commit} is not on the first-parent chain of {UPSTREAM}; launch only from "
                          "a pushed merge commit of a reviewed PR")


def list_ids(root, name, guard):
    p = Path(root) / name
    if not p.is_file():
        raise LaunchError(f"guard {guard}: {name} is missing")
    return [s for ln in read_text_strict(p).splitlines() for s in ids_in(ln.split("#", 1)[0])]  # comments skipped


def code_and_data_guards(root, git, expected):
    """Guards 1 (tree), 2, 3, 4 and 5, shared by every stage that writes or verifies."""
    if git(["status", "--porcelain", "--untracked-files=no"]).stdout.strip():
        raise LaunchError("guard 1: the tracked tree has uncommitted changes; run from a clean commit")
    for rel in (SELF, RUN_N5, ANALYZE_N5, TRAIN_PY):
        if git(["diff", "--quiet", "HEAD", "--", rel], check=False).returncode != 0:
            raise LaunchError(f"guard 1: {rel} differs from its committed blob at HEAD")
    for rel, pin in ((TRAIN_PY, TRAIN_PY_SHA256), (RUN_N5, RUN_N5_SHA256), (ANALYZE_N5, ANALYZE_N5_SHA256)):
        if git_sha(git, f"HEAD:{rel}") != pin:
            raise LaunchError(f"guard 2: HEAD's {rel} is not the pinned blob {pin[:12]}...")
    tree = {}
    for ln in git(["ls-tree", "HEAD", f"{SRC_DIR}/"]).stdout.decode().splitlines():
        meta, name = ln.split("\t", 1)
        tree[Path(name).name] = meta.split()[2] if meta.split()[1] == "blob" else f"<{meta.split()[1]}>"
    if tree != SRC_BLOBS:
        diff = sorted(k for k in set(tree) | set(SRC_BLOBS) if tree.get(k) != SRC_BLOBS.get(k))
        raise LaunchError(f"guard 2: {SRC_DIR}/ at HEAD is not fc71b98's nine blobs (differs: {diff})")
    block = coordinator_block(Path(root) / COORDINATOR)
    if block != expected:
        raise LaunchError("guard 3: analyze_n5.PROTECTED differs from the COORDINATOR.md protected block")
    absent = sorted(k for k in expected if not (Path(root) / k).is_file())
    if absent:
        raise LaunchError(f"guard 3: protected files missing under {root}: {absent[:3]} (run from the main checkout)")
    now = protected_now(root, expected)
    if now != expected:
        raise LaunchError(f"guard 3: protected files differ: {sorted(k for k in expected if now[k] != expected[k])}")
    if git_blob(git, f"HEAD:{EXCLUDED_CROPS}") != EXCLUDED_CROPS_BLOB or \
            git_sha(git, f"HEAD:{EXCLUDED_CROPS}") != EXCLUDED_CROPS_SHA256:
        raise LaunchError(f"guard 4: HEAD's {EXCLUDED_CROPS} is not blob {EXCLUDED_CROPS_BLOB[:8]} (the excluded_crops pin)")
    for rel, pin in ((TRAIN_LIST, TRAIN_LIST_BLOB), (VAL_LIST, VAL_LIST_BLOB)):
        if git_blob(git, f"HEAD:{rel}") != pin:
            raise LaunchError(f"guard 4: HEAD's {rel} is not fc71b98's blob {pin[:8]}")
    if epochs_for(CONTROL_TRAIN_FRAMES, TARGET_STEPS) != (EPOCHS, BATCHES) or CONTROL_TRAIN_FRAMES != 2081:
        raise LaunchError(f"guard 4: epochs_for({CONTROL_TRAIN_FRAMES}, {TARGET_STEPS}) is not ({EPOCHS}, {BATCHES})")
    train, val = list_ids(root, TRAIN_LIST, 5), list_ids(root, VAL_LIST, 5)
    if sorted(s for s in train if s not in val) != CONTROL_IDS or val != VAL_IDS:  # load_splits drops val from train
        raise LaunchError("guard 5: the train or validation list is not analyze_n5's CONTROL_IDS / VAL_IDS")
    forbidden = set()
    for name in FORBIDDEN_LISTS:
        ids = list_ids(root, name, 5)
        if not ids:
            raise LaunchError(f"guard 5: {name} names no storm")
        overlap = sorted(set(ids) & set(train + val))
        if overlap:
            raise LaunchError(f"guard 5: the train or validation list shares {overlap} with {name}")
        forbidden |= set(ids)
    late = [s for s in train + val if s.endswith("2026")]
    if late:
        raise LaunchError(f"guard 5: the train or validation list names 2026 storms {late}")
    return forbidden


def environment_guards(root, cuda_ok, child_import):
    """Guard 9 (as F-31's common_guards): the .venv's editable install points at the main checkout's src."""
    if not (Path(root) / ".git").is_dir():
        raise LaunchError(f"guard 9: {root} is not the main checkout (a worktree has a .git file)")
    pkg = child_import(root)
    if not pkg or not Path(pkg).resolve().is_relative_to((Path(root) / "src").resolve()):
        raise LaunchError(f"guard 9: a child python imports hurricane_intensity from {pkg or '(nowhere)'}, not "
                          f"{Path(root) / 'src'}")
    if not cuda_ok():
        raise LaunchError("guard 9: torch.cuda.is_available() is False; the runs need the GPU")


def frozen_evidence(root, git):
    """Guard 8: the members file on disk, tracked, or on origin/phase2-experiments."""
    found = [f"disk:{MEMBERS}"] if (Path(root) / MEMBERS).exists() else []
    if git(["ls-files", "--error-unmatch", MEMBERS], check=False).returncode == 0:
        found.append(f"tracked:{MEMBERS}")
    if git(["cat-file", "-e", f"{UPSTREAM}:{MEMBERS}"], check=False).returncode == 0:
        found.append(f"{UPSTREAM}:{MEMBERS}")
    return found


def preflight(root, git, expected, cuda_ok, child_import):
    """launch's guards, all before the first run (module docstring)."""
    environment_guards(root, cuda_ok, child_import)
    forbidden = code_and_data_guards(root, git, expected)
    frozen = frozen_evidence(root, git)
    if frozen:
        raise LaunchError(f"guard 8: v2 is frozen ({frozen}); launch refuses. Only `unfreeze` can reopen it, and "
                          "only before the freeze PR merges")
    pending = pending_aborts(root)
    if pending:
        raise LaunchError(f"guard 6: unfinished aborts {pending}; run the same abort command again to finish them")
    check_output_paths(planned_outputs())
    commit = git(["rev-parse", "HEAD"]).stdout.decode().strip()
    states = {s: run_state(root, s, commit, expected) for s in SEEDS}
    bad = [s for s, st in states.items() if st == "bad_record"]
    if bad:
        raise LaunchError(f"guard 6: training records that fail a check (or were made at another commit than HEAD): "
                          f"seeds {bad}; move each aside with `run_v2_recipe.py abort-run SEED --reason ...`")
    partial = [s for s, st in states.items() if st == "partial"]
    if partial:
        raise LaunchError(f"guard 6: unfinished runs have files on disk: seeds {partial}; move each aside with "
                          "`run_v2_recipe.py abort-run SEED --reason ...`")
    first_parent(git, commit, 1)
    return dict(commit=commit, expected=expected, forbidden=forbidden, git=git, states=states)


# ---- training ----------------------------------------------------------------------------------------------------
def protected_changed(root, ctx):
    now = protected_now(root, ctx["expected"])
    return sorted(k for k, v in ctx["expected"].items() if now[k] != v)


def train_one(seed, ctx, root, hooks):
    """One run: train to 99 epochs, check the log, re-validate every epoch on validation only, check the native pick
    and its weights, hash and delete every epoch file, write the record. Nothing held out is read."""
    root, py, p, order = Path(root), sys.executable, run_paths(seed), order_of(seed)
    existing = [v for v in p.values() if (root / v).exists()]
    if existing:
        raise LaunchError(f"refusing to overwrite {existing}")
    before = protected_now(root, ctx["expected"])
    if before != ctx["expected"]:
        raise LaunchError(f"guard 3: protected hashes differ before seed {seed}; not started")
    cmd = train_cmd(py, seed)
    check_command(cmd, py, seed, ctx["forbidden"])
    for d in ("logs/v2_recipe/runs", "models/v2_recipe"):
        (root / d).mkdir(parents=True, exist_ok=True)
    log = root / p["train_log"]
    with open(log, "xb"):
        pass
    rc = run_step(hooks["runner"], cmd, log, root, [log_header(seed), f"START {now_iso()} | {hooks['snapshot']()}"])
    with open(log, "ab") as f:
        write_line(f, f"TRAIN_FAILED rc={rc} {now_iso()}" if rc != 0 else f"END {now_iso()} | {hooks['snapshot']()}")
    changed = protected_changed(root, ctx)
    if changed:
        raise LaunchError(f"guard 3: protected hashes changed during seed {seed}: {changed}")
    if rc != 0:
        raise LaunchError(f"training failed: {p['train_log']}")
    try:
        ep, saved = parse_train_log(log, seed)
    except DesignError as e:
        raise LaunchError(f"seed {seed}: {e}; nothing deleted") from None
    epoch_dir = root / p["epoch_dir"]
    if epoch_dir.resolve() != root.resolve() / p["epoch_dir"]:
        raise LaunchError(f"seed {seed}: {p['epoch_dir']} resolves to {epoch_dir.resolve()}, outside {root}; "
                          "nothing deleted")
    files = {e: epoch_file(seed, e) for e in range(1, EPOCHS + 1)}
    missing = [rel for rel in files.values() if not (root / rel).is_file()]
    if missing:
        raise LaunchError(f"seed {seed}: epoch files missing: {missing[:3]}; nothing deleted")
    extra = sorted(f.name for f in epoch_dir.iterdir() if f.relative_to(root).as_posix() not in files.values())
    if extra:
        raise LaunchError(f"seed {seed}: {p['epoch_dir']} holds files other than epoch_01-{EPOCHS}.pt: {extra[:3]}; "
                          "nothing deleted")
    reval = hooks["revalidate_fn"](epoch_dir, EPOCHS, TRAIN_LIST)
    logged = {e: va for e, _, va in ep}
    if sorted(reval) != list(range(1, EPOCHS + 1)) or any(abs(round(v, 2) - logged[e]) > 0.0051
                                                          for e, v in reval.items()):
        raise LaunchError(f"seed {seed}: re-validated values do not round to the logged ones; nothing deleted")
    try:
        pick, tie = native_pick(reval, saved, EPOCHS)
    except DesignError as e:
        raise LaunchError(f"seed {seed}: {e}; nothing deleted") from None
    if not hooks["same_weights_fn"](root / p["native_checkpoint"], root / files[pick]):
        raise LaunchError(f"seed {seed}: {p['native_checkpoint']} is not tensor-equal to {files[pick]}; nothing deleted")
    deleted = {str(e): sha256_file(root / rel) for e, rel in files.items()}  # every epoch file, before any delete
    for rel in files.values():
        (root / rel).unlink()
    epoch_dir.rmdir()
    kept = [p["train_log"], p["native_checkpoint"]]
    rec = {"arm": "control", "seed": seed, "order": order, "commit": ctx["commit"], "train_list": TRAIN_LIST,
           "val_list": VAL_LIST, "frame_list_sha256": None, "epochs": EPOCHS, "train_py_sha256": TRAIN_PY_SHA256,
           "native_same_weights": True, "train_log": p["train_log"], "native_checkpoint": p["native_checkpoint"],
           "epoch_dir": p["epoch_dir"], "val_mae_revalidated": {str(e): reval[e] for e in sorted(reval)},
           "native_epoch": pick, "tie_rule": tie, "under_trained": flagged(pick, EPOCHS),
           "files_sha256": {f: sha256_file(root / f) for f in kept}, "deleted_epoch_sha256": deleted,
           "protected_before": before, "protected_after": protected_now(root, ctx["expected"]),
           "aborted_attempts": aborted_attempts(root, seed)}
    if rec["protected_after"] != ctx["expected"]:
        raise LaunchError(f"guard 3: protected hashes changed during seed {seed} (re-validation or deletion)")
    write_json_new(root / p["record"], rec)
    for f in kept:  # an accidental write now fails instead of breaking the record's hash
        os.chmod(root / f, stat.S_IREAD)
    print(f"order {order}: seed {seed} trained; native epoch {pick}"
          + (" (under-training flag)" if rec["under_trained"] else "") + (f"; tie rule {tie}" if tie else ""),
          flush=True)
    return rec


def launch(root=ROOT, *, git=None, expected=None, runner=subprocess.run, revalidate_fn=revalidate,
           snapshot=nvidia_snapshot, cuda_ok=cuda_available, child_import=child_package_path,
           same_weights_fn=same_weights):
    """Guards 1-9, then seeds 30-39 in order (a run with a verified record is skipped). The keyword hooks exist for
    the synthetic tests only; main() exposes none."""
    root = Path(root)
    git, expected = git or real_git(root), dict(expected or PROTECTED)
    hooks = dict(runner=runner, revalidate_fn=revalidate_fn, snapshot=snapshot, same_weights_fn=same_weights_fn)
    with Lock(root):
        ctx = preflight(root, git, expected, cuda_ok, child_import)
        print(f"HEAD {ctx['commit']}; train.py {TRAIN_PY_SHA256[:12]}... (pinned); seeds {SEEDS[0]}-{SEEDS[-1]}",
              flush=True)
        for s in SEEDS:
            if run_state(root, s, ctx["commit"], expected) == "trained":
                print(f"order {order_of(s)}: seed {s} already trained; skipped", flush=True)
                continue
            train_one(s, ctx, root, hooks)
    records = [run_paths(s)["record"] for s in SEEDS]
    print("LAUNCH DONE. Run `run_v2_recipe.py verify`, then `run_v2_recipe.py freeze`:", *records, sep="\n  ",
          flush=True)
    return records


# ---- verify and freeze -------------------------------------------------------------------------------------------
def verified_records(root, git, expected):
    """All 10 records from one launch commit L on the first-parent chain, each passing check_training at L. HEAD may
    differ from L only in the decisions docs and the v2 outputs themselves (the freeze branch commits them)."""
    missing = [run_paths(s)["record"] for s in SEEDS if not (Path(root) / run_paths(s)["record"]).is_file()]
    if missing:
        raise LaunchError(f"not all 10 runs have finished: missing {missing}")
    try:
        recs = {s: load_record(root, s) for s in SEEDS}
    except DesignError as e:
        raise LaunchError(str(e)) from None
    commits = sorted({str(r.get("commit")) for r in recs.values()})
    if len(commits) != 1:
        raise LaunchError(f"the 10 records come from more than one launch commit: {commits}")
    launch_commit = commits[0]
    first_parent(git, launch_commit, 1)
    head = git(["rev-parse", "HEAD"]).stdout.decode().strip()
    if head != launch_commit:
        changed = [p for p in git(["diff", "--no-renames", "-z", "--name-only", launch_commit, "HEAD"]).stdout
                   .decode("utf-8").split("\0") if p]
        other = sorted(p for p in changed if not (doc_path(p) or p.startswith("logs/v2_recipe/") or p == MEMBERS))
        if other:
            raise LaunchError(f"HEAD differs from the launch commit {launch_commit} in more than the decisions docs "
                              f"and the v2 outputs: {other[:5]}")
    rows = []
    for s in SEEDS:
        try:
            rows.append(check_training(recs[s], s, root, launch_commit, expected))
        except DesignError as e:
            raise LaunchError(f"verify: {e}") from None
    return launch_commit, recs, rows


def verify(root=ROOT, *, git=None, expected=None):
    """Every training-only check on the 10 records. Writes nothing."""
    root = Path(root)
    git, expected = git or real_git(root), dict(expected or PROTECTED)
    code_and_data_guards(root, git, expected)
    launch_commit, _, rows = verified_records(root, git, expected)
    print(f"VERIFIED: 10 records from launch commit {launch_commit}")
    for r in rows:
        print(f"  seed {r['seed']} (order {r['order']}): native epoch {r['native_epoch']}, validation MAE at the pick "
              f"{r['val_at_pick']:.4f} kt (validation, the selection set, not an estimate)"
              + ("; under-training flag" if r["under_trained"] else "") + (f"; tie {r['tie_rule']}" if r["tie_rule"] else ""))
    n = sum(r["under_trained"] for r in rows)
    print(f"  under-training flag (pick > {UNDERTRAIN_FRACTION} x {EPOCHS}): {n}/10" +
          (" (>= 5: still budget-limited on validation at 6534 steps; descriptive only)" if n >= 5 else ""))
    return rows


def members_text(root):
    """design §4: one line per member, `<sha256> *models/v2_recipe/seed<s>.pt`, seeds ascending, LF, final newline."""
    return "".join(f"{sha256_file(Path(root) / c)} *{c}\n" for c in member_paths())


def freeze(root=ROOT, *, git=None, expected=None):
    """Write the members file once all 10 records pass every training-only check."""
    root = Path(root)
    git, expected = git or real_git(root), dict(expected or PROTECTED)
    with Lock(root):
        code_and_data_guards(root, git, expected)
        pending = pending_aborts(root)
        if pending:
            raise LaunchError(f"unfinished aborts {pending}; run the same abort command again to finish them")
        frozen = frozen_evidence(root, git)
        if frozen:
            raise LaunchError(f"guard 8: already frozen ({frozen}); refusing to write the members file again")
        launch_commit, recs, rows = verified_records(root, git, expected)
        for s, c in zip(SEEDS, member_paths()):
            if sha256_file(root / c) != recs[s]["files_sha256"][c]:
                raise LaunchError(f"{c} changed since its record")
        text = members_text(root)
        (root / MEMBERS).parent.mkdir(parents=True, exist_ok=True)
        with open(root / MEMBERS, "x", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.chmod(root / MEMBERS, stat.S_IREAD)
    n = sum(r["under_trained"] for r in rows)
    print(f"FROZEN (pending the freeze PR): {MEMBERS}, SHA-256 {sha256_file(root / MEMBERS)}; launch commit "
          f"{launch_commit}; under-training flag {n}/10. Every later use passes --checkpoints "
          f"{' '.join(member_paths())} and --expect-sha {MEMBERS}; never this file as an output path.")
    return text


# ---- aborts ------------------------------------------------------------------------------------------------------
def move_aside(root, dest_rel, info, rels):
    """abort.json (pending, every file's SHA-256) first, then the moves, then status complete."""
    root = Path(root)
    check_output_paths([dest_rel + "/"])
    (root / dest_rel).mkdir(parents=True, exist_ok=True)
    present = [rel for rel in rels if (root / rel).exists()]
    for rel in present:
        src = root / rel
        for f in ([src] if src.is_file() else sorted(f for f in src.rglob("*") if f.is_file())):
            r = f.relative_to(root).as_posix()
            info["moved"][r] = {"to": f"{dest_rel}/{r}", "sha256": sha256_file(f)}
    write_json_replace(root / dest_rel / "abort.json", info)
    for rel in present:
        (root / dest_rel / rel).parent.mkdir(parents=True, exist_ok=True)
        os.rename(root / rel, root / dest_rel / rel)
    info["status"] = "complete"
    write_json_replace(root / dest_rel / "abort.json", info)
    print(f"aborted ({info['kind']}): {len(info['moved'])} files -> {dest_rel}")
    return dest_rel


def resume_or_new(root, seed, new_dest, new_info):
    pending = [f for f in abort_files(root, seed) if json.loads(f.read_text(encoding="utf-8")).get("status") != "complete"]
    if pending:
        return pending[0].parent.relative_to(root).as_posix(), json.loads(pending[0].read_text(encoding="utf-8")), True
    return new_dest, new_info, False


def abort_run(seed, reason, root=ROOT, *, git=None, expected=None):
    """Move one unfinished run, or one whose record fails a check, aside (design §2: retrained only for a
    verification failure, the failed attempt kept). launch then retrains it with its seed, in its order position."""
    root = Path(root)
    git, expected = git or real_git(root), dict(expected or PROTECTED)
    if seed not in SEEDS or not reason:
        raise LaunchError(f"abort-run: unknown seed {seed}, or no --reason")
    with Lock(root):
        frozen = frozen_evidence(root, git)
        if frozen:
            raise LaunchError(f"abort-run: v2 is frozen ({frozen}); `unfreeze` first (only before the freeze PR merges)")
        commit = git(["rev-parse", "HEAD"]).stdout.decode().strip()
        if run_state(root, seed, commit, expected) == "trained":
            raise LaunchError(f"abort-run: seed {seed} has a record that passes every check; finished runs are not "
                              "aborted")
        k = len(abort_files(root, seed)) + 1
        dest, info, resumed = resume_or_new(root, seed, f"{ABORTED}/seed{seed}_attempt{k}", {
            "kind": "attempt", "seed": seed, "order": order_of(seed), "attempt": k, "reason": reason,
            "time": now_iso(), "status": "pending", "moved": {}})
        rels = [run_paths(seed)[key] for key in TRAIN_KEYS]
        if not resumed and not any((root / r).exists() for r in rels):
            raise LaunchError(f"abort-run: seed {seed} has no files to move")
        return move_aside(root, dest, info, rels)


def later_outputs(root, git):
    """H-EP or clean-test output, on disk or on origin/phase2-experiments (unfreeze refuses if any exists)."""
    found = [p for p in LATER_OUTPUTS if (Path(root) / p).exists()]
    r = git(["ls-tree", "-r", "-z", "--name-only", UPSTREAM, "--", *LATER_OUTPUTS], check=False)
    if r.returncode != 0:
        raise LaunchError(f"unfreeze: can't read {UPSTREAM}'s tree; refusing")
    return found + [f"{UPSTREAM}:{n}" for n in r.stdout.decode("utf-8", "replace").split("\0") if n]


def unfreeze(reason, root=ROOT, *, git=None):
    """Withdraw the members file (design §4, guard 8's procedure): only while the freeze PR is unmerged, decided after
    a fetch; never once a freeze entry exists, or any H-EP or clean-test output."""
    root = Path(root)
    git = git or real_git(root)
    if not reason:
        raise LaunchError("unfreeze: no --reason")
    with Lock(root):
        if git(["fetch", "origin"], check=False).returncode != 0:
            raise LaunchError("unfreeze: git fetch origin failed; can't confirm the freeze PR is unmerged; refusing")
        if not (root / MEMBERS).is_file():
            raise LaunchError(f"unfreeze: {MEMBERS} is not on disk; nothing to withdraw")
        if git(["cat-file", "-e", f"{UPSTREAM}:{MEMBERS}"], check=False).returncode == 0:
            raise LaunchError(f"unfreeze: {MEMBERS} is on {UPSTREAM}: the freeze PR has merged, so v2 is frozen for "
                              "good; report a defect in decisions.md instead")
        for ref in ("HEAD", UPSTREAM):
            r = git(["show", f"{ref}:{DECISIONS}"], check=False)
            if r.returncode != 0:
                raise LaunchError(f"unfreeze: can't read {ref}'s {DECISIONS}; refusing")
            if FROZEN_ENTRY.search(r.stdout.decode("utf-8", "replace")):
                raise LaunchError(f"unfreeze: {ref}'s {DECISIONS} already holds the 'v2 Atlantic ensemble frozen' entry")
        later = later_outputs(root, git)
        if later:
            raise LaunchError(f"unfreeze: H-EP or clean-test output exists: {later[:3]}; refusing")
        pending = [f for f in abort_files(root, None) if json.loads(f.read_text(encoding="utf-8")).get("status") != "complete"]
        if pending:
            dest, info = pending[0].parent.relative_to(root).as_posix(), json.loads(pending[0].read_text(encoding="utf-8"))
        else:
            n = len(abort_files(root, None)) + 1
            dest = f"{ABORTED}/freeze_{n}"
            info = {"kind": "freeze", "attempt": n, "reason": reason, "time": now_iso(),
                    "members_sha256": sha256_file(root / MEMBERS), "status": "pending", "moved": {}}
        os.chmod(root / MEMBERS, stat.S_IREAD | stat.S_IWRITE)
        check_output_paths([f"{dest}/members_sha256.txt"])
        (root / dest).mkdir(parents=True, exist_ok=True)
        info["moved"][MEMBERS] = {"to": f"{dest}/members_sha256.txt", "sha256": sha256_file(root / MEMBERS)}
        write_json_replace(root / dest / "abort.json", info)
        os.rename(root / MEMBERS, root / dest / "members_sha256.txt")
        info["status"] = "complete"
        write_json_replace(root / dest / "abort.json", info)
    print(f"UNFROZEN: {MEMBERS} -> {dest}/members_sha256.txt. Drop the file from the freeze branch in a new commit "
          "(never a force-push), retrain any failed run under abort-run, then run freeze again.")
    return dest


# ---- plan and the CLI --------------------------------------------------------------------------------------------
def plan():
    """Every command, in order. No git, GPU, guard or file write."""
    print(f"v2 Atlantic recipe: {EPOCHS} epochs = epochs_for({CONTROL_TRAIN_FRAMES}, {TARGET_STEPS}) "
          f"({BATCHES} batches, {EPOCHS * BATCHES} steps); seeds {SEEDS[0]}-{SEEDS[-1]}; native selection; train.py "
          f"pinned at {TRAIN_PY_SHA256[:12]}...")
    for s in SEEDS:
        p = run_paths(s)
        print(f"order {order_of(s)}: seed {s}")
        print("  train:", train_cmd("PY", s), "->", p["train_log"])
        print(f"  re-validate epochs 1-{EPOCHS} on {VAL_LIST} only; native pick and tie rule; tensor-equal check; "
              "hash and delete all epoch files")
        print("  record:", p["record"])
    print(f"freeze: {MEMBERS} (after verify passes on all 10 records)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["plan", "launch", "verify", "freeze", "abort-run", "unfreeze"])
    ap.add_argument("seed", nargs="?", type=int, help="abort-run only")
    ap.add_argument("--reason", help="abort-run and unfreeze only: why (recorded in abort.json)")
    args = ap.parse_args(argv)
    if (args.seed is not None) != (args.stage == "abort-run") or \
            bool(args.reason) != (args.stage in ("abort-run", "unfreeze")):
        ap.error("abort-run takes SEED --reason TEXT; unfreeze takes --reason TEXT; the other stages take neither")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    if args.stage == "plan":
        plan()
    elif args.stage == "launch":
        launch()
    elif args.stage == "verify":
        verify()
    elif args.stage == "freeze":
        freeze()
    elif args.stage == "abort-run":
        abort_run(args.seed, args.reason)
    else:
        unfreeze(args.reason)


if __name__ == "__main__":
    sys.exit(main())
