"""F-31 driver: the 40 nested-budget training runs, their H9 slot evaluations and per-run manifests
(reports/v2_survey/f31/design.md, approved by John 2026-10-19 23:50; §8 and §10).

    python scripts/run_f31.py plan          # print every command of the 40 runs; runs nothing
    python scripts/run_f31.py launch        # phase 1: the 40 trainings; resumable; then stops
    python scripts/run_f31.py evaluate      # phase 2: H9, only once the 40 training records are committed and pushed
    python scripts/run_f31.py abort-run ARM SEED --reason TEXT    # move an unfinished training run aside
    python scripts/run_f31.py abort-all --reason TEXT             # move every run aside (fix-commit restart)
    python scripts/run_f31.py abort-eval ARM SEED --reason TEXT   # move a crashed H9 evaluation's outputs aside

This is run_n5.py's launch-safety design (closed by John, 2026-10-19 17:03), carried over: the same two phases with the
records committed between them, the same guards, aborts, lock, look evidence and phase-2 marker. What differs is
forced by the F-31 design: the paths (models/f31, results/f31, logs/f31), seeds 10-29 in analyze_f31.expected_order,
one 4x run per seed with --epoch-dir in both arms, the nested-budget slots, per-epoch H9 outputs, run-level rule (A),
and the pinned train.py. Reuse route (design §10): the N5-free helpers are imported from scripts/run_n5.py, whose
blob is pinned (RUN_N5_BLOB_SHA256) and checked by guard 1; everything bound to N5's paths is rewritten here.
run_n5.py is never edited.

Both phases refuse to start unless:
  1. the tracked tree is clean; scripts/analyze_f31.py, scripts/analyze_n5.py and scripts/run_n5.py equal their
     committed blobs at HEAD; HEAD's run_n5.py is the pinned blob and HEAD's analyze_n5.py is analyze_f31's pin;
  2. HEAD's scripts/train.py hashes to analyze_f31.TRAIN_PY_SHA256 (John, 2026-10-19 23:50: the read-at-1x/2x/4x
     equivalence holds only while train.py's learning rate is independent of the total epoch count), and the
     atlantic frame list exists with its pinned SHA-256;
  3. analyze_f31's PROTECTED equals the protected block in the main checkout's COORDINATOR.md, and every protected
     file has that hash;
  6. the checkout is the main checkout, not a worktree; a child python imports hurricane_intensity from ROOT/src;
     torch.cuda.is_available() is True;
and no abort is pending (4). launch also requires:
  4. every output path is under {models,results,logs}/f31/ and outside the never-written paths; no unfinished run,
     unverifiable record or pending abort is on disk; every existing record or manifest has HEAD's commit; no H9
     look has happened while any run is still to be trained; and the phase-2 marker does not exist;
  5. after `git fetch origin`, HEAD is on the first-parent chain of origin/phase2-experiments (a merge commit).
evaluate also requires (5e) the 40 records to come from one launch commit L on that first-parent chain, HEAD to be
pushed, `git diff --no-renames L HEAD` to touch only the 40 records, the decisions docs and the deviations file, and
every record to be tracked and identical to HEAD's blob. Then, before the first H9 evaluation, it applies every
training-side check of the analysis (analyze_f31.check_training to all 40 records, verify_self and check_train_py at
L).

Phase 1 trains the 40 runs in order and evaluates nothing on H9. Each run trains to 4x (control 99 epochs, atlantic
52) with --epoch-dir; every epoch is re-validated on validation only; the slots are analyze_f31.selections (native at
1x/2x/4x with the tie rule against train.py's 'saved' lines, matched and smoothed at 4x). Before anything is deleted,
the epoch dir must resolve inside ROOT and seed<s>.pt must hold the native 4x epoch's tensors. Every epoch file is
hashed; the non-slot ones are deleted and their hashes recorded under deleted_epoch_sha256 (never files_sha256, which
lists exactly the train log, seed<s>.pt and the slot epochs' files). The kept files become read-only once the record,
logs/f31/runs/<arm>_seed<s>.train.json, is written. launch then stops and prints the 40 record paths: commit and push
them, then run evaluate.

Phase 2 evaluates each distinct slot epoch of each run once, from its epoch file (never seed<s>.pt), writes the
manifest = the record plus evals, eval_attempts and protected_after_eval (nothing else: analyze_f31.load_run
requires manifest == record on every other key), and checks it with analyze_f31.load_run. The protected hashes are
taken before and after every step; any mismatch stops the driver. One lock file (logs/f31/run_f31.lock) serialises
every stage that writes.

Aborts: each moves files, unchanged, to logs/f31/aborted/<arm>_seed<s>_<attempt|eval><k>/, after writing abort.json
there with every file's SHA-256 and status "pending"; the status becomes "complete" once all are moved, and a pending
abort is resumed by running the same command again (every other stage refuses while one is pending). abort-run and
abort-all refuse once H9 has been looked at: any H9 output or manifest, any aborted H9 evaluation, or
logs/f31/phase2_started.json, which evaluate writes before its first H9 step and nothing moves; launch refuses while
that marker exists. abort-run refuses a run whose record verifies. abort-eval only takes a run whose record verifies
and whose H9 outputs exist without a manifest, or with a manifest load_run refuses; the same slot epochs are then
evaluated again.

Rule (A) (evaluate --deviations logs/f31/deviations.json): only after a look; the committed file, identical to its
blob at HEAD; read and checked by analyze_f31.load_deviations, so the driver and the analysis share one format and one
trigger (a file the record hashed is missing or changed). The named runs are left out of every check and H9 step.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL_LIST = "storms_val_expanded.txt"
HOLDOUT_LIST = "storms_holdout_h9.txt"
FRAME_LIST = "results/n5_atlantic/frame_list_atlantic.txt"  # N5's committed list, read only
ANALYSIS = "scripts/analyze_f31.py"
ANALYZE_N5 = "scripts/analyze_n5.py"
RUN_N5 = "scripts/run_n5.py"
RUN_N5_BLOB_SHA256 = "e8f13e06b80c29101ef1a4efaeac7aaa17e5922f0a8bebdc26a19f270d6d3ed7"  # blob 2ebd8cea, at 482a501
TRAIN_PY = "scripts/train.py"
UPSTREAM = "origin/phase2-experiments"
ABORTED = "logs/f31/aborted"
LOCK = "logs/f31/run_f31.lock"
PHASE2_MARKER = "logs/f31/phase2_started.json"  # written once, before the first H9 step; never moved
RESTORE_AFTER_LOOK = ("H9 has been looked at, so it can't be retrained: restore the exact bytes of the changed files "
                      "(their SHA-256s are in the record); if that is impossible, rule (A) applies: the run is "
                      "dropped as a named deviation (evaluate --deviations logs/f31/deviations.json)")
OUTPUT_ROOTS = ("models/f31/", "results/f31/", "logs/f31/")
NEVER_WRITTEN = ("best_model.pt", "models/baseline_9kt.pt", "results/heldout_predictions.csv", "models/storms/",
                 "results/storms/", "results/ensemble/", "results/final_2025/", "site/", "models/n5_atlantic/",
                 "results/n5_atlantic/", "logs/n5_atlantic/", "data/", "results/hep/")
TRAIN_KEYS = ("train_log", "native_checkpoint", "epoch_dir", "record")  # phase 1 writes these


class LaunchError(SystemExit):
    pass


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # every module loaded here has a __main__ guard; importing runs nothing
    return module


# The N5-free helpers, from the pinned run_n5.py (guard 1 checks HEAD's blob): hashing, git, CUDA, the child import
# path, same_weights, the GPU snapshot, timestamps, the child env, binary log writes, JSON writes, one subprocess step,
# validation-only re-validation (run_n5.revalidate: train.py's validation loop, no held-out storm), first_parent and
# doc_path (the same upstream and decisions docs).
RN5 = load_module("run_n5_for_f31", Path(__file__).resolve().parent / "run_n5.py")
sha256_file, protected_now, real_git = RN5.sha256_file, RN5.protected_now, RN5.real_git
cuda_available, child_package_path, same_weights = RN5.cuda_available, RN5.child_package_path, RN5.same_weights
nvidia_snapshot, now_iso, child_env, write_line = RN5.nvidia_snapshot, RN5.now_iso, RN5.child_env, RN5.write_line
write_json_new, write_json_replace, run_step, revalidate = (RN5.write_json_new, RN5.write_json_replace, RN5.run_step,
                                                            RN5.revalidate)
doc_path = RN5.doc_path


def analysis(root):
    return load_module("analyze_f31", Path(root) / ANALYSIS)


def schedule(A):
    """(order, arm, seed) in analyze_f31.expected_order: round-robin by seed, arm order alternating C/A, A/C."""
    return sorted((A.expected_order(arm, s), arm, s) for s in A.SEEDS for arm in A.ARMS)


def run_paths(A, arm, seed):
    p = dict(A.pinned_paths(arm, seed))
    p.update(manifest=f"logs/f31/runs/{arm}_seed{seed}.json", record=f"logs/f31/runs/{arm}_seed{seed}.train.json")
    return p


def h9_files(root, arm, seed):
    """Every H9 output of one run on disk: its per-epoch CSVs and eval logs, and its manifest."""
    root = Path(root)
    found = [f for pat in (f"results/f31/{arm}_seed{seed}_e*_h9.csv", f"logs/f31/{arm}_seed{seed}_e*_eval_h9.log")
             for f in root.glob(pat)]
    man = root / f"logs/f31/runs/{arm}_seed{seed}.json"
    return sorted(f.relative_to(root).as_posix() for f in found + ([man] if man.exists() else []))


def train_cmd(py, A, arm, seed):
    p = run_paths(A, arm, seed)
    cmd = [py, "-u", "scripts/train.py", "--train-list", A.N5.TRAIN_LISTS[arm], "--val-list", VAL_LIST,
           "--epochs", str(A.EPOCHS[arm][4]), "--seed", str(seed), "--loss", "l1"]
    if arm == "atlantic":
        cmd += ["--frame-list", FRAME_LIST]
    return cmd + ["--out", p["native_checkpoint"], "--epoch-dir", p["epoch_dir"]]


def eval_cmd(py, checkpoint, train_list, out):
    return [py, "scripts/evaluate.py", "--checkpoint", checkpoint, "--train-list", train_list,
            "--holdout-list", HOLDOUT_LIST, "--out", out]


def check_output_paths(paths):
    """Every output is under the F-31 roots and never under a protected, frozen, N5 or data path (design §9)."""
    bad = [p for p in paths if not p.startswith(OUTPUT_ROOTS) or any(p.startswith(n) for n in NEVER_WRITTEN)]
    if bad:
        raise LaunchError(f"refusing output paths outside {OUTPUT_ROOTS} or under a never-written path: {bad}")


def planned_outputs(A):
    out = [v for _, arm, s in schedule(A) for v in run_paths(A, arm, s).values()]
    out += [v for _, arm, s in schedule(A) for e in range(1, A.EPOCHS[arm][4] + 1)
            for k, v in A.epoch_paths(arm, s, e).items()]
    return out + [ABORTED + "/", LOCK, PHASE2_MARKER]


def h9_outputs(root, A):
    return [f for _, arm, s in schedule(A) for f in h9_files(root, arm, s)]


def looked(root, A):
    """Evidence of an H9 look: any H9 output or manifest, any aborted H9 evaluation, or the phase-2 marker."""
    found = h9_outputs(root, A)
    found += sorted(f.relative_to(root).as_posix() for f in Path(root).glob(f"{ABORTED}/*_eval*/abort.json"))
    return found + ([PHASE2_MARKER] if (Path(root) / PHASE2_MARKER).exists() else [])


def abort_files(root, arm, seed, kind):
    """abort.json paths of one run's aborts of one kind ("attempt" or "eval"), in attempt order."""
    found = Path(root).glob(f"{ABORTED}/{arm}_seed{seed}_{kind}*/abort.json")
    return sorted(found, key=lambda f: int(f.parent.name.rsplit(kind, 1)[1]))


def aborted_attempts(root, arm, seed, kind="attempt"):
    return [f.relative_to(root).as_posix() for f in abort_files(root, arm, seed, kind)]


def pending_aborts(root):
    return sorted(f.relative_to(root).as_posix() for f in Path(root).glob(f"{ABORTED}/*/abort.json")
                  if json.loads(f.read_text(encoding="utf-8")).get("status") != "complete")


def verify_record(root, A, arm, seed, expected):
    """A training record is valid only while every file it hashed is unchanged and its protected hashes held."""
    rel = run_paths(A, arm, seed)["record"]
    try:
        rec = json.loads((Path(root) / rel).read_text(encoding="utf-8"))
    except ValueError as e:
        raise LaunchError(f"training record {rel} is not JSON ({e})") from None
    if (rec.get("arm"), rec.get("seed")) != (arm, seed):
        raise LaunchError(f"training record {rel} is for {rec.get('arm')} seed {rec.get('seed')}")
    changed = [f for f, h in rec.get("files_sha256", {}).items()
               if not (Path(root) / f).is_file() or sha256_file(Path(root) / f) != h]
    if changed or not rec.get("files_sha256"):
        raise LaunchError(f"training record {rel}: files missing or changed since training: {changed}")
    if rec.get("protected_after") != expected:
        raise LaunchError(f"training record {rel}: protected hashes differ")
    return rec


def run_state(root, A, arm, seed, expected):
    """fresh (nothing on disk), partial (files but no record), bad_record (a record that does not verify), trained,
    eval_partial (H9 outputs but no manifest) or evaluated."""
    p = run_paths(A, arm, seed)
    if not (Path(root) / p["record"]).exists():
        on_disk = any((Path(root) / v).exists() for k, v in p.items() if k != "manifest") or h9_files(root, arm, seed)
        return "partial" if on_disk else "fresh"
    try:
        verify_record(root, A, arm, seed, expected)
    except LaunchError:
        return "bad_record"
    if (Path(root) / p["manifest"]).exists():
        return "evaluated"
    return "eval_partial" if h9_files(root, arm, seed) else "trained"


class Lock:
    """One run_f31 stage that writes at a time. A stale lock (a killed process) is removed by hand, after checking
    that no run_f31 is running."""

    def __init__(self, root):
        self.path = Path(root) / LOCK

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise LaunchError(f"{LOCK} exists: another run_f31 stage is running (or one was killed; remove the file by "
                              "hand only after checking that none is running)") from None
        os.write(fd, f"pid {os.getpid()} {now_iso()}\n".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink()


def common_guards(root, git, main_root, check, cuda_ok, child_import):
    """Guards 1, 2, 3 and 6, and pending aborts, shared by both phases. Returns what both need."""
    if git(["status", "--porcelain", "--untracked-files=no"]).stdout.strip():
        raise LaunchError("guard 1: the tracked tree has uncommitted changes; launch from a clean commit")
    for rel in (ANALYSIS, ANALYZE_N5, RUN_N5, TRAIN_PY):
        if git(["diff", "--quiet", "HEAD", "--", rel], check=False).returncode != 0:
            raise LaunchError(f"guard 1: {rel} differs from its committed blob at HEAD")
    if hashlib.sha256(git(["show", f"HEAD:{RUN_N5}"]).stdout).hexdigest() != RUN_N5_BLOB_SHA256:
        raise LaunchError(f"guard 1: HEAD's {RUN_N5} is not the pinned blob {RUN_N5_BLOB_SHA256[:12]}... (the helpers "
                          "this driver imports)")
    A = analysis(root)
    if hashlib.sha256(git(["show", f"HEAD:{ANALYZE_N5}"]).stdout).hexdigest() != A.N5_BLOB_SHA256:
        raise LaunchError(f"guard 1: HEAD's {ANALYZE_N5} is not analyze_f31's pinned N5 blob")
    design = A.design_constants()
    if hashlib.sha256(git(["show", f"HEAD:{TRAIN_PY}"]).stdout).hexdigest() != A.TRAIN_PY_SHA256:
        raise LaunchError(f"guard 2: HEAD's {TRAIN_PY} is not the pinned blob {A.TRAIN_PY_SHA256[:12]}...; the "
                          "nested 1x/2x/4x reading holds only for it (John, 2026-10-19 23:50)")
    frame_list = Path(root) / FRAME_LIST
    if not frame_list.is_file() or sha256_file(frame_list) != design["frame_list_sha"]["atlantic"]:
        raise LaunchError(f"guard 2: {FRAME_LIST} is missing or its SHA-256 is not the pinned one")
    expected = dict(check.load_expected(Path(main_root) / "COORDINATOR.md"))
    if expected != design["protected"]:
        raise LaunchError("guard 3: analyze_f31's PROTECTED differs from the COORDINATOR.md protected block")
    actual = protected_now(main_root, expected)
    if actual != expected:
        raise LaunchError(f"guard 3: protected files differ: {sorted(k for k in expected if actual[k] != expected[k])}")
    if Path(root).resolve() != Path(main_root).resolve() or not (Path(root) / ".git").is_dir():
        raise LaunchError(f"guard 6: {root} is not the main checkout {main_root} (a worktree has a .git file)")
    pkg = child_import(root)
    if not pkg or not Path(pkg).resolve().is_relative_to((Path(root) / "src").resolve()):
        raise LaunchError(f"guard 6: a child python imports hurricane_intensity from {pkg or '(nowhere)'}, not "
                          f"{Path(root) / 'src'}")
    if not cuda_ok():
        raise LaunchError("guard 6: torch.cuda.is_available() is False; the runs need the GPU")
    pending = pending_aborts(root)
    if pending:
        raise LaunchError(f"guard 4: unfinished aborts {pending}; run the same abort command again to finish them")
    commit = git(["rev-parse", "HEAD"]).stdout.decode().strip()
    return dict(A=A, design=design, expected=expected, commit=commit, git=git)


def first_parent(git, commit):
    """Guard 5: after a fetch, commit is on origin/phase2-experiments' first-parent chain (merge commits only)."""
    if git(["fetch", "origin"], check=False).returncode != 0:
        raise LaunchError("guard 5: git fetch origin failed; can't check the launch commit against the remote")
    chain = git(["rev-list", "--first-parent", UPSTREAM], check=False)
    if chain.returncode != 0 or commit not in chain.stdout.decode().split():
        raise LaunchError(f"guard 5: {commit} is not on the first-parent chain of {UPSTREAM}; launch only from a "
                          "merge commit of a reviewed PR")


def preflight(root, git, main_root, check, cuda_ok, child_import):
    """launch's guards (module docstring)."""
    ctx = common_guards(root, git, main_root, check, cuda_ok, child_import)
    A, expected = ctx["A"], ctx["expected"]
    check_output_paths(planned_outputs(A))
    if (Path(root) / PHASE2_MARKER).exists():
        raise LaunchError(f"guard 4: {PHASE2_MARKER} exists: phase 2 has started, so no training may run")
    states = {(arm, s): run_state(root, A, arm, s, expected) for _, arm, s in schedule(A)}
    bad = [(arm, s) for (arm, s), st in states.items() if st == "bad_record"]
    if bad:
        raise LaunchError(f"guard 4: training records that do not verify: {bad}; " + (
            RESTORE_AFTER_LOOK if looked(root, A) else
            "move each run aside with `run_f31.py abort-run ARM SEED --reason ...` and launch again"))
    partial = [(arm, s) for (arm, s), st in states.items() if st == "partial"]
    if partial:
        raise LaunchError(f"guard 4: unfinished runs have files on disk: {partial[:5]}"
                          f"{' ...' if len(partial) > 5 else ''}; move each run aside with "
                          "`run_f31.py abort-run ARM SEED --reason ...`")
    stale = []
    for _, arm, s in schedule(A):
        for k in ("record", "manifest"):
            f = Path(root) / run_paths(A, arm, s)[k]
            if f.exists():
                commit = json.loads(f.read_text(encoding="utf-8")).get("commit")
                if commit != ctx["commit"]:  # the commit fixes every blob; verify_self checks them at it in phase 2
                    stale.append((run_paths(A, arm, s)[k], commit))
    if stale:
        raise LaunchError(f"guard 4: runs recorded at another commit than HEAD {ctx['commit']}: {stale[:3]}; relaunch "
                          "only at their commit, or (before any H9 look) move every run aside with "
                          "`run_f31.py abort-all --reason ...` and restart all 40")
    seen = looked(root, A)
    if seen and any(st == "fresh" for st in states.values()):
        raise LaunchError(f"guard 4: H9 has been looked at while runs remain to be trained: {seen[:3]}; no training "
                          "may follow an H9 evaluation")
    first_parent(git, ctx["commit"])
    return ctx


def evaluate_step(runner, py, root, checkpoint, train_list, out, log):
    with open(Path(root) / log, "xb"):
        pass
    if run_step(runner, eval_cmd(py, checkpoint, train_list, out), Path(root) / log, root) != 0:
        with open(Path(root) / log, "ab") as f:
            write_line(f, f"EVAL_FAILED {now_iso()}")
        raise LaunchError(f"evaluation failed: {log}")


def train_one(order, arm, seed, ctx, root, main_root, hooks):
    """Phase 1 for one run: train to 4x, re-validate every epoch, fix the slots, delete the non-slot epoch files,
    write the training record. No H9."""
    A, design, py, root = ctx["A"], ctx["design"], sys.executable, Path(root)
    p = run_paths(A, arm, seed)
    e_max = A.EPOCHS[arm][4]
    existing = [v for k, v in p.items() if (root / v).exists()] + h9_files(root, arm, seed)
    if existing:
        raise LaunchError(f"refusing to overwrite {existing}")
    seen = looked(root, A)
    if seen:
        raise LaunchError(f"phase 1: H9 has been looked at ({seen[:3]}); no training may follow an H9 evaluation")
    before = protected_now(main_root, ctx["expected"])
    if before != ctx["expected"]:
        raise LaunchError(f"protected hashes differ before {arm} seed {seed}")
    for d in ("logs/f31/runs", f"models/f31/{arm}", "results/f31"):
        (root / d).mkdir(parents=True, exist_ok=True)
    log = root / p["train_log"]
    with open(log, "xb"):
        pass
    header = [f"RUN arm={arm} seed={seed} order={order} list={design['train_lists'][arm]} val={VAL_LIST} "
              f"epochs={e_max}", f"START {now_iso()} | {hooks['snapshot']()}"]
    rc = run_step(hooks["runner"], train_cmd(py, A, arm, seed), log, root, header)
    with open(log, "ab") as f:
        write_line(f, f"TRAIN_FAILED rc={rc} {now_iso()}" if rc != 0 else f"END {now_iso()} | {hooks['snapshot']()}")
    if rc != 0:
        raise LaunchError(f"training failed: {p['train_log']}")
    try:
        ep, saved = A.parse_train_log(log, arm, seed, design)
    except A.DesignError as e:
        raise LaunchError(f"{arm} seed {seed}: {e}") from None

    epoch_dir = root / p["epoch_dir"]
    if epoch_dir.resolve() != root.resolve() / p["epoch_dir"]:
        raise LaunchError(f"{arm} seed {seed}: {p['epoch_dir']} resolves to {epoch_dir.resolve()}, outside {root}; "
                          "nothing deleted")
    files = {e: A.epoch_paths(arm, seed, e)["checkpoint"] for e in range(1, e_max + 1)}
    missing = [rel for rel in files.values() if not (root / rel).is_file()]
    if missing:
        raise LaunchError(f"{arm} seed {seed}: epoch files missing: {missing[:3]}; nothing deleted")
    reval = hooks["revalidate_fn"](epoch_dir, e_max, design["train_lists"][arm])
    logged = {e: va for e, _, va in ep}
    if sorted(reval) != list(range(1, e_max + 1)) or any(abs(round(v, 2) - logged[e]) > 0.0051
                                                         for e, v in reval.items()):
        raise LaunchError(f"{arm} seed {seed}: re-validated values do not round to the logged ones; nothing deleted")
    try:
        slots, ties, near = A.selections(arm, reval, saved)
    except A.DesignError as e:
        raise LaunchError(f"{arm} seed {seed}: {e}; nothing deleted") from None
    native = files[slots["native_4x"]]
    if not hooks["same_weights_fn"](root / p["native_checkpoint"], root / native):
        raise LaunchError(f"{arm} seed {seed}: {p['native_checkpoint']} does not hold {native}'s weights; train.py's "
                          "'saved' lines were not parsed right; nothing deleted")
    kept_epochs = sorted(set(slots.values()))
    all_sha = {e: sha256_file(root / rel) for e, rel in files.items()}  # every epoch file, hashed before any delete
    deleted = {str(e): h for e, h in all_sha.items() if e not in kept_epochs}
    for e in map(int, deleted):
        (root / files[e]).unlink()
    kept = [p["train_log"], p["native_checkpoint"]] + [files[e] for e in kept_epochs]
    rec = {"arm": arm, "seed": seed, "order": order, "commit": ctx["commit"],
           "train_list": design["train_lists"][arm], "frame_list_sha256": design["frame_list_sha"][arm],
           "train_py_sha256": A.TRAIN_PY_SHA256, "native_same_weights": True,
           "train_log": p["train_log"], "native_checkpoint": p["native_checkpoint"], "epoch_dir": p["epoch_dir"],
           "val_mae_revalidated": {str(e): reval[e] for e in sorted(reval)}, "slots": slots, "tie_rule": ties,
           "near_ties": near, "files_sha256": {f: sha256_file(root / f) for f in kept},
           "deleted_epoch_sha256": deleted, "protected_before": before,
           "protected_after": protected_now(main_root, ctx["expected"]),
           "aborted_attempts": aborted_attempts(root, arm, seed), "eval_attempts": []}
    if rec["protected_after"] != ctx["expected"]:
        changed = sorted(k for k, v in ctx["expected"].items() if rec["protected_after"][k] != v)
        raise LaunchError(f"protected hashes changed during {arm} seed {seed}: {changed}")
    write_json_new(root / p["record"], rec)
    for f in kept:  # an accidental write now fails instead of breaking the record's hash
        os.chmod(root / f, stat.S_IREAD)
    print(f"order {order}: {arm} seed {seed} trained; slots {slots}"
          + (f"; tie rule {({b: t for b, t in ties.items() if t})}" if any(ties.values()) else ""), flush=True)
    return rec


def train_phase(ctx, root, main_root, hooks):
    A = ctx["A"]
    for order, arm, s in schedule(A):
        if run_state(root, A, arm, s, ctx["expected"]) in ("trained", "eval_partial", "evaluated"):
            print(f"order {order}: {arm} seed {s} already trained; skipped", flush=True)
            continue
        train_one(order, arm, s, ctx, root, main_root, hooks)


def records_evidence(root, git, A):
    """Guard 5e: the 40 records are the pushed witness that all trainings finished before any H9 look."""
    recs = []
    for _, arm, s in schedule(A):
        rel = run_paths(A, arm, s)["record"]
        if not (Path(root) / rel).is_file():
            raise LaunchError(f"phase 2: H9 evaluation refuses to start until all 40 training runs have finished; "
                              f"{rel} is missing")
        if git(["ls-files", "--error-unmatch", rel], check=False).returncode != 0:
            raise LaunchError(f"guard 5e: {rel} is not tracked; commit and push the 40 training records first")
        if git(["show", f"HEAD:{rel}"], check=False).stdout != (Path(root) / rel).read_bytes().replace(b"\r\n", b"\n"):
            raise LaunchError(f"guard 5e: {rel} differs from its committed blob at HEAD")
        recs.append(json.loads((Path(root) / rel).read_text(encoding="utf-8")))
    try:
        launch_commit = A.single_commit(recs)
    except A.DesignError as e:
        raise LaunchError(f"phase 2: {e}") from None
    first_parent(git, launch_commit)
    if git(["merge-base", "--is-ancestor", "HEAD", UPSTREAM], check=False).returncode != 0:
        raise LaunchError(f"guard 5e: HEAD is not pushed to {UPSTREAM}; push the commit with the 40 records first")
    allowed = {run_paths(A, arm, s)["record"] for _, arm, s in schedule(A)} | {A.DEVIATIONS_PATH}
    changed = [p for p in git(["diff", "--no-renames", "-z", "--name-only", launch_commit, "HEAD"]).stdout
               .decode("utf-8").split("\0") if p]
    other = sorted(p for p in changed if p not in allowed and not doc_path(p))
    if other:
        raise LaunchError(f"guard 5e: HEAD differs from the launch commit {launch_commit} in more than the 40 "
                          f"records, the decisions docs and the deviations file: {other[:5]}")
    return launch_commit, recs


def deviations_in(root, git, A, design, path):
    """Rule (A) in phase 2: after a look only, the committed deviations file, read by analyze_f31.load_deviations
    (the analysis's own format, path and trigger). Returns the named (arm, seed) runs."""
    if path is None:
        return set()
    if not looked(root, A):  # before a look a damaged run is retrained (abort-run), keeping n = 20 and confirmatory
        raise LaunchError("phase 2: no H9 look yet: abort-run and retrain instead; rule (A) applies only after a look")
    if Path(path).as_posix() != A.DEVIATIONS_PATH:
        raise LaunchError(f"--deviations must be {A.DEVIATIONS_PATH}, not {path}")
    if git(["show", f"HEAD:{A.DEVIATIONS_PATH}"], check=False).stdout != \
            (Path(root) / A.DEVIATIONS_PATH).read_bytes().replace(b"\r\n", b"\n"):
        raise LaunchError(f"{A.DEVIATIONS_PATH} must be committed, and identical to its blob at HEAD")
    try:
        dropped, _ = A.load_deviations(Path(root) / A.DEVIATIONS_PATH, root, Path(root) / "logs/f31/runs", git, design)
    except A.DesignError as e:
        raise LaunchError(f"phase 2: {e}") from None
    return set(dropped)


def evaluate_phase(ctx, root, main_root, runner, deviations=None):
    """No H9 evaluation runs until all 40 training runs have finished and every training-side check the analysis will
    apply has passed on all 40 records. Under rule (A) the named runs are left out of every check and step here, as
    the analysis leaves them out."""
    A, git, root = ctx["A"], ctx["git"], Path(root)
    order = schedule(A)
    states = {(arm, s): run_state(root, A, arm, s, ctx["expected"]) for _, arm, s in order}
    skip = deviations_in(root, git, A, ctx["design"], deviations)
    order = [(k, arm, s) for k, arm, s in order if (arm, s) not in skip]
    states = {key: st for key, st in states.items() if key not in skip}
    bad = [key for key, st in states.items() if st == "bad_record"]
    if bad:
        raise LaunchError(f"phase 2: training records that do not verify: {bad}; " + (
            RESTORE_AFTER_LOOK if looked(root, A) else
            "no H9 output exists yet, so move each run aside with `run_f31.py abort-run ARM SEED --reason ...` and "
            "launch again"))
    unfinished = [(k, arm, s) for k, arm, s in order if states[(arm, s)] not in ("trained", "eval_partial", "evaluated")]
    if unfinished:
        raise LaunchError(f"phase 2: H9 evaluation refuses to start until all {len(order)} training runs have "
                          f"finished; unfinished: {unfinished}")
    half = [key for key, st in states.items() if st == "eval_partial"]
    if half:
        raise LaunchError(f"phase 2: H9 outputs without a manifest for {half}; move them aside with "
                          "`run_f31.py abort-eval ARM SEED --reason ...`, then evaluate again")
    launch_commit, recs = records_evidence(root, git, A)
    try:
        A.verify_self(launch_commit, root, git)
        A.check_train_py(launch_commit, git, ctx["design"])
        for (_, arm, s), rec in zip(schedule(A), recs):
            if (arm, s) not in skip:
                A.check_training(rec, arm, s, ctx["design"], root, run_paths(A, arm, s)["record"])
    except A.DesignError as e:
        raise LaunchError(f"phase 2 entry, before any H9 evaluation: {e}") from None
    if not (root / PHASE2_MARKER).exists():
        write_json_new(root / PHASE2_MARKER, {"head": ctx["commit"], "launch_commit": launch_commit, "time": now_iso()})
    mans = []
    for k, arm, s in order:
        p = run_paths(A, arm, s)
        if states[(arm, s)] == "evaluated":
            try:
                A.load_run(root / p["manifest"], arm, s, ctx["design"], root)
            except A.DesignError as e:
                raise LaunchError(f"phase 2: {e}; move its H9 outputs and manifest aside with "
                                  f"`run_f31.py abort-eval {arm} {s} --reason ...` to re-evaluate the same slots") from None
            mans.append(json.loads((root / p["manifest"]).read_text(encoding="utf-8")))
            print(f"order {k}: {arm} seed {s} already evaluated; skipped", flush=True)
        else:
            mans.append(evaluate_one(k, arm, s, ctx, root, main_root, runner))
    return mans


def evaluate_one(order, arm, seed, ctx, root, main_root, runner):
    """Phase 2 for one run: H9 evaluation of each distinct slot epoch, from its epoch file, then the manifest."""
    A, design, py, root = ctx["A"], ctx["design"], sys.executable, Path(root)
    p = run_paths(A, arm, seed)
    rec = verify_record(root, A, arm, seed, ctx["expected"])
    attempts = aborted_attempts(root, arm, seed, "eval")
    try:
        A.check_attempts({"eval_attempts": attempts}, root, p["record"])
    except A.DesignError as e:
        raise LaunchError(f"before the H9 evaluation of {arm} seed {seed}: {e}") from None
    before = protected_now(main_root, ctx["expected"])
    if before != ctx["expected"]:
        raise LaunchError(f"protected hashes differ before the H9 evaluation of {arm} seed {seed}")
    evals = {}
    for e in sorted(set(rec["slots"].values())):
        ep = A.epoch_paths(arm, seed, e)
        evaluate_step(runner, py, root, ep["checkpoint"], rec["train_list"], ep["csv"], ep["eval_log"])
        evals[str(e)] = {"csv_sha256": sha256_file(root / ep["csv"]), "eval_log_sha256": sha256_file(root / ep["eval_log"])}
    man = dict(rec, evals=evals, eval_attempts=attempts, protected_after_eval=protected_now(main_root, ctx["expected"]))
    write_json_new(root / p["manifest"], man)
    if man["protected_after_eval"] != ctx["expected"]:
        raise LaunchError(f"protected hashes changed during the H9 evaluation of {arm} seed {seed}")
    try:
        A.load_run(root / p["manifest"], arm, seed, design, root)  # the analysis's own linkage checks, now
    except A.DesignError as e:
        raise LaunchError(f"after the H9 evaluation of {arm} seed {seed}: {e}") from None
    print(f"order {order}: {arm} seed {seed} evaluated on H9 at epochs {sorted(map(int, evals))}", flush=True)
    return man


def resolve_hooks(root, git, main_root, check, **hooks):
    root = Path(root)
    git = git or real_git(root)
    check = check or load_module("check", root / "scripts" / "check.py")
    main_root = Path(main_root) if main_root else check.find_main_root(None)
    return root, git, main_root, check, hooks


def launch(root=ROOT, *, git=None, main_root=None, check=None, runner=subprocess.run, revalidate_fn=revalidate,
           snapshot=nvidia_snapshot, cuda_ok=cuda_available, child_import=child_package_path,
           same_weights_fn=same_weights):
    """Phase 1 only: all 40 trainings, then stop. The keyword hooks exist for the synthetic tests only; main()
    exposes none."""
    root, git, main_root, check, hooks = resolve_hooks(root, git, main_root, check, runner=runner, snapshot=snapshot,
                                                       revalidate_fn=revalidate_fn, same_weights_fn=same_weights_fn)
    with Lock(root):
        ctx = preflight(root, git, main_root, check, cuda_ok, child_import)
        print(f"HEAD {ctx['commit']}; train.py {ctx['A'].TRAIN_PY_SHA256[:12]}... (pinned)", flush=True)
        train_phase(ctx, root, main_root, hooks)
    records = [run_paths(ctx["A"], arm, s)["record"] for _, arm, s in schedule(ctx["A"])]
    print("PHASE 1 DONE. Commit these 40 training records, push them to phase2-experiments, then run "
          "`run_f31.py evaluate`:", *records, sep="\n  ", flush=True)
    return records


def evaluate(root=ROOT, *, deviations=None, git=None, main_root=None, check=None, runner=subprocess.run,
             cuda_ok=cuda_available, child_import=child_package_path):
    """Phase 2 only. deviations: rule (A)'s committed file (main() passes --deviations). The other keyword hooks
    exist for the synthetic tests only; main() exposes none."""
    root, git, main_root, check, _ = resolve_hooks(root, git, main_root, check)
    with Lock(root):
        ctx = common_guards(root, git, main_root, check, cuda_ok, child_import)
        print(f"HEAD {ctx['commit']}", flush=True)
        return evaluate_phase(ctx, root, main_root, runner, deviations)


def _abort_one(root, A, arm, seed, reason, kind, rels):
    """Move one run's files `rels` to aborted/<arm>_seed<s>_<kind><k>/: abort.json (pending, every file's SHA-256)
    first, then the moves, then status complete. A pending abort of this run and kind is resumed."""
    root = Path(root)
    order = A.expected_order(arm, seed)
    pending = [f for f in abort_files(root, arm, seed, kind)
               if json.loads(f.read_text(encoding="utf-8")).get("status") != "complete"]
    if pending:
        dest_rel = pending[0].parent.relative_to(root).as_posix()
        info = json.loads(pending[0].read_text(encoding="utf-8"))
    else:
        dest_rel = f"{ABORTED}/{arm}_seed{seed}_{kind}{len(abort_files(root, arm, seed, kind)) + 1}"
        info = {"arm": arm, "seed": seed, "order": order, "kind": kind, "attempt": int(dest_rel.rsplit(kind, 1)[1]),
                "reason": reason, "time": now_iso(), "status": "pending", "moved": {}}
    present = [rel for rel in rels if (root / rel).exists()]
    if not present and not pending:
        raise LaunchError(f"abort: {arm} seed {seed} has no files to move")
    check_output_paths([dest_rel + "/"])
    (root / dest_rel).mkdir(parents=True, exist_ok=True)
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
    print(f"aborted {arm} seed {seed} (order {order}, {kind} {info['attempt']}): {len(info['moved'])} files -> {dest_rel}")
    return dest_rel


def _known(A, arm, seed, reason, stage):
    if (arm, seed) not in {(a, s) for _, a, s in schedule(A)} or not reason:
        raise LaunchError(f"{stage}: unknown run {arm} seed {seed}, or no --reason")


def abort_run(arm, seed, reason, root=ROOT):
    """Move one unfinished training run (or one whose record does not verify) aside; launch restarts it in its order
    position with its seed."""
    root = Path(root)
    with Lock(root):
        A = analysis(root)
        _known(A, arm, seed, reason, "abort-run")
        seen = looked(root, A)
        if seen:
            raise LaunchError(f"abort-run: H9 has been looked at ({seen[:3]}); no run may be aborted after an H9 evaluation")
        if run_state(root, A, arm, seed, A.N5.PROTECTED) in ("trained", "eval_partial", "evaluated"):
            raise LaunchError(f"abort-run: {arm} seed {seed} has a training record that verifies; finished runs are "
                              "not aborted (abort-all restarts all 40)")
        return _abort_one(root, A, arm, seed, reason, "attempt", [run_paths(A, arm, seed)[k] for k in TRAIN_KEYS])


def abort_all(reason, root=ROOT):
    """Before any H9 look, move every run aside, finished ones included, so a relaunch from a fix commit restarts all
    40 in the same order."""
    root = Path(root)
    with Lock(root):
        A = analysis(root)
        if not reason:
            raise LaunchError("abort-all: no --reason")
        seen = looked(root, A)
        if seen:
            raise LaunchError(f"abort-all: H9 has been looked at ({seen[:3]}); no run may be aborted after an H9 evaluation")
        done = []
        for _, arm, s in schedule(A):
            if run_state(root, A, arm, s, A.N5.PROTECTED) != "fresh" or any(
                    json.loads(f.read_text(encoding="utf-8")).get("status") != "complete"
                    for f in abort_files(root, arm, s, "attempt")):
                done.append(_abort_one(root, A, arm, s, f"abort-all: {reason}", "attempt",
                                       [run_paths(A, arm, s)[k] for k in TRAIN_KEYS]))
        if not done:
            raise LaunchError("abort-all: no run has files to move")
        return done


def abort_eval(arm, seed, reason, root=ROOT):
    """Move a crashed H9 evaluation's outputs aside (hashed, never deleted); evaluate then re-evaluates the same
    recorded slot epochs, and the manifest lists the attempt."""
    root = Path(root)
    with Lock(root):
        A = analysis(root)
        _known(A, arm, seed, reason, "abort-eval")
        state = run_state(root, A, arm, seed, A.N5.PROTECTED)
        pending = any(json.loads(f.read_text(encoding="utf-8")).get("status") != "complete"
                      for f in abort_files(root, arm, seed, "eval"))
        if state == "evaluated":
            try:
                A.load_run(root / run_paths(A, arm, seed)["manifest"], arm, seed, A.design_constants(), root)
            except A.DesignError:
                state = "evaluated, refused by load_run"
        if state not in ("eval_partial", "evaluated, refused by load_run") and not (pending and state == "trained"):
            raise LaunchError(f"abort-eval: {arm} seed {seed} is {state}; only a run with a verified training record "
                              "and H9 outputs without a manifest (or with a manifest load_run refuses) can be "
                              "re-evaluated")
        return _abort_one(root, A, arm, seed, reason, "eval", h9_files(root, arm, seed))


def plan(root=ROOT):
    """Every command of both phases, in order. No git, GPU, guard or file write."""
    A = analysis(root)
    py = "PY"
    print(f"F-31 budgets: control epochs {A.EPOCHS['control']}, atlantic epochs {A.EPOCHS['atlantic']}; seeds "
          f"{A.SEEDS[0]}-{A.SEEDS[-1]}; train.py pinned at {A.TRAIN_PY_SHA256[:12]}...")
    print("PHASE 1 (launch): the 40 trainings; no H9 evaluation")
    for order, arm, s in schedule(A):
        p = run_paths(A, arm, s)
        print(f"order {order}: {arm} seed {s}")
        print("  train:", train_cmd(py, A, arm, s), "->", p["train_log"])
        print(f"  re-validate epochs 1-{A.EPOCHS[arm][4]} on {VAL_LIST} only; slots by analyze_f31.selections: "
              f"{list(A.SLOTS[arm])}; delete the non-slot epoch files (hashes -> deleted_epoch_sha256)")
        print("  record:", p["record"])
    print("BETWEEN THE PHASES: commit the 40 records and push them to phase2-experiments")
    print("PHASE 2 (evaluate): only once the 40 committed records pass every training-side check")
    for order, arm, s in schedule(A):
        p = run_paths(A, arm, s)
        ep = A.epoch_paths(arm, s, 0)
        print(f"order {order}: {arm} seed {s}: <= {len(A.SLOTS[arm])} H9 evaluations, one per distinct slot epoch NN "
              "(known only after training)")
        print("  eval:", eval_cmd(py, ep["checkpoint"].replace("epoch_00", "epoch_NN"), A.N5.TRAIN_LISTS[arm],
                                  ep["csv"].replace("_e00_", "_eNN_")),
              "->", ep["eval_log"].replace("_e00_", "_eNN_"))
        print("  manifest:", p["manifest"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["plan", "launch", "evaluate", "abort-run", "abort-all", "abort-eval"])
    ap.add_argument("arm", nargs="?", choices=["control", "atlantic"], help="abort-run and abort-eval only")
    ap.add_argument("seed", nargs="?", type=int, help="abort-run and abort-eval only")
    ap.add_argument("--reason", help="abort stages only: why (recorded in abort.json)")
    ap.add_argument("--deviations", help="evaluate only: rule (A)'s committed logs/f31/deviations.json")
    args = ap.parse_args(argv)
    if args.deviations and args.stage != "evaluate":
        ap.error("--deviations is for evaluate only")
    run_args = args.stage in ("abort-run", "abort-eval")
    if (args.arm is not None, args.seed is not None) != (run_args, run_args) or \
            bool(args.reason) != args.stage.startswith("abort-"):
        ap.error("abort-run/abort-eval take ARM SEED --reason TEXT; abort-all takes --reason TEXT; the other stages "
                 "take none of them")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    if args.stage == "plan":
        plan()
    elif args.stage == "abort-run":
        abort_run(args.arm, args.seed, args.reason)
    elif args.stage == "abort-all":
        abort_all(args.reason)
    elif args.stage == "abort-eval":
        abort_eval(args.arm, args.seed, args.reason)
    elif args.stage == "evaluate":
        evaluate(deviations=args.deviations)
    else:
        launch()


if __name__ == "__main__":
    sys.exit(main())
