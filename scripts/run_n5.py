"""N5 driver: the 20 training runs, their H9 evaluations and per-run manifests (reports/v2_survey/n5/design.md §8 and
its 2026-10-19 amendment, §14).

    python scripts/run_n5.py plan          # print every command of the 20 runs; runs nothing
    python scripts/run_n5.py build-spec    # print the build and exclusion steps (§12.3-4); runs nothing
    python scripts/run_n5.py launch        # phase 1: the 20 trainings; resumable; then stops
    python scripts/run_n5.py evaluate      # phase 2: H9, only once the 20 training records are committed and pushed
    python scripts/run_n5.py abort-run ARM SEED --reason TEXT    # move an unfinished training run aside
    python scripts/run_n5.py abort-all --reason TEXT             # move every run aside (fix-commit restart, §14)
    python scripts/run_n5.py abort-eval ARM SEED --reason TEXT   # move a crashed H9 evaluation's outputs aside

Both phases refuse to start unless:
  1. the tracked tree is clean, and scripts/analyze_n5.py equals its committed blob at HEAD;
  2. analyze_n5.design_constants() is final (ARM_TRAIN_FRAMES and FRAME_LIST_SHA256 filled), and the frame list
     exists with that SHA-256;
  3. analyze_n5.PROTECTED equals the protected block in the main checkout's COORDINATOR.md, and every protected
     file has that hash;
  6. the checkout is the main checkout, not a worktree; a child python imports hurricane_intensity from ROOT/src;
     torch.cuda.is_available() is True.
launch also requires:
  4. every output path is under {models,results,logs}/n5_atlantic/ and outside the never-written paths; no
     unfinished run, unverifiable record or pending abort is on disk; every existing record has HEAD's commit and
     analysis blob; and no H9 output exists while any run is still to be trained;
  5. after `git fetch origin`, HEAD is on the first-parent chain of origin/phase2-experiments (a merge commit).
evaluate also requires (5e) the 20 records to come from one launch commit L on that first-parent chain, HEAD to be
pushed (an ancestor of origin/phase2-experiments), `git diff --no-renames L HEAD` to touch only the 20 records and
the decisions docs (logs/decisions.md, reports/decisions/*.md), and every record
to be tracked and identical to HEAD's blob. Then, before the first H9 evaluation, it applies every training-side
check of the analysis (analyze_n5.check_training, single_commit, verify_self) to all 20 records.

Phase 1 trains the 20 runs in the §4 order and evaluates nothing on H9. Each step is a subprocess with list
arguments (no shell); its stdout and stderr go to a log opened in binary (UTF-8, no BOM; the child gets
PYTHONUTF8=1). The control trains with --epoch-dir; all 25 epochs are re-validated on validation only and the
selection follows analyze_n5.control_selection (the tie rule, §14). Before the even epochs are deleted, seed<s>.pt
must hold the same tensors as epoch_<native>.pt. Each finished run gets a training record,
logs/n5_atlantic/runs/<arm>_seed<s>.train.json. launch then stops and prints the 20 record paths: commit and push
them, then run evaluate. Phase 2 evaluates each run's recorded checkpoint (and the control's native one) on H9,
writes the manifest and checks it with analyze_n5.load_run. The protected hashes are taken before and after every
step; any mismatch stops the driver. One lock file (logs/n5_atlantic/run_n5.lock) serialises every stage that
writes.

Aborts (§14): each moves files, unchanged, to logs/n5_atlantic/aborted/<arm>_seed<s>_<attempt|eval><k>/, after
writing abort.json there with every file's SHA-256 and status "pending"; the status becomes "complete" once all are
moved, and a pending abort is resumed by running the same command again (every other stage refuses while one is
pending). abort-run and abort-all refuse once H9 has been looked at: any pinned H9 output or manifest, any aborted H9
evaluation, or logs/n5_atlantic/phase2_started.json, which evaluate writes before its first H9 step and nothing
moves; launch refuses while that marker exists. abort-run refuses a run whose record verifies. abort-eval only takes
a run whose record verifies and whose H9 outputs exist without a manifest, or with a manifest load_run refuses; the
same recorded checkpoint is then evaluated again.
"""
import argparse
import datetime
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
FRAME_LIST = "results/n5_atlantic/frame_list_atlantic.txt"
BUILD_IDS = "storms_n5_new.txt"
SELF_ANALYSIS = "scripts/analyze_n5.py"
UPSTREAM = "origin/phase2-experiments"
ABORTED = "logs/n5_atlantic/aborted"
LOCK = "logs/n5_atlantic/run_n5.lock"
PHASE2_MARKER = "logs/n5_atlantic/phase2_started.json"  # written once, before the first H9 step; never moved
RESTORE_AFTER_LOOK = ("H9 has been looked at, so it can't be retrained: restore the exact bytes of the changed files "
                      "(their SHA-256s are in the record); if that is impossible, §14's rule (A) applies: the seed "
                      "pair is dropped as a named deviation (analyze_n5.py --deviations)")
DOC_PATHS = ("logs/decisions.md",)  # 5e: docs a merge may add between L and HEAD; never executed (coordinator)
OUTPUT_ROOTS = ("models/n5_atlantic/", "results/n5_atlantic/", "logs/n5_atlantic/")
NEVER_WRITTEN = ("best_model.pt", "models/baseline_9kt.pt", "results/heldout_predictions.csv", "models/storms/",
                 "results/storms/", "results/ensemble/", "results/final_2025/", "site/")
TRAIN_KEYS = ("train_log", "native_checkpoint", "epoch_dir", "record")  # phase 1 writes these
H9_KEYS = ("h9_csv", "eval_log", "native_h9_csv", "native_eval_log", "manifest")  # phase 2 writes these


class LaunchError(SystemExit):
    pass


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # both have a __main__ guard; importing runs nothing
    return module


def analysis(root):
    return load_module("analyze_n5", Path(root) / SELF_ANALYSIS)


def schedule(seeds):
    """(order, arm, seed): seeds interleaved round-robin, arm order alternating C/A, A/C, ... (design §4)."""
    out = []
    for s in seeds:
        for arm in (("control", "atlantic") if s % 2 == 0 else ("atlantic", "control")):
            out.append((len(out) + 1, arm, s))
    return out


def run_paths(arm, seed):
    p = {"train_log": f"logs/n5_atlantic/{arm}_seed{seed}.log",
         "native_checkpoint": f"models/n5_atlantic/{arm}/seed{seed}.pt",
         "h9_csv": f"results/n5_atlantic/{arm}_seed{seed}_h9.csv",
         "eval_log": f"logs/n5_atlantic/{arm}_seed{seed}_eval_h9.log",
         "manifest": f"logs/n5_atlantic/runs/{arm}_seed{seed}.json",
         "record": f"logs/n5_atlantic/runs/{arm}_seed{seed}.train.json"}
    if arm == "control":
        p.update(epoch_dir=f"models/n5_atlantic/control/seed{seed}_epochs",
                 native_h9_csv=f"results/n5_atlantic/control_seed{seed}_native_h9.csv",
                 native_eval_log=f"logs/n5_atlantic/control_seed{seed}_native_eval_h9.log")
    return p


def train_cmd(py, arm, seed, train_lists, epochs):
    p = run_paths(arm, seed)
    cmd = [py, "-u", "scripts/train.py", "--train-list", train_lists[arm], "--val-list", VAL_LIST,
           "--epochs", str(epochs[arm]), "--seed", str(seed), "--loss", "l1"]
    if arm == "atlantic":
        cmd += ["--frame-list", FRAME_LIST]
    cmd += ["--out", p["native_checkpoint"]]
    if arm == "control":
        cmd += ["--epoch-dir", p["epoch_dir"]]
    return cmd


def eval_cmd(py, checkpoint, train_list, out):
    return [py, "scripts/evaluate.py", "--checkpoint", checkpoint, "--train-list", train_list,
            "--holdout-list", HOLDOUT_LIST, "--out", out]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_now(main_root, paths):
    return {p: sha256_file(Path(main_root) / p) for p in paths}


def real_git(root):
    def git(args, check=True):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=check)
    return git


def cuda_available():
    import torch
    return torch.cuda.is_available()


def child_package_path(root):
    """Where a child process (the env train.py and evaluate.py get, cwd ROOT) imports hurricane_intensity from."""
    r = subprocess.run([sys.executable, "-c", "import hurricane_intensity; print(hurricane_intensity.__file__)"],
                       capture_output=True, cwd=str(root), env=child_env())
    return r.stdout.decode("utf-8", "replace").strip()


def same_weights(a, b):
    """True when two checkpoints hold the same state_dict tensors."""
    import torch
    x, y = torch.load(a, map_location="cpu"), torch.load(b, map_location="cpu")
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def nvidia_snapshot():
    q = "timestamp,temperature.gpu,clocks.current.sm,power.draw"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True)
        return r.stdout.decode("utf-8", "replace").strip() or "nvidia-smi: no output"
    except OSError as e:
        return f"nvidia-smi unavailable ({e.__class__.__name__})"


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def revalidate(epoch_dir, epochs, train_list):
    """Validation MAE (kt, unrounded) of every epoch_NN.pt, computed exactly as train.py's epoch loop does: the same
    validation storms in the same order, TCDataset unmasked, batch 32, no shuffle, eval mode, the batch-weighted mean
    of L1 on labels / 185, times 185. No held-out storm is touched."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18
    from hurricane_intensity.dataset import TCDataset
    from hurricane_intensity.splits import load_splits

    _, val_storms, _ = load_splits(train_list=train_list, val_list=VAL_LIST)  # train.py:40, same defaults
    val_ds = TCDataset("data/processed", val_storms, circular_mask=False)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae = nn.L1Loss()
    out = {}
    for e in range(1, epochs + 1):
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(Path(epoch_dir) / f"epoch_{e:02d}.pt", map_location=device))
        model = model.to(device)
        model.eval()
        total = 0.0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                total += mae(model(images).squeeze(1), labels).item() * images.size(0)
        out[e] = total / len(val_ds) * 185
    return out


def check_output_paths(paths):
    """Every output is under the N5 roots and never under a protected or frozen path (design §8)."""
    bad = [p for p in paths if not p.startswith(OUTPUT_ROOTS) or any(p.startswith(n) for n in NEVER_WRITTEN)]
    if bad:
        raise LaunchError(f"refusing output paths outside {OUTPUT_ROOTS} or under a never-written path: {bad}")


def planned_outputs(seeds):
    return [v for _, arm, s in schedule(seeds) for v in run_paths(arm, s).values()]


def h9_outputs(root, seeds):
    """Every H9 output (CSV, eval log or manifest) of any run that exists on disk."""
    return [p[k] for _, arm, s in schedule(seeds) for p in [run_paths(arm, s)] for k in H9_KEYS
            if k in p and (Path(root) / p[k]).exists()]


def looked(root, seeds):
    """Evidence of an H9 look: any pinned H9 output or manifest, any aborted H9 evaluation, or the phase-2 marker.
    abort-eval moves pinned H9 outputs aside, so the pinned paths alone can't say whether H9 was seen (item 16b)."""
    found = h9_outputs(root, seeds)
    found += sorted(f.relative_to(root).as_posix() for f in Path(root).glob(f"{ABORTED}/*_eval*/abort.json"))
    return found + ([PHASE2_MARKER] if (Path(root) / PHASE2_MARKER).exists() else [])


def doc_path(p):
    return p in DOC_PATHS or (p.startswith("reports/decisions/") and p.endswith(".md") and p.count("/") == 2)


def abort_files(root, arm, seed, kind):
    """abort.json paths of one run's aborts of one kind ("attempt" or "eval"), in attempt order."""
    found = Path(root).glob(f"{ABORTED}/{arm}_seed{seed}_{kind}*/abort.json")
    return sorted(found, key=lambda f: int(f.parent.name.rsplit(kind, 1)[1]))


def aborted_attempts(root, arm, seed, kind="attempt"):
    return [f.relative_to(root).as_posix() for f in abort_files(root, arm, seed, kind)]


def pending_aborts(root):
    return sorted(f.relative_to(root).as_posix() for f in Path(root).glob(f"{ABORTED}/*/abort.json")
                  if json.loads(f.read_text(encoding="utf-8")).get("status") != "complete")


def verify_record(root, arm, seed, expected):
    """A training record is valid only while every file it hashed is unchanged and its protected hashes held."""
    rel = run_paths(arm, seed)["record"]
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


def run_state(root, arm, seed, expected):
    """fresh (nothing on disk), partial (files but no record), bad_record (a record that does not verify), trained,
    eval_partial (H9 outputs but no manifest) or evaluated."""
    p = run_paths(arm, seed)
    if not (Path(root) / p["record"]).exists():
        return "partial" if any((Path(root) / v).exists() for v in p.values()) else "fresh"
    try:
        verify_record(root, arm, seed, expected)
    except LaunchError:
        return "bad_record"
    if (Path(root) / p["manifest"]).exists():
        return "evaluated"
    return "eval_partial" if any((Path(root) / p[k]).exists() for k in H9_KEYS if k in p) else "trained"


class Lock:
    """One run_n5 stage that writes at a time. A stale lock (a killed process) is removed by hand, after checking
    that no run_n5 is running."""

    def __init__(self, root):
        self.path = Path(root) / LOCK

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise LaunchError(f"{LOCK} exists: another run_n5 stage is running (or one was killed; remove the file by "
                              "hand only after checking that none is running)") from None
        os.write(fd, f"pid {os.getpid()} {now_iso()}\n".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink()


def head_blob(git, commit):
    blob = git(["show", f"{commit}:{SELF_ANALYSIS}"]).stdout
    blob_id = git(["rev-parse", f"{commit}:{SELF_ANALYSIS}"]).stdout.decode().strip()
    return hashlib.sha256(blob).hexdigest(), blob_id


def common_guards(root, git, main_root, check, cuda_ok, child_import):
    """Guards 1, 2, 3 and 6, shared by both phases. Returns what both need."""
    if git(["status", "--porcelain", "--untracked-files=no"]).stdout.strip():
        raise LaunchError("guard 1: the tracked tree has uncommitted changes; launch from a clean commit")
    if git(["diff", "--quiet", "HEAD", "--", SELF_ANALYSIS], check=False).returncode != 0:
        raise LaunchError(f"guard 1: {SELF_ANALYSIS} differs from its committed blob at HEAD")
    A = analysis(root)
    try:
        design = A.design_constants()
    except A.DesignError as e:
        raise LaunchError(f"guard 2: {e} (ARM_TRAIN_FRAMES and FRAME_LIST_SHA256 come from the build)") from None
    frame_list = root / FRAME_LIST
    if not frame_list.is_file() or sha256_file(frame_list) != A.FRAME_LIST_SHA256:
        raise LaunchError(f"guard 2: {FRAME_LIST} is missing or its SHA-256 is not FRAME_LIST_SHA256")
    expected = dict(check.load_expected(Path(main_root) / "COORDINATOR.md"))
    if expected != A.PROTECTED:
        raise LaunchError("guard 3: analyze_n5.PROTECTED differs from the COORDINATOR.md protected block")
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
    blob_sha, blob_id = head_blob(git, commit)
    return dict(A=A, design=design, expected=expected, commit=commit, blob_sha=blob_sha, blob_id=blob_id, git=git)


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
    check_output_paths(planned_outputs(A.SEEDS))
    if (root / PHASE2_MARKER).exists():
        raise LaunchError(f"guard 4: {PHASE2_MARKER} exists: phase 2 has started, so no training may run "
                          "(2026-10-19 amendment, condition (b))")
    states = {(arm, s): run_state(root, arm, s, expected) for _, arm, s in schedule(A.SEEDS)}
    bad = [(arm, s) for (arm, s), st in states.items() if st == "bad_record"]
    if bad:
        raise LaunchError(f"guard 4: training records that do not verify: {bad}; " + (
            RESTORE_AFTER_LOOK if looked(root, A.SEEDS) else
            "move each run aside with `run_n5.py abort-run ARM SEED --reason ...` and launch again"))
    partial = [v for (arm, s), st in states.items() if st == "partial"
               for v in run_paths(arm, s).values() if (root / v).exists()]
    if partial:
        raise LaunchError(f"guard 4: unfinished runs have files on disk: {partial[:5]}{' ...' if len(partial) > 5 else ''}"
                          "; move each run aside with `run_n5.py abort-run ARM SEED --reason ...`")
    stale = []
    for (arm, s), st in states.items():
        for k in ("record", "manifest"):
            f = root / run_paths(arm, s)[k]
            if f.exists():
                rec = json.loads(f.read_text(encoding="utf-8"))
                if (rec.get("commit"), rec.get("analysis_blob_sha256")) != (ctx["commit"], ctx["blob_sha"]):
                    stale.append((run_paths(arm, s)[k], rec.get("commit")))
    if stale:
        raise LaunchError(f"guard 4: runs recorded at another commit or analysis blob than HEAD {ctx['commit']}: "
                          f"{stale[:3]}; relaunch only at their commit, or (before any H9 output) move every run aside "
                          "with `run_n5.py abort-all --reason ...` and restart all 20 (§14, decided by John)")
    seen = looked(root, A.SEEDS)
    if seen and any(st == "fresh" for st in states.values()):
        raise LaunchError(f"guard 4: H9 has been looked at while runs remain to be trained: {seen[:3]}; no training "
                          "may follow an H9 evaluation (2026-10-19 amendment, condition (b))")
    first_parent(git, ctx["commit"])
    return ctx


def child_env():
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return env


def write_line(f, text):
    f.write((text + "\n").encode("utf-8"))
    f.flush()


def write_json_new(path, obj):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1) + "\n")


def write_json_replace(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def run_step(runner, cmd, log_path, root, header=None):
    """One subprocess with stdout+stderr appended, unchanged, to log_path. Returns the return code."""
    with open(log_path, "ab") as f:
        for line in header or ():
            write_line(f, line)
        return runner(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(root), env=child_env()).returncode


def evaluate_step(runner, py, root, checkpoint, train_list, out, log):
    with open(root / log, "xb"):
        pass
    if run_step(runner, eval_cmd(py, checkpoint, train_list, out), root / log, root) != 0:
        with open(root / log, "ab") as f:
            write_line(f, f"EVAL_FAILED {now_iso()}")
        raise LaunchError(f"evaluation failed: {log}")


def train_one(order, arm, seed, ctx, root, main_root, hooks):
    """Phase 1 for one run: train, re-validate and select (control), write the training record. No H9."""
    A, design, py = ctx["A"], ctx["design"], sys.executable
    p = run_paths(arm, seed)
    existing = [v for v in p.values() if (root / v).exists()]
    if existing:
        raise LaunchError(f"refusing to overwrite {existing}")
    seen = looked(root, A.SEEDS)
    if seen:
        raise LaunchError(f"phase 1: H9 has been looked at ({seen[:3]}); no training may follow an H9 evaluation")
    before = protected_now(main_root, ctx["expected"])
    if before != ctx["expected"]:
        raise LaunchError(f"protected hashes differ before {arm} seed {seed}")
    for d in ("logs/n5_atlantic/runs", f"models/n5_atlantic/{arm}", "results/n5_atlantic"):
        (root / d).mkdir(parents=True, exist_ok=True)
    log = root / p["train_log"]
    with open(log, "xb"):
        pass
    header = [f"RUN arm={arm} seed={seed} order={order} list={design['train_lists'][arm]} val={VAL_LIST} "
              f"epochs={design['epochs'][arm]}", f"START {now_iso()} | {hooks['snapshot']()}"]
    rc = run_step(hooks["runner"], train_cmd(py, arm, seed, design["train_lists"], design["epochs"]), log, root, header)
    with open(log, "ab") as f:
        write_line(f, f"TRAIN_FAILED rc={rc} {now_iso()}" if rc != 0 else f"END {now_iso()} | {hooks['snapshot']()}")
    if rc != 0:
        raise LaunchError(f"training failed: {p['train_log']}")
    ep, native_epoch = A.parse_train_log(log, arm, seed, design)

    rec = {"arm": arm, "seed": seed, "order": order, "commit": ctx["commit"],
           "analysis_blob_sha256": ctx["blob_sha"], "analysis_blob_id": ctx["blob_id"],
           "train_list": design["train_lists"][arm], "frame_list_sha256": design["frame_list_sha"][arm],
           "train_log": p["train_log"], "native_checkpoint": p["native_checkpoint"],
           "aborted_attempts": aborted_attempts(root, arm, seed)}
    kept = [p["train_log"], p["native_checkpoint"]]
    if arm == "control":
        reval = hooks["revalidate_fn"](root / p["epoch_dir"], A.CONTROL_EPOCHS, design["train_lists"][arm])
        logged = {e: va for e, _, va in ep}
        if sorted(reval) != list(range(1, A.CONTROL_EPOCHS + 1)) or \
                any(abs(round(v, 2) - logged[e]) > 0.0051 for e, v in reval.items()):
            raise LaunchError(f"control seed {seed}: re-validated values do not round to the logged ones")
        try:
            sel, tie = A.control_selection(reval, native_epoch)
        except A.DesignError as e:
            raise LaunchError(f"control seed {seed}: {e}") from None
        epoch_dir = root / p["epoch_dir"]
        if epoch_dir.resolve() != root.resolve() / p["epoch_dir"]:
            raise LaunchError(f"control seed {seed}: {p['epoch_dir']} resolves to {epoch_dir.resolve()}, outside "
                              f"{root}; nothing deleted")
        if not hooks["same_weights_fn"](root / p["native_checkpoint"], epoch_dir / f"epoch_{native_epoch:02d}.pt"):
            raise LaunchError(f"control seed {seed}: {p['native_checkpoint']} does not hold epoch_{native_epoch:02d}.pt's "
                              "weights; train.py's last 'saved' was not parsed right; nothing deleted")
        for e in range(2, A.CONTROL_EPOCHS + 1, 2):  # even epochs are not candidates; odd ones stay for the analysis
            (epoch_dir / f"epoch_{e:02d}.pt").unlink()
        kept += [f"{p['epoch_dir']}/epoch_{e:02d}.pt" for e in A.CONTROL_GRID]
        selected = p["native_checkpoint"] if tie else f"{p['epoch_dir']}/epoch_{sel:02d}.pt"
        rec.update(selected_epoch=sel, selected_checkpoint=selected, tie_rule=tie,
                   off_grid_primary=bool(tie) and sel % 2 == 0,
                   odd_grid_near_ties=A.odd_grid_near_ties(reval),
                   val_mae_revalidated={str(e): reval[e] for e in sorted(reval)})
    else:
        rec.update(selected_epoch=native_epoch, selected_checkpoint=p["native_checkpoint"])
    for k in ("train_log", "native_checkpoint", "selected_checkpoint"):
        rec[f"{k}_sha256"] = sha256_file(root / rec[k])
    rec["files_sha256"] = {f: sha256_file(root / f) for f in kept}
    rec["protected_before"] = before
    rec["protected_after"] = protected_now(main_root, ctx["expected"])
    if rec["protected_after"] != ctx["expected"]:
        changed = sorted(k for k, v in ctx["expected"].items() if rec["protected_after"][k] != v)
        raise LaunchError(f"protected hashes changed during {arm} seed {seed}: {changed}")
    write_json_new(root / p["record"], rec)
    for f in kept:  # an accidental write now fails instead of breaking the record's hash (item 16c)
        os.chmod(root / f, stat.S_IREAD)
    print(f"order {order}: {arm} seed {seed} trained; selected epoch {rec['selected_epoch']} (native {native_epoch})"
          + (f"; tie rule fired {tie}" if rec.get("tie_rule") else ""), flush=True)
    return rec


def train_phase(ctx, root, main_root, hooks):
    for order, arm, s in schedule(ctx["A"].SEEDS):
        if run_state(root, arm, s, ctx["expected"]) in ("trained", "eval_partial", "evaluated"):
            print(f"order {order}: {arm} seed {s} already trained; skipped", flush=True)
            continue
        train_one(order, arm, s, ctx, root, main_root, hooks)


def records_evidence(root, git, A):
    """Guard 5e: the 20 records are the pushed witness that all trainings finished before any H9 look."""
    recs = []
    for _, arm, s in schedule(A.SEEDS):
        rel = run_paths(arm, s)["record"]
        if not (root / rel).is_file():
            raise LaunchError(f"phase 2: H9 evaluation refuses to start until all 20 training runs have finished; "
                              f"{rel} is missing")
        if git(["ls-files", "--error-unmatch", rel], check=False).returncode != 0:
            raise LaunchError(f"guard 5e: {rel} is not tracked; commit and push the 20 training records first")
        if git(["show", f"HEAD:{rel}"], check=False).stdout != (root / rel).read_bytes().replace(b"\r\n", b"\n"):
            raise LaunchError(f"guard 5e: {rel} differs from its committed blob at HEAD")
        recs.append(json.loads((root / rel).read_text(encoding="utf-8")))
    try:
        launch_commit = A.single_commit(recs)
    except A.DesignError as e:
        raise LaunchError(f"phase 2: {e}") from None
    first_parent(git, launch_commit)
    if git(["merge-base", "--is-ancestor", "HEAD", UPSTREAM], check=False).returncode != 0:
        raise LaunchError(f"guard 5e: HEAD is not pushed to {UPSTREAM}; push the commit with the 20 records first")
    allowed = {run_paths(arm, s)["record"] for _, arm, s in schedule(A.SEEDS)} | {A.DEVIATIONS_PATH}
    changed = [p for p in git(["diff", "--no-renames", "-z", "--name-only", launch_commit, "HEAD"]).stdout
               .decode("utf-8").split("\0") if p]
    other = sorted(p for p in changed if p not in allowed and not doc_path(p))
    if other:
        raise LaunchError(f"guard 5e: HEAD differs from the launch commit {launch_commit} in more than the 20 "
                          f"records and the decisions docs: {other[:5]}")
    return launch_commit, recs


def deviations_in(root, git, A, path, states):
    """Rule (A) in phase 2: the committed DEVIATIONS_PATH names the seed pairs left out; each must have a run whose
    record no longer verifies (hc-adversary item 17). Returns the left-out (arm, seed) runs."""
    if path is None:
        return set()
    if Path(path).as_posix() != A.DEVIATIONS_PATH:
        raise LaunchError(f"--deviations must be {A.DEVIATIONS_PATH}, not {path}")
    if git(["ls-files", "--error-unmatch", A.DEVIATIONS_PATH], check=False).returncode != 0 or \
            git(["show", f"HEAD:{A.DEVIATIONS_PATH}"], check=False).stdout != \
            (root / A.DEVIATIONS_PATH).read_bytes().replace(b"\r\n", b"\n"):
        raise LaunchError(f"{A.DEVIATIONS_PATH} must be committed, and identical to its blob at HEAD")
    try:
        dropped = A.load_deviations(root / A.DEVIATIONS_PATH)
    except A.DesignError as e:
        raise LaunchError(f"phase 2: {e}") from None
    healthy = [s for s in dropped if all(states[(arm, s)] != "bad_record" for arm in A.ARMS)]
    if healthy:
        raise LaunchError(f"phase 2: {A.DEVIATIONS_PATH} names seeds {healthy}, but none of their runs has a record "
                          "that fails; rule (A) drops only such a pair")
    return {(arm, s) for s in dropped for arm in A.ARMS}


def evaluate_phase(ctx, root, main_root, runner, deviations=None):
    """Condition (b): no H9 evaluation runs until all 20 training runs have finished, AND every training-side check
    the analysis will apply has passed on all 20 records (hc-adversary item 16). Under rule (A) the pairs named in
    the deviations file are left out of every check and step here, as the analysis leaves them out."""
    A, git, order = ctx["A"], ctx["git"], schedule(ctx["A"].SEEDS)
    states = {(arm, s): run_state(root, arm, s, ctx["expected"]) for _, arm, s in order}
    skip = deviations_in(root, git, A, deviations, states)
    order = [(k, arm, s) for k, arm, s in order if (arm, s) not in skip]
    states = {(arm, s): st for (arm, s), st in states.items() if (arm, s) not in skip}
    bad = [(arm, s) for (arm, s), st in states.items() if st == "bad_record"]
    if bad:
        raise LaunchError(f"phase 2: training records that do not verify: {bad}; " + (
            RESTORE_AFTER_LOOK if looked(root, A.SEEDS) else
            "no H9 output exists yet, so move each run aside with `run_n5.py abort-run ARM SEED --reason ...` and "
            "launch again"))
    unfinished = [(k, arm, s) for k, arm, s in order if states[(arm, s)] not in ("trained", "eval_partial", "evaluated")]
    if unfinished:
        raise LaunchError(f"phase 2: H9 evaluation refuses to start until all {len(order)} training runs have "
                          f"finished; unfinished: {unfinished}")
    half = [(arm, s) for (arm, s), st in states.items() if st == "eval_partial"]
    if half:
        raise LaunchError(f"phase 2: H9 outputs without a manifest for {half}; move them aside with "
                          "`run_n5.py abort-eval ARM SEED --reason ...`, then evaluate again")
    launch_commit, recs = records_evidence(root, git, A)
    if {r.get("analysis_blob_sha256") for r in recs} != {ctx["blob_sha"]}:
        raise LaunchError(f"phase 2: the records' analysis blob is not HEAD's {ctx['blob_sha']}")
    try:
        A.verify_self(launch_commit, root, git)
        for (_, arm, s), rec in zip(schedule(A.SEEDS), recs):
            if (arm, s) not in skip:
                A.check_training(rec, arm, s, ctx["design"], root, run_paths(arm, s)["record"])
    except A.DesignError as e:
        raise LaunchError(f"phase 2 entry, before any H9 evaluation: {e}") from None
    if not (root / PHASE2_MARKER).exists():
        write_json_new(root / PHASE2_MARKER, {"head": ctx["commit"], "launch_commit": launch_commit, "time": now_iso()})
    mans = []
    for k, arm, s in order:
        p = run_paths(arm, s)
        if states[(arm, s)] == "evaluated":
            try:
                A.load_run(root / p["manifest"], arm, s, ctx["design"], root)
            except A.DesignError as e:
                raise LaunchError(f"phase 2: {e}; move its H9 outputs and manifest aside with "
                                  f"`run_n5.py abort-eval {arm} {s} --reason ...` to re-evaluate the same checkpoint") from None
            mans.append(json.loads((root / p["manifest"]).read_text(encoding="utf-8")))
            print(f"order {k}: {arm} seed {s} already evaluated; skipped", flush=True)
        else:
            mans.append(evaluate_one(k, arm, s, ctx, root, main_root, runner))
    return mans


def evaluate_one(order, arm, seed, ctx, root, main_root, runner):
    """Phase 2 for one run: H9 evaluation of the recorded selection (and the control's native), then the manifest."""
    A, design, py = ctx["A"], ctx["design"], sys.executable
    p = run_paths(arm, seed)
    man = verify_record(root, arm, seed, ctx["expected"])
    man.update(h9_csv=p["h9_csv"], eval_log=p["eval_log"], eval_attempts=aborted_attempts(root, arm, seed, "eval"))
    try:
        A.check_attempts({"eval_attempts": man["eval_attempts"]}, root, p["record"])
    except A.DesignError as e:
        raise LaunchError(f"before the H9 evaluation of {arm} seed {seed}: {e}") from None
    if arm == "control":
        man.update(native_h9_csv=p["native_h9_csv"], native_eval_log=p["native_eval_log"])
    before = protected_now(main_root, ctx["expected"])
    if before != ctx["expected"]:
        raise LaunchError(f"protected hashes differ before the H9 evaluation of {arm} seed {seed}")
    evaluate_step(runner, py, root, man["selected_checkpoint"], man["train_list"], p["h9_csv"], p["eval_log"])
    if arm == "control":
        evaluate_step(runner, py, root, p["native_checkpoint"], man["train_list"], p["native_h9_csv"], p["native_eval_log"])
    for k in ("train_log", "native_checkpoint", "selected_checkpoint", "h9_csv", "eval_log", "native_h9_csv",
              "native_eval_log"):
        if k in man:
            man[f"{k}_sha256"] = sha256_file(root / man[k])
    man["protected_after_eval"] = protected_now(main_root, ctx["expected"])
    write_json_new(root / p["manifest"], man)
    if man["protected_after_eval"] != ctx["expected"]:
        raise LaunchError(f"protected hashes changed during the H9 evaluation of {arm} seed {seed}")
    A.load_run(root / p["manifest"], arm, seed, design, root)  # the analysis's own linkage checks, now
    print(f"order {order}: {arm} seed {seed} evaluated on H9", flush=True)
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
    """Phase 1 only: all 20 trainings, then stop. The keyword hooks exist for the synthetic tests only; main()
    exposes none."""
    root, git, main_root, check, hooks = resolve_hooks(root, git, main_root, check, runner=runner, snapshot=snapshot,
                                                       revalidate_fn=revalidate_fn, same_weights_fn=same_weights_fn)
    with Lock(root):
        ctx = preflight(root, git, main_root, check, cuda_ok, child_import)
        print(f"HEAD {ctx['commit']}; analysis blob sha256 {ctx['blob_sha']} (id {ctx['blob_id']})", flush=True)
        train_phase(ctx, root, main_root, hooks)
    records = [run_paths(arm, s)["record"] for _, arm, s in schedule(ctx["A"].SEEDS)]
    print("PHASE 1 DONE. Commit these 20 training records, push them to phase2-experiments, then run "
          "`run_n5.py evaluate`:", *records, sep="\n  ", flush=True)
    return records


def evaluate(root=ROOT, *, deviations=None, git=None, main_root=None, check=None, runner=subprocess.run,
             cuda_ok=cuda_available, child_import=child_package_path):
    """Phase 2 only. deviations: rule (A)'s committed file (main() passes --deviations). The other keyword hooks
    exist for the synthetic tests only; main() exposes none."""
    root, git, main_root, check, _ = resolve_hooks(root, git, main_root, check)
    with Lock(root):
        ctx = common_guards(root, git, main_root, check, cuda_ok, child_import)
        print(f"HEAD {ctx['commit']}; analysis blob sha256 {ctx['blob_sha']} (id {ctx['blob_id']})", flush=True)
        return evaluate_phase(ctx, root, main_root, runner, deviations)


def _abort_one(root, A, arm, seed, reason, kind, keys):
    """Move one run's files for `keys` to aborted/<arm>_seed<s>_<kind><k>/: abort.json (pending, every file's
    SHA-256) first, then the moves, then status complete. A pending abort of this run and kind is resumed."""
    p = run_paths(arm, seed)
    order = {(a, s): k for k, a, s in schedule(A.SEEDS)}[(arm, seed)]
    pending = [f for f in abort_files(root, arm, seed, kind)
               if json.loads(f.read_text(encoding="utf-8")).get("status") != "complete"]
    if pending:
        dest_rel = pending[0].parent.relative_to(root).as_posix()
        info = json.loads(pending[0].read_text(encoding="utf-8"))
    else:
        dest_rel = f"{ABORTED}/{arm}_seed{seed}_{kind}{len(abort_files(root, arm, seed, kind)) + 1}"
        info = {"arm": arm, "seed": seed, "order": order, "kind": kind, "attempt": int(dest_rel.rsplit(kind, 1)[1]),
                "reason": reason, "time": now_iso(), "status": "pending", "moved": {}}
    present = [p[k] for k in keys if k in p and (root / p[k]).exists()]
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


def abort_run(arm, seed, reason, root=ROOT):
    """Move one unfinished training run (or one whose record does not verify) aside; launch restarts it in its order
    position with its seed."""
    root = Path(root)
    with Lock(root):
        A = analysis(root)
        if (arm, seed) not in {(a, s) for _, a, s in schedule(A.SEEDS)} or not reason:
            raise LaunchError(f"abort-run: unknown run {arm} seed {seed}, or no --reason")
        seen = looked(root, A.SEEDS)
        if seen:
            raise LaunchError(f"abort-run: H9 has been looked at ({seen[:3]}); no run may be aborted after an H9 evaluation")
        if run_state(root, arm, seed, A.PROTECTED) in ("trained", "eval_partial", "evaluated"):
            raise LaunchError(f"abort-run: {arm} seed {seed} has a training record that verifies; finished runs are "
                              "not aborted (abort-all restarts all 20, §14)")
        return _abort_one(root, A, arm, seed, reason, "attempt", TRAIN_KEYS)


def abort_all(reason, root=ROOT):
    """§14, decided by John 2026-10-19 (option (i)): before any H9 output, move every run aside, finished ones
    included, so a
    relaunch from a fix commit restarts all 20 in the same order."""
    root = Path(root)
    with Lock(root):
        A = analysis(root)
        if not reason:
            raise LaunchError("abort-all: no --reason")
        seen = looked(root, A.SEEDS)
        if seen:
            raise LaunchError(f"abort-all: H9 has been looked at ({seen[:3]}); no run may be aborted after an H9 evaluation")
        done = []
        for _, arm, s in schedule(A.SEEDS):
            if run_state(root, arm, s, A.PROTECTED) != "fresh" or any(
                    json.loads(f.read_text(encoding="utf-8")).get("status") != "complete"
                    for f in abort_files(root, arm, s, "attempt")):
                done.append(_abort_one(root, A, arm, s, f"abort-all: {reason}", "attempt", TRAIN_KEYS))
        if not done:
            raise LaunchError("abort-all: no run has files to move")
        return done


def abort_eval(arm, seed, reason, root=ROOT):
    """Move a crashed H9 evaluation's outputs aside (hashed, never deleted); evaluate then re-evaluates the same
    recorded checkpoint, and the manifest lists the attempt."""
    root = Path(root)
    with Lock(root):
        A = analysis(root)
        if (arm, seed) not in {(a, s) for _, a, s in schedule(A.SEEDS)} or not reason:
            raise LaunchError(f"abort-eval: unknown run {arm} seed {seed}, or no --reason")
        state = run_state(root, arm, seed, A.PROTECTED)
        pending = any(json.loads(f.read_text(encoding="utf-8")).get("status") != "complete"
                      for f in abort_files(root, arm, seed, "eval"))
        if state == "evaluated":
            try:
                A.load_run(root / run_paths(arm, seed)["manifest"], arm, seed, A.design_constants(), root)
            except A.DesignError:
                state = "evaluated, refused by load_run"
        if state not in ("eval_partial", "evaluated, refused by load_run") and not (pending and state == "trained"):
            raise LaunchError(f"abort-eval: {arm} seed {seed} is {state}; only a run with a verified training record "
                              "and H9 outputs without a manifest (or with a manifest load_run refuses) can be "
                              "re-evaluated")
        return _abort_one(root, A, arm, seed, reason, "eval", H9_KEYS)  # the same recorded checkpoint is re-evaluated


def plan(root=ROOT):
    A = analysis(root)
    epochs = {"control": A.CONTROL_EPOCHS,
              "atlantic": A.epochs_for(A.ARM_TRAIN_FRAMES)[0] if A.ARM_TRAIN_FRAMES else "<epochs_for(ARM_TRAIN_FRAMES)>"}
    py = "PY"
    print("PHASE 1 (launch): the 20 trainings; no H9 evaluation")
    for order, arm, s in schedule(A.SEEDS):
        p = run_paths(arm, s)
        print(f"order {order}: {arm} seed {s}")
        print("  train:", train_cmd(py, arm, s, A.TRAIN_LISTS, epochs), "->", p["train_log"])
        if arm == "control":
            print("  re-validate all epochs on", VAL_LIST, "only; analyze_n5.control_selection: argmin must be the last "
                  f"'saved' (tie window {A.TIE_KT} kt: train.py's checkpoint), else the first minimum over",
                  list(A.CONTROL_GRID))
        print("  record:", p["record"])
    print("BETWEEN THE PHASES: commit the 20 records and push them to phase2-experiments")
    print("PHASE 2 (evaluate): only once the 20 committed records pass every training-side check")
    for order, arm, s in schedule(A.SEEDS):
        p = run_paths(arm, s)
        print(f"order {order}: {arm} seed {s}")
        if arm == "control":
            print("  eval:", eval_cmd(py, "<selected_checkpoint>", A.TRAIN_LISTS[arm], p["h9_csv"]), "->", p["eval_log"])
            print("  native:", eval_cmd(py, p["native_checkpoint"], A.TRAIN_LISTS[arm], p["native_h9_csv"]),
                  "->", p["native_eval_log"])
        else:
            print("  eval:", eval_cmd(py, p["native_checkpoint"], A.TRAIN_LISTS[arm], p["h9_csv"]), "->", p["eval_log"])
        print("  manifest:", p["manifest"])


def build_spec():
    """The build and exclusion steps as run (2026-10-19). Nothing here downloads, builds or scans."""
    print("BUILD (design §12.3; done 2026-10-19 in the main checkout; not run by this script):")
    print("  hash data/interim/hurdat2_atlantic.txt before; it must be ee35819c...0bcb5c")
    print("  ", ["PY", "scripts/build_batch.py", "--ids-file", BUILD_IDS, "--sample-interval", "3h"])
    print("   GOES-16 default; never --force-hurdat2-refresh or --hurdat2-url")
    print("  hash data/interim/hurdat2_atlantic.txt after; it must be unchanged")
    print("EXCLUSIONS (design §12.4; done): ['PY', 'check_nan.py', '--processed-root', <main>/data/processed]")
    print("  rescans all non-spent storms; the diff of excluded_crops.txt is additions under the 31 new IDs only")
    print("FRAME LIST (design §12.5; done): ['PY', 'scripts/n5_frame_list.py'] drops the 4 Marco AL142020 frames")
    print("  concurrent with a protected storm (scripts/concurrency.py, 650 km box) and writes", FRAME_LIST)
    print("  (tracked, -text); its count and SHA-256 are ARM_TRAIN_FRAMES and FRAME_LIST_SHA256 in", SELF_ANALYSIS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["plan", "build-spec", "launch", "evaluate", "abort-run", "abort-all", "abort-eval"])
    ap.add_argument("arm", nargs="?", choices=["control", "atlantic"], help="abort-run and abort-eval only")
    ap.add_argument("seed", nargs="?", type=int, help="abort-run and abort-eval only")
    ap.add_argument("--reason", help="abort stages only: why (recorded in abort.json)")
    ap.add_argument("--deviations", help="evaluate only: rule (A)'s committed logs/n5_atlantic/deviations.json")
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
    elif args.stage == "build-spec":
        build_spec()
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
