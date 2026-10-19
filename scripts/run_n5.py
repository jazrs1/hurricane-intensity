"""N5 driver: the 20 training runs, their H9 evaluations and per-run manifests (reports/v2_survey/n5/design.md §8 and
its 2026-10-19 amendment).

    python scripts/run_n5.py plan          # print every command of the 20 runs; runs nothing
    python scripts/run_n5.py build-spec    # print the build and exclusion steps (§12.3-4); runs nothing
    python scripts/run_n5.py launch        # phase 1, then phase 2; resumable; refuses until PR B fills the constants
    python scripts/run_n5.py abort-run ARM SEED --reason TEXT   # move an unfinished run aside (amendment, abort)

launch refuses to start unless:
  1. the tracked tree is clean, and scripts/analyze_n5.py equals its committed blob at HEAD;
  2. analyze_n5.design_constants() is final (ARM_TRAIN_FRAMES and FRAME_LIST_SHA256 filled), and the frame list
     exists with that SHA-256;
  3. analyze_n5.PROTECTED equals the protected block in the main checkout's COORDINATOR.md, and every protected
     file has that hash;
  4. every output path of all 20 runs is under {models,results,logs}/n5_atlantic/ and outside the design's
     never-written paths; no unfinished run has files on disk (abort-run moves them aside); and no H9 output exists
     while any run is still to be trained;
  5. after `git fetch origin`, HEAD is an ancestor of origin/phase2-experiments (amendment guard B);
  6. the checkout is the main checkout, not a worktree, and torch.cuda.is_available() is True (amendment guard C).

Phase 1 trains the 20 runs in the §4 order and evaluates nothing on H9. Each step is a subprocess with list
arguments (no shell); its stdout and stderr go to a log opened in binary, so the bytes are written unchanged (UTF-8,
no BOM; the child gets PYTHONUTF8=1). The driver writes RUN/START before and END (or TRAIN_FAILED) after each
training run. The control trains with --epoch-dir; all 25 epochs are then re-validated on validation only, and the
selection follows analyze_n5.control_selection: the unrounded argmin must be train.py's own pick (its last "saved"),
except within the tie window (|d| < TIE_KT), where train.py's saved checkpoint is the one evaluated; otherwise the
first minimum over the odd epochs is. The 12 even-epoch files are deleted after that check; the 13 odd ones are kept
until the analysis has run. A finished run gets a training record, logs/n5_atlantic/runs/<arm>_seed<s>.train.json.
Phase 2 starts only when all 20 training records exist and verify. It evaluates each run's selected checkpoint (and
the control's native one) on H9, writes the manifest and checks it with analyze_n5's own load_run. The protected
hashes are taken before and after every step; any mismatch stops the driver.

A failed run stops the driver. Abort procedure (design.md, 2026-10-19 amendment): `abort-run ARM SEED --reason`
moves that run's files, unchanged, to logs/n5_atlantic/aborted/<arm>_seed<s>_attempt<k>/ and records them with their
SHA-256s in abort.json there; `launch` then skips the finished runs and restarts that run in its original order
position with its original seed. abort-run refuses once any H9 output exists, and refuses a finished run.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
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


def aborted_attempts(root, arm, seed):
    found = Path(root).glob(f"{ABORTED}/{arm}_seed{seed}_attempt*/abort.json")
    return [f.relative_to(root).as_posix() for f in sorted(found, key=lambda f: int(f.parent.name.rsplit("attempt", 1)[1]))]


def verify_record(root, arm, seed, expected):
    """A training record is valid only while every file it hashed is unchanged and its protected hashes held."""
    rel = run_paths(arm, seed)["record"]
    rec = json.loads((Path(root) / rel).read_text(encoding="utf-8"))
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
    """fresh (nothing on disk), partial (files but no record), trained, eval_partial or evaluated."""
    p = run_paths(arm, seed)
    if not (Path(root) / p["record"]).exists():
        return "partial" if any((Path(root) / v).exists() for v in p.values()) else "fresh"
    verify_record(root, arm, seed, expected)
    if (Path(root) / p["manifest"]).exists():
        return "evaluated"
    return "eval_partial" if any((Path(root) / p[k]).exists() for k in H9_KEYS if k in p) else "trained"


def preflight(root, git, main_root, check, cuda_ok):
    """Guards 1-6 (module docstring), in order. Returns what launch needs."""
    if git(["status", "--porcelain", "--untracked-files=no"]).stdout.strip():
        raise LaunchError("guard 1: the tracked tree has uncommitted changes; launch from a clean commit")
    if git(["diff", "--quiet", "HEAD", "--", SELF_ANALYSIS], check=False).returncode != 0:
        raise LaunchError(f"guard 1: {SELF_ANALYSIS} differs from its committed blob at HEAD")
    A = analysis(root)
    try:
        design = A.design_constants()
    except A.DesignError as e:
        raise LaunchError(f"guard 2: {e} (PR B fills ARM_TRAIN_FRAMES and FRAME_LIST_SHA256)") from None
    frame_list = root / FRAME_LIST
    if not frame_list.is_file() or sha256_file(frame_list) != A.FRAME_LIST_SHA256:
        raise LaunchError(f"guard 2: {FRAME_LIST} is missing or its SHA-256 is not FRAME_LIST_SHA256")
    expected = dict(check.load_expected(Path(main_root) / "COORDINATOR.md"))
    if expected != A.PROTECTED:
        raise LaunchError("guard 3: analyze_n5.PROTECTED differs from the COORDINATOR.md protected block")
    actual = protected_now(main_root, expected)
    if actual != expected:
        raise LaunchError(f"guard 3: protected files differ: {sorted(k for k in expected if actual[k] != expected[k])}")
    check_output_paths(planned_outputs(A.SEEDS))
    states = {(arm, s): run_state(root, arm, s, expected) for _, arm, s in schedule(A.SEEDS)}
    partial = [v for (arm, s), st in states.items() if st == "partial"
               for v in run_paths(arm, s).values() if (root / v).exists()]
    if partial:
        raise LaunchError(f"guard 4: unfinished runs have files on disk: {partial[:5]}{' ...' if len(partial) > 5 else ''}"
                          "; move each run aside with `run_n5.py abort-run ARM SEED --reason ...`")
    looked = h9_outputs(root, A.SEEDS)
    if looked and any(st == "fresh" for st in states.values()):
        raise LaunchError(f"guard 4: H9 outputs exist while runs remain to be trained: {looked[:3]}; no training "
                          "may follow an H9 evaluation (2026-10-19 amendment, condition (b))")
    if git(["fetch", "origin"], check=False).returncode != 0:
        raise LaunchError("guard 5: git fetch origin failed; can't check the launch commit against the remote")
    if git(["merge-base", "--is-ancestor", "HEAD", UPSTREAM], check=False).returncode != 0:
        raise LaunchError(f"guard 5: HEAD is not an ancestor of {UPSTREAM}; launch only from a merged commit")
    if Path(root).resolve() != Path(main_root).resolve() or not (Path(root) / ".git").is_dir():
        raise LaunchError(f"guard 6: {root} is not the main checkout {main_root} (a worktree has a .git file)")
    if not cuda_ok():
        raise LaunchError("guard 6: torch.cuda.is_available() is False; the runs need the GPU")
    commit = git(["rev-parse", "HEAD"]).stdout.decode().strip()
    blob = git(["show", f"{commit}:{SELF_ANALYSIS}"]).stdout
    blob_id = git(["rev-parse", f"{commit}:{SELF_ANALYSIS}"]).stdout.decode().strip()
    return dict(A=A, design=design, expected=expected, commit=commit,
                blob_sha=hashlib.sha256(blob).hexdigest(), blob_id=blob_id)


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


def run_step(runner, cmd, log_path, root, header=None):
    """One subprocess with stdout+stderr appended, unchanged, to log_path. Returns the return code."""
    with open(log_path, "ab") as f:
        for line in header or ():
            write_line(f, line)
        return runner(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(root), env=child_env()).returncode


def evaluate(runner, py, root, checkpoint, train_list, out, log):
    with open(root / log, "xb"):
        pass
    if run_step(runner, eval_cmd(py, checkpoint, train_list, out), root / log, root) != 0:
        with open(root / log, "ab") as f:
            write_line(f, f"EVAL_FAILED {now_iso()}")
        raise LaunchError(f"evaluation failed: {log}")


def train_one(order, arm, seed, ctx, root, main_root, runner, revalidate_fn, snapshot):
    """Phase 1 for one run: train, re-validate and select (control), write the training record. No H9."""
    A, design, py = ctx["A"], ctx["design"], sys.executable
    p = run_paths(arm, seed)
    existing = [v for v in p.values() if (root / v).exists()]
    if existing:
        raise LaunchError(f"refusing to overwrite {existing}")
    looked = h9_outputs(root, A.SEEDS)
    if looked:
        raise LaunchError(f"phase 1: H9 outputs exist ({looked[:3]}); no training may follow an H9 evaluation")
    before = protected_now(main_root, ctx["expected"])
    if before != ctx["expected"]:
        raise LaunchError(f"protected hashes differ before {arm} seed {seed}")
    for d in ("logs/n5_atlantic/runs", f"models/n5_atlantic/{arm}", "results/n5_atlantic"):
        (root / d).mkdir(parents=True, exist_ok=True)
    log = root / p["train_log"]
    with open(log, "xb"):
        pass
    header = [f"RUN arm={arm} seed={seed} order={order} list={design['train_lists'][arm]} val={VAL_LIST} "
              f"epochs={design['epochs'][arm]}", f"START {now_iso()} | {snapshot()}"]
    rc = run_step(runner, train_cmd(py, arm, seed, design["train_lists"], design["epochs"]), log, root, header)
    with open(log, "ab") as f:
        write_line(f, f"TRAIN_FAILED rc={rc} {now_iso()}" if rc != 0 else f"END {now_iso()} | {snapshot()}")
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
        reval = revalidate_fn(root / p["epoch_dir"], A.CONTROL_EPOCHS, design["train_lists"][arm])
        logged = {e: va for e, _, va in ep}
        if sorted(reval) != list(range(1, A.CONTROL_EPOCHS + 1)) or \
                any(abs(round(v, 2) - logged[e]) > 0.0051 for e, v in reval.items()):
            raise LaunchError(f"control seed {seed}: re-validated values do not round to the logged ones")
        try:
            sel, tie = A.control_selection(reval, native_epoch)
        except A.DesignError as e:
            raise LaunchError(f"control seed {seed}: {e}") from None
        for e in range(2, A.CONTROL_EPOCHS + 1, 2):  # even epochs are not candidates; odd ones stay for the analysis
            (root / p["epoch_dir"] / f"epoch_{e:02d}.pt").unlink()
        kept += [f"{p['epoch_dir']}/epoch_{e:02d}.pt" for e in A.CONTROL_GRID]
        selected = p["native_checkpoint"] if tie else f"{p['epoch_dir']}/epoch_{sel:02d}.pt"
        rec.update(selected_epoch=sel, selected_checkpoint=selected, tie_rule=tie,
                   odd_grid_near_ties=sorted(e for e in A.CONTROL_GRID
                                             if e != sel and abs(reval[e] - reval[sel]) < A.TIE_KT),
                   val_mae_revalidated={str(e): reval[e] for e in sorted(reval)})
    else:
        rec.update(selected_epoch=native_epoch, selected_checkpoint=p["native_checkpoint"])
    rec["files_sha256"] = {f: sha256_file(root / f) for f in kept}
    rec["protected_before"] = before
    rec["protected_after"] = protected_now(main_root, ctx["expected"])
    if rec["protected_after"] != ctx["expected"]:
        changed = sorted(k for k, v in ctx["expected"].items() if rec["protected_after"][k] != v)
        raise LaunchError(f"protected hashes changed during {arm} seed {seed}: {changed}")
    write_json_new(root / p["record"], rec)
    print(f"order {order}: {arm} seed {seed} trained; selected epoch {rec['selected_epoch']} (native {native_epoch})"
          + (f"; tie rule fired {tie}" if rec.get("tie_rule") else ""), flush=True)
    return rec


def evaluate_one(order, arm, seed, ctx, root, main_root, runner):
    """Phase 2 for one run: H9 evaluation of the recorded selection (and the control's native), then the manifest."""
    A, design, py = ctx["A"], ctx["design"], sys.executable
    p = run_paths(arm, seed)
    man = verify_record(root, arm, seed, ctx["expected"])
    man.update(h9_csv=p["h9_csv"], eval_log=p["eval_log"])
    if arm == "control":
        man.update(native_h9_csv=p["native_h9_csv"], native_eval_log=p["native_eval_log"])
    evaluate(runner, py, root, man["selected_checkpoint"], man["train_list"], p["h9_csv"], p["eval_log"])
    if arm == "control":
        evaluate(runner, py, root, p["native_checkpoint"], man["train_list"], p["native_h9_csv"], p["native_eval_log"])
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


def train_phase(ctx, root, main_root, runner, revalidate_fn, snapshot):
    for order, arm, s in schedule(ctx["A"].SEEDS):
        if run_state(root, arm, s, ctx["expected"]) in ("trained", "eval_partial", "evaluated"):
            print(f"order {order}: {arm} seed {s} already trained; skipped", flush=True)
            continue
        train_one(order, arm, s, ctx, root, main_root, runner, revalidate_fn, snapshot)


def evaluate_phase(ctx, root, main_root, runner):
    """Condition (b): no H9 evaluation runs until every one of the 20 training runs has a verified record."""
    order = schedule(ctx["A"].SEEDS)
    states = {(arm, s): run_state(root, arm, s, ctx["expected"]) for _, arm, s in order}
    unfinished = [(k, arm, s) for k, arm, s in order if states[(arm, s)] not in ("trained", "evaluated")]
    if unfinished:
        raise LaunchError(f"phase 2: H9 evaluation refuses to start until all {len(order)} training runs have "
                          f"finished; unfinished: {unfinished}")
    half = [(arm, s) for (arm, s), st in states.items() if st == "eval_partial"]
    if half:
        raise LaunchError(f"phase 2: H9 outputs without a manifest for {half}; inspect them before evaluating again")
    mans = []
    for k, arm, s in order:
        if states[(arm, s)] == "evaluated":
            ctx["A"].load_run(root / run_paths(arm, s)["manifest"], arm, s, ctx["design"], root)
            mans.append(json.loads((root / run_paths(arm, s)["manifest"]).read_text(encoding="utf-8")))
            print(f"order {k}: {arm} seed {s} already evaluated; skipped", flush=True)
        else:
            mans.append(evaluate_one(k, arm, s, ctx, root, main_root, runner))
    return mans


def launch(root=ROOT, *, git=None, main_root=None, check=None, runner=subprocess.run, revalidate_fn=revalidate,
           snapshot=nvidia_snapshot, cuda_ok=cuda_available):
    """Phase 1 (all 20 trainings), then phase 2 (H9). The keyword hooks exist for the synthetic tests only; main()
    exposes none."""
    root = Path(root)
    git = git or real_git(root)
    check = check or load_module("check", root / "scripts" / "check.py")
    main_root = Path(main_root) if main_root else check.find_main_root(None)
    ctx = preflight(root, git, main_root, check, cuda_ok)
    print(f"commit {ctx['commit']}; analysis blob sha256 {ctx['blob_sha']} (id {ctx['blob_id']})", flush=True)
    train_phase(ctx, root, main_root, runner, revalidate_fn, snapshot)
    return evaluate_phase(ctx, root, main_root, runner)


def abort_run(arm, seed, reason, root=ROOT):
    """Move an unfinished run's files aside, unchanged and hashed, so launch restarts it in its order position."""
    root = Path(root)
    A = analysis(root)
    order = {(a, s): k for k, a, s in schedule(A.SEEDS)}
    if (arm, seed) not in order or not reason:
        raise LaunchError(f"abort-run: unknown run {arm} seed {seed}, or no --reason")
    looked = h9_outputs(root, A.SEEDS)
    if looked:
        raise LaunchError(f"abort-run: H9 outputs exist ({looked[:3]}); no run may be aborted after an H9 evaluation")
    p = run_paths(arm, seed)
    if (root / p["record"]).exists():
        raise LaunchError(f"abort-run: {arm} seed {seed} has a training record; finished runs are not aborted")
    present = [p[k] for k in TRAIN_KEYS if k in p and (root / p[k]).exists()]
    if not present:
        raise LaunchError(f"abort-run: {arm} seed {seed} has no files to move")
    attempt = len(list(root.glob(f"{ABORTED}/{arm}_seed{seed}_attempt*"))) + 1
    dest_rel = f"{ABORTED}/{arm}_seed{seed}_attempt{attempt}"
    check_output_paths([dest_rel + "/"])
    (root / dest_rel).mkdir(parents=True)
    moved = {}
    for rel in present:
        src = root / rel
        files = [src] if src.is_file() else sorted(f for f in src.rglob("*") if f.is_file())
        for f in files:
            r = f.relative_to(root).as_posix()
            moved[r] = {"to": f"{dest_rel}/{r}", "sha256": sha256_file(f)}
        (root / dest_rel / rel).parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, root / dest_rel / rel)
    write_json_new(root / dest_rel / "abort.json", {"arm": arm, "seed": seed, "order": order[(arm, seed)],
                                                    "attempt": attempt, "reason": reason, "time": now_iso(),
                                                    "moved": moved})
    print(f"aborted {arm} seed {seed} (order {order[(arm, seed)]}, attempt {attempt}): {len(moved)} files -> {dest_rel}")
    return dest_rel


def plan(root=ROOT):
    A = analysis(root)
    epochs = {"control": A.CONTROL_EPOCHS,
              "atlantic": A.epochs_for(A.ARM_TRAIN_FRAMES)[0] if A.ARM_TRAIN_FRAMES else "<epochs_for(ARM_TRAIN_FRAMES)>"}
    py = "PY"
    print("PHASE 1: the 20 trainings; no H9 evaluation")
    for order, arm, s in schedule(A.SEEDS):
        p = run_paths(arm, s)
        print(f"order {order}: {arm} seed {s}")
        print("  train:", train_cmd(py, arm, s, A.TRAIN_LISTS, epochs), "->", p["train_log"])
        if arm == "control":
            print("  re-validate all epochs on", VAL_LIST, "only; analyze_n5.control_selection: argmin must be the last "
                  f"'saved' (tie window {A.TIE_KT} kt: train.py's checkpoint), else the first minimum over",
                  list(A.CONTROL_GRID))
        print("  record:", p["record"])
    print("PHASE 2: only after all 20 training records verify")
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
    """Specification only (PR A). Nothing here downloads, builds or scans."""
    print("BUILD (design §12.3; specification only, not run by this script):")
    print("  hash data/interim/hurdat2_atlantic.txt before; it must be ee35819c...0bcb5c")
    print("  ", ["PY", "scripts/build_batch.py", "--ids-file", BUILD_IDS, "--sample-interval", "3h"])
    print("   GOES-16 default; never --force-hurdat2-refresh or --hurdat2-url")
    print("  hash data/interim/hurdat2_atlantic.txt after; it must be unchanged")
    print("EXCLUSIONS (design §12.4; specification only): ['PY', 'check_nan.py'] (the merged safe version)")
    print("  rescans all non-spent storms; the diff of excluded_crops.txt must be additions under the 31 new IDs only")
    print("FRAME LIST (design §12.5; PR B): drop the Marco AL142020 frames concurrent with a protected storm")
    print("  (scripts/concurrency.py, 650 km box) from the built manifests; write", FRAME_LIST)
    print("  then fill ARM_TRAIN_FRAMES and FRAME_LIST_SHA256 in", SELF_ANALYSIS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["plan", "build-spec", "launch", "abort-run"])
    ap.add_argument("arm", nargs="?", choices=["control", "atlantic"], help="abort-run only")
    ap.add_argument("seed", nargs="?", type=int, help="abort-run only")
    ap.add_argument("--reason", help="abort-run only: why the run is aborted (recorded)")
    args = ap.parse_args(argv)
    given = (args.arm is not None, args.seed is not None, bool(args.reason))
    if given != ((args.stage == "abort-run"),) * 3:
        ap.error("abort-run takes ARM SEED --reason TEXT; the other stages take none of them")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    if args.stage == "plan":
        plan()
    elif args.stage == "build-spec":
        build_spec()
    elif args.stage == "abort-run":
        abort_run(args.arm, args.seed, args.reason)
    else:
        launch()


if __name__ == "__main__":
    sys.exit(main())
