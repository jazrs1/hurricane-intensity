"""Deliberate-breakage proof for scripts/run_f31.py: every guard, disabled on its own, fails at least one test.

    .venv\\Scripts\\python.exe -I -B reports/v2_survey/f31/run_f31_breakage.py [--jobs 8]

For each row below, a copy of run_f31.py with that one guard disabled (one exact text replacement, which must occur
exactly once) is written to a temporary folder, beside a copy of run_n5.py (the driver imports its helpers from its
own folder). tests/test_run_f31.py is then run against the copy (HC_RUN_F31_UNDER_TEST), restricted with -k to the
row's named tests. A row passes when pytest reports at least one failed test and no collection error: the guard is
load-bearing. A row whose tests still pass would mean the guard is untested. First, the same -k selections are run on
the unmodified driver and must all pass, so a failure is caused by the breakage and nothing else.

Writes reports/v2_survey/f31/run_f31_breakage.txt (the table). Synthetic only: no training, evaluation or H9 read.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "scripts" / "run_f31.py"
TESTS = "tests/test_run_f31.py"

# (guard, exact text in run_f31.py, replacement, pytest -k expression)
ROWS = [
    ("1: tracked tree clean", 'if git(["status", "--porcelain", "--untracked-files=no"]).stdout.strip():', "if False:",
     "uncommitted_or_unpinned and dirty"),
    ("1: analysis/N5/run_n5/train.py equal HEAD", 'if git(["diff", "--quiet", "HEAD", "--", rel], check=False).returncode != 0:',
     "if False:", "uncommitted_or_unpinned and (analysis or analyze_n5 or run_n5 or train_py)"),
    ("1: run_n5.py is the pinned blob",
     'if hashlib.sha256(git(["show", f"HEAD:{RUN_N5}"]).stdout).hexdigest() != RUN_N5_BLOB_SHA256:', "if False:",
     "run_n5_blob"),
    ("1: analyze_n5.py is analyze_f31's pin",
     'if hashlib.sha256(git(["show", f"HEAD:{ANALYZE_N5}"]).stdout).hexdigest() != A.N5_BLOB_SHA256:', "if False:",
     "analyze_n5_blob"),
    ("2: train.py is the pinned blob (John 23:50)",
     'if hashlib.sha256(git(["show", f"HEAD:{TRAIN_PY}"]).stdout).hexdigest() != A.TRAIN_PY_SHA256:', "if False:",
     "train_py_blob"),
    ("2: frame list SHA-256",
     'if not frame_list.is_file() or sha256_file(frame_list) != design["frame_list_sha"]["atlantic"]:', "if False:",
     "frame_list"),
    ("3: PROTECTED == COORDINATOR.md block", 'if expected != design["protected"]:', "if False:", "protected_copy"),
    ("3: protected files have their hashes", "    if actual != expected:\n", "    if False:\n", "changed_protected_file"),
    ("6: main checkout, not a worktree",
     'if Path(root).resolve() != Path(main_root).resolve() or not (Path(root) / ".git").is_dir():', "if False:",
     "worktree"),
    ("6: child imports ROOT/src",
     'if not pkg or not Path(pkg).resolve().is_relative_to((Path(root) / "src").resolve()):', "if False:",
     "child_that_imports"),
    ("6: CUDA available", "    if not cuda_ok():\n", "    if False:\n", "without_cuda"),
    ("4: no pending abort (both phases)", "    pending = pending_aborts(root)\n    if pending:\n",
     "    pending = pending_aborts(root)\n    if False:\n", "interrupted_abort or pending_eval_abort"),
    ("4: outputs confined to the F-31 roots",
     "bad = [p for p in paths if not p.startswith(OUTPUT_ROOTS) or any(p.startswith(n) for n in NEVER_WRITTEN)]",
     "bad = []", "output_paths_are_confined"),
    ("4: no launch after the phase-2 marker", "    if (Path(root) / PHASE2_MARKER).exists():\n        raise LaunchError(f\"guard 4",
     "    if False:\n        raise LaunchError(f\"guard 4", "bytes_back or cannot_launder"),
    ("4: no launch over a bad record", '    if bad:\n        raise LaunchError(f"guard 4: training records',
     '    if False:\n        raise LaunchError(f"guard 4: training records', "fails_verification"),
    ("4: no launch over an unfinished run", "    if partial:\n", "    if False:\n",
     "existing_output or restart_keeps_seed"),
    ("4: records/manifests at HEAD's commit", 'if commit != ctx["commit"]:', "if False:", "relaunch_at_another"),
    ("4: no training after a look", 'if seen and any(st == "fresh" for st in states.values()):', "if False:",
     "never_follows"),
    ("5: fetch succeeds", 'if git(["fetch", "origin"], check=False).returncode != 0:', "if False:", "fetch_fails"),
    ("5: first-parent chain", "if chain.returncode != 0 or commit not in chain.stdout.decode().split():", "if False:",
     "not_first_parent or launch_not_first_parent"),
    ("phase 1: refuse existing outputs", "    if existing:\n", "    if False:\n", "appears_mid_launch"),
    ("phase 1: no training after a look (per run)",
     '    seen = looked(root, A)\n    if seen:\n        raise LaunchError(f"phase 1',
     '    seen = looked(root, A)\n    if False:\n        raise LaunchError(f"phase 1', "planted_mid_phase1"),
    ("phase 1: train.py failure stops", "    if rc != 0:\n        raise LaunchError(f\"training failed",
     "    if False:\n        raise LaunchError(f\"training failed", "training_failure"),
    ("phase 1: epoch dir inside ROOT", 'if epoch_dir.resolve() != root.resolve() / p["epoch_dir"]:', "if False:",
     "resolving_outside"),
    ("phase 1: every epoch file present", "    if missing:\n", "    if False:\n", "missing_epoch_file"),
    ("phase 1: re-validation rounds to the log",
     "if sorted(reval) != list(range(1, e_max + 1)) or any(abs(round(v, 2) - logged[e]) > 0.0051",
     "if False and any(abs(round(v, 2) - logged[e]) > 0.0051", "does_not_round"),
    ("phase 1: seed<s>.pt == native 4x epoch", 'if not hooks["same_weights_fn"](root / p["native_checkpoint"], root / native):',
     "if False:", "native_checkpoint_must_hold"),
    ("phase 1: protected unchanged during a run", '    if rec["protected_after"] != ctx["expected"]:',
     "    if False:", "changed_during_revalidation"),
    ("phase 1: protected unchanged during the training step",
     "    changed = protected_changed(main_root, ctx)  # after the training step\n    if changed:",
     "    changed = protected_changed(main_root, ctx)  # after the training step\n    if False:", "changed_during_a_run"),
    ("phase 1: protected unchanged before a run", '    if before != ctx["expected"]:\n        raise LaunchError(f"protected hashes differ before {arm}',
     '    if False:\n        raise LaunchError(f"protected hashes differ before {arm}', "drift_before_a_run"),
    ("phase 2: protected unchanged before each H9 step",
     "        changed = protected_changed(main_root, ctx)  # before each H9 step\n        if changed:",
     "        changed = protected_changed(main_root, ctx)  # before each H9 step\n        if False:",
     "drift_before_an_h9_step"),
    ("phase 2: protected unchanged during each H9 step",
     "        changed = protected_changed(main_root, ctx)  # after it\n        if changed:",
     "        changed = protected_changed(main_root, ctx)  # after it\n        if False:", "during_an_h9_step"),
    ("import: run_n5.py hashed before it is executed", "    if got != RUN_N5_BLOB_SHA256:", "    if False:",
     "tampered_run_n5"),
    ("phase 1: kept files read-only", "        os.chmod(root / f, stat.S_IREAD)", "        pass",
     "fails_verification"),
    ("phase 1: deleted hashes never in files_sha256", '"files_sha256": {f: sha256_file(root / f) for f in kept},',
     '"files_sha256": {f: sha256_file(root / f) for f in kept + [p["epoch_dir"] + "/gone"]},', "end_to_end"),
    ("phase 1: non-slot epoch files deleted", "        (root / files[e]).unlink()", "        pass", "end_to_end"),
    ("5e: records tracked", 'if git(["ls-files", "--error-unmatch", rel], check=False).returncode != 0:', "if False:",
     "committed_records_as_evidence and untracked"),
    ("5e: records equal HEAD's blob",
     'if git(["show", f"HEAD:{rel}"], check=False).stdout != (Path(root) / rel).read_bytes().replace(b"\\r\\n", b"\\n"):',
     "if False:", "committed_records_as_evidence and differs"),
    ("5e: launch commit on the first-parent chain", "    first_parent(git, launch_commit)\n", "\n",
     "launch_not_first_parent"),
    ("5e: HEAD pushed", 'if git(["merge-base", "--is-ancestor", "HEAD", UPSTREAM], check=False).returncode != 0:',
     "if False:", "not_pushed"),
    ("5e: L..HEAD only records, docs, deviations", "    if other:\n", "    if False:\n", "code_changed"),
    ("rule (A): only after a look", "    if not looked(root, A):", "    if False:", "refused_before_any_look"),
    ("rule (A): the fixed path", "    if Path(path).as_posix() != A.DEVIATIONS_PATH:", "    if False:", "other_path"),
    ("rule (A): identical to HEAD's blob", '    if git(["show", f"HEAD:{A.DEVIATIONS_PATH}"], check=False).stdout != \\',
     "    if False and None != \\", "deviation_rule_a_does_not_cover and differs"),
    ("rule (A): analyze_f31.load_deviations",
     '        dropped, _ = A.load_deviations(Path(root) / A.DEVIATIONS_PATH, root, Path(root) / "logs/f31/runs", git, design)',
     '        dropped = {(r["arm"], r["seed"]): "" for r in json.loads((Path(root) / A.DEVIATIONS_PATH).read_text())["dropped_runs"]}',
     "deviation_rule_a_does_not_cover and (healthy or restorable or untracked)"),
    ("phase 2: no evaluation over a bad record", '    if bad:\n        raise LaunchError(f"phase 2: training records',
     '    if False:\n        raise LaunchError(f"phase 2: training records', "fails_verification or bytes_back"),
    ("phase 2: all 40 trained first", "    if unfinished:\n", "    if False:\n", "while_any_training_run_is_unfinished"),
    ("phase 2: no half-evaluated run", "    if half:\n", "    if False:\n", "crashed_evaluation"),
    ("phase 2 entry: verify_self at L", "        A.verify_self(launch_commit, root, git)\n", "\n",
     "verify_self_at_the_launch"),
    ("phase 2 entry: check_training on all 40",
     "                A.check_training(rec, arm, s, ctx[\"design\"], root, run_paths(A, arm, s)[\"record\"])",
     "                pass", "training_side_checks_before_any_h9"),
    ("phase 2 entry: the phase-2 marker", '        write_json_new(root / PHASE2_MARKER, {"head"',
     '        dict(x={"head"', "cannot_launder or marker_alone"),
    ("phase 2: an evaluated run must pass load_run",
     '                A.load_run(root / p["manifest"], arm, s, ctx["design"], root)\n            except',
     '                pass\n            except', "takes_a_manifest_that_load_run_refuses"),
    ("evaluate_one: eval attempts complete", '        A.check_attempts({"eval_attempts": attempts}, root, p["record"])',
     "        pass", "checks_its_eval_attempts"),
    ("evaluate_one: the epoch file, never seed<s>.pt",
     'evaluate_step(runner, py, root, ep["checkpoint"], rec["train_list"], ep["csv"], ep["eval_log"])',
     'evaluate_step(runner, py, root, p["native_checkpoint"], rec["train_list"], ep["csv"], ep["eval_log"])',
     "end_to_end"),
    ("evaluate_one: manifest = record + 3 keys",
     'man = dict(rec, evals=evals, eval_attempts=attempts, protected_after_eval=protected_now(main_root, ctx["expected"]))',
     'man = dict(rec, evals=evals, eval_attempts=attempts, protected_after_eval=protected_now(main_root, ctx["expected"]), h9="x")',
     "end_to_end"),
    ("evaluate_one: load_run after the evaluation",
     '        A.load_run(root / p["manifest"], arm, seed, design, root)  # the analysis',
     '        pass  # the analysis', "manifest_the_analysis_refuses"),
    ("lock: exclusive create", "os.O_CREAT | os.O_EXCL | os.O_WRONLY", "os.O_CREAT | os.O_WRONLY", "held_lock"),
    ("abort-run: refused after a look", '            raise LaunchError(f"abort-run: H9 has been looked at',
     '            print(f"abort-run: H9 has been looked at', "abort_run_refusals or cannot_launder"),
    ("abort-run: refused for a verified record",
     'if run_state(root, A, arm, seed, A.N5.PROTECTED) in ("trained", "eval_partial", "evaluated"):', "if False:",
     "abort_run_refusals"),
    ("abort-all: refused after a look", '            raise LaunchError(f"abort-all: H9 has been looked at',
     '            print(f"abort-all: H9 has been looked at', "abort_all_refuses or marker_alone"),
    ("abort-eval: only a half-evaluated or refused run",
     'if state not in ("eval_partial", "evaluated, refused by load_run") and not (pending and state == "trained"):',
     "if False:", "crashed_evaluation or takes_a_manifest"),
    ("abort: abort.json pending before any move", '    write_json_replace(root / dest_rel / "abort.json", info)\n    for rel in present:',
     "    for rel in present:", "interrupted_abort"),
    ("verify_record: hashed files unchanged", "    if changed or not rec.get(\"files_sha256\"):",
     "    if False:", "fails_verification"),
]


def run_row(i, row, base, jobs_tmp):
    name, old, new, k = row
    src = DRIVER.read_text(encoding="utf-8")
    if row is not None and old is not None:
        n = src.count(old)
        if n != 1:
            return (name, k, f"BAD ROW: the text occurs {n} times", False, "")
        src = src.replace(old, new)
    d = Path(jobs_tmp) / f"row{i:02d}"
    d.mkdir(parents=True)
    (d / "run_f31.py").write_text(src, encoding="utf-8", newline="\n")
    shutil.copy(ROOT / "scripts" / "run_n5.py", d / "run_n5.py")
    env = dict(os.environ, HC_RUN_F31_UNDER_TEST=str(d / "run_f31.py"))
    r = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", TESTS, "-k", k,
                        "--basetemp", str(d / "tmp")], cwd=str(ROOT), env=env, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    failed = re.search(r"(\d+) failed", last)
    errors = "error" in last
    if old is None:
        ok = r.returncode == 0 and "passed" in last
    else:
        ok = r.returncode == 1 and failed is not None and not errors
    first = next((ln.split(" - ")[0].replace("FAILED tests/test_run_f31.py::", "")
                  for ln in r.stdout.splitlines() if ln.startswith("FAILED")), "")
    return (name, k, last, ok, first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="hc_f31_breakage_") as tmp:
        with ThreadPoolExecutor(args.jobs) as ex:
            controls = list(ex.map(lambda ir: run_row(1000 + ir[0], (ir[1][0], None, None, ir[1][3]), None, tmp),
                                   enumerate(ROWS)))
            broken = list(ex.map(lambda ir: run_row(ir[0], ir[1], None, tmp), enumerate(ROWS)))
    lines = ["run_f31.py deliberate-breakage table (reports/v2_survey/f31/run_f31_breakage.py). Each guard disabled on",
             "its own; the row's tests must fail on the broken copy and pass on the unmodified driver.", "",
             "| # | guard | -k | unmodified | broken | first failing test |", "|---|---|---|---|---|---|"]
    bad = 0
    for i, (c, b) in enumerate(zip(controls, broken), 1):
        cs = "pass" if c[3] else f"NOT PASS ({c[2]})"
        bs = f"fails ({b[2]})" if b[3] else f"NOT CAUGHT ({b[2]})"
        bad += (not c[3]) + (not b[3])
        lines.append(f"| {i} | {b[0]} | `{b[1]}` | {cs} | {bs} | {b[4] if len(b) > 4 else ''} |")
    lines += ["", f"{len(ROWS)} guards; {len(ROWS) - bad if bad <= len(ROWS) else 0} rows OK; problems: {bad}"]
    out = Path(__file__).with_name("run_f31_breakage.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
