"""F-31 budget study: pre-registered analysis. DRAFT (launch PR A); not yet approved.

Design: reports/v2_survey/f31/design.md (approved by John, 2026-10-19 23:50). Read-only, CPU only, deterministic:
no torch and no random numbers. Its launch hash is the SHA-256 of the committed blob, from the bytes of
`git show <commit>:scripts/analyze_f31.py` (verify_self), as for N5.

What it reads: the 40 per-run manifests that scripts/run_f31.py writes, logs/f31/runs/<arm>_seed<s>.json
(seeds 10-29; schema in check_training and load_run). Each run trained once to 4x with --epoch-dir. The driver
re-validated every epoch on validation only, fixed the H9 slots, kept only the slot epochs' files and evaluated each
distinct slot epoch on H9 once. Every link is re-checked here, and every selection is recomputed from the recorded
validation values. Also read, both pinned:
  - results/n5_atlantic/analysis/report.txt (SHA-256 N5_REPORT_SHA256): N5's verdict, quoted verbatim beside S1/S2;
  - results/ensemble/expanded10_h9_metrics.csv: the frozen ensemble's committed H9 figures (FROZEN_ENSEMBLE),
    quoted, unpaired.

Reused from scripts/analyze_n5.py, whose blob is pinned (N5_BLOB_SHA256), pure functions and constants only:
epochs_for, signflip_p and signflip_ci (n always passed explicitly), sign_matrix (rule (A)'s power only), load_csv,
read_text_strict, check_eval_log, metrics, per_storm, the trivial helpers sha256_file and ids_in, DesignError, and the design constants (H9 and its pin,
the training and validation IDs, frame counts, the frame-list SHA-256, TIE_KT, PROTECTED). Rewritten here, not
imported: verify_self, the deviations rule, the train-log parser, pinned paths and everything bound to N5's
SELF_PATH, SEEDS or DEVIATIONS_PATH. analyze_n5.py itself is never modified.

    python scripts/analyze_f31.py --manifest-dir logs/f31/runs --out-dir results/f31/analysis
        [--deviations logs/f31/deviations.json]

The record/manifest contract run_f31.py must write (phase-1 record <arm>_seed<s>.train.json; the manifest
<arm>_seed<s>.json adds "evals"):
  arm, seed, order, commit, train_list, frame_list_sha256 (null for the control), train_py_sha256 (TRAIN_PY_SHA256),
  native_same_weights (true: seed<s>.pt holds the native 4x epoch's tensors), train_log, native_checkpoint, epoch_dir
  (pinned_paths), val_mae_revalidated {epoch: unrounded val MAE, 1..E_4x}, slots {slot: epoch} (SLOTS), tie_rule
  {"1x"|"2x"|"4x": null or {revalidated_argmin, saved_epoch, delta_kt}}, near_ties {"matched_4x"|"smoothed_4x":
  [epochs]}, files_sha256 {path: sha} (exactly the train log, seed<s>.pt and the slot epochs' files),
  deleted_epoch_sha256 {epoch: sha} (every other epoch; those files are gone), protected_before, protected_after,
  aborted_attempts, eval_attempts, and in the manifest evals {epoch: {csv_sha256, eval_log_sha256}} for exactly the
  distinct slot epochs (paths: epoch_paths) and protected_after_eval (the protected hashes after the run's H9
  evaluations, as run_n5.evaluate_one writes; must equal PROTECTED).
  Spelled out (hc-adversary item 23c):
  - run_f31.py evaluate applies analyze_f31.check_training to all 40 phase-1 records before the first H9 evaluation,
    so no training-side refusal can first appear after the look (N5 item 16).
  - Every H9 evaluation, the native 4x slot included, evaluates the epoch file epoch_paths(...)["checkpoint"], never
    seed<s>.pt. seed<s>.pt is kept only for native_same_weights.
  - near_ties has exactly the keys {"matched_4x", "smoothed_4x"} for the control and {"smoothed_4x"} for the atlantic
    arm; tie_rule exactly {"1x", "2x", "4x"}.
  - order is the run's position in the schedule (expected_order: round-robin by seed, arm order alternating C/A, A/C,
    from 1 to 40); it is checked and reported.
  - Each train log has an "END " line and no TRAIN_FAILED; each attempt JSON named in aborted_attempts or
    eval_attempts exists with "status": "complete".
  - The manifest equals its committed phase-1 record on every key except evals, eval_attempts and
    protected_after_eval (phase 2 adds those), and the commit's scripts/train.py (git show) hashes to TRAIN_PY_SHA256.

--deviations (the dead-end rule (A), carried over from N5, design.md §8) is the committed, clean file
logs/f31/deviations.json, {"dropped_runs": [{"arm": <arm>, "seed": <s>, "reason": "<why>"}, ...]}. It may name only a
run whose training record no longer verifies. That run leaves P1/P2 (within-run), and its seed pair leaves S1/S2. The
report labels every verdict NON-CONFIRMATORY, gives the file's blob SHA-256, states each test's reduced n and its
effect on power (design §5's n = 20 values, quoted, against power.py's method at the reduced n), and prints the
pre-look BUDGET-LIMITED counts from all 40 records beside the adjusted ones.

Writes into --out-dir (a new directory under results/f31/; refuses to overwrite): primary_tests.csv,
sensitivity.csv, secondary.csv, selection.csv, per_run.csv, budget_curve.csv, ensemble.csv, per_storm.csv,
val_trace.csv, report.txt. All LF.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N5_PATH = "scripts/analyze_n5.py"


def _load_n5():
    path = Path(__file__).resolve().parent / "analyze_n5.py"
    spec = importlib.util.spec_from_file_location("analyze_n5_for_f31", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module level is constants and functions; main() is behind __main__
    return mod


N5 = _load_n5()
DesignError = N5.DesignError

# ---- fixed before launch -----------------------------------------------------------------------------------
N5_BLOB_SHA256 = "0130641583cf09c8abb5791b9ea9467fd6410b953bc8a57c3a2ae68fb3bb1288"  # N5's launch blob (679e2169)
TRAIN_PY_SHA256 = "afceb4b19ea7d8c732c6f26b7d52e4b3b6d313d58153e2d0518887a3a6db231f"  # scripts/train.py at c4c4cff
SEEDS = tuple(range(10, 30))  # design §5: 20 per arm, new seeds (N5 used 0-9)
ARMS = ("control", "atlantic")
BUDGETS = (1, 2, 4)  # x 1640 steps, nested in one run (design §2)
FRAMES = {"control": N5.CONTROL_TRAIN_FRAMES, "atlantic": N5.ARM_TRAIN_FRAMES}  # 2081, 4044
EXPECTED_EPOCHS = {"control": {1: 25, 2: 50, 4: 99}, "atlantic": {1: 13, 2: 26, 4: 52}}  # design §2 table
SLOTS = {"control": ("native_1x", "native_2x", "native_4x", "matched_4x", "smoothed_4x"),
         "atlantic": ("native_1x", "native_2x", "native_4x", "smoothed_4x")}  # design §3, r2
SMOOTH_HALF = 2  # smoothed rule: median of validation MAE over epochs [e-2, e+2] within the budget
ALPHA = 0.025  # P1, P2 and their sensitivities: Bonferroni 0.05 / 2, two-sided exact sign-flip, 97.5% CIs
SECONDARY_ALPHA = 0.05  # S1-S4, uncorrected, labelled secondary
SESOI_MAE = 0.9  # N5's SESOI, for comparability only (design §5)
SESOI_BIAS = 3.0  # N5's, for S2
UNDERTRAIN_FRACTION = N5.UNDERTRAIN_FRACTION  # 0.85: flagged when the selected epoch > 0.85 x E_k
CI_GRID = 4001  # signflip_ci grid, as N5
N5_REPORT = "results/n5_atlantic/analysis/report.txt"
N5_REPORT_SHA256 = "fde430de92bab775977dfc84bef9eeb7a46907ae7abfb06cb2907db2d1722bb1"  # stored -text: exact bytes
FROZEN_METRICS = "results/ensemble/expanded10_h9_metrics.csv"
FROZEN_ENSEMBLE = {"overall MAE": 8.570941059931508, ">96 MAE": 10.815341209800918, ">96 bias": -7.715010003062787}
DEVIATIONS_PATH = "logs/f31/deviations.json"
TRAIN_PY = "scripts/train.py"
# Rule (A)'s power statement (John, 2026-10-19 17:03: "state the reduced n and its effect on power in the report").
# design §5's tables at n = 20 and the SESOI, quoted, not re-simulated: {sd_diff: power}
POWER_N20 = {"P1/P2": dict(alpha=0.025, sesoi=0.9, cells={0.54: 1.00, 0.85: 0.99, 1.12: 0.87}),
             "S1": dict(alpha=0.05, sesoi=0.9, cells={0.85: 0.99, 1.12: 0.92, 1.40: 0.76}),
             "S2": dict(alpha=0.05, sesoi=3.0, cells={2.84: 0.99, 4.79: 0.76, 6.00: 0.57})}
POWER_SEED = 20261019  # power.py's seed; one fresh generator per cell, as N5's rule-(A) power_at
POWER_REPS = 2000  # power.py's reps per cell: SE <= sqrt(0.25 / 2000) = 0.011
POWER_CHUNK = 16  # reps per sign-matrix product: 2^19 x 16 doubles = 67 MB at n = 19 (N5's unchunked form: 84 GB)
SELF_PATH = "scripts/analyze_f31.py"
# ------------------------------------------------------------------------------------------------------------

OUTPUTS = ["primary_tests.csv", "sensitivity.csv", "secondary.csv", "selection.csv", "per_run.csv",
           "budget_curve.csv", "ensemble.csv", "per_storm.csv", "val_trace.csv", "report.txt"]
HEX64 = re.compile(r"[0-9a-f]{64}")


def epochs_table():
    """E_k per arm and budget by N5's budget rule at k x 1640 steps; must equal the design's table."""
    e = {arm: {k: N5.epochs_for(FRAMES[arm], k * N5.TARGET_STEPS)[0] for k in BUDGETS} for arm in ARMS}
    if e != EXPECTED_EPOCHS:
        raise DesignError(f"budget rule gives {e}, not the design's {EXPECTED_EPOCHS}")
    return e


EPOCHS = epochs_table()
BATCHES = {arm: N5.epochs_for(FRAMES[arm])[1] for arm in ARMS}  # 66, 127


def design_constants(**override):
    """The recorded design as one dict. run() takes overrides from the tests only; main() exposes none."""
    d = dict(h9=N5.H9, h9_frames=N5.H9_FRAMES, h9_pin=N5.H9_PIN, val_ids=N5.VAL_IDS, val_frames=N5.VAL_FRAMES,
             ids={"control": N5.CONTROL_IDS, "atlantic": N5.ARM_IDS}, train_lists=N5.TRAIN_LISTS, frames=FRAMES,
             frame_list_sha={"control": None, "atlantic": N5.FRAME_LIST_SHA256}, protected=N5.PROTECTED,
             train_py_sha=TRAIN_PY_SHA256, n5_report_sha=N5_REPORT_SHA256, frozen=FROZEN_ENSEMBLE, ci_grid=CI_GRID,
             power_reps=POWER_REPS)
    d.update(override)
    return d


# ---- selection (validation only) ---------------------------------------------------------------------------
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
    if not abs(delta) < N5.TIE_KT:
        raise DesignError(f"re-validated minimum over epochs 1-{e_k} is epoch {argmin}, not train.py's pick {pick}: "
                          "the re-validation path does not reproduce training")
    return pick, {"revalidated_argmin": argmin, "saved_epoch": pick, "delta_kt": delta}


def matched_pick(reval, e_k):
    """(epoch, near ties): the first minimum over the odd epochs <= E_k (N5's matched grid); near ties are odd epochs
    within TIE_KT of it. They are recorded; the first minimum stands (John's decision 3)."""
    grid = range(1, e_k + 1, 2)
    pick = min(grid, key=lambda e: (reval[e], e))
    return pick, sorted(e for e in grid if e != pick and abs(reval[e] - reval[pick]) < N5.TIE_KT)


def smoothed_values(reval, e_k):
    return {e: float(np.median([reval[j] for j in range(max(1, e - SMOOTH_HALF), min(e_k, e + SMOOTH_HALF) + 1)]))
            for e in range(1, e_k + 1)}


def smoothed_pick(reval, e_k):
    """(epoch, near ties): the first minimum over e <= E_k of the median validation MAE over [e-2, e+2] within the
    budget (design §3). H9-motivated (N5's seed-4 swing), but selected on validation only; a sensitivity."""
    m = smoothed_values(reval, e_k)
    pick = min(m, key=lambda e: (m[e], e))
    return pick, sorted(e for e in m if e != pick and abs(m[e] - m[pick]) < N5.TIE_KT)


def selections(arm, reval, saved):
    """Every slot of one run, recomputed from validation values only: ({slot: epoch}, {budget: tie}, {slot: ties})."""
    e = EPOCHS[arm]
    slots, ties, near = {}, {}, {}
    for k in BUDGETS:
        slots[f"native_{k}x"], ties[f"{k}x"] = native_pick(reval, saved, e[k])
    if arm == "control":
        slots["matched_4x"], near["matched_4x"] = matched_pick(reval, e[4])
    slots["smoothed_4x"], near["smoothed_4x"] = smoothed_pick(reval, e[4])
    return slots, ties, near


def flagged(epoch, e_k):
    """Under-training flag (N5's rule): the selected epoch lies in the last 15% of the budget."""
    return epoch > UNDERTRAIN_FRACTION * e_k


def val_stability(reval, epoch, e_k):
    """Design §3's validation-only stability figures for one pick: val MAE at the pick, the min/max over its +-2-epoch
    neighbours within the budget, and the sd of val MAE over the budget's last 15% (the flagged epochs)."""
    nb = [reval[j] for j in range(max(1, epoch - 2), min(e_k, epoch + 2) + 1) if j != epoch]
    tail = [reval[j] for j in range(1, e_k + 1) if flagged(j, e_k)]
    return {"val_at_pick": reval[epoch], "val_neighbours_min": min(nb) if nb else np.nan,
            "val_neighbours_max": max(nb) if nb else np.nan,
            "val_sd_last15": float(np.std(tail, ddof=1)) if len(tail) > 1 else np.nan}


# ---- statistics --------------------------------------------------------------------------------------------
def p_floor(d):
    """The smallest two-sided p the exact sign-flip test can give: 2^(1-m) with m non-zero differences."""
    m = int(np.count_nonzero(np.asarray(d, float)))
    return (2.0 ** (1 - m) if m else 1.0), m


def paired_test(name, d, alpha, better_sign, sesoi, branch, ci_grid, extra=None):
    """One paired test over the per-seed differences d. The sign-flip p over all 2^n assignments equals the p over the
    m non-zero seeds (zeros add nothing to any sign assignment). The CI inverts the test over all n differences."""
    d = np.asarray(d, float)
    n = len(d)
    p = N5.signflip_p(d, n=n)
    lo, hi = N5.signflip_ci(d, alpha, n_grid=ci_grid, n=n)
    floor, m = p_floor(d)
    level = f"{100 * (1 - alpha):g}%"
    direction = "better" if np.sign(d.mean()) == better_sign else ("worse" if d.mean() != 0 else "none")
    sig = p < alpha
    if sesoi is None:
        verdict = f"{'SIGNIFICANT' if sig else 'not significant'} ({direction}); no SESOI in the design"
    else:
        inside = -sesoi < lo and hi < sesoi
        if sig and inside:
            verdict = f"SIGNIFICANT BUT BELOW SESOI ({direction}; real but smaller than {sesoi} kt)"
        elif sig:
            verdict = direction.upper()
        elif inside:
            verdict = f"NULL (informative; the {level} CI excludes effects of {sesoi} kt or more)"
        else:
            verdict = f"NULL (inconclusive; effects of {sesoi} kt are not excluded)"
    if floor >= alpha:
        verdict += f" [only {m} of {n} seeds differ: p floor 2^(1-{m}) = {floor:.4f} >= {alpha}, significance unreachable]"
    row = {"test": name, "branch": branch, "n": n, "m_nonzero": m, "p_floor": floor, "mean_diff": d.mean(),
           "sd_diff": d.std(ddof=1) if n > 1 else np.nan, "ci_level": level, "ci_lo": lo, "ci_hi": hi,
           "p_signflip": p, "alpha": alpha, "sesoi": sesoi, "n_seeds_better": int((np.sign(d) == better_sign).sum()),
           "significant": sig, "direction": direction, "verdict": verdict}
    row.update(extra or {})
    return row


def bias_contrast(new, old):
    """N5's fixed bias branch: both means < 0 -> d = new - old, better d > 0 (less under-prediction); otherwise
    (a sign crossing) d = |new| - |old| per seed, better d < 0."""
    new, old = np.asarray(new, float), np.asarray(old, float)
    if new.mean() < 0 and old.mean() < 0:
        return new - old, +1, "both negative: d = bias_new - bias_ref, d > 0 better"
    return np.abs(new) - np.abs(old), -1, "sign crossing: d = |bias_new| - |bias_ref|, d < 0 better"


def power_reduced(n, delta, sd, alpha, reps):
    """power.py's method at a reduced n: reps normal draws of n paired differences (one fresh generator seeded
    POWER_SEED), each tested by the two-sided exact sign-flip over all 2^n assignments; the share with p < alpha. The
    sign-matrix product is chunked, POWER_CHUNK reps at a time."""
    d = np.random.default_rng(POWER_SEED).normal(delta, sd, (reps, n))
    signs = N5.sign_matrix(n)
    hits = 0
    for i in range(0, reps, POWER_CHUNK):
        c = d[i:i + POWER_CHUNK]
        p = (np.abs(signs @ c.T) / n >= np.abs(c.mean(axis=1)) - 1e-12).mean(axis=0)
        hits += int((p < alpha).sum())
    return hits / reps


def expected_order(arm, seed):
    """Schedule position 1..40: round-robin by seed, arm order alternating (C/A for the 1st seed, A/C for the 2nd)."""
    i = SEEDS.index(seed)
    first = ARMS[i % 2]
    return 2 * i + (1 if arm == first else 2)


# ---- input checks ------------------------------------------------------------------------------------------
def pinned_paths(arm, seed):
    return {"train_log": f"logs/f31/{arm}_seed{seed}.log",
            "native_checkpoint": f"models/f31/{arm}/seed{seed}.pt",
            "epoch_dir": f"models/f31/{arm}/seed{seed}_epochs"}


def epoch_paths(arm, seed, e):
    return {"checkpoint": f"models/f31/{arm}/seed{seed}_epochs/epoch_{e:02d}.pt",
            "csv": f"results/f31/{arm}_seed{seed}_e{e:02d}_h9.csv",
            "eval_log": f"logs/f31/{arm}_seed{seed}_e{e:02d}_eval_h9.log"}


def parse_train_log(path, arm, seed, design):
    """The F-31 train log (run_f31.py header + train.py stdout). Returns ([(epoch, train, val)], [saved epochs])."""
    txt = N5.read_text_strict(path)
    lines = txt.splitlines()
    ep = [(int(e), float(tr), float(va)) for e, tr, va in
          re.findall(r"^epoch (\d+): train ([\d.]+) kt\s+val ([\d.]+) kt", txt, flags=re.M)]
    saved = [int(m.group(1)) for i, ln in enumerate(lines) if ln.strip() == "saved" and i > 0
             for m in [re.match(r"epoch (\d+):", lines[i - 1])] if m]
    m_seed = re.search(r"^seed: (\d+)\s+checkpoint: (\S+)", txt, flags=re.M)
    m_val = re.search(r"^val storms:\s+(\[.*\])", txt, flags=re.M)
    m_tr = re.search(r"^train storms: (\[.*\])", txt, flags=re.M)
    m_n = re.findall(r"^train samples: (\d+)\s+val samples: (\d+)", txt, flags=re.M)
    m_fl = re.search(r"^frame list: (\S+): (\d+) frames from (\d+) storms", txt, flags=re.M)
    flags = re.search(r"loss: (\w+)\s+circular_mask: (\w+)\s+rotate: (\w+)", txt)
    problems = []
    if not m_seed or int(m_seed.group(1)) != seed:
        problems.append("seed")
    if not m_seed or Path(m_seed.group(2)).as_posix() != pinned_paths(arm, seed)["native_checkpoint"]:
        problems.append("checkpoint path")
    if not m_val or N5.ids_in(m_val.group(1)) != design["val_ids"]:
        problems.append("validation storms")
    if not m_tr or sorted(N5.ids_in(m_tr.group(1))) != design["ids"][arm]:
        problems.append("training storm set (not the recorded list)")
    if not m_n or int(m_n[-1][0]) != design["frames"][arm] or int(m_n[-1][1]) != design["val_frames"]:
        problems.append("train/val frame counts")
    if arm == "atlantic" and (not m_fl or Path(m_fl.group(1)).as_posix() != "results/n5_atlantic/frame_list_atlantic.txt"
                              or int(m_fl.group(2)) != design["frames"][arm]):
        problems.append("frame list line")
    if arm == "control" and m_fl:
        problems.append("control used a frame list")
    if [e for e, _, _ in ep] != list(range(1, EPOCHS[arm][4] + 1)):
        problems.append(f"epochs ({len(ep)} logged, {EPOCHS[arm][4]} expected)")
    if not flags or flags.groups() != ("l1", "False", "False"):
        problems.append("loss/mask/rotation")
    if "TRAIN_FAILED" in txt or not re.search(r"^END ", txt, flags=re.M):
        problems.append("TRAIN_FAILED or unfinished")
    if not saved:
        problems.append("no 'saved' line")
    if problems:
        raise DesignError(f"{path}: does not match the recorded design: {', '.join(problems)}")
    return ep, saved


def check_attempts(man, root, where):
    for key in ("aborted_attempts", "eval_attempts"):
        for rel in man.get(key, []):
            p = Path(root) / rel
            if not p.is_file() or json.loads(N5.read_text_strict(p)).get("status") != "complete":
                raise DesignError(f"{where}: {key} {rel} is missing or not complete")


def check_training(man, arm, seed, design, root, where):
    """Every training-side check, on a manifest or on run_f31's phase-1 record (the same keys), so no training-side
    refusal can first appear after the H9 look. Returns (epoch rows, saved epochs, slots)."""
    root = Path(root)
    if (man.get("arm"), man.get("seed")) != (arm, seed):
        raise DesignError(f"{where}: record is for {man.get('arm')} seed {man.get('seed')}")
    for k, want in pinned_paths(arm, seed).items():
        if Path(str(man.get(k))).as_posix() != want:
            raise DesignError(f"{where}: {k} is {man.get(k)}, not the design's {want}")
    if man.get("train_list") != design["train_lists"][arm]:
        raise DesignError(f"{where}: train list {man.get('train_list')} is not {design['train_lists'][arm]}")
    if man.get("frame_list_sha256") != design["frame_list_sha"][arm]:
        raise DesignError(f"{where}: frame list SHA-256 differs from the recorded one")
    if man.get("train_py_sha256") != design["train_py_sha"]:
        raise DesignError(f"{where}: train.py blob {man.get('train_py_sha256')} is not the pinned one; the nested "
                          "read-at-1x/2x/4x equivalence holds only for it (John, 2026-10-19 23:50)")
    if man.get("order") != expected_order(arm, seed):
        raise DesignError(f"{where}: order {man.get('order')} is not the schedule's {expected_order(arm, seed)}")
    if man.get("native_same_weights") is not True:
        raise DesignError(f"{where}: native_same_weights must be true (seed<s>.pt == the native 4x epoch file)")
    for when in ("protected_before", "protected_after"):
        got = man.get(when, {})
        bad = sorted(k for k, v in design["protected"].items() if got.get(k) != v)
        if bad:
            raise DesignError(f"{where}: {when} protected hashes differ: {bad}")
    check_attempts(man, root, where)
    ep, saved = parse_train_log(root / pinned_paths(arm, seed)["train_log"], arm, seed, design)

    e_max = EPOCHS[arm][4]
    reval = {int(k): float(v) for k, v in (man.get("val_mae_revalidated") or {}).items()}
    logged = {e: va for e, _, va in ep}
    if sorted(reval) != list(range(1, e_max + 1)) or any(abs(round(v, 2) - logged[e]) > 0.0051 for e, v in reval.items()):
        raise DesignError(f"{where}: re-validated values missing or inconsistent with the train log")
    try:
        slots, ties, near = selections(arm, reval, saved)
    except DesignError as e:
        raise DesignError(f"{where}: {e}") from None
    if man.get("slots") != slots:
        raise DesignError(f"{where}: slots {man.get('slots')} are not the validation-only selections {slots}")
    core = ("revalidated_argmin", "saved_epoch")
    got = {b: (None if t is None else {k: t.get(k) for k in core}) for b, t in (man.get("tie_rule") or {}).items()}
    if got != {b: (None if t is None else {k: t[k] for k in core}) for b, t in ties.items()}:
        raise DesignError(f"{where}: tie_rule {man.get('tie_rule')} does not match the re-validated values ({ties})")
    if man.get("near_ties") != near:
        raise DesignError(f"{where}: near_ties {man.get('near_ties')} do not match the re-validated values ({near})")

    # Disk (John's choice 4): keep only selected and H9-evaluated checkpoints, plus seed<s>.pt; every other epoch file
    # is hashed under deleted_epoch_sha256 (never files_sha256) and gone.
    kept_epochs = sorted(set(slots.values()))
    pins = pinned_paths(arm, seed)
    want_kept = {pins["train_log"], pins["native_checkpoint"]} | {epoch_paths(arm, seed, e)["checkpoint"]
                                                                 for e in kept_epochs}
    files = man.get("files_sha256") or {}
    if set(files) != want_kept:
        raise DesignError(f"{where}: files_sha256 must name exactly the train log, seed<s>.pt and the slot epochs' "
                          f"files {sorted(want_kept)}")
    for rel, h in files.items():
        if not (root / rel).is_file() or N5.sha256_file(root / rel) != h:
            raise DesignError(f"{where}: kept file {rel} is missing or its SHA-256 differs from the record")
    deleted = man.get("deleted_epoch_sha256") or {}
    want_deleted = {str(e) for e in range(1, e_max + 1) if e not in kept_epochs}
    if set(deleted) != want_deleted or not all(isinstance(h, str) and HEX64.fullmatch(h) for h in deleted.values()):
        raise DesignError(f"{where}: deleted_epoch_sha256 must hash every non-slot epoch 1-{e_max}")
    present = [e for e in map(int, want_deleted) if (root / epoch_paths(arm, seed, e)["checkpoint"]).exists()]
    if present:
        raise DesignError(f"{where}: non-slot epoch files still on disk: {sorted(present)[:5]}")
    return ep, saved, slots


def load_run(manifest_path, arm, seed, design, root):
    root = Path(root)
    man = json.loads(N5.read_text_strict(manifest_path))
    ep, saved, slots = check_training(man, arm, seed, design, root, manifest_path)
    rec_path = Path(manifest_path).with_name(f"{arm}_seed{seed}.train.json")
    if not rec_path.is_file():
        raise DesignError(f"{manifest_path}: no phase-1 record {rec_path}")
    rec = json.loads(N5.read_text_strict(rec_path))
    phase2 = {"evals", "eval_attempts", "protected_after_eval"}
    diff = sorted(k for k in (set(man) | set(rec)) - phase2 if man.get(k) != rec.get(k))
    if diff:
        raise DesignError(f"{manifest_path}: differs from its phase-1 record {rec_path.name} in {diff}; the slots and "
                          "every training-side field are fixed before the look")
    after = man.get("protected_after_eval") or {}
    bad = sorted(k for k, v in design["protected"].items() if after.get(k) != v)
    if bad or set(after) != set(design["protected"]):
        raise DesignError(f"{manifest_path}: protected_after_eval differs from the protected hashes: "
                          f"{bad or sorted(set(after) ^ set(design['protected']))}")
    epochs = sorted(set(slots.values()))
    evals = man.get("evals") or {}
    if sorted(evals) != sorted(map(str, epochs)):
        raise DesignError(f"{manifest_path}: evals {sorted(evals)} must be exactly the distinct slot epochs {epochs}")
    # Nothing but a slot epoch is ever read on H9: any other CSV or eval log for this run is refused.
    for pattern, key in ((f"results/f31/{arm}_seed{seed}_e*_h9.csv", "csv"),
                         (f"logs/f31/{arm}_seed{seed}_e*_eval_h9.log", "eval_log")):
        found = {p.relative_to(root).as_posix() for p in root.glob(pattern)}
        allowed = {epoch_paths(arm, seed, e)[key] for e in epochs}
        if found - allowed:
            raise DesignError(f"{manifest_path}: H9 output for a non-slot epoch: {sorted(found - allowed)}")
    preds = {}
    for e in epochs:
        p = epoch_paths(arm, seed, e)
        rec = evals[str(e)]
        for key, rel in (("csv", p["csv"]), ("eval_log", p["eval_log"])):
            if not (root / rel).is_file() or N5.sha256_file(root / rel) != rec.get(f"{key}_sha256"):
                raise DesignError(f"{manifest_path}: {rel} is missing or its SHA-256 differs from the manifest")
        d = N5.load_csv(root / p["csv"], design)
        N5.check_eval_log(root / p["eval_log"], p["checkpoint"], p["csv"], d)
        preds[e] = d
    return dict(man=man, ep=ep, saved=saved, slots=slots, preds=preds)


def single_commit(mans):
    commits = {m.get("commit") for m in mans}
    if len(commits) != 1:
        raise DesignError(f"runs come from more than one commit: {sorted(map(str, commits))}")
    return commits.pop()


def record_fails(rec_path, root):
    """N5's trigger, per run: a file the record hashed in files_sha256 is missing or changed."""
    files = json.loads(N5.read_text_strict(rec_path)).get("files_sha256") or {}
    return any(not (Path(root) / f).is_file() or N5.sha256_file(Path(root) / f) != h for f, h in files.items())


def load_deviations(path, root, manifest_dir, git, design):
    """Rule (A): {(arm, seed): reason} from the committed, clean DEVIATIONS_PATH, naming only runs whose training
    record no longer verifies. Returns (dropped, the file's blob SHA-256 at HEAD)."""
    if Path(path).resolve() != (Path(root) / DEVIATIONS_PATH).resolve():
        raise DesignError(f"--deviations must be {DEVIATIONS_PATH} in the repo, not {path}")
    if git(["ls-files", "--error-unmatch", DEVIATIONS_PATH], check=False).returncode != 0 or \
            git(["status", "--porcelain", "--", DEVIATIONS_PATH]).stdout.strip():
        raise DesignError(f"{DEVIATIONS_PATH} must be committed and unchanged before the analysis runs")
    d = json.loads(N5.read_text_strict(path))
    rows = d.get("dropped_runs") if isinstance(d, dict) else None
    if not isinstance(rows, list) or not rows:
        raise DesignError(f"{path}: needs a non-empty \"dropped_runs\" list of {{arm, seed, reason}}")
    out = {}
    for r in rows:
        r = r if isinstance(r, dict) else {}
        key, why = (r.get("arm"), r.get("seed")), str(r.get("reason", "")).strip()
        if key[0] not in ARMS or type(key[1]) is not int or key[1] not in SEEDS or key in out or not why:
            raise DesignError(f"{path}: each dropped run needs a distinct arm, a seed in {list(SEEDS)} and a reason: {r}")
        out[key] = why
    for arm in ARMS:
        if sum(a == arm for a, _ in out) > len(SEEDS) - 2:
            raise DesignError(f"{path}: fewer than 2 {arm} runs would remain")
    healthy = []
    for arm, s in out:
        rec = Path(manifest_dir) / f"{arm}_seed{s}.train.json"
        if not rec.is_file():  # as N5: a run with no record is not "kept files that can't be restored"
            raise DesignError(f"{DEVIATIONS_PATH}: {arm} seed {s} has no training record {rec}; rule (A) drops only "
                              "a run whose record exists and fails")
        if not record_fails(rec, root):
            healthy.append((arm, s))
    if healthy:
        raise DesignError(f"{DEVIATIONS_PATH}: {healthy}: every file their record hashed is present and unchanged; "
                          "rule (A) drops only a run whose kept files can't be restored (anything else is fixed, "
                          "not dropped)")
    return out, hashlib.sha256(git(["show", f"HEAD:{DEVIATIONS_PATH}"]).stdout).hexdigest()


def verify_self(commit, root, git, check_path=True):
    """This file and the imported analyze_n5.py must be the launch commit's blobs (clean, no diff), and analyze_n5.py
    must be N5's pinned blob. check_path=False is the test hook; main() always checks the paths."""
    root = Path(root)
    if check_path:
        for rel, me in ((SELF_PATH, Path(__file__)), (N5_PATH, Path(N5.__file__))):
            if me.resolve() != (root / rel).resolve():
                raise DesignError(f"run {rel} from the repo, not {me.resolve()}")
    out = {}
    for rel in (SELF_PATH, N5_PATH):
        if git(["status", "--porcelain", "--", rel]).stdout.strip():
            raise DesignError(f"{rel} has uncommitted changes")
        if git(["diff", "--quiet", commit, "--", rel], check=False).returncode != 0:
            raise DesignError(f"{rel} differs from the launch commit {commit}")
        blob = git(["show", f"{commit}:{rel}"]).stdout
        out[rel] = (hashlib.sha256(blob).hexdigest(), git(["rev-parse", f"{commit}:{rel}"]).stdout.decode().strip())
    if out[N5_PATH][0] != N5_BLOB_SHA256:
        raise DesignError(f"{N5_PATH} at {commit} is not N5's pinned blob {N5_BLOB_SHA256[:12]}...")
    return out


def check_train_py(commit, git, design):
    """The launch commit's scripts/train.py must be the pinned blob (John, 2026-10-19 23:50), not only as recorded."""
    got = hashlib.sha256(git(["show", f"{commit}:{TRAIN_PY}"]).stdout).hexdigest()
    if got != design["train_py_sha"]:
        raise DesignError(f"{TRAIN_PY} at {commit} hashes to {got[:12]}..., not the pinned {design['train_py_sha'][:12]}...")


def real_git(root):
    def git(args, check=True):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=check)
    return git


def n5_verdict_lines(root, design):
    """N5's verdict, verbatim from its pinned report: the two primary lines and the JOINT VERDICT line."""
    p = Path(root) / N5_REPORT
    if not p.is_file() or N5.sha256_file(p) != design["n5_report_sha"]:
        raise DesignError(f"{N5_REPORT} is missing or not N5's pinned report ({design['n5_report_sha'][:12]}...)")
    lines = N5.read_text_strict(p).splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.startswith("PRIMARIES"))
    return [lines[i + 1], lines[i + 2], next(ln for ln in lines if ln.startswith("JOINT VERDICT:"))]


def frozen_figures(root, design):
    p = Path(root) / FROZEN_METRICS
    if not p.is_file():
        raise DesignError(f"missing {FROZEN_METRICS}")
    f = pd.read_csv(p).set_index("metric")["ensemble"]
    for k, v in design["frozen"].items():
        if not math.isclose(float(f[k]), v, rel_tol=0, abs_tol=1e-9):
            raise DesignError(f"{FROZEN_METRICS}: {k} {f[k]} is not the committed {v}")
    return dict(design["frozen"])


# ---- main analysis -----------------------------------------------------------------------------------------
def run(manifest_dir, out_dir, root=None, design=None, git=None, deviations=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    design = design or design_constants()
    out_dir = Path(out_dir)
    clash = [f for f in OUTPUTS if (out_dir / f).exists()]
    if clash:
        raise DesignError(f"refusing to overwrite {clash} in {out_dir}")
    g = git or real_git(root)
    dropped, dev_sha = load_deviations(deviations, root, manifest_dir, g, design) if deviations else ({}, None)
    n5_lines = n5_verdict_lines(root, design)
    frozen = frozen_figures(root, design)

    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            if (arm, s) in dropped:
                continue
            p = Path(manifest_dir) / f"{arm}_seed{s}.json"
            if not p.exists():
                raise DesignError(f"missing run manifest {p}")
            runs[(arm, s)] = load_run(p, arm, s, design, root)
    commit = single_commit([r["man"] for r in runs.values()])
    blobs = verify_self(commit, root, g, check_path=git is None)
    check_train_py(commit, g, design)

    ref = next(iter(runs.values()))["preds"]
    ref = ref[min(ref)]
    vecs = {}
    for (arm, s), r in runs.items():
        for e, d in r["preds"].items():
            if not d["crop_path"].equals(ref["crop_path"]):
                raise DesignError(f"{arm} seed {s} epoch {e}: H9 frame order differs")
            vecs[(arm, s, e)] = d["pred"].to_numpy()
    for (k1, v1), (k2, v2) in itertools.combinations(vecs.items(), 2):  # distinct checkpoints, distinct predictions
        if np.allclose(v1, v2, atol=1e-6):
            raise DesignError(f"identical prediction vectors for {k1} and {k2}: inputs are mislabelled or duplicated")

    # Per run and slot: the selected epoch, its flag and the H9 metrics of that epoch's single evaluation.
    met = {(arm, s, e): N5.metrics(d) for (arm, s), r in runs.items() for e, d in r["preds"].items()}
    sel_rows, run_rows, storm_rows, traces = [], [], [], []
    for (arm, s), r in runs.items():
        reval = {int(k): float(v) for k, v in r["man"]["val_mae_revalidated"].items()}
        for slot, e in r["slots"].items():
            k = int(slot.split("_")[1][:-1])
            e_k = EPOCHS[arm][k]
            sel_rows.append({"arm": arm, "seed": s, "order": r["man"]["order"], "slot": slot, "epoch": e,
                             "budget_epochs": e_k,
                             "selected_step": e * BATCHES[arm], "budget_steps": e_k * BATCHES[arm],
                             "flagged": flagged(e, e_k),
                             "tie_rule": json.dumps(r["man"]["tie_rule"].get(f"{k}x")) if slot.startswith("native")
                             else "", "near_ties": " ".join(map(str, r["man"]["near_ties"].get(slot, []))),
                             **val_stability(reval, e, e_k)})
            run_rows.append({"arm": arm, "seed": s, "order": r["man"]["order"], "slot": slot, "epoch": e,
                             **met[(arm, s, e)]})
            storm_rows.append({"arm": arm, "slot": slot, **N5.per_storm(r["preds"][e])})
        traces += [{"arm": arm, "seed": s, "epoch": e, "train_mae": tr, "val_mae": va} for e, tr, va in r["ep"]]
    selection = pd.DataFrame(sel_rows)
    per_run = pd.DataFrame(run_rows)

    def vals(arm, slot, metric, seeds):
        t = per_run[(per_run.arm == arm) & (per_run.slot == slot)].set_index("seed")
        return t.loc[list(seeds), metric].to_numpy(float)

    seeds_of = {arm: [s for s in SEEDS if (arm, s) in runs] for arm in ARMS}
    paired_seeds = [s for s in SEEDS if all((a, s) in runs for a in ARMS)]
    ci = design["ci_grid"]
    tag = "NON-CONFIRMATORY (rule (A): runs dropped as a named deviation): " if dropped else ""

    # Flags per (arm, budget) under the native rule; the qualifier fires at >= half of the arm's runs (10/20).
    qual, flag_counts = {}, {}
    for arm in ARMS:
        n_arm = len(seeds_of[arm])
        for k in BUDGETS:
            c = int(selection[(selection.arm == arm) & (selection.slot == f"native_{k}x")].flagged.sum())
            flag_counts[(arm, f"native_{k}x")] = c
            qual[(arm, k)] = c >= math.ceil(n_arm / 2)
        flag_counts[(arm, "smoothed_4x")] = int(selection[(selection.arm == arm) &
                                                          (selection.slot == "smoothed_4x")].flagged.sum())
    flag_counts[("control", "matched_4x")] = int(selection[(selection.arm == "control") &
                                                           (selection.slot == "matched_4x")].flagged.sum())

    def budget_effect(name, arm, new_slot, ref_slot, alpha):
        ss = seeds_of[arm]
        d = vals(arm, new_slot, "overall MAE", ss) - vals(arm, ref_slot, "overall MAE", ss)
        row = paired_test(name, d, alpha, -1, SESOI_MAE, f"d = MAE({new_slot}) - MAE({ref_slot}), d < 0 better", ci,
                          {"arm": arm, "ref_mean": vals(arm, ref_slot, "overall MAE", ss).mean(),
                           "new_mean": vals(arm, new_slot, "overall MAE", ss).mean()})
        if qual[(arm, int(new_slot.split("_")[1][:-1]))]:
            row["verdict"] += f" | BUDGET-LIMITED at {new_slot.split('_')[1]} in the {arm} arm: read 'at this budget'"
        return row

    prim = [budget_effect("P1 control (frozen-ensemble configuration): overall MAE, 4x - 1x", "control",
                          "native_4x", "native_1x", ALPHA),
            budget_effect("P2 atlantic: overall MAE, 4x - 1x", "atlantic", "native_4x", "native_1x", ALPHA)]
    for r in prim:
        r["verdict"] = tag + r["verdict"]
    sens = []
    for arm, lab in (("control", "P1"), ("atlantic", "P2")):
        sens.append(budget_effect(f"{lab} sensitivity, 2x: overall MAE, 2x - 1x", arm, "native_2x", "native_1x", ALPHA))
        sens.append(budget_effect(f"{lab} sensitivity, smoothed 4x vs native 1x (rule and budget change together)",
                                  arm, "smoothed_4x", "native_1x", ALPHA))

    # Secondary, alpha 0.05: S1/S2 atlantic - control at 4x (control on the matched grid); S3/S4 per arm, 4x - 1x.
    ps = paired_seeds
    s1d = vals("atlantic", "native_4x", "overall MAE", ps) - vals("control", "matched_4x", "overall MAE", ps)
    sec = [paired_test("S1 atlantic - control at 4x: overall MAE (control matched)", s1d, SECONDARY_ALPHA, -1,
                       SESOI_MAE, "d = MAE_atl(native_4x) - MAE_ctl(matched_4x), d < 0 better", ci)]
    d2, sign2, br2 = bias_contrast(vals("atlantic", "native_4x", ">96 bias", ps),
                                   vals("control", "matched_4x", ">96 bias", ps))
    sec.append(paired_test("S2 atlantic - control at 4x: >96 kt bias (control matched)", d2, SECONDARY_ALPHA, sign2,
                           SESOI_BIAS, br2.replace("new", "atl").replace("ref", "ctl"), ci))
    for arm in ARMS:
        ss = seeds_of[arm]
        d3, sign3, br3 = bias_contrast(vals(arm, "native_4x", ">96 bias", ss), vals(arm, "native_1x", ">96 bias", ss))
        sec.append(paired_test(f"S3 {arm}: >96 kt bias, 4x - 1x", d3, SECONDARY_ALPHA, sign3, None,
                               br3.replace("new", "4x").replace("ref", "1x"), ci))
        d4 = vals(arm, "native_4x", ">96 MAE", ss) - vals(arm, "native_1x", ">96 MAE", ss)
        sec.append(paired_test(f"S4 {arm}: >96 kt MAE, 4x - 1x", d4, SECONDARY_ALPHA, -1, None,
                               "d = >96 MAE(4x) - >96 MAE(1x), d < 0 better", ci))
    s1n = vals("atlantic", "native_4x", "overall MAE", ps) - vals("control", "native_4x", "overall MAE", ps)
    sens.append(paired_test("S1 sensitivity, atlantic - control at 4x: control at its native 4x pick", s1n, SECONDARY_ALPHA, -1, SESOI_MAE,
                            "d = MAE_atl(native_4x) - MAE_ctl(native_4x), d < 0 better", ci))
    for r in sec + sens:
        r["verdict"] = tag + r["verdict"]

    # Descriptive. No atlantic - control difference, p or CI at 1x or 2x: F-31 is not a rerun of N5.
    curve = per_run.drop(columns=["seed", "order", "epoch"]).groupby(["arm", "slot"]).agg(["mean", "std"])
    curve.columns = [f"{m} {s}" for m, s in curve.columns]
    curve = curve.reset_index()
    truth = ref["truth"].to_numpy()
    ens = []
    for arm in ARMS:
        groups = {"seeds 10-19": [s for s in seeds_of[arm] if s < 20], "seeds 20-29": [s for s in seeds_of[arm] if s >= 20],
                  "all": seeds_of[arm]}
        for k in BUDGETS:
            slot = f"native_{k}x"
            for gname, ss in groups.items():
                if not ss:
                    continue
                members = [vecs[(arm, s, runs[(arm, s)]["slots"][slot])] for s in ss]
                mean_pred = np.mean(members, axis=0)
                m = N5.metrics(ref.assign(pred=mean_pred))
                member_mae = np.mean([met[(arm, s, runs[(arm, s)]["slots"][slot])]["overall MAE"] for s in ss])
                jack = [N5.metrics(ref.assign(pred=np.mean(members[:i] + members[i + 1:], axis=0)))["overall MAE"]
                        for i in range(len(members))] if gname != "all" and len(members) > 2 else []
                low = truth < 64
                ens.append({"arm": arm, "slot": slot, "members": gname, "n_members": len(ss),
                            "overall MAE": m["overall MAE"], "fixes-only MAE": m["fixes-only MAE"],
                            ">96 MAE": m[">96 MAE"], ">96 bias": m[">96 bias"], "spearman >96": m["spearman >96"],
                            "<64 MAE": float(np.abs(mean_pred - truth)[low].mean()),
                            "<64 bias": float((mean_pred - truth)[low].mean()),
                            "member mean MAE": member_mae, "ensemble gain": member_mae - m["overall MAE"],
                            "jackknife MAE min": min(jack) if jack else np.nan,
                            "jackknife MAE max": max(jack) if jack else np.nan})
    ens = pd.DataFrame(ens)
    storms = pd.DataFrame(storm_rows).groupby(["arm", "slot"]).mean().T
    trace = pd.DataFrame(traces).groupby(["arm", "epoch"])[["train_mae", "val_mae"]].agg(["mean", "std"]).reset_index()
    trace.columns = [" ".join(x).strip() for x in trace.columns]

    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(index=False, lineterminator="\n", float_format="%.6f")
    pd.DataFrame(prim).to_csv(out_dir / "primary_tests.csv", **kw)
    pd.DataFrame(sens).to_csv(out_dir / "sensitivity.csv", **kw)
    pd.DataFrame(sec).to_csv(out_dir / "secondary.csv", **kw)
    selection.to_csv(out_dir / "selection.csv", **kw)
    per_run.to_csv(out_dir / "per_run.csv", **kw)
    curve.to_csv(out_dir / "budget_curve.csv", **kw)
    ens.to_csv(out_dir / "ensemble.csv", **kw)
    storms.to_csv(out_dir / "per_storm.csv", lineterminator="\n", float_format="%.6f")
    trace.to_csv(out_dir / "val_trace.csv", **kw)

    def line(r):
        return (f"  {r['test']}: d {r['mean_diff']:+.3f} (sd {r['sd_diff']:.3f}, n {r['n']}, {r['m_nonzero']} non-zero, "
                f"p floor {r['p_floor']:.2g}), {r['ci_level']} CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], "
                f"p {r['p_signflip']:.5f} -> {r['verdict']}   [{r['branch']}]")

    def mean_of(arm, slot):
        return per_run[(per_run.arm == arm) & (per_run.slot == slot)]["overall MAE"].mean()

    ties = {f"{a} seed {s}": {b: t for b, t in r["man"]["tie_rule"].items() if t} for (a, s), r in runs.items()
            if any(r["man"]["tie_rule"].values())}
    near = {f"{a} seed {s}": {k: v for k, v in r["man"]["near_ties"].items() if v} for (a, s), r in runs.items()
            if any(r["man"]["near_ties"].values())}
    attempts = {k: {f"{a} seed {s}": r["man"][k] for (a, s), r in runs.items() if r["man"].get(k)}
                for k in ("aborted_attempts", "eval_attempts")}
    zeros = {arm: int(sum(runs[(arm, s)]["slots"]["native_4x"] == runs[(arm, s)]["slots"]["native_1x"]
                          for s in seeds_of[arm])) for arm in ARMS}
    deviation = []
    if dropped:
        # Pre-look counts from all 40 phase-1 records (what decisions.md recorded before the look), beside these.
        pre = {}
        for arm in ARMS:
            for k in BUDGETS:
                recs = [json.loads(N5.read_text_strict(Path(manifest_dir) / f"{arm}_seed{s}.train.json"))
                        for s in SEEDS]
                pre[(arm, k)] = sum(flagged(rr["slots"][f"native_{k}x"], EPOCHS[arm][k]) for rr in recs)
        reps = design["power_reps"]
        sizes = {"P1/P2": [("P1 control", len(seeds_of["control"])), ("P2 atlantic", len(seeds_of["atlantic"]))],
                 "S1": [("S1", len(paired_seeds))], "S2": [("S2", len(paired_seeds))]}
        deviation = [
            "DEVIATION (rule (A), John 2026-10-19 17:03): runs dropped: "
            + "; ".join(f"{a} seed {s}: {why}" for (a, s), why in sorted(dropped.items())),
            f"  Every verdict is NON-CONFIRMATORY. {DEVIATIONS_PATH} blob sha256 {dev_sha}.",
            f"  Reduced n and power at the SESOI (n = {len(SEEDS)}: design §5's table, quoted; reduced n: two-sided exact "
            f"sign-flip, normal model, {reps} reps per cell, seed {POWER_SEED}, as power.py; SE <= "
            f"{math.sqrt(0.25 / reps):.3f}):"]
        for key, spec in POWER_N20.items():
            for lab, n_red in sizes[key]:
                cells = ", ".join(
                    f"sd_diff {sd:.2f}: {pw:.2f} -> "
                    + (f"{pw:.2f} (n unchanged)" if n_red == len(SEEDS) else
                       f"{power_reduced(n_red, spec['sesoi'], sd, spec['alpha'], reps):.2f}")
                    for sd, pw in spec["cells"].items())
                deviation.append(f"    {lab} (alpha {spec['alpha']}, {spec['sesoi']} kt), n {len(SEEDS)} -> {n_red}: "
                                 f"{cells}")
        deviation.append("  BUDGET-LIMITED before the look (all 40 phase-1 records, >= 10/20) vs after the drop "
                         "(>= half the remaining runs):")
        deviation += [f"    {arm} {k}x: pre-look {pre[(arm, k)]}/{len(SEEDS)}"
                      f"{' -> BUDGET-LIMITED' if pre[(arm, k)] >= len(SEEDS) // 2 else ''}; adjusted "
                      f"{flag_counts[(arm, f'native_{k}x')]}/{len(seeds_of[arm])}"
                      f"{' -> BUDGET-LIMITED' if qual[(arm, k)] else ''}" for arm in ARMS for k in BUDGETS]
    lines = [
        "F-31 budget study: each arm read at 1x, 2x and 4x x 1640 steps inside one run per seed, H9 pooled. "
        "Pre-registered analysis output, unedited.",
        f"seeds {list(SEEDS)}; control epochs {EPOCHS['control']}, atlantic epochs {EPOCHS['atlantic']}",
        f"analysis blob sha256 {blobs[SELF_PATH][0]} (blob id {blobs[SELF_PATH][1]}); imported {N5_PATH} blob sha256 "
        f"{blobs[N5_PATH][0]} (blob id {blobs[N5_PATH][1]}); commit {commit}",
        "F-31 is a new experiment, not a rerun or rescue of N5. N5's verdict stands whatever F-31 finds.",
    ]
    lines += deviation
    lines += [
        f"PRIMARIES (Bonferroni, alpha {ALPHA} each, two-sided exact sign-flip, native selection; SESOI {SESOI_MAE} kt, "
        "N5's, for comparability):",
        *[line(r) for r in prim],
        f"  Zero seeds (4x pick <= the 1x budget, d = 0 exactly): control {zeros['control']}, atlantic "
        f"{zeros['atlantic']}. With m non-zero seeds the p floor is 2^(1-m); p < {ALPHA} needs m >= 7.",
        "BUDGET-LIMITED (native rule; flagged when the selected epoch > 0.85 x E_k; qualifier at >= half the runs):",
        *[f"  {arm} {k}x (E_k {EPOCHS[arm][k]}): {flag_counts[(arm, f'native_{k}x')]}/{len(seeds_of[arm])} flagged"
          f"{' -> BUDGET-LIMITED' if qual[(arm, k)] else ''}" for arm in ARMS for k in BUDGETS],
        f"  also reported: control matched_4x {flag_counts[('control', 'matched_4x')]}, smoothed_4x control "
        f"{flag_counts[('control', 'smoothed_4x')]}, atlantic {flag_counts[('atlantic', 'smoothed_4x')]}",
        "SENSITIVITY (pre-registered; does not change any verdict):",
        *[line(r) for r in sens],
        f"SECONDARY (alpha {SECONDARY_ALPHA}, uncorrected; reported beside the verdicts, never replacing them):",
        line(sec[0]), line(sec[1]),
        "  N5's verdict, verbatim from its report (N5's verdict stands; same H9, so S1/S2 are not an independent "
        "replication):",
        *[f"    {ln}" for ln in n5_lines],
        *[line(r) for r in sec[2:]],
        "NO atlantic - control test at 1x or 2x (F-31 is not a rerun of N5). Arm means, side by side, descriptive:",
        *[f"  {k}x native: control {mean_of('control', f'native_{k}x'):.3f}, atlantic "
          f"{mean_of('atlantic', f'native_{k}x'):.3f}" for k in (1, 2)],
        "H9 SPLIT (descriptive): original three (Ian, Nicole, Claudette) vs the six added later, mean overall MAE:",
        *[f"  {arm} {slot}: H3 {per_run[(per_run.arm == arm) & (per_run.slot == slot)]['H3 MAE'].mean():.3f}, added6 "
          f"{per_run[(per_run.arm == arm) & (per_run.slot == slot)]['added6 MAE'].mean():.3f}"
          for arm in ARMS for slot in ("native_1x", "native_4x")],
        "VALIDATION STABILITY (descriptive, validation only): val at each pick vs its +-2-epoch neighbours, and the sd "
        "over each budget's last 15%: selection.csv",
        "FROZEN ENSEMBLE (committed H9 figures, results/ensemble; unpaired: different seeds, an older code version, the "
        "same H9; not a test): " + ", ".join(f"{k} {v:.3f}" for k, v in frozen.items()),
        "  F-31 ensembles of the control arm (= the frozen recipe) by budget: ensemble.csv",
        f"Tie rule (|d| < {N5.TIE_KT} kt; train.py's own pick evaluated): {ties or 'none'}",
        f"Near ties (matched_4x, smoothed_4x; the first minimum is kept): {near or 'none'}",
        f"Aborted training attempts: {attempts['aborted_attempts'] or 'none'}; aborted H9 evaluation attempts: "
        f"{attempts['eval_attempts'] or 'none'}",
        "H9 is read only on the slot epochs fixed before the look (native at 1x/2x/4x, matched and smoothed at 4x);"
        " some runs' slots are adjacent epochs. No held-out storm is evaluated per epoch; traces are validation only.",
        "PER STORM (exploratory; >96 stats only with >= 10 frames > 96 kt): per_storm.csv. Sam (AL182021) had "
        "per-epoch exposure on 10-14.",
        "DESCRIPTIVE (not tests): budget_curve.csv, per_run.csv, selection.csv, ensemble.csv, val_trace.csv",
    ]
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return dict(primary=prim, sensitivity=sens, secondary=sec, per_run=per_run, selection=selection, ensemble=ens,
                qualifiers=qual, flag_counts=flag_counts, zeros=zeros, blobs=blobs, report=lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deviations", default=None, help="rule (A): JSON naming the runs to drop, with reasons")
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to((root / "results/f31").resolve()) or out == (root / "results/f31").resolve():
        ap.error(f"--out-dir {args.out_dir}: must be a new directory under results/f31/")
    run(args.manifest_dir, args.out_dir, deviations=args.deviations)


if __name__ == "__main__":
    sys.exit(main())
