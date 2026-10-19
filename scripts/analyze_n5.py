"""N5 Atlantic-only arm: pre-registered analysis. DRAFT r2 (reports/v2_survey/n5/design.md); not yet approved.

Read-only, CPU only, deterministic (no torch, no random numbers). At launch this file is committed as
scripts/analyze_n5.py. Its hash is the SHA-256 of the committed blob, computed in Python from the bytes of
`git show <commit>:scripts/analyze_n5.py`. It is not the hash of the Windows working-tree file, whose bytes differ
under core.autocrlf. At run time the script refuses to run unless the working file is clean and identical to the
launch commit's blob (verify_self).

Between the "fixed before launch" markers, only ARM_TRAIN_FRAMES and FRAME_LIST_SHA256 may change after approval.
They are filled in from the build, recorded in logs/decisions.md, and the approved blob must differ from the
launch blob in exactly those two lines. ARM_EPOCHS is computed from ARM_TRAIN_FRAMES by the budget rule.

Inputs: the per-run manifests that scripts/run_n5.py writes, logs/n5_atlantic/runs/<arm>_seed<s>.json (20 of them).
Each manifest names its train log, eval logs, checkpoints and H9 prediction CSVs, with SHA-256s and the commit.
Every link is verified (see load_run). Optional input: --frozen-dir results/storms (the frozen expanded members'
expanded_seed<s>_h9.csv), for a descriptive, unpaired reproducibility check only.

    python scripts/analyze_n5.py --manifest-dir logs/n5_atlantic/runs --out-dir results/n5_atlantic/analysis
        [--frozen-dir results/storms]

Writes into --out-dir (refuses to overwrite): primary_tests.csv, sensitivity.csv, per_run.csv, per_arm.csv,
paired_differences.csv, per_storm.csv, ensemble.csv, val_trace.csv, frozen_vs_control.csv (with --frozen-dir),
report.txt.
"""
import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---- fixed before launch -----------------------------------------------------------------------------------
SEEDS = tuple(range(10))
ARMS = ("control", "atlantic")
# H9 per-storm frame counts as evaluated in all 20 results/storms/{control,expanded}_seed*_h9.csv (2026-10-14), and
# the SHA-256 of their sorted (crop_path, truth) pairs (h9_pin below). They hold only while H9's exclusions are
# unchanged; the N5 exclusion step may add entries for the 31 new storms only.
H9 = {"AL092022": 197, "AL172022": 122, "AL032021": 74,  # original three (H3): Ian, Nicole, Claudette
      "AL022024": 260, "AL132023": 249, "AL182021": 295, "AL072022": 232, "AL122024": 177, "AL312020": 146}
H3 = ("AL092022", "AL172022", "AL032021")
ADDED6 = tuple(s for s in H9 if s not in H3)
H9_FRAMES = 1752
H9_PIN = "267f154be4a2a3dd5d11fa3d4af38c942f05041ada35e778fac31dc64646b04f"
VAL_IDS = ["AL192020", "AL162023", "AL132020", "AL202020", "AL082023"]  # storms_val_expanded.txt, file order
VAL_FRAMES = 940
CONTROL_IDS = sorted(["AL142018", "AL142024", "AL052019", "AL062020", "AL092021", "AL142021", "AL102023", "AL012024",
                      "AL042024", "AL062018", "AL092019", "AL132019", "AL262020", "AL292020", "AL122021", "AL062022",
                      "AL202023", "AL092024", "AL182024"])  # storms_train_expanded.txt minus validation
NEW_IDS = sorted(["AL072021", "AL272020", "AL282020", "AL082018", "AL162018", "AL032018", "AL062024", "AL102019",
                  "AL102024", "AL132024", "AL172020", "AL052024", "AL152023", "AL082020", "AL092020", "AL132018",
                  "AL142023", "AL152022", "AL052021", "AL052022", "AL132022", "AL162022", "AL162024", "AL022018",
                  "AL022019", "AL052023", "AL082021", "AL092018", "AL142020", "AL162020", "AL252020"])
ARM_IDS = sorted(CONTROL_IDS + NEW_IDS)
TRAIN_LISTS = {"control": "storms_train_expanded.txt", "atlantic": "storms_train_n5_atlantic.txt"}
CONTROL_TRAIN_FRAMES = 2081
CONTROL_EPOCHS = 25
CONTROL_GRID = tuple(range(1, CONTROL_EPOCHS + 1, 2))  # primary: control re-selected on odd epochs (13 candidates)
TIE_KT = 1e-4  # 2026-10-19 amendment (a): re-validation argmin != train.py's pick is a tie only when |d| < TIE_KT kt
ARM_TRAIN_FRAMES = None  # filled from the build (post-exclusion, post-Marco frame list), before hashing
FRAME_LIST_SHA256 = None  # SHA-256 of results/n5_atlantic/frame_list_atlantic.txt, filled with ARM_TRAIN_FRAMES
BATCH = 32
TARGET_STEPS = 1640
PRIMARIES = ("overall MAE", ">96 bias")  # H9 pooled, per member, paired atlantic - control over seeds 0-9
ALPHA = 0.025  # each: Bonferroni 0.05 / 2; two-sided exact sign-flip; 97.5% CIs by test inversion
SESOI = {"overall MAE": 0.9, ">96 bias": 3.0}  # kt, power-derived (not a meaningfulness claim)
SECONDARY = (">96 MAE",)  # alpha 0.05, uncorrected, labelled secondary
SECONDARY_ALPHA = 0.05
UNDERTRAIN_FRACTION = 0.85  # flag: selected checkpoint in the last 15% of steps
UNDERTRAIN_RUNS = 5  # >= 5/10 runs flagged in either arm -> the result is "budget-limited"
HIGH = 96  # ">96" means truth > 96 kt, as in scripts/storms_analysis.py
PER_STORM_MIN_HIGH = 10  # per-storm >96 stats only with >= 10 frames > 96 kt (2025 entry's rule)
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"),
         (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
PROTECTED = {  # FEATURES.md / COORDINATOR.md protected block; must be identical before and after every run
    "models/baseline_9kt.pt": "f95e6c7e2672c37e28aaf138e89af4468e8f3c9211c24d1052440ae8cc18e9cc",
    "best_model.pt": "f95e6c7e2672c37e28aaf138e89af4468e8f3c9211c24d1052440ae8cc18e9cc",
    "results/heldout_predictions.csv": "dc12980781c7685974ea50a07a429297104f8210995253dbc5473157d7c287ef",
    "models/storms/expanded/seed0.pt": "316b511884b62b032ccae78e9f59207ac48a2f7a05e649641d5eaa0ded24bdfa",
    "models/storms/expanded/seed1.pt": "f7d8cc6058dae7937df8cbc90cdefcd1649b2aac8404dca9a00da3b02294b7f5",
    "models/storms/expanded/seed2.pt": "a2c73717b72cb65e0d0a5d9be67c56abdf8424141d20b560a63845e5f7bce6a4",
    "models/storms/expanded/seed3.pt": "5003eaf26206b163f243daebaa340913550d4935110f111f62c3516cfab10f07",
    "models/storms/expanded/seed4.pt": "ecf4156dac078b1633bb87eab67ff3d09d9a4ce5edafe1630a9f097f736c87ef",
    "models/storms/expanded/seed5.pt": "c5f5909c0fa83b66ca2ad60eb9a9a40db69d7345d03d25363151e853b5bdf8e2",
    "models/storms/expanded/seed6.pt": "57c7461dabcef83974058c6ec9585e6ce273c6a1ebe4eafc3a57d934c1fda6d1",
    "models/storms/expanded/seed7.pt": "e6786815f384c1dc8a5bbd78ce26cd39f00333e6986d32f17ec1ecef7aa9fd1b",
    "models/storms/expanded/seed8.pt": "462e529a9bb7cd807d65dc053e67bf9e3c2e5c98d275105eb163b096a92f98f3",
    "models/storms/expanded/seed9.pt": "3fab4e0d272e77808b9dc082dc4349048024878bb76ca44bedd4efce44d96e2c",
    "data/interim/hurdat2_atlantic.txt": "ee35819cba2b61ab0c2175601084def682813727ba209c2dab7d3af71d0bcb5c",
}
# ------------------------------------------------------------------------------------------------------------

REQUIRED = ["atcf_id", "storm_name", "sample_time", "wind_kt", "is_interpolated", "crop_path",
            "pred", "truth", "signed_error", "abs_error"]
SIGNS = np.array(list(itertools.product((1.0, -1.0), repeat=len(SEEDS))))
OUTPUTS = ["primary_tests.csv", "sensitivity.csv", "per_run.csv", "per_arm.csv", "paired_differences.csv",
           "per_storm.csv", "ensemble.csv", "val_trace.csv", "frozen_vs_control.csv", "report.txt"]
SELF_PATH = "scripts/analyze_n5.py"


class DesignError(SystemExit):
    pass


def epochs_for(frames, target=TARGET_STEPS, batch=BATCH):
    """Budget rule (10-13 night / 10-14 15:07): whole epochs whose step total is nearest the target."""
    b = math.ceil(frames / batch)
    best = min(range(1, 200), key=lambda e: (abs(e * b - target), e))
    return best, b


def design_constants(**override):
    """The recorded design as one dict. run() takes overrides only from the tests (main() exposes none)."""
    d = dict(h9=H9, h9_frames=H9_FRAMES, h9_pin=H9_PIN, val_ids=VAL_IDS, val_frames=VAL_FRAMES,
             ids={"control": CONTROL_IDS, "atlantic": ARM_IDS}, train_lists=TRAIN_LISTS,
             frames={"control": CONTROL_TRAIN_FRAMES, "atlantic": ARM_TRAIN_FRAMES},
             frame_list_sha={"control": None, "atlantic": FRAME_LIST_SHA256}, protected=PROTECTED)
    d.update(override)
    if d["frames"]["atlantic"] is None or d["frame_list_sha"]["atlantic"] is None:
        raise DesignError("ARM_TRAIN_FRAMES / FRAME_LIST_SHA256 are not filled in: the design is not final")
    d["epochs"] = {"control": CONTROL_EPOCHS, "atlantic": epochs_for(d["frames"]["atlantic"])[0]}
    d["batches"] = {a: math.ceil(d["frames"][a] / BATCH) for a in ARMS}
    return d


# ---- statistics --------------------------------------------------------------------------------------------
def signflip_p(d):
    """Two-sided exact paired sign-flip p over all 2^10 sign assignments."""
    d = np.asarray(d, float)
    if d.shape != (len(SEEDS),):
        raise DesignError(f"expected {len(SEEDS)} paired differences, got {d.shape}")
    return float((np.abs(SIGNS @ d) / len(d) >= abs(d.mean()) - 1e-12).mean())


def signflip_ci(d, alpha, n_grid=4001):
    """(1 - alpha) CI for the mean paired difference: the shifts g in [min d, max d] (where it must lie) that the
    exact sign-flip test on d - g does not reject at alpha, on a fixed grid."""
    d = np.asarray(d, float)
    keep = [g for g in np.linspace(d.min(), d.max(), n_grid) if signflip_p(d - g) >= alpha]
    return (float(min(keep)), float(max(keep))) if keep else (np.nan, np.nan)


def unpaired_p(a, b):
    """Two-sided exact permutation p for a difference of means over every split (descriptive use only)."""
    pool = np.concatenate([a, b]).astype(float)
    n, k = len(pool), len(b)
    idx = np.array(list(itertools.combinations(range(n), k)))
    mb = pool[idx].mean(1)
    ma = (pool.sum() - mb * k) / (n - k)
    return float((np.abs(mb - ma) >= abs(np.mean(b) - np.mean(a)) - 1e-12).mean())


# ---- input checks ------------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def h9_pin(d):
    s = "".join(f"{c}\t{t:.3f}\n" for c, t in zip(d["crop_path"], d["truth"]))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_text_strict(path):
    """Logs must be UTF-8 without BOM (run_n5.py writes them so). UTF-16 and BOMs are refused by name."""
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:4096]:
        raise DesignError(f"{path}: looks like UTF-16 (PowerShell redirection?); logs must be UTF-8 without BOM")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise DesignError(f"{path}: has a UTF-8 BOM; logs must be UTF-8 without BOM")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DesignError(f"{path}: not valid UTF-8 ({e})") from None


def load_csv(path, design):
    path = Path(path)
    if not path.exists():
        raise DesignError(f"missing predictions: {path}")
    d = pd.read_csv(path)
    miss = [c for c in REQUIRED if c not in d.columns]
    if miss:
        raise DesignError(f"{path}: missing columns {miss}")
    ids = d["atcf_id"].astype(str).str.upper()
    late = sorted({s for s in ids if not re.fullmatch(r"[A-Z]{2}\d{6}", s) or int(s[4:]) >= 2025})
    if late:
        raise DesignError(f"{path}: refuses 2025-or-later or malformed storm ids {late}")
    extra = sorted(set(ids) - set(design["h9"]))
    if extra:
        raise DesignError(f"{path}: refuses storms outside H9 {extra}")
    if ids.value_counts().to_dict() != design["h9"]:
        raise DesignError(f"{path}: H9 frame counts differ from the recorded {design['h9']}")
    if len(d) != design["h9_frames"] or d["crop_path"].duplicated().any():
        raise DesignError(f"{path}: {len(d)} rows or duplicate crop paths; expected {design['h9_frames']} frames")
    if not np.allclose(d["truth"], d["wind_kt"], atol=1e-3):
        raise DesignError(f"{path}: truth does not match wind_kt")
    if not np.allclose(d["abs_error"], (d["pred"] - d["truth"]).abs(), atol=1e-3):
        raise DesignError(f"{path}: abs_error does not match |pred - truth|")
    d["atcf_id"] = ids
    d = d.sort_values("crop_path").reset_index(drop=True)
    if h9_pin(d) != design["h9_pin"]:
        raise DesignError(f"{path}: H9 frames/labels differ from the pinned set (sha {h9_pin(d)[:12]}...)")
    return d


def ids_in(line):
    return re.findall(r"[A-Z]{2}\d{6}", line)


def parse_train_log(path, arm, seed, design):
    txt = read_text_strict(path)
    lines = txt.splitlines()
    ep = [(int(e), float(tr), float(va)) for e, tr, va in
          re.findall(r"^epoch (\d+): train ([\d.]+) kt\s+val ([\d.]+) kt", txt, flags=re.M)]
    saved_after = [int(m.group(1)) for i, ln in enumerate(lines) if ln.strip() == "saved" and i > 0
                   for m in [re.match(r"epoch (\d+):", lines[i - 1])] if m]
    m_seed = re.search(r"^seed: (\d+)\s+checkpoint: (\S+)", txt, flags=re.M)
    m_val = re.search(r"^val storms:\s+(\[.*\])", txt, flags=re.M)
    m_tr = re.search(r"^train storms: (\[.*\])", txt, flags=re.M)
    m_n = re.findall(r"^train samples: (\d+)\s+val samples: (\d+)", txt, flags=re.M)
    m_fl = re.search(r"^frame list: (\S+): (\d+) frames from (\d+) storms", txt, flags=re.M)
    flags = re.search(r"loss: (\w+)\s+circular_mask: (\w+)\s+rotate: (\w+)", txt)
    epochs = design["epochs"][arm]
    problems = []
    if not m_seed or int(m_seed.group(1)) != seed:
        problems.append("seed")
    if not m_seed or Path(m_seed.group(2)).as_posix() != f"models/n5_atlantic/{arm}/seed{seed}.pt":
        problems.append("checkpoint path")
    if not m_val or ids_in(m_val.group(1)) != design["val_ids"]:
        problems.append("validation storms")
    if not m_tr or sorted(ids_in(m_tr.group(1))) != design["ids"][arm]:
        problems.append("training storm set (not the recorded list)")
    if not m_n or int(m_n[-1][0]) != design["frames"][arm] or int(m_n[-1][1]) != design["val_frames"]:
        problems.append("train/val frame counts")
    if arm == "atlantic" and (not m_fl or int(m_fl.group(2)) != design["frames"][arm]):
        problems.append("frame list line")
    if arm == "control" and m_fl:
        problems.append("control used a frame list")
    if [e for e, _, _ in ep] != list(range(1, epochs + 1)):
        problems.append(f"epochs ({len(ep)} logged, {epochs} expected)")
    if not flags or flags.groups() != ("l1", "False", "False"):
        problems.append("loss/mask/rotation")
    if "TRAIN_FAILED" in txt or not re.search(r"^END ", txt, flags=re.M):
        problems.append("TRAIN_FAILED or unfinished")
    if not saved_after:
        problems.append("no 'saved' line")
    if problems:
        raise DesignError(f"{path}: does not match the recorded design: {', '.join(problems)}")
    return ep, saved_after[-1]


def check_eval_log(path, checkpoint, csv_rel, d):
    """Links an evaluate.py stdout log to its checkpoint AND to the CSV's content: the logged checkpoint, H9 storms,
    sample count, overall MAE (2 dp, evaluate.py:51) and the final 'wrote <out> (<n> rows)' line must all match."""
    txt = read_text_strict(path)
    m = re.search(r"^checkpoint: (\S+)", txt, flags=re.M)
    h = re.search(r"^held-out storms: (\[.*\])", txt, flags=re.M)
    n = re.search(r"^held-out samples: (\d+)", txt, flags=re.M)
    mae = re.search(r"^overall MAE:\s+([\d.]+) kt", txt, flags=re.M)
    wrote = re.search(r"^wrote (\S+)\s+\((\d+) rows\)", txt, flags=re.M)
    if not m or Path(m.group(1)).as_posix() != Path(checkpoint).as_posix():
        raise DesignError(f"{path}: evaluated checkpoint {m.group(1) if m else None} is not {checkpoint}")
    if not h or sorted(ids_in(h.group(1))) != sorted(H9):
        raise DesignError(f"{path}: held-out storms are not H9")
    if not wrote or Path(wrote.group(1)).as_posix() != Path(csv_rel).as_posix() or int(wrote.group(2)) != len(d):
        raise DesignError(f"{path}: 'wrote' line does not name {csv_rel} with {len(d)} rows")
    if not n or int(n.group(1)) != len(d):
        raise DesignError(f"{path}: held-out sample count does not match {csv_rel}")
    if not mae or abs(float(mae.group(1)) - round(float(d["abs_error"].mean()), 2)) > 0.0051:
        raise DesignError(f"{path}: logged overall MAE {mae.group(1) if mae else None} does not match the content of "
                          f"{csv_rel} ({d['abs_error'].mean():.3f}): CSV and log are not from the same evaluation")


def control_selection(reval, native_epoch):
    """(selected epoch, tie record or None) for the control (2026-10-19 amendment, condition (a)). The unrounded
    re-validation argmin must be train.py's pick (its last 'saved'). When it is not, but the two differ by less than
    TIE_KT, the tie rule fires: the checkpoint train.py saved (seed<s>.pt) is the one evaluated, never a re-ranked
    one. Otherwise the selection is the first minimum over the odd epochs (the matched grid)."""
    argmin = min(reval, key=lambda e: (reval[e], e))
    if argmin == native_epoch:
        return min(CONTROL_GRID, key=lambda e: (reval[e], e)), None
    delta = reval[native_epoch] - reval[argmin]
    if not abs(delta) < TIE_KT:
        raise DesignError(f"re-validation minimum is not train.py's native selection (epoch {native_epoch}): the "
                          "re-validation path does not reproduce training")
    return native_epoch, {"revalidated_argmin": argmin, "native_epoch": native_epoch, "delta_kt": delta}


def pinned_paths(arm, seed):
    """Every manifest path is fixed by arm and seed (design §8); nothing may point elsewhere."""
    p = {"train_log": f"logs/n5_atlantic/{arm}_seed{seed}.log",
         "native_checkpoint": f"models/n5_atlantic/{arm}/seed{seed}.pt",
         "h9_csv": f"results/n5_atlantic/{arm}_seed{seed}_h9.csv",
         "eval_log": f"logs/n5_atlantic/{arm}_seed{seed}_eval_h9.log"}
    if arm == "control":
        p.update(native_h9_csv=f"results/n5_atlantic/control_seed{seed}_native_h9.csv",
                 native_eval_log=f"logs/n5_atlantic/control_seed{seed}_native_eval_h9.log")
    else:
        p["selected_checkpoint"] = p["native_checkpoint"]
    return p


def odd_grid_near_ties(reval):
    """Odd epochs within TIE_KT of the odd-grid minimum (that minimum excluded). Recorded and reported; the
    pre-registered first minimum stands (2026-10-19 amendment, open for John)."""
    g = min(CONTROL_GRID, key=lambda e: (reval[e], e))
    return sorted(e for e in CONTROL_GRID if e != g and abs(reval[e] - reval[g]) < TIE_KT)


def single_commit(mans):
    """All runs must come from one commit (design §4)."""
    commits = {m.get("commit") for m in mans}
    if len(commits) != 1:
        raise DesignError(f"runs come from more than one commit: {sorted(map(str, commits))}")
    return commits.pop()


def check_attempts(man, root, where):
    """Every recorded abort (training attempt or H9 evaluation attempt) must exist and be complete."""
    for key in ("aborted_attempts", "eval_attempts"):
        for rel in man.get(key, []):
            p = Path(root) / rel
            if not p.is_file() or json.loads(read_text_strict(p)).get("status") != "complete":
                raise DesignError(f"{where}: {key} {rel} is missing or not complete")


def check_training(man, arm, seed, design, root, where):
    """Every training-side check, on a manifest or on run_n5's training record (the same keys). load_run applies it,
    and run_n5 applies it to all 20 records before the first H9 evaluation, so no training-side refusal can first
    appear after a look at H9 (hc-adversary item 16). Returns (epoch rows, native epoch, selected epoch)."""
    if (man.get("arm"), man.get("seed")) != (arm, seed):
        raise DesignError(f"{where}: manifest is for {man.get('arm')} seed {man.get('seed')}")
    pins = pinned_paths(arm, seed)
    for k in ("train_log", "native_checkpoint", "selected_checkpoint"):
        if k in pins and Path(str(man.get(k))).as_posix() != pins[k]:
            raise DesignError(f"{where}: {k} is {man.get(k)}, not the design's {pins[k]}")
    for k in ("train_log", "native_checkpoint", "selected_checkpoint"):
        p = root / man[k]
        if not p.exists() or sha256_file(p) != man.get(f"{k}_sha256"):
            raise DesignError(f"{where}: {k} {man[k]} is missing or its SHA-256 differs from the manifest")
    if man.get("train_list") != design["train_lists"][arm]:
        raise DesignError(f"{where}: train list {man.get('train_list')} is not {design['train_lists'][arm]}")
    if man.get("frame_list_sha256") != design["frame_list_sha"][arm]:
        raise DesignError(f"{where}: frame list SHA-256 differs from the recorded one")
    for when in ("protected_before", "protected_after"):
        got = man.get(when, {})
        bad = sorted(k for k, v in design["protected"].items() if got.get(k) != v)
        if bad:
            raise DesignError(f"{where}: {when} protected hashes differ: {bad}")
    check_attempts(man, root, where)
    ep, native_epoch = parse_train_log(root / man["train_log"], arm, seed, design)
    if arm == "control":
        # Primary: re-selected on validation only, over the odd epochs. The driver re-validates ALL 25 epochs'
        # saved weights (no held-out storm). Their unrounded values must round to the logged ones, and their
        # first minimum must be train.py's own pick (last 'saved', made on unrounded values): that proves the
        # re-validation path reproduces in-training validation end to end. Only then is the odd grid used.
        reval = {int(k): float(v) for k, v in man["val_mae_revalidated"].items()}
        logged = {e: va for e, _, va in ep}
        if sorted(reval) != list(range(1, CONTROL_EPOCHS + 1)) or                 any(abs(round(v, 2) - logged[e]) > 0.0051 for e, v in reval.items()):
            raise DesignError(f"{where}: re-validated values missing or inconsistent with the train log")
        try:
            sel, tie = control_selection(reval, native_epoch)
        except DesignError as e:
            raise DesignError(f"{where}: {e}") from None
        if "tie_rule" not in man:
            raise DesignError(f"{where}: no tie_rule field (2026-10-19 amendment); null when it did not fire")
        core = ("revalidated_argmin", "native_epoch")
        got = None if man["tie_rule"] is None else {k: man["tie_rule"].get(k) for k in core}
        if got != (None if tie is None else {k: tie[k] for k in core}):
            raise DesignError(f"{where}: tie_rule {man['tie_rule']} does not match the re-validated values ({tie})")
        if man.get("odd_grid_near_ties") != odd_grid_near_ties(reval):
            raise DesignError(f"{where}: odd_grid_near_ties {man.get('odd_grid_near_ties')} does not match the "
                              f"re-validated values ({odd_grid_near_ties(reval)})")
        want = man["native_checkpoint"] if tie else f"models/n5_atlantic/control/seed{seed}_epochs/epoch_{sel:02d}.pt"
        if man.get("selected_epoch") != sel or Path(man["selected_checkpoint"]).as_posix() != want:
            raise DesignError(f"{where}: selected checkpoint is not the validation minimum on the odd-epoch grid"
                              " (or, when the tie rule fired, train.py's own saved checkpoint)")
    else:
        sel = native_epoch
        if man.get("selected_epoch") != sel or man["selected_checkpoint"] != man["native_checkpoint"]:
            raise DesignError(f"{where}: atlantic selected checkpoint must be the native one (last 'saved')")
    return ep, native_epoch, sel


def load_run(manifest_path, arm, seed, design, root):
    man = json.loads(read_text_strict(manifest_path))
    if (man.get("arm"), man.get("seed")) != (arm, seed):
        raise DesignError(f"{manifest_path}: manifest is for {man.get('arm')} seed {man.get('seed')}")
    for k, want in pinned_paths(arm, seed).items():  # every pinned path first, the H9 ones included
        if Path(str(man.get(k))).as_posix() != want:
            raise DesignError(f"{manifest_path}: {k} is {man.get(k)}, not the design's {want}")
    ep, native_epoch, sel = check_training(man, arm, seed, design, root, manifest_path)
    keys = ["h9_csv", "eval_log"] + (["native_h9_csv", "native_eval_log"] if arm == "control" else [])
    for k in keys:
        p = root / man[k]
        if not p.exists() or sha256_file(p) != man.get(f"{k}_sha256"):
            raise DesignError(f"{manifest_path}: {k} {man[k]} is missing or its SHA-256 differs from the manifest")
    d = load_csv(root / man["h9_csv"], design)
    check_eval_log(root / man["eval_log"], man["selected_checkpoint"], man["h9_csv"], d)
    native = d
    if arm == "control":
        native = load_csv(root / man["native_h9_csv"], design)
        check_eval_log(root / man["native_eval_log"], man["native_checkpoint"], man["native_h9_csv"], native)
    return dict(man=man, ep=ep, sel=sel, native_epoch=native_epoch, d=d, native=native)


def verify_self(commit, root, git, check_path=True):
    """The executing file must be the launch commit's blob: clean working copy, no diff against the commit.
    check_path=False is the test hook (the draft lives under reports/); main() always checks the path."""
    me = Path(__file__).resolve()
    if check_path and me != (root / SELF_PATH).resolve():
        raise DesignError(f"run {SELF_PATH} from the repo, not {me}")
    if git(["status", "--porcelain", "--", SELF_PATH]).stdout.strip():
        raise DesignError(f"{SELF_PATH} has uncommitted changes")
    if git(["diff", "--quiet", commit, "--", SELF_PATH], check=False).returncode != 0:
        raise DesignError(f"{SELF_PATH} differs from the launch commit {commit}")
    blob = git(["show", f"{commit}:{SELF_PATH}"]).stdout
    blob_id = git(["rev-parse", f"{commit}:{SELF_PATH}"]).stdout.decode().strip()
    return hashlib.sha256(blob).hexdigest(), blob_id


def real_git(root):
    def git(args, check=True):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=check)
    return git


# ---- metrics -----------------------------------------------------------------------------------------------
def metrics(d):
    t, p = d["truth"].to_numpy(), d["pred"].to_numpy()
    e = p - t
    fix = ~d["is_interpolated"].astype(bool).to_numpy()
    hi = t > HIGH
    m = {"overall MAE": np.abs(e).mean(), "fixes-only MAE": np.abs(e)[fix].mean(), "overall bias": e.mean(),
         ">96 n": int(hi.sum()), ">96 MAE": np.abs(e)[hi].mean(), ">96 bias": e[hi].mean(),
         ">96 |bias|": abs(e[hi].mean()), "spearman >96": stats.spearmanr(p[hi], t[hi])[0]}
    for lo, up, name in BANDS:
        k = (t >= lo) & (t < up)
        m[f"MAE {name}"] = np.abs(e)[k].mean() if k.any() else np.nan
        m[f"bias {name}"] = e[k].mean() if k.any() else np.nan
    for tag, ids in (("H3", H3), ("added6", ADDED6)):
        k = d["atcf_id"].isin(ids).to_numpy()
        m[f"{tag} MAE"] = np.abs(e)[k].mean()
        m[f"{tag} >96 MAE"] = np.abs(e)[k & hi].mean() if (k & hi).any() else np.nan
        m[f"{tag} >96 bias"] = e[k & hi].mean() if (k & hi).any() else np.nan
    return m


def per_storm(d):
    t, e = d["truth"].to_numpy(), (d["pred"] - d["truth"]).to_numpy()
    out = {}
    for s in sorted(H9):
        k = (d["atcf_id"] == s).to_numpy()
        hk = k & (t > HIGH)
        out[f"{s} MAE"] = np.abs(e[k]).mean()
        enough = hk.sum() >= PER_STORM_MIN_HIGH
        out[f"{s} >96 n"] = int(hk.sum())
        out[f"{s} >96 MAE"] = np.abs(e[hk]).mean() if enough else np.nan
        out[f"{s} >96 bias"] = e[hk].mean() if enough else np.nan
    return out


def test_primary(name, a_vals, c_vals):
    """Returns the row for one primary. The bias branch is fixed in advance:
    both arm means < 0 -> d = bias_atl - bias_ctl, better is d > 0 (less under-prediction);
    otherwise (a sign crossing) -> d = |bias_atl| - |bias_ctl| per seed, better is d < 0."""
    a, c = np.asarray(a_vals[name], float), np.asarray(c_vals[name], float)
    if name == ">96 bias":
        if a.mean() < 0 and c.mean() < 0:
            d, better_sign, branch = a - c, +1, "both negative: d = bias_atl - bias_ctl, d > 0 better"
        else:
            d = np.abs(a) - np.abs(c)
            better_sign, branch = -1, "sign crossing: d = |bias_atl| - |bias_ctl|, d < 0 better"
    else:
        d, better_sign, branch = a - c, -1, "d = MAE_atl - MAE_ctl, d < 0 better"
    p = signflip_p(d)
    lo, hi = signflip_ci(d, ALPHA)
    s = SESOI[name]
    inside = -s < lo and hi < s
    direction = "better" if np.sign(d.mean()) == better_sign else "worse"
    if p < ALPHA and inside:
        verdict = f"SIGNIFICANT BUT BELOW SESOI ({direction}; real but smaller than {s} kt)"
    elif p < ALPHA:
        verdict = direction.upper()
    elif inside:
        verdict = f"NULL (informative; the 97.5% CI excludes effects of {s} kt or more)"
    else:
        verdict = f"NULL (inconclusive; effects of {s} kt are not excluded)"
    return {"metric": name, "branch": branch, "control_mean": c.mean(), "atlantic_mean": a.mean(),
            "mean_diff": d.mean(), "sd_diff": d.std(ddof=1), "ci97.5_lo": lo, "ci97.5_hi": hi, "p_signflip": p,
            "alpha": ALPHA, "sesoi": s, "n_seeds_better": int((np.sign(d) == better_sign).sum()),
            "significant": p < ALPHA, "direction": direction, "verdict": verdict}


def joint_verdict(rows, budget_limited):
    mae, bias = rows["overall MAE"], rows[">96 bias"]
    sig = {k: (r["direction"] if r["significant"] else None) for k, r in rows.items()}
    m, b = sig["overall MAE"], sig[">96 bias"]
    if m == "better" and b == "worse":
        v = "TRADE-OFF: overall MAE better, >96 kt bias worse"
    elif m == "worse" and b == "better":
        v = "REVERSE TRADE-OFF: overall MAE worse, >96 kt bias better"
    elif m and m == b:
        v = f"BOTH {m.upper()}: overall MAE and >96 kt bias"
    elif m or b:
        name, d = ("overall MAE", m) if m else (">96 kt bias", b)
        v = f"{name} {d}; the other primary shows no significant difference"
    else:
        v = "NO SIGNIFICANT DIFFERENCE on either primary"
    if budget_limited:
        v += (" | BUDGET-LIMITED: the under-training flag fired in >= 5/10 runs of an arm; a 'worse' result is read"
              " as 'worse at this budget', not as harm from the added storms; not rerun or rescued")
    return v


# ---- main analysis -----------------------------------------------------------------------------------------
def run(manifest_dir, out_dir, frozen_dir=None, root=None, design=None, git=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    design = design or design_constants()
    out_dir = Path(out_dir)
    clash = [f for f in OUTPUTS if (out_dir / f).exists()]
    if clash:
        raise DesignError(f"refusing to overwrite {clash} in {out_dir}")
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            p = Path(manifest_dir) / f"{arm}_seed{s}.json"
            if not p.exists():
                raise DesignError(f"missing run manifest {p}")
            runs[(arm, s)] = load_run(p, arm, s, design, root)
    commit = single_commit([r["man"] for r in runs.values()])
    blob_sha, blob_id = verify_self(commit, root, git or real_git(root), check_path=git is None)

    ref = runs[("control", 0)]["d"]
    vecs = {}
    for k, r in runs.items():
        for which in ("d", "native"):
            if not r[which]["crop_path"].equals(ref["crop_path"]):
                raise DesignError(f"{k}: H9 frame order differs")
        vecs[k] = r["d"]["pred"].to_numpy()
    allvecs = dict(vecs)
    for (arm, s), r in runs.items():
        if arm == "control":
            allvecs[(arm, s, "native")] = r["native"]["pred"].to_numpy()
    # When train.py's native pick is an odd epoch, the matched grid picks the same epoch, so a control seed's matched
    # and native vectors MUST be identical (same weights). That pair is exempt from the identity refusal and gets the
    # opposite check instead; when the picks differ, the pair is compared like every other (item 7b).
    same_pick = {s for (arm, s), r in runs.items() if arm == "control" and r["sel"] == r["native_epoch"]}
    for s in sorted(same_pick):
        if not np.allclose(allvecs[("control", s)], allvecs[("control", s, "native")], atol=1e-6):
            raise DesignError(f"control seed {s}: matched and native picks are both epoch {runs[('control', s)]['sel']} "
                              "but their H9 predictions differ; they must be identical")
    for (k1, v1), (k2, v2) in itertools.combinations(allvecs.items(), 2):
        if k1[0] == k2[0] == "control" and k1[1] == k2[1] and k1[1] in same_pick:
            continue
        if np.allclose(v1, v2, atol=1e-6):
            raise DesignError(f"identical prediction vectors for {k1} and {k2}: inputs are mislabelled or duplicated")

    rows, storm_rows, traces = [], [], []
    for (arm, s), r in runs.items():
        e_total = design["epochs"][arm]
        flag = r["sel"] * design["batches"][arm] > UNDERTRAIN_FRACTION * e_total * design["batches"][arm]
        rows.append({"arm": arm, "seed": s, "order": r["man"].get("order"), "selected_epoch": r["sel"],
                     "native_selected_epoch": r["native_epoch"], "epochs": e_total,
                     "steps_total": e_total * design["batches"][arm], "selected_step": r["sel"] * design["batches"][arm],
                     "undertrain_flag": bool(flag), **metrics(r["d"])})
        storm_rows.append({"arm": arm, "seed": s, **per_storm(r["d"])})
        traces += [{"arm": arm, "seed": s, "epoch": e, "train_mae": tr, "val_mae": va} for e, tr, va in r["ep"]]
    per_run = pd.DataFrame(rows)
    by = {arm: per_run[per_run.arm == arm].set_index("seed").sort_index() for arm in ARMS}
    under = {arm: int(by[arm].undertrain_flag.sum()) for arm in ARMS}
    budget_limited = any(v >= UNDERTRAIN_RUNS for v in under.values())

    prim = {k: test_primary(k, by["atlantic"], by["control"]) for k in PRIMARIES}
    joint = joint_verdict(prim, budget_limited)
    primary = pd.DataFrame(list(prim.values())).assign(joint_verdict=joint)

    native_ctl = pd.DataFrame([{"seed": s, **metrics(runs[("control", s)]["native"])} for s in SEEDS]).set_index("seed")
    sens = {k: test_primary(k, by["atlantic"], native_ctl) for k in PRIMARIES}
    sensitivity = pd.DataFrame(list(sens.values())).assign(
        analysis="sensitivity: control at its native selection (all 25 epochs)", joint_verdict=joint_verdict(sens, budget_limited))

    paired = []
    skip = {"seed", "arm", "order", "undertrain_flag", ">96 n", "epochs", "steps_total"}
    for k in [c for c in per_run.columns if c not in skip]:
        dd = (by["atlantic"][k] - by["control"][k]).to_numpy(float)
        if np.isnan(dd).any():
            continue
        role = "primary" if k in PRIMARIES else ("secondary" if k in SECONDARY else "descriptive")
        alpha = ALPHA if role == "primary" else SECONDARY_ALPHA
        lo, hi = signflip_ci(dd, alpha)
        paired.append({"metric": k, "role": role, "alpha_ci": alpha, "control_mean": by["control"][k].mean(),
                       "atlantic_mean": by["atlantic"][k].mean(), "mean_diff": dd.mean(), "sd_diff": dd.std(ddof=1),
                       "ci_lo": lo, "ci_hi": hi, "p_signflip": signflip_p(dd)})
    paired = pd.DataFrame(paired)

    per_arm = per_run.drop(columns=["seed", "order"]).groupby("arm").agg(["mean", "std", "min", "max"])
    per_arm.columns = [f"{m} {s}" for m, s in per_arm.columns]
    storms = pd.DataFrame(storm_rows).drop(columns=["seed"]).groupby("arm").mean().T

    truth = ref["truth"].to_numpy()
    ens = []
    for arm in ARMS:
        mp = np.mean([vecs[(arm, s)] for s in SEEDS], axis=0)
        ens.append({"arm": arm, **metrics(ref.assign(pred=mp, signed_error=mp - truth, abs_error=np.abs(mp - truth)))})
    ens = pd.DataFrame(ens)
    trace = pd.DataFrame(traces).groupby(["arm", "epoch"])[["train_mae", "val_mae"]].agg(["mean", "std"]).reset_index()
    trace.columns = [" ".join(x).strip() for x in trace.columns]

    frozen = None
    if frozen_dir is not None:
        fz = pd.DataFrame([{"seed": s, **metrics(load_csv(Path(frozen_dir) / f"expanded_seed{s}_h9.csv", design))}
                           for s in SEEDS])
        frozen = pd.DataFrame([{"metric": k, "frozen_expanded_mean": fz[k].mean(), "frozen_sd": fz[k].std(ddof=1),
                                "retrained_native_control_mean": native_ctl[k].mean(),
                                "retrained_native_sd": native_ctl[k].std(ddof=1),
                                "p_unpaired_exact": unpaired_p(fz[k].to_numpy(), native_ctl[k].to_numpy())}
                               for k in ("overall MAE", ">96 bias", ">96 MAE")])

    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(index=False, lineterminator="\n", float_format="%.6f")
    primary.to_csv(out_dir / "primary_tests.csv", **kw)
    sensitivity.to_csv(out_dir / "sensitivity.csv", **kw)
    per_run.to_csv(out_dir / "per_run.csv", **kw)
    per_arm.to_csv(out_dir / "per_arm.csv", lineterminator="\n", float_format="%.6f")
    paired.to_csv(out_dir / "paired_differences.csv", **kw)
    storms.to_csv(out_dir / "per_storm.csv", lineterminator="\n", float_format="%.6f")
    ens.to_csv(out_dir / "ensemble.csv", **kw)
    trace.to_csv(out_dir / "val_trace.csv", **kw)
    if frozen is not None:
        frozen.to_csv(out_dir / "frozen_vs_control.csv", **kw)

    ties = [s for (arm, s), r in runs.items() if arm == "control" and r["man"]["tie_rule"]]
    offgrid = [s for s in ties if runs[("control", s)]["sel"] % 2 == 0]
    near = {s: r["man"]["odd_grid_near_ties"] for (arm, s), r in runs.items()
            if arm == "control" and r["man"]["odd_grid_near_ties"]}
    attempts = {k: {f"{arm} seed {s}": r["man"][k] for (arm, s), r in runs.items() if r["man"].get(k)}
                for k in ("aborted_attempts", "eval_attempts")}

    def prow(r):
        return (f"  {r['metric']}: control {r['control_mean']:.3f}, atlantic {r['atlantic_mean']:.3f}, "
                f"d {r['mean_diff']:+.3f} (sd {r['sd_diff']:.3f}), 97.5% CI [{r['ci97.5_lo']:+.3f}, "
                f"{r['ci97.5_hi']:+.3f}], p {r['p_signflip']:.5f}, SESOI {r['sesoi']} -> {r['verdict']}"
                f"   [{r['branch']}]")
    lines = [
        "N5 Atlantic-only arm vs retrained control, H9, seeds 0-9 paired. Pre-registered analysis output, unedited.",
        f"analysis blob sha256 {blob_sha} (blob id {blob_id}) at commit {commit}",
        f"PRIMARIES (Bonferroni, alpha {ALPHA} each, two-sided exact sign-flip; control re-selected on odd epochs):",
        *[prow(r) for r in prim.values()],
        f"JOINT VERDICT: {joint}",
        f"Under-training flag (selected step in the last 15%): control {under['control']}/10, "
        f"atlantic {under['atlantic']}/10",
        f"Tie rule (2026-10-19 amendment, |d| < {TIE_KT} kt; train.py's own checkpoint evaluated): control seeds "
        f"{ties or 'none'}; off-grid primary (even epoch, the same checkpoint as the sensitivity): {offgrid or 'none'}",
        f"Odd-grid near ties (within {TIE_KT} kt of the odd-grid minimum; the first minimum is kept): "
        f"{near or 'none'}",
        f"Aborted training attempts (restarted with the same seed and order): {attempts['aborted_attempts'] or 'none'};"
        f" aborted H9 evaluation attempts (same checkpoint re-evaluated): {attempts['eval_attempts'] or 'none'}",
        "SENSITIVITY (pre-registered; control at its native 25-epoch selection; does not change the verdict):",
        *[prow(r) for r in sens.values()],
        "SECONDARY (alpha 0.05, uncorrected):",
        *[f"  {r.metric}: d {r.mean_diff:+.3f}, 95% CI [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}], p {r.p_signflip:.5f}"
          for r in paired[paired.role == "secondary"].itertuples()],
        "H9 SPLIT (descriptive): original three (Ian, Nicole, Claudette) vs the six added later:",
        *[f"  {r.metric}: control {r.control_mean:.3f}, atlantic {r.atlantic_mean:.3f}, d {r.mean_diff:+.3f}, "
          f"p {r.p_signflip:.5f}" for r in paired[paired.metric.isin(
              ["H3 MAE", "added6 MAE", "H3 >96 MAE", "added6 >96 MAE", "H3 >96 bias", "added6 >96 bias"])].itertuples()],
        "PER STORM (exploratory; >96 stats only with >= 10 frames > 96 kt): per_storm.csv. Sam (AL182021) was traced",
        "  per epoch on 10-14, so it has had per-epoch exposure; read its row with that in mind.",
        "DESCRIPTIVE (not tests): paired_differences.csv, per_arm.csv, ensemble.csv, val_trace.csv"
        + (", frozen_vs_control.csv (frozen members vs the NATIVE retrained control; unpaired, distribution-level)" if frozen is not None else ""),
        "Per-epoch traces are validation only (storms_val_expanded.txt); no held-out storm is evaluated per epoch.",
    ]
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return dict(primary=primary, sensitivity=sensitivity, paired=paired, per_run=per_run, frozen=frozen,
                joint=joint, blob_sha=blob_sha)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frozen-dir", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out_dir).resolve().as_posix().lower()
    if any(x in out for x in ("/models/", "/results/storms", "/results/ensemble", "/results/final_2025")) \
            or out.endswith("/results") or "/n5_atlantic/" not in out + "/":
        ap.error(f"--out-dir {args.out_dir}: must be a new directory under results/n5_atlantic/")
    run(args.manifest_dir, args.out_dir, frozen_dir=args.frozen_dir)


if __name__ == "__main__":
    sys.exit(main())
