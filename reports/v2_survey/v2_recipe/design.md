# v2 Atlantic recipe: design (r2 + tightening after hc-adversary item 31b; not approved; nothing run)

**Status:** design only.
- No training, evaluation or analysis runs before John's yes (his decision 2, 2026-10-21).
- The driver and its tests come in a separate launch PR, reviewed before anything runs (§9).
- r2 folds in hc-adversary item 31, which is round 2 of 2.

**Source of the decisions:** John's answers to `reports/v2_survey/f31/v2_implications.md` §5 (recorded in PR #30), verbatim:
"1. Yes: the 4× control recipe (99 epochs, native selection, 10 members) is the v2 Atlantic candidate, confirmed only on the 2026 clean test. 2. Yes: hc-survey drafts its design, hc-adversary reviews it (max two rounds unless there's a concrete defect), then it comes to me. No compute before my yes. 3. Yes: E3 uses the same fixed budget, with an Atlantic-only comparator at that budget (the v2 ensemble may serve). 4. Yes: start the 2026 clean-test protocol draft; the no-2026-imagery rule is unchanged. …"

## 0. What this is, and what it is not

**What it is:** the recipe, seeds, outputs, guards and freeze procedure for one new 10-member ensemble, "v2 Atlantic". It is trained with the frozen ensemble's configuration at F-31's 4× budget.

**What it is not:**
- **Not a replacement of the frozen ensemble.** `models/storms/expanded/seed0-9.pt` stays the frozen ensemble, with every I3/I6 pin and `check.py` check unchanged. Whether v2 replaces it is decided only after the clean test, under that test's protocol.
- **Not a test.** Nothing here estimates v2's skill on new storms. The only confirmation is the clean test (§4).
- **Not a new budget or rule choice.** The budget (4×) and the selection rule (native) are fixed by F-31 and John's decision 1. Nothing in this design is chosen on H9, H-EP or 2026 data.

**How H9 entered, stated plainly:**
- The recipe was chosen *because of* F-31's H9 result: 4× beat 1× on H9 in the control arm (P1 BETTER). That is why only an unspent set can confirm it.
- The v2 checkpoints themselves are read on H9 **zero** times under this design. That holds under transfer-gap option (a), the recommendation (§3). Option (b), if John picks it in E3's design, would add exactly one H9 evaluation of the frozen v2 ensemble.

## 1. The recipe (fixed)

| Item | Value | Source |
|---|---|---|
| Train list | `storms_train_expanded.txt`: 2081 frames after `excluded_crops.txt` | frozen-ensemble / F-31 control configuration |
| Validation list | `storms_val_expanded.txt`: 940 frames | the same |
| Model and optimiser | ResNet-18 IMAGENET1K_V1; Adam lr 1e-4 constant; batch 32; L1; unmasked; unrotated; no frame list; `dataset.py` preprocessing unchanged | `train.py` and `src/hurricane_intensity` at the pinned blobs |
| Budget | **99 epochs** = `analyze_n5.epochs_for(2081, 4 × 1640)`: 66 batches per epoch, 6534 steps | F-31 §2, 4× |
| Selection | **native:** `train.py`'s own `--out` checkpoint, the first strict minimum of unrounded validation MAE over epochs 1–99 | John, decision 1 |
| Members | **10**, seeds **30–39** (§2) | John, decision 1; seeds here |
| Ensemble | unweighted mean of the 10 members per frame (`ensemble_eval.py`); no weighting, member selection, recalibration or test-time augmentation | as the frozen ensemble and H-EP protocol §1 |

- **Smoothed was offered, not chosen.** The smoothed rule was offered (implications note §2). Native is what F-31's P1 tested at 4×.
- **The budget is tied to the frame count.** 99 is `epochs_for(2081, 6560)`. If the train-frame count changed, the epoch count would change. So the guards pin the inputs that set the count, and each run's log must show it (§6).

**Code pins: what "the same procedure as F-31's control run" rests on.**
- **`scripts/train.py`:** the driver refuses unless the SHA-256 of `git show <launch commit>:scripts/train.py` is `afceb4b19ea7d8c732c6f26b7d52e4b3b6d313d58153e2d0518887a3a6db231f` (blob `500bbc1e`). That is F-31's pin.
- **`src/hurricane_intensity/*.py`:** train.py imports `hurricane_intensity.dataset` and `.splits` (`train.py:10-11`), and `revalidate` imports the same. So every `src/hurricane_intensity/*.py` blob at the launch commit must equal its blob at **fc71b98** (F-31's launch commit L). The nine blob ids:

  | File | Blob id |
  |---|---|
  | `__init__` | `7a6fe639` |
  | `config` | `305d3c49` |
  | `crop` | `101be8fc` |
  | `dataset` | `458ed363` |
  | `goes_index` | `bba3b98f` |
  | `hurdat2` | `11dee801` |
  | `pipeline` | `58746c60` |
  | `projection` | `29cc9b6c` |
  | `splits` | `355f5f63` |

  The full SHA-256s go in the pre-launch entry, and no file may be added or removed.
- **`excluded_crops.txt`:** its blob at the launch commit must be `cc9dfcff` (SHA-256 `10200977…0cea`), the same as at fc71b98. This is the "excluded_crops pin" of guard 4. It is defined here, because it is not one of the 14 protected files and nothing else pins it.
- **Unchanged at 512593f and at d6f62b6:** `git diff fc71b98 <ref> -- src scripts/train.py excluded_crops.txt storms_train_expanded.txt storms_val_expanded.txt` is empty at both.
- **A change to any of these is not a pin update.** It needs a reviewed PR that re-argues the equivalence below. Then a new budget is chosen on validation only, in a new design.

**The learning-rate condition** (John, 2026-10-19 23:50, carried over):
- The 4× budget's meaning depends on `train.py`'s learning rate (and schedule) being independent of the total epoch count.
- At the pinned blob it is:
  - Adam at a constant 1e-4 (`train.py:88`), with no scheduler;
  - the epoch count is used only as the loop bound;
  - `--epoch-dir` only calls `torch.save` (no RNG).
- So a 99-epoch v2 run is the same procedure as F-31's control run read at 4×. **Any schedule change reopens the budget.**

**cuDNN is non-deterministic** (decisions.md 2026-10-13, "Groundwork"). So members are identified by hash (§4), never by "seed s of this recipe".

## 2. Decision 1: new seeds, not F-31's checkpoints

**Recommendation:** train 10 new members, seeds 30–39, in ascending order.

**The alternative, stated fairly:**
- F-31 holds 20 control 4× native checkpoints: `models/f31/control/seed<s>.pt`, seeds 10–29. Each has `native_same_weights: true`.
- A rule declared in advance (say "seeds 10–19") would pick 10 of them without compute.

**Why not reuse:**
1. **v2 would be an H9-favoured draw of the recipe.**
   - All 20 checkpoints have H9 evaluations. `results/f31/analysis/ensemble.csv` already prints both candidate halves at native 4×: seeds 10–19 at 8.029 kt, seeds 20–29 at 8.141 kt overall H9 MAE.
   - Any rule declared now is declared after those numbers, and seeds 10–19 is also the better half.
   - A clean test of a *fixed* ensemble stays unbiased for that ensemble however it was picked. The cost lies elsewhere: the ensemble tested would be one the H9 results favoured, so the clean-test result would likely overstate what a typical ensemble built from this recipe achieves. New seeds give a typical draw.
2. **They are outside every H-EP slot.**
   - The implications note (§3) and F-31 §0 say F-31's checkpoints fit no H-EP slot and are never scored on H-EP.
   - John's decision 3 lets the v2 ensemble be E3's comparator. Reusing F-31 checkpoints there would contradict both.
3. **Cost is small:** about 1.6 h of GPU time (§7).

**Why seeds 30–39:**
- Seeds 0–29 are all used by this configuration's data and initialisation path:
  - 0–9: the frozen ensemble and N5, at 25 epochs;
  - 10–29: F-31, at 99 epochs, H9-evaluated.
- `train.py` seeds the init and a dedicated shuffle generator from the seed alone (`train.py:49-55`). So with the same list, a reused seed would reproduce an H9-evaluated F-31 run, or a frozen member's first 25 epochs, up to cuDNN noise.
- Seeds 30–39 appear in no model directory, no log and no commit message (checked by hc-adversary, item 31).

**All ten runs are the ten members:**
- No run is dropped, replaced or reordered for its validation or any other score.
- A run is retrained only for a verification failure (§6), with the failed attempt kept and recorded.

## 3. Decision 2: no H9 evaluation of the v2 ensemble

**Recommendation: none.**
- Neither the ensemble nor any member is evaluated on H9 before the clean test, and this design plans none after it.
- That is under transfer-gap option (a) below. Option (b) would be one H9 evaluation of the frozen v2 ensemble inside the H-EP pass.

**Made structural, and tested behaviourally:**
- **No evaluation stage.** The driver never builds an `evaluate.py` or `ensemble_eval.py` command.
- **Output roots.** Its output-root check refuses any path outside the `v2_recipe` roots (§5). `results/v2_recipe/` holds only the members file.
- **Behavioural test** (launch PR): `run_v2_recipe.py launch` runs on a synthetic repo with a recording runner, as F-31's runner hooks allow. The test asserts that **every** subprocess the driver starts is `train.py` with exactly the §6 argument list for its seed, in seed order, and nothing else.
  - This catches calls made through imported code. A source grep would miss them, since `run_n5` itself defines `eval_cmd` and `evaluate_step`.
- **Name check, in addition:** the driver's source references none of `eval_cmd`, `evaluate_step`, `evaluate_phase`, `h9_*`, `HOLDOUT_LIST`, `evaluate.py`, `ensemble_eval.py` or `storms_holdout_`.

**Why none:**
- An H9 figure for v2 is not an estimate for new storms (H9 is reused; F-31 §4).
- It would invite comparison with F-31's 8.05 kt, or the frozen 8.57 kt, as if it were evidence.
- It cannot be used for anything: the recipe, seeds and rule are fixed, and choosing between ensembles on H9 is what §2 avoids.

**Where descriptive numbers may come from:**
- **F-31's committed control-arm 4× figures,** labelled "F-31, H9-derived, different seeds, not v2". Examples: the 20-member 4× ensemble at 8.05 kt, and P1's 0.99 kt member gain.
- **The v2 members' own training-side figures:** validation MAE at the pick, the pick epoch, the under-training flag count (§6) and the per-epoch validation traces. These are labelled "validation, the selection set, not an estimate".

**Where they may not come from:**
- any H9 read of a v2 checkpoint (except (b), if chosen);
- any H-EP read, except through the H-EP pass's own pre-registered outputs;
- any clean-test read before the clean-test protocol is committed and its single pass runs.

**A conflict for John (choice), settled in E3's design:**
- **The conflict:** the H-EP pass computes each evaluation's transfer gap from "E's H9 per-frame CSV" (protocol §2f, §6: `hep_report.py --h9`). v2 has none under this design. E3 has none either: §2f names H9 sources only for E1 and E2. So E3's design settles the transfer gap for E3 and its comparator together.
- **(a) Recommended:** no transfer-gap row for E3 or its comparator. The E3-vs-comparator contrast is direct on H-EP and needs no H9 figure. "Zero H9 reads of v2" holds.
- **(b) The alternative:** one H9 evaluation each of the frozen comparator (and of E3), inside the H-EP pass only. It runs after v2's freeze, E3's freeze and the clean-test protocol's commit, so it can shape none of them. It is descriptive only.
  - **Costs:** it breaks "zero"; `run_hep.py` would need an H9 stage; and it invites the comparison this section avoids.
- **Either way it is a protocol edit before H-EP gate 6.** The protocol is still a draft.

## 4. Decision 3: freeze, E3 comparator, and the timeline

**Freeze:** one step, before anything reads the members.
- **The members file.** After the 10 runs pass every training-only check (§6), the driver writes `results/v2_recipe/members_sha256.txt`.
  - Format: exactly `results/ensemble/members_sha256.txt`'s. One line per member, `<sha256> *models/v2_recipe/seed<s>.pt`, seeds 30–39 ascending, LF, with a final newline.
  - `ensemble_eval.py:42-47` compares the file's non-blank lines with `f"{sha256(c)} *{c.as_posix()}"`, built from `--checkpoints` as passed.
- **Every later use passes exactly:**
  - `--checkpoints models/v2_recipe/seed30.pt … models/v2_recipe/seed39.pt` (relative to the repo root, ascending);
  - `--expect-sha results/v2_recipe/members_sha256.txt`.
  - Any other path spelling or order refuses, so a mistake fails safe.
- **Never pass the members file as `--sha-out`.** `ensemble_eval.py:94` would rewrite it with `write_text`, as CRLF on Windows. Run-time hashes go to the evaluating protocol's own `members_sha256_at_run.txt`.
- **The freeze PR commits:**
  - the 10 training records;
  - the 10 train logs;
  - the members file;
  - a decisions.md entry, "v2 Atlantic ensemble frozen". It holds the members file's **blob** SHA-256 (from `git show` bytes), the 10 member SHA-256s, the launch commit and the train list.
- **After the freeze PR merges:**
  - the member files are read-only;
  - they are never retrained, fine-tuned, re-selected or replaced.
  - A defect found after the merge is reported in decisions.md. Any fix is a new ensemble under a new design, never an edit of v2.
- **The freeze precedes any H-EP evaluation and any clean-test build.**

**A defect found while the freeze PR is under review (guard 8's procedure):**
- **The command.** `run_v2_recipe.py unfreeze --reason "<text>"` is the only way to remove the members file. It refuses unless:
  - the members file is not on `origin/phase2-experiments` (the freeze PR is unmerged). This is decided only **after a `git fetch origin`**, and unfreeze refuses if the fetch fails;
  - neither HEAD's nor `origin/phase2-experiments`' `logs/decisions.md` already holds the "v2 Atlantic ensemble frozen" entry;
  - no H-EP or clean-test output exists.
- **What it does:**
  - it moves the file to `logs/v2_recipe/aborted/freeze_<n>/members_sha256.txt`, with an `abort.json` (reason, time, the file's SHA-256, status `complete`);
  - the freeze branch then drops the file in a new commit, never by a force-push.
- **Then:**
  - a run with a verification failure is retrained under the abort procedure (§6);
  - the freeze stage is run again.
- Both attempts are named in the freeze entry.

**As E3's comparator (H-EP protocol 3901f4cf, §1, row E3):**
- **What the row needs:** "checkpoint hashes (blob-hashed members file) and train list". The freeze entry supplies both.
- **E3's design, not this one, declares the comparator.** This design only makes v2 eligible.
- **No H-EP information is used:**
  - The recipe was fixed by F-31 (H9 and validation only) and John's decisions.
  - The driver refuses `storms_holdout_hep.txt`, any H-EP ID and `data/hep_sealed` (§6).
  - Protocol gate 6's rule ("E3's design may not use any H-EP label or pixel information") applies to the comparator through E3's design.
- **Disclosure, by line.** While drafting r1, I read protocol.md by section (§1, §4, §6). A search also displayed:
  - **l.84–88:** the §2a viewing-angle reference rows. These are training, validation and H9 rows, not H-EP rows.
  - **l.101:** one sentence on two H-EP storms' high-end viewing angles, which is H-EP intensity-derived and geometry information.
  - No v2 choice depends on them: every recipe item is fixed in §1 by F-31 and John.
  - The pre-launch entry repeats this disclosure, so John can weigh it when choosing who drafts E3's design.
  - From r2 on, protocol.md is read by line range only, never by search. Its l.37 marks the estimates table before §1 as H-EP label information.

**"The same fixed budget" for E3 (decision 3): two readings, for John.** This design does not decide which. E3's design settles it.

| Reading | E3's budget | Consequence |
|---|---|---|
| **Step-matched** (F-31's definition of 4×) | `epochs_for(E3's train frames, 4 × 1640)`, about 6560 steps, fewer epochs than 99 | E3 has more frames than 2081, so it is likely BUDGET-LIMITED, as F-31's atlantic arm was at 4× (12/20 vs control 5/20). E3 vs comparator would then mix the East Pacific effect with under-training: the asymmetry John's pre-look note named. |
| **Epoch-matched** | 99 epochs | E3 gets more steps than the comparator (6534), so the two are not "at that budget" in steps. |

- v2 is the same under either reading: 99 epochs, 6534 steps.

**Ordering with the clean-test protocol:** the clean-test protocol is committed **before** the H-EP pass, **unconditionally**, and it cannot be amended afterwards. The frozen ensemble is E1 in H-EP whatever happens, so H-EP results always precede the clean pass. So no H-EP result can shape it, whether or not v2 is E3's comparator (implications note §4; item 31, follow-up to finding 7; clean-test protocol draft r2 §8 step 2).

**Timeline against E3's deadline** (about end-January 2027; the date is set when the H-EP protocol is committed at its gate 6):
- v2 needs none of H-EP gates 1–7.

| Step | Duration |
|---|---|
| design review and John | about 1 week |
| launch PR, review and John | 1–2 weeks |
| training | about 1.6 h |
| freeze PR | 1–2 days |

- **Target:** frozen by mid-December 2026.
- **If the freeze misses E3's deadline,** v2 cannot be E3's comparator (the slot rule), but it remains the clean-test candidate.

## 5. Decision 4: outputs and line endings

**New paths (only these are written; the driver refuses any other):**

| Path | What it is | Committed? |
|---|---|---|
| `models/v2_recipe/seed<s>.pt` | the native pick; kept, and read-only after its record | no (`models/` is ignored) |
| `models/v2_recipe/seed<s>_epochs/epoch_NN.pt` | transient; each file is hashed into the record, then deleted | no |
| `logs/v2_recipe/seed<s>.log` | the driver's header, `train.py` stdout captured as bytes (CRLF on Windows), and the driver's footer | yes |
| `logs/v2_recipe/runs/seed<s>.train.json` | the per-run training record, LF | yes |
| `logs/v2_recipe/aborted/<n>/` | failed attempts and a withdrawn members file (§4), each with `abort.json` | yes. A moved checkpoint stays ignored: `models/` matches inside `aborted/` (`.gitignore:3`, checked in item 31). |
| `logs/v2_recipe/run_v2.lock` | the driver's own lock | no (gitignored) |
| `results/v2_recipe/members_sha256.txt` | the freeze file (§4) | yes |
| `reports/decisions/V2-*.md` | decision packages | yes. **Every committed v2 package must be named `V2-*.md`;** the freeze-PR check refuses any other new file under `reports/decisions/`. |

**Never written:**
- everything F-31 §9 lists as never written, plus all of `models/f31`, `logs/f31` (including `logs/f31/run_f31.lock`) and `results/f31`;
- `data/*`, `excluded_crops.txt`, `results/hep/*`, `best_model.pt`, the 14 protected files, `site/`, and John's untracked `scripts/export_*.py`.

**`.gitattributes` rows (added in the launch PR):**
```
# v2 recipe (reports/v2_survey/v2_recipe/design.md §5): the records pin the SHA-256 of each train log and member, and
# the members file's blob hash is recorded at the freeze; every committed output is stored and checked out byte-for-byte.
logs/v2_recipe/*.log -text
logs/v2_recipe/runs/*.json -text
logs/v2_recipe/aborted/** -text
results/v2_recipe/* -text
reports/decisions/V2-*.md -text
```

**Coverage and byte checks.** They cover **committed outputs only**; `models/` and the lock are not committed.
- **Launch PR test:** `git check-attr text` is `unset` for a synthetic example of each committed kind above (log, record, abort file, members file, `V2-*.md`).
- **Freeze-PR check:** it enumerates the files the PR actually commits (`git diff --name-only <base> HEAD`), not examples.
  - Every file under `logs/v2_recipe/`, `results/v2_recipe/` and `reports/decisions/V2-*.md` must have `text` unset, and its committed blob must equal its working-file bytes.
  - **`logs/decisions.md` is exempt from the bytes check.** It has no `-text` row, so under `core.autocrlf` its working copy is CRLF while its blob is LF. It is pinned by blob hash instead.
  - Any other committed path refuses.
- **The members file** is LF in its blob and in every checkout. So its file hash and blob hash agree, unlike E1's members file (protocol §1).

## 6. Decision 5: driver, launch guards and training-only checks

**The driver: `scripts/run_v2_recipe.py`** (launch PR).
- Stages: `launch` (guards, then train the 10 seeds in order, each re-validated and checked), `verify` (all 10 records), `freeze` (writes the members file), and `abort-run` / `unfreeze`.

**Imports: exactly these, each from a module hashed before it is executed.**
- **How each module is loaded.** It is read as bytes and hashed with CRLF read as LF; the hash must be its pin. A fresh module object gets `__file__` set to the file's path **before** execution (`run_n5` derives its `ROOT` from `__file__`). The module is then compiled and executed from exactly those bytes, with no re-read and no `.pyc`. That is `run_f31.load_pinned_run_n5`'s method (`run_f31.py:122-138`), written into the v2 driver.

| Module | Pin (SHA-256) | Names used |
|---|---|---|
| `scripts/run_n5.py` | `e8f13e06…` (blob `2ebd8cea`) | `sha256_file`, `protected_now`, `same_weights`, `write_json_new`, `write_json_replace`, `run_step`, `child_env`, `write_line`, `now_iso`, `nvidia_snapshot`, `cuda_available`, `child_package_path`, `real_git`, `revalidate` (validation list only, `run_n5.py:188-217`), `doc_path` |
| `scripts/analyze_n5.py` | `01306415…` (blob `679e2169`) | `epochs_for`, `TIE_KT`, `UNDERTRAIN_FRACTION`, `read_text_strict`, `ids_in`, `PROTECTED`, `VAL_IDS`, `VAL_FRAMES`, `CONTROL_IDS`, `CONTROL_TRAIN_FRAMES`, `DesignError` |

- **No other import.** `run_f31` and `analyze_f31` are **not** imported:
  - their lock, record, abort, path and log-parser helpers are bound to F-31's paths;
  - `analyze_f31` executes `analyze_n5` on import without a hash check.
- **Written in the v2 driver, on v2 paths:**
  - its lock (`logs/v2_recipe/run_v2.lock`);
  - `run_paths` and the record writer and verifier;
  - the abort helpers (`logs/v2_recipe/aborted/`);
  - the train-log parser, expecting `models/v2_recipe/seed<s>.pt` and the v2 driver's own header and END footer;
  - the selection functions `last_saved`, `native_pick` and `flagged`. These are copied from `analyze_f31.py:153-211` (pure, 20 lines). A launch-PR test checks that they agree with `analyze_f31`'s on the 20 F-31 control records' re-validation tables. The test reads those committed records and imports `analyze_f31` in the test only, after hashing it and `analyze_n5`. None of the 20 records has a non-null `tie_rule`, so the records exercise only the common path. **The test adds synthetic validation tables** for the remaining paths, run through both implementations, which must agree:
    - a table whose re-validated first minimum differs from the last `saved` epoch by less than `TIE_KT`. The tie branch must return `train.py`'s pick, with the tie record;
    - a table where they differ by `TIE_KT` or more. Both must refuse.
- **`revalidate` imports `hurricane_intensity`** from the checkout's `src`. That package is covered by the src pin (§1; guard 2).

**Launch guards (all before the first run; each refuses):**
1. **Tree:** the tracked tree is clean, and the driver equals its HEAD blob. HEAD is on `phase2-experiments`' first-parent chain and pushed.
2. **Code pins:** HEAD's `scripts/train.py` is `afceb4b1…`, and every `src/hurricane_intensity/*.py` blob equals fc71b98's (§1), with no file added or removed. `run_n5.py` and `analyze_n5.py` match their pins before execution.
3. **Protected files:** the 14 protected files match the COORDINATOR.md block, before and after every run.
4. **Data pins:**
   - HEAD's `excluded_crops.txt` blob is `cc9dfcff` (§1);
   - `storms_train_expanded.txt` and `storms_val_expanded.txt` equal their fc71b98 blobs;
   - `epochs_for(2081, 6560) == (99, 66)`.
5. **Lists:** the train and validation lists intersect no held-out or spent set: H9, the 2025 test, H-EP (`storms_holdout_hep.txt`), and any 2026 ID (I5 forbidden sets and `test_no_2026_id_in_any_list`).
6. **Paths:** every output path is under the `v2_recipe` roots. No `models/v2_recipe/seed<s>.pt` exists unless its record verifies (resume).
7. **No held-out argument:** nothing held-out appears in any command (§3's behavioural test).
8. **Not frozen:** no `results/v2_recipe/members_sha256.txt` exists, tracked or on disk. Once frozen, `launch` refuses; only `unfreeze` (§4) can reopen it, and only before the freeze PR merges.
9. **Run environment** (as F-31's `common_guards`):
   - the run root is the main checkout, with a `.git` directory, not a worktree;
   - a child Python imports `hurricane_intensity` from `<root>/src` (`child_package_path`);
   - CUDA is available.
   - **Why:** the `.venv` editable install's `.pth` points at the main checkout's `src`. A launch from a worktree would pass guard 2 on the worktree's blobs while training on the main checkout's `src`.
   - The guard tests cover each refusal: a worktree root, a child import from elsewhere, and no CUDA.

**Per run (seed s, ascending 30–39):**
```
[PY, -u, scripts/train.py, --train-list, storms_train_expanded.txt, --val-list, storms_val_expanded.txt,
 --epochs, 99, --seed, s, --loss, l1, --out, models/v2_recipe/seed<s>.pt,
 --epoch-dir, models/v2_recipe/seed<s>_epochs]                              -> logs/v2_recipe/seed<s>.log
```
- There is no `--frame-list`, `--circular-mask` or `--rotate`.
- After training, every epoch file is re-validated on validation only (`revalidate`, about 2 min).

**Training-only checks.** The record must pass all of them. They are modelled on `analyze_f31.check_training` and `parse_train_log`, written for v2.
- **From the train log:**
  - `seed: <s>  checkpoint: models/v2_recipe/seed<s>.pt`;
  - `val storms:` equals `VAL_IDS`, and the `train storms:` set equals `CONTROL_IDS`;
  - **`train samples: 2081  val samples: 940`.** This is the only per-run proof that this run saw 2081 frames, i.e. 66 batches × 99 = 6534 steps;
  - `loss: l1  circular_mask: False  rotate: False`;
  - no `frame list:` line;
  - epoch lines exactly 1, 2, …, 99, in order;
  - at least one `saved` line;
  - no `TRAIN_FAILED`, and the driver's `END` footer present.
- **Record identity:**
  - arm `control`, seed s;
  - `order` equals its position in the seed schedule (30 → 1, …, 39 → 10);
  - train and validation lists as designed;
  - `frame_list_sha256` is `None`;
  - epochs 99;
  - `commit` is the launch commit;
  - `train_py_sha256` is the pin;
  - protected-before and protected-after hashes match.
- **Re-validation:** `val_mae_revalidated` covers epochs 1–99 and agrees with the log's rounded values to 0.0051 kt.
- **Native pick and tie:**
  - `native_pick(reval, saved, 99)`: the re-validated first minimum must be the last `saved` epoch, or differ from it by less than `TIE_KT` (N5's amendment (a)).
  - The record's `tie_rule` (`None`, or `{revalidated_argmin, saved_epoch, delta_kt}`) must equal the recomputed one.
- **Same weights:** the pick's epoch file is **tensor-equal** to `seed<s>.pt` (`run_n5.same_weights`: the same state_dict keys, every tensor `torch.equal`), as F-31 checked (`native_same_weights: true`). The check runs before the epoch files are deleted. r2 said "byte-identical", which this function does not test.
- **Files:**
  - `files_sha256` names exactly the train log and `seed<s>.pt`, with matching hashes;
  - `deleted_epoch_sha256` hashes all 99 epoch files, and none remains on disk.
- **Attempts:** `aborted_attempts` lists every earlier attempt for the seed. Each listed `abort.json` exists with status `complete`, and no abort directory for the seed is unlisted (as `check_attempts`).
- **Under-training flag:** pick > 0.85 × 99, i.e. epoch ≥ 85. It is computed and reported per run and as a count out of 10.
  - **Descriptive only.** If it fires in ≥ 5 of 10 runs, the freeze entry says the recipe is still budget-limited on validation at 6534 steps for these seeds.
  - It changes nothing. F-31's control was 5/20 at 4×.

**Disk:**
- **Kept:** about 450 MB (10 × 44.8 MB).
- **Transient:** at most 99 epoch files, about 4.4 GB, one run at a time (John's F-31 choice 4).
- **The alternative,** keeping all epoch files, is about 44 GB; not recommended.

## 7. Compute

- **Per run:** training about 7–9 min (F-31 control at 4×: median 7.2 min, about 9 min start to start), plus re-validation about 2 min.
- **10 runs:** about 1.6 h; plan on 1.5–2 h.
- **No build, no download,** and no H9, H-EP or 2026 evaluation.

## 8. Known limits

- **The recipe was selected on H9** (§0): only the clean test confirms it.
- **The clean test compares v2 with the frozen ensemble,** which differs in budget, code version and seeds. It answers "is v2 better than what is frozen", not "is 4× better than 1× in isolation". F-31 answered the latter on H9.
- **Validation is blind at Cat 5** (no frames ≥ 137 kt), as in F-31.
- **Ten members:** the ensemble's member spread is estimated from 10, as for the frozen ensemble.

## 9. Prerequisites, in order (none done)

0. **Design approval:** this design approved by John, after hc-adversary (two rounds).
1. **Launch PR:**
   - `scripts/run_v2_recipe.py` and its tests: synthetic records, every guard above (guard 9 included), the behavioural no-evaluation test, the selection-agreement test with its synthetic tie and refusal tables, the coverage test, and the unfreeze procedure with its fetch-or-refuse rule;
   - the `.gitattributes` rows and the `.gitignore` row for the lock.
   - It goes to hc-adversary and John before anything runs.
2. **Checks:** `check.py` PASS (FAST and `--full`) on the launch commit.
3. **decisions.md pre-launch entry:**
   - this design's blob hash;
   - the driver's blob hash;
   - the `run_n5`, `analyze_n5`, `train.py`, `src/hurricane_intensity/*.py` and `excluded_crops.txt` pins;
   - the seeds and order;
   - the protocol.md disclosure by line (§4).
4. **Training** (`run_v2_recipe.py launch`), after John's yes. No H9, H-EP or 2026 read.
5. **Freeze PR:** the records, logs, members file and the freeze entry (§4). From then on, v2 is frozen.
6. **The clean-test protocol** is committed before any clean-test build, and before the H-EP pass, unconditionally (§4). Only then does v2 meet held-out data.

## Files in this draft

- `design.md`: this file. No script, no data.
