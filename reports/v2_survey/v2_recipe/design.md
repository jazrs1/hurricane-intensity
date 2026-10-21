# v2 Atlantic recipe: design (r1, draft for hc-adversary; not approved; nothing run)

**Status:** design only. No training, evaluation or analysis runs before John's yes (his decision 2, 2026-10-21). The driver and its tests come in a separate launch PR, reviewed before anything runs (§9).

**Source of the decisions:** John's answers to `reports/v2_survey/f31/v2_implications.md` §5 (recorded in PR #30), verbatim:
"1. Yes: the 4× control recipe (99 epochs, native selection, 10 members) is the v2 Atlantic candidate, confirmed only on the 2026 clean test. 2. Yes: hc-survey drafts its design, hc-adversary reviews it (max two rounds unless there's a concrete defect), then it comes to me. No compute before my yes. 3. Yes: E3 uses the same fixed budget, with an Atlantic-only comparator at that budget (the v2 ensemble may serve). 4. Yes: start the 2026 clean-test protocol draft; the no-2026-imagery rule is unchanged. …"

## 0. What this is, and what it is not

**What it is.** The recipe, seeds, outputs, guards and freeze procedure for one new 10-member ensemble, "v2 Atlantic", trained with the frozen ensemble's configuration at F-31's 4× budget.

**What it is not:**
- **Not a replacement of the frozen ensemble.** `models/storms/expanded/seed0-9.pt` stays the frozen ensemble, with every I3/I6 pin and the `check.py` checks unchanged. Whether v2 replaces it is decided only after the 2026 clean test, under that test's protocol.
- **Not a test.** Nothing here estimates v2's skill on new storms. The only confirmation is the 2026 clean test (§4).
- **Not a new budget or rule choice.** The budget (4×) and the selection rule (native) are fixed by F-31 and John's decision 1. Nothing in this design is chosen on H9, H-EP or 2026 data.

**How H9 entered, stated plainly.** The recipe was chosen *because of* F-31's H9 result: 4× beat 1× on H9 in the control arm (P1 BETTER). So the recipe carries H9 selection, and that is why only an unspent set can confirm it. The v2 checkpoints themselves are read on H9 **zero** times under this design (§3).

## 1. The recipe (fixed)

| Item | Value | Source |
|---|---|---|
| Train list | `storms_train_expanded.txt` (2081 frames after `excluded_crops.txt`) | frozen-ensemble / F-31 control configuration |
| Validation list | `storms_val_expanded.txt` (940 frames) | the same |
| Model and optimiser | ResNet-18 IMAGENET1K_V1, Adam lr 1e-4 constant, batch 32, L1 loss, unmasked, unrotated, `dataset.py` preprocessing unchanged | `train.py` at the pinned blob |
| Budget | **99 epochs** (6534 steps, 66 batches per epoch), = `analyze_n5.epochs_for(2081, 4 × 1640)` | F-31 §2, 4× |
| Selection | **native:** `train.py`'s own `--out` checkpoint, the first strict minimum of unrounded validation MAE over epochs 1–99 | John, decision 1 |
| Members | **10**, seeds **30–39** (§2) | John, decision 1; seeds here |
| Ensemble | unweighted mean of the 10 members per frame (`ensemble_eval.py`), no weighting, member selection, recalibration or test-time augmentation | as the frozen ensemble and H-EP protocol §1 |

**The smoothed rule was offered** (implications note §2) **and not chosen.** Native is what F-31's P1 tested at 4×.

**The budget is tied to the frame count.** 99 is `epochs_for(2081, 6560)`. If the train-frame count changes (a change to `excluded_crops.txt` or the list), the epoch count would change, so the launch guards pin both the 2081 count and the 99 (§6).

**train.py pin and the learning-rate condition** (John, 2026-10-19 23:50, carried over):
- The driver refuses unless the SHA-256 of `git show <launch commit>:scripts/train.py` is `afceb4b19ea7d8c732c6f26b7d52e4b3b6d313d58153e2d0518887a3a6db231f` (blob `500bbc1e`), F-31's pin, unchanged at 512593f.
- **The 4× budget's meaning depends on `train.py`'s learning rate (and schedule) being independent of the total epoch count.** At the pinned blob it is: Adam at a constant 1e-4 (`train.py:88`), no scheduler, and nothing in the loop depends on `--epochs` except the loop count. So a 99-epoch v2 run is the same procedure as F-31's control run read at 4×.
- **Any schedule change reopens the budget.** A new `train.py` blob is not a pin update: it needs a reviewed PR that re-checks this property, and a budget chosen on validation only in a new design.
- `--epoch-dir` only writes files (`train.py`, after each epoch's validation). It does not change training, so writing epoch files (§5) leaves the procedure equal to `train.py --out` alone.

**cuDNN is non-deterministic** (decisions.md 2026-10-13, "Groundwork"), so no run reproduces bit-for-bit. That is why the members are identified by hash (§4), never by "seed s of this recipe".

## 2. Decision 1: new seeds, not F-31's checkpoints

**Recommendation: train 10 new members, seeds 30–39, in ascending order.**

**The alternative, stated fairly.** F-31 already holds 20 control 4× native checkpoints: `models/f31/control/seed<s>.pt`, seeds 10–29. Each record has `native_same_weights: true`, so `seed<s>.pt` is the native 4× pick. A pre-declared rule (say "seeds 10–19") would pick 10 of them without compute.

**Why not reuse:**
1. **Their H9 scores are known, and the choice would be made after seeing them.** All 20 have H9 evaluations at the native 4× slot. F-31's `results/f31/analysis/ensemble.csv` already prints the two candidate ensembles: seeds 10–19 at 8.029 kt and seeds 20–29 at 8.141 kt overall H9 MAE. Any rule declared now (a seed range, first-in-schedule, and so on) is declared after those numbers exist. No wording makes it blind, and seeds 10–19 is also the better half. This is selection optimism by construction, however innocent the rule looks.
2. **They are outside every H-EP slot.** The implications note (§3) and F-31 §0 say F-31's checkpoints fit no H-EP slot and are never scored on H-EP. John's decision 3 lets the v2 ensemble be E3's comparator. Reusing F-31 checkpoints there would contradict both statements. New members avoid that.
3. **Cost is small:** about 1.6 h of GPU time (§7).

**Why seeds 30–39:**
- Seeds 0–29 are all used by recipes that share this configuration's data and initialisation path:
  - 0–9: frozen ensemble and N5 (25 epochs);
  - 10–29: F-31 (99 epochs, H9-evaluated).
- **A reused seed would correlate the members:** with the same seed, a v2 member's init and shuffle order would match an H9-evaluated F-31 run, or a frozen member's first 25 epochs. That correlation is unwanted in a v2-vs-frozen comparison on 2026. cuDNN non-determinism only partly breaks it.
- Seeds 30–39 appear in no model directory (checked: `models/*` holds seeds 0–29 only).

**The ten members are all ten runs.** No run is dropped, replaced or reordered for its validation or any other score. A run is retrained only for a verification failure (§6), with the failed attempt kept and recorded.

## 3. Decision 2: no H9 evaluation of the v2 ensemble

**Recommendation: none.** Neither the ensemble nor any member is evaluated on H9 at any point before the 2026 clean test, and this design plans none after it.

**Made structural, not promised:**
- The driver has no evaluation stage. It never calls `evaluate.py` or `ensemble_eval.py`.
- Its output-root check refuses any path except the three `v2_recipe` roots (§5). `results/v2_recipe/` holds only the members file.
- The launch PR's tests assert that the driver's source contains no `evaluate.py` / `ensemble_eval.py` / `storms_holdout_` argument.

**Why none:**
- An H9 figure for v2 is not an estimate for new storms (H9 is reused; F-31 §4).
- It invites comparison with F-31's 8.05 kt, or with the frozen 8.57 kt, as if it were evidence. It is not.
- It cannot be used for anything: the recipe, seeds and rule are fixed, and selection between ensembles on H9 is exactly what §2 avoids.

**Where descriptive numbers may come from:**
- **F-31's committed control-arm 4× figures**, labelled "F-31, H9-derived, different seeds, not v2". Examples are the 20-member 4× ensemble at 8.05 kt and P1's 0.99 kt member gain.
- **The v2 members' own training-side figures:** validation MAE at the pick, the pick epoch, the under-training flag count (§6), and per-epoch validation traces. These are labelled "validation, the selection set, not an estimate".

**Where they may not come from:**
- any H9 read of a v2 checkpoint;
- any H-EP read, before or after the H-EP pass, except through that pass's own pre-registered outputs;
- any 2026 read before the 2026 protocol is committed and its single pass runs.

**A conflict for John (choice):**
- **The conflict:** the H-EP protocol's pass computes each evaluation's *transfer gap* from "E's H9 per-frame CSV" (protocol §2f and §6: `hep_report.py --h9`). The v2 ensemble has no H9 CSV under this design. If v2 is E3's comparator, one of these must give:
  - **(a) Recommended:** the comparator row has no transfer gap. E3's design declares that its comparison with the comparator is the direct H-EP contrast (E3 vs comparator, both on H-EP), which needs no H9 figure. This is an E3-design / protocol item, not a v2 run.
  - **(b)** One H9 evaluation of the frozen v2 ensemble, inside the H-EP pass and nowhere else. It would be descriptive only. It would run after the v2 members are frozen (§4) *and* after the 2026 protocol is committed, so it can shape neither.
- (a) keeps "zero H9 reads" true. (b) gives the comparator the same transfer-gap row as E1/E2. Either is John's choice when E3's design is drafted; nothing here depends on it.

## 4. Decision 3: freeze, E3 comparator, and the timeline

**Freeze (one step, before anything reads the members):**
- After the 10 runs pass every training-only check (§6), the driver writes `results/v2_recipe/members_sha256.txt`.
  - Format: exactly `results/ensemble/members_sha256.txt`'s: one line per member, `<sha256> *models/v2_recipe/seed<s>.pt`, seeds ascending, LF, final newline.
  - That is the line format `ensemble_eval.py --expect-sha` compares (`ensemble_eval.py:34-47`), so the same file serves the 2026 pass and the H-EP pass unchanged.
- **The freeze PR** commits:
  - the 10 training records;
  - the 10 train logs;
  - the members file;
  - a decisions.md entry "v2 Atlantic ensemble frozen". It holds the members file's **blob** SHA-256 (computed from `git show` bytes, as the H-EP protocol does for E1), the 10 member SHA-256s, the launch commit, and the train list.
- **After the freeze:**
  - the member files are read-only;
  - they are never retrained, fine-tuned, re-selected or replaced;
  - any later use passes `--expect-sha results/v2_recipe/members_sha256.txt`.
- The freeze precedes **any** H-EP evaluation and **any** 2026 build.

**As E3's comparator (H-EP protocol 3901f4cf, §1, row E3):**
- **What the row needs:** "checkpoint hashes (blob-hashed members file) and train list". The freeze entry supplies both: the `members_sha256.txt` blob hash and `storms_train_expanded.txt`.
- **E3's design, not this one, declares the comparator.** This design only makes v2 eligible.
- **No H-EP information is used.** The recipe was fixed by F-31 (H9 and validation only) and John's decisions. The driver refuses `storms_holdout_hep.txt`, any H-EP ID, and `data/hep_sealed` (§6). Protocol gate 6's rule ("E3's design may not use any H-EP label or pixel information") applies to the comparator through E3's design, and this design satisfies it.
  - **Disclosure:** while drafting, protocol.md was read for its slot table, gates and paths (§1, §4, §6). A search also displayed its §2a viewing-angle reference rows and one sentence on two H-EP storms' high-end viewing angles. No v2 choice depends on them: every recipe item is fixed in §1 by F-31 and John.
- **"The same fixed budget" for E3 (decision 3):** E3 applies the same rule, `epochs_for(E3's train frames, 4 × 1640)`. That is the step-matched 4× budget, not "99 epochs", since E3's frame count differs. The v2 comparator's 6534 steps and E3's step-matched epochs then share a budget in steps, as F-31's arms did.
- **Ordering with the 2026 protocol:** if v2 is E3's comparator, the 2026 clean-test protocol is committed **before** the H-EP pass. Then no H-EP result can shape it (implications note §4).

**Timeline against E3's deadline** (about end-January 2027; the date is set when the H-EP protocol is committed at its gate 6):
- v2 needs none of H-EP gates 1–7 (antimeridian fix, EP cache, builds). It can train as soon as John approves this design and the launch PR.
- **Expected path:**

  | Step | Duration |
  |---|---|
  | design review (≤ 2 rounds) and John | about 1 week |
  | launch PR (driver, tests, `.gitattributes` rows), review and John | 1–2 weeks |
  | training | about 1.6 h |
  | freeze PR | 1–2 days |

- **Target:** frozen by mid-December 2026, about six weeks before E3's deadline.
- If the freeze has not happened by E3's deadline, v2 cannot be E3's comparator (the slot rule), but it remains the 2026 candidate.

## 5. Decision 4: outputs and line endings

**New paths (only these are written; the driver refuses any other):**
- `models/v2_recipe/seed<s>.pt`: the native pick. It is kept, and read-only after its record.
- `models/v2_recipe/seed<s>_epochs/epoch_NN.pt`: transient, for re-validation (§6). Each is hashed into the record, then deleted.
- `logs/v2_recipe/seed<s>.log`: `train.py`'s stdout, captured as bytes. It is CRLF on Windows.
- `logs/v2_recipe/runs/seed<s>.train.json`: the per-run training record, written by the driver as LF.
- `logs/v2_recipe/aborted/<seed>_<n>/`: failed attempts, kept.
- `logs/v2_recipe/run_v2.lock`: the run lock; gitignored, never committed.
- `results/v2_recipe/members_sha256.txt`: the freeze file (§4).
- `reports/decisions/V2-*.md`: decision packages, if any.

**Never written:**
- everything F-31 §9 lists as never written, plus all of `models/f31`, `logs/f31` and `results/f31`;
- `data/*`, `excluded_crops.txt`, `results/hep/*`, `best_model.pt`, the 14 protected files, `site/`, and John's untracked `scripts/export_*.py`.

**`.gitattributes` rows (added in the launch PR, as F-31's were):**
```
# v2 recipe (reports/v2_survey/v2_recipe/design.md §5): the records pin the SHA-256 of each train log and member, and
# the members file's blob hash is recorded at the freeze; every committed output is stored and checked out byte-for-byte.
logs/v2_recipe/*.log -text
logs/v2_recipe/runs/*.json -text
logs/v2_recipe/aborted/** -text
results/v2_recipe/* -text
reports/decisions/V2-*.md -text
```
- **A coverage test,** so no committed output can miss a rule (the gap PR #29 hit with `reports/decisions/F31-result.md`, which no `-text` row covered):
  - the launch PR's tests run `git check-attr text` on one example of every output path above and require `unset`;
  - the freeze PR's check repeats it on every file it commits;
  - every committed blob must equal its file's bytes.
- `members_sha256.txt` is LF in its blob and in every checkout, so its file hash and blob hash agree. That avoids the E1 members file's working-copy/blob split (protocol §1).

## 6. Decision 5: launch guards and training-only checks

**The driver: `scripts/run_v2_recipe.py`** (launch PR).
- Phases: launch → train 10 → verify → freeze-file.
- **Reused by import, pure functions only:**
  - from `analyze_f31`: `last_saved`, `native_pick`, `flagged`, `parse_train_log`;
  - from `run_f31`: the protected-hash, lock and record helpers, and `revalidate` (which `run_f31` takes from `run_n5`).
- The imported blobs are pinned and checked at run time: `analyze_f31` `feee9baf…`, `run_f31` `85db105a…`, and through them `run_n5` `e8f13e06…` and `analyze_n5` `01306415…`. Neither file is edited; F-31's results cite their blobs.

**Launch guards (all before the first run; each refuses):**
1. **Tree:** the tracked tree is clean, and the driver and its imports equal their HEAD blobs. HEAD is on `phase2-experiments`' first-parent chain and pushed.
2. **train.py:** HEAD's `scripts/train.py` hashes to `afceb4b1…` (§1).
3. **Protected files:** the 14 protected files match the COORDINATOR.md block, before and after every run.
4. **Data counts:** the `excluded_crops.txt` pin holds, giving 2081 train and 940 validation frames. `epochs_for(2081, 6560) == 99` and `batches == 66`.
5. **Lists:** the train and validation lists equal their committed blobs. They intersect no held-out or spent set: H9, the 2025 test, H-EP (`storms_holdout_hep.txt`), and any 2026 ID (I5 forbidden sets and `test_no_2026_id_in_any_list`).
6. **Paths:** every output path is under the three `v2_recipe` roots. No `models/v2_recipe/seed<s>.pt` exists, unless its record verifies (resume).
7. **Nothing held-out in the command:** no `--holdout-list`, no `data/hep_sealed`, no `evaluate.py` or `ensemble_eval.py`.
8. **Not frozen yet:** no `results/v2_recipe/members_sha256.txt` exists, tracked or on disk. Once frozen, the driver refuses to train.

**Per run (seed s, ascending 30–39):**
```
[PY, -u, scripts/train.py, --train-list, storms_train_expanded.txt, --val-list, storms_val_expanded.txt,
 --epochs, 99, --seed, s, --loss, l1, --out, models/v2_recipe/seed<s>.pt,
 --epoch-dir, models/v2_recipe/seed<s>_epochs]                              -> logs/v2_recipe/seed<s>.log
```
Then the run is re-validated on validation only: every epoch file, as F-31 did (about 2 min).

**Training-only checks (the record must pass all; modelled on `analyze_f31.check_training`):**
- **Identity:**
  - seed, train list, val list and epochs match the design;
  - the commit is the launch commit;
  - `train_py_sha256` is the pin;
  - protected-before and protected-after hashes match.
- **The log:** exactly 99 epoch lines.
- **Re-validation:** `val_mae_revalidated` covers epochs 1–99, consistent with the log to 0.0051 kt.
- **The native pick:** `native_pick(reval, saved, 99)` equals the last `saved` epoch. N5's tie rule applies, as in F-31.
- **Same weights:** the pick's epoch file is byte-identical to `seed<s>.pt` (`native_same_weights: true`).
- **Files:**
  - `files_sha256` names exactly the train log and `seed<s>.pt`, with matching hashes;
  - `deleted_epoch_sha256` hashes all 99 epoch files, and none remains on disk.
- **The under-training flag** (pick > 0.85 × 99, i.e. epoch ≥ 85) is computed and reported per run and as a count out of 10. It is descriptive only.
  - If it fires in ≥ 5/10 runs, the freeze entry says the recipe is still budget-limited on validation at 6534 steps for these seeds.
  - It changes nothing: no extension, no reselection. F-31's control was 5/20 at 4×.

**Disk:**
- **Kept:** about 450 MB (10 × 44.8 MB).
- **Transient:** at most 99 epoch files, about 4.4 GB, one run at a time. This follows John's F-31 choice 4: keep only what is selected.
- **Alternative (not recommended):** keep all epoch files, about 44 GB. Nothing in this design reads a non-selected epoch.

## 7. Compute

- **Per run:**
  - training: 99 × 4.6 s ≈ 7.6 min (F-31 control at 4×, measured: 7–9 min with epoch files);
  - re-validation: 99 × 1.2 s ≈ 2.0 min;
  - total: about 10 min.
- **10 runs:** about 1.6 h; plan on 1.5–2 h.
- **No build, no download.** Every storm was built for N5 and F-31. No H9, H-EP or 2026 evaluation.

## 8. Known limits

- **H9 selection of the recipe** (§0): only the 2026 test confirms it.
- **The 2026 comparison is v2 (4×, 2026-era code) against the frozen ensemble (1×, its own code version).** It answers "is v2 better than what is frozen", not "is 4× better than 1× in isolation". F-31 answered that on H9.
- **Validation is blind at Cat 5** (no frames ≥ 137 kt), as in F-31.
- **Ten members:** the ensemble's member spread is estimated from 10, as for the frozen ensemble.

## 9. Prerequisites, in order (none done)

0. This design approved by John, after hc-adversary (≤ 2 rounds unless there's a concrete defect).
1. **Launch PR:**
   - `scripts/run_v2_recipe.py` and its tests: synthetic records, the guards above, the coverage test, and no evaluation path;
   - the `.gitattributes` rows;
   - the `.gitignore` row for the lock.
   - It goes to hc-adversary and John before anything runs.
2. `check.py` PASS (FAST and `--full`) on the launch commit.
3. **decisions.md pre-launch entry:** this design's blob hash, the driver and pinned-import blob hashes, the seeds, and the order.
4. **Training** (`run_v2_recipe.py launch`) after John's yes. No H9, H-EP or 2026 read.
5. **Freeze PR:** records, logs, members file and the freeze entry (§4). From then on, v2 is frozen.
6. **The 2026 clean-test protocol** is committed before any 2026 build, and before the H-EP pass if v2 is E3's comparator. Only then does v2 meet held-out data.

## Files in this draft

- `design.md`: this file. No script, no data.
