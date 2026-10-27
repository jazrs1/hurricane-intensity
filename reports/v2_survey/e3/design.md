# E3 (Atlantic + East Pacific training): design (r2: review of PR #43 folded in; not approved; nothing run)

**Status:** design only. No build, download, training or evaluation before John's yes. The selection script, driver and tests come in reviewed PRs (§9).

**Source (John's decision 3, verbatim, v2 design header):** "E3 uses the same fixed budget, with an Atlantic-only comparator at that budget (the v2 ensemble may serve)."

**Design rule (John):** every E3 choice is justified from training/validation data or prior findings in this repo (F-31, N5, v2), never from properties of H-EP test storms. Each choice below carries its justification in a "Why" column, for hc-adversary to check one by one.

**What was read:** `src/`, `scripts/{train,analyze_n5,build_dataset,check}.py` (parts), the train/val lists, v2 design §1–§3 and §7–§9, F-31 design (budget lines), `results/f31/analysis/report.txt`, `results/n5_atlantic/analysis/report.txt`. Nothing under the forbidden paths. See Disclosure.

## 0. Summary of recommendations

| Open item | Recommendation | Section |
|---|---|---|
| Budget reading | **Step-matched:** `epochs_for(E3 train frames, 4 × 1640)` | §2 |
| Transfer gap | **(a) none:** E3 vs comparator is a direct H-EP contrast; zero H9 reads of E3 and v2 | §3 |
| Comparator | **Frozen v2** (`results/v2_recipe/members_sha256.txt`, seeds 30–39); no new Atlantic-only ensemble | §4 |
| EP training data | Mechanical rule with a tested concurrency gate, EP frames targeted at 1963 (E3 total ≈ 4044, F-31's atlantic-arm size, or fewer if EP data are short) | §5 |
| EP satellite | **GOES-18, 2023–2024** (the scoring sensor; VZA inside the Atlantic training range) | §5a |
| Everything else | Identical to v2 (one-factor contrast: only the train list changes) | §1 |

## 1. The recipe: v2 with one change

| Item | E3 value | Why |
|---|---|---|
| Train list | `storms_train_expanded.txt` (2081 frames) **plus** the EP storms of §5, in one new list `storms_train_e3.txt` | The one factor E3 tests |
| Validation list | `storms_val_expanded.txt`, 940 frames, unchanged; **no EP validation storms** | Same selection set as the comparator, so selection is not a second factor. The native rule's behaviour at 4× is documented only on this set (F-31 `selection.csv`, `val_trace.csv`). Cost, stated: E3's pick never sees EP frames. Alternative (EP val storms by the §5 rule) left to John; not recommended |
| Model, optimiser, loss | ResNet-18 IMAGENET1K_V1; Adam 1e-4 constant, no scheduler; batch 32; L1; unmasked; unrotated; no frame list beyond §5's frame rules | v2 §1. Any change would need a new comparator (§4) |
| High-end handling | **None** (no reweighting, no loss change, no >96 kt oversampling) | No repo finding supports one: N5 >96 bias NULL (d −0.95, CI [−5.5, +3.6]); F-31 S2 NULL (d +0.77, CI [−0.6, +2.2]); F-31 S3 (budget) NULL on >96 bias in both arms. A change would also break comparator equivalence |
| Budget | §2 | |
| Selection | Native: `train.py --out`, first strict minimum of unrounded validation MAE | v2 §1 (John, decision 1) |
| Members, seeds | 10, seeds **40–49**, ascending | Seeds 0–39 are used by this data path (0–9 frozen/N5, 10–29 F-31, 30–39 v2; v2 §2). A reused seed would reproduce part of a known run. The launch PR checks 40–49 appear in no model directory, log or commit message (as done for 30–39) |
| Ensemble | Unweighted per-frame mean of 10 members (`ensemble_eval.py`) | v2 §1 |
| All 10 runs are the members | No run dropped or replaced for any score; retrain only on a verification failure, failed attempt kept | v2 §2 |

**Code pins (E3 cannot copy v2's guard 2).** `git diff fc71b98 HEAD -- src` is not empty: PRs #40/#41 changed `config.py`, `hurdat2.py`, `pipeline.py` (EP HURDAT2 reader). So E3's guard is per file:

| File | Pin at E3's launch commit |
|---|---|
| `scripts/train.py` | SHA-256 `afceb4b1…231f` (blob `500bbc1e`), F-31's and v2's pin |
| `src/hurricane_intensity/{__init__,dataset,splits,crop,projection,goes_index}.py` | blob equal to fc71b98's (v2 §1 table) |
| `src/hurricane_intensity/{config,hurdat2,pipeline}.py`, `scripts/concurrency.py` | the blobs at E3's launch commit, **taken after John's pending antimeridian fix merges** (it changes `pipeline.py` and `concurrency.py`); no EP crop is built before that merge. The launch PR shows `git diff fc71b98` on the src files is confined to the EP-reader and antimeridian changes, with the Atlantic default paths unchanged. If those changes alter any Atlantic crop, the §4 byte-identity proof fails and E3 stops |
| `excluded_crops.txt` | blob `cc9dfcff` (Atlantic frames unchanged); EP exclusions go in a new pinned `excluded_crops_e3.txt`, never in the old file |

- **The learning-rate condition** (v2 §1) carries over: constant Adam, epoch count only a loop bound. Any schedule change reopens the budget.
- **The Atlantic crops** are the same files v2 trained and validated on. The launch guard hashes every Atlantic train/val crop and compares with a record made at v2's launch if one exists; otherwise it rebuilds a declared sample byte-for-byte, as check.py F2 does for Michael. (Unsure which applies; see "Open".)

## 2. Budget: two readings

| | Step-matched (**recommended**) | Epoch-matched |
|---|---|---|
| Epochs | `epochs_for(2081 + EP frames, 6560)`; at the §5 cap (4044 frames, 127 batches) **52 epochs, ≈6604 steps**; fewer EP frames give more epochs at the same steps | 99 |
| Steps vs v2 (6534) | equal to within one epoch's batches (the `epochs_for` rule, as N5 and F-31 arms) | ≈12,500, about 1.9× v2 |
| Comparator at that budget | v2 itself (same steps) | v2 matches in epochs only; a step-matched Atlantic-only comparator would be a new ≈190-epoch ensemble, a budget no arm in this repo has run |
| What a difference means | the effect of the added EP storms at fixed compute | EP storms **and** ≈1.9× compute, confounded. F-31 P1: compute alone moved control H9 MAE by −0.99 kt (1×→4×) |
| Budget-limited risk | **likely.** F-31's atlantic arm at this exact size and budget (4044 frames, 52 epochs): 12/20 flagged BUDGET-LIMITED | lower, but no repo evidence beyond 4× in either arm |
| Prior finding at this size and budget | F-31 S1: atlantic arm (4044 frames, 52 ep) beat control (2081, 99 ep) on H9 by 0.21 kt (95% CI [−0.36, −0.05]) while flagged budget-limited. So step-matched with ≈2× frames is not known to be harmful | none |

**Recommendation: step-matched.**
- **Why:** "budget" is steps in every prior design here (`analyze_n5.epochs_for`, TARGET_STEPS; N5; F-31 §2 compared arms of different sizes at equal steps). It keeps v2 a valid comparator, and isolates the data factor.
- **Carried rule (F-31):** the under-training flag (selected epoch > 0.85 × E) is reported per member, descriptively. If it fires in ≥ 5/10, the freeze entry and the H-EP report read any E3 deficit as "worse **at this budget**", not as harm from EP storms. Not rerun or rescued (N5, F-31 §2).
- **The exact epoch count** is filled by the launch guard from the built frame count: `epochs_for(N_e3, 6560)`, logged per run (v2 §1: the budget is tied to the frame count).

## 3. Transfer gap

| Option | What | Cost |
|---|---|---|
| **(a) none (recommended)** | No transfer-gap row for E3 or v2. The H-EP result is the direct contrast E3 − v2 on EP | none; "zero H9 reads of v2" (v2 §3) holds, and E3 also gets zero |
| (b) one H9 evaluation each | E3 and v2 each evaluated once on H9, inside the H-EP pass only, after both freezes; descriptive | breaks "zero"; `run_hep.py` needs an H9 stage; H9 is reused (F-31 §4), so the figure is not an estimate; invites comparison with F-31's 8.05 kt (v2 §3) |

**Why (a):** the question E3 answers is "does adding EP training storms help on EP storms, at fixed budget?" The paired contrast on the same H-EP frames answers it without an Atlantic reference. An H9 figure for E3 would mostly measure whether EP storms cost Atlantic skill, a separate question H9 (reused, H9-selected recipe) cannot answer cleanly. Either option is an H-EP protocol edit before gate 6 (v2 §3).

## 4. Comparator: frozen v2

| Option | Verdict | Why |
|---|---|---|
| **Frozen v2** (seeds 30–39, `results/v2_recipe/members_sha256.txt`) | **chosen** | Identical recipe except the train list (§1), at the same step budget (§2). Already frozen before any EP data exists, with zero H9/H-EP reads. John's decision 3 names it |
| New Atlantic-only ensemble at E3's budget | not needed | Under step-matched it would be the v2 recipe again, a second draw costing ≈1.6 h GPU with nothing gained. Needed only if E3 changed something besides the train list, which §1 forbids, or under epoch-matched (§2) |

- **Seed pairing:** none. E3 seeds 40–49 and v2 seeds 30–39 differ (a shared seed would not share an init path across different lists anyway). The contrast is ensemble vs ensemble.
- **Preprocessing (John's rule 5).** v2's pinned commit (fc71b98) has no EP HURDAT2 reader, so it **cannot build EP crops**. So the H-EP pass takes the second branch: before scoring, it proves the crops it evaluates are byte-identical between current code and each model's pinned crop path, for **every evaluated storm**:
  - v2: the crop modules (`crop`, `projection`, `goes_index`, `dataset`) are fc71b98's blobs (§1 pin); the proof rebuilds each evaluated crop with a harness importing fc71b98's crop modules on the same scan and fix and compares bytes;
  - E3: the same, against E3's launch commit;
  - any mismatch refuses the pass. No crop is built twice for scoring.

## 5. EP training storms: mechanical rule

Applied later by a reviewed script (`scripts/select_e3_ep.py`, not written). This design enumerates and looks up no EP storm.

| Step | Rule | Why (training data / prior finding) |
|---|---|---|
| Source | the pinned nepac HURDAT2 (`config.HURDAT2_EP_*`), read through the #41 reader, which refuses the H-EP set and its concurrency exclusions (Octave, Raymond 2025) by ID | gate (d); not a choice |
| Basin | ID prefix `EP` | E3's question is EP |
| Years | 2018–2024, then narrowed to the satellite's operational years (§5a: **2023–2024** under the recommendation) | The Atlantic train list spans exactly 2018–2024 (`storms_train_expanded.txt`), so E3 adds no new period. Not derived from H-EP (the reader gate excludes its IDs regardless) |
| Concurrency (**hard gate**) | drop every EP training frame whose box overlaps, at the same time, the box of any storm in the H-EP set or its concurrency exclusions: `scripts/concurrency.py` (N4 rule: 600 km box + 50 km margin, `boxes_overlap`/`concurrent`, positions interpolated as the pipeline does) | ID exclusion is not enough: the training years overlap the H-EP period, so a concurrent storm's crop can carry H-EP pixels (review_b1 I-1, the Marco/Laura case; N4/A4 precedent). See below |
| Status fixes | `config.DEFAULT_INCLUDE_STATUSES` (LO TD SD TS SS HU) | Same as every Atlantic build |
| Sampling, tolerance | the interval and scan tolerance recorded in the Atlantic train manifests (3-hourly per the list header); the script reads them, not this design | same frame construction as the Atlantic data |
| Satellite | §5a: **`noaa-goes18`** (recommended) | §5a |
| VZA handling | drop EP **fixes** whose viewing zenith angle from the §5a satellite (fix lat/lon, the satellite's nominal sub-point: 137.2° W for GOES-18) exceeds the **maximum GOES-16 VZA (75.2° W) over the fixes of the 19 Atlantic train + 5 val storms**, computed by the script with the same formula. No other angle handling | Keeps EP inputs inside the angle range the comparator was trained and selected on (training and validation data) |
| Intensity floor | storm's peak best-track intensity ≥ the **minimum peak over the 19 Atlantic train storms** (`storms_train_expanded.txt` minus its 5 val storms; Atlantic HURDAT2 cache) | same intensity floor as the comparator's training storms; val storms are excluded because they are the selection set, not training data |
| Minimum frames | ≥ the **minimum per-storm post-exclusion frame count over the same 19 train storms** | same floor, same set |
| Bad crops | EP crops failing the existing NaN/exclusion check go in `excluded_crops_e3.txt` | as `excluded_crops.txt` |
| Order and cap | eligible storms sorted by SHA-256 of the ID string; add whole storms in that order until EP frames ≥ **1963** (= 4044 − 2081) | Target makes E3's train size equal F-31's atlantic arm (4044), the only larger-data size with prior evidence at this budget (§2). Hash order is fixed before any storm is seen, so no intensity or angle preference enters. Without a cap, all eligible EP storms could outnumber Atlantic frames several times and push epochs far below 52, outside any prior finding |

The script writes `storms_train_e3.txt` and a record (rule inputs, eligible list, hash order, final frame count); its output is pinned at launch. If the eligible EP frames total less than 1963, all are used and `epochs_for` sets the epochs; E3's size then lies between F-31's control (2081) and atlantic (4044) arms, both studied at 4× (5/20 and 12/20 flagged).

**Concurrency gate (D1), mechanical and tested:**
- `select_e3_ep.py` loads the excluded storms' tracks (the H-EP set and its concurrency exclusions) as **(time, lat, lon) only**, through `concurrency.track` on the pinned nepac file, and calls `concurrency.concurrent` against each candidate EP training track at the build sampling grid.
- Every overlapping training frame is dropped **before** frame counting, the floors and the cap. Whole storms are not dropped unless the floors then fail.
- **It prints and records no excluded storm's ID, name, wind or position:** only per-candidate counts of dropped frames.
- **Hard gate:** after selection the script re-runs the test on every kept frame and refuses to write `storms_train_e3.txt` if any overlaps. The launch guard re-runs the same check on the pinned list.
- **Tests (synthetic tracks):** an overlapping frame is dropped; a frame 651 km away is kept; a non-overlapping time is kept; the refusal fires on an injected overlap; the output contains no excluded-storm field.
- **Needs:** the #41 reader refuses H-EP IDs outside the sealed pass. The gate needs a reviewed positions-only path (time, lat, lon, no ID or intensity returned) for this one use. That is a reader-gate PR, before step 2 of §9.

## 5a. Satellite for EP training imagery (choice for John)

Protocol fact (from the coordinator, not read here): H-EP imagery is built from `noaa-goes18` for all its storms. That is a property of the scoring platform, not of any test storm. Nominal sub-points: GOES-16 75.2° W, GOES-17/18 137.2° W (GOES-West). Typical EP genesis/track region ≈ 95–135° W.

| Option | Years usable | Frames available | VZA vs Atlantic training (GOES-16 on Atlantic storms) | Transfer confound in E3 − v2 |
|---|---|---|---|---|
| GOES-16 (`noaa-goes16`) | 2018–2024 | most (7 seasons); cap 1963 likely reached | **oblique-view proxy:** EP storms 20–60° of longitude from 75.2° W, so many fixes near or past the Atlantic maximum and dropped by the cap; kept fixes cluster at the oblique end | E3 learns EP from a sensor and geometry H-EP is not scored on: train and test views differ for E3's EP data, which understates what EP data could add |
| **GOES-18 (`noaa-goes18`), recommended** | 2023–2024 (GOES-West from 2023-01-04) | fewest (2 seasons); cap may not be reached (handled above) | near-nadir to moderate for EP, the same geometric regime as GOES-16 on the Atlantic training storms; the cap should drop few fixes | E3 sees the scoring sensor; v2 sees only GOES-16. So E3 − v2 includes "EP storms viewed as H-EP views them". That is the question E3 asks, and it is stated in the report as part of the treatment |
| GOES-17 (`noaa-goes17`) | 2019–2022 (GOES-West 2019-02-12 to 2023-01-04) | middle (4 seasons) | as GOES-18 | a third sensor, seen by neither the comparator nor H-EP; no read-path check in this repo (check.py F3 covers G18 only) |
| GOES-17 + GOES-18 (operational West at fix time) | 2019–2024 | most West frames | as GOES-18 | two sensors; GOES-17's band-13 cooling anomaly needs a quality rule this repo has no finding for |

**Recommendation: GOES-18, 2023–2024.**
- **Why, from train/val data and prior findings:** (1) the VZA cap is defined by the Atlantic train/val geometry, and only a GOES-West view keeps EP fixes inside it without discarding most of the oblique range; (2) the GOES-18 read/crop path has a regression check (check.py F3), GOES-17 none; (3) F-31 S1 says more storms at fixed steps did not hurt at 4044 frames, and §5's floor case (fewer EP frames) lies between two studied sizes, so fewer frames is a smaller risk than an unchecked sensor.
- **Cost, stated plainly:** fewer EP storms; E3's size is set by the data, not the 1963 target, if short.
- GOES-16 stays the fallback only by John's choice, never by the script.

## 6. Reporting strata (E3 vs v2 on H-EP, descriptive beyond the protocol's primary)

| Stratum | Edges | Why |
|---|---|---|
| Overall | all frames | primary, as N5/F-31 |
| > 96 kt | best-track > 96 kt | the high-end threshold of N5 and F-31 (S2, S4) |
| VZA | angle from the **§5a satellite (GOES-18 under the recommendation, the same satellite H-EP is scored on)**; edges = tertiles of E3's EP training fixes' VZA from that satellite, written to the selection record before freeze. Each H-EP frame is binned by its own GOES-18 VZA | training-data edges; fixed before any H-EP read; the binned angle is the one each model's scored input was actually viewed at |

No stratum edge or new stratum is chosen after the freeze.

## 7. Freeze and timeline

- **Freeze**, as v2: `results/e3/members_sha256.txt` (LF in blob and checkout) plus a decisions entry with the design blob, driver blob, pins, seeds, the selection record hash and `N_e3`. After it, E3 is frozen; `launch` refuses.
- v2's freeze check refuses any decision package not named `V2-*.md`; E3's driver extends it to `E3-*.md`.
- **Order:** E3 frozen before the H-EP pass, and before any H-EP protocol edit for §3. **Deadline:** about end-January 2027.

## 8. Compute (estimate)

- Training ≈ v2's per run (same steps, same val set): ≈7–9 min + ≈2 min revalidation; 10 runs ≈ 1.5–2 h.
- The EP build (download + crops for ≈1963 frames) is separate and comes first.

## 9. Prerequisites, in order (none done)

0. This design approved by John, after hc-adversary.
1. EP HURDAT2 downloaded under gate (d); John's antimeridian fix merged; the positions-only reader path for the concurrency gate merged (§5).
2. `select_e3_ep.py` + tests (reviewed PR); run; `storms_train_e3.txt`, record, `excluded_crops_e3.txt` committed.
3. EP crop build (reviewed); check.py FAST and `--full` PASS.
4. Launch PR: E3 driver (v2's, with §1 pins, `epochs_for(N_e3, 6560)`, seeds 40–49, no evaluation stage) + tests; hc-adversary and John.
5. Training after John's yes. No H9, H-EP or 2026 read.
6. Freeze PR (§7).
7. H-EP protocol edit for §3 (a) and §4's crop-identity proof, before gate 6.

## Open / unsure

- Whether v2's launch recorded per-crop hashes for the Atlantic train/val crops (§1); if not, the launch PR adds a rebuild sample.
- §5a's satellite fact (H-EP built from `noaa-goes18`) comes from the coordinator's review, not from a read of the protocol.
- How many GOES-18 EP frames survive the floors, VZA cap and concurrency gate is unknown until the script runs; the design accepts any count (§5).
- Whether every Atlantic train storm meets its own floors trivially (it must, by construction); the script asserts it.

## Disclosure

- No forbidden path was opened. A `data/interim` read (Atlantic HURDAT2, peak winds) was refused by the sandbox and not performed.
- H-EP is **mentioned** without labels, angles or frame counts at `src/hurricane_intensity/config.py:39`, `src/hurricane_intensity/dataset.py:15`, `scripts/check.py:84–98`, and in v2 design §3 (protocol §2f/§6 references).
- `scripts/check.py:94–97` shows one EP fix (EP052023, 2023-08-06 06Z, GOES-18) with label 130 kt. Read before knowing whether it is in the H-EP set; listed for completeness. It is a 2023 storm, so it may be an E3 training candidate under §5.
