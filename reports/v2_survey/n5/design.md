# N5: Atlantic-only training arm, design (DRAFT r2, not approved, nothing run)

r1 drafted 2026-10-18 by hc-survey. r2 is the same day, after hc-adversary item 5 (22 findings) and John's additions A1–A4, relayed by the coordinator. John approved running the arm at 22:47; he sees this design before any build or training.

- **Nothing has been run:** no build, download, training, evaluation, model load, list file or git write.
- **Format:** modelled on logs/decisions.md 2026-10-14 15:07 / 18:10 / 18:58 / 19:08 and the 2026-10-13 night budget section.
- **When accepted:** the text becomes a decisions.md entry before launch, together with the analysis blob hash (§9).

**Changed in r2:** §0, §2 (Marco), §3, §4, §5, §6, §7, §8, §9, §10 and §11, plus new §12 (launch prerequisites) and §13 (launch PR contents). The new power figures are in §5.

**Changed in r3** (hc-adversary item 7 and the coordinator's batch):
- §4: re-validation of all 25 control epochs, checked against train.py's own pick; epoch files kept until the analysis; disk cost.
- §5: power rows relabelled; Monte Carlo footnote.
- §6: the frozen check uses the native control.
- §8: pinned manifest paths; the CSV↔eval-log content link; `val_mae_revalidated`.
- §9: the new linkage refusals; manual blob-hash comparison; 45 tests.

**Changed in r4** (adversary 7b): §9: odd native epochs. The matched and native control vectors must be identical when the picks coincide, and that pair is exempt from the identity refusal; 47 tests.
- §12 and §13: the driver and exclusion step are specification only, and the launch PR returns for review; the H-EP list and its guards land before or inside the launch PR; PR #4; no `__pycache__`.

## 0. Confounds: read this first

**The arm tests a bundle, not storm count.** No unused GOES-16 Atlantic hurricanes are left to build a storm-count-matched comparison (survey §1), so storm count, frame volume, intensity mix, exposure and sampling density all move together. At equal steps (control 1650, atlantic 1664), the arm changes:

- **Storms:** 19 → 50.
- **Frames:** 2081 → about 4072.
- **Intensity mix:** 1912 of the about 2002 new frames are below 83 kt, and only 8 are at ≥ 96 kt. The ≥ 96 kt share of training falls from 18.9% to about 9.9%.
- **Absolute high-end exposure halves.** The control makes 25 passes over 394 frames at ≥ 96 kt, which is 9850 high-end presentations. The atlantic arm makes 13 passes over about 402, about 5226. **Every existing storm, at every intensity, is seen about half as often** (13 passes vs 25).
- **Sampling density:** hourly frames (the original nine storms, 1307 frames) fall from 63% of training to about 32%. That is the variety-against-volume axis (10-14 18:58/19:08), so the two contrasts are entangled.
- **Selection grid:** the atlantic arm has 13 candidate checkpoints, the control natively 25. A coarser grid biases against the atlantic arm. A validation-only proxy shows about +0.13 kt at selection (expanded logs re-selected on every other epoch: 8.52 → 8.66; adversary item 5, finding 13). The primary therefore uses a **matched grid** (§4).
- **Under-training is asymmetric.** The control is already at its limit: 6 of 10 expanded runs selected epoch 24–25. The atlantic arm makes 13 passes over twice the data, so it is probably further from convergence. That handling is pre-registered (§4, §7).

**Consequence:** a difference cannot be attributed to "more distinct storms" rather than "more frames", "a weaker mix", "less exposure per storm" or "less training". A "worse" result in particular is not evidence that the added storms are harmful (§7).

## 1. Question and expectation

**Question.** Starting from the frozen expanded configuration, does adding all remaining usable GOES-16 Atlantic hurricanes (31 storms, 3-hourly) change H9 overall MAE and H9 > 96 kt bias at equal steps?

**Expectation, stated in advance; both tests are two-sided:**
- Overall MAE should fall, mostly on TD/TS/Cat 1 frames.
- The > 96 kt bias may get worse, meaning more under-prediction: the high-end exposure halves (§0). The more-distinct-storms gain came with added high-end frames (10-14 18:53).
- **John's case is a trade-off:** MAE better and bias worse. It is pre-registered as a joint verdict (§7).

## 2. Storms (PROPOSED)

**Source.** All candidates are GOES-16 (2018–2024), with labels from the pinned Atlantic HURDAT2 (`hurdat2-1851-2025-02272026.txt`, SHA-256 `ee35819c…0bcb5c`; survey §7 shows zero label changes against the newer release).

**Frames.** Counts are pipeline-grid estimates from `reports/v2_survey/n5/estimate_frames.py` → `candidate_frames.csv`. They are re-fixed after the build.

**The candidates.** All unlisted 2018–2024 Atlantic hurricanes from survey §1 except three:
- AL022022 Bonnie, which is aliased to EP042022;
- AL192019 Rebekah, whose 70 kt was extratropical;
- AL182019 Pablo, at the GOES-16 edge.

All 31 are built 3-hourly.

| ATCF | Name | Peak kt (tropical) | Frames (3h) | TD | TS | Cat 1 | Cat 2 | Cat 3 | Cat 4 | Cat 5 |
|---|---|---|---|---|---|---|---|---|---|---|
| AL072021 | Grace | 105 | 71 | 23 | 29 | 13 | 2 | 4 | 0 | 0 |
| AL272020 | Epsilon | 100 | 77 | 24 | 16 | 29 | 5 | 3 | 0 | 0 |
| AL282020 | Zeta | 100 | 42 | 4 | 20 | 14 | 3 | 1 | 0 | 0 |
| AL082018 | Helene | 95 | 71 | 4 | 38 | 16 | 13 | 0 | 0 | 0 |
| AL162018 | Oscar | 95 | 39 | 0 | 16 | 16 | 7 | 0 | 0 | 0 |
| AL032018 | Chris | 90 | 59 | 24 | 20 | 10 | 5 | 0 | 0 | 0 |
| AL062024 | Francine | 90 | 32 | 1 | 21 | 7 | 3 | 0 | 0 | 0 |
| AL102019 | Jerry | 90 | 89 | 30 | 47 | 8 | 4 | 0 | 0 | 0 |
| AL102024 | Isaac | 90 | 35 | 0 | 18 | 10 | 7 | 0 | 0 | 0 |
| AL132024 | Leslie | 90 | 91 | 16 | 35 | 35 | 5 | 0 | 0 | 0 |
| AL172020 | Paulette | 90 | 174 | 32 | 105 | 22 | 15 | 0 | 0 | 0 |
| AL052024 | Ernesto | 85 | 67 | 2 | 24 | 33 | 8 | 0 | 0 | 0 |
| AL152023 | Nigel | 85 | 55 | 14 | 10 | 26 | 5 | 0 | 0 | 0 |
| AL082020 | Hanna | 80 | 33 | 9 | 15 | 9 | 0 | 0 | 0 | 0 |
| AL092020 | Isaias | 80 | 50 | 0 | 30 | 20 | 0 | 0 | 0 | 0 |
| AL132018 | Leslie | 80 | 161 | 10 | 111 | 40 | 0 | 0 | 0 | 0 |
| AL142023 | Margot | 80 | 91 | 10 | 54 | 27 | 0 | 0 | 0 | 0 |
| AL152022 | Lisa | 80 | 46 | 15 | 24 | 7 | 0 | 0 | 0 | 0 |
| AL052021 | Elsa | 75 | 75 | 2 | 65 | 8 | 0 | 0 | 0 | 0 |
| AL052022 | Danielle | 75 | 63 | 8 | 17 | 38 | 0 | 0 | 0 | 0 |
| AL132022 | Julia | 75 | 32 | 6 | 20 | 6 | 0 | 0 | 0 | 0 |
| AL162022 | Martin | 75 | 15 | 0 | 8 | 7 | 0 | 0 | 0 | 0 |
| AL162024 | Oscar | 75 | 31 | 0 | 18 | 13 | 0 | 0 | 0 | 0 |
| AL022018 | Beryl | 70 | 103 | 34 | 60 | 9 | 0 | 0 | 0 | 0 |
| AL022019 | Barry | 65 | 43 | 14 | 27 | 2 | 0 | 0 | 0 | 0 |
| AL052023 | Don | 65 | 113 | 31 | 79 | 3 | 0 | 0 | 0 | 0 |
| AL082021 | Henri | 65 | 81 | 29 | 45 | 7 | 0 | 0 | 0 | 0 |
| AL092018 | Isaac | 65 | 61 | 10 | 44 | 7 | 0 | 0 | 0 | 0 |
| AL142020 | Marco | 65 | 41 | 16 | 22 | 3 | 0 | 0 | 0 | 0 |
| AL162020 | Nana | 65 | 23 | 3 | 18 | 2 | 0 | 0 | 0 | 0 |
| AL252020 | Gamma | 65 | 38 | 13 | 24 | 1 | 0 | 0 | 0 | 0 |
| **total** | 31 storms | | **2002** | 384 | 1080 | 448 | 82 | 8 | 0 | 0 |

Notes:
- **Julia:** AL132022 only. EP182022 is never listed (alias map).
- **Name collisions:** Beryl 2018 and Helene 2018 share names with listed storms (AL022024 is H9, AL092024 is training). Everything is keyed on ATCF ID.
- **Size:** Paulette (174) and Leslie 2018 (161) are the largest new storms, each about 4%. Dorian's share falls from 17.2% to about 8.8%.
- **Expected yield:** about 1995 frames after the build (the earlier 3-hourly build matched 775/777 and NaN-excluded 1/774).

**Marco rule (N4, decided A4).** Drop every Marco AL142020 training frame whose 600 km box overlaps a validation, H9, 2025-test or H-EP storm's box at the same time, with a **50 km margin**.
- **Test:** flat-earth at the mean latitude; |dx| < 650 km and |dy| < 650 km between centres.
- **Why a margin:** it covers the flat-earth and linear-interpolation approximations.
- **Estimate:** only one pair qualifies, Marco vs Laura AL132020 (validation). Closest approach is 732.5 km. On the hourly grid: 12 h under the 650 km box test and 8 h under plain overlap. That is **4 frames on the 3-hourly grid: 2020-08-25 09Z, 12Z, 15Z, 18Z** (`concurrency.csv`).
- **The wording that was dropped:** a centre-to-centre "700 km" rule drops nothing. Two 600 km boxes can overlap at centre distances up to 849 km.
- **Frames come from the built manifest:** the dropped frames are chosen from the built Marco manifest's sample_times (exact fixes included), not from this estimate. The concurrency function is the F-10 function, reused.
- **Mechanism:** the drop is applied with `train.py --frame-list`.

## 3. Splits and counts

| Split | List | Storms | Frames | ≥ 96 / ≥ 113 / ≥ 137 |
|---|---|---|---|---|
| Control training | `storms_train_expanded.txt` (minus val) | 19 | 2081 | 394 / 220 / 67 |
| Atlantic-only training (PROPOSED) | `storms_train_n5_atlantic.txt` = expanded + the 31 above (written in the launch PR, §13) | 50 | about 4072 (2081 + about 1995 − 4 Marco) | about 402 / 220 / 67 |
| Validation, both arms | `storms_val_expanded.txt` (unchanged) | 5 | 940 | 124 / 56 / 0 |
| Held-out, both arms | `storms_holdout_h9.txt` (unchanged) | 9 | 1752 | 653 frames > 96 kt; 379 / 31 at ≥ 113 / ≥ 137 |

- **Unchanged and kept out of training:** H9, `storms_val_expanded.txt` and `storms_test_2025.txt`. Nothing is added to them.
- **Never in training:** any 2025 or 2026 storm, EP/CP storm, or alias of a held-out or validation storm.
- **Checkpoint selection** uses `storms_val_expanded.txt` only, which has no frames ≥ 137 kt.

## 4. Arms, budget and selection

Same in both arms:
- ResNet-18 IMAGENET1K_V1, L1, Adam lr 1e-4, batch 32;
- unmasked, unrotated;
- same preprocessing; exclusion rule unchanged (any NaN pixel).

| Arm | Train list | Frames | Batches/epoch | Epochs | Steps | Checkpoint used for the primary |
|---|---|---|---|---|---|---|
| control | `storms_train_expanded.txt` | 2081 | 66 | 25 | 1650 (+0.6%) | **matched grid:** best validation MAE over the **odd epochs 1, 3, …, 25** (13 candidates, every 132 steps) |
| atlantic | `storms_train_n5_atlantic.txt` + `--frame-list` | about 4072 | 128 | **13** (computed) | 1664 (+1.5%) | native: best validation MAE over its 13 epochs (every 128 steps) |

**Budget rule (10-13 night / 10-14 15:07).** Use the whole number of epochs whose step total is nearest 1640.
- **In code:** `analyze_n5.epochs_for(frames)` computes it. It gives 13 epochs for anything from 4033 to 4096 frames.
- **Only one constant moves after the build:** `ARM_TRAIN_FRAMES`, together with the frame list's SHA (§9).

**Matched selection grid (B6).**
- **Control training:** the control is trained with `--epoch-dir`. In `train.py:133-134`, that only writes `epoch_NN.pt` (state dict) every epoch. The epoch loop evaluates **validation only**.
- **Driver re-validation:** after training, the driver re-validates **all 25** epoch checkpoints on `storms_val_expanded.txt` alone. That gives unrounded values; the log prints 2 decimals.
- **End-to-end check:** the unrounded minimum over all 25 must equal `train.py`'s own native pick, its last `saved` line, which `train.py` made on unrounded values. That proves the re-validation path reproduces in-training validation, not just to the cent.
- **Re-selection:** only then does the driver take the first minimum over the odd epochs, and it evaluates **only that one checkpoint** on H9.
- **Analysis cross-checks:** `analyze_n5.py` checks that the re-validated values round to the logged ones, that their minimum is the native pick, and that the chosen epoch is the odd-grid minimum.
- **Sensitivity, pre-registered:** the control's native selection (best of all 25 epochs, `train.py`'s own `best_val.pt`) gets one extra H9 evaluation per seed. It is reported as a sensitivity analysis and does not change the verdict.
- **No held-out storm is evaluated per epoch:** H9 sees exactly two control checkpoints per seed (matched and native) and one atlantic checkpoint.
- **Disk:**
  - all 25 epoch files (about 1.1 GB) exist per seed until re-validation;
  - the 12 even-epoch files are then deleted;
  - the 13 odd-epoch files (the candidates) are **kept until the analysis has run**, so the selected checkpoint's SHA can be verified. That is about 13 × 10 × 45 MB ≈ **5.9 GB** transient, deleted after the results entry.
- **John's alternative:** the coordinator offers John "native control selection as the primary". Under that alternative, the matched-grid analysis becomes the sensitivity.

**Under-training flag, in steps (B7).**
- **Definition:** a run is flagged when its selected checkpoint falls in the **last 15% of its steps**. For the control's matched grid, that means epochs 23 and 25. For the atlantic arm, epochs 12 and 13.
- **Consequence, pre-registered:** if the flag fires in **≥ 5/10 runs of either arm**, the result is reported as **budget-limited**. It is not rerun or rescued, and an extended budget would need a new design.
- **How a budget-limited "worse" is read:** as "worse at this budget", not as harm from the added storms.

**Budget provenance.** The 1640-step budget was first chosen from per-epoch held-out MAE on Ian/Nicole/Claudette (finding 7). The H9 entry (adversary item 4, COORDINATOR.md 22:55) now bounds that favour near zero: validation alone gives the same window shape and the same budget. §7 still discounts a gain confined to H3, since that costs nothing.

**Seeds and order (O1 decided: retrain).**
- Seeds 0–9 in both arms; 20 runs.
- The runs are **interleaved round-robin by seed, with arm order alternating** (C/A, A/C, …).
- All 20 run under **one commit**, with `scripts/check.py` PASS and the regenerated `excluded_crops.txt` committed.
- **Pairing:** atlantic seed s is compared with control seed s. The pairing is valid because it was fixed in advance, not because seeds carry shared variance: the between-arm seed correlation was +0.001 last time, and "the seed carries little" (decisions.md:405-407).

## 5. Primary tests (A1: two named primaries, Bonferroni)

Both primaries are on H9 pooled (1752 frames), **per member** (one value per training run), using the matched-grid control. Each uses the paired differences over seeds 0–9 and a **two-sided exact sign-flip test** (all 1024 sign assignments) at **α = 0.025** (Bonferroni 0.05/2). Each gets a **97.5% CI by test inversion**. All of this is implemented in code (`PRIMARIES`, `ALPHA`, `SESOI` in `analyze_n5.py`).

| Primary | Statistic | Better means | Smallest effect of interest (power-derived, not a meaningfulness claim) |
|---|---|---|---|
| 1. H9 overall MAE | d = MAE(atl) − MAE(ctl) | d < 0 | 0.9 kt |
| 2. H9 > 96 kt bias (truth > 96 kt, 653 frames) | d = bias(atl) − bias(ctl) | **d > 0 while both arms' mean bias is negative** (less under-prediction) | 3.0 kt |

**Sign crossing, fixed now.** If either arm's mean > 96 kt bias is ≥ 0, primary 2 is instead run on the per-seed d = |bias(atl)| − |bias(ctl)|, where d < 0 is better. The report names which branch applied. Today both are negative (expanded −7.7).

**Power at n = 10, α = 0.025** (`power.py`; deterministic normal-model Monte Carlo, 2000 reps per cell). The sd inputs are observed (results/storms):
- overall MAE: sd 0.633 (control2), 0.286 (expanded), sd of per-seed differences 0.694, correlation +0.001;
- > 96 bias: 2.01 / 1.43, sd of differences 2.44, correlation +0.025.

**Overall MAE: 71–100% power to detect a 0.9 kt difference (plan on 71%).**

| sd_diff | Scenario | 0.5 kt | 0.6 kt | **0.9 kt** | 1.2 kt | 80% MDE | Informative null at a true 0 (97.5% CI inside ±0.9) |
|---|---|---|---|---|---|---|---|
| 0.40 | both arms at 0.29, the expanded configuration (the retrained control IS that configuration) | 0.87 | 0.96 | 1.00 | 1.00 | 0.50 | 1.00 |
| 0.69 | one arm at 0.63, one at 0.29 (last time's two arms, control2 vs expanded) | 0.38 | 0.52 | 0.89 | 0.99 | 0.80 | 0.78 |
| **0.85** | **both arms at 0.6 (conservative planning case)** | 0.23 | 0.35 | **0.71** | 0.93 | 1.05 | 0.50 |

**> 96 kt bias: 71–95% power to detect a 3.0 kt difference (plan on 71%).**

| sd_diff | Scenario | 2.0 kt | 2.5 kt | **3.0 kt** | 3.5 kt | 4.0 kt | 80% MDE | Informative null at 0 (SESOI 3.0 / 2.5) |
|---|---|---|---|---|---|---|---|---|
| 2.03 | both arms at 1.43, the expanded configuration | 0.64 | 0.84 | 0.95 | 0.99 | 1.00 | 2.5 | 0.89 / 0.68 |
| 2.44 | one arm at 2.01, one at 1.43 (last time's two arms) | 0.47 | 0.68 | 0.83 | 0.94 | 0.98 | 2.9 | 0.69 / 0.40 |
| **2.84** | **both arms at 2.01 (conservative planning case)** | 0.36 | 0.53 | **0.71** | 0.85 | 0.93 | 3.3 | 0.48 / 0.21 |

Notes on the power figures:
- **Monte Carlo error:** the standard error at p ≈ 0.7 with 2000 reps is √(0.21/2000) ≈ 0.01. So 0.70 (my rerun at sd 2.84 and 3.0 kt) and 0.71 (the adversary's) are the same figure. The wording is John's and stands as written: "71–95% power to detect 3.0 kt (plan on 71%)", and for the MAE, "71–100% power to detect 0.9 kt (plan on 71%)".
- **Why the bias plans on 2.84:** the atlantic arm's bias spread is expected to be larger, because its high-end exposure halves.
- **Why the MAE plans on 0.85:** for the same caution.
- **3.0 kt context:** it is about 55% of the last experiment's +5.40 kt bias gain.
- **Bonferroni is conservative here:** the MAE and bias differences correlated −0.59 across seeds last time.

## 6. Secondary and descriptive

- **Secondary, pre-registered** (α 0.05, uncorrected, labelled secondary): H9 **> 96 kt MAE**.
- **Descriptive**, labelled; paired sign-flip p printed, not tests:
  - fixes-only MAE, overall bias, > 96 |bias|, > 96 kt Spearman;
  - per-band MAE and bias (bands as in `evaluate.py`);
  - selected epochs and steps;
  - 10-member ensemble-mean metrics per arm;
  - the validation-only per-epoch trace.
- **H9 split (A2):** the original three (Ian, Nicole, Claudette) vs the six added later. Overall MAE and > 96 kt MAE/bias for each, printed in `report.txt`.
- **Per storm (A2, exploratory):**
  - every H9 storm's MAE;
  - > 96 kt MAE and bias **only for storms with ≥ 10 frames > 96 kt** (the 2025 entry's rule);
  - Sam AL182021, the largest > 96 kt storm, was traced per epoch on 10-14 (adversary item 4). It has had per-epoch exposure, and its row is read with that in mind.
- **Reproducibility check (A3):** the retrained control **at its native selection** (best of all 25 epochs, the same rule the frozen members were selected by; item 7 A2) vs the frozen expanded members (`results/storms/expanded_seed*_h9.csv`), on overall MAE, > 96 bias and > 96 MAE. It is **unpaired and distribution-level only**, with a descriptive exact permutation p. It is not "same seeds".

## 7. Interpretation, fixed in advance

**Per primary** (α 0.025, 97.5% CI, SESOI 0.9 kt or 3.0 kt):

| Outcome | Condition | Reading |
|---|---|---|
| **BETTER / WORSE** | p < 0.025 | direction by the §5 convention |
| **SIGNIFICANT BUT BELOW SESOI** | p < 0.025 and the 97.5% CI lies inside ±SESOI | real but smaller than the effect of interest |
| **NULL (informative)** | p ≥ 0.025 and the CI lies inside ±SESOI | changes are under the SESOI in either direction |
| **NULL (inconclusive)** | p ≥ 0.025 and the CI reaches ±SESOI | no conclusion; reported, not rerun |

**Joint verdict.** A primary counts as better or worse here when it is significant, including below-SESOI results:

| Joint verdict | Condition |
|---|---|
| **TRADE-OFF** | MAE better, > 96 kt bias worse (John's case) |
| **REVERSE TRADE-OFF** | MAE worse, bias better |
| **BOTH BETTER / BOTH WORSE** | both primaries move the same way |
| One primary only | one moves, and the other shows no significant difference |
| **NO SIGNIFICANT DIFFERENCE** | neither primary moves |

**Budget-limited qualifier (§4).** It is appended when ≥ 5/10 runs of either arm are flagged. Any "worse" is then read as "worse at this budget".

**Attribution.** Every reading is about the bundle in §0, not storm count. A gain confined to H3 is discounted.

**Secondary and sensitivity results** are reported beside the verdict and never replace it.

## 8. Paths and commands

**Never written:**
- `best_model.pt`, `models/baseline_9kt.pt`, `results/heldout_predictions.csv`;
- anything under `models/storms/` (including `models/storms/expanded/`), `results/storms/`, `results/ensemble/`, `results/final_2025/`, or `site/`.

**New:**
- `models/n5_atlantic/{control,atlantic}/`;
- `models/n5_atlantic/control/seed<s>_epochs/`, transient;
- `results/n5_atlantic/`;
- `logs/n5_atlantic/`, including `logs/n5_atlantic/runs/`.

**Every step runs through Python with list arguments.** There is no shell redirection (PowerShell 5.1 `>` writes UTF-16LE) and no `{seed}` patterns (PowerShell parses braces as script blocks). The driver is `scripts/run_n5.py` (B4), written in the launch PR (§13):

- **Subprocess calls:** `subprocess.run([sys.executable, "-u", "scripts/train.py", ...], stdout=f, stderr=subprocess.STDOUT)`, with `f` opened in binary and the bytes written unchanged (UTF-8, no BOM). Every argument is an explicit list element, using forward-slash relative paths from the repo root.
- **Per-run log lines:**
  - `RUN arm=<a> seed=<s> order=<k> list=<train list> val=storms_val_expanded.txt epochs=<n>`;
  - `START <iso time> | <nvidia-smi snapshot>`, then `END ...` or `TRAIN_FAILED`.
- **Refuses existing outputs:** before each run, it checks that every output path (checkpoint, epoch dir, CSVs, logs, manifest) is absent. `train.py`'s `torch.save`, `evaluate.py`'s CSV and any rerun would otherwise overwrite silently.
- **Protected hashes:** recorded from the COORDINATOR.md protected block before and after every run, into the manifest. A mismatch stops the driver.
- **Analysis hash:** records the commit (`git rev-parse HEAD`) and the analysis blob hash at launch.
- **Order:** the 20 runs in the §4 order.

**Argument lists run_n5.py passes, per seed s.** `PY` = `sys.executable`; paths are relative to the repo root.

```
control, train:     [PY, -u, scripts/train.py, --train-list, storms_train_expanded.txt, --val-list, storms_val_expanded.txt,
                     --epochs, 25, --seed, s, --loss, l1, --out, models/n5_atlantic/control/seed<s>.pt,
                     --epoch-dir, models/n5_atlantic/control/seed<s>_epochs]
                     -> logs/n5_atlantic/control_seed<s>.log
control, re-select: validation-only re-validation of ALL epoch_01 ... epoch_25 (same TCDataset/val loader as train.py)
                     -> manifest val_mae_revalidated; require its unrounded argmin == the last 'saved' epoch;
                     then the first minimum over the odd epochs -> selected_epoch, selected_checkpoint
                     = models/n5_atlantic/control/seed<s>_epochs/epoch_<NN>.pt
control, eval:      [PY, scripts/evaluate.py, --checkpoint, <selected_checkpoint>, --train-list, storms_train_expanded.txt,
                     --holdout-list, storms_holdout_h9.txt, --out, results/n5_atlantic/control_seed<s>_h9.csv]
                     -> logs/n5_atlantic/control_seed<s>_eval_h9.log
control, native:    [PY, scripts/evaluate.py, --checkpoint, models/n5_atlantic/control/seed<s>.pt, --train-list,
                     storms_train_expanded.txt, --holdout-list, storms_holdout_h9.txt,
                     --out, results/n5_atlantic/control_seed<s>_native_h9.csv]
                     -> logs/n5_atlantic/control_seed<s>_native_eval_h9.log
atlantic, train:    [PY, -u, scripts/train.py, --train-list, storms_train_n5_atlantic.txt, --val-list, storms_val_expanded.txt,
                     --epochs, 13, --seed, s, --loss, l1, --frame-list, results/n5_atlantic/frame_list_atlantic.txt,
                     --out, models/n5_atlantic/atlantic/seed<s>.pt]
                     -> logs/n5_atlantic/atlantic_seed<s>.log
atlantic, eval:     [PY, scripts/evaluate.py, --checkpoint, models/n5_atlantic/atlantic/seed<s>.pt, --train-list,
                     storms_train_n5_atlantic.txt, --holdout-list, storms_holdout_h9.txt,
                     --out, results/n5_atlantic/atlantic_seed<s>_h9.csv]
                     -> logs/n5_atlantic/atlantic_seed<s>_eval_h9.log
manifest:           logs/n5_atlantic/runs/<arm>_seed<s>.json
```

- **Arguments:** every train/evaluate call carries every path argument (I2). No `--circular-mask` or `--rotate`.
- **Evaluation stdout is captured (B1).** `evaluate.py:23-24` prints `checkpoint:` and `held-out storms:`, and the analysis checks both against the manifest.
- **Atlantic epochs:** the `13` in the atlantic command comes from `epochs_for(ARM_TRAIN_FRAMES)`.

**Manifest schema** (`logs/n5_atlantic/runs/<arm>_seed<s>.json`):
- **Run fields:** `arm`, `seed`, `order`, `commit`, `train_list`, `frame_list_sha256` (null for the control).
- **Files, each with its SHA-256 in `<key>_sha256`:** `train_log`, `native_checkpoint`, `selected_checkpoint`, `selected_epoch`, `h9_csv`, `eval_log`.
- **Control only:** `native_h9_csv`, `native_eval_log`, and `val_mae_revalidated` (epochs 1–25 → unrounded validation MAE).
- **Every path is pinned** to its arm and seed: `train_log`, `eval_log`, `h9_csv`, the native files and the checkpoints, exactly as in the argument lists above. A manifest naming any other path is refused (item 7 A1).
- **Each CSV is content-linked to its eval log.** The `evaluate.py` stdout carries `held-out samples: N` and `overall MAE: x.xx kt` (evaluate.py:50-51), plus a final `wrote <out> (<n> rows)`. The logged MAE must equal the CSV's MAE to 2 decimals, the wrote-path must equal the pinned CSV path, and the row counts must match. A CSV swapped between arms then fails, because its log's MAE belongs to the other arm.
- **Protected hashes:** `protected_before`, `protected_after` (file → SHA-256).

**Analysis, after all 20 runs:**
```
[PY, scripts/analyze_n5.py, --manifest-dir, logs/n5_atlantic/runs, --out-dir, results/n5_atlantic/analysis,
 --frozen-dir, results/storms]
```

**Compute:**
- build ≈ 2.6 h (about 2002 sample times × about 4.6 s);
- 20 runs ≈ 40 min, plus 10 extra H9 evaluations and 250 validation-only re-validations (all 25 epochs × 10 seeds), a few minutes;
- analysis under a minute.

## 9. Analysis script, its hash and its tests

`reports/v2_survey/n5/analyze_n5.py` (r2) becomes `scripts/analyze_n5.py` in the launch PR. It is deterministic, CPU only, with no torch and no random numbers.

**Input refusals:**
- any 2025-or-later ID; any storm outside H9;
- per-storm count drift; duplicate frames;
- **an H9 frame/label set whose SHA-256 differs from the pinned `267f154b…646b04f`**. That pin covers the sorted (crop_path, truth) pairs, identical in all 20 `results/storms/*_h9.csv`.

**Linkage refusals (B1):**
- a manifest whose files are missing or whose SHA-256s differ;
- runs from more than one commit;
- a train list name, or a frame-list SHA, that is not the recorded one;
- protected hashes not equal to the pinned block before and after;
- a train log whose `checkpoint:` path, seed, **exact sorted training-ID set** (pinned per arm), validation IDs, train/val frame counts, frame-list line, epochs, loss/mask/rotation, `END` line or last `saved` line differs from the design;
- **any manifest path that is not the pinned path** for its arm and seed (r3);
- an eval log whose `checkpoint:` is not the manifest's selected (or native) checkpoint, or whose held-out storms are not H9;
- **an eval log whose sample count, overall MAE (2 dp) or `wrote` path/rows do not match its CSV** (r3);
- re-validated values that don't round to the logged ones, **whose unrounded minimum is not `train.py`'s native pick** (r3), or a control selection that is not the odd-grid minimum;
- **identical prediction vectors across any two runs, the control's native CSVs included** (r3). One pair is exempt: when train.py's native pick is an odd epoch, the matched grid picks the same epoch, and seed s's matched and native vectors are then **required** to be identical. Differing vectors are refused (r4). This would otherwise fire on almost every real run: P(all ten native picks even) is about 1/1024.

**Other refusals:**
- logs in UTF-16 or with a BOM, refused by name;
- overwriting outputs;
- an out-dir outside `results/n5_atlantic/`.

**Selected epoch.** It is taken from the last `saved` line (`train.py:140`), not from re-ranking the 2-decimal log values.

**Constants after the build (B3).** Only `ARM_TRAIN_FRAMES` and `FRAME_LIST_SHA256` may change after approval; `ARM_EPOCHS` is computed.
- At approval, the coordinator records the SHA-256 of the r2 blob (both constants `None`).
- At launch, the approved blob must differ from it in exactly those two lines, and the decisions.md entry records both hashes and the diff.
- **What stays manual:** the script prints its own blob SHA-256 into `report.txt`, but it cannot check that value against decisions.md. The coordinator compares them by hand.
- The script refuses to run while either constant is `None`.

**Hash definition.**
- The launch hash is the SHA-256 of the committed blob, computed **in Python from bytes**: `hashlib.sha256(subprocess.run(["git", "show", f"{commit}:scripts/analyze_n5.py"], capture_output=True).stdout)`. The blob id from `git rev-parse <commit>:scripts/analyze_n5.py` is recorded as well.
- Not the working-tree file: under `core.autocrlf=true` with no `.gitattributes`, its bytes differ.
- Not `| sha256sum`: that works only in Git Bash, and a PowerShell 5.1 pipe re-encodes the bytes.
- **At run time:**
  - the script refuses unless `git status --porcelain -- scripts/analyze_n5.py` is empty and `git diff --quiet <commit> -- scripts/analyze_n5.py` passes, for the manifests' commit;
  - it must be run as `scripts/analyze_n5.py` in the repo;
  - it prints the blob SHA-256 and id into `report.txt`.

**Test hooks.** `run(..., design=, git=, root=)` exists for the synthetic tests only; `main()` exposes no override.

**Tests.** `reports/v2_survey/n5/test_analyze_n5.py` has **47 tests, all passing** (`.venv\Scripts\python.exe -I -B -m pytest -q reports/v2_survey/n5/test_analyze_n5.py -p no:cacheprovider`). `-B` is used because `-I` ignores `PYTHONDONTWRITEBYTECODE`, so `-B` is what stops `__pycache__` being written. They build the full run_n5.py layout synthetically under `n5/synthetic/` (deleted after), with a fake git. They cover:
- **Planted outcomes:** MAE gain; **trade-off** (MAE better, bias about 4 kt worse); reverse trade-off; the sign-crossing branch; significant but below SESOI; informative and inconclusive nulls; the budget-limited flag.
- **Reports:** sensitivity and frozen reproducibility outputs; the per-storm ≥ 10 rule; determinism and LF-only output; overwrite refusal.
- **The adversary's exact attack is refused:** the same CSVs for both arms, with 31 EP IDs in the arm's training storms. So are identical vectors.
- **Linkage refusals:** checkpoint path in the train log; eval-log checkpoint; a file changed after its manifest; frame-list SHA; mixed commits; protected hash change; control grid selection; re-validated values vs the log; a dirty or different self.
- **Item 7's three attack variants are refused:**
  - CSV contents swapped between arms per seed, with consistent SHAs: refused on the eval-log MAE content link;
  - the atlantic CSV pointing at the control's native CSV: refused on the pinned path;
  - an eval log at an arbitrary path: refused on the pinned path.
- **Also refused:** identical native vectors; a re-validation minimum that moves off `train.py`'s pick while still rounding to the log.
- **Input refusals:** 2025/2026/validation IDs; a missing frame; a changed label (via the pin); five off-design train logs; UTF-16 and BOM logs; the unfinished design; protected out-dirs.
- **Values:** the budget rule's values and exact sign-flip values.

## 10. Choices: decided, and still open

- **O1, decided (A3): retrain the control**, interleaved, arm order alternating, all 20 runs under one commit. The reasons are session, run-order and code drift:
  - **Run order:** the 10-14 control arm's H9 MAE rose with seed/order (Spearman +0.72, decisions.md:1525, the 18:53 results entry). The 18:57 robustness entry (decisions.md:1571-1591) then showed the ranking result halving and failing on the earlier control seeds alone.
  - **Code drift:** PR #1, PR #4 (check_nan safe), PR #3 (N6 required args and block tests) and PR #5 (TCDataset 2025 block) touch `train.py`, `evaluate.py`, `load_splits`, `TCDataset` and `check_nan.py`, and `excluded_crops.txt` is regenerated.
  - **Comparison:** the frozen members are reported against the retrained control as an unpaired, descriptive check (§6).
- **O2, decided (A4):** box overlap with the 50 km margin (|dx|, |dy| < 650 km), giving 4 frames in the estimate. The frames are chosen from the built manifest.
- **O3, settled by A1:** the test is the paired sign-flip test, for both primaries.
- **Open, for John (the coordinator's offer):** the control's native selection as the primary, instead of the matched grid (§4). My recommendation is the matched grid, because the coarse-grid bias runs against the atlantic arm and has a known direction.

## 11. Known limits

- **Confounds:** the §0 bundle (storms, volume, mix, exposure, density) can't be separated here.
- **Validation is blind at Cat 5.**
- **Weak-storm crops:** frames from weak, sheared storms may include frontal or land cloud that the NaN rule doesn't catch.
- **Matched grid is approximate:** spacing 132 vs 128 steps, and the first candidate at 66 vs 128 steps.
- **Estimates re-fixed after the build:** frame counts and the Marco frames.

## 12. Launch prerequisites (in order; none done)

0. **Process.** `scripts/run_n5.py` and the safe exclusion step are **specification only** in this draft. The launch PR (§13) that implements them, with the matched-grid re-validation, the manifest writer and the exclusion step, **returns to hc-adversary and John for review before any build**.
1. **PR #1, PR #4 (check_nan safe), PR #3 (N6 required args and block tests) and PR #5 (TCDataset 2025 block) merged.** `check.py` PASS on the launch commit.
1b. **H-EP guards land first (item 8, finding 10).** `storms_train_n5_atlantic.txt` is a v2 train list, so all of the following land **before or inside the N5 launch PR**:
   - the H-EP list `storms_holdout_hep.txt`;
   - its I5 forbidden-set entry;
   - its registry role.
2. **New list and registry entry committed before the build:** `storms_train_n5_atlantic.txt` (the 24 IDs of `storms_train_expanded.txt` + the 31 above), and its `VAL_LIST_FOR` entry in `tests/test_list_invariants.py`, mapped to `storms_val_expanded.txt`. The I5 tests must pass with it, and the concurrency/alias checks (F-10) must pass for it.
3. **Build the 31 storms** (`scripts/build_batch.py --sample-interval 3h`, GOES-16 default, no `--force-hurdat2-refresh`, no `--hurdat2-url`). Hash the Atlantic cache before and after.
4. **Exclusion step (B5): all non-spent storms are rescanned; additions only for the 31 new storms (removal refused). By default it runs through PR #4's `check_nan.py` ("check_nan safe"; merge order PR #2 → PR #4 → PR #3 → PR #5).**
   - **What PR #4 already does:** it never loads the six spent 2025 storms, and it refuses any removal of an existing entry.
   - **What it is getting** (from hc-reviewer): `--processed-root`, plus carry-forward of entries for storms absent from the scanned root.
   - **The old `check_nan.py` is never run.** It unlinks the tracked `excluded_crops.txt` first (check_nan.py:8), then loads every storm on disk, the six spent 2025 storms included. Once PR #5's TCDataset 2025 block lands (worker-written; John changed this 23:57), it would raise after the unlink, and every later run would apply no exclusions.
   - **Fallback only, if PR #4's mode is not available at launch:** a launch-PR script (`scripts/n5_exclusions.py`):
     - loads `TCDataset` for the 31 new storms only, so spent storms are never loaded;
     - writes the existing entries plus any new NaN frames to a temp file, then `os.replace`;
     - asserts the diff against HEAD is additions only, all under the 31 new IDs.
   - **Either way:** the 21 existing entries must be unchanged, so H9's pin holds. The regenerated file is committed before launch. It is tracked, written without a trailing newline, and LF in the repo.
5. **Frame list:** the Marco frames are chosen from the built manifest by the A4 rule. Write `results/n5_atlantic/frame_list_atlantic.txt`, then fill `ARM_TRAIN_FRAMES` and `FRAME_LIST_SHA256`.
6. **decisions.md entry:** this design, the r2 and launch blob hashes and their diff, the frame-list SHA and the final counts. Committed (§13).
7. **Launch** `scripts/run_n5.py` on that commit, after the launch PR's review (step 0).

## 13. Launch PR contents

- `scripts/analyze_n5.py` (from this draft, with the two constants filled in).
- `scripts/run_n5.py`, the driver per §8. It includes the validation-only re-selection step and the manifest writer.
- `scripts/n5_exclusions.py`, the safe exclusion step (§12.4), plus the regenerated `excluded_crops.txt`.
- `storms_train_n5_atlantic.txt` and its `VAL_LIST_FOR` entry in `tests/test_list_invariants.py`.
- `results/n5_atlantic/frame_list_atlantic.txt`.
- The decisions.md entry with the blob hash (r2 hash, launch hash, two-line diff) and the frame-list SHA.
- The synthetic tests (`tests/test_analyze_n5.py`, adapted from the n5 draft's tests).
- **Never included:** `reports/v2_survey/n5/__pycache__/` or `n5/synthetic/`. The `__pycache__` folder came from earlier pytest runs; tests now run with `-B`.
- **Before any build,** the PR goes back to hc-adversary and John (§12 step 0).

## 14. Amendment, 2026-10-19 (John): tie rule and abort

This section amends §4, §8 and §9. The text above is unchanged; where they disagree, this section applies. It includes hc-adversary item 16's fixes. John's decision, verbatim:

> "ASK A: approved with two conditions.
> (a) Tie rule: when it fires, the checkpoint train.py actually saved is the one evaluated on H9, never a post-hoc choice; this applies to both arms, including the control's matched-grid re-selection.
> (b) Abort: no H9 evaluation runs until all 20 training runs have finished, so an abort can never follow a look at held-out results; restarted runs keep their original seed and run-order position.
> Record both as a dated amendment in decisions.md with the new design.md and analyze_n5.py hashes, in PR B, before launch."

**Why a tie rule.** Re-validating on the GPU can differ from train.py's in-training validation in the last digits. If two epochs are within that noise, the §4 check "the unrounded re-validation minimum equals train.py's pick" would stop a valid run.

**(a) Tie rule.** It is implemented once, in `analyze_n5.control_selection`, and both `run_n5.py` and the analysis call it.
- **When it fires:** the unrounded re-validation minimum over all 25 control epochs is not train.py's pick (its last `saved`), and the two differ by less than `TIE_KT` = 1e-4 kt. All 25 re-validated values must still round to the logged ones (the §4 check, unchanged).
- **What is evaluated:** train.py's own `models/n5_atlantic/control/seed<s>.pt` becomes the primary's checkpoint. `selected_epoch` is train.py's epoch, whether it is odd or even. The re-validated values never re-rank the epochs once the rule has fired.
- **Weights check:** before the even epoch files are deleted, the driver checks that `seed<s>.pt` holds the same tensors as `epoch_<native>.pt`. That verifies the parse of train.py's last `saved`, which the rule rests on.
- **What is recorded:**
  - `tie_rule` is `{revalidated_argmin, native_epoch, delta_kt}` when the rule fired, and `null` otherwise;
  - `off_grid_primary` is true exactly when the rule fired on an even native epoch;
  - `odd_grid_near_ties` lists the odd epochs within `TIE_KT` of the odd-grid minimum (`analyze_n5.odd_grid_near_ties`).
- **What the analysis refuses** (`analyze_n5.check_training`):
  - a control manifest without `tie_rule`, or with a `tie_rule`, `off_grid_primary` or `odd_grid_near_ties` that does not match the re-validated values;
  - a selected epoch or checkpoint other than train.py's when the rule fired.
- **What `report.txt` lists:**
  - the seeds where the rule fired;
  - the seeds whose primary is therefore off the odd grid (an even epoch, the same checkpoint as the sensitivity);
  - the odd-grid near ties.
- **Outside the window** (|d| ≥ 1e-4 kt), the driver stops and the analysis refuses, as before.
- **Atlantic arm:** there is no re-validation. Its evaluated checkpoint is always train.py's own (§4), so the rule holds by construction.
- **Decided by John, 2026-10-19 15:48** (verbatim: "1. Even-epoch tie: (i), evaluate train.py's own checkpoint, marked off-grid. 2. Fix-commit restart: (i), restart all 20 from the fix commit, only before any H9 output exists. 3. Odd-epoch near tie: keep the first minimum and record it."):
  - **Even native epoch (his 1).** When the tie rule fires and train.py's pick is an even epoch, that seed's primary evaluates train.py's checkpoint, which is off the odd grid. The manifest sets `off_grid_primary`, and `report.txt` lists the seed as an "off-grid primary".
  - **Odd-grid near ties (his 3).** Sometimes train.py's pick is reproduced, but two odd epochs are within `TIE_KT` of each other. train.py never chose between odd epochs: it saved every epoch file, and its `saved` line only marks running minima over all 25. The pre-registered first minimum over the odd grid stands. The manifest records the other epochs as `odd_grid_near_ties`; the analysis verifies them and `report.txt` lists them.
  - **Fix-commit restart (his 2):** `abort-all`, below.

**(b) Two phases, two commands, with committed evidence between them.**
- **Phase 1, `run_n5.py launch`:** trains the 20 runs in the §4 order and evaluates nothing on H9.
  - After each run it re-validates and selects (control), checks the weights, then deletes the even epoch files.
  - It then writes a training record, `logs/n5_atlantic/runs/<arm>_seed<s>.train.json`. The record holds:
    - the commit and analysis blob;
    - the selection, `tie_rule` and `odd_grid_near_ties`;
    - the SHA-256 of every kept file (train log, native checkpoint, the 13 odd epoch files);
    - the protected hashes before and after;
    - `aborted_attempts`.

    A run whose protected hashes change gets no record.
  - Phase 1 refuses to train once H9 has been **looked at**. It checks at launch and again before each run. A look is any of:
    - a pinned H9 output (CSV, eval log or manifest);
    - any aborted H9 evaluation (`aborted/*_eval*/abort.json`);
    - the marker `logs/n5_atlantic/phase2_started.json`.

    `abort-eval` moves pinned outputs aside, so the pinned paths alone can't show a look (hc-adversary item 16b).
  - `launch` then **stops** and prints the 20 record paths.
- **Between the phases:** the 20 records are committed and pushed to phase2-experiments. That pushed commit witnesses that all 20 trainings finished before any H9 look. Retraining after a look would then mean rewriting pushed history.
- **Phase 2, `run_n5.py evaluate`, guard 5e.** It refuses unless:
  - every record is tracked and identical to its blob at HEAD;
  - all 20 records come from one launch commit L, and L is on the first-parent chain of `origin/phase2-experiments`;
  - HEAD is pushed (an ancestor of `origin/phase2-experiments`);
  - `git diff --no-renames -z L HEAD` touches nothing but the 20 records and the decisions docs (`logs/decisions.md`, `reports/decisions/*.md`; never executed; the coordinator's decision). A decisions entry recording the launch then doesn't block phase 2. A code change alongside it still does.
- **Phase-2 marker:** after the entry checks below, and before its first H9 step, `evaluate` creates `logs/n5_atlantic/phase2_started.json` (exclusive create, with HEAD, the launch commit and the time). No command moves or deletes it. `launch`, `abort-run` and `abort-all` refuse while it exists.
- **Phase-2 entry, before the first H9 evaluation:** every training-side check the analysis will later apply is run over all 20 records, through the analysis's own code (`check_training`, `single_commit`, `verify_self` at L). That covers:
  - the pinned paths, file hashes, train list, frame-list SHA and protected hashes;
  - the train-log parse, `control_selection`, `tie_rule` and `odd_grid_near_ties`;
  - the recorded aborts.

  So no training-side refusal can first appear after a look at H9. Then, in the §4 order, phase 2:
  - evaluates each run's recorded checkpoint on H9, plus the control's native one;
  - writes the §8 manifest (the record plus the H9 files, their SHA-256s and `eval_attempts`);
  - checks it with `load_run`.
- **Procedural rule:** no `evaluate.py` or `ensemble_eval.py` is run on `models/n5_atlantic/*` except through `run_n5.py evaluate`. An evaluation to an unpinned path would leave no trace the driver can see.

**Resuming and relaunching.**
- **Resuming:** `launch` and `evaluate` are resumable. Runs that already have a valid record (or manifest) are skipped. Refused until moved aside:
  - any run with files but no record;
  - any record that no longer verifies;
  - any unfinished abort.
- **Relaunching:** `launch` refuses while any record or manifest carries a commit or analysis blob other than HEAD's. A relaunch happens at the launch commit, or after `abort-all` (below).

**Abort procedure (b).** Every abort moves files unchanged to `logs/n5_atlantic/aborted/<arm>_seed<s>_<attempt|eval><k>/` (same relative paths). The steps:
1. It writes `abort.json` there first, with status `pending`, the arm, seed, order, reason, time and every file's SHA-256.
2. It moves the files.
3. It sets the status to `complete`.

An interrupted abort is resumed by running the same command again, and `launch` and `evaluate` refuse while one is pending. `evaluate` also checks, before each run's H9 step, that every evaluation attempt it will record is complete. A lock file, `logs/n5_atlantic/run_n5.lock`, lets only one of `launch`, `evaluate` and the abort commands run at a time. The analysis checks that every recorded attempt's `abort.json` exists and is complete, and `report.txt` lists the attempts.
1. **`abort-run ARM SEED --reason`:** for a run that failed (`TRAIN_FAILED`, a refused re-validation or weights check, a protected-hash change, a crash) or whose record no longer verifies. Its record is moved too. `launch` then trains it again with the **same seed, in the same order position**, at the pinned paths, which are empty again. Its record and manifest list `aborted_attempts`.
2. **`abort-all --reason`, decided by John (his 2: option (i) of hc-adversary item 16).** The procedure for a relaunch from a fix commit. Only while no H9 output exists, it moves every run aside, finished ones included, and `launch` from the fix commit restarts all 20 in the §4 order with their seeds. Records already committed show up as deletions; commit the deletions through a PR before relaunch.
3. **`abort-eval ARM SEED --reason`:** for a crashed H9 evaluation, or a manifest that `load_run` refuses (for example, a CSV changed after its manifest). It applies only to a run whose record verifies and which has H9 outputs without a manifest, or a manifest that `load_run` refuses. The outputs (and that manifest) are moved, never deleted. `evaluate` then evaluates the same recorded checkpoint again. The manifest lists `eval_attempts`. An eval abort is itself a look (above).
4. **What is refused:**
   - `abort-run` and `abort-all` once H9 has been looked at (above), so an abort can never follow a look at held-out results;
   - `abort-run` on a run whose record verifies (finished runs are not aborted one by one);
   - any abort with nothing to move.

**Guards added by this amendment (John approved B and C):**
- **Guard 5 (B):** `git fetch origin`, then HEAD must be on `git rev-list --first-parent origin/phase2-experiments`, which means a merge commit, not a PR branch's own commit. HEAD's SHA and the analysis blob SHA-256 are printed at launch. The coordinator checks the commit against John's approval.
- **Guard 6 (C):** the checkout must be the main checkout (its `.git` is a directory, not a worktree's file). A child python with the runs' environment, started from ROOT, must import `hurricane_intensity` from `ROOT/src`. `torch.cuda.is_available()` must be True.
- **Epoch directory:** it must resolve inside ROOT before any file in it is deleted.

**Hashes.** The SHA-256 of the committed `design.md`, `scripts/analyze_n5.py` and `tests/test_analyze_n5.py` blobs, before and after this amendment, are recorded in `logs/decisions.md` (2026-10-19 entry).
