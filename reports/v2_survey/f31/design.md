# F-31: budget study, design (DRAFT r1, not approved, nothing run)

Drafted 2026-10-19 by hc-survey for the coordinator. It goes to hc-adversary, then John. **No compute until John approves this design.**

- **Nothing has been run:** no training, evaluation, model load, build, download or GPU use. The only computation was `power.py` (CPU, synthetic numbers) and reading committed CSVs and logs.
- **John's brief, verbatim:** "F-31 budget study: have hc-survey draft the pre-registration. It covers longer training for both N5 arms and for the frozen ensemble configuration, with epoch sensitivity included."
- **Where it comes from:**
  - the N5 BUDGET-LIMITED entry, decisions.md 2026-10-19 20:16: "longer training for both arms, as a future, separately pre-registered experiment. It is not a rerun of N5";
  - the N5 results entry (20:46) and John's priority note: F-31 "should also cover the frozen ensemble's budget".
- **When accepted:** this text becomes a decisions.md entry before launch, together with the analysis and driver blob hashes (§10).

## 0. What F-31 is, and what it is not

**F-31 is a new experiment, not a rerun or rescue of N5.**
- **N5's verdict stands whatever F-31 finds.** That verdict is "NO SIGNIFICANT DIFFERENCE on either primary | BUDGET-LIMITED", with the sensitivity WORSE "at this budget".
- **No N5 run, checkpoint, CSV or seed is reused as F-31 data.** F-31 trains its own runs with new seeds (§5).
- **Nothing is pooled with N5.**
- **N5's committed files are read-only inputs:** the frame list and the H9 pin. Nothing under `models/n5_atlantic/`, `results/n5_atlantic/` or `logs/n5_atlantic/` is written.

**The question.** Does training longer than N5's 1640-step budget change H9 overall MAE?
- It is asked for each of the two N5 training configurations.
- One of the two is the frozen ensemble's configuration (§1).
- **Epoch sensitivity is built in:** each run is read at three nested budgets and under more than one validation-only selection rule (§3).

**What F-31 cannot do:**
- **Change the frozen ensemble.** `models/storms/expanded/seed0-9.pt` was frozen by the 2026-10-15 entry and tested once on 2025.
- **Change the 2025 result, N5's verdict, or any H-EP slot.**
- **Speak to new storms.** H9 is not an unseen set (§4). If a longer budget looks better here, the retrained model that would follow is a new model. It needs its own test on an unspent set, under its own protocol (§4, option D).

## 1. Configurations: three named, two trained

The brief names three configurations. Two of them have the same recipe.

| Name | Train list | Frames | Validation (selection) | Recipe | Selection used |
|---|---|---|---|---|---|
| Frozen ensemble members (more-distinct-storms expanded arm, `scripts/storms_experiment.sh`) | `storms_train_expanded.txt` | 2081 | `storms_val_expanded.txt` | 25 epochs, L1 | native (best of 25) |
| N5 control (`run_n5.py`) | `storms_train_expanded.txt` | 2081 | `storms_val_expanded.txt` | 25 epochs, L1 | matched odd grid (primary); native (sensitivity) |
| N5 atlantic | `storms_train_n5_atlantic.txt` + `--frame-list results/n5_atlantic/frame_list_atlantic.txt` | 4044 | `storms_val_expanded.txt` | 13 epochs, L1 | native (best of 13) |

All three share the rest of the recipe: ResNet-18 IMAGENET1K_V1, Adam lr 1e-4, batch 32, unmasked, unrotated, `dataset.py` preprocessing unchanged.

- **The frozen-ensemble configuration is the N5 control configuration.** N5 §5 says so: "the retrained control IS that configuration".
  - So F-31 trains **two arms: control (= frozen-ensemble configuration) and atlantic**.
  - The frozen ensemble enters as an **outcome layer** on the control arm: the 10-member ensemble mean at each budget (§6).
  - Its members are **not re-evaluated** anywhere. Their committed H9 figures (`results/ensemble/`) are quoted beside F-31's, unpaired and descriptive.
- **What "the frozen ensemble's budget" can mean here:** would the frozen recipe, trained longer and ensembled the same way, score differently on H9?
  - The arithmetic from `results/ensemble/findings.md` carries over to every budget.
    - The ensemble's bias equals its member-mean bias exactly. So the ensemble's bias question *is* the member question.
    - The ensemble's MAE is at most the member-mean MAE. Only that ensemble gain can move separately from the members.
  - There is one ensemble per arm per budget (two, with the 20 seeds of §5). So the ensemble figures are **descriptive, never tested** (§6).

## 2. Budgets: nested inside one run, step-matched as in N5

**Nesting.** Each run trains once to the longest budget, with `--epoch-dir`, and is read at three nested budgets.
- **Why this is legitimate:** `train.py` has no learning-rate schedule (constant Adam 1e-4, `train.py:88`), and nothing in its loop depends on `--epochs` except the loop count. So the first E epochs of a 99-epoch run *are* an E-epoch run with that seed and data order. That is exact, not approximate.
- **It is not a reproduction of N5's runs:** cuDNN is non-deterministic (decisions.md 2026-10-13, "Groundwork": seed 0 run twice diverges inside the first epoch). Different seeds make that moot anyway (§5).
- **Consequence:** the budget contrasts are **paired within a run**. The 1× checkpoint and the 4× checkpoint come from one trajectory.

**Step-matching rule, extended.** N5's rule is "whole epochs whose step total is nearest 1640" (`analyze_n5.epochs_for(frames, target)`; the function already takes the target). F-31 applies it at k × 1640 for k = 1, 2, 4:

| Budget | Target steps | Control: epochs (steps), 66 batches/epoch | Atlantic: epochs (steps), 127 batches/epoch | Control odd-grid candidates / atlantic candidates |
|---|---|---|---|---|
| **1×** (N5's budget, the frozen recipe) | 1640 | 25 (1650) | 13 (1651) | 13 / 13 |
| **2×** | 3280 | 50 (3300) | 26 (3302) | 25 / 26 |
| **4×** (trained length) | 6560 | **99** (6534) | **52** (6604) | 50 / 52 |

- **Trained length:** the control trains 99 epochs and the atlantic arm 52. Both carry `--epoch-dir`. In N5 only the control did.
- **The 1× rows are N5's budget exactly:** 25 and 13 epochs, 1650 and 1651 steps.
- **Why 4× is the maximum:**
  - **Evidence:** the only longer-budget record usable for a choice is validation-only. The 2026-10-13 60-epoch traces, validation columns only (the 2026-10-18 23:21 entry), show validation MAE flat from about epoch 31 to 60 (41 batches per epoch: about 1270–2460 steps). 4× gives the control 99 passes and the atlantic arm 52, well past that.
  - **Cost:** the phase-1 time stays inside one night (§9).
  - **What these traces are not used for:** their held-out columns (H3, per epoch) are **not used** for this or any other F-31 choice. That follows the 2026-10-18 rule.
- **If 4× is still budget-limited:** that is a finding (§3). F-31 does not extend again; a longer budget would need a new design.

## 3. Selection and epoch sensitivity (validation only)

**Re-validation, as in N5.** After each run, the driver re-validates **every** epoch file on `storms_val_expanded.txt` alone (unrounded MAE), with N5's loader. That is 99 control and 52 atlantic re-validations per seed. Every rule below reads only those values.

**Rules, all on validation, all fixed now:**

| Rule | Definition at budget E_k | Arms | Budgets |
|---|---|---|---|
| **native** | first argmin of re-validated val MAE over epochs 1..E_k | both | 1×, 2×, 4× |
| **matched** | first argmin over the odd epochs ≤ E_k (N5's matched grid, extended) | control | 1×, 2×, 4× |
| **smoothed** (sensitivity) | first argmin over e ≤ E_k of the median of re-validated val MAE over epochs [e−2, e+2] ∩ [1, E_k]; the checkpoint evaluated is epoch e | both | 4× |

**End-to-end check at every budget, not just the last.**
- `train.py` prints `saved` at every new running minimum of unrounded val MAE (`train.py:137-140`).
- So **the last `saved` line at an epoch ≤ E_k is `train.py`'s own pick at budget E_k**.
- The native re-validated argmin must equal it at 1×, 2× and 4×. This is N5's check, extended to the nested budgets.
- **N5's tie rule carries over:** a mismatch smaller than TIE_KT = 1e-4 kt means `train.py`'s pick is evaluated (amendment (a), 2026-10-19 14:20). A larger mismatch is refused before any H9 look.
- **Odd-grid near ties are recorded** and the first minimum kept (John's decision 3, 15:48). The same applies to the smoothed rule.

**Exactly which checkpoints see H9.**
- **The slots:**
  - a control run has 7 slots: native at 1×, 2× and 4×, matched at 1×, 2× and 4×, and smoothed at 4×;
  - an atlantic run has 4: native at 1×, 2× and 4×, and smoothed at 4×.
- **Slots are fixed before any look.** Each slot's epoch is written into the phase-1 training record, before any H9 evaluation (§8).
- **Each distinct epoch is evaluated once.** Coinciding slots share one CSV, and the record maps slot → epoch. So at most 7 H9 evaluations per control run and 4 per atlantic run. That is at most 220 for the 40 runs.
- **Nothing else is evaluated on H9:** no other epoch, and no per-epoch trace. That follows the 2026-10-18 rule ("No held-out storm is evaluated per epoch").

**What "epoch sensitivity" means here.**
- **Budget sensitivity:** each primary is read at 4×. The 2× reading is a pre-registered sensitivity, and the 1×/2×/4× curve is reported.
- **Selection-rule sensitivity:** native vs matched for the control at every budget, and native vs smoothed at 4×.
  - N5 showed H9 MAE can move by up to 1.7 kt between adjacent epochs (seed 4, epochs 23 vs 24).
  - The smoothed rule shows whether a pick less driven by single-epoch validation noise reads differently on H9.
- **Validation-only stability, descriptive:**
  - pick positions across budgets;
  - val MAE at the pick against its ±2-epoch neighbours;
  - the sd of val MAE over the last 15% of each budget;
  - the per-epoch validation trace by arm.
- **What it does not include:** H9 MAE at the neighbours of a pick. That would be a held-out per-epoch trace (§11, open item 3).

**Under-training flag at the new budgets.**
- **Rule, as in N5** (`analyze_n5.py`): a run is flagged at budget E_k when its selected epoch is > 0.85 × E_k. Flagged epochs:
  - control: 1× 22–25, 2× 43–50, 4× 85–99 (odd grid: 23 and 25; 43–49; 85–99);
  - atlantic: 1× 12–13, 2× 23–26, 4× 45–52.
- **The qualifier:**
  - BUDGET-LIMITED at (arm, budget) fires when **≥ 10 of 20 runs** are flagged under the primary rule (native). That is N5's "half the runs" (5/10).
  - For the control, the matched-rule count is also reported.
- **Reference for a flat validation curve:** if the pick were uniform over the budget, P(flag) = 0.15 per run, and P(≥ 10/20) = 0.0002 (and P(≥ 5/10) = 0.0099) (`power.py`). A qualifier firing at 4× is therefore evidence that validation was still improving at 6600 steps.
- **Known before any H9 look:** every flag and qualifier is computed from the phase-1 records. It is recorded in decisions.md before phase 2, as N5's 20:16 entry was.
- **Reading:** a primary at a BUDGET-LIMITED budget is read "at this budget". If 4× fires in an arm, F-31 reports that the arm was still budget-limited at 6600 steps, and does not extend.

## 4. The evaluation set

**Recommendation: H9 (`storms_holdout_h9.txt`, 9 storms, 1752 frames, pin `267f154b…646b04f`), with the reading below.** No other set is usable (the alternatives follow).

**What H9 has already been used for:**
- **Arm choice:** the more-distinct-storms experiment, where H9 favoured the expanded arm. That arm became the frozen ensemble, so H9 is optimistic for that configuration.
- **Exploratory evaluations:** the frozen ensemble on H9 (`results/ensemble/findings.md`), and the variety, imbalance, loss and rotation runs (read on H3/H4/H6, which H9 contains).
- **N5:** 20 runs plus 10 native evaluations.
- **The 1640-step budget itself:** it was first chosen from per-epoch held-out traces on Ian, Nicole and Claudette (decisions.md 2026-10-13 night). The 2026-10-18 23:21 entry bounds that favour near zero: validation alone gives the same window shape and the same budget. **F-31's 1× budget inherits exactly that exposure**, no more.
- **Sam AL182021** was traced per epoch on 2026-10-14. Its per-storm row is read with that in mind, as in N5.
- **F-31's control at 1× is another H9 look at the frozen configuration,** by new runs. No frozen member is re-evaluated.

**What reusing H9 means for F-31:**
- **Absolute H9 levels are not estimates for new storms.** No F-31 number is set beside the 2025 result or the site's 9.00 kt.
- **Every pre-registered quantity is a difference** between budgets of one configuration (the primaries), or between arms at one budget (secondary). Each checkpoint is chosen on validation only.
  - **Budgets above 1640 steps were seen once:** when 1640 was chosen (2026-10-13 night), per-epoch held-out H3 traces were read up to 60 epochs (about 2460 steps). The choice made then was 1640, and the 2026-10-18 23:21 entry bounds that favour near zero.
  - **Not chosen on H9 or H3:** the 2× and 4× ladder, the 4× cap (validation columns only, §2), every selection rule here and the seeds.
  - So **the F-31 contrasts are not selected on H9**. The earlier looks bias H9's levels for the frozen configuration; they do not pick F-31's comparisons.
- **The inference is over training randomness (seeds), on these 9 storms.** "Longer is better on H9" means better on Ian, Sam, Beryl and the rest. H9's > 96 kt frames are dominated by a few storms (in the original three, > 96 kt is Ian alone; N5-result note (d)). The tests do not generalise over storms. The same was true of N5.
- **F-31 was prompted partly by N5's H9 outcome.** It was pre-registered after John read N5 (his priority note) as well as because of the BUDGET-LIMITED flag known before the look.
  - The **primaries** (budget effects within a configuration) were not the contrast N5 looked at.
  - The **atlantic − control contrast at 4×** was. On the same nine storms, it is not independent confirmation or refutation of N5. It is pre-registered as a **secondary** for that reason (§5).

**Alternatives considered:**

| Option | Status | Why not |
|---|---|---|
| A. Unspent Atlantic storms, 2018–2024 | none usable | After N5, the survey §1 leftovers with hurricane frames are Bonnie AL022022 (aliased to EP042022), Rebekah AL192019 (70 kt extratropical) and Pablo AL182019 (GOES-16 edge). The rest are 76 tropical storms and depressions with no frames > 96 kt, so they could not test the high end, and their MAE mix is unlike H9's. All 31 N5 storms are atlantic-arm training. |
| B. Validation (`storms_val_expanded.txt`) | not an evaluation set | It is the selector for every rule in §3. |
| C. 2025 test set (`storms_test_2025.txt`) | **SPENT; never evaluated again** | The 2026-10-16 entry. TCDataset raises on the six IDs (`dataset.py:16`, `SPENT_2025`). F-31 never passes the list. |
| H-EP (`storms_holdout_hep.txt`) | **sealed; not used** | It is scored once under `reports/v2_survey/hep/protocol.md`. F-31's checkpoints fit none of its slots: E1 is the frozen members, E2c/E2a are N5's, and E3 is a later Atlantic + East Pacific arm. They are never evaluated on it. |
| D. 2026 Atlantic season | not part of F-31 | No 2026 imagery may be downloaded or cropped under the current rules, and the 2026 best tracks are not final before spring 2027. **Offered to John as a possible future confirmatory set** for whichever budget F-31 favours, under its own single-pass protocol designed before any 2026 storm is built. Not proposed here. |

**Guards that keep F-31 off the spent and sealed sets:**
- the driver passes only `storms_holdout_h9.txt`;
- the analysis refuses any ID outside H9, any 2025-or-later ID and any EP/CP ID, and checks the H9 pin;
- TCDataset refuses the spent 2025 IDs;
- the H-EP crops are built only under `data/hep_sealed/`, which no F-31 command names;
- **procedural rule:** no `evaluate.py` or `ensemble_eval.py` on `models/f31/*` except through `run_f31.py evaluate`.

## 5. Hypotheses, tests, seeds and power

**Primaries: two, Bonferroni α = 0.025 each.** Both are H9 pooled overall MAE (1752 frames), per run, native selection, two-sided:

| Primary | Statistic, per seed s | Better means | SESOI (power-derived, not a meaningfulness claim) |
|---|---|---|---|
| **P1, control (= frozen-ensemble configuration)** | d = MAE(4×) − MAE(1×), same run | d < 0 | 0.9 kt |
| **P2, atlantic** | d = MAE(4×) − MAE(1×), same run | d < 0 | 0.9 kt |

- **Test:** exact paired sign-flip over all 2^20 sign assignments (n = 20), with 97.5% CIs by test inversion, as in N5.
- **Zeros:** a seed whose 4× pick is ≤ E_1 has d = 0 exactly. Zeros make the test conservative.
- **The SESOI is N5's, for comparability:** John's reading of N5 was "the CI rules out an overall-MAE gain larger than about 0.4 kt at this budget".
- **Expectation, stated in advance:** both d < 0, and larger in the atlantic arm, which made 13 passes against the control's 25 and fired the flag in 5/10 runs. Both tests are two-sided.
- **Why overall MAE only:** that is the quantity the budget-limited reading and John's reading are about.
  - The > 96 kt bias budget effect is a secondary (S3).
  - The 2026-10-13 held-out trace's remark that "more training does nothing for the strong end" is **not** used as a reason; that trace is held-out (§2).

**Secondary** (α 0.05, uncorrected, labelled secondary; reported beside the verdict, never replacing it):
- **S1: atlantic − control, H9 overall MAE, at 4×,** control on the matched odd grid (N5's primary rule), paired by seed. SESOI 0.9.
- **S2: atlantic − control, H9 > 96 kt bias, at 4×,** with N5's sign-crossing branch. SESOI 3.0.
- **S3:** > 96 kt bias, 4× − 1×, per arm.
- **S4:** > 96 kt MAE, 4× − 1×, per arm.

Every S1/S2 line in `report.txt` carries N5's verdict beside it, verbatim, with "N5's verdict stands" (§7).

**Pre-registered sensitivities** (do not change any verdict):
- P1/P2 at 2× (d = MAE(2×) − MAE(1×));
- P1/P2 with the smoothed rule at 4× (d = MAE(4×, smoothed) − MAE(1×, native));
- S1 with the control at its native 4× pick.

**Seeds: 20 per arm, seeds 10–29. 40 runs.**
- **New seeds:** N5 used 0–9. Reusing them would give F-31 the same initial weights and batch orders as N5's runs, though not the same runs (cuDNN). New seeds keep F-31 independent of N5, and keep it from looking like N5's seeds rerun.
- **Order:** round-robin by seed, with arm order alternating (C/A, A/C, …), all under **one commit**, as N5.
- **Why 20 and not 10:** at the sd N5 actually observed, n = 10 gives about 45% power at 0.9 kt (table below). N5 planned on 0.85 / 2.84 and observed 1.12 / 4.79, so it under-planned on both. 20 seeds roughly double phase 1 (§9). **n = 10 (seeds 10–19) is open item 1 for John.**
- **Pairing:** P1/P2 are within-run. S1/S2 pair atlantic seed s with control seed s. As in N5, that pairing is valid because it was fixed in advance; the seed carries little shared variance.

**Power** (`power.py`; N5 §5's method: normal model, exact two-sided sign-flip, seed 20261019, 2000 reps per cell, 80% MDE from 1000 reps per step). The informative-null column is the rate at which the CI lies inside ±SESOI when the true difference is 0.

**The sd inputs, all observed:**
- **0.54:** two selections within one run (N5 control, native − matched, seeds 0–9; picks 0–5 epochs apart). This is the optimistic row: 4× and 1× picks will be further apart.
- **0.85:** N5's planning case.
- **1.12:** N5's observed atlantic − control sd.
- **Planning value: 1.12.**

**P1/P2, overall MAE, 4× − 1× within one configuration (α 0.025): at n = 20, 87–100% power to detect a 0.9 kt difference (plan on 87%). At n = 10: 45–98% (plan on 45%).**

| n | sd_diff | Scenario | 0.5 kt | 0.7 kt | **0.9 kt** | 1.2 kt | 80% MDE | Informative null at 0 (SESOI 0.9) |
|---|---|---|---|---|---|---|---|---|
| 20 | 0.54 | two selections within one run (optimistic) | 0.95 | 1.00 | 1.00 | 1.00 | 0.45 | 1.00 |
| 20 | 0.85 | N5 planning case | 0.58 | 0.89 | 0.99 | 1.00 | 0.65 | 0.97 |
| **20** | **1.12** | **N5 observed (planning)** | 0.36 | 0.64 | **0.87** | 0.98 | 0.85 | 0.73 |
| 10 | 0.54 | | 0.58 | 0.89 | 0.98 | 1.00 | 0.65 | 0.96 |
| 10 | 0.85 | | 0.26 | 0.47 | 0.73 | 0.94 | 1.00 | 0.44 |
| 10 | 1.12 | | 0.15 | 0.31 | 0.45 | 0.73 | 1.35 | 0.13 |

**S1, overall MAE, atlantic − control at 4× (secondary, α 0.05): at n = 20, 76–99% power at 0.9 kt (plan on 92%, sd 1.12).**

| n | sd_diff | 0.5 kt | **0.9 kt** | 1.2 kt | 80% MDE | Informative null at 0 (SESOI 0.9) |
|---|---|---|---|---|---|---|
| 20 | 0.85 | 0.71 | 0.99 | 1.00 | 0.60 | 0.99 |
| **20** | **1.12** | 0.46 | **0.92** | 1.00 | 0.80 | 0.86 |
| 20 | 1.40 | 0.33 | 0.76 | 0.96 | 0.95 | 0.58 |
| 10 | 0.85 | 0.37 | 0.84 | 0.97 | 0.85 | 0.68 |
| 10 | 1.12 | 0.23 | 0.60 | 0.85 | 1.15 | 0.29 |
| 10 | 1.40 | 0.17 | 0.43 | 0.64 | 1.40 | 0.10 |

**S2, > 96 kt bias, atlantic − control at 4× (secondary, α 0.05): at n = 20, 57–99% power at 3.0 kt (plan on 76%, sd 4.79). An informative null at the observed sd is about a coin toss (0.50).**

| n | sd_diff | 2.0 kt | **3.0 kt** | 4.0 kt | 5.0 kt | 80% MDE | Informative null at 0 (SESOI 3.0) |
|---|---|---|---|---|---|---|---|
| 20 | 2.84 | 0.85 | 0.99 | 1.00 | 1.00 | 1.9 | 0.99 |
| **20** | **4.79** | 0.43 | **0.76** | 0.95 | 0.99 | 3.1 | 0.50 |
| 20 | 6.00 | 0.30 | 0.57 | 0.80 | 0.94 | 3.9 | 0.17 |
| 10 | 2.84 | 0.48 | 0.83 | 0.97 | 1.00 | 2.9 | 0.68 |
| 10 | 4.79 | 0.21 | 0.43 | 0.65 | 0.84 | 4.8 | 0.07 |
| 10 | 6.00 | 0.16 | 0.29 | 0.46 | 0.63 | 6.2 | 0.01 |

Notes:
- **Monte Carlo error:** about ±0.01 at p ≈ 0.8 with 2000 reps (√(0.16/2000)). The 80% MDE comes from a 0.05 kt (0.1 kt for the bias) grid at 1000 reps, so read it to that step.
- **The normal model is an approximation.** N5's per-seed differences were not normal (seed 4's +1.7 kt swing). The power figures are planning ranges, not guarantees.
- **The S1/S2 rows assume N5's 1× sd also holds at 4×.** No data says whether a longer budget shrinks or widens it.

## 6. Descriptive outputs (labelled; paired sign-flip p printed where paired, not tests)

- **Budget curve per arm and rule:** mean H9 overall MAE, > 96 kt MAE and bias, fixes-only MAE, and per-band MAE/bias at 1×, 2× and 4×.
- **The frozen-ensemble layer (control arm):**
  - **The ensembles:** at each budget, two 10-member ensemble means, seeds 10–19 and 20–29 (the frozen ensemble's size), and the 20-member mean, under native selection.
  - **Metrics:** overall MAE, fixes-only MAE, > 96 kt MAE/bias, Spearman > 96 kt, < 64 kt MAE/bias, and the ensemble's gain over its member mean.
  - **Spread:** a deterministic leave-one-seed-out jackknife range for the 10-member figures.
  - **The atlantic arm:** the same figures.
  - **The frozen ensemble's committed H9 figures** (`results/ensemble/expanded10_h9_metrics.csv`, overall MAE 8.57) are quoted beside them, labelled **unpaired: different seeds, an older code version, and the same H9**. They are not a test.
- **1× atlantic − control on the matched grid:** a descriptive re-measure of N5's contrast by new runs. It is reported once, labelled "not N5, not pooled with N5". A difference from N5's d = +0.531 is run-to-run and code-version variation.
- **Selection:** the epoch picked by every rule at every budget, the flag counts per budget and rule, tie-rule firings and near ties, and the validation-only stability figures of §3.
- **H9 split** (Ian/Nicole/Claudette vs the six added later) and **per storm** (> 96 kt figures only with ≥ 10 frames > 96 kt; Sam's caveat), as in N5.

## 7. Interpretation, fixed in advance

**Per primary:** N5 §7's four outcomes, with α 0.025 and the 97.5% CI against ±0.9 kt:
- **BETTER / WORSE:** p < 0.025.
- **SIGNIFICANT BUT BELOW SESOI:** p < 0.025 and the CI inside ±0.9.
- **NULL (informative):** p ≥ 0.025 and the CI inside ±0.9.
- **NULL (inconclusive):** p ≥ 0.025 and the CI reaching ±0.9.

The BUDGET-LIMITED qualifier is appended per arm at 4× (§3).

**Readings:**

| Result | Reading | What it does not mean |
|---|---|---|
| P1 BETTER | The frozen recipe was budget-limited at 1640 steps, on H9, for this code version. | That the frozen ensemble is worse than it could be on new storms, or that it should be replaced. The frozen ensemble is not changed. A longer-budget ensemble would be a new model needing its own unspent test (§4 D). |
| P1 WORSE | Longer training, with native selection, reads worse on H9. The smoothed sensitivity and the val-at-pick-vs-neighbours figures (§3) show whether selection optimism (§12), rather than training itself, drove it. | That 1640 steps is optimal, or that longer training harms. |
| P1 NULL (informative) | Training 4× longer does not change the frozen recipe's H9 MAE by 0.9 kt or more. | That 1640 steps is optimal. |
| P2 BETTER | The atlantic configuration gains from a longer budget. | Anything about N5's verdict. |
| S1 at 4× shows no atlantic deficit (or a gain), while N5 at 1× showed d = +0.531 | Descriptive: the contrast at 4× differs from the contrast at 1×. | That N5 was wrong. N5's verdict was "at this budget" and stands. On the same H9, this is not an independent replication. |
| S1 at 4× WORSE | The atlantic deficit persists at 4× on H9. | Harm from the added storms: N5 §0's bundle (frames, intensity mix, exposure, density) still confounds it. |
| 4× still BUDGET-LIMITED in an arm | That arm was still improving on validation at 6600 steps. | A reason to extend F-31. A longer budget needs a new design. |

- **Attribution:** every reading is about these two recipes on H9. Nothing here is about storm count (N5 §0).
- **Precedence:** secondary, sensitivity and descriptive results are reported beside the verdicts and never replace them.

## 8. What carries over from N5 unchanged, and what has to differ

The launch-safety design was closed by John after N5 (decisions.md 2026-10-19 17:03: "only a concrete defect reopens it"). F-31 reuses it. Only the rows marked **differs** are new, and each is forced by nesting or by n = 20.

| Machinery (N5 §8, §9, §14) | F-31 |
|---|---|
| Two phases: `launch` trains and writes records, no H9; records committed and pushed; `evaluate` refuses unless HEAD differs from L only by the records and the decisions docs | unchanged |
| Records read-only once written; `phase2_started.json` (exclusive create) marks the look; launch refuses while it exists | unchanged (under `logs/f31/`) |
| abort-run / abort-all (only before any look) / abort-eval; `abort.json` pending → moved → complete; one lock | unchanged |
| Dead-end rule (A), `deviations.json` | unchanged in form. **Differs:** a bad record drops only that run from P1/P2 (within-run contrasts), and that seed pair from S1/S2. |
| Guards: first-parent chain after a fetch, main checkout, child imports `hurricane_intensity` from ROOT/src, CUDA available, epoch dir inside ROOT, refuse existing outputs, protected hashes before/after every run | unchanged |
| `seed<s>.pt` equals `epoch_<native>.pt` | unchanged, at 4×. **Differs:** at 1×/2× the evaluated file is the epoch file. |
| Re-validation of every epoch; argmin == `train.py`'s pick; tie rule TIE_KT; odd-grid near ties kept as the first minimum | **differs:** applied at three nested budgets, against the last `saved` ≤ E_k (§3), and both arms carry `--epoch-dir` |
| Pinned paths per (arm, seed); CSV ↔ eval-log content link (MAE 2 dp, sample count, `wrote` path/rows); identity refusal across runs | **differs:** paths are pinned per (arm, seed, epoch). Slots map to epochs in the record, so coinciding slots share one CSV and the "same-pick exemption" becomes structural. Distinct epochs must give distinct vectors. |
| Exact sign-flip, CIs by inversion, Bonferroni, verdict table | **differs:** n = 20 (2^20 assignments; `analyze_n5.sign_matrix(n)` already takes n) |
| verify_self, hash as SHA-256 of `git show` bytes, LF outputs, `.gitattributes -text` for records/logs/CSVs | unchanged, for `scripts/analyze_f31.py` and `scripts/run_f31.py` (plus `scripts/analyze_n5.py` / `scripts/run_n5.py` if imported; §10) |
| Disk: delete non-candidates after re-validation; keep candidates | **differs:** kept = the selected epochs (≤ 7 control, ≤ 4 atlantic per run). The SHA-256 of every epoch file is recorded before deletion (§9). |
| H-EP guards, 2025 block, protected set | unchanged |

## 9. Compute, disk and paths

**Compute, RTX 5070, from N5's own logs:**
- **Measured in N5:**
  - control 25 epochs took 114 s (19:09:07–19:11:01, about 4.6 s/epoch with epoch files), and its 25 re-validations about 29 s (about 1.2 s each);
  - atlantic 13 epochs took 102 s (about 7.8 s/epoch);
  - phase 1 was 41 min for 20 runs.
- **F-31 per seed:**
  - control 99 × 4.6 s ≈ 7.6 min, plus 99 re-validations ≈ 2.0 min;
  - atlantic 52 × 7.8 s ≈ 6.8 min, plus 52 re-validations ≈ 1.0 min;
  - about 17.4 min per seed.
- **Phase 1 (40 runs): about 5.8 h; plan on 5.5–7 h**, allowing for back-to-back GPU heat (decisions.md 2026-10-13 run-order note). n = 10 would be about 2.9 h.
- **Phase 2:** at most 220 H9 evaluations at about 6.5 s each (N5: 30 in 3 min 14 s), so at most 25 min.
- **Analysis:** CPU only, a few minutes (2^20 sign-flip enumeration and CI inversion).
- **No build and no download:** every storm was built for N5.

**Disk (checkpoint 44.8 MB):**
- **Transient, one run at a time:** at most 99 epoch files ≈ 4.4 GB.
- **After re-validation:** the driver writes every epoch file's SHA-256 into the run's record, keeps only the selected epochs (read-only), and deletes the rest.
- **Kept:** at most 11 files per seed pair × 20, plus 40 `seed<s>.pt`: **at most about 12 GB**. The C: drive had 781 GB free on 2026-10-19.
- **Alternative:** keep all epoch files until the results entry (about 134 GB), so every candidate stays re-verifiable. It is affordable; open item 4.

**New paths (only these are written):**
- `models/f31/{control,atlantic}/seed<s>.pt` and `seed<s>_epochs/epoch_NN.pt`;
- `logs/f31/<arm>_seed<s>.log`;
- `logs/f31/<arm>_seed<s>_e<NN>_eval_h9.log`;
- `logs/f31/runs/<arm>_seed<s>.train.json` (phase-1 record) and `<arm>_seed<s>.json` (manifest);
- `logs/f31/phase2_started.json`, `deviations.json`, `aborted/`, `run_f31.lock`;
- `results/f31/<arm>_seed<s>_e<NN>_h9.csv` and `results/f31/analysis/`.

**Inputs read, unchanged:**
- `storms_train_expanded.txt`, `storms_train_n5_atlantic.txt` and `storms_val_expanded.txt`;
- `storms_holdout_h9.txt`;
- `results/n5_atlantic/frame_list_atlantic.txt` (SHA-256 `84e5f730…cc87`, pinned);
- `excluded_crops.txt` (SHA-256 of the committed bytes at fbe0da9 `10200977…0cea`, blob id `cc9dfcff`; the H9 pin, 2081 control, 940 validation and 4044 atlantic frames must all hold at launch);
- `results/ensemble/expanded10_h9_metrics.csv` and `results/storms/expanded_seed*_h9.csv` (descriptive only).

**Never written:**
- `best_model.pt`, `models/baseline_9kt.pt`, `results/heldout_predictions.csv`;
- the 14 protected files (COORDINATOR.md block), checked before and after every run;
- `models/storms/*` (including the frozen members), `results/storms/*`, `results/ensemble/*`, `results/final_2025/*`, `site/`;
- **everything N5 wrote or committed:** `models/n5_atlantic/*`, `results/n5_atlantic/*`, `logs/n5_atlantic/*`;
- `data/*` (including `data/processed`, `data/interim/hurdat2_atlantic.txt` and `data/hep_sealed`), `excluded_crops.txt`, and `results/hep/*`.

The driver's output check refuses any path outside the three `f31` roots, as `run_n5.py`'s OUTPUT_ROOTS does.

## 10. Analysis script and driver: the plan

This PR is the design only. The scripts come in the launch PR, after John approves this design. **Both are hashed before launch, as with N5.**

**`scripts/analyze_f31.py`** (deterministic, CPU, no torch, no random numbers):
- **Reused from `scripts/analyze_n5.py` by import:** `epochs_for`, `sign_matrix`/`signflip_p`/`signflip_ci` (these take n), `load_csv` (H9 pin and input refusals), `read_text_strict`, `check_eval_log`, `metrics`, `per_storm`, `verify_self` and the hash definition, `PROTECTED`, and the deviations loader.
  - Where a function hard-codes N5 (pinned paths, `SEEDS = range(10)`, `TRAIN_LISTS` keyed to `n5_atlantic`, `CONTROL_EPOCHS = 25`), F-31 gets its own function instead of editing `analyze_n5.py`. `analyze_n5.py` is not modified: N5's results cite its blob.
  - **Because F-31's behaviour depends on imported code,** the blob SHA-256 of `scripts/analyze_n5.py` (`01306415…1288`, blob id `679e2169`: N5's launch blob, unchanged at fbe0da9, verified from `git show` bytes) is pinned in `analyze_f31.py`, checked at run time, and recorded in `report.txt`.
- **New:**
  - nested-budget selection (native, matched, smoothed) from the record's re-validation table, with the `saved`-line cross-check at each budget;
  - the slot → epoch map, with each distinct epoch evaluated exactly once;
  - a refusal of any H9 CSV for an epoch that is not a slot;
  - the per-budget flag and qualifier;
  - within-run P1/P2;
  - S1–S4 and the sensitivities;
  - the ensemble layer with its jackknife;
  - `report.txt` with N5's verdict quoted beside S1/S2.
- **Synthetic tests** (`tests/test_analyze_f31.py`, the full `run_f31.py` layout built under a per-process temp folder, as N5's tests do):
  - planted budget gain in one arm only, planted null, and planted 4× BUDGET-LIMITED;
  - nested selection against hand-computed picks, the `saved`-line cross-check, and tie-rule firing at 1×;
  - a refused per-epoch H9 CSV, and refused 2025, EP and validation IDs;
  - exact p values at n = 20 against brute force on small cases;
  - determinism and LF output.

**Argument lists per seed s** (`PY` = `sys.executable`; list arguments, binary-captured stdout, as N5 §8):

```
control, train:   [PY, -u, scripts/train.py, --train-list, storms_train_expanded.txt, --val-list, storms_val_expanded.txt,
                   --epochs, 99, --seed, s, --loss, l1, --out, models/f31/control/seed<s>.pt,
                   --epoch-dir, models/f31/control/seed<s>_epochs]                  -> logs/f31/control_seed<s>.log
atlantic, train:  [PY, -u, scripts/train.py, --train-list, storms_train_n5_atlantic.txt, --val-list, storms_val_expanded.txt,
                   --epochs, 52, --seed, s, --loss, l1, --frame-list, results/n5_atlantic/frame_list_atlantic.txt,
                   --out, models/f31/atlantic/seed<s>.pt, --epoch-dir, models/f31/atlantic/seed<s>_epochs]
                                                                                     -> logs/f31/atlantic_seed<s>.log
re-select:        validation-only re-validation of every epoch file -> record: val_mae_revalidated, slots -> epochs
eval, per distinct slot epoch NN:
                  [PY, scripts/evaluate.py, --checkpoint, models/f31/<arm>/seed<s>_epochs/epoch_NN.pt,
                   --train-list, <arm's train list>, --holdout-list, storms_holdout_h9.txt,
                   --out, results/f31/<arm>_seed<s>_eNN_h9.csv]               -> logs/f31/<arm>_seed<s>_eNN_eval_h9.log
analysis:         [PY, scripts/analyze_f31.py, --manifest-dir, logs/f31/runs, --out-dir, results/f31/analysis]
```

No `--circular-mask` or `--rotate`. The 99 and 52 come from `epochs_for(frames, 4 * 1640)`.

**`scripts/run_f31.py`:** `run_n5.py`'s phases, guards and abort commands, with F-31's paths, epochs and seeds, `--epoch-dir` for both arms, and the nested re-selection. The same rule applies: import what is not N5-bound, pin `run_n5.py`'s blob if imported, and never edit `run_n5.py`.

**An alternative to re-validating from files** (not recommended): add an unrounded per-epoch validation line to `train.py`. That is a training-code change between N5 and F-31, and re-validation is cheap (about 3 min per seed).

## 11. Open items for John (recommendation first)

1. **Seeds: 20 per arm (recommended) or 10.** n = 10 halves phase 1 (about 2.9 h). The cost is power: at the observed sd of 1.12 kt, a 0.9 kt budget effect has about 45% power at n = 10, against 87% at n = 20.
2. **Primaries: the two within-configuration budget effects (recommended), or three** (add atlantic − control at 4× as P3, each at α 0.0167). Making the N5 contrast primary on the same H9, right after N5's look, reads as a second attempt at N5's question. As a secondary it is reported in full, beside N5's verdict.
3. **Epoch sensitivity: validation-only plus the per-budget and per-rule H9 slots (recommended).** The alternative, H9 at the neighbours of each pick (±1 epoch), would show the adjacent-epoch swing directly. But it is a held-out per-epoch trace, which the 2026-10-18 rule forbids. It would need John to lift that rule for F-31, and I don't recommend it.
4. **Disk: keep only the selected epoch files, about 12 GB (recommended), or all of them until the results entry, about 134 GB.**
5. **Evaluation set: H9 as in §4 (recommended; the only usable set).** Option D (a future 2026-season set) would be a separate protocol.

## 12. Known limits

- **H9 is reused** (§4). The results are about these 9 storms and these two recipes.
- **The confounds remain:** N5 §0's bundle confounds every atlantic − control reading.
- **Validation is blind at Cat 5** (no frames ≥ 137 kt).
- **The budget ladder is coarse:** 1×, 2×, 4×. An optimum between them is not located.
- **Selection pool grows with the budget:** at 4× the native rule takes the minimum of 99 (control) or 52 (atlantic) validation values; at 1×, of 25 or 13. The minimum of a larger noisy pool is more optimistic on validation (winner's curse), so a 4× − 1× contrast bundles "more training" with "more candidates". The smoothed rule and the validation-stability figures (§3) are there to show how much of a contrast is selection.
- **Within-run zeros:** in a run whose 4× pick is ≤ 25 (control) or ≤ 13 (atlantic) epochs, the budget effect is exactly 0 for that seed.
- **Code version:** F-31 runs on its own launch commit. Differences from N5's or the frozen members' code are part of why 1× is re-measured inside F-31, rather than taken from N5.

## 13. Launch prerequisites (in order; none done)

0. This design approved by John after hc-adversary's review.
1. **Launch PR:**
   - `scripts/analyze_f31.py` and `tests/test_analyze_f31.py`;
   - `scripts/run_f31.py` and its tests;
   - the `.gitattributes` rows for `results/f31/` and `logs/f31/`.
   It goes back to hc-adversary and John before anything runs.
2. `check.py` PASS on the launch commit. The H9 pin, the frame-list SHA and the four frame counts hold.
3. **decisions.md entry:** this design, the analysis, driver and (if imported) N5 blob hashes, and the seeds and order. Committed.
4. **Phase 1** (`run_f31.py launch`) on that commit, after John's yes. Then the records are committed and pushed.
5. **The flags and qualifiers known before the look** (§3) are recorded in decisions.md. Then John's separate yes to phase 2.
6. **Phase 2** (`run_f31.py evaluate`), then `analyze_f31.py`, then the results entry.

## Files in this draft

- `design.md`: this file.
- `power.py`: the §5 power tables. Run it with `.venv\Scripts\python.exe -I -B reports/v2_survey/f31/power.py`. CPU only; it reads nothing. Its output (about 12 min) is `power_output.txt`.
