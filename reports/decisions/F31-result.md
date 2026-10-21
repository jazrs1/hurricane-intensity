# F-31 result (phase 2 done; results PR next)

**No decision needed yet.** The results PR goes to hc-adversary first, then to you.

## Run facts

- **Phase 2:** `run_f31.py evaluate` at H = 079f631 (the PR #28 merge).
  - Ran 22:00:34–22:18:04 EDT, exit 0.
  - 40/40 runs evaluated: 165 H9 evaluations, 40 manifests.
  - `logs/f31/phase2_started.json` written.
  - No aborted training or evaluation attempts.
- **Analysis:** `analyze_f31.py --manifest-dir logs/f31/runs --out-dir results/f31/analysis`, no deviations file.
  - Ran 22:18:19–22:23:38 EDT, exit 0.
  - It printed the analysis blob feee9baf… and commit L = fc71b98.
  - results/f31/analysis/report.txt is identical to the printed output.
- **After the run:**
  - check.py FAST PASS (14/14 protected hashes);
  - block fa31620e8163;
  - main == fbb39c4;
  - tracked tree clean.

## Pre-registered output, verbatim (excerpt: header, primaries, BUDGET-LIMITED, sensitivity, secondary)

The complete output is in results/f31/analysis/report.txt (it goes into the results PR).

```
PRIMARIES (Bonferroni, alpha 0.025 each, two-sided exact sign-flip, native selection; SESOI 0.9 kt, N5's, for comparability):
  P1 control (frozen-ensemble configuration): overall MAE, 4x - 1x: d -0.993 (sd 0.346, n 20, 20 non-zero, p floor 1.9e-06), 97.5% CI [-1.181, -0.814], p 0.00000 -> BETTER   [d = MAE(native_4x) - MAE(native_1x), d < 0 better]
  P2 atlantic: overall MAE, 4x - 1x: d -1.364 (sd 0.611, n 20, 20 non-zero, p floor 1.9e-06), 97.5% CI [-1.699, -1.034], p 0.00000 -> BETTER | BUDGET-LIMITED at 4x in the atlantic arm: read 'at this budget'   [d = MAE(native_4x) - MAE(native_1x), d < 0 better]
  Zero seeds (4x pick <= the 1x budget, d = 0 exactly): control 0, atlantic 0. With m non-zero seeds the p floor is 2^(1-m); p < 0.025 needs m >= 7.
BUDGET-LIMITED (native rule; flagged when the selected epoch > 0.85 x E_k; qualifier at >= half the runs):
  control 1x (E_k 25): 12/20 flagged -> BUDGET-LIMITED
  control 2x (E_k 50): 8/20 flagged
  control 4x (E_k 99): 5/20 flagged
  atlantic 1x (E_k 13): 8/20 flagged
  atlantic 2x (E_k 26): 13/20 flagged -> BUDGET-LIMITED
  atlantic 4x (E_k 52): 12/20 flagged -> BUDGET-LIMITED
  also reported: control matched_4x 4, smoothed_4x control 7, atlantic 12
SENSITIVITY (pre-registered; does not change any verdict):
  P1 sensitivity, 2x: overall MAE, 2x - 1x: d -0.665 (sd 0.396, n 20, 19 non-zero, p floor 3.8e-06), 97.5% CI [-0.882, -0.450], p 0.00000 -> SIGNIFICANT BUT BELOW SESOI (better; real but smaller than 0.9 kt)   [d = MAE(native_2x) - MAE(native_1x), d < 0 better]
  P1 sensitivity, smoothed 4x vs native 1x (rule and budget change together): d -0.515 (sd 0.788, n 20, 20 non-zero, p floor 1.9e-06), 97.5% CI [-0.937, -0.079], p 0.01044 -> BETTER   [d = MAE(smoothed_4x) - MAE(native_1x), d < 0 better]
  P2 sensitivity, 2x: overall MAE, 2x - 1x: d -0.769 (sd 0.617, n 20, 19 non-zero, p floor 3.8e-06), 97.5% CI [-1.110, -0.437], p 0.00001 -> BETTER | BUDGET-LIMITED at 2x in the atlantic arm: read 'at this budget'   [d = MAE(native_2x) - MAE(native_1x), d < 0 better]
  P2 sensitivity, smoothed 4x vs native 1x (rule and budget change together): d -0.951 (sd 0.834, n 20, 20 non-zero, p floor 1.9e-06), 97.5% CI [-1.405, -0.504], p 0.00008 -> BETTER | BUDGET-LIMITED at 4x in the atlantic arm: read 'at this budget'   [d = MAE(smoothed_4x) - MAE(native_1x), d < 0 better]
  S1 sensitivity, atlantic - control at 4x: control at its native 4x pick: d -0.206 (sd 0.330, n 20, 20 non-zero, p floor 1.9e-06), 95% CI [-0.359, -0.051], p 0.01284 -> SIGNIFICANT BUT BELOW SESOI (better; real but smaller than 0.9 kt)   [d = MAE_atl(native_4x) - MAE_ctl(native_4x), d < 0 better]
SECONDARY (alpha 0.05, uncorrected; reported beside the verdicts, never replacing them):
  S1 atlantic - control at 4x: overall MAE (control matched): d -0.260 (sd 0.393, n 20, 20 non-zero, p floor 1.9e-06), 95% CI [-0.444, -0.075], p 0.00912 -> SIGNIFICANT BUT BELOW SESOI (better; real but smaller than 0.9 kt)   [d = MAE_atl(native_4x) - MAE_ctl(matched_4x), d < 0 better]
  S2 atlantic - control at 4x: >96 kt bias (control matched): d +0.772 (sd 2.972, n 20, 20 non-zero, p floor 1.9e-06), 95% CI [-0.619, +2.160], p 0.25911 -> NULL (informative; the 95% CI excludes effects of 3.0 kt or more)   [both negative: d = bias_atl - bias_ctl, d > 0 better]
  N5's verdict, verbatim from its report (N5's verdict stands; same H9, so S1/S2 are not an independent replication):
      overall MAE: control 9.571, atlantic 10.102, d +0.531 (sd 1.120), 97.5% CI [-0.413, +1.479], p 0.17383, SESOI 0.9 -> NULL (inconclusive; effects of 0.9 kt are not excluded)   [d = MAE_atl - MAE_ctl, d < 0 better]
      >96 bias: control -6.905, atlantic -7.853, d -0.947 (sd 4.792), 97.5% CI [-5.515, +3.633], p 0.53125, SESOI 3.0 -> NULL (inconclusive; effects of 3.0 kt are not excluded)   [both negative: d = bias_atl - bias_ctl, d > 0 better]
    JOINT VERDICT: NO SIGNIFICANT DIFFERENCE on either primary | BUDGET-LIMITED: the under-training flag fired in >= 5/10 runs of an arm; a 'worse' result is read as 'worse at this budget', not as harm from the added storms; not rerun or rescued
  S3 control: >96 kt bias, 4x - 1x: d +0.299 (sd 3.645, n 20, 20 non-zero, p floor 1.9e-06), 95% CI [-1.403, +2.012], p 0.71677 -> not significant (better); no SESOI in the design   [both negative: d = bias_4x - bias_1x, d > 0 better]
  S4 control: >96 kt MAE, 4x - 1x: d -1.437 (sd 1.109, n 20, 20 non-zero, p floor 1.9e-06), 95% CI [-1.957, -0.928], p 0.00001 -> SIGNIFICANT (better); no SESOI in the design   [d = >96 MAE(4x) - >96 MAE(1x), d < 0 better]
  S3 atlantic: >96 kt bias, 4x - 1x: d +1.152 (sd 3.511, n 20, 20 non-zero, p floor 1.9e-06), 95% CI [-0.503, +2.799], p 0.15700 -> not significant (better); no SESOI in the design   [both negative: d = bias_4x - bias_1x, d > 0 better]
  S4 atlantic: >96 kt MAE, 4x - 1x: d -1.582 (sd 1.191, n 20, 20 non-zero, p floor 1.9e-06), 95% CI [-2.141, -1.026], p 0.00002 -> SIGNIFICANT (better); no SESOI in the design   [d = >96 MAE(4x) - >96 MAE(1x), d < 0 better]
NO atlantic - control test at 1x or 2x (F-31 is not a rerun of N5). Arm means, side by side, descriptive:
  1x native: control 9.551, atlantic 9.717
  2x native: control 8.887, atlantic 8.948
H9 SPLIT (descriptive): original three (Ian, Nicole, Claudette) vs the six added later, mean overall MAE:
  control native_1x: H3 9.105, added6 9.681
  control native_4x: H3 8.550, added6 8.561
  atlantic native_1x: H3 9.370, added6 9.818
  atlantic native_4x: H3 8.152, added6 8.411
FROZEN ENSEMBLE (committed H9 figures, results/ensemble; unpaired: different seeds, an older code version, the same H9; not a test): overall MAE 8.571, >96 MAE 10.815, >96 bias -7.715
```

## John's note, recorded before the look (goes into the results entry)

John's phase-2 yes, read at 21:49 EDT, before H9 was evaluated at 22:00. It was logged in COORDINATOR.md at that time:

> Control is not BUDGET-LIMITED at 2× or 4×, while atlantic still is at 4× (12/20), so the S1/S2 comparison at 4× remains asymmetric in training.

## Per storm (exploratory): mean overall MAE (kt), native 1× → 4×, from per_storm.csv

| storm | control 1× → 4× | atlantic 1× → 4× |
|---|---|---|
| Beryl AL022024 | 14.49 → 12.74 | 14.28 → 12.15 |
| Claudette AL032021 (orig. 3) | 6.52 → 6.14 | 7.03 → 6.30 |
| Fiona AL072022 | 7.90 → 7.06 | 7.59 → 6.68 |
| Ian AL092022 (orig. 3) | 9.14 → 8.61 | 9.61 → 8.50 |
| Kirk AL122024 | 6.70 → 6.27 | 7.40 → 6.50 |
| Lee AL132023 | 9.43 → 7.99 | 10.14 → 8.40 |
| Nicole AL172022 (orig. 3) | 10.62 → 9.91 | 10.41 → 8.71 |
| Sam AL182021 (traced per epoch 2026-10-14) | 10.01 → 9.11 | 10.28 → 8.99 |
| Iota AL312020 | 7.31 → 6.14 | 6.86 → 5.68 |

- **4× is better than 1× on all 9 storms in both arms.**
- Atlantic native 4× vs control native 4× (descriptive): atlantic is lower on 6 of 9; it is higher on Claudette, Kirk and Lee.

## Descriptive ensembles (ensemble.csv; not tests)

| 20-member ensemble, overall MAE | 1× | 2× | 4× |
|---|---|---|---|
| control | 8.532 | 8.239 | 8.053 |
| atlantic | 8.514 | 8.245 | 7.848 |

The frozen ensemble's committed figure is 8.571. That comparison is unpaired and not a test.

## Next

- The results PR, via hc-survey (runbook step 7). It contains:
  - the 40 manifests;
  - the 165 H9 CSVs and evaluation logs;
  - the phase-2 marker;
  - the analysis outputs, all -text;
  - a decisions.md results entry with the verbatim output and John's pre-look note.
- **How "at this budget" applies** (design §3 line 142 and §7 line 304; corrected after hc-adversary item 29):
  - **As printed:** P2 (4×) and the P2 sensitivities (2×, smoothed 4×) carry BUDGET-LIMITED.
  - **P1:** control 4× is 5/20, so P1 carries no qualifier. Its BETTER is read per §7: the frozen recipe was budget-limited at 1640 steps, on H9, for this code version (control 1× 12/20).
  - **S1/S2:** these are secondaries and carry no qualifier. Their training asymmetry is John's pre-look note.
  - An earlier mapping put control 1× on P1 and atlantic 2×/4× on S1/S2. It came from the item-28 review and went beyond the pre-registration; it is withdrawn.
- Review: hc-adversary, then John.
- The 40 train logs and the kept checkpoints are outside phase-2's outputs. Whether to commit the train logs is a question for the PR, following N5's practice.
