# N5 result, and committing it

**Needs John, two separate answers:**
1. **Results PR: yes or no.** Commit the N5 outputs through one tier-2 PR, through hc-adversary and then you. **Recommendation: yes.**
2. **Delete the 130 odd-epoch control checkpoints** (5.5 GB, models/n5_atlantic/control/seed*_epochs/) once the results PR has merged. Design §4 says they are "kept until the analysis has run … deleted after the results entry". The 20 final models (seed<s>.pt) are kept. **Recommendation: yes, after the merge.** This is a destructive action, so it needs your explicit yes.

## The verdict (report.txt, unedited; sha256 fde430de92bab775…)

```
PRIMARIES (Bonferroni, alpha 0.025 each, two-sided exact sign-flip; control re-selected on odd epochs):
  overall MAE: control 9.571, atlantic 10.102, d +0.531 (sd 1.120), 97.5% CI [-0.413, +1.479], p 0.17383, SESOI 0.9 -> NULL (inconclusive; effects of 0.9 kt are not excluded)   [d = MAE_atl - MAE_ctl, d < 0 better]
  >96 bias: control -6.905, atlantic -7.853, d -0.947 (sd 4.792), 97.5% CI [-5.515, +3.633], p 0.53125, SESOI 3.0 -> NULL (inconclusive; effects of 3.0 kt are not excluded)   [both negative: d = bias_atl - bias_ctl, d > 0 better]
JOINT VERDICT: NO SIGNIFICANT DIFFERENCE on either primary | BUDGET-LIMITED: the under-training flag fired in >= 5/10 runs of an arm; a 'worse' result is read as 'worse at this budget', not as harm from the added storms; not rerun or rescued
SENSITIVITY (pre-registered; control at its native 25-epoch selection; does not change the verdict):
  overall MAE: control 9.288, atlantic 10.102, d +0.814 (sd 0.891), 97.5% CI [+0.095, +1.587], p 0.01367, SESOI 0.9 -> WORSE
  >96 bias: control -6.758, atlantic -7.853, d -1.095 (sd 4.836), 97.5% CI [-5.587, +3.256], p 0.47852, SESOI 3.0 -> NULL
SECONDARY (alpha 0.05, uncorrected):
  >96 MAE: d +1.267, 95% CI [-0.166, +2.718], p 0.07227
```

The per-storm table and the split between the original three and the added six storms were sent in the 20:30 STATUS. Both are in results/n5_atlantic/analysis/report.txt and per_storm.csv.

## hc-adversary item 19: OK

**An independent recomputation reproduces every number** in report.txt. It used its own metrics, its own exact sign-flip test and its own confidence intervals, and imported nothing from the repo.
- **Agreement:** per-run values agree to 5e-7. Every p-value matches exactly. The confidence intervals agree to about 0.002 kt.
- **No post-hoc checkpoint choice:**
  - every manifest's evaluated checkpoint is the one in its training record;
  - every eval log names that checkpoint;
  - every hash matches the files on disk.
- **The H9 frame set is the pinned 1752 frames.** Its hash recomputes to the pinned value, and it is identical to the 10-14 frozen frame set.
- **The verdict text is exactly what the pre-registered rules produce,** including NULL, the joint verdict, BUDGET-LIMITED, and "WORSE" in the sensitivity analysis, which by design doesn't change the verdict.
- **Nothing protected was touched;** the 2025 test set is untouched.

**Reading notes from the adversary for you (none changes the verdict):**
- **(b) The null is inconclusive, not "no effect".** The overall-MAE confidence interval allows the Atlantic arm to be up to 1.48 kt worse but at most 0.41 kt better. Both analyses put the point estimate on the "worse" side. The sensitivity analysis (Atlantic worse by 0.81 kt, worse in 8 of 10 seeds) is reported beside the verdict and never replaces it. Under BUDGET-LIMITED it reads "worse at this budget".
- **(c) Why the primary and sensitivity analyses differ.** The odd-epoch pick costs the control 0.28 kt of H9 MAE on average (9.571 vs 9.288 at its native epoch). The largest single case is seed 4: H9 MAE is 11.04 at epoch 23 and 9.37 at epoch 24. So at this budget, H9 swings by up to 1.7 kt between adjacent epochs. This is the pre-registered matched grid working as designed.
- **(d) In the original-three split, ">96 kt" means Ian alone.** Nicole and Claudette have no frames above 96 kt.
- **(a) A cosmetic follow-up.** The confidence-interval grid puts each edge up to about 0.002 kt inward. No edge is near a threshold.

## What the results PR would contain (answer 1)

- results/n5_atlantic/analysis/ (9 files) and the 30 H9 CSVs in results/n5_atlantic/;
- logs/n5_atlantic/: the 20 manifests, the 20 training logs, the 30 eval logs and phase2_started.json;
- a logs/decisions.md results entry:
  - the verdict as printed;
  - the output hashes;
  - the adversary's recomputation;
  - reading notes b–d;
  - the budget follow-up (F-31);
- reports/decisions/PR10–PR14.md and this file.

This follows the earlier experiments, whose logs/ and results/ are tracked. The models stay untracked. Committing results/ is tier 2.

## Follow-ups logged (not done)

- **F-31 (from you):** longer training for both arms, as a future separately pre-registered experiment.
- **F-23 to F-30:** carried from earlier.
- **F-32:** the confidence-interval grid edge (note a).
