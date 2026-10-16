# Final test: the 2025 GOES-19 storms

Run once, as designed in `logs/decisions.md` (2026-10-15 design, its
corrections and approvals; 2026-10-16 pre-evaluation note).

Frozen before the run, and checked by hash just before it:
- the ten expanded-arm checkpoints (`results/ensemble/members_sha256.txt`;
  `members_sha256_at_run.txt` here is identical);
- `scripts/ensemble_eval.py` (658770d7...), `scripts/ensemble_report.py`
  (854c166f...) and `best_model.pt` (f95e6c7e...).

Commands as pinned in the corrections entry. Outputs in this folder,
logs in `logs/final_2025/`.
- `ensemble_2025.csv` / `ensemble_2025_metrics.csv`: the test.
- `best_model_2025.csv` / `best_model_2025_metrics.csv`: the reference.
  The metrics come from the same frozen `ensemble_report.py`, run on a
  copy of the CSV with the prediction duplicated as its one member
  column (`best_model_2025_report_input.csv`).

**best_model.pt is a reference, not part of the test.**

No result here is inferential: six storms of autocorrelated hourly
frames and one model give no sampling distribution. The comparison with
H9 is descriptive, and H9's numbers are optimistic because H9 selected
the arm.

## Test set

1045 frames from six storms, all GOES-19, none excluded. Gabrielle is
missing two frames with no scan within 10 min. 272 frames are above
96 kt.

| band | TD | TS | Cat 1 | Cat 2 | Cat 3 | Cat 4 | Cat 5 |
|---|---|---|---|---|---|---|---|
| 2025 frames | 22 | 456 | 166 | 129 | 84 | 137 | 51 |
| H9 frames | 180 | 506 | 264 | 149 | 274 | 348 | 31 |

## Result, pooled

| metric | ensemble 2025 | ensemble H9 | 2025 - H9 | members 2025, mean (sd) | best_model.pt 2025 (reference) |
|---|---|---|---|---|---|
| overall MAE | **9.78** | 8.57 | +1.20 | 10.42 (0.26) | 12.13 |
| fixes-only MAE | 9.79 | 8.45 | +1.34 | 10.42 (0.45) | 11.71 |
| overall MAE reweighted to H9 bands | 10.05 | 8.57 | +1.48 | 10.64 (0.34) | 12.26 |
| overall bias | -2.27 | -2.93 | +0.66 | | -2.96 |
| >96 kt MAE | 12.65 | 10.82 | +1.83 | 13.18 (0.87) | 15.36 |
| >96 kt bias | -8.54 | -7.72 | -0.82 | (same by construction) | -11.83 |
| Spearman, frames > 96 kt | 0.729 | 0.601 | +0.128 | 0.711 (0.028) | 0.684 |
| < 64 kt MAE | 8.53 | 7.21 | +1.32 | 9.02 (0.86) | 10.72 |
| < 64 kt bias | +3.80 | +1.31 | +2.49 | | +6.03 |

- **Error on new storms and a new satellite:** overall MAE rose by
  1.2 kt from H9, and by 1.5 kt once the intensity mix is reweighted to
  H9's.
- **The ensemble's advantage replicated:** it beats all ten members on
  overall MAE, as on H9.
- **Against the reference:** the ensemble is better than `best_model.pt`
  on every pooled metric, and better or equal on every per-band MAE.

## Result, by band

| band | bias 2025 | bias H9 | MAE 2025 | MAE H9 | best_model.pt bias / MAE |
|---|---|---|---|---|---|
| TD | +7.59 | +7.37 | 7.59 | 8.22 | +8.27 / 8.28 |
| TS | +3.62 | -0.85 | 8.58 | 6.86 | +5.92 / 10.84 |
| Cat 1 | -6.29 | -2.31 | 9.42 | 6.64 | -7.63 / 10.01 |
| Cat 2 | -6.42 | -2.62 | 8.78 | 8.41 | -11.56 / 13.28 |
| Cat 3 | -7.90 | -4.19 | 15.29 | 9.42 | -12.42 / 17.84 |
| Cat 4 | -6.25 | -9.50 | 9.85 | 11.20 | -9.03 / 12.69 |
| Cat 5 | -15.75 | -18.85 | 15.81 | 18.85 | -18.39 / 18.43 |

The shift from H9 is not uniform. Tropical storms are over-predicted
more (TS +3.6 against -0.8). Cat 1-3 are under-predicted more, by about
4 kt each, and Cat 3 MAE is the worst of any band below Cat 5 (15.3).
Cat 4 and Cat 5 are under-predicted less than on H9. By the note
recorded before the run, that smaller high-end bias is not read as
better skill: the 2025 crops have a colder cold tail in every band,
strongest at Cat 5, and colder convection may raise high-end predictions.
The 2025 Cat 5 frames are also mostly one storm (37 of 51 are Melissa).

## Result, per storm

| storm | frames | MAE | >96 kt frames | >96 kt MAE | >96 kt bias | >96 kt Spearman | best_model.pt MAE |
|---|---|---|---|---|---|---|---|
| Dexter AL042025 | 73 | 6.13 | 0 | | | | 8.29 |
| Erin AL052025 | 277 | 9.88 | 68 | 17.24 | -17.09 | 0.525 | 13.66 |
| Gabrielle AL072025 | 185 | 8.24 | 44 | 8.39 | -1.60 | 0.820 | 10.37 |
| Humberto AL082025 | 163 | 11.73 | 80 | 9.54 | -7.42 | 0.722 | 13.43 |
| Imelda AL092025 | 103 | 8.91 | 0 | | | | 10.71 |
| Melissa AL132025 | 244 | 10.98 | 80 | 14.20 | -6.21 | 0.821 | 12.61 |

Above 96 kt, Erin carries the largest error: it is under-predicted by
17 kt with weak ranking. Gabrielle, Humberto and Melissa are ranked well
(0.72-0.82).

The pooled Spearman (0.729, against 0.601 on H9) mixes ranking within
storms with differences between storms, and none of the 2025 storms
behaves like Ian on H9 (0.04). The per-storm values above are the
cleaner measure.

## Exploratory (not pre-registered): saturation at Melissa's peak

The note recorded before the run expected saturation above the 160 kt
training maximum.
- The highest ensemble prediction anywhere in 2025 is 143.3 kt (H9:
  139.7); the highest single-member prediction is 154.2.
- Melissa's 8 frames above 160 kt (truth mean 162.6) get 133-143 kt
  (mean 139.6, bias -23.0).
- Her 27 frames between 137 and 160 kt (truth mean 150.8) get a mean of
  132.9 (bias -17.9).
- The two frames at exactly 160 kt fall in neither group.

This is consistent with saturation. It can't be separated from the
general Cat 5 underestimation, which is -18.8 kt on H9 as well.

Cat 5 by storm:
- Melissa: 37 frames, truth 153.9, prediction 134.7.
- Humberto: 11 frames, truth 139.4, prediction 137.0.
- Erin: 3 frames, truth 138.6, prediction 116.6.

The smaller pooled Cat 5 bias comes mostly from Humberto, whose Cat 5
frames are barely above the 137 kt threshold.

## Limitations

- **Satellite and storms are confounded.** The test cannot separate the
  GOES-19 change (a different instrument and band-13 spectral response)
  from these storms being new. Every 2025 frame is GOES-19.
- **Six storms, one season.** The per-storm spread (MAE 6.1-11.7) is as
  large as the change from H9.
- **The models never saw Cat 5 during validation,** and Melissa
  exceeds the training maximum.
- **The 2025 set is now spent.** It is not used for model selection.
