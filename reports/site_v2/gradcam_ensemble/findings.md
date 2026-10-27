# Grad-CAM for the frozen ensemble: results

Produced by `scripts/gradcam_ensemble.py` (`control`, then `ensemble`), following
`reports/site_v2/gradcam_ensemble_method.md` at 6e8f58b. The inputs, all read only, were:

- `best_model.pt` and the ten members, with hashes verified against `COORDINATOR.md` and
  `results/ensemble/members_sha256.txt`;
- `data/processed`.

Device: CUDA (RTX 5070).

## Control: `best_model.pt` with today's code, reproduced exactly

The tolerances were 1e-3 on the arrays and 5e-4 on the mean. The control was well inside both:

- **Peak-frame arrays:** all four are bit-identical to `results/gradcam/<stem>.npy` (max abs difference 0.0).
- **Site PNGs:** all four overlays are pixel-identical to the site PNGs.
- **454-frame survey:** mean r is +0.574891 against +0.574891 (difference 1e-16). The largest per-frame difference
  from the committed CSVs is 8e-17.

Files: `control_summary.json` and `control_best_model_survey_96_999.csv`.

## Ensemble: renormalized mean of the ten members' normalized CAMs, same 454 frames

| | ensemble | original (`best_model.pt`) |
|---|---|---|
| **mean pearson(CAM, coldness)** | **+0.514** | **+0.575** |
| median | +0.529 | +0.607 |
| frames with r < 0 | 6 of 454 (1.3%) | 3 of 454 (0.7%) |
| IoU minus chance, mean | +0.300 | +0.258 |
| frames with IoU below chance | 6 | 5 |
| area at CAM >= 0.5, median share of frame | 24% | 40% |
| CAM peak to coldest pixel, median | 31 px | 53 px |

The ensemble's mean r is 0.061 below the original's.

- **Per storm:** the means are Ian +0.501, Beryl +0.453, Lee +0.528 and Sam +0.542.
- **Per frame:** the ensemble's r is lower than the original's in 72.7% of frames.
- **Area:** the ensemble's highlighted area is smaller in 78.2% of frames.

### Peak frames

| frame | truth kt | ensemble pred kt | pearson |
|---|---|---|---|
| Ian, 2022-09-28 12:00 | 140 | 118.0 | +0.583 |
| Beryl, 2024-07-02 09:45 | 145 | 123.9 | +0.492 |
| Lee, 2023-09-08 06:00 | 145 | 133.6 | +0.462 |
| Sam, 2021-09-26 18:00 | 135 | 134.8 | +0.533 |

These predictions match `pred` in `results/ensemble/expanded10_h9.csv` for the same frames.

### Reading

- **Alignment:** the ensemble's attribution still sits on the coldest cloud tops far more than chance (r > 0 in
  98.7% of frames). But its correlation is lower than the single model's.
- **Shape:** its maps are tighter and more central. That fits a lower all-pixel correlation together with a higher
  IoU against chance, though this survey does not show it is the cause.
- **Status:** these are held-out development storms for the ensemble, not a test.

Files:

- `ensemble_summary.json` and `ensemble_survey_96_999.csv`;
- `<stem>.npy`, the four renormalized ensemble CAMs;
- `ensemble_<figure>.png`, copied to `site/figures/`.
