# Grad-CAM for the frozen ensemble: method (step 1, no compute)

This repeats the analysis behind the site's current Grad-CAM section (`site/index.html`, section `#gradcam`) on the
frozen 10-member ensemble, `models/storms/expanded/seed0-9.pt`. Nothing here has been run on the ensemble yet.
hc-adversary checks this file before any compute.

## Where the current section comes from

- **Commit:** a8a6dc0 (2026-09-13), "Add Grad-CAM attribution section and four-storm alignment analysis". It added
  `scripts/gradcam.py`, `results/gradcam/*`, `storms_gradcam.txt`, the six `site/figures` Grad-CAM PNGs and the
  section text.
- **Model:** `best_model.pt`, loaded at `scripts/gradcam.py` L44, L118-124 (a8a6dc0).
- **Since a8a6dc0:** the only change to `gradcam.py` is 0d17b78. It reads `HOLDOUT_STORMS` and `load_storm_list`
  from `splits.py` instead of hard-coding them. No method line changed (`git diff a8a6dc0 HEAD -- scripts/gradcam.py`).
- **Site figures:** each 224 x 224 site PNG is pixel-identical to the top 224 rows of the matching
  `results/gradcam/<stem>.png`. Checked for all five: ian_140kt, beryl_145kt, lee_145kt, sam_135kt and ian_30kt. So
  the site images are `render_threshold_image` output, without the strip that `render_threshold` adds below.
- **The +0.57:** the mean pearson(CAM, coldness) over the 454 valid frames in two CSVs:
  - `convection_alignment_survey_96_999.csv` (Ian, 47);
  - `convection_alignment_survey_96_999_gradcam_storms.csv` (Beryl 104 valid of 106, Lee 115, Sam 188).

  It is +0.5749 (median +0.6069).
  - `findings.md` L89-95 prints +0.569. That is the pool of the three new storms only (407 valid). The 454-frame
    figure itself is not written anywhere in the repo.
  - Both round to +0.57. The ensemble comparison uses +0.575, the 454-frame number the site quotes.

## Method, item by item

All line numbers are `scripts/gradcam.py` at a8a6dc0.

| item | original | source |
|---|---|---|
| Layer | `model.layer4`; forward and full-backward hooks | L142-143 |
| Score | the scalar regression output itself (no class logit), `score.backward()` | L147-150 |
| Weights | gradient averaged over the 7 x 7 grid, per channel | L154 |
| CAM | ReLU of the weighted channel sum | L155 |
| Upsampling | bilinear to 224 x 224, `align_corners=False` | L156, `FRAME_SIZE` L46 |
| Blur | Gaussian, kernel 33 px, sigma 8.0 | L157, L107-108 |
| Per-frame normalization | `cam -= cam.min()`, then `/= cam.max()` if max > 0 | L163-166 |
| Input | `TCDataset(data/processed, storms)` default: no circular mask, no rotation | L427 (survey); the single-frame path in `main` builds the same dataset |
| Coldness | `1 - gray/255`, where gray = `clip(image[0] * 255, 0, 255)` as uint8 | L436 and L550 (gray), L333-334 |
| Correlation | Pearson over all 224 x 224 pixels, `np.corrcoef(cam, cold)`; NaN if either is constant | L318-343 |
| IoU vs coldest | pixels with CAM >= 0.5, against the same count of coldest pixels, with its chance level | L344-357, `THRESHOLD` L97 |
| Survey band | `96 <= wind_kt < 999`, true fixes and interpolated hours | L428 (`--survey 96 999`) |
| Overlay | `render_threshold_image`: plasma LUT (L78-83), transparent below 0.5, alpha 0.22 at 0.5 up to 0.85 at 1.0 | L270-284, L97-99 |
| Prediction shown | `model(x) * 185.0` | L546 |

The ensemble uses the same architecture, built and loaded as `scripts/ensemble_eval.py` L61-67 does. It also uses
the same unmasked `TCDataset` input, as at `ensemble_eval.py` L58.

## The four peak frames

From `results/gradcam/convection_alignment.csv`, same stems as the site PNGs:

| site figure | storm | frame (UTC) | truth kt | best_model.pt pred kt | pearson |
|---|---|---|---|---|---|
| ian_140kt | AL092022 Ian | 2022-09-28 12:00 | 140 | 126.1 | +0.641 |
| beryl_145kt | AL022024 Beryl | 2024-07-02 09:45 | 145 | 114.0 | +0.637 |
| lee_145kt | AL132023 Lee | 2023-09-08 06:00 | 145 | 127.0 | +0.599 |
| sam_135kt | AL182021 Sam | 2021-09-26 18:00 | 135 | 139.7 | +0.677 |

The original commands were `gradcam.py <id> <timestamp>` for these frames. For the survey they were
`gradcam.py --survey 96 999` (Ian, then the holdout default) and
`gradcam.py --survey 96 999 --storms-file storms_gradcam.txt --out-name convection_alignment_survey_96_999_gradcam_storms`.

## The 454 frames

- **Storms:** AL092022 (Ian, 47 frames), AL022024 (Beryl, 104), AL132023 (Lee, 115) and AL182021 (Sam, 188).
- **Excluded lists:**
  - All four are Atlantic storms from 2021-2024.
  - None is in `storms_test_2025.txt` or `storms_holdout_hep.txt`, and none is a 2026 storm.
  - None is in `storms_train_expanded.txt` or `storms_val_expanded.txt`.
  - All four are in `storms_holdout_h9.txt`, so for the ensemble they are held-out development storms, not a test.
- **Same frames today:**
  - `TCDataset` on those four storms with `96 <= wind_kt < 999` gives 454 frames today.
  - They are the same (storm, time) set as the 454 valid CSV rows. The lowest wind is 96.67 kt.
  - The two Beryl frames that were NaN at the time (2024-07-02 17:00 and 18:00) are now in `excluded_crops.txt`
    (438cb23), so they drop out of the dataset instead of giving a NaN.

## Ensemble CAM

- **Per member:** `compute_gradcam` exactly as above. Each member's CAM is normalized to [0, 1] on its own (L163-166).
- **Ensemble CAM:** the mean of the ten normalized CAMs.
- **Survey statistic:** pearson(ensemble CAM, coldness) per frame over the same 454 frames, then the mean. The
  result is compared with +0.575 whichever way it comes out.

## Before inference

- **Member hashes:** checked against `results/ensemble/members_sha256.txt`, as `ensemble_eval.py --expect-sha` does.
- **GPU:** one GPU job at a time.

## What cannot be matched exactly, and open points

1. **Pearson is scale-free, but the 0.5 threshold is not.**
   - The mean of ten [0, 1] maps peaks below 1 wherever the members' peaks disagree.
   - So the IoU (CAM >= 0.5) and the overlay's transparent-below-0.5 rule would highlight less than for one model.
   - Open: either renormalize the mean once, min-max as for a member, before the IoU and the overlay, or don't.
     Pearson is the same either way.
2. **Legend.**
   - `site/figures/gradcam_legend.png` (760 x 76) has no committed generator.
   - `_threshold_strip` at 760 or 224 px does not reproduce it or the results strip pixel for pixel, most likely
     because of fonts.
   - The ensemble section would reuse the existing legend unchanged, since it uses the same colour scale.
3. **Weak-frame panel.**
   - The current section also shows Ian at 30 kt and says the weak band (< 64 kt) is noisy. Its source is
     `convection_alignment_survey_0_64.csv`: 277 frames on the old holdout of Ian, Nicole and Claudette.
   - The brief does not cover it. It is out of scope unless added.
4. **Proposed control before the ensemble run.**
   - Recompute `best_model.pt`'s CAMs for the four peak frames with today's code and compare them with the
     committed `.npy` files.
   - Then recompute its 454-frame mean and check it reproduces +0.5749.
   - That shows today's code and environment are the original method before the ensemble number is read.
