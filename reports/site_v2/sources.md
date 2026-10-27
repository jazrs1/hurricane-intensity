# Site v2: where each number comes from

Every number that this rewrite added or changed in `site/index.html`, with the committed file and line it comes from (or the CSV and the computation). Abbreviations:
- F = `results/final_2025/findings.md` (the 2025 test, run once);
- E = `results/ensemble/findings.md` (the ensemble on H9);
- D = `logs/decisions.md`;
- X = the printed output of `scripts/export_site_v2.py`, which reads `results/ensemble/expanded10_h9.csv`.

Numbers this rewrite didn't touch (Michael's frames, the method steps 01–05 and the Grad-CAM section) are unchanged from the previous page.

## Header, metrics, standfirst

| Page text | Value | Source |
|---|---|---|
| meta description, metric 1, Results lead | 9.78 kt ensemble MAE, 2025 | F:42 (`ensemble_2025_metrics.csv`, overall MAE 9.775…) |
| meta description, metric 2, Results lead | 12.13 kt, original model (best_model.pt), same frames | F:42; D:2121 |
| metric 3 | −15.7 kt bias at category 5, 2025 | F:70 (−15.75) |
| standfirst | 2,081 crops after exclusion | D:1443–1445 ("Actual counts after exclusion … expanded training: 2081") |
| standfirst | 19 storms | D:1476 ("storms (9 / 19)") |
| standfirst | 2018–2024 | `storms_train_expanded.txt` minus `storms_val_expanded.txt`: the ATCF-ID years run 2018–2024 |
| method 06 | ten seeds, validation-selected, equal-weight mean, frozen by hash | E:10, E:14, E:16 |
| method 02, Results lead | GOES-19 for 2025 | F:29, F:128 |

## Results

| Page text | Value | Source |
|---|---|---|
| lead | six storms, 1,045 frames | F:29 |
| category table, frames | 22 / 456 / 166 / 129 / 84 / 137 / 51 | F:35 |
| category table, MAE, bias and original MAE | | F:64–70 (bias 2025, MAE 2025, best_model.pt MAE) |
| category caption | "better or equal in every category" | F:58 |
| storm table | frames, MAE, original MAE | F:85–90 |
| storm caption | spread 6.1–11.7 kt; 1.2 kt change from H9 | F:129, F:52 (+1.20 at F:42) |
| scatter caption | 1,752 frames, nine storms, 315 fixes | X ("H9: 1752 frames, 9 storms, 315 fixes"); E:18 |
| scatter caption | H9 years 2020–2024, GOES-16 | `expanded10_h9.csv`: the minimum `sample_time` year per storm is 2020–2024, and every `source_key` is `_G16_` |
| scatter caption | 8.57 kt, optimistic | E:28; E:3–6 (exploratory, the configuration H9 favoured) |

## Ian scrubber

| Page text | Value | Source |
|---|---|---|
| frames | 197, the `pred` of each from the ensemble | X ("ian.json: 197 frames"); the script checks time, truth and label kind against ian.json |
| "within about seven knots on average" before the peak | MAE 6.93 | X |
| "at the 140-knot peak on 28 September it reads 118" | 140 kt, 118.0 | X |
| "highest reading anywhere is 126" | 126.0 | X |
| "about eleven knots low" after the peak | bias −11.42 | X |

## Where it fails

| Page text | Value | Source |
|---|---|---|
| category 5: about 16 kt low on 2025, 19 on H9 | −15.75, −18.85 | F:70; E:45 |
| a shared bias survives the average | | E (Reading: ensemble bias = member mean) |
| Melissa peak 165 kt; training max 160 | | D:2105–2106 |
| eight frames above 160 kt got 133–143 | | F:107 |
| can't be separated from the general Cat 5 shortfall | | F:113–114 |
| weak systems: TD about +8, TS about +4, 2025 | +7.59, +3.62 | F:64–65 |
| Category 3 worst below Cat 5, 15.3 | | F:74 |
| 2025 is 1.2 kt higher than H9; GOES-19 vs GOES-16 confounded | | F:52, F:126–128 |

## What changed in phase 2

| Page text | Value | Source |
|---|---|---|
| 33 reruns, mean 10.05 kt, one in 33 at or below 9.00 | | D:403–405 |
| 19 storms instead of 9: −1.6 kt overall on the nine held-out development storms (H9), high-end bias +5.4 kt, all three tests pass | | `results/storms/findings.md`:23, 29, 30; D:1476 |
| more storms and more frames confounded | | D:1562–1565 |
| 19 → 50 storms, same budget: +0.53 kt, not significant, under-trained | | D:2481, D:2483; `reports/v2_survey/n5/design.md`:25 |
| four times longer: −0.99 kt over 20 paired seeds, on H9 | | D:2850 |
| new ensemble frozen; tested only in 2026–2027 | | D:3511; D:3616 |
| Cat 5 about 19 kt low on H9 | −18.85 | E:45 |
| East Pacific work under way; test set sealed under a fixed protocol | | D:3638 (the East Pacific HURDAT2 download plan); D:3343 ("H-EP stays sealed under protocol.md 3901f4cf") |
| phase-2 design, script and thresholds committed before training; output reported as printed; a separate AI review session (hc-adversary) tries to break each design and recomputes | | `results/storms/findings.md`:3–6 (script hash at launch); D:2837 ("pre-registered analysis, unedited"); D:2510 ("An independent recompute reproduces every number") |

## Figures

`scripts/export_site_v2.py` regenerates `site/figures/heldout_scatter.png` and `heldout_scatter_narrow.png` (in `make_scatter.py`'s style and at the same pixel sizes) and the `pred` fields of `site/frames/ian.json`, from `results/ensemble/expanded10_h9.csv` only. It runs no model, and reads no 2025, East Pacific or 2026 data. The Ian frame PNGs and every Grad-CAM image are unchanged; the Grad-CAM section is labelled as the original model (best_model.pt).
