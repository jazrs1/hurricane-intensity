chose L2 CMIP over L1b, already calibrated to Kelvin, Planck inversion includes silent error risk.

Defaulted to hourly frames rather than depending on 6-hourly fixes. Compromise between 15-min frames which are near duplicates and the 6-hour fixes which leads to less training data. 

+/- 10 min matching tolerance as storm motion over 30 min can decenter the crop; offsets logged per sample for later filtering

Test crop came out to be 274x309 (not a square). May have to resize to fixed 224x224 as each sample can differ in crop size.

HURDAT2 is versioned by filename, resolver scrapes the index page, best tracks are subject to post-season reanalysis so results are tied to a specific database version. 

122 samples, 22 exact fixes, 100 interpolated, zero skips, offsets ~0.6 min

Keep depression-stage frames? Label-image relationship in these frames is weak, center may not be in crop.

Frame 19: Hot towers forming, 35kts storm. While evidence of RI is present, the lag for RI analysis is due to an exact interpretation rather than implied based on the features.

Frames 65-70, prominent hot towers rotating around a developing eye. Signature of intensification.

BT clamped to [180, 320] and scaled to [0, 1], bilinear resize to 224x224 as crops vary in pixel size with latitude, single channel replicated to 3 for ImageNet weights with labels normalized by fixed 185 kt.

Dataset currently includes post-landfall frames, but a storm over land has a different IR appearance than one over water at the same wind speed. Should they be included? 

In training loop, split by storm to avoid leakage and deceptive MAE. Shuffle for training to ensure model is not just memorizing
cycles and truly learning what different storm structures correspond to in terms of intensity. 

Identified and excluded 6 degraded scans from GOES-16, all at 0:500Z which corresponds to a cooldown and calibration window for the satellite. These scans previously caused all of the weights in ResNet to become NaN.

Excluded bad scans via generated exclusion list.

First 20 epochs, best val 9.44 kt at epoch 20, train 4.47.

Held-out results: 9.00 kt overall MAE, 9.50 kt on true HURDAT2 fixes only
(n=77). Fixes-only close to overall, so interpolated labels are not
inflating the score.

Systematic underestimation at high intensity: bias -14.58 kt (Cat 1),
-19.49 (Cat 4), -18.25 (Cat 5). Model hedges toward the mean. Two likely
causes: class imbalance (270 of 393 held-out samples below hurricane
strength) and the single-frame limitation — one IR frame carries no
information about whether a storm is intensifying.

Open: Cat 1 MAE (15.76) is worse than Cat 2 and Cat 3 despite being
closer to the data bulk. Need to check which storm those 58 samples come from.

2026-10-13. Class-imbalance experiment, design. Written before any training.
Nothing has been run and no code has been changed.

Question: does the high-intensity underestimation shrink when the training
set has more Cat 3+ frames? This tests the class-imbalance explanation only.
It says nothing about the single-frame limitation.

What changes: the training data, and nothing else. Beryl (AL022024) and Lee
(AL132023) move from the Grad-CAM list into training. Sam (AL182021) moves
from the Grad-CAM list into the held-out set. All three already had crops
built; the baseline checkpoint was trained on none of them.

Frame counts as TCDataset loads them, binned with evaluate.py's BANDS.
"unbinned" is interpolated hours whose wind falls between the integer band
edges (e.g. 33.5 kt): evaluate.py's per-band table skips them, they still
count in the overall MAE.

| split | TD | TS | Cat 1 | Cat 2 | Cat 3 | Cat 4 | Cat 5 | unbinned | total | >96 kt |
|---|---|---|---|---|---|---|---|---|---|---|
| train, baseline | 294 | 458 | 155 | 122 | 83 | 104 | 66 | 25 | 1307 | 258 |
| train, experiment | 315 | 567 | 238 | 187 | 208 | 164 | 93 | 44 | 1816 | 477 |
| validation (both) | 81 | 190 | 64 | 19 | 8 | 17 | 0 | 7 | 386 | 25 |
| held-out, original 3 | 83 | 187 | 58 | 9 | 25 | 15 | 4 | 12 | 393 | 47 |
| held-out, expanded 4 | 93 | 210 | 85 | 52 | 67 | 160 | 4 | 17 | 688 | 235 |

Training frames above 96 kt go from 258 of 1307 (19.7%) to 477 of 1816 (26.3%).

What is held fixed: architecture (ResNet-18, IMAGENET1K_V1 weights, one
output), loss (L1), optimizer (Adam, lr 1e-4), batch size 32, 20 epochs,
checkpoint chosen on best validation MAE, validation storms (Sally, Ophelia,
Laura), preprocessing in dataset.py, labels scaled by 185 kt. No loss
reweighting, no resampling, no augmentation.

Seed: intended to be held fixed, but it cannot be yet. train.py sets no seed
and the 9.00 kt baseline was an unseeded run with shuffle=True, so there is
no baseline seed to match. Has to be decided before the run: seed the new
run (then it is seeded vs. unseeded) or rerun the baseline under the same
seed so the two differ only in data. Until run-to-run spread is known, a
small change in MAE cannot be told apart from seed noise.

Epochs fixed is not updates fixed. 20 epochs is 41 batches per epoch on the
baseline set and 57 on the expanded one, about 39% more gradient steps. That
comes with adding data and is part of the treatment, not a separate knob.

Reporting, two ways:
1. Original three held-out storms (Ian, Nicole, Claudette; 393 frames).
   Same frames as the 9.00 kt baseline (9.50 on true fixes, n=77), so this is
   the comparable number.
2. Expanded four-storm held-out set (adds Sam; 688 frames). This is the new
   claim. It needs its own baseline: the baseline checkpoint never saw Sam,
   so it gets evaluated on the same four storms rather than comparing the new
   model's four-storm number against 9.00.
Both with overall MAE, fixes-only MAE, and per-band MAE and bias.

Limits on what this can show, known in advance:
- Validation has 25 frames above 96 kt, all from Laura (8 Cat 3, 17 Cat 4,
  4 of the 25 true fixes) and no Cat 5 at all. Checkpoint selection is nearly
  blind to the range the experiment targets. Validation MAE is not the
  readout; the held-out comparison is.
- The expanded held-out set is mostly Sam's Cat 4 plateau: 145 of its 160
  Cat 4 frames and 188 of its 235 frames above 96 kt are Sam. Cat 5 held-out
  is still Ian's 4 frames. The four-storm number must be reported per storm
  as well as pooled, and the claim is about Cat 3-4, not Cat 5.
- Sam was picked after seeing where the baseline failed, as a Cat 4+ storm.
  The four-storm set is not a neutral sample of intensities, which is why the
  three-storm number is reported alongside it.
- The Grad-CAM findings on Beryl, Lee and Sam hold for the baseline
  checkpoint. For the new model Beryl and Lee are training data.

Two Beryl frames (2024-07-02 17:00 and 18:00, 5% and 52% NaN pixels) were not
in excluded_crops.txt. Added by hand; all 2898 crops on disk rescanned, 8
have NaN pixels and all 8 are now listed. check_nan.py's ALL_STORMS does not
include Beryl, Lee or Sam, so rerunning it rewrites the file without these
two.

Not yet done, needed before the run (code left untouched on purpose):
- train.py: TRAIN_STORMS is hardcoded and does not read storms_train.txt.
  Beryl and Lee have to be added there or the run trains on the old nine.
- train.py saves to best_model.pt. That file is the baseline checkpoint and
  is not in git. The new run needs a different output path, or a copy of the
  baseline kept first.
- evaluate.py: HOLDOUT_STORMS is hardcoded to three storms, loads
  best_model.pt and overwrites results/heldout_predictions.csv. It needs to
  take a checkpoint and a storm list and write to separate outputs.
- scripts/gradcam.py and scripts/export_scrubber.py carry their own copies of
  HOLDOUT_STORMS.

2026-10-13, later. Groundwork before the experiment. The experiment itself
has still not been run.

Baseline checkpoint copied to models/baseline_9kt.pt (same SHA-256 as
best_model.pt). best_model.pt itself is unchanged.

Band fix. evaluate.py's bands were closed integer ranges with gaps, so 12 of
the 393 held-out frames (interpolated, fractional kt) were in no band. Now
contiguous, lower bound inclusive, upper exclusive, at 34/64/83/96/113/137.
Overall MAE never used the bands and is unchanged: 9.00 kt, 9.50 on fixes
(n=77). results/heldout_predictions.csv came out byte-identical. Corrected
per-band table for the baseline checkpoint, original three storms:

| band | n | MAE | bias | was (n, MAE, bias) |
|---|---|---|---|---|
| TD | 88 | 5.45 | +2.40 | 83, 5.50, +2.38 |
| TS | 189 | 7.64 | -2.85 | 187, 7.59, -2.74 |
| Cat 1 | 60 | 15.35 | -14.13 | 58, 15.76, -14.58 |
| Cat 2 | 9 | 3.96 | +0.24 | 9, 3.96, +0.24 |
| Cat 3 | 26 | 10.40 | +7.08 | 25, 10.50, +7.68 |
| Cat 4 | 17 | 18.40 | -18.40 | 15, 19.49, -19.49 |
| Cat 5 | 4 | 18.25 | -18.25 | 4, 18.25, -18.25 |

This supersedes the per-band numbers above (-14.58 / -19.49, "270 of 393
below hurricane strength" is 277, "those 58 samples" are 60: Ian 48,
Nicole 12).

Storm lists are live. train.py, evaluate.py, gradcam.py and
export_scrubber.py read them through src/hurricane_intensity/splits.py.
storms_train.txt is every storm a checkpoint sees, validation included;
storms_val.txt (new: Sally, Ophelia, Laura, the same three as before) is
taken out of it to get the training set; storms_holdout.txt is the held-out
set. The loader refuses a held-out storm that is also in the train list.
Because the live lists now describe the experiment (11 training storms, 4
held-out), the original split is frozen in storms_train_baseline.txt and
storms_holdout_baseline.txt, passed with --train-list / --holdout-list.
Plain `python scripts/evaluate.py` now evaluates four storms and overwrites
results/heldout_predictions.csv; the site numbers need --holdout-list
storms_holdout_baseline.txt.

check_nan.py scans every storm with a manifest on disk. Rerun: 8 of 2898,
the same 8 crops.

train.py takes --seed (python, numpy, torch, and a dedicated generator for
the shuffle) and --out. Default is unseeded and best_model.pt, as before.
The seed fixes the fc init and the batch order (checked: identical across
processes). It does not make a GPU run repeatable: seed 0 run twice diverges
inside the first epoch (cuDNN is non-deterministic by default) and ends at
10.43 and 10.61 kt held-out. Left that way; forcing deterministic cuDNN is a
separate choice.

Run-to-run spread of the baseline. Original nine training storms, same
code, seeds 0/1/2, each evaluated on the original three held-out storms
(same 393 frames as the 9.00). Checkpoints models/baseline_seed{0,1,2}.pt,
logs in logs/seed_baseline/, predictions in results/seed_baseline/.

| | 9.00 run | seed 0 | seed 1 | seed 2 | seed range |
|---|---|---|---|---|---|
| overall MAE | 9.00 | 10.43 | 9.94 | 10.57 | 0.63 |
| fixes-only MAE (n=77) | 9.50 | 10.63 | 10.93 | 11.20 | 0.57 |
| overall bias | -3.50 | +2.43 | -3.46 | -3.78 | 6.21 |
| best val MAE (epoch) | 9.44 (20) | 9.40 (18) | 9.20 (20) | 8.48 (20) | |
| TD bias (n=88) | +2.40 | +9.15 | +2.80 | +2.27 | 6.88 |
| TS bias (n=189) | -2.85 | +1.90 | -3.60 | -4.17 | 6.07 |
| Cat 1 bias (n=60) | -14.13 | -10.25 | -12.78 | -10.98 | 2.53 |
| Cat 2 bias (n=9) | +0.24 | +8.26 | +3.53 | +3.97 | 4.74 |
| Cat 3 bias (n=26) | +7.08 | +19.09 | +9.58 | +6.79 | 12.31 |
| Cat 4 bias (n=17) | -18.40 | -6.96 | -21.19 | -21.04 | 14.24 |
| Cat 5 bias (n=4) | -18.25 | -11.81 | -20.26 | -23.50 | 11.69 |

What this says:
- Overall MAE moves 0.63 kt across three seeds (sd 0.33). An experimental
  change in overall MAE smaller than roughly that is not a result.
- The per-band bias the experiment is meant to move is far less stable than
  the overall number. Cat 4 bias ranges from -7 to -21 kt and Cat 3 from +7
  to +19 with no change to the data. Seed 0 trades the Cat 4 underestimate
  for a Cat 3 overestimate. A single experimental run with a better Cat 4
  bias would prove nothing.
- All of Cat 3-5 held-out is 47 frames of one storm (Ian). That is part of
  why these numbers swing.
- 9.00 is better than all three seeds, and than the seed 0 repeat (10.61),
  by 0.9 to 1.6 kt. The four reruns average 10.4. Either the published run
  was a favorable draw or something differs between then and now that I
  have not found; the checkpoint itself still reproduces 9.00 exactly. Its
  validation MAE (9.44) was ordinary. The experiment has to be compared with
  seeded baselines run under the same code, not with 9.00.
- Validation MAE does not rank the runs: seed 2 has the best validation MAE
  (8.48) and the worst held-out MAE of the three.

Consequence for the design above: one run per arm cannot answer the
question. Each arm needs several seeds and the comparison is between the two
sets of runs, per band, with the per-storm breakdown.

2026-10-13, later still. Did the data change since the 9.00 run? No sign
of it.

- Every crop and manifest for the nine training, three validation and three
  held-out storms was written 2026-08-18 to 08-25 (creation and modification
  times agree, files written in manifest order, storm after storm).
  best_model.pt was written 09-11 15:08. Nothing for those 15 storms is newer.
- The uncommitted edits to crop.py and goes_index.py are dated 09-13 20:17,
  after every crop on disk, Beryl/Lee/Sam (09-13 13:05-14:13) included. They
  add a bucket argument that defaults to the same bucket and a sector filter
  that defaults to full disk. Neither touches the crop geometry or the
  extraction. Every source_key in every manifest is ABI-L2-CMIPF, G16, C13.
  The rest of the pipeline code is unchanged since the 08-24 commit.
- Geometry: each crop's shape matches its manifest, and matches the shape
  projection.crop_box gives for its lat/lon at 600 km, for all 2898 crops
  (one Ian frame off by a pixel in my reconstruction of the grid). So the
  three build batches (08-18/19, 08-24/25, 09-13) used the same crop.
- Held-out pixels: the 197 Ian frames in site/frames/ian (exported 09-12
  from the model input) are pixel-identical to what the dataset yields now.
  The committed predictions CSV reproduces byte for byte.
- Training pixels: no hash of the training crops was recorded at the time,
  so for those the evidence is the timestamps and geometry, plus the three
  Michael Grad-CAM arrays from 09-13, which recompute exactly.
- Environment: torch and torchvision were installed 09-11 13:39 and the
  ResNet-18 weights cached 14:38, both before the checkpoint. numpy was
  reinstalled 09-13; requirements.txt already pinned the same version.
- best_model.pt was created 15:07:56 and last written 15:08:44. That 48 s
  span is what a 20-epoch run takes here from its first save to its last
  (44-48 s in the seeded runs), so the checkpoint is the last epoch of one
  20-epoch run, as the log says.
- Not verifiable: train.py, the exclusion code in dataset.py and
  excluded_crops.txt were first committed 09-11 20:50, five and a half hours
  after the checkpoint. What exactly ran at 15:07 rests on this log.

So 9.00 against 10.4 from four reruns is not explained by data, pipeline
code or installed versions. (It did not need explaining; see the correction
below.)

2026-10-13, last entry. Does validation MAE track held-out MAE?

The three seeded runs above kept only their selected checkpoint and a seed
does not reproduce a run here, so they cannot be traced. Three new runs
(seeds 0/1/2, original nine storms) with every epoch's weights saved
(train.py --epoch-dir, off by default), each epoch evaluated on validation
and on the original three held-out storms. Recomputed validation MAE matches
what train.py printed to 0.005 kt. Per-epoch numbers in
results/epoch_traces/epoch_trace.csv.

Within a run, validation tracks held-out. It is not selecting at random.
- All 20 epochs: Spearman 0.86-0.89 per run. That is mostly the shared
  descent and proves little.
- Epochs 11-20 only, each run's mean removed, pooled (n=30): Pearson +0.90,
  Spearman +0.73. Shuffling held-out across epochs within a run never
  reaches that in 20000 tries.
- The selected epoch was the 1st, 2nd and 3rd best of the ten late epochs on
  held-out. Held-out MAE at the selected epoch averages 9.24 kt; a random
  late epoch averages 10.81; the final epoch 10.95; the best available 9.06.
- Why it works: late in training the whole model still shifts from one
  epoch to the next, and both sets feel it. Overall bias on validation vs
  overall bias on held-out, late epochs: +0.98. Held-out MAE swings by 2 kt
  (sd) between late epochs, with single epochs far worse (9.48 at epoch 19,
  13.77 at epoch 20 in one run). Validation catches those.
- Where it stops working: among late epochs whose validation MAE is within
  1 kt of the run's best (n=12), the correlation is +0.30 and held-out sd is
  0.45 kt. Validation rejects bad epochs; it cannot rank good ones more
  finely than about half a knot.

Between runs, validation says nothing. Over all ten reruns, validation MAE
of the selected checkpoint vs its held-out MAE: Pearson +0.14 (bootstrap
95% interval -0.70 to +0.76). Held-out minus validation ranges from 0.04 to
2.09 kt between runs. The "best validation, worst held-out" observation
above was this, not a failure of selection inside a run. Validation MAE
must not be used to compare runs or arms.

Epochs. Validation selected epochs 20, 19 and 16; held-out was lowest at 20,
13 and 12, costing 0.00, 0.01 and 0.55 kt. No consistent direction. Across
all ten reruns validation picked epoch 19 or 20 seven times and never
before 15, and training MAE is still falling at epoch 20 (5-6 kt), so 20
epochs ends the run before it settles. Held fixed for now; noted.

Correction to the spread and to what I wrote about 9.00. Ten reruns of the
baseline now exist (4 earlier, 3 traced, 3 controls with per-epoch saving
off): held-out MAE 8.55 to 10.61, mean 9.85, sd 0.61. The earlier figure
(range 0.63, sd 0.33, from three seeds) was too small. 9.00 is inside that
range, 1.4 sd below the mean, with one rerun better. It is a good draw, not
an anomaly, and there is nothing left to explain. The four earliest reruns
are the four worst of the ten (10.39 mean vs 9.50 for the later six), on
identical commands and data. I have no mechanism for that; the controls
were run to check that per-epoch saving does not move the result (9.60,
9.82, 9.85, overlapping both groups).

One revision to the design entry, which called checkpoint selection
"nearly blind" above 96 kt. Within a run it is not: the >96 kt bias on
validation's 25 Laura frames and on held-out's 47 Ian frames move together
across late epochs (+0.98, and +0.92 after taking out the overall bias).
The limit is that it is 25 frames of one storm with no Cat 5, not that it
carries no signal.

For the experiment: selection by validation can stay. A difference between
arms has to be read on held-out, over several seeds per arm, against a
run-to-run sd of about 0.6 kt overall and much more per band (>96 kt bias
at the selected epoch in the three traced runs: +2.2, -0.4, -11.0).

2026-10-13, night. Epoch budget and run order. Experiment still not run.

Epoch budget. Three seeds to 60 epochs (train.py --epochs, default 20),
original nine storms, every epoch evaluated on validation and on the
original three held-out storms. results/epoch_traces_60/epoch_trace.csv.
Held-out MAE by ten-epoch window:

| epochs | seed 0 | seed 1 | seed 2 | mean | median | sd within run | train MAE | >96 kt MAE |
|---|---|---|---|---|---|---|---|---|
| 11-20 | 10.55 | 10.35 | 10.72 | 10.54 | 10.32 | 0.98 | 6.48 | 14.27 |
| 21-30 | 10.11 | 10.31 | 10.03 | 10.15 | 9.94 | 0.85 | 4.54 | 14.46 |
| 31-40 | 9.89 | 9.39 | 9.74 | 9.67 | 9.69 | 0.39 | 3.86 | 14.57 |
| 41-50 | 9.63 | 9.74 | 9.55 | 9.64 | 9.53 | 0.42 | 3.20 | 14.49 |
| 51-60 | 10.16 | 8.97 | 10.39 | 9.84 | 9.46 | 1.00 | 3.05 | 14.71 |

- 20 epochs is on the slope, as suspected. Held-out keeps falling to about
  epoch 30-35 and is flat from there to 60: slope over epochs 21-60 is
  -0.03, -0.37 and +0.06 kt per ten epochs. Smoothed minimum (5-epoch
  median) at epochs 43, 54 and 44.
- It does not rise. Training MAE goes on down to about 3 kt and the gap to
  held-out widens, but held-out does not get worse through epoch 60. The
  51-60 mean is up because the bad single epochs come back in two of three
  runs (11.6, 11.6, 12.2, 14.7); the median still edges down. Epochs 31-50
  are the calm stretch: no epoch above 11 kt in any run.
- What the experiment would actually use, the validation-selected
  checkpoint, mean of three runs by budget: 20 epochs 10.23; 25 9.61;
  30 9.68; 40 9.58; 50 9.68; 60 9.30 (one run's pick at epoch 58). All
  three runs improve from a 20 to a 40 budget (by 0.62, 1.11, 0.23).
- More training does nothing for the strong end. Held-out MAE above 96 kt
  is 14.3-14.7 in every window from 11-20 on and the >96 kt bias stays near
  -3 with large swings. The gain past 20 epochs is in the TD and TS frames.
- In steps: 41 batches per epoch on the baseline set, 57 on the expanded
  one. At a 20-epoch budget the experiment arm would get 1140 steps, what
  the baseline reaches at epoch 28, and over that stretch the baseline is
  still improving by about 0.4 kt from steps alone (10.23 at a 20-epoch
  budget, 9.87 at 28). That would be credited to the new data. At a
  40-epoch budget both arms are on the flat part.
- Budget to use: decided by the user; my recommendation is 40 baseline
  epochs (1640 steps) and equal steps in the other arm (29 epochs), with the
  per-epoch trace kept for the experiment arm since its curve is unmeasured.

Run order and GPU state. GPU temperature, clocks, power and throttle flags
sampled every 2 s through 13 runs; snapshot at the start and end of each.
Five baselines each after at least 3 min idle, then the same five seeds back
to back. logs/run_order/, results/run_order/runs.csv.

| | spaced | back to back |
|---|---|---|
| held-out MAE, seeds 0-4 | 9.91, 10.63, 10.19, 9.93, 9.68 | 9.28, 10.73, 10.53, 10.08, 10.72 |
| mean (sd) | 10.07 (0.36) | 10.27 (0.61) |
| GPU temp at start | 35-39 C | 42-45 C |
| GPU temp peak | 54-56 C | 55-56 C |
| core clock under load | 2896 MHz | 2896 MHz |
| power capping during run | 0.1-1.5 s | none |

- Difference +0.20 kt, exact permutation p = 0.55. With five against five a
  difference under about 0.65 kt would not show, so this rules out an effect
  the size of the earlier pattern (0.9 kt), not a small one.
- Nothing thermal happened to find. No thermal slowdown in any sample; the
  driver's thermal-slowdown counters read 0 since boot on 09-20, which
  covers every run of the day. The card peaks at 56 C and returns to idle
  within 3 minutes. Clock under load is the same in both blocks.
- Order across the day: 23 baseline reruns in sequence, held-out MAE vs
  position Spearman +0.22, p = 0.31. Eight of the 19 later runs are no
  better than the best of the first four, and five are worse than the first
  run. The "four earliest are the four worst" pattern was a run of luck in
  the six that followed them.
- One of ten logged quantities, wall-clock run length (50-57 s), correlated
  with held-out MAE at -0.75 in those ten runs, and survived dropping any
  one run. Ten things had been tried, so it was tested rather than believed:
  ten more back-to-back runs (seeds 0-9) with millisecond timing gave +0.25
  (p = 0.50). It does not replicate. results/run_order/replication_runs.csv.
- With those ten added, 33 reruns in sequence: held-out MAE vs position
  Spearman +0.07, p = 0.71.

Spread, again revised. 33 reruns at 20 epochs: held-out mean 10.05, sd
0.59, range 8.55 to 11.37. 9.00 is 1.8 sd below the mean and one rerun in
33 is at or below it. The typical 20-epoch result of this configuration is
about 10.0 kt. Same seed run three times (seeds 0-4) does not give similar
results (correlations +0.2 to +0.4): the seed carries little.

2026-10-13, 22:10. Class-imbalance experiment, final design. Written before
launch. This replaces the single-run plan in the first design entry.

Three arms, ten seeds each (0-9), run interleaved by seed (A0, B0, C0, A1,
...) so nothing that drifts over the session lines up with an arm:

| arm | training storms | frames | epochs | steps |
|---|---|---|---|---|
| A baseline | original nine (storms_train_baseline.txt) | 1307 | 40 | 1640 |
| B equal steps | eleven, Beryl and Lee added (storms_train.txt) | 1816 | 29 | 1653 |
| C equal epochs | eleven (storms_train.txt) | 1816 | 40 | 2280 |

Same in every arm: ResNet-18 IMAGENET1K_V1, L1 loss, Adam lr 1e-4, batch
32, validation storms (Sally, Ophelia, Laura), checkpoint = best validation
MAE within the arm's budget, preprocessing, exclusions. Every epoch's
weights saved.

    python scripts/train.py --train-list <list> --epochs <n> --seed <s>
        --out models/imbalance/<arm>/seed<s>/best_val.pt
        --epoch-dir models/imbalance/<arm>/seed<s>

Each selected checkpoint evaluated twice with scripts/evaluate.py:
--holdout-list storms_holdout_baseline.txt (Ian, Nicole, Claudette; 393
frames; comparable across arms and with everything above) and
--holdout-list storms_holdout.txt (adds Sam; 688 frames; the new claim).
Sam is in no training or validation list.

Read-out, fixed in advance: per arm mean, sd and range over the ten seeds of
overall MAE, fixes-only MAE and per-band bias, on both sets. The band the
experiment is about is >96 kt, reported first: MAE and bias, pooled and per
storm (the four-storm >96 kt band is 47 Ian frames and 188 Sam frames, so
pooled is mostly Sam). Arms are compared as independent samples of ten.
Reference spread is 0.59 kt (33 baseline reruns at 20 epochs); each arm's
own sd is reported too. B against A is the data effect at equal steps; C
against B is what more steps buy on the larger set.

2026-10-13, 23:40. Class-imbalance experiment, results.

All 30 runs completed; no train or eval failures. best_model.pt SHA-256
unchanged before and after. Integrity: every arm used the intended storms,
frames and epochs (A 9/1307/40, B 11/1816/29, C 11/1816/40). The three-storm
rows of each four-storm CSV match the three-storm CSV to 0.01 kt. Per-run and
per-arm tables: results/imbalance/summary_per_run.csv,
summary_by_arm_h3.csv, summary_by_arm_h4.csv, arm_differences.csv.

What was added. The loader's training frames by band:

| band | original nine | eleven | added |
|---|---|---|---|
| Cat 3 (96-112) | 84 | 212 | +128 (Lee 89) |
| Cat 4 (113-136) | 108 | 172 | +64 (Beryl 47) |
| Cat 5 (137+) | 66 | 93 | +27 |
| all >96 | 258 (19.7%) | 477 (26.3%) | +219 |

Cat 5 as a share of training is unchanged (5.0% -> 5.1%). Cat 4 goes from
8.3% to 9.5%. Most of the addition is Cat 3.

The >96 kt band. Mean (sd) over ten runs; per storm, since the two held-out
storms disagree.

| | A 9 st, 40 ep | B 11 st, 29 ep | C 11 st, 40 ep |
|---|---|---|---|
| Ian >96 MAE (47) | 14.66 (1.18) | 17.09 (1.37) | 17.80 (0.98) |
| Ian Cat 3 bias (26) | +8.06 (3.29) | +8.70 (3.01) | +12.40 (4.16) |
| Ian Cat 4 bias (17) | -18.86 (4.69) | -22.30 (4.04) | -19.68 (3.41) |
| Ian Cat 5 bias (4) | -21.93 (5.32) | -26.54 (4.95) | -24.84 (2.84) |
| Sam >96 MAE (188) | 17.04 (1.68) | 16.19 (2.41) | 15.16 (2.38) |
| Sam >96 bias | -16.14 (2.09) | -14.76 (2.79) | -12.49 (4.05) |
| Sam Cat 3 bias (43) | -14.73 (2.41) | -13.33 (2.80) | -10.69 (3.44) |
| Sam Cat 4 bias (145) | -16.56 (2.15) | -15.18 (2.85) | -13.02 (4.33) |
| 4-storm >96 MAE (235) | 16.56 (1.53) | 16.37 (1.99) | 15.68 (1.94) |
| 4-storm >96 bias | -13.76 (2.30) | -12.91 (2.61) | -10.46 (3.87) |

Primary tests, named in the design: >96 MAE and bias, two sets, three pairs
= 12. Bonferroni threshold about 0.004. Survivors: Ian >96 MAE B-A +2.44
(permutation p 0.0005) and C-A +3.14 (p < 0.0001). Both are worse. All ten C
runs are above A's worst run on this metric. Not surviving: 4-storm >96 bias
C-A +3.30 (p 0.034), Sam >96 bias C-A +3.66 (p 0.021), Sam >96 MAE C-A -1.88
(p 0.056). B against A on Sam: bias +1.38 (p 0.22), MAE -0.85 (p 0.38).
Everything else in arm_differences.csv (about 100 comparisons) is
exploratory.

Reading. Underestimation of Ian's Cat 4 and Cat 5 frames did not shrink in
either arm. The two arms make Ian's >96 MAE worse in different ways.
Weighting each band by its frame count (26/17/4 of 47):
- B-A +2.44 = Cat 3 +0.80, Cat 4 +1.24, Cat 5 +0.39. At equal steps, B's
  rise is mostly Cat 4/5 under-prediction getting deeper (Cat 4 -18.9 ->
  -22.3, p 0.10; Cat 5 -21.9 -> -26.5, p 0.06).
- C-A +3.14 = Cat 3 +2.60, Cat 4 +0.30, Cat 5 +0.25. At equal epochs, C's
  rise is Ian's Cat 3 frames, already over-predicted, pushed higher (Cat 3
  MAE 10.8 -> 15.5, p 0.002).
On Sam, which is under-predicted at both Cat 3 and Cat 4, C's upward shift
reduces the error. C moved high-end predictions up rather than separating
Cat 4/5 from Cat 3. This fits an addition that is mostly Cat 3 and leaves
the Cat 5 share unchanged.

Overall MAE. Mean (sd) [range]:

| | A | B | C |
|---|---|---|---|
| 3-storm overall | 9.39 (0.26) [8.87-9.71] | 10.02 (0.46) [9.25-10.70] | 9.93 (0.62) [9.07-10.83] |
| 3-storm fixes | 10.12 (0.47) | 10.79 (0.49) | 10.76 (0.61) |
| 4-storm overall | 11.45 (0.69) [10.55-12.73] | 11.63 (0.83) [10.37-13.16] | 11.34 (0.93) [10.09-13.19] |
| 4-storm fixes | 11.54 (0.75) | 11.79 (0.76) | 11.49 (0.95) |

Against 0.59 kt. On three storms, three differences exceed it: B-A overall
+0.62 (p 0.002), B-A fixes +0.67 (p 0.005) and C-A fixes +0.64 (p 0.018).
C-A overall, +0.54 (p 0.022), does not. C-B is -0.08. All of these are in
the direction of the expanded data doing worse on the original held-out
storms. On four storms nothing exceeds 0.59; the largest difference is
0.30 (fixes, C-B). The 0.59 figure is a 20-epoch spread; A's own sd at 40 epochs is
0.26. It is not a yardstick for band metrics, where arm sds are 1-2 kt for
MAE and 2-4 kt for bias.

B against C (does the larger set need more steps?). Overall MAE is
indistinguishable on both sets (C-B -0.08 and -0.29). B selected epochs
19-29, seven of ten at 24 or later; C selected 21-39, mean 31. So B may be
short of converged, but forty epochs bought nothing measurable overall. On
Sam >96, C is 1.0 kt better than B (p 0.35).

Other notes.
- Validation MAE is worse in both expanded arms (A 8.24, B 8.82, C 8.64).
  This is consistent in direction with the three-storm held-out result. It
  is not used for the comparison.
- A at 40 epochs: mean 9.39. The published 9.00 is 1.5 of A's sd below
  that. This is consistent with the earlier 9.58 estimate from three 40-epoch
  traces.
- Within C, held-out MAE against run order has Spearman -0.75: seeds 0-2
  ran first and are the three worst. Within A it is -0.30, within B -0.07.
  GPU snapshots show no throttle flags, and C durations are flat at
  135-139 s. Within an arm, order and seed are the same thing, so the two
  cannot be separated. Dropping C seeds 0-2 changes no sign. It does halve
  3-storm C-A overall (+0.54 -> +0.22) and moves 4-storm C-A overall from
  -0.12 to -0.57. Ian >96 MAE C-A stays at +2.79.
- Disk: models/imbalance is 47 GB. Per-epoch weights are kept until it is
  decided whether to extract per-epoch held-out traces for B and C.

2026-10-14, 00:30. Imbalance checkpoints deleted; write-up.

Before deleting models/imbalance (47 GB), every saved epoch of all 30 runs
was evaluated on validation and the four held-out storms
(scripts/imbalance_epoch_trace.py; inference only). Output:
results/imbalance/epoch_trace.csv (1090 rows, metrics on both held-out
sets) and heldout_predictions_by_epoch.csv.gz (749,920 rows). Checks: every
best_val.pt bit-identical to its selected epoch; recomputed 4-storm MAE at
the selected epoch equals evaluate.py's to 0.000 kt. models/ now holds only
baseline_9kt.pt, hash f95e6c7e...cc18e9cc, unchanged.

Paired within run, arm C's validation-selected checkpoint at a 29-epoch
budget vs 40: -0.01 kt (3 storms), +0.21 kt (4 storms).

Analysis scripts moved from the session scratchpad into scripts/:
imbalance_analysis.py, imbalance_extra.py. New: scripts/site_figures.py
recomputes every site figure from the 9.00 run for the 33 reruns and arm A
(results/site_figures/). The three control runs' per-sample predictions
were copied from the scratchpad to
results/epoch_traces/control_heldout_predictions_seed{0,1,2}.csv. Write-up:
results/imbalance/findings.md.

2026-10-14, 11:41. Loss-function experiment, design. Written before any training
run; not launched until approved.

Hypothesis. L1 optimizes the conditional median, which biases predictions
low in the right tail of a skewed label distribution. If that drives the
>96 kt underestimation, Huber or MSE should reduce it on the same data.

Arms. Everything except the loss is held fixed.

| arm | loss | train.py |
|---|---|---|
| L1 (control) | L1, as now | --loss l1 (default) |
| Huber | Huber, delta = 10 kt = 10/185 = 0.054 in label units | --loss huber |
| MSE | mean squared error | --loss mse |

The Huber delta is fixed here and is not tuned after seeing results. It is
quadratic below a 10 kt error and linear above (checked numerically).

Fixed in every arm:
- Training: original nine storms, storms_train_baseline.txt. The live
  storms_train.txt is the 11-storm list and is not used.
- Validation: unchanged (Sally, Ophelia, Laura; storms_val.txt).
- Optimization: ResNet-18 IMAGENET1K_V1, Adam lr 1e-4, batch 32, 40
  epochs.
- Data: same preprocessing and exclusions.
- Seeds 0-9, ten runs per arm, 30 in total.
The learning rate is not retuned per loss. Huber and MSE gradients are
far smaller than L1's in label units, but Adam normalizes each
parameter's step by its own gradient scale, so lr 1e-4 sets a comparable
step size for all three.
The L1 arm is the same configuration as arm A of the class-imbalance
experiment; its H3 numbers are a consistency check against A, not a test.

Epoch selection. Every arm selects its checkpoint on validation MAE in
knots, not on its own training loss. Before this change, train.py selected
on criterion(val), which equals validation MAE only for L1. train.py now
computes validation MAE separately and selects on it for every loss. The
logged "train X kt / val Y kt" figures are MAE in kt for every loss; for
non-L1 arms the loss values are appended to the line. For --loss l1, the
selection and the logged numbers are computed exactly as before.

Code changes, made before launch:
- train.py: --loss {l1,huber,mse}, default l1.
- evaluate.py: --train-list (default storms_train.txt, so existing
  behaviour is unchanged). The held-out overlap check runs against the list
  the checkpoint was trained on. Without this, H6 is refused, because Beryl
  and Lee are in the live 11-storm list.
- storms_holdout_h6.txt: new.
- scripts/loss_experiment.sh: the driver.
Only each run's validation-selected checkpoint is kept
(models/loss/<loss>/seed<s>.pt, about 1.3 GB in total). No per-epoch
weights.

Run order. Round-robin by seed: L1 s0, Huber s0, MSE s0, L1 s1, ... Every
arm runs in every round, so drift over the session hits all arms alike and
cannot appear as an arm difference. This does not remove the within-arm
seed/order confound. With one run per seed, seed s is always an arm's
(s+1)th run under any schedule, and last time's arms were already
interleaved this way. What changes is the check, which is fixed in advance:
held-out MAE against round, per arm and pooled. A trend that appears in
every arm is a session effect. A trend in one arm only, like C's last
time, is read as chance within that arm. GPU START/END snapshots are kept
as before.

Held-out sets. Every run is evaluated on both:
- H3: Ian, Nicole, Claudette (storms_holdout_baseline.txt), 393 frames,
  for continuity with earlier numbers.
- H6: H3 plus Beryl, Lee and Sam (storms_holdout_h6.txt), 1197 frames.
  None of the six is in the nine-storm training list or in validation.
  Beryl's two NaN crops near its peak stay excluded (excluded_crops.txt).

H6 by storm (frames from the loader):

| storm | frames | fixes | >96 kt | Cat 3 | Cat 4 | Cat 5 |
|---|---|---|---|---|---|---|
| Ian | 197 | 38 | 47 | 26 | 17 | 4 |
| Nicole | 122 | 25 | 0 | 0 | 0 | 0 |
| Claudette | 74 | 14 | 0 | 0 | 0 | 0 |
| Beryl | 260 | 47 | 104 | 39 | 47 | 18 |
| Lee | 249 | 42 | 115 | 89 | 17 | 9 |
| Sam | 295 | 50 | 188 | 43 | 145 | 0 |
| H6 | 1197 | 216 | 454 | 197 | 226 | 31 |

Primary tests, named before running. All six are on H6, pooled:

| # | metric | comparison |
|---|---|---|
| 1 | >96 kt bias | Huber - L1 |
| 2 | >96 kt MAE | Huber - L1 |
| 3 | overall MAE | Huber - L1 |
| 4 | >96 kt bias | MSE - L1 |
| 5 | >96 kt MAE | MSE - L1 |
| 6 | overall MAE | MSE - L1 |

- ">96 kt" means best-track label > 96 kt.
- Test: difference of arm means, ten runs against ten, treating runs as
  independent samples. Two-sided permutation p from 40,000 permutations.
- Bonferroni threshold: 0.05 / 6 = 0.0083.
- Expected direction under the hypothesis: >96 kt bias moves toward zero
  and >96 kt MAE falls. Overall MAE tests whether any gain costs accuracy
  elsewhere.
- Everything else is exploratory and labelled that way. That includes H3,
  per-storm results, per-category bands, fixes-only MAE, MSE against
  Huber, and L1 against arm A.

Detectable effect. With ten runs per arm, two-sided alpha 0.0083 and 80%
power, the smallest reliably detectable difference is about 1.7 times the
between-run sd. Using arm A's sds on the four-storm set as a guide (H6 sds
are not yet known), that is about 1.2 kt for overall MAE, 2.6 kt for >96 kt
MAE and 3.9 kt for >96 kt bias. Smaller real effects will likely read as
not significant.

Report:
- Per arm, mean +- sd [range] of overall MAE, fixes-only MAE and
  per-category bias, on H3 and H6.
- The >96 kt band per storm (Ian, Beryl, Lee, Sam), MAE and bias.
- The six primary tests with p against 0.0083, saying plainly whether any
  survive.
- The run-order check.
All 30 runs are reported whatever happens. No run is dropped or repeated
after the fact. If an arm diverges (for example, MSE training unstable),
that is a result and is reported as one.

What a null would and would not mean. If neither Huber nor MSE moves >96 kt
bias, the median-vs-mean effect of L1 is not what drives the
underestimation at this data size. A conditional-mean regressor also
shrinks toward the bulk of the training distribution when the image is
ambiguous, so a null does not rule out scarcity or ambiguity explanations.

2026-10-14, 11:46. Loss-function experiment, approved, with two
amendments. Written before launch.

- Tests are two-sided, as designed.
- Arm order rotates each round: L/H/M, H/M/L, M/L/H, repeating. Run order:
  l0 h0 m0 h1 m1 l1 m2 l2 h2 l3 h3 m3 ... L1 runs first in rounds 0, 3, 6
  and 9; Huber in 1, 4 and 7; MSE in 2, 5 and 8.
- scripts/loss_analysis.py was written before launch. The six primary
  tests and THRESHOLD = 0.0083 are hard-coded and are not edited after
  results exist. It was smoke-tested on synthetic predictions (a planted
  +5 kt high-end shift was detected and classified), then the synthetic
  files were deleted. No real results existed at that point.
- Added at approval, exploratory: for each arm, does any >96 kt change
  separate Cat 3 from Cat 4/5, or move the high end together? Ian is the
  check, being over-predicted at Cat 3 and under-predicted at Cat 4/5.
  Measures, against L1:
  - the shift in Cat 3 bias and in Cat 4/5 bias (same sign = moves
    together);
  - the shift in the predicted gap, mean prediction on Cat 4/5 frames minus
    mean on Cat 3 frames (wider = separates).
  Shown for Beryl, Lee and Sam too.

2026-10-14, 12:44. Loss-function experiment, results.

All 30 runs completed, no failures, no NaN. Both protected checkpoints'
SHA-256 unchanged before and after. Integrity: nine training storms, 1307
frames, 386 validation, 40 epochs and the logged loss match every arm. The
H3 rows of each H6 CSV match the H3 CSV to 0.017 kt. Output from
scripts/loss_analysis.py, unedited since launch. Tables in results/loss/.

Verdict: the hypothesis is not supported, and the effect runs the other
way. On H6, Huber and MSE make the >96 kt underestimation worse.

Primary tests: H6 pooled, two-sided, threshold 0.0083. The p column is the
pre-registered permutation p (40,000); the exact column enumerates all
184,756 splits.

| test | L1 | arm | difference | p | exact p |
|---|---|---|---|---|---|
| >96 bias, Huber - L1 | -14.46 | -16.66 | -2.20 | 0.0078 | 0.0084 |
| >96 MAE, Huber - L1 | 17.04 | 18.40 | +1.37 | 0.027 | 0.027 |
| overall MAE, Huber - L1 | 12.20 | 12.85 | +0.65 | 0.033 | 0.034 |
| >96 bias, MSE - L1 | -14.46 | -18.25 | -3.79 | < 0.0001 | 0.00004 |
| >96 MAE, MSE - L1 | 17.04 | 20.05 | +3.02 | < 0.0001 | 0.00005 |
| overall MAE, MSE - L1 | 12.20 | 14.00 | +1.80 | < 0.0001 | 0.00003 |

- Clear survivors: the three MSE tests, all worse than L1.
- Huber >96 bias sits on the threshold: 0.0078 by the pre-registered
  procedure, 0.0084 exactly. It is not counted as a clear survivor. Its
  direction is worse, not better.
- No test shows an improvement.

Exploratory:
- H3 alone would have pointed the other way. Ian's >96 MAE falls with Huber
  (-1.40, p 0.02) and MSE (-1.78, p 0.01). The cause is Ian's Cat 3
  over-prediction shrinking (Cat 3 bias +9.5 -> +6.7 -> +5.3), while Ian's
  Cat 4/5 bias is unchanged (shift -0.10 and +0.19). On H6, every other
  high-end storm gets worse: >96 bias Beryl -21.8 -> -24.9 -> -27.7, Lee
  -10.8 -> -12.7 -> -13.3, Sam -15.5 -> -17.5 -> -19.3.
- Cat 3 vs Cat 4/5 separation, Ian as the check. In L1, Ian's mean
  prediction on its Cat 4/5 frames is 3.5 kt below its Cat 3 frames, while
  the best track has them 24.6 kt above. Within Ian's high end, L1 does not
  rank Cat 4/5 above Cat 3 at all.
  - Huber and MSE widen that predicted gap (+2.75 kt, p 0.01; +4.43 kt,
    p < 0.001) only by lowering Cat 3. Cat 4/5 does not move up.
  - On Beryl and Sam, both bands move down together. On Lee, Cat 4/5 moves
    down further than Cat 3, so the gap narrows (-2.8, -7.5).
  - Neither loss raises any storm's Cat 4/5 predictions.
- H6 per-category bias, L1 / Huber / MSE: Cat 4 -18.9 / -21.3 / -23.0;
  Cat 5 (31 frames) -20.1 / -23.2 / -27.3.
- Consistency: the L1 arm's H3 mean is 9.46 +- 0.45 against arm A's 9.39 +-
  0.26 (p 0.69).
- Run order: no pooled trend (H3 Spearman -0.15, p 0.42; H6 -0.20, p 0.29).
  No position-in-round effect. One single-arm trend (Huber, H6, -0.66) is
  read as chance, per the rule fixed in advance.
- Validation MAE at the selected epoch: L1 8.25, Huber 8.56, MSE 9.52.
  MSE picks the latest epochs (33-40).

2026-10-14, 12:54. Embedding-neighbour test of the ambiguity
explanation, design. Written before the script runs. No training.

Question: is the high-end underestimation a representation problem (strong
storms' embeddings sit among training frames of widely varying intensity)
or a head problem (neighbours are high, but the fc layer still predicts
low)?

Checkpoint: one L1 checkpoint from models/loss/. Rule: the L1 run whose H6
>96 kt bias is closest to the L1 arm mean, which is seed 2 (-14.38 against
-14.46). Chosen by that rule before any embedding was computed.

Method:
- Penultimate features: the 512-d avgpool output before fc, eval mode,
  same preprocessing as training.
- Embeddings for every training frame (nine storms, 1307 frames) and every
  H6 frame. fc(embedding) must reproduce results/loss/l1_seed2_h6.csv.
- For each held-out frame, its k = 20 nearest training frames by Euclidean
  distance on the raw features (the head is linear on these). Cosine
  distance is a one-line sensitivity.
- Per frame, from the neighbours' labels: mean, median, sd, fraction
  >= 113 kt (Cat 4 or stronger), distinct training storms, distinct
  6-hourly fixes.
- Decomposition: pred - truth = (pred - neighbour mean) + (neighbour mean
  - truth). The first term is the head term, the second the
  representation term.
- The training set has 174 frames at >= 113 kt (108 Cat 4, 66 Cat 5), so an
  all-strong neighbour set is possible in principle.

Groups: held-out Cat 3, Cat 4 and Cat 5 (H6 >96 kt), Cat 4 also by storm.

References:
- Held-out frames < 64 kt, where the model is accurate.
- Training Cat 4/5 frames, with neighbours drawn from the other eight
  training storms only.

Decision rule, per held-out band:
- Representation, if the representation term is the larger in magnitude
  and the neighbour sets are wide (fraction >= 113 well below that of the
  training Cat 4/5 reference, label sd above the < 64 kt reference).
- Head, if the neighbour mean is near truth (fraction >= 113 high) and the
  head term carries most of the bias.
- Mixed, if neither term dominates.

Also reported: neighbour median minus mean per band. Positive means the
local label distribution is skewed downward, which bears on the
interpretation in results/loss/findings.md. This is reported, not used
for the verdict.

2026-10-14, 12:56. Embedding-neighbour test, results. Inference only.
Script: scripts/embedding_neighbours.py. Output: results/embedding_neighbours/.

Checkpoint models/loss/l1/seed2.pt, chosen by the rule. fc(embedding)
reproduces evaluate.py's H6 predictions to 0.0000 kt over 1197 frames.

k = 20 nearest training frames, Euclidean distance on the 512-d features.
Error = head term (pred - neighbour mean) + representation term (neighbour
mean - truth).

| group | n | truth | pred | nbr mean | nbr sd | frac >= 113 | head | repr |
|---|---|---|---|---|---|---|---|---|
| held-out Cat 3 | 197 | 103.5 | 95.7 | 94.2 | 10.2 | 0.17 | +1.6 | -9.3 |
| held-out Cat 4 | 226 | 123.8 | 104.5 | 102.8 | 11.1 | 0.32 | +1.7 | -20.9 |
| held-out Cat 5 | 31 | 141.4 | 120.1 | 123.3 | 11.6 | 0.79 | -3.2 | -18.1 |
| ref: held-out < 64 kt | 449 | 42.2 | 45.5 | 38.4 | 9.7 | 0.00 | +7.1 | -3.8 |
| ref: training Cat 4/5, other storms' nbrs | 174 | 133.7 | 132.6 | 123.7 | 11.7 | 0.78 | +8.9 | -10.0 |

Verdict against the rule fixed in advance:
- Cat 3 and Cat 4: representation. The representation term carries nearly
  all the bias; Cat 4's fraction >= 113 (0.32) is well below the training
  reference (0.78).
- Neighbour spread is not wide. Label sd is 10-12 kt, the same as both
  references. Held-out strong frames are not among mixed intensities; they
  sit in a neighbourhood centred too low. Held-out Cat 4 frames land among
  training frames averaging 103 kt. This is misplacement, not spread.
- Cat 5: fits neither of the user's two cases. 79% of neighbours are
  >= 113 kt, as in the training reference, but they centre on 123 kt, 18 kt
  below truth. The representation term still dominates (-18.1 against
  -3.2). The embedding does not separate Cat 5 from Cat 4. Caveat: at the
  top of the label range a neighbour mean is pulled down mechanically, and
  the training reference shows -10 kt even for frames the model fits.
- Predictions follow the neighbourhood: corr(pred, nbr mean) +0.92 over
  all >96 kt frames, against corr(truth, nbr mean) +0.45. The head is not
  where the bias arises; at most a small share at Cat 5 (-3.2; cosine
  distance -6.3).
- Ian: the inverted Cat 3 / Cat 4 gap is in the representation. Its Cat 3
  frames' neighbours average 110.9 kt and its Cat 4 frames' 108.3. Head
  terms are -0.7 and -3.2.
- Cosine distance gives the same picture: representation term -6.9 /
  -18.8 / -15.0 against head term -0.9 / -0.4 / -6.3.
- Neighbour sets are dominated by one storm: top-storm share about 0.58,
  3-4 distinct storms, about 12 distinct 6-hourly fixes. Dorian supplies
  56% of the neighbour slots for held-out Cat 4/5 frames.

Skew check, bearing on the interpretation in results/loss/findings.md.
Neighbour median minus mean: Cat 3 -0.02, Cat 4 -0.66, Cat 5 +0.41, all
>96 kt -0.31. Under 1 kt and mixed in sign. This local estimate does not
show the downward skew that interpretation needs, and is far too small to
account for MSE's -3.8 kt shift. findings.md is not yet changed.

Limitations: one checkpoint of ten. k-NN is a readout of the embedding,
not of the linear head, so the head term measures the gap between a
linear readout and a local average.

Addendum, same test: training reference split by band. Training frames
with neighbours drawn from the other eight storms:
- Training Cat 4 (108): truth 125.4, neighbour mean 118.6, representation
  term -6.8, fraction >= 113 0.68.
- Training Cat 5 (66; Dorian 35, Milton 29, Michael 2): truth 147.2,
  neighbour mean 132.1, representation term -15.1, fraction 0.94.
Reading:
- Held-out Cat 4 (-20.9, fraction 0.32) is about 14 kt worse placed than
  training Cat 4 frames are across storms: genuine misplacement.
- Held-out Cat 5 (-18.1) is only about 3 kt worse than the training Cat 5
  reference. Most of the Cat 5 shortfall is a ceiling that training Cat 5
  frames share. Looked up across storms, even they land about 15 kt low.
  The model fits them (pred 147.6) only through their own storm.

2026-10-14, 13:07. Storm-identity clustering in the embedding.
Results. CPU only. Script: scripts/storm_clustering.py; output:
results/embedding_neighbours/storm_clustering_l1_seed2.csv.

Embeddings: the saved L1 seed-2 training embeddings (1307 frames, nine
storms).

Method:
- For each training frame, its 20 nearest training frames, Euclidean,
  excluding itself and all same-storm frames within +-6 h.
- Measured: the fraction that come from the same storm.
- Chance, naive: the storm's share of the training set, as specified.
- Chance, intensity-matched (added): the storm's share among eligible
  frames within +-10 kt of the query's label. That is the same-storm
  fraction expected if neighbours were picked by intensity alone.
- Chance depends only on labels and timing, never on the embedding.
- A first run showed 0.000 everywhere: pandas parses these times at
  microsecond resolution, so the +-6 h window covered whole storms. Fixed,
  with an assert that consecutive frames are 1 h apart.

| band | n | observed | chance naive | chance intensity | obs/naive | obs/intensity |
|---|---|---|---|---|---|---|
| all | 1307 | 0.367 | 0.147 | 0.204 | 2.5 | 1.8 |
| TD | 304 | 0.416 | 0.111 | 0.186 | 3.8 | 2.2 |
| TS | 462 | 0.287 | 0.132 | 0.120 | 2.2 | 2.4 |
| Cat 1 | 157 | 0.222 | 0.151 | 0.156 | 1.5 | 1.4 |
| Cat 2 | 126 | 0.473 | 0.216 | 0.439 | 2.2 | 1.1 |
| Cat 3 | 84 | 0.353 | 0.190 | 0.314 | 1.9 | 1.1 |
| Cat 4 | 108 | 0.444 | 0.168 | 0.190 | 2.6 | 2.3 |
| Cat 5 | 66 | 0.729 | 0.196 | 0.418 | 3.7 | 1.7 |

Sensitivity, excluding same-storm frames within +-24 h: overall 0.200
against 0.147 naive (1.4x) and 0.135 intensity-matched (1.5x). Cat 4 is
2.0x and Cat 5 3.6x intensity-matched chance; Cat 1-3 fall to chance or
below.

Reading:
- Same-storm neighbours are above chance overall: 2.5x naive at +-6 h.
- Part of that is a storm resembling itself within a day: the excess
  halves when +-24 h is excluded.
- Part is intensity: storms with long plateaus at one intensity, such as
  Cat 2 and Cat 3, sit at about intensity-matched chance.
- Storm identity beyond time and intensity is clearest at Cat 4 and Cat 5
  (2.0x and 3.6x at +-24 h). That is where the embedding is most organized
  by storm, and it fits the earlier finding that training Cat 5 frames are
  fit through their own storm.
- The embedding is moderately organized by storm, concentrated at the high
  end. It is not far above chance everywhere.
- Caveat: same-storm clustering cannot tell storm-specific cues (framing,
  orientation, background) from a storm's real physical consistency over
  days.

2026-10-14, 13:08. Rotation-augmentation experiment, design. Written before
any code change or training run. Not launched until approved.

Hypothesis. The embedding is partly organized by storm identity, most at
Cat 4/5 (storm-clustering entry above). Held-out strong storms are placed
among moderate training frames (embedding-neighbour entry). If some of the
storm-specific signal is orientation -- how a particular storm's bands,
outflow or background sit in the frame -- then training on randomly
rotated frames removes it, and the network has to use features that carry
across storms. That should reduce the >96 kt underestimation on unseen
storms.

Arms:

| arm | circular mask | rotation |
|---|---|---|
| control | train, validation, held-out | none |
| rotation | train, validation, held-out | uniform 0-360 deg per training sample; none at validation or held-out |

No flips. A mirrored Northern Hemisphere storm turns the wrong way;
rotation keeps the sense of rotation. The mask is identical in both arms,
so the only difference is the rotation. Both arms are new; neither is
comparable to earlier unmasked runs.

Fixed in every run:
- L1 loss.
- Nine training storms (storms_train_baseline.txt), validation unchanged.
- ResNet-18 IMAGENET1K_V1, Adam lr 1e-4, batch 32, 40 epochs.
- Checkpoint selected on validation MAE.
- Seeds 0-9: 20 runs in total.
- Order: round-robin by seed, arm order alternating each round (C/R,
  R/C, ...).
- Only each run's selected checkpoint is kept (about 0.9 GB). About 45
  minutes in total.

Preprocessing, to be implemented after approval:
- One bilinear resampling for every frame. The native crop goes straight
  to 224 x 224 through an affine sampling grid (affine_grid + grid_sample),
  with angle 0 for the control arm, for validation and for held-out, and
  a random angle for rotation-arm training.
- Both arms therefore interpolate once. Rotating after the existing resize
  would blur only the rotation arm's training images.
- Rotation happens in the grid's normalized coordinates, which span the
  crop's square ground box, so it is a rotation on the ground even though
  the native pixels are not square.
- The inscribed circle (radius 112 px) lies inside the original square at
  every angle, so no masked-in pixel is ever sampled from outside the crop.
- Angles come from the seeded torch generator, so they are reproducible
  per seed.

Open decision, the mask fill value. Labels are BT normalized so 0 = 180 K
(the coldest value) and 1 = 320 K. A literal zero outside the circle would
paint every frame's corners as the coldest possible cloud tops, the
feature the network attends to most at high intensity. Recommendation:
fill with 1.0 (320 K, warm, clear-sky-like), so the mask reads as "no
cloud" rather than "deepest convection". Both arms get the same fill
either way. To be confirmed before code is written.

Checks before launch (no training):
- With angle 0 and no mask, the new path reproduces the current
  F.interpolate path to float tolerance.
- The rotation is proper, not a reflection. Check on a synthetic
  asymmetric pattern that the sense of rotation is preserved and the
  transform's determinant is +1.
- A few masked control frames and rotated training frames rendered to the
  scratchpad, to look at.
- evaluate.py and the neighbour scripts apply the same mask, given a flag.

Held-out set: H6 (Ian, Nicole, Claudette, Beryl, Lee, Sam; 1197 frames,
454 above 96 kt), masked and unrotated. H3 figures come from the H3 rows
of the H6 CSVs, as exploratory continuity.

Primary tests, named now:

| # | metric (H6 pooled) | comparison |
|---|---|---|
| 1 | >96 kt bias | rotation - control |
| 2 | >96 kt MAE | rotation - control |
| 3 | overall MAE | rotation - control |

- Two-sided. Bonferroni threshold 0.05 / 3 = 0.0167.
- Test: exact permutation test on the difference of arm means, 10 runs
  against 10, enumerating all 184,756 splits. Fixed in advance so the
  borderline Monte Carlo case of the loss experiment cannot recur.

Pre-registered secondary (one test, alpha 0.05, outside the Bonferroni
family):
- Metric: the same-storm neighbour fraction. For each run, embed the
  1307 training frames (masked, unrotated). For each frame, take its 20
  nearest training frames, Euclidean on the 512-d features, excluding
  itself and same-storm frames within +-6 h, and record the fraction from
  the same storm. Average over frames: one value per run.
- Comparison: rotation - control, exact permutation, two-sided.
- Expected direction if rotation works as hypothesized: down.
- Chance depends only on labels and timing, so it is identical in both
  arms and the raw fractions compare directly.
- Its run-to-run sd is not yet known. Only one checkpoint has been
  measured.

Exploratory, labelled as such: per-storm >96 kt MAE and bias; per-category
bias; fixes-only MAE; H3; the same-storm fraction at +-24 h and by band
(Cat 4/5 especially); the head/representation decomposition per arm;
selected epochs per arm; run order.

Detectable effect, with 10 runs per arm, alpha 0.0167 and 80% power:
about 1.57 times the run-to-run sd. With the unmasked L1 arm's H6 sds as a
guide, that is about 0.9 kt overall, 1.7 kt >96 kt MAE and 2.3 kt >96 kt
bias. The masked arms' sds may differ.

Risks fixed in advance:
- Rotation also removes physically meaningful orientation: north-south
  asymmetry, the usual direction of shear, where land sits in the frame.
  A null or negative result therefore would not refute the storm-identity
  explanation, only this route to fixing it.
- Augmentation can need more epochs. If most rotation runs select epoch
  38-40, that is reported as possibly under-trained. It is not a reason to
  rerun or extend.
- All 20 runs are reported. None is dropped or repeated.
"""

2026-10-14, 13:10. Rotation-augmentation experiment, approved, with
amendments. Written before any code change.

- Mask fill: 300 K, normalized (300 - 180) / 140 = 0.857 (6/7). That is
  like clear tropical ocean; 320 K is warmer than any real sea surface.
  Same fill in both arms and at validation and held-out.
- Added, pre-registered exploratory comparison: masked control arm against
  the unmasked L1 arm of the loss experiment (results/loss/l1_seed*_h6.csv).
  - Metrics on H6: overall MAE, >96 kt MAE and >96 kt bias, as specified.
  - Also < 64 kt MAE and bias (added here): masking removes the frame
    corners, where the weak-storm Grad-CAM attribution sat, so the weak
    band is where corner use would show.
  - Caveat: the two arms come from different sessions, so this is not an
    interleaved comparison.
- Before launch: the angle-0 resampling path must match the current
  F.interpolate resize. The measured maximum difference is reported.
- Secondary metric plumbing: after each run, the driver computes that run's
  same-storm neighbour fraction (scripts/storm_fraction.py; masked,
  unrotated training frames). The analysis script reads those values. The
  metric is validated first by reproducing the seed-2 L1 value (0.367)
  without the mask.

2026-10-14, 13:14. Rotation experiment, pre-launch checks. No
training before this entry.

- Resampling:
  - The affine-grid path at angle 0 is NOT bit-identical to the current
    F.interpolate resize. The maximum difference over all 2890 train, val
    and H6 frames is 1.1e-5 normalized, 0.0016 K: float rounding in the
    grid arithmetic.
  - So angle 0 goes through F.interpolate itself. The masked control, all
    validation and all held-out frames are bit-identical to the current
    resize inside the circle (2890/2890 frames) and exactly 300 K outside.
  - The grid path is used only for rotated training samples.
  - The default, unmasked dataset path is unchanged (checked equal).
- Rotation is proper. At 90 deg it matches rot90 to 1.5e-5, and misses
  both flips and the transpose by 1.0. Determinant +1. Angles are
  reproducible per seed; rotation angles use their own generator
  (seed + 10000).
- Rendered frames checked by eye (Michael, 136 kt): the eye stays
  centred, the corners are 300 K, and the spiral keeps its
  counter-clockwise sense under rotation.
- Secondary metric validated. scripts/storm_fraction.py on L1 seed 2,
  unmasked, gives 0.3669 at +-6 h and 0.1998 at +-24 h, reproducing
  storm_clustering.py. The same checkpoint with the mask gives 0.3782.
- scripts/rotation_analysis.py written before launch, with the three
  primary tests, THRESHOLD = 0.0167, the exact permutation test and the
  secondary hard-coded. It was smoke-tested on synthetic outputs (a
  planted +4 kt high-end lift and a -0.05 fraction drop were recovered),
  then the synthetic files were deleted.
- Code: dataset.py (circular_mask, rotate, resample(), MASK_FILL = 6/7);
  train.py (--circular-mask, --rotate); evaluate.py (--circular-mask);
  scripts/storm_fraction.py; scripts/rotation_experiment.sh.

2026-10-14, 13:58. Rotation-augmentation experiment, results.

All 20 runs completed, no failures. Both protected hashes unchanged.
Integrity: masks, rotation flags, nine storms, 1307 / 386 frames and 40
epochs as intended; every storm-fraction file computed on masked frames.
Output from scripts/rotation_analysis.py, unedited since launch. Tables in
results/rotation/. 20 checkpoints kept, 855 MB.

Primary tests: H6 pooled, rotation - control, two-sided exact permutation,
threshold 0.0167.

| test | control | rotation | difference | exact p | survives |
|---|---|---|---|---|---|
| >96 kt bias | -13.93 | -9.25 | +4.67 | 0.0028 | yes |
| >96 kt MAE | 16.87 | 14.67 | -2.21 | 0.019 | no |
| overall MAE | 12.32 | 11.83 | -0.49 | 0.25 | no |

Rotation reduces the >96 kt underestimation on H6 pooled by 4.7 kt. This
is the first intervention to survive correction in the improving
direction. The >96 kt MAE improvement (-2.2) does not survive (0.019
against 0.0167). Overall MAE does not change.

Secondary, pre-registered: same-storm neighbour fraction 0.436 -> 0.425
(-0.011, exact p 0.23). Not significant. The improvement did not come with
a measurable drop in storm clustering overall, so the pre-registered
mechanism is not supported. Exploratory: by band, the fraction drops at
Cat 4 (-0.05, p 0.03) and Cat 5 (-0.07, p 0.02), uncorrected.

Exploratory (uncorrected):
- Storms disagree. >96 kt bias, control -> rotation:
  - Beryl -19.1 -> -13.2 (p < 0.001)
  - Lee -10.1 -> -5.4 (p 0.005)
  - Sam -16.2 -> -10.5 (p 0.001)
  - Ian -2.8 -> -5.2 (p 0.17); Ian's >96 kt MAE gets worse, 15.5 -> 18.9
    (p < 0.001).
- H3 alone would have said rotation is harmful: overall MAE 9.72 -> 11.29
  (p < 0.001). H6 says it helps the high end.
- Ian as the check. Cat 3 bias is unchanged (-0.2); Cat 4/5 bias gets
  worse (-5.0, p 0.02). The predicted gap, already inverted, widens the
  wrong way (-4.8 -> -9.6). No separation on Ian.
- Other storms:
  - Beryl and Lee: Cat 4/5 lifts more than Cat 3 (+7.4 against +3.3 and
    +3.9), so the gap widens (p 0.002, 0.02). Partial separation.
  - Sam: both bands lift together (+5.3, +5.9).
- Rotation raises predictions at every intensity. Overall bias -8.2 ->
  -4.5. TD bias +5.0 -> +7.0. < 64 kt MAE worse, 7.96 -> 9.90
  (p < 0.001). H6 band bias shifts run from +2.0 (TD) to +7.3 (Cat 5),
  larger at the high end. So it is partly an upward block shift, steeper
  at the high end.
- Validation MAE improves (8.37 -> 7.54). Training MAE is higher (3.36 ->
  5.43), as expected under augmentation. Rotation selected epochs 26-40,
  with 3 of 10 at 38 or later, so not flagged as under-trained by the
  pre-set rule.
- Rotation runs are more consistent. H6 overall sd 1.24 -> 0.48; >96 kt
  bias sd 3.52 -> 2.45. The masked control's H6 overall MAE splits into
  two groups (10.8-11.9 and 13.0-14.0).

Pre-registered exploratory: masked control against the unmasked L1 arm
(loss experiment; different sessions). No difference anywhere:
- overall +0.11 (p 0.79)
- >96 kt MAE -0.17 (p 0.84)
- >96 kt bias +0.53 (p 0.66)
- < 64 kt MAE +0.01 (p 0.97)
- < 64 kt bias -1.27 (p 0.10)
Removing the frame corners did not change weak-storm accuracy.

Run order: pooled trend +0.32 (p 0.17). The first run in each round
averaged 0.75 kt worse than the second (exact p 0.064). Arm order
alternates, so each arm ran first five times, and this cannot bias the
arm comparison.

2026-10-14, 14:25. Rotation calibration check, design. Written before
computing. Inference only; no training.

Question: was rotation's >96 kt gain a recalibration (a shift any linear
correction would give the control too) or better discrimination?

Method, per run (20):
- Predict the validation frames (Sally, Ophelia, Laura; masked, unrotated)
  with the run's selected checkpoint. Check that the recomputed validation
  MAE matches the logged one at the selected epoch.
- Fit OLS truth = a + b * pred on that run's validation predictions only,
  never held-out. Apply it to the run's saved H6 predictions.
- Repeat the three primary comparisons on the corrected predictions: H6
  pooled >96 kt bias, >96 kt MAE and overall MAE, rotation - control,
  two-sided exact permutation, threshold 0.0167.
- Report a and b per arm.

Decision rule:
- Discrimination: rotation still wins a corrected high-end test (>96 kt
  bias or MAE) at 0.0167.
- Mostly recalibration: neither corrected high-end test survives, and the
  corrected high-end differences are under half the uncorrected ones.
- Mixed: anything else.

Exploratory, labelled as such:
- Pearson and Spearman correlation of prediction with truth on H6, overall
  and within >96 kt, per arm. A linear correction cannot change these, so
  they are a calibration-free measure of discrimination.
- Corrected < 64 kt MAE.
- Corrected per-storm >96 kt bias.

Caveat to report: validation above 96 kt is Laura only, so a and b are
fitted mostly on weaker frames.

2026-10-14, 14:27. Rotation calibration check, results. Inference
only. Script: scripts/rotation_calibration.py. Output:
results/rotation/calibration_per_run.csv, calibration_tests.csv and
<arm>_seed<s>_val.csv.

Integrity: recomputed validation MAE matches the logged value at the
selected epoch to 0.005 kt in all 20 runs.

Validation above 96 kt: 25 frames, all Laura; 17 at >= 113 kt; maximum
130 kt.

Correction truth = a + b * pred, fitted per run on validation:
- control: a +4.0 (sd 1.9), b 0.939 (sd 0.045)
- rotation: a +2.7 (sd 2.3), b 0.937 (sd 0.039)
The slopes are the same; both corrections shrink predictions slightly
toward the mean, so on H6 the corrected high end is lower in both arms.

Primary comparisons, rotation - control, exact permutation, threshold
0.0167:

| metric | raw difference (p) | corrected difference (p) | corrected / raw |
|---|---|---|---|
| >96 kt bias | +4.67 (0.0028) | +2.86 (0.00004) | 61% |
| >96 kt MAE | -2.21 (0.019) | -1.68 (0.0008) | 76% |
| overall MAE | -0.49 (0.25) | -0.26 (0.27) | |

The correction also cut the control's run-to-run sd (>96 kt bias sd 3.52
-> 1.46), which is why corrected p values are smaller even though the
differences shrank.

Verdict against the rule fixed in advance: discrimination. Both corrected
high-end tests survive; >96 kt MAE did not survive raw and does after
correction. About 40% of the bias gain and 25% of the MAE gain go away
under a linear recalibration; the rest remains.

Qualifier, from the calibration-free measures (exploratory). A linear
correction cannot change correlation:
- Overall Pearson r falls with rotation, 0.920 -> 0.905 (p 0.0001), as do
  Spearman's, from the weak-band degradation.
- Within >96 kt, Pearson 0.520 -> 0.542 (p 0.10) and Spearman 0.506 ->
  0.517 (p 0.46). No significant gain.
So rotation does not rank frames better, overall or within the high end.
What survives the correction is the high end sitting higher relative to
the rest: band bias shifts grow from +2.0 at TD to +7.3 at Cat 5, so the
response is less compressed at the top. One straight line fitted mostly
on validation's weaker frames cannot reproduce that for the control. It
is between-band separation, not within-band ranking.

Corrected per storm (exploratory). >96 kt bias, control -> rotation:
- Beryl -21.3 -> -17.3
- Lee -12.1 -> -9.3
- Sam -18.5 -> -14.7
- Ian -5.7 -> -9.5 (p 0.0001)
Ian's >96 kt MAE is worse by 3.8 (p 0.0001). Ian still goes the other way.
Corrected < 64 kt MAE stays worse with rotation, 7.66 -> 9.49
(p < 0.0001).

Caveat: validation's high end is one storm, Laura, 25 frames, maximum
130 kt. The correction is fitted mostly on weaker frames, so it can only
test linear recalibration. A correction specific to the high end cannot
be fitted without high-end validation data.

2026-10-14, 15:07. More-distinct-storms experiment, design (approved
with changes). Written before the build and before any training.

Hypothesis. The high-end bias is a representation problem: held-out Cat 4
frames sit among training frames averaging 103 kt, and the embedding is
organized by storm, most at Cat 4/5. Training on more distinct storms
should improve how the embedding places and ranks strong storms it has not
seen.

Storms (all GOES-16, 2018-2024, from the 2025 HURDAT2 release):
- New training, 3-hourly: Florence 2018, Humberto 2019, Lorenzo 2019,
  Delta 2020, Eta 2020, Larry 2021, Earl 2022, Tammy 2023, Helene 2024,
  Rafael 2024.
- Validation, both arms, hourly: Sally, Ophelia, Laura (as before), plus
  Teddy 2020 and Franklin 2023.
- Held-out H9, hourly: H6 (Ian, Nicole, Claudette, Beryl, Lee, Sam) plus
  Fiona 2022, Kirk 2024 and Iota 2020.
- Beryl, Lee and Sam are never in training.

Lorenzo was moved from held-out to training at approval, to add a fourth
distinct Cat 5 storm to training. Its Cat 5 is a single special HURDAT2
fix: 140 kt at 2019-09-29 03:00, with 130 kt six hours either side. So it
adds ONE frame at >= 137 kt at either sampling (training Cat 5 frames
66 -> 67), plus 14 frames at >= 113 kt and 27 at >= 96 kt at 3-hourly.
This experiment therefore tests storm variety mostly at Cat 3/4; it does
not materially change Cat 5 scarcity.

Per split. Frames as sampled, and storm-hours on the hourly grid. Counts
for storms not yet built are pipeline-grid estimates, which reproduced
existing manifests exactly; actual counts are recorded after the build.

| split | storms | frames | frames >=96 / >=113 / >=137 | storm-hours >=96 / >=113 / >=137 |
|---|---|---|---|---|
| control training (original nine, hourly) | 9 | 1307 | 258 / 174 / 66 | 260 / 174 / 66 |
| expanded training: control + new ten (3-hourly) | 19 | 2084 (1307 + 777) | 394 / 220 / 67 | 672 / 313 / 67 |
| validation, both arms (hourly) | 5 | 952 (386 + 566) | 134 / 61 / 0 | 134 / 61 / 0 |
| held-out H9 (hourly) | 9 | 1756 (1197 + 559) | 653 / 379 / 31 | 659 / 381 / 31 |

New storms, frames as sampled (>=96 / >=113 / >=137):
- Training, 3-hourly: Florence 146 (34/18/0), Humberto 59 (13/0/0),
  Lorenzo 77 (27/14/1), Delta 49 (12/1/0), Eta 104 (11/10/0), Larry 86
  (28/0/0), Earl 63 (0), Tammy 106 (0), Helene 37 (5/3/0), Rafael 50
  (6/0/0).
- Validation, hourly: Teddy 259 (61/15/0), Franklin 307 (48/29/0).
- Held-out, hourly: Fiona 232 (89/61/0), Kirk 181 (78/35/0), Iota 146
  (32/26/0).
- H9 above 96 kt: 653 frames from 7 storms (Ian 47, Beryl 104, Lee 115,
  Sam 188, Fiona 89, Kirk 78, Iota 32).

Sampling and weighting:
- New training storms are 3-hourly; validation and held-out storms are
  hourly. The original nine stay hourly as built.
- Dorian's share of training: 27.3% (control) -> 17.1% (expanded).
- The largest new storm is Florence, 7.0%. The new storms are 37% of
  expanded training frames and 10 of 19 storms.
- Known consequence: each original storm is still sampled three times as
  densely as each new one.

Arms (L1, unmasked, unrotated, ResNet-18 IMAGENET1K_V1, Adam lr 1e-4,
batch 32, checkpoint selected on validation MAE; seeds 0-9; 20 runs;
round-robin by seed, arm order alternating each round):

| arm | train list | frames | batches/epoch | epochs | steps |
|---|---|---|---|---|---|
| control | storms_train_control2.txt | 1307 | 41 | 40 | 1640 |
| expanded | storms_train_expanded.txt | ~2084 | 66 | 25 | 1650 (+0.6%) |

Both arms use --val-list storms_val_expanded.txt. The expanded arm's final
epoch count is fixed after the build: the whole number of epochs whose
step total is nearest 1640, from the actual post-exclusion frame count.
It is recorded here before training.

Validation change. Both arms select on the expanded validation set, so the
control differs from the expanded arm in training storms only. This
control is NOT identical to earlier L1 runs (the 9.00 kt baseline, the
seed and epoch reruns, arm A, the loss experiment's L1 arm), because
validation changed. Checkpoint selection now sees Teddy and Franklin.

Primary tests: on H9 pooled, expanded - control, two-sided exact
permutation test on the difference of arm means (all 184,756 splits),
Bonferroni 0.05 / 3 = 0.0167.
1. Spearman correlation between prediction and best track over all H9
   frames > 96 kt (653 frames). The metric no intervention has moved yet.
2. >96 kt bias.
3. Overall MAE.

Pre-registered secondary (alpha 0.05): the same-storm neighbour fraction
(k = 20, same-storm frames within +-6 h excluded, Euclidean on the 512-d
features). It is computed on a frame set common to both arms: the original
nine storms' 1307 training frames, with neighbours searched only among
them, so the chance level is identical. (Over each arm's own training set
it would be confounded: more storms lowers chance mechanically.) Expected
direction: down.

Exploratory, labelled as such:
- per-storm >96 kt Spearman, MAE and bias, and the mean of per-storm
  Spearman values;
- per-band bias;
- the H6 and H3 subsets;
- the Ian Cat 3 / Cat 4-5 check;
- selected epochs (most runs at 24-25 is flagged as possibly
  under-trained, not rerun);
- run order.
All 20 runs are reported.

Approximate detectable effects at alpha 0.0167 and 80% power (1.57 x
run-to-run sd): Spearman about 0.05, >96 kt bias about 2.3-5 kt, overall
MAE about 0.9 kt.

Not chosen:
- Leslie 2018 and Paulette 2020 (481 and 518 frames of Cat 1-2 would
  dominate).
- Bonnie 2022 (major phase in the East Pacific).
- Grace 2021, Epsilon and Zeta 2020 (short major phases; alternates).

2025 / GOES-19 storms: Dexter, Erin (140 kt), Gabrielle (120), Humberto
(140, 11 h >= 137), Imelda and Melissa (165 kt, 37 h >= 137). GOES-19
replaced GOES-16 as GOES-East on 2025-04-07, and the pipeline reads the
noaa-goes16 bucket by default. These storms are a candidate future
cross-satellite test set. They are left untouched, not downloaded, and
in no split, until that test is designed.

Code and data for this design:
- train.py --val-list.
- List files storms_val_expanded.txt, storms_train_control2.txt,
  storms_train_expanded.txt and storms_holdout_h9.txt. Checked: the
  control trains on exactly the original nine; both arms share validation
  and H9; no overlaps.
- Build with scripts/build_batch.py: the ten training storms at
  --sample-interval 3h; Teddy, Franklin, Fiona, Kirk and Iota at 1h.
  About 1902 frames, 2.5-2.9 h.
- Then check_nan.py regenerates excluded_crops.txt.
- No training until the build is done and the exclusions have been
  reviewed.

2026-10-14, 17:31. More-distinct-storms build and exclusions. No
training yet.

Build (scripts/build_batch.py; logs in logs/storm_build/), all 15 storms
succeeded:
- Ten training storms at 3h: 775 of 777 sample times matched (Helene
  36/37, Rafael 49/50). 15:07-16:06.
- Five validation and held-out storms at 1h: 1121 of 1125 matched (Kirk
  177/181). 16:06-17:29.
- About 4.6 s per frame.

check_nan.py regenerated excluded_crops.txt: 21 of 4794 frames on disk,
was 8. None removed; 13 added, all crops with missing pixels in the source
full-disk scan (blocks of 6-100% of the crop):
- Teddy (validation), 10 frames, 2020-09-17 18Z to 09-18 14Z, 108-120 kt,
  including the 120 kt peak fix. NaN blocks of 6-57% that move from hour
  to hour: data dropouts across consecutive scans. Teddy keeps 51 of its 61
  frames >= 96 kt and 10 of 15 >= 113 kt.
- Franklin (validation), 2 frames: 2023-08-31 05Z (86 kt, 33% NaN) and
  09-01 05Z (71 kt, 100% NaN).
- Larry (training), 1 frame: 2021-09-11 06Z (65 kt, 100% NaN).

Actual counts after exclusion. Frames, then >= 96 / >= 113 / >= 137 kt:
- control training: 1307; 258 / 174 / 66
- expanded training: 2081 (1307 + 774); 394 / 220 / 67
- validation: 940; 124 / 56 / 0 (Teddy's exclusions take 10 of the 134
  frames >= 96 and 5 of the 61 >= 113)
- H9: 1752; 653 / 379 / 31 (Fiona, Kirk, Iota: 555; 199 / 122 / 0)

Step budget, fixed now:
- control: 40 epochs x 41 batches = 1640 steps
- expanded: 25 epochs x 66 batches = 1650 steps (+0.6%)
Dorian is 27.3% of control training and 17.2% of expanded.

2026-10-14, 18:08. Exclusions approved; the rule is unchanged (any NaN pixel
excludes the frame), including the ten Teddy frames at its peak.

Validation has no frames at or above 137 kt: Sally, Ophelia, Laura, Teddy
(peak 120) and Franklin (peak 130). Epoch selection is therefore blind at
Cat 5 in both arms. Validation MAE says nothing about how a checkpoint
does on Cat 5 frames, and any Cat 5 result on H9 (31 frames: Ian 4,
Beryl 18, Lee 9) comes from checkpoints chosen without seeing that range.

2026-10-14, 18:10. More-distinct-storms experiment, pre-launch. No training
before this entry.

- scripts/storms_analysis.py was written before launch, with the three
  primary tests (H9 pooled: Spearman over frames > 96 kt, >96 kt bias,
  overall MAE; expanded - control), THRESHOLD = 0.0167, the exact
  permutation test and the secondary (same-storm fraction on the original
  nine storms' 1307 frames) hard-coded.
  - SHA-256 77aeb599585b861ed50c8f9c42cd0a46707713acc7eead419488b09ce26a946f.
    The driver also records it at launch, in
    logs/storms/analysis_sha256_at_launch.txt.
  - It checks every run against the recorded design: lists, validation
    list, epochs (40 / 25), storms (9 / 19), frames (1307 / 2081),
    validation frames (940).
  - Smoke-tested on synthetic outputs; the planted Spearman, bias and
    secondary effects were all recovered. The synthetic files were
    deleted.
- scripts/storms_experiment.sh (SHA-256 9416dde0...2301ae8):
  - L1, unmasked, unrotated, --val-list storms_val_expanded.txt for both
    arms.
  - Control: storms_train_control2.txt, 40 epochs. Expanded:
    storms_train_expanded.txt, 25 epochs.
  - Seeds 0-9, arm order alternating each round (C/E, E/C, ...).
  - Evaluated on storms_holdout_h9.txt; the secondary comes from
    storm_fraction.py.
  - Keeps only each run's selected checkpoint (models/storms/), and
    records both protected checkpoint hashes before and after.
- Pre-training setup is committed and pushed (a9d07f4).

2026-10-14, 18:53. More-distinct-storms experiment, results.

All 20 runs completed, no failures. Both protected hashes unchanged. The
analysis script's SHA-256 is identical at launch and after (77aeb599...).
Every run matched the recorded design (lists, validation list, epochs
40 / 25, storms 9 / 19, frames 1307 / 2081, validation 940). Output from
scripts/storms_analysis.py, unedited. Tables in results/storms/. 20
checkpoints kept in models/storms/ (855 MB).

Primary tests: H9 pooled, expanded - control, two-sided exact
permutation, threshold 0.0167.

| test | control | expanded | difference | exact p | survives |
|---|---|---|---|---|---|
| Spearman, frames > 96 kt | 0.523 (0.053) | 0.580 (0.024) | +0.057 | 0.0024 | yes |
| >96 kt bias | -13.12 (2.01) | -7.72 (1.43) | +5.40 | 0.00003 | yes |
| overall MAE | 10.87 (0.63) | 9.27 (0.29) | -1.60 | 0.00001 | yes |

All three survive, in the improving direction. It is the first time any
intervention has moved high-end ranking. All ten expanded runs have lower
overall MAE than the best control run.

Secondary, pre-registered: same-storm neighbour fraction on the original
nine storms' frames, 0.365 -> 0.369 (+0.004, p 0.78). Null. The
improvement did not come with a less storm-organized embedding of those
storms, so the pre-registered mechanism is not supported.

Flags fixed in advance:
- Under-training: 6 of 10 expanded runs selected epoch 24 or 25 (control
  19-40). Flagged as possibly under-trained, not rerun. If so, the effect
  is understated, not overstated.
- Run order: in the control arm, H9 overall MAE rises with seed (Spearman
  +0.72). The expanded arm is flat (+0.01). Pooled against run order, arm
  means removed: +0.44 (p 0.05). No position-in-round effect (-0.00 kt,
  p 0.99). By the rule fixed in advance, a trend in one arm only is read
  as chance within that arm. It is strong here, so stated plainly: seed
  and order are confounded within an arm and cannot be separated. It does
  not decide the overall-MAE result: the best control run (9.88) is worse
  than the worst expanded run (9.64).

Exploratory (uncorrected):
- Ranking within storms, > 96 kt Spearman: mean per-storm 0.523 -> 0.570
  (p 0.0006).
  - Gains on Sam (+0.089, p 0.0009), Fiona (+0.119, p 0.0006) and Lee
    (+0.078, p 0.06).
  - Unchanged on Kirk (0.821 both), Iota (+0.033) and Ian (0.03 -> 0.07;
    Ian has essentially no ranking above 96 kt in either arm).
  - Beryl -0.026 (p 0.14).
  - Pooled Pearson above 96 kt +0.034 (p 0.08).
- Bias by band, H9: TD 0.0, TS -0.4, Cat 1 +2.9, Cat 2 +3.5, Cat 3 +4.5,
  Cat 4 +6.3, Cat 5 +2.7 (p 0.07). < 64 kt MAE unchanged (7.89 -> 7.84,
  p 0.83). Unlike rotation, the high-end gain costs nothing at the weak
  end.
- Cat 5 underestimation remains: -21.6 -> -18.8, not significant.
  Validation is blind at Cat 5, and Lorenzo added one frame.
- > 96 kt MAE per storm, control -> expanded: Sam 16.8 -> 10.7, Kirk 10.5
  -> 6.3, Iota 10.1 -> 6.2, Beryl 24.1 -> 21.3, Fiona 9.1 -> 6.4, Lee 13.4
  -> 11.1, Ian 14.4 -> 14.5 (no change).
- Ian check: Cat 3 +2.0 and Cat 4/5 +2.1 (neither significant); the
  inverted predicted gap is unchanged (-2.8 -> -2.7). No separation on
  Ian. Fiona's gap widens (+2.9, p 0.02); the other storms move together.
- H6 subset: overall MAE -1.77, > 96 kt MAE -3.72, > 96 kt bias +5.33,
  Spearman +0.053 (p 0.016).
- H3 subset: overall MAE -0.52 (p 0.036); nothing above 96 kt, which on
  H3 is Ian only.
- Expanded runs are more consistent (H9 overall sd 0.63 -> 0.29).
  Validation MAE at the selected epoch is 9.53 (control) and 8.52
  (expanded).

Caveat on attribution. The expanded arm has more distinct storms, but
also more training frames (2081 against 1307) and more high-end frames
(394 against 258 at >= 96 kt; 220 against 174 at >= 113), at equal steps.
This design cannot separate "more distinct storms" from "more frames" or
"more high-end frames". Cross-experiment and exploratory: adding Beryl
and Lee hourly in the class-imbalance experiment (+509 frames, +219 >= 96)
did not help, which suggests variety matters more than volume, but those
experiments used different held-out sets and are not directly comparable.

2026-10-14, 18:57. More-distinct-storms experiment, run-order robustness
check. Asked for after the results; not a primary test. Script:
scripts/storms_robustness.py; output:
results/storms/robustness_control_seeds0-4.csv.

The three primary comparisons were repeated using only control seeds 0-4
(the earlier and better runs; run orders 1, 4, 5, 8, 9) against all ten
expanded runs. Same two-sided exact permutation test (all 3003 splits of
5 against 10), same 0.0167 threshold.

| metric | control 0-4 | expanded | difference | exact p | passes |
|---|---|---|---|---|---|
| Spearman, frames > 96 kt | 0.555 | 0.580 | +0.025 | 0.053 | no |
| >96 kt bias | -11.93 | -7.72 | +4.22 | 0.0010 | yes |
| overall MAE | 10.40 | 9.27 | -1.13 | 0.0003 | yes |

The bias and overall-MAE results hold. The ranking result does not: its
gain halves and misses the threshold. That result depends partly on the
control's later, worse runs and is treated as suggestive, not
established, in results/storms/findings.md.

2026-10-14, 18:58. Variety-against-volume experiment, design (DRAFT, not
approved, nothing run). Uses only data already on disk; no downloads.

Question: the more-distinct-storms experiment improved >96 kt bias,
overall MAE and (less robustly) high-end ranking, but its expanded arm had
more storms, more frames and more high-end frames at once. Holding the
number of frames and their intensity mix fixed, does drawing them from
many storms at sparse sampling beat drawing them from fewer storms at
dense sampling?

Pools (training storms only; never validation or H9):
- Dense pool: the original nine storms at hourly, as built. 1307 frames.
- Sparse pool: all 19 training storms at 3-hourly. The original nine are
  thinned to their 3-hourly grid points, which are a subset of the hourly
  crops on disk because the 3-hour grid and every exact fix are on the
  hourly grid. That gives 457 frames, plus the ten new storms as built
  (774), for 1231 frames.

Matching, per intensity band: each arm draws exactly target = min(dense
pool, sparse pool) frames.

| band | dense pool | sparse pool | target | dense kept | sparse kept |
|---|---|---|---|---|---|
| TD | 304 | 192 | 192 | 63% | 100% |
| TS | 462 | 437 | 437 | 95% | 100% |
| Cat 1 | 157 | 220 | 157 | 100% | 71% |
| Cat 2 | 126 | 152 | 126 | 100% | 83% |
| Cat 3 | 84 | 119 | 84 | 100% | 71% |
| Cat 4 | 108 | 85 | 85 | 79% | 100% |
| Cat 5 | 66 | 26 | 26 | 39% | 100% |
| total | 1307 | 1231 | 1107 | 85% | 90% |

Both arms therefore have 1107 frames: 195 above 96 kt, 111 at >= 113 kt,
26 at >= 137 kt, identical in every band.

| arm | storms | storms with frames >= 96 / >= 113 / >= 137 | largest storm (expected) | sampling |
|---|---|---|---|---|
| dense (few storms) | 9 | 5 / 5 / 3 | Dorian ~29% | ~0.85 frames per storm-hour |
| sparse (many storms) | 19 | 13 / 10 / 4 | Florence ~12%; Dorian ~10% | ~0.33 frames per storm-hour |

Known limit: at Cat 5 both arms draw from the same storms (Michael,
Dorian, Milton; the sparse arm also has Lorenzo's single frame), and the
dense arm keeps only 39% of its Cat 5 frames. So there is almost no
variety contrast at Cat 5. The contrast lives in Cat 3/4 and below.

Draws: stratified random sampling without replacement within each band,
drawn separately for each training seed (subset seed = training seed), so
the comparison generalizes over the thinning, not one draw. All 20 frame
lists are written before launch, and checked: per-band counts identical
between arms for every seed, 1107 frames each, no validation or H9 frame.
They are recorded with their hashes.

Arms: L1, unmasked, unrotated, ResNet-18 IMAGENET1K_V1, Adam lr 1e-4,
batch 32, seeds 0-9 (20 runs), round-robin with alternating arm order,
--val-list storms_val_expanded.txt for both arms (no Cat 5 in
validation, as before), evaluated on H9. Equal steps: 1107 frames is 35
batches per epoch; 47 epochs is 1645 steps, the same in both arms and
+0.3% on the 1640 used before.

Primary tests, unchanged from the more-distinct-storms experiment: on H9
pooled, sparse - dense, two-sided exact permutation (all 184,756 splits),
Bonferroni 0.05 / 3 = 0.0167.
1. Spearman correlation over H9 frames > 96 kt.
2. >96 kt bias.
3. Overall MAE.

Pre-registered secondary (alpha 0.05), changed from last time. The
same-storm neighbour fraction is computed on the H9 frames (k = 20,
Euclidean on the 512-d features; neighbours among the other H9 frames;
same-storm frames within +-6 h excluded). Last time it was computed on
the original nine storms' frames. Here the dense arm trains on about 85%
of those frames and the sparse arm on about a third, so measuring there
would be biased toward the dense arm. Neither arm sees H9 in training,
and the chance level is identical. Expected direction if variety acts
through less storm-specific features: lower in the sparse arm.

Interpretation, fixed in advance:
- Sparse beats dense on a primary: at fixed volume and intensity mix,
  variety helps.
- No difference on any primary: the more-distinct-storms gain is better
  explained by volume (more frames, more high-end frames) than by variety.
  This design can say that only at its own power: about 0.9 kt overall,
  2-5 kt bias, 0.05 Spearman.
- Dense beats sparse: density, within-storm detail, matters more.

Exploratory, labelled as such:
- per-storm > 96 kt Spearman, MAE and bias, and the mean of per-storm
  Spearman values;
- per-band bias and < 64 kt MAE;
- H6 and H3 subsets;
- the Ian check;
- selected epochs (most runs at epoch 46-47 flagged as possibly
  under-trained);
- run order, with the control-seeds-0-4 robustness check repeated in
  advance as: each arm's seeds 0-4 against the other arm's ten runs.

Code needed before launch (after approval):
- train.py --frame-list: restrict the training dataset to listed crop
  paths.
- scripts/variety_subsets.py: writes and checks the 20 frame lists.
- storm_fraction.py: a held-out frame-set option.
- The driver, and the analysis script written and hashed before launch.

2026-10-14, 19:08. Variety-against-volume experiment, approved with two
changes; final design with the resulting counts. Supersedes the draft
above where they differ. No training before this entry.

Change 1, per-seed draws. Each arm's training frames are drawn fresh for
every seed, deterministic from the seed (numpy default_rng([seed, arm])),
so the result averages over frame selection. All 20 frame lists are
recorded in results/variety/frame_lists/<arm>_seed<s>.txt, with SHA-256s
in sha256.txt (itself 2ec78fe0...0e). Re-running scripts/variety_subsets.py
reproduces every list byte for byte (checked).

Change 2, matching on 5 kt bins. Bins are 5 kt wide, split at the band
edges (34/64/83/96/113/137) so every bin sits inside one Saffir-Simpson
band: 36 bins. In every bin where both pools have frames, each arm draws
min(dense, sparse) frames.
- 33 bins match exactly.
- 3 bins cannot, because only the dense pool has frames in them. All are
  narrow slivers at band edges, holding hourly-interpolated labels the
  3-hourly pool never produces: [34, 35) TS, 9 frames; [83, 85) Cat 2, 6;
  [113, 115) Cat 4, 3.
- Fallback: those orphan bins were pooled within their band and matched
  by count. The sparse pool has no orphans in those bands, so the fallback
  matched 0 frames, and the 18 dense frames are dropped. No frame is
  matched at band level; every frame in both arms is matched within its
  5 kt bin.

Resulting per-arm counts, identical in every seed, checked:

| band | frames per arm | dense mean kt | sparse mean kt | sparse - dense | storms dense / sparse |
|---|---|---|---|---|---|
| TD | 191 | 26.88 | 26.49 | -0.39 | 8 / 17 |
| TS | 371 | 46.57 | 46.02 | -0.55 | 9 / 19 |
| Cat 1 | 155 | 72.38 | 72.05 | -0.33 | 7 / 17 |
| Cat 2 | 118 | 89.11 | 88.55 | -0.56 | 5 / 14.7 |
| Cat 3 | 72 | 103.59 | 102.99 | -0.61 | 5 / 12.6 |
| Cat 4 | 80 | 124.41 | 123.50 | -0.91 | 5 / 10 |
| Cat 5 | 26 | 148.45 | 147.85 | -0.60 | 2.4 / 4 |
| all | 1013 | 64.58 | 64.06 | -0.52 | 9 / 19 |

Means and storm counts are averaged over the ten draws.
- Both arms: 1013 frames; 178 at >= 96 kt, 106 at >= 113, 26 at >= 137.
- Largest single storm: dense 28.4% (Dorian), sparse 11.5% (Florence).
- Residual mismatch: the sparse arm is 0.3-0.9 kt lower within every
  band, systematically. Within a 5 kt bin, 3-hourly labels sit more often
  on best-track values at the bin floor, while hourly interpolated labels
  spread across the bin. It is small against the effects this design can
  detect (about 2 kt bias); it is reported, not corrected.

Step budget: 1013 frames is 32 batches per epoch; 51 epochs is 1632
steps (-0.5% on 1640), the same in both arms. Under-training flag, fixed
now: most runs of an arm selecting epoch 50 or 51.

Everything else as in the draft:
- L1, unmasked, unrotated; --val-list storms_val_expanded.txt for both
  arms; evaluated on H9; seeds 0-9 round-robin with alternating arm
  order.
- Primary tests: H9 pooled, sparse - dense, two-sided exact permutation,
  0.0167. (1) Spearman over frames > 96 kt, (2) >96 kt bias, (3) overall
  MAE.
- Secondary (alpha 0.05): same-storm neighbour fraction on the H9 frames.
- Robustness, not primary: each arm's seeds 0-4 against the other arm's
  ten runs, same test and threshold.

Code, all written before launch:
- train.py --frame-list. Checked with --epochs 0, which trains nothing:
  it filters to the 1013 listed frames from 19 storms, and it refuses a
  list containing a validation frame.
- storm_fraction.py --storms-list. The default is unchanged: it reproduces
  the storms-experiment control seed 0 value, 0.3443, exactly. With
  storms_holdout_h9.txt it runs on 1752 frames from 9 storms.
- scripts/variety_subsets.py (SHA-256 417ca282...8b).
- scripts/variety_experiment.sh (fa87e8d3...ef). It records the analysis
  and frame-list hashes at launch and checks the frame lists are
  unchanged at the end.
- scripts/variety_analysis.py, SHA-256
  2ba4d24709fc2c8be5ee2cab937e214373ba3f62951c4aa4714b54f651fe17a5. Its
  primary tests, threshold, secondary and robustness check are
  hard-coded. Smoke-tested on synthetic outputs (the planted effects were
  recovered); the synthetic files were deleted.

2026-10-14, 19:59. Variety-against-volume experiment, results.

All 20 runs completed, no failures.
- Protected checkpoint hashes, the analysis SHA-256 (2ba4d247...) and all
  20 frame lists are unchanged since launch.
- Every run matched the recorded design: its own seed's frame list, 1013
  frames from 9 (dense) or 19 (sparse) storms, 51 epochs, 940 validation
  frames, L1, unmasked, unrotated, secondary on 1752 H9 frames.
- Output from scripts/variety_analysis.py, unedited. Tables in
  results/variety/. 20 checkpoints in models/variety/ (855 MB).

Primary tests: H9 pooled, sparse - dense, two-sided exact permutation,
threshold 0.0167.

| test | dense (9 storms) | sparse (19 storms) | difference | exact p | survives |
|---|---|---|---|---|---|
| Spearman, frames > 96 kt | 0.532 (0.024) | 0.564 (0.040) | +0.031 | 0.047 | no |
| >96 kt bias | -11.14 (2.64) | -6.88 (2.34) | +4.26 | 0.0017 | yes |
| overall MAE | 10.75 (0.66) | 9.55 (0.39) | -1.21 | 0.00016 | yes |

By the rule fixed in advance ("sparse beats dense on a primary: at fixed
volume and intensity mix, variety helps"), variety helps: on >96 kt bias
and overall MAE, at an identical frame count and identical counts in
every 5 kt bin. Ranking above 96 kt is not established (p 0.047 against
0.0167), as in the robustness check of the more-distinct-storms
experiment.

Secondary, pre-registered: same-storm neighbour fraction on H9, 0.336 ->
0.314 (-0.022, p 0.30). Null, the third time this mechanism has not
shown up.

Robustness, fixed in advance (each arm's seeds 0-4 against the other
arm's ten):
- Bias passes both ways: +3.82 (p 0.009) and +5.11 (p 0.003).
- Overall MAE passes both ways: -0.99 (p 0.0003) and -1.27 (p 0.003).
- Spearman fails both ways: p 0.13 and 0.23.

Flags fixed in advance:
- Under-training: epoch 50 or 51 selected by 1 of 10 dense and 3 of 10
  sparse runs. Not flagged.
- Run order: within arm, dense +0.42 and sparse +0.02. Pooled +0.27
  (p 0.25).
- The first run of each round was 0.52 kt better than the second (exact
  p 0.021). Arm order alternated, so each arm ran first five times, and
  this cannot bias the arm comparison.

Residual intensity mismatch: the sparse arm's training labels are 0.3-0.9
kt lower within every band. That would push the sparse arm's predictions
down, the opposite of the observed +4.3 kt bias gain, so it cannot explain
the result.

Exploratory (uncorrected):
- Bias by band, sparse - dense: TD -0.4, TS -0.7, Cat 1 +1.5, Cat 2 +2.5,
  Cat 3 +3.5 (p 0.004), Cat 4 +5.1 (p 0.001), Cat 5 +1.9 (p 0.32). The
  gain is at Cat 3/4. At Cat 5, where both arms draw on the same three
  storms, nothing is detectable.
- Weak storms improved slightly: < 64 kt MAE 8.50 -> 7.91 (p 0.016).
- > 96 kt per storm, dense -> sparse:
  - Sam: MAE 15.8 -> 11.6 (p 0.0001), bias +5.4
  - Kirk: MAE 9.8 -> 6.7 (p 0.0001), bias +5.6
  - Iota: MAE 9.9 -> 5.7 (p 0.004), bias +8.4, Spearman +0.105 (p 0.0001)
  - Fiona: Spearman +0.100 (p 0.0007), bias +3.7
  - Lee: bias +3.7 (p 0.004)
  - Beryl: MAE -1.3 (p 0.17), Spearman -0.026 (p 0.02)
  - Ian: MAE 13.9 -> 14.3, Spearman 0.17 -> 0.07 (p 0.11)
- Ian check: Cat 3 +1.0 and Cat 4/5 +0.6 (neither significant); gap -1.5
  -> -1.8. No separation on Ian, again. Fiona's gap widens (+3.6,
  p 0.001); the other storms move as a block.
- H3 subset: overall MAE -0.43; nothing above 96 kt (Ian only).
- Validation MAE at the selected epoch: dense 9.86, sparse 8.35.
- Against the more-distinct-storms experiment (cross-experiment; the
  frame counts and step budgets differ, so not a test):
  - that experiment's 9 -> 19 storms with more frames gave bias +5.40 and
    MAE -1.60;
  - here, 9 -> 19 storms with equal frames gives +4.26 and -1.21, about
    three-quarters of each.
  Variety alone accounts for most of that gain.

2026-10-15, 22:57. Final test on the 2025 GOES-19 storms, design. Not run.
No 2025 test frame has been built, and no model has seen any 2025 data.

Question: does the more-distinct-storms model carry over to storms it
has never seen, imaged by a different satellite? H9 can no longer
answer that. It has been used across several experiments and it
favoured the expanded arm, so its numbers are optimistic. The 2025 set is
used once, and then it is spent.

Model, frozen by this entry:
- An ensemble of the ten expanded-arm checkpoints of the
  more-distinct-storms experiment (models/storms/expanded/seed0-9.pt). Their
  SHA-256s are in results/ensemble/members_sha256.txt, committed with this
  entry; models/ is not in git, so that file is the definition.
- The prediction is the unweighted mean of the ten members' predictions
  per frame. No recalibration, bias correction, weighting, member
  selection or test-time augmentation. Preprocessing is dataset.py's,
  unchanged: unmasked, unrotated, 180-320 K scaling, bilinear to 224.
- Code: scripts/ensemble_eval.py (SHA-256 658770d7...0a84) runs it, with
  --expect-sha results/ensemble/members_sha256.txt, which refuses to run on
  any other checkpoints. scripts/ensemble_report.py (SHA-256
  854c166f...4429) computes the metrics. Both hashes are recorded here and
  checked before the run. On H9 (exploratory, results/ensemble/findings.md)
  each member reproduces its committed single-model CSV exactly.
- Reference, fixed now, evaluated in the same single pass: best_model.pt
  (f95e6c7e...cc18e9cc), with scripts/evaluate.py. It is reported beside
  the ensemble; it is not the test, and it does not change what the test
  reads.

Test set (storms_test_2025.txt, committed with this entry):

| ATCF id | name | first fix | HURDAT2 entries |
|---|---|---|---|
| AL042025 | Dexter | 2025-08-02 | 43 |
| AL052025 | Erin | 2025-08-11 | 68 |
| AL072025 | Gabrielle | 2025-09-17 | 45 |
| AL082025 | Humberto | 2025-09-24 | 29 |
| AL092025 | Imelda | 2025-09-26 | 38 |
| AL132025 | Melissa | 2025-10-21 | 49 |

- HURDAT2 entry counts include statuses the build drops.
- Only header lines and first-fix dates were read; no intensities.
- Labels come from the cached HURDAT2 copy, data/interim/hurdat2_atlantic.txt
  (SHA-256 ee35819c...0bcb5c), which contains the 2025 season. It must
  not be refreshed before the build.
- Humberto 2025 shares its name with Humberto 2019 (AL092019, a training
  storm). Everything is keyed on ATCF id; ensemble_report.py's per-storm
  rows already are.
- load_splits confirms no overlap with storms_train.txt or
  storms_train_expanded.txt.

GOES-19 check (results/goes19_check/):
- One GOES-19 scan (2025-06-15 12:00, outside every test storm's
  lifetime, cropped at an arbitrary point) was read through
  goes_index.find_nearest_scan(bucket="noaa-goes19") and crop.py, and
  compared field by field with a GOES-16 scan. The files were deleted. No
  model was run.
- Same file naming (platform G19 matches the existing regex), same key
  layout, same Mode 6 full disk.
- Identical projection metadata (lon_0 -75.0), grid and CMI packing.
  Same crop shape.
- What differs: the instrument (FM4 against FM1) and its band-13 Planck
  coefficients (fk1 10860.40 against 10803.30). Both are calibrated
  brightness temperature, but the spectral response differs slightly, so
  the same scene can read slightly differently. Not corrected; this is
  part of what the test measures.
- noaa-goes16 has no band-13 full-disk scans on 2025-09-01. These storms
  exist only on GOES-19.
- Gap: pipeline.run_pipeline does not pass a bucket to goes_index or crop,
  and build_dataset.py has no flag for it.

Order of operations, each step committed before the next:
1. This entry: the members' hashes, both scripts' hashes, the storm list.
2. Pipeline change only: a --bucket flag on build_dataset.py, passed
   through run_pipeline to find_nearest_scan and process_one_scan /
   download_scan. The default stays noaa-goes16. Check that the default
   is unchanged: for a sample of existing G16 manifest rows, the index
   returns the same source_key. No existing storm is rebuilt.
3. Build the six storms, settings identical to H9 except the bucket:
   - hourly grid plus every exact fix (H9 manifests are hourly);
   - ABI-L2-CMIPF band 13, full disk;
   - 600 km box, 10 min tolerance, statuses LO,TD,SD,TS,SS,HU;
   - --bucket noaa-goes19.
   Then check_nan.py regenerates excluded_crops.txt under the unchanged
   rule (any NaN pixel excludes the frame). The 21 existing entries must
   be unchanged; the new ones are reported per storm. Per storm, report
   sample times, matched frames and missing scans. Missing scans stay
   missing. Nothing is re-cropped or rebuilt after any prediction is
   seen.
4. Domain-shift diagnostic, fixed now and computed before any model
   runs. Per band, 2025 against H9: crop BT mean, sd, 5th percentile and
   fraction of pixels below 200 K. Descriptive only.
5. Evaluate once:
   - ensemble_eval.py --expect-sha results/ensemble/members_sha256.txt
     --train-list storms_train_expanded.txt --holdout-list
     storms_test_2025.txt;
   - evaluate.py --checkpoint best_model.pt --holdout-list
     storms_test_2025.txt;
   - ensemble_report.py --reweight-to results/ensemble/expanded10_h9.csv.
   Outputs go to results/final_2025/, all committed, every number
   reported.

Metrics, fixed now:
- Overall MAE, fixes-only MAE, overall bias.
- >96 kt MAE and bias; Spearman over frames > 96 kt, pooled; <64 kt MAE
  and bias.
- Per-band bias, MAE and frame count.
- Overall MAE reweighted to H9's band shares (2025 per-band MAE, H9
  weights), so a change in intensity mix is not read as a change in
  skill.
- Per storm, by ATCF id: MAE; and, where a storm has at least 10 frames
  > 96 kt, >96 kt MAE, bias and Spearman.
- The ten members' mean, sd and range on each metric (from the pred_seed
  columns).

Reading, fixed now:
- There is no inferential test. Six storms of autocorrelated hourly
  frames and one model give no sampling distribution to test against.
  The results are compared descriptively with the ensemble's H9 values:
  overall 8.57, fixes-only 8.45, >96 kt MAE 10.82, >96 kt bias -7.72,
  Spearman 0.601, per-band bias TD +7.4 to Cat 5 -18.8.
- Some degradation is expected: H9 picked the arm, and the satellite
  changes. The test reports how much and where (by band, by storm),
  rather than pass or fail.
- Validation has no frames at or above 137 kt, so the models are blind
  there. Cat 5 frames are reported separately and also shown within the
  reweighted figure.

If the result is poor, it is still the result:
- No retraining or tuning on 2025 data.
- No second evaluation.
- No change of members, averaging or preprocessing.
- If a bug in the evaluation code is found afterwards, it is fixed and
  rerun, and both runs are reported with the reason.
- The 2025 set is not used for model selection afterwards.

2026-10-15, 22:59. Final test on the 2025 GOES-19 storms, design: three
corrections. These supersede the entry above where they differ. Still
nothing built or run.

1. Output paths are pinned. Without --out, evaluate.py writes
   results/heldout_predictions.csv, which must never be overwritten. Step 5
   runs exactly:
   - ensemble_eval.py --expect-sha results/ensemble/members_sha256.txt
     --train-list storms_train_expanded.txt --holdout-list
     storms_test_2025.txt --out results/final_2025/ensemble_2025.csv
     --sha-out results/final_2025/members_sha256_at_run.txt
   - evaluate.py --checkpoint best_model.pt --train-list storms_train.txt
     --holdout-list storms_test_2025.txt --out
     results/final_2025/best_model_2025.csv
   - ensemble_report.py --csv results/final_2025/ensemble_2025.csv
     --reweight-to results/ensemble/expanded10_h9.csv --out
     results/final_2025/ensemble_2025_metrics.csv
2. The domain-shift diagnostic gets its own script, written and hashed
   before the build: a new step between the pipeline change (2) and the
   build (3). Its SHA-256 goes in decisions.md before any 2025 crop exists.
3. HURDAT2: hurdat2.download_hurdat2 returns the cached file without
   touching the network when it exists and force is False, so the build
   uses data/interim/hurdat2_atlantic.txt as long as --force-hurdat2-refresh
   is not passed. Its SHA-256 (ee35819c...0bcb5c) is rechecked after the
   build and reported.

Known limit of the frozen ensemble_report.py: the reweighted overall MAE
sums over every band H9 has. If the 2025 set has no frames in one of
those bands, it returns NaN. That is reported as NaN and not patched; the
per-band MAEs stand on their own.

2026-10-15, 23:05. Final test on the 2025 GOES-19 storms: user decisions,
bucket plumbing, diagnostic script. Written and committed before any 2025
crop is built. data/processed/ has no 2025 storm.

User decisions on the design above:
- best_model.pt stays as a reference. It is evaluated in the same single
  pass and labelled as a reference, not part of the test.
- All three extras stand:
  - the domain-shift diagnostic, with its script written and hashed now;
  - the overall MAE reweighted to H9's band shares;
  - storms_test_2025.txt, already committed in 12e579e.

Step 2, bucket plumbing (src/hurricane_intensity/pipeline.py, crop.py,
scripts/build_dataset.py):
- build_dataset.py gets --bucket. run_pipeline(bucket=...) passes it to
  goes_index.find_nearest_scan and to crop.process_one_scan, which passes
  it to download_scan.
- Every default is config.GOES_BUCKET (noaa-goes16).
- Check that the default path is unchanged: three existing GOES-16 crops
  were rebuilt through the real run_pipeline, with no bucket given, into
  a scratch out_root. Only the sample grid was narrowed to the one sample
  time. All three matched the crops on disk byte for byte, from the same
  source scans:
  - Michael AL142018 2018-10-09 12:00 (scan mode 3), 649563 bytes
  - Ian AL092022 2022-09-27 06:00, 663107 bytes
  - Kirk AL122024 2024-10-05 12:00, 650255 bytes
  No existing storm was rebuilt in place.

Step 2b, domain-shift diagnostic: scripts/domain_shift.py, SHA-256
d439e7361d6a66e02e3fc7e2cddc9fd36b4f5f9d1a97b847e62e360614c7c3b4.
- It uses the same frame set TCDataset uses (manifest rows with a crop,
  minus excluded_crops.txt) and the native crops, before resampling and
  before the 180-320 K clip.
- Per frame: mean BT, BT sd, 5th percentile, fraction below 200 K,
  fraction outside 180-320 K. These are averaged per band and overall.
- It never loads a model.
- Scale reference, run now on existing data
  (results/final_2025/domain_shift_reference_h3_vs_h9.csv): H3 against H9
  differs by up to about 8 K in band-mean BT (all frames +10.5 K, mostly
  composition). Two sets of GOES-16 storms already differ this much, so a
  2025 difference is read against that, not against zero.
- Run after the build and check_nan.py, before any evaluation:
  domain_shift.py --set H9 storms_holdout_h9.txt --set 2025
  storms_test_2025.txt --out results/final_2025/domain_shift_2025_vs_h9.csv

Step 3, the build, exactly as fixed. For each of AL042025, AL052025,
AL072025, AL082025, AL092025, AL132025:
- build_dataset.py --atcf-id <id> --bucket noaa-goes19
- All defaults: 1h grid plus exact fixes; ABI-L2-CMIPF band 13; 600 km;
  10 min; statuses LO,TD,SD,TS,SS,HU; no --force-hurdat2-refresh.
- Logs go to logs/final_2025/build_<id>.log.
- Then check_nan.py; the HURDAT2 cache SHA-256 (ee35819c..., unchanged
  now) is rechecked.
- The 21 existing exclusions must be unchanged.
- Nothing is re-cropped or rebuilt after any prediction is seen.
- The evaluation (step 5) waits for the user.

2026-10-16, 00:27. Final test on the 2025 GOES-19 storms: build done,
pre-evaluation note. Written before any prediction on 2025 data.

Build:
- 1047 sample times, 1045 crops, all GOES-19. The 2 missing are
  Gabrielle 2025-09-22 17:00 and 18:00, with no scan within 10 min.
- check_nan.py excluded no 2025 frame. The 21 existing exclusions are
  unchanged (5839 frames).
- The HURDAT2 cache SHA-256 is unchanged (ee35819c...).
- 272 frames are > 96 kt. Frames per band, TD to Cat 5: 22 / 456 / 166 /
  129 / 84 / 137 / 51.
- Logs in logs/final_2025/.

Domain-shift diagnostic (results/final_2025/domain_shift_2025_vs_h9.csv):
- The cold tail is colder in 2025 in every band: the 5th percentile is
  3-8 K lower in every band, and in every band but TD more pixels are
  below 200 K.
- The effect is strongest at Cat 5: 0.174 of pixels below 200 K against
  0.042 in H9.
- Mean brightness is within normal variation (the H3-vs-H9 reference
  differs by up to about 8 K per band).
- No pixel in either set lies outside the 180-320 K scaling range.

Expected effects, stated now:
- The model's attribution at high intensity concentrates on cold
  convection. A colder tail may therefore raise predictions and partly
  offset the known low bias. A smaller 2025 high-end bias is not to be
  read as better skill without this caveat.
- Melissa peaked above the maximum training intensity: 165 kt against
  160 kt (Dorian). 8 Melissa frames are above 160 kt; validation's
  maximum is 130 kt. Predictions there are expected to saturate.

2026-10-16, 00:29. Final test on the 2025 GOES-19 storms, results. Run
once.
- The members, both scripts and best_model.pt matched their frozen hashes
  just before the run. The commands were exactly as pinned.
- 1045 frames from six storms. The full write-up is in
  results/final_2025/findings.md.
- best_model.pt is a reference, not part of the test. Its metrics come
  from the same frozen ensemble_report.py, run on a copy of its CSV with
  the prediction as the one member column.

| metric | ensemble 2025 | ensemble H9 | members 2025, mean (sd) | best_model.pt 2025 |
|---|---|---|---|---|
| overall MAE | 9.78 | 8.57 | 10.42 (0.26) | 12.13 |
| fixes-only MAE | 9.79 | 8.45 | 10.42 (0.45) | 11.71 |
| MAE reweighted to H9 bands | 10.05 | 8.57 | 10.64 (0.34) | 12.26 |
| >96 kt MAE | 12.65 | 10.82 | 13.18 (0.87) | 15.36 |
| >96 kt bias | -8.54 | -7.72 | same | -11.83 |
| Spearman > 96 kt | 0.729 | 0.601 | 0.711 (0.028) | 0.684 |

Per-band bias, TD to Cat 5:
- 2025: +7.6, +3.6, -6.3, -6.4, -7.9, -6.2, -15.7.
- H9: +7.4, -0.8, -2.3, -2.6, -4.2, -9.5, -18.8.

Per storm, MAE (and >96 kt bias / Spearman where there are at least 10
frames above 96 kt):
- Dexter 6.13.
- Erin 9.88 (-17.1 / 0.53).
- Gabrielle 8.24 (-1.6 / 0.82).
- Humberto 11.73 (-7.4 / 0.72).
- Imelda 8.91.
- Melissa 10.98 (-6.2 / 0.82).

Reading, descriptive as fixed:
- Error rose by 1.2 kt from H9 (1.5 kt at H9's intensity mix).
- The ensemble beat all ten members again, and it beats best_model.pt on
  every pooled metric.
- The shift is not uniform: TS over-prediction grows, Cat 1-3
  under-prediction grows by about 4 kt, and Cat 4/5 under-prediction
  shrinks. Per the pre-evaluation note, the smaller high-end bias is not
  read as better skill (colder 2025 cold tail).
- Exploratory: Melissa's 8 frames above 160 kt are predicted at 133-143 kt
  (bias -23). That is consistent with saturation, but it can't be
  separated from the general Cat 5 underestimation.

The 2025 set is now spent.

2026-10-18, 23:21. Held-out storms in the epoch-budget choice; protocol
fix. No training, no evaluation.
- The 40-epoch budget (2026-10-13, night) was chosen from per-epoch and
  per-budget held-out MAE on Ian AL092022, Nicole AL172022 and Claudette
  AL032021, the original three held-out storms, all now in H9. No
  list-membership check can catch that. Found in review on 2026-10-18.
- Exposed: any held-out result that contains the original three and
  comes from a model trained to 40 epochs or about 1640 steps. That
  covers control (storms_train_control2.txt, 40 epochs) and expanded
  (25 epochs, step-matched), so the ten ensemble members
  (models/storms/expanded/) too. It also covers the variety-against-
  volume runs (read on H9) and the imbalance, loss and rotation runs
  (read on H3, H4 or H6, which contain the three). The traced
  configuration was the nine-storm baseline, so any favour in the
  more-distinct experiment goes to the control-like arm and can't
  inflate expanded minus control.
- The size can be bounded, and the bound is close to zero. The same
  three runs logged validation MAE at every epoch
  (results/epoch_traces_60/epoch_trace.csv). Ten-epoch window means,
  validation / held-out:
  - 11-20: 10.00 / 10.54
  - 21-30: 9.52 / 10.15
  - 31-40: 8.88 / 9.67
  - 41-50: 8.80 / 9.64
  - 51-60: 9.02 / 9.84
  Validation alone shows the same shape and would have pointed to the
  same budget.
- Sam AL182021 was also evaluated per epoch (2026-10-14 00:30, imbalance
  traces). No recorded choice used it.
- The 2025 GOES-19 test result (2026-10-16, 00:29) is unaffected. H9 and
  its traces informed the budget; no 2025 storm was used in any model,
  budget or ensemble choice, and the 2025 set was built and evaluated
  once, after the ensemble was frozen.
  - Caveat for reading it: every comparison with H9 in that entry uses H9
    as the reference (overall and band-mix MAE, Spearman, per-band
    bias). The 2025 design already called H9 optimistic, because H9
    picked the arm. The budget adds a second source, bounded near zero
    above.
  - Separately, the record contradicts itself on whether 2025
    intensities were read before the test. The 2026-10-14
    more-distinct-storms design lists their peaks, while the 10-15
    design says no intensities were read.
    The peaks were seen; no model choice used them.
- From now on:
  - H9 results are reported pooled and split into the original three
    (Ian, Nicole, Claudette) and the six added later (Sam AL182021,
    Beryl AL022024, Lee AL132023, Fiona AL072022, Kirk AL122024, Iota
    AL312020).
  - Per-epoch traces are computed on validation storms only. No
    held-out storm is evaluated per epoch.
  - The existing held-out traces (results/epoch_traces*/,
    results/imbalance/epoch_trace.csv,
    results/imbalance/heldout_predictions_by_epoch.csv.gz) are not
    used for any further choice.
  - The rule is procedural for now: epoch_trace_tag.py and
    imbalance_epoch_trace.py still default to held-out lists.
Recorded at the user's request, 2026-10-18.

2026-10-19, 14:20. N5 Atlantic-only arm: design amendment (John, ASK A).
No build step, training or evaluation run by this entry.
- John's decision, verbatim: "ASK A: approved with two conditions.
  (a) Tie rule: when it fires, the checkpoint train.py actually saved is
  the one evaluated on H9, never a post-hoc choice; this applies to both
  arms, including the control's matched-grid re-selection. (b) Abort: no
  H9 evaluation runs until all 20 training runs have finished, so an
  abort can never follow a look at held-out results; restarted runs keep
  their original seed and run-order position. Record both as a dated
  amendment in decisions.md with the new design.md and analyze_n5.py
  hashes, in PR B, before launch."
- Implemented in PR B (PR #12) as reports/v2_survey/n5/design.md
  section 14. Sections 0-13 are unchanged. hc-adversary item 16's fixes
  are included.
  - (a) analyze_n5.control_selection, TIE_KT = 1e-4 kt. The driver and
    the analysis share this one function. When the re-validated argmin
    misses train.py's pick by less than TIE_KT, train.py's
    models/n5_atlantic/control/seed<s>.pt is evaluated.
    - The manifest records tie_rule, and odd_grid_near_ties (measured
      against the odd-grid minimum).
    - report.txt lists both, and the seeds whose primary is off the
      odd grid.
    - The atlantic arm always evaluates train.py's own checkpoint.
  - (b) Two commands, with committed evidence between them:
    - run_n5.py launch is phase 1 only: the 20 trainings, each with a
      training record, and no H9. It then stops.
    - The 20 records are committed and pushed.
    - run_n5.py evaluate is phase 2. It refuses unless:
      - every record is tracked and identical to its blob at HEAD;
      - the records come from one launch commit L on the first-parent
        chain;
      - HEAD is pushed;
      - HEAD differs from L only by the records.
    - Before the first H9 evaluation, evaluate runs every
      training-side check the analysis will apply, over all 20 records
      (analyze_n5.check_training, single_commit, verify_self at L).
  - launch refuses records or manifests from another commit or
    analysis blob.
  - Aborts are recorded in abort.json, written as pending first, then
    the files are moved, then it is marked complete. One lock covers
    every stage. The commands:
    - abort-run: an unfinished run, or one whose record no longer
      verifies. It restarts with the same seed in the same order
      position.
    - abort-all: every run, before any H9 output, so all 20 restart
      from a fix commit (John's decision 2 below).
    - abort-eval: a crashed H9 evaluation. The same checkpoint is
      re-evaluated.
    No abort is allowed after any H9 output, except abort-eval.
  - Procedural rule: no evaluate.py or ensemble_eval.py on
    models/n5_atlantic/* except through run_n5.py evaluate.
  - Guards:
    - HEAD must be on origin/phase2-experiments' first-parent chain
      after a fetch.
    - The checkout must be the main checkout, a child python must
      import hurricane_intensity from ROOT/src, and CUDA must be
      available.
    - seed<s>.pt must hold the same tensors as epoch_<native>.pt.
    - The epoch directory must resolve inside ROOT.
- Decided by John, 2026-10-19 15:48 (relayed by the coordinator),
  verbatim: "1. Even-epoch tie: (i), evaluate train.py's own
  checkpoint, marked off-grid. 2. Fix-commit restart: (i), restart all
  20 from the fix commit, only before any H9 output exists. 3.
  Odd-epoch near tie: keep the first minimum and record it."
  - 1: when the tie rule fires on an even native epoch, that seed's
    primary is train.py's checkpoint, off the odd grid. The manifest
    sets off_grid_primary; the analysis verifies it; report.txt lists
    the seed as an "off-grid primary".
  - 2: abort-all, legal only while no H9 output exists; launch from the
    fix commit then retrains all 20 in the original order.
  - 3: two odd epochs within TIE_KT while train.py's pick is
    reproduced: the pre-registered first minimum over the odd grid
    stands; odd_grid_near_ties (against the odd-grid minimum) is
    recorded, verified and reported.
- Hashes, as SHA-256 of `git show <commit>:<path>` bytes (blob id in
  brackets), before (e8e4fc1) -> after (the PR #12 commit that records
  them):
  - reports/v2_survey/n5/design.md:
    c08312358429067522691b761a512538c9e9e9e98ed783d280058e77778ff525
    [4638f3d8] ->
    a6e11b3763d9bba78e98de6444877d6f5259d0e68a37a1861f57909a43de0017
    [6eee3a47]
  - scripts/analyze_n5.py:
    cc9083cabcc48ec442d3ca8c24bdb1347d33edc859117a5a6aa51f14d98ee6dc
    [5692d6d3] ->
    2a70bf187beace35023e46572e0c2caec1433edd1bb75465759e2971c6147a82
    [97a8dd2f]
  - tests/test_analyze_n5.py:
    343059b8008ed7365fd92dbea07e61ab214660ab5a992d0a3f4aa02d3ad89f38
    [d2d15bde] ->
    b10aaaabd84e088660e31a57e038cdd36c61a61a5c7f9f2815316b8171b5addb
    [64c95e95]
- Notes:
  - The approved draft test file
    (reports/v2_survey/n5/test_analyze_n5.py, d4d1d56b...3cbe) differs
    from the moved tests/test_analyze_n5.py at e8e4fc1 (343059b8...) in
    imports and paths only (PR #11).
  - The test file now writes its synthetic trees under a per-process
    folder, so concurrent runs don't collide.
  - scripts/analyze_n5.py still carries the draft's "DRAFT r2 ... not yet
    approved" docstring. It was left byte-identical apart from this
    amendment (hc-adversary item 15).
  - TIE_KT was added between the "fixed before launch" markers. The
    launch blob will differ from 2a70bf18... in exactly the
    ARM_TRAIN_FRAMES and FRAME_LIST_SHA256 lines. Those are filled from
    the build in PR B and recorded with the frame-list SHA-256 before
    launch.
