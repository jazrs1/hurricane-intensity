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
      - HEAD differs from L only by the records and the decisions docs
        (logs/decisions.md, reports/decisions/*.md; never executed),
        per git diff --no-renames -z.
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
    No abort-run or abort-all is allowed once H9 has been looked at:
    any pinned H9 output or manifest, any aborted H9 evaluation, or
    logs/n5_atlantic/phase2_started.json, which evaluate writes
    (exclusive create) before its first H9 step and nothing moves;
    launch refuses while it exists. evaluate also refuses pending
    aborts, and checks each run's eval attempts before its H9 step.
    abort-eval also takes a manifest that load_run refuses.
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
    615e0f6f52b438125a76e32bb259f55a77cd732c137bf206575c1022142aaaea
    [f35f0050]
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

2026-10-19, 16:51. N5 Atlantic-only arm: build results, exclusions,
frame list and the two post-build constants (PR B part 2). No training
or evaluation.
- Build (coordinator, main checkout, 14:04-16:40). 31/31 storms built,
  0 failed. 2002 sample times: 1990 matched, 12 missing_scan. The new
  folders under data/processed are exactly the 31 IDs in
  storms_n5_new.txt; all 6143 pre-existing files are byte-unchanged.
  The HURDAT2 cache (ee35819c...), its source.json and
  excluded_crops.txt are unchanged. Scans are G16 only, 2018-2024.
- Exclusions: the merged check_nan.py, run from the PR worktree with
  --processed-root <main>/data/processed. It read the main checkout's
  data and wrote only the worktree's excluded_crops.txt. The main
  checkout's data/ (8203 files, size and mtime), its git status and
  its excluded_crops.txt were identical before and after.
  - 44 of 6784 frames excluded. The 21 existing entries are kept, in
    order.
  - 23 entries added, all under the 31 new IDs: Don AL052023 (1),
    Leslie AL132018 (4), Paulette AL172020 (18). The design estimated
    about 1.
  - excluded_crops.txt blob:
    102009773f843410149f8f8cadeb1b3c981252fcb8d7c53cc8d43e018df00cea.
- Unchanged by the new exclusions: control 2081 frames, validation
  940, H9 1752 with the per-storm counts and the H9 pin 267f154b...
- Frame list: scripts/n5_frame_list.py writes
  results/n5_atlantic/frame_list_atlantic.txt.
  - It contains every training frame of storms_train_n5_atlantic.txt
    after exclusions (50 storms), minus the Marco AL142020 frames whose
    650 km box overlaps a validation, H9, 2025-test or H-EP storm.
    Marco's positions come from its built manifest.
  - Dropped: exactly the design's 4 frames, 2020-08-25 09Z, 12Z, 15Z
    and 18Z, all against Laura AL132020. H-EP storms are East Pacific
    (2023, 2025) and can't share a time with Marco.
  - 4044 frames, so epochs_for gives 13 epochs of 127 batches (1651
    steps).
  - SHA-256 (also the blob):
    84e5f730edb42129bc96186b4705241759c696f5a15383c466215fc8b662cc87.
  - Tracked. .gitattributes marks it -text, so checkout keeps its LF
    bytes and run_n5.py guard 2 sees this SHA in the main checkout.
- scripts/analyze_n5.py, the two constants: ARM_TRAIN_FRAMES = 4044 and
  FRAME_LIST_SHA256 = "84e5f730...cc87". At commit 438cb23 these were
  the only two changed lines against the amendment blob 2a70bf18...
  (blob f214379f...).
- tests/test_analyze_n5.py: test_refuses_unfinished_design now passes
  the unfilled constants as overrides, since the real ones are filled.
  tests/test_n5_frame_list.py reproduces the committed frame list byte
  for byte from the main checkout's data.
- hc-adversary item 16c follow-ups (the coordinator folded them into
  part 2), in design.md section 14:
  - phase2_started.json alone is a look, even with no H9 output;
  - kept training files are read-only once their record is written,
    and the abort moves still work;
  - after a look, a record that no longer verifies asks for the exact
    bytes back, never abort-run;
  - a post-look bad record that can't be restored: decided below.
- Decided by John, 2026-10-19 17:03 (relayed by the coordinator),
  verbatim: "3. Dead-end rule: (A). Drop that seed pair as a named
  deviation, label the verdict non-confirmatory, and state the reduced
  n and its effect on power in the report. 4. Phase-2 marker PR: no.
  The launch-safety design is now closed. From here, only a concrete
  defect reopens it, not new hypothetical cases. Log further edge-case
  ideas as follow-ups."
  - Rule (A) in analyze_n5.py and run_n5.py. A pair is dropped only
    through the one committed file logs/n5_atlantic/deviations.json,
    {"dropped_pairs": [{seed, reason}]}, never inferred.
    - Both `run_n5.py evaluate --deviations` and `analyze_n5.py
      --deviations` read it. It must be committed (evaluate: identical
      to HEAD's blob; analysis: clean), and its path is in guard 5e's
      allowed set.
    - It may name only a seed with a run whose training record no
      longer verifies.
    - evaluate leaves both runs of a named seed out of its checks and
      its H9 steps and evaluates the rest. So rule (A) completes even
      when the bad record appears part-way through phase 2
      (hc-adversary item 17).
  - Both arms of each named seed are left out and the tests run on the
    remaining n. report.txt names the deviation and labels the verdict
    NON-CONFIRMATORY and gives the deviations file's blob SHA-256.
    primary_tests.csv gets n_pairs, confirmatory and dropped_seeds.
  - Power at the SESOI is stated for each design section-5 sd_diff
    row. The n = 10 column is section 5's own table, quoted as
    constants. The reduced n is simulated with power.py's method
    (two-sided exact sign-flip, alpha 0.025, normal model, seed
    20261018) at 20000 reps, SE at most 0.0036.
    - One dropped pair (n = 9): overall MAE 0.9 kt goes 1.00 / 0.89 /
      0.71 -> 1.00 / 0.83 / 0.64; >96 bias 3.0 kt goes 0.95 / 0.83 /
      0.71 -> 0.91 / 0.78 / 0.64.
- Launch blobs, SHA-256 of the committed bytes [blob id]:
  - scripts/analyze_n5.py:
    0130641583cf09c8abb5791b9ea9467fd6410b953bc8a57c3a2ae68fb3bb1288
    [679e2169]. That is the two constants plus rule (A) with item 17's
    fixes.
  - reports/v2_survey/n5/design.md:
    334e51f01c841d47b12beed90ede0a966151dddc5ec9e9f29cc66daa34f4a271
    [49a5d80e]. Section 14 has the 16c wording, rule (A) (with item
    17's fixes) and John's closing note; it is still a pure append
    against e8e4fc1.
  - tests/test_analyze_n5.py:
    fa1c079ff7b97ce26f1d0c9c206cc16030acf9e2234348a787fdaffdbf16e8fb
    [4d531f39].

2026-10-19, 19:55. N5 Atlantic-only arm: phase 1 (training) done;
training records committed. No H9 evaluation.
- Phase 1 ran 19:09:04-19:49:57 EDT in the main checkout, at the launch
  commit L = 2ce5eb6 (analysis blob 01306415...), after John's yes. EXIT 0.
- 20 training records, logs/n5_atlantic/runs/<arm>_seed<s>.train.json,
  committed here byte-identical to the main checkout's files.
  - Every record has commit 2ce5eb6, analysis blob 01306415..., and
    protected_before == protected_after.
  - The coordinator verified that all 170 hashed kept files match and
    are read-only.
- Selected epochs, from the records (validation only):
  - control, matched odd grid, selected (train.py's native) by seed
    0-9: 25 (24), 23 (24), 21 (18), 15 (15), 23 (24), 23 (23),
    25 (25), 21 (24), 17 (22), 13 (13). Native is the re-validated
    argmin, which equals train.py's last 'saved'; the tie rule never
    fired.
  - atlantic, native, by seed 0-9: 13, 5, 13, 13, 9, 12, 8, 13, 9, 11.
- No tie-rule firing, no off-grid primary, no odd-grid near ties, no
  aborted attempts. No H9 output and no phase2_started.json.
- Phase 2 not run; awaits John's separate yes.

2026-10-19, 20:16. N5 Atlantic-only arm: BUDGET-LIMITED known before the
H9 look; John's yes to phase 2. No H9 evaluation.
- Known before the H9 look, from the committed training records
  (validation only): the pre-registered BUDGET-LIMITED qualifier fires
  in both arms.
  - The rule (analyze_n5.py): a run is flagged when its selected epoch
    > 0.85 x its epochs; UNDERTRAIN_RUNS = 5 flagged runs in an arm
    fire the qualifier.
  - Atlantic: 5/10, seeds 0, 2, 3, 5 and 7 (selected epoch >= 12 of
    13).
  - Control: 5/10, seeds 0, 1, 4, 5 and 6 (selected epoch >= 22 of
    25).
  - So the joint verdict will carry the BUDGET-LIMITED qualifier
    whatever H9 shows. Identified by hc-adversary item 18.
- John's decision, verbatim (relayed by the coordinator; the relay did
  not give the time of his message, which came before this entry):
  "Phase 2: yes. I'm proceeding knowing both arms are flagged
  BUDGET-LIMITED (5/10 each); record that in the decisions entry as
  known before the look."
- Follow-up, per John: longer training for both arms, as a future,
  separately pre-registered experiment. It is not a rerun of N5.
- No H9 evaluation had run when this entry was written.

2026-10-19, 20:46. N5 Atlantic-only arm: results entry.
- Commits: launch L = 2ce5eb6; training records e06d6c5 (PR #14);
  BUDGET-LIMITED note eb482a3 (PR #15). Phase 2 (H9 evaluation) ran
  20:22:47-20:26:01 EDT in the main checkout; the pre-registered
  analysis ran at 20:26 (analysis blob 01306415..., at commit 2ce5eb6).
- report.txt, quoted exactly (PRIMARIES, JOINT VERDICT, SENSITIVITY):
```
PRIMARIES (Bonferroni, alpha 0.025 each, two-sided exact sign-flip; control re-selected on odd epochs):
  overall MAE: control 9.571, atlantic 10.102, d +0.531 (sd 1.120), 97.5% CI [-0.413, +1.479], p 0.17383, SESOI 0.9 -> NULL (inconclusive; effects of 0.9 kt are not excluded)   [d = MAE_atl - MAE_ctl, d < 0 better]
  >96 bias: control -6.905, atlantic -7.853, d -0.947 (sd 4.792), 97.5% CI [-5.515, +3.633], p 0.53125, SESOI 3.0 -> NULL (inconclusive; effects of 3.0 kt are not excluded)   [both negative: d = bias_atl - bias_ctl, d > 0 better]
JOINT VERDICT: NO SIGNIFICANT DIFFERENCE on either primary | BUDGET-LIMITED: the under-training flag fired in >= 5/10 runs of an arm; a 'worse' result is read as 'worse at this budget', not as harm from the added storms; not rerun or rescued
SENSITIVITY (pre-registered; control at its native 25-epoch selection; does not change the verdict):
  overall MAE: control 9.288, atlantic 10.102, d +0.814 (sd 0.891), 97.5% CI [+0.095, +1.587], p 0.01367, SESOI 0.9 -> WORSE   [d = MAE_atl - MAE_ctl, d < 0 better]
  >96 bias: control -6.758, atlantic -7.853, d -1.095 (sd 4.836), 97.5% CI [-5.587, +3.256], p 0.47852, SESOI 3.0 -> NULL (inconclusive; effects of 3.0 kt are not excluded)   [both negative: d = bias_atl - bias_ctl, d > 0 better]
```
- SHA-256 of every analysis output, results/n5_atlantic/analysis/:
  - ensemble.csv: c687dbac9fbf6fa82165f03ed31ae51128f3402be11e1824391f5c558eab54eb
  - paired_differences.csv: 517281c7d861ae286a7660c8c1a71267d5b8341bb13be5129de1e1a0b0fab792
  - per_arm.csv: 27fb1ae54c60b5ce409b507db03ec9387e5149072b2eccfd690af4cb7fbd80c7
  - per_run.csv: c8d77dab23b6facd092a0ea2c3cdc476abcb3db34f0eb609a683bb6330a97555
  - per_storm.csv: bf87dfb7bc57f90e83ebc58ba32a7bc51cc33ffcb7ce6dc362db2caeff45c822
  - primary_tests.csv: 2bb71c0055c9ec7fc30ae5fe2608c10f7bd144f0a573d2d82c10517376818ea4
  - report.txt: fde430de92bab775977dfc84bef9eeb7a46907ae7abfb06cb2907db2d1722bb1
  - sensitivity.csv: 5bd4fcbf32bde9e154877d73df2fe3d3279f691b45e3ed6a9a6d9e34fd5188df
  - val_trace.csv: 069e5345576cdf6fbafe7f99007448a4a790feb82941739e1ba3c2ee0ef36581
- Committed with this entry:
  - the 9 analysis outputs and the 30 H9 CSVs;
  - the 20 manifests, the 20 training logs, the 30 eval logs and
    phase2_started.json;
  - reports/decisions/PR10-PR14.md and N5-result.md.
  All are byte-identical to the main checkout, and all are stored
  -text (.gitattributes): train.py and evaluate.py wrote CRLF, the
  driver wrote LF, and the recorded SHA-256s (manifests, the outputs
  above) must verify on any checkout, core.autocrlf or not. The
  manifest pattern (runs/*_seed[0-9].json) leaves the 20 training
  records of PR #14 as they were.
- hc-adversary item 19: OK.
  - An independent recompute reproduces every number in report.txt.
  - No post-hoc checkpoint choice: every evaluated checkpoint is the
    one in its training record.
  - The H9 frame set is the pinned 1752 frames.
  - The 2025 test set is untouched.
- Reading notes from item 19 (reports/decisions/N5-result.md):
  - (b) The null is inconclusive, not "no effect". The overall-MAE CI
    allows the atlantic arm to be up to 1.48 kt worse but at most
    0.41 kt better. Both analyses put the point estimate on the
    "worse" side. The sensitivity (atlantic worse by 0.81 kt, worse in
    8 of 10 seeds) is reported beside the verdict and never replaces
    it. Under BUDGET-LIMITED it reads "worse at this budget".
  - (c) The odd-epoch pick costs the control 0.28 kt of H9 MAE on
    average (9.571 vs 9.288 at its native epoch). The largest case is
    seed 4: 11.04 at epoch 23 vs 9.37 at epoch 24. So at this budget
    H9 MAE can swing by as much as 1.7 kt between adjacent epochs
    (seed 4). That is the pre-registered matched grid working as
    designed.
  - (d) In the original-three split, ">96 kt" is Ian alone: Nicole
    and Claudette have no frames above 96 kt.
- John's reading, verbatim: "the CI rules out an overall-MAE gain
  larger than about 0.4 kt at this budget, and the result does not
  show harm."
  [Coordinator: John's reading refers to the pre-registered verdict.
  The primaries show no significant difference, though the overall-MAE
  CI does not exclude up to 1.48 kt worse; the sensitivity's WORSE is,
  under BUDGET-LIMITED, read as 'worse at this budget', not as harm
  from the added storms.]
- John's priority note, verbatim: "it lowers the priority of v2 lever
  2 (remaining Atlantic storms) and raises F-31 (the budget study),
  which should also cover the frozen ensemble's budget."
- Odd-epoch control checkpoints (130), hashes recorded before deletion.
  - Per John (2026-10-19): the 10 primary-evaluated control checkpoints
    are KEPT, together with the 20 final models seed<s>.pt, because the
    primary analysis evaluated odd-epoch files. John's earlier "keep the
    20 final models" had missed that. The other 120 are DELETED after
    this PR merges.
  - John (21:05 EDT), verbatim: "keep the 10 primary-evaluated control
    checkpoints (seeds 0–9 at the epochs listed) plus the 20 final
    models; delete the other 120 after PR #16 merges. My earlier 'keep
    the 20 final models' missed that the primary used odd-epoch files;
    the results entry should say the 10 were kept for that reason and
    list the hashes of all 130, marking which were deleted."
  - The 10 kept files, from each control training record's
    selected_checkpoint, by seed 0-9: epoch_25, 23, 21, 15, 23, 23, 25, 21, 17, 13.
  - SHA-256 of models/n5_atlantic/control/seed<s>_epochs/epoch_XX.pt;
    all 130 equal their training record's files_sha256.
  seed 0:
    epoch_01 c7fd052d88bf1ee5176adc9e7aa939052fe198234207b4fb8392d9e35a6f830c DELETED
    epoch_03 fb0fa3953f461ce8bf8abd7842126b3740ab7bfa7f82f80c57b0962670ec647f DELETED
    epoch_05 e32dc091409374d68fce020862d43259bae8d890832ee50e267e5ec40779c9f3 DELETED
    epoch_07 1c381bf87df18ecef013f5539e4071b0d92769cd65957ed58fd370c323d2e8c6 DELETED
    epoch_09 1d388d02be34ed841d20313c6279189868c8d9b1eef5b650e05e02c53dac2667 DELETED
    epoch_11 eb0fd6bd46b9e4dcdebf6ca67a21178dd440837328b744bd41fac409c497a4c2 DELETED
    epoch_13 0926caf90a99b8392efd14a7d9bad9c6adfcbae41bdb0a314e72751396196709 DELETED
    epoch_15 6aad221a859df33d889a988bfb323c9425dea1e48a9cc974c7ba663271218547 DELETED
    epoch_17 bc56b3ce9813376dea63249507c93b815cffd9e6438dcb61a5fd4300f9dcf4af DELETED
    epoch_19 7ecbfff9a8c7adfd3e0c9690516c74f877e66d5040f024b5fe757dd00eb01ac2 DELETED
    epoch_21 7148e0428599d38f7d87fc91de9528f34f5996e42b9ba8e4a14b08fb7f5beaac DELETED
    epoch_23 f87ceeca61bd309f84ad76b7d9918a7f8eaf0c9c4bcd8b7b835d7ea0e3e6cac2 DELETED
    epoch_25 0b3caf50628a3fa8a0c9ac55985a42632d65b0088a9790325e06641545ee9f1f KEPT
  seed 1:
    epoch_01 718eefab495cacfdadd240761a599b8f7709c03c2312b3d7389a5fde755db063 DELETED
    epoch_03 1a76206bb9765dec7bdd043de56c2a6294a69172bdd33dbddc328424ce0b1b9e DELETED
    epoch_05 38dcfc2fd33dd1b2367e998ae2af4b4d44350af87e1c4a5f4e2032dfb29ad8c1 DELETED
    epoch_07 5b0eaf33bdfcceb16a721bfeffe925d8f6db89206f393d7b653555c51aa4ee11 DELETED
    epoch_09 3583e7b3608f933cab74beb3fcb8d62b86a9abf585d93763b9c12992779dc4db DELETED
    epoch_11 3cd156e020654294f73a25bc4ae5102ee52d62efa005d23c7620a2e4bb58e7b2 DELETED
    epoch_13 ee5334d775e7b699dca9e37608fcefafe13d61a1865f0676257160d81d83f31b DELETED
    epoch_15 3ef41ac85ee3be8ccc2d536c489199bc2e0e700f4bcc85db99265331a1d5d494 DELETED
    epoch_17 c641df181e7cf403ef91628ac22a565025e93cd88b56addbed7411a1ccd68ded DELETED
    epoch_19 323e7341e6107790d409f05ee50878eb07d22980cb830fd2666fa3be2db64355 DELETED
    epoch_21 210c05266a59e283fc524f196ad3aa53900810a4d7c8b86bad80d2d3c3f31aec DELETED
    epoch_23 9961e12542a7175caf4534c8b1cd726e7515425ff3aa1ef059806bd8f0d9c3d7 KEPT
    epoch_25 19ac1e363de9ec0639f820023c2ed9c08a59472acbd9ad898484703f767f49e0 DELETED
  seed 2:
    epoch_01 10b2c97f54c1eaa0553d11688b2eff0869294584a54e93789b38f8231ace462e DELETED
    epoch_03 8d961da4def0253d266509509b196ce4104bd5507a57a55323f35c0364838491 DELETED
    epoch_05 6ef75ad1f0fbb1600634e818f640a8074368a29107a197c458715d390b9bb090 DELETED
    epoch_07 dabd98c0bd288306445fc3df15849df810de9dc90d35525ecd2972ac906cdd75 DELETED
    epoch_09 b34b2da86bd070bb12ed474ae8774e9e68faae71b6fe196fc8ed35c4ab1f327b DELETED
    epoch_11 bc305d6fed9447b9ce95f03b4c2b2850f2dbb229f2b6b47f52d73b65fd03aae6 DELETED
    epoch_13 0fb8e37dfc0364f7c6330c43cad69c80e9ce580219622b390e042395b80bde28 DELETED
    epoch_15 d20f8cd3bf7a9d275d6c1d7c67d37b0b511581f46f2a405635cd5d67f17a1535 DELETED
    epoch_17 e4840445cb1099fa7e0c84c33d00716ce9a5aefabd156cb036cbc9e9f00a5c01 DELETED
    epoch_19 86f182d811aefc0bfde1ee163b6b22d04a505588a7d6624290e5f8fc07a22ea4 DELETED
    epoch_21 c63ad61415a795e354a5c94f47bf4fa3ed76f809a460ca271f5ceacca516a633 KEPT
    epoch_23 f440f1dd9b9ca727251daf05f21824b5cd52c7baae0c51bf3f8aa029bdd7ab65 DELETED
    epoch_25 3f7222a0a2725d67666266cf23c3136b47fcee2aab9b362b7ca5683d899d3da8 DELETED
  seed 3:
    epoch_01 1658803f1ca48dd589e38a2a850ac008c4104e62b38551ae14fe8bbc49696a52 DELETED
    epoch_03 dc5646067a1a4fc9bb8fb62b8d67b43fff9a874407cc1a921de123917c52fe8b DELETED
    epoch_05 3d4877b62722676606d960282a9f5c3dbc3a9c1704ff8973134a0c9ca512775a DELETED
    epoch_07 58dc5afbc81957373113493209bb4de0f09ce4c1cb3e8c95a4bd995ad64c8184 DELETED
    epoch_09 26a95baa004251c9275460e55a33a2e968f7fff0509befecb7c4f2a019e3a1cb DELETED
    epoch_11 3a38f8848701d60a7c8a169af1a03fda603a79acb10a09ea91147d3d08716dae DELETED
    epoch_13 74bcb63a242947f14e592f9d38ff555130e58d334f59a94414fd6033d204b50c DELETED
    epoch_15 c670e23377d9c3d4725dfec8f89fb5ae9dac98632237c1b3aaca65d3889c32e8 KEPT
    epoch_17 4dcc3dc5f883a80fe4734375c7842abdc160680aa56232c7232dc9f38a45fd23 DELETED
    epoch_19 15a00dcfe6030ec873447282dc3d3712c1344f0c096d9dfaffb648b9dba9d92d DELETED
    epoch_21 b2676984523dd991de26cba1a249e43d5f3b3424c886cbaa73cc6099d9f085f1 DELETED
    epoch_23 f7843592e77b51191fe2f1f733dca7c83dbbeae2604631d6a95591b5b95b32c3 DELETED
    epoch_25 3242b2bb5520c7551c5ad2167e429d1c8a089823b54309f2807c0fe1fa9c94d0 DELETED
  seed 4:
    epoch_01 10e68efccf4f3b7989075d7ad5561665519b9d66c0204cd15527278ae6de5309 DELETED
    epoch_03 ea5dadfcb8bcec7797b36dc920dfa85199acd74d8c9feecaf4b450d0192aaf45 DELETED
    epoch_05 c3e48682bccd8722640cfaa8fc3bacf21cf9291f7077168b1fcb7998e0ad171e DELETED
    epoch_07 bef7f6c7115fe617429784602d4f1e3b9fec07a635319211e386bd569c7b1db4 DELETED
    epoch_09 cd51e62e3e8c96c2bc42f8e33b38150ac2d473f32b358619a1acf1113e07346b DELETED
    epoch_11 4b9b288bbf9dd27eb0d936d6a46ade7a992a023787ef75836a14fe782e8e88d7 DELETED
    epoch_13 ff27ac272d5d80bcc5d777d150c5e2eeabf9556ff3110f401028d71b8ad79964 DELETED
    epoch_15 0c48ec440b24508b63b78797506e4198ade8ac610e28f955582cabf853b3c6ff DELETED
    epoch_17 f28bfb26f02b6563ce3d259c3f63ff47b8ff0ec26e11bdc6e94de75c22857bbc DELETED
    epoch_19 d176711c322f7e179da90cdf0fcce3a84c846ceeacfb1f97fc459f405b220ada DELETED
    epoch_21 a472430c58e9b2f6882e114508570dbf5123afd1499005232be81f1bdee7c1e0 DELETED
    epoch_23 866518e9f05906ec69b1360270beaff862d49eb255394d677e2d79f4abb8dc4d KEPT
    epoch_25 d021c5bc78184501cdbbef3cb56b7ca82eaf4a47d1bbd3a71ba63864e7ee44bb DELETED
  seed 5:
    epoch_01 905d6b3a439dc3d285669b59b1a7efb5ac78a8a893653060051a043c2cf4c6f1 DELETED
    epoch_03 748fafcbbff75bb73e33a20e8dff9e4f8efb37ddfb653bce93e356a2f6afb80d DELETED
    epoch_05 37504f5262a3bf37a8ef6f695869c47b8057d0ac166b96692b6cc128f51657d2 DELETED
    epoch_07 29a79c6e894a05cdd557fe0bae02e1894fdf9b3d0fa129ee389afd828905ef45 DELETED
    epoch_09 669550a7543592867c3b65a860453a6f3f5785bcdac4fd309f7986b6eb311cca DELETED
    epoch_11 26603b5071fcffd935bcfb82247954e2dd4537a70a75ead2e3664924012b0fd2 DELETED
    epoch_13 6de4bd4b7f17bc2e8848964022bc196209bcca59f94fee584e0e296bee329635 DELETED
    epoch_15 50a9e39642d2e55cefd2fc9dcb9378546cf648559d4308b5f312a8940fdb9169 DELETED
    epoch_17 26eef1866313bbf62941983b473d57465d5253c4f63f9c686c37c5fcd41a1ddf DELETED
    epoch_19 0350e41a13409f36fb96bd00e077d84675afd7f6e97f7a3568ecc18848d11148 DELETED
    epoch_21 a9c09e07b1be895108c78d5a8eafc5b115908a75d78342cf6221b28b81f8e508 DELETED
    epoch_23 d19d395a36a6ebfab1de22a61a4a65c14180f47e419ce10c3a5930f557219a06 KEPT
    epoch_25 8b3fd531646889f78c36fffcee79a8758b9bec8ae9820833cc69bd654660afee DELETED
  seed 6:
    epoch_01 3b1f9fc9877e792ddeebf2e520816601555a2bf79ca7f42d0edd5e0254744598 DELETED
    epoch_03 be1e9caf667b6bb90fd51424245ba5cd72445765eca32317286d63715e4da7f7 DELETED
    epoch_05 df549c97430be45568c13a7bcb993e16a26066f7bfb08aa482646a0a0559fe0e DELETED
    epoch_07 a2cde85aa55b1798a9883780f03178ac59a1ee884a60217e091fa63f0a82d800 DELETED
    epoch_09 dfa24002fb27d1055cb1ab7f6da678fd03c0aae1030adf7cda9e95629bd20bd8 DELETED
    epoch_11 b9ffa0b7f0c9b2857befe89b7cf6f7eda17aea9ca1467cffb2957cb7043b01b0 DELETED
    epoch_13 51ed069ffd409e7c7415220e09fd211cec034e44a904bcacb210d9fd96f0d087 DELETED
    epoch_15 980d013190a7a6bb9e44c653b501e59f0860b77326884b204f7490843b445849 DELETED
    epoch_17 52c45cd960bf3f9903af45ee909a25e6fa34fd72256ac4de9bf136661120a593 DELETED
    epoch_19 c2fb5520aa5557010009bca04e0de4422096d56df0579033229a5f0812220580 DELETED
    epoch_21 62ec209da1cede0ec2d2f674c88e7d708cc3a38402c2ec1194c0bcb307104b6d DELETED
    epoch_23 89f20faf8f59591b81b44ea057f8e06fd258a72b2d4fd2121de96bddf2aceffc DELETED
    epoch_25 f7d79ceada1a26f3cc190bffae1dd72e0f0ba4ccdf1562782f0938d3caba02fd KEPT
  seed 7:
    epoch_01 1b30090666dc10f0098e089e2e3c820159ebd9f81fd2520fffc4965a182929f0 DELETED
    epoch_03 e4d38c22b4d1963023082b172141c7736039766d1ce4cf86e0f14bcbf3b09029 DELETED
    epoch_05 5f92a22a3261e073e4be95fbf39479838ce4405480ade9bb5ce06730fe5941b2 DELETED
    epoch_07 df09affcd56fe4e05045577a3289510c1bc5a4d52ec8dab14254c58ff486c37e DELETED
    epoch_09 4147f8faa681a718f86607f69cbb4fcb487f76492b6a6f3ccf84de47819e3d5b DELETED
    epoch_11 9e0c3def173caa83f8b6fea4bfa3644a6581d62cbf5fd01a9d63ab7b8ab6bec3 DELETED
    epoch_13 0f3c129d29b2e09e7f0995af95760df6af1414dc5e294de7f2c89e2170bb9da3 DELETED
    epoch_15 4fea407612f34b4251fe01a592d87dbae158f1c863f7cb48780fae71c6e93a6c DELETED
    epoch_17 203c62189821f54646cca6133af6ca199023bfd1ca6416639829ced07bf70d1e DELETED
    epoch_19 9e3db36ff63775d7ba6b59efcca0104ac304360dcd6b6755ddc7b71db9c8c181 DELETED
    epoch_21 fe9ebad2f1b71611360e140cea42508ec06a229ba2b31549cf2100d5b68f942e KEPT
    epoch_23 1f162c5c3520093785c9c820b060cbc05f20f7b71bb8ae17ad9066415c7b405f DELETED
    epoch_25 7666f6666be44be5e56467cdc0334928f3d3754c4454e1538735a9b0282c3091 DELETED
  seed 8:
    epoch_01 f855e50657f6bb63f72301a87fa808f29e21cd63ba6429c90e8af0dc971287cf DELETED
    epoch_03 13de8b856936a8c07e5f42ce2d2284f9ec8de3f5744ec00b5d2384a251e50ca5 DELETED
    epoch_05 fe7de2f861670f8f53ef07b413072714645b97fabe9d774d362866c4260f7c0d DELETED
    epoch_07 345f5ab49a8ec50c6c578fb852c702dd0035fc1be0b6dfd193a8402d6854941a DELETED
    epoch_09 45002b4c5a39a6ea7cda2c2079ec6986468c3146ccf0073ed94e6e6322014537 DELETED
    epoch_11 ec0eb3db005827282f17309b5da41649d05936757de3836e18c0dededd485887 DELETED
    epoch_13 dddf1012e0d18fefcd4d79dc644bd0fd5e98c1254c315237245070563646d5ec DELETED
    epoch_15 b36475f377770a1f5dc7ce4028476f79c17f6ecf9d783e746939fc4606264cdf DELETED
    epoch_17 a21630d9956a56b9ebbd5c2413d992092aa9a6954ec2096309261c6ce93769b5 KEPT
    epoch_19 0b0770fd57c9abd187e8cc29880bb20555ff6c51d1fff0f1dd4e999e46563060 DELETED
    epoch_21 8474e122dd5775f70885c835d0414ef981666acd12f0d56cfd9d268a6eefccf5 DELETED
    epoch_23 655213b555edc7694746137b0ecb49cacfc5257836cc19d444e082af06696229 DELETED
    epoch_25 2eaf118846ac8f644d980a4fd3272bf9f3063aa18087bebb5c0596be6f3a59b8 DELETED
  seed 9:
    epoch_01 7b144178a32ae3d9b17912aff639acac904c50a2e7a1dccfb2de016fa38246c5 DELETED
    epoch_03 f569771a605928de5f5bd53e11759ca1622837deb53e8e1bb8717bf140fd0a54 DELETED
    epoch_05 6a1da845d6546ed6065bb74c01dacbd54d1887727a94a1e935e083499afb4e65 DELETED
    epoch_07 f2b78b4e980efd3d9dcf007a037c9792136a0a99cb8bfbc69a035034ee535724 DELETED
    epoch_09 e271f96ff63efbd9950787519b4d40a011f18de9729d64fcc4023014536dcbbf DELETED
    epoch_11 3578d2aab30617c2240198b856bae75f1f07bd10f94b2b4397517095ae98909d DELETED
    epoch_13 09b98dd8da249b1d4275ec1fe497cbcc7dfc966f3a86986c9831fcbeecae904c KEPT
    epoch_15 3d7fe483569edeeccce8b10538210b6ae217dc51c29064cc1937576e39164978 DELETED
    epoch_17 63fdc0826cd97dc292b1db1af0242d147468f813b2712a05e589b35e08ed2a26 DELETED
    epoch_19 18987b2cdb393767ad0ec798d1c684e3e84d52b557faed9d73aab40cc5f24e41 DELETED
    epoch_21 e083fddbe3935d47fc2f9c01e120fb2eeab3c6c388b64bac5f44c6cfae9d422a DELETED
    epoch_23 fff7c692d003afcad5dd415d337687c5961d1e080a82bd40b506ae785346c2ea DELETED
    epoch_25 c6bde3a2d4f58735c437e450f7aacabdeb6253f79775e7ff09ec1afdf86f2f5b DELETED
- Follow-ups:
  - F-31: a budget study for both arms and for the frozen ensemble.
    It is a future, separately pre-registered experiment, not a rerun
    of N5.
  - F-32: the CI grid puts each edge up to about 0.002 kt inward
    (cosmetic; no edge is near a threshold).

2026-10-19, 23:50. F-31 budget study: design approved (John). No training,
evaluation or H9 read.
- PR #18 merged as c4c4cff (parents fbe0da9 + d83ff06).
- John's decision, verbatim (relayed by the coordinator): "1. PR #18:
  yes, merge pinned to d83ff06 (design.md 475f1e44…). 2. §11 choices:
  all six as you recommend. 20 seeds; two primaries (4× vs 1× within
  each arm); validation-based picks plus the fixed H9 checkpoints, with
  no ±1-epoch H9 reads; keep only selected and H9-evaluated
  checkpoints; H9 as the evaluation set; matched-grid checkpoints at 1×
  and 2× left dropped. Record in the design that the read-at-1×/2×/4×
  equivalence depends on train.py's learning rate being independent of
  total epochs; any schedule change reopens it."
- This PR changes reports/v2_survey/f31/design.md only as follows:
  - the title line now reads approved;
  - section 11 now records the six decisions;
  - section 2 now carries John's equivalence rule, with a launch guard:
    run_f31.py refuses unless scripts/train.py's blob SHA-256 equals a
    pinned constant (at c4c4cff afceb4b19ea7d8c732c6f26b7d52e4b3b6d313d58153e2d0518887a3a6db231f,
    blob 500bbc1e);
  - section 8 and section 10 name that guard.
- Hashes, as SHA-256 of `git show <commit>:<path>` bytes [blob id]:
  - reports/v2_survey/f31/design.md: 475f1e44dda61c539ed0624d9244c1423422797f632ae8dbb5e224757dddc61e
    [9f73e1ce] (d83ff06, approved) ->
    7e1ef4c10c3866dfc2addffcf6605c69b955131dcdd6911b5d56f05f9c1f215e (this PR).
  - reports/v2_survey/f31/power.py: aaf738ab4b0deb7c30fbab0b035e58b5eaee7aeed98e38191d83954dc679246f
    [c0ee034e], unchanged.
  - reports/v2_survey/f31/power_output.txt: a2baf27c53ecf0735d12fd6be2acd7ab83ff8102920c3eee6cba06b04a6f504a
    [b7269885], unchanged.
- Next: launch PR A (scripts/analyze_f31.py and its synthetic tests),
  then the driver. Each goes to hc-adversary and John before anything
  runs.

2026-10-20, 15:12. F-31 budget study: pre-launch record (design §10). No training,
evaluation or H9 read by this entry.
- Merged so far: design PR #18 (c4c4cff) and docs PR #19 (d03ccba,
  John's §11 decisions and the train.py-pin rule); analysis PR #20
  (482a501); driver PR #25 (2fc2863); look-evidence tightening PR #26
  (9fd119e, merged on coordinator review as tightening only, after
  hc-adversary items 27 and a real-git check of the unmerged-results
  case).
- Launch blobs, SHA-256 of the committed bytes at 9fd119e [blob id]:
  - reports/v2_survey/f31/design.md
    7e1ef4c10c3866dfc2addffcf6605c69b955131dcdd6911b5d56f05f9c1f215e [9cc0b05c]
  - scripts/analyze_f31.py
    feee9bafabee141dd393a86c3c469b4cafd17c160c8f97076cab55da0f8016e1 [fd7d4c93]
  - scripts/run_f31.py
    85db105ae40cab81431385de56feb8e7c3bcea204b29b7cb076cc2c90fdec435 [3e80bf03]
  - scripts/run_n5.py (pinned by run_f31)
    e8f13e06b80c29101ef1a4efaeac7aaa17e5922f0a8bebdc26a19f270d6d3ed7 [2ebd8cea]
  - scripts/analyze_n5.py (pinned by analyze_f31)
    0130641583cf09c8abb5791b9ea9467fd6410b953bc8a57c3a2ae68fb3bb1288 [679e2169]
  - scripts/train.py (pinned, John's 2026-10-19 23:50 rule)
    afceb4b19ea7d8c732c6f26b7d52e4b3b6d313d58153e2d0518887a3a6db231f [500bbc1e]
  - scripts/evaluate.py
    d2ea083178b575cd0e30902ab4c4dea59411fdcedbbcc8f64476a2fdcd68390d [af5332d6]
  - results/n5_atlantic/frame_list_atlantic.txt
    84e5f730edb42129bc96186b4705241759c696f5a15383c466215fc8b662cc87 [9e3275c4]
  - reports/v2_survey/f31/runbook.md
    e3ead7c0f04dfb1f8cfce0c9dd83f91c8ffaafe82d7977354f14674e30f27bf8 [20e0c019]
- The launch commit L is the merge commit of this entry's PR. The
  driver prints HEAD and the analysis blob at launch; the coordinator
  checks them against this entry.

2026-10-20, 21:02. F-31 pre-look record (design §3), from the 40 phase-1
records, before any H9 look. No training, evaluation or H9 read by this
entry.
- Phase 1: `run_f31.py launch` at L = fc71b989feaaf1c450d2da0df5b59d2f405b46a6
  finished 20:58:53 EDT, exit 0, 40/40 trained. All 40 records name
  commit L. analyze_f31.check_training passes on all 40 (0 problems).
  `run_f31.py looked`: none.
- Flags: selected epoch > 0.85 x E_k (analyze_f31.flagged). Qualifier
  BUDGET-LIMITED at (arm, budget) when >= 10/20 native picks are flagged.
  - control: 1x (E 25) 12/20 BUDGET-LIMITED; 2x (E 50) 8/20;
    4x (E 99) 5/20.
  - atlantic: 1x (E 13) 8/20; 2x (E 26) 13/20 BUDGET-LIMITED;
    4x (E 52) 12/20 BUDGET-LIMITED.
  - Also reported (not qualifiers): control matched_4x 4/20, control
    smoothed_4x 7/20, atlantic smoothed_4x 12/20.
- Reading (design §3): the atlantic arm is still budget-limited at 4x
  (6600 steps). F-31 reports that and does not extend.
- Zero seeds (native 4x = native 1x): control 0, atlantic 0, so m = 20
  in both arms and significance is reachable.
- Record SHA-256s (`sha256sum` of the committed bytes, sorted by path);
  SHA-256 of this 40-line listing, LF-terminated:
  0a3085dda4acb729f20f21e7d4a5abb94e4e832697a3d08cb66915fabfc3621d
  - 08d7a68b42f0af9e1b09ad1d4f70895bfd7ee06de40d4909933ca909c872cb76  logs/f31/runs/atlantic_seed10.train.json
  - a3e8b1543c27f6a078fa85528d0765d4b44eeef34ce9363a1870f9b3d9393526  logs/f31/runs/atlantic_seed11.train.json
  - 9a6dd8bb85da651dd3014bec52843dfb057e78956a634cd426e911a44a7fe26b  logs/f31/runs/atlantic_seed12.train.json
  - c3de574d20496e344796d9ea02819a735d08ba58734e6e8877a02cfb5150ca53  logs/f31/runs/atlantic_seed13.train.json
  - ee21225cff4b752f3706340edca958ff56a13c529b0fbe6a834ade3775ad6858  logs/f31/runs/atlantic_seed14.train.json
  - a90edcc7984fa2edb6b77dca700034e467cddff3a8860d46e54d41b72c5826f4  logs/f31/runs/atlantic_seed15.train.json
  - 211146e2bcd49fc00193d37469b6bf3db844b25ce9b5ac01d13063147fb26c38  logs/f31/runs/atlantic_seed16.train.json
  - 92de021f8a8e2c86431a727cc2c4e60d9387fe07bf0ceb997f5601792cd9915b  logs/f31/runs/atlantic_seed17.train.json
  - 217319cc7a7b572bac91f57ca2a0c7577eb563d2653ecc94eed6b70bebe52890  logs/f31/runs/atlantic_seed18.train.json
  - 7a01bf231e731313cfe14d34bf278b01dcf05571aa26f8cda9a05a2d2289cbf0  logs/f31/runs/atlantic_seed19.train.json
  - f0db0555e8cd66572acff10ec9745517e07c1c3479219dfd6325f79ba35078e6  logs/f31/runs/atlantic_seed20.train.json
  - e20243642898114df2aea85335e56d211026e2ece60a3b4f74ef1e2f5968b910  logs/f31/runs/atlantic_seed21.train.json
  - 21d780d8664c57f7a151fd67fb862c090cb0407a6fb35d9f5e0308c0cbd4cdba  logs/f31/runs/atlantic_seed22.train.json
  - 4a2d07fb63155c057853ea39062129c1a6339dea4430f1841192be61f7567ad9  logs/f31/runs/atlantic_seed23.train.json
  - 80c4a97d1754f000620bf378664d1c656bcd5ca0f225bb198f2c4869badc98ac  logs/f31/runs/atlantic_seed24.train.json
  - 65b7c59af8bfce8df54864d48178c5e94413ff05e4fb5a8a709fd3decb681c6a  logs/f31/runs/atlantic_seed25.train.json
  - c4d3bc168c9996fb81506c38f79d490c36b905dd43b711e2bd35ed5ad5b4e220  logs/f31/runs/atlantic_seed26.train.json
  - db32aa84068d8c3721505f0d7c3d0ce887450a6c3b49ddca16c52db24c305749  logs/f31/runs/atlantic_seed27.train.json
  - 169f26760a900774745dce4a0863598cd6af1b9d62337368b216e21dbeb507e5  logs/f31/runs/atlantic_seed28.train.json
  - ce5c7dab6836e80d2efbf849167b94472f8d99c776c6367fbaeac27fc79ce134  logs/f31/runs/atlantic_seed29.train.json
  - a61609fb3897cf7af0fc39c4e6f442d55fe4765fbeb1bde67781406d726fad24  logs/f31/runs/control_seed10.train.json
  - 2df05e948d18e2cc0302b65588dc51f9764e4610a76442cd83b735f6595a9329  logs/f31/runs/control_seed11.train.json
  - 7c97b7ae23efa886052f2d6a69b8ec8ef690484d5f106cbbcb09c520314db88b  logs/f31/runs/control_seed12.train.json
  - 934ad92587886589e6ef80c9c8c389b86e1cbfc17a4497efe0d194dfa96187e1  logs/f31/runs/control_seed13.train.json
  - de96828f6aed6e98eec598e9c506ae2278ce339f71a634ccee0329cdf6d10a46  logs/f31/runs/control_seed14.train.json
  - bab33b8261775e4347f7bfd7706e20b365c4841de509e209e5819c8bc2013922  logs/f31/runs/control_seed15.train.json
  - e966c324de4ee0e2fac1d0e57dc4cd0ab1c0bc40bb129b17d7a3678819aad127  logs/f31/runs/control_seed16.train.json
  - dbd640285caae2d002010b3e5d85ae5b5475fed78eab719494ecaab8f59f8fd7  logs/f31/runs/control_seed17.train.json
  - b672b09b8c9bec916a4160c97f7a619acb02771dc337502e2b1a367904339beb  logs/f31/runs/control_seed18.train.json
  - 6bffe2e0e97368853942212726c35752ddae6a896427ff49340947a2c3ca0b32  logs/f31/runs/control_seed19.train.json
  - 3b42bf329a3cfa70e4de68706cf3a0b1b973abfb2317d87b9a3ccc927cdc8065  logs/f31/runs/control_seed20.train.json
  - d6096e3aeb1e1b7f84cf52f829f3ff526b273aef2c1ea07f7d06873df1579744  logs/f31/runs/control_seed21.train.json
  - 6629baeefacb007caccfb2a0f204afc6654dbab222de9fb8fbff19f00dc9315a  logs/f31/runs/control_seed22.train.json
  - 7bdc07432f1c7f7620bee34418a9b6bf5973f244163a02f1a9b80fbe06968d0c  logs/f31/runs/control_seed23.train.json
  - 2a3a135344ce409e5303f59f1c35bbf0509920c8b0c438adcdbd38b657199f80  logs/f31/runs/control_seed24.train.json
  - e58a48c3e7c98100d4252a1a94649d078fcedb8fadf599e2c184dcfcb18dc4d6  logs/f31/runs/control_seed25.train.json
  - f61fdaff511cf7f0812fc039957c5943faa1523bf258dafc0144497c7f03bca2  logs/f31/runs/control_seed26.train.json
  - f91a66b9e497ec54d9ac7475fb5a81566c8d7c2d73797e1c695e1b015363189b  logs/f31/runs/control_seed27.train.json
  - ddcce64a17da44f1f8ba9d2c7cbdf46a7598d39c12d1b948c31368202f2d150a  logs/f31/runs/control_seed28.train.json
  - 18d285de97dfadd7a23159c8ad070874a061eb04e4e6af1f2586105950bac2d7  logs/f31/runs/control_seed29.train.json
- Correction to the 2026-10-20 pre-launch entry: run_f31.py printed
  HEAD (fc71b98…) and the train.py pin (afceb4b19ea7…), not the
  analysis blob. The coordinator verified the analysis blob
  independently at launch: the working analyze_f31.py = the HEAD blob =
  feee9baf…, as in that entry, and launch guard 1 requires it to equal
  HEAD. Follow-up F-38: a future driver prints it.
- Next: John's separate yes to phase 2 (runbook step 4).

2026-10-21, 14:25. F-31 results (pre-registered analysis, unedited). Nothing was
re-run for this entry; it records the outputs already on disk.
- Phase 2: `run_f31.py evaluate` at H = 079f63167496f93ca4167d08bc0738d7b0fe33e3
  (the PR #28 merge), 22:00:34-22:18:04 EDT 2026-10-20, exit 0: 40
  manifests, 165 H9 evaluations, logs/f31/phase2_started.json written.
- Analysis: `analyze_f31.py --manifest-dir logs/f31/runs --out-dir
  results/f31/analysis`, no deviations file, 22:18:19-22:23:38 EDT, exit
  0. results/f31/analysis/report.txt equals the printed output.
- The PRIMARIES, BUDGET-LIMITED, SENSITIVITY and SECONDARY sections of
  report.txt, verbatim:

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
```

- John's note, recorded before the look (his phase-2 yes was read at
  21:49 EDT, before the first H9 evaluation at 22:00): "control is not
  BUDGET-LIMITED at 2× or 4×, while atlantic still is at 4× (12/20), so
  the S1/S2 comparison at 4× remains asymmetric in training."
- Reading (design §3 line 142, §7 line 304): as printed, P2 (4x) and
  the P2 sensitivities (2x, smoothed 4x) carry BUDGET-LIMITED and are
  read "at this budget". P1 carries no qualifier (control 4x 5/20); its
  BETTER is read per §7: the frozen recipe was budget-limited at 1640
  steps, on H9, for this code version (control 1x 12/20). S1/S2 are
  secondaries and carry no qualifier; their training asymmetry is John's
  pre-look note above. The atlantic arm was still budget-limited at 6600
  steps; F-31 does not extend.
- Per-storm and ensemble figures: reports/decisions/F31-result.md and
  results/f31/analysis.
- Committed files (all except this entry), SHA-256 of the committed bytes,
  sorted by path; SHA-256 of this 422-line listing, LF-terminated:
  dad9f871c48cf04dd8f79f70b654ab58da2d582e0ec978c4e77895099612d893
  - 15f0379a56967c94b2340f32122ece3610364b906f38c4254b455e9f49f930f4  logs/f31/atlantic_seed10.log
  - e0b0a8b75719f43ac1fdc97dcc57100998ecc9b0b346f3281354cf72a3971f54  logs/f31/atlantic_seed10_e12_eval_h9.log
  - 4151d887dd5a000144073a6640be1d637f218f56852ca660386af4c04d090a90  logs/f31/atlantic_seed10_e25_eval_h9.log
  - 9c337acca8bbb3fb8d641453faa6f20634ee631151c6ca402be59c4526a11ae0  logs/f31/atlantic_seed10_e30_eval_h9.log
  - cf93a74adb0663ea92f3049693075da45ebe081145b2a208ac730721dd2f6154  logs/f31/atlantic_seed10_e47_eval_h9.log
  - 0000af89e399a88779a1d9c4b31f3f28e6ad2c95627744bc0ecf8d60930831ed  logs/f31/atlantic_seed11.log
  - 5004f05d1388dfbc39eec2d9c518ca68c011514d4676a1a6e27456e7b2a2c410  logs/f31/atlantic_seed11_e10_eval_h9.log
  - ad0166ad45a1eee49ddd4f715cf1649d6143b0343d7efa20d8aa664c80acd657  logs/f31/atlantic_seed11_e28_eval_h9.log
  - 3e230f7b560fd27814b5b887f4a83ccd185ee9bc7278d50a2999ee8be8eb946a  logs/f31/atlantic_seed11_e43_eval_h9.log
  - 7b1cf53ca78a491f79398a3a16b45456f7e348457be856f53952e5563cc04f6a  logs/f31/atlantic_seed12.log
  - 063156f2de42a2912f4f1b65d22eb8801e248b0a4eeb2fd299bee732e56d4764  logs/f31/atlantic_seed12_e13_eval_h9.log
  - 3433a76714b5ec2c5c45d5b1e1ccb671a955c7f77e9205d096f9df71ff086aa0  logs/f31/atlantic_seed12_e24_eval_h9.log
  - e0e208fc94629f62210bce9b0b9bd62ab9bb9ce7b4020403508c8fd57f87052f  logs/f31/atlantic_seed12_e38_eval_h9.log
  - 183b646f42d163501983df9cb1eb3d74c457fc70a279d1e6e2420bc969a4ab03  logs/f31/atlantic_seed12_e44_eval_h9.log
  - b047f7ab6579d3e6bea938b4b68893e30e9d3ed9024d221dd9c12b009ce14ee6  logs/f31/atlantic_seed13.log
  - 144fca6e1272c8d05e9e753bb945a57158c70b9e42b21e3e6b5e960908233689  logs/f31/atlantic_seed13_e13_eval_h9.log
  - c7aeead6a47cfc6f3feed2482317cc8ce9e4460362a473c7b7412a31c2c8fb1c  logs/f31/atlantic_seed13_e18_eval_h9.log
  - c5d8f1fe4231125a020f5d957d994531e6a4a3cd82ae78798dabe7cb31db7693  logs/f31/atlantic_seed13_e46_eval_h9.log
  - e60e80c618821de8ef4eb5742bb6473878ed254ff9fe7f480339febba1d41f46  logs/f31/atlantic_seed13_e48_eval_h9.log
  - e8eadee3c428034cef832398fb232ba13b8251f57ebbe303a38013948fc6e6bf  logs/f31/atlantic_seed14.log
  - d24021fd81923ac9a6d7a311f384a97ca35929fe4eda698f007e11a00b6b4257  logs/f31/atlantic_seed14_e10_eval_h9.log
  - 65058d03d43e3ff159d056b41c71e524735257220116c7220af534e10c9dc366  logs/f31/atlantic_seed14_e24_eval_h9.log
  - f3835fd03dda6f0d3a697c0c7d929f8e3dd9efec68d10c582048a811dbfc7dd9  logs/f31/atlantic_seed14_e31_eval_h9.log
  - 77254fdfcf695e81d59d65314cb0bdd53bf7fcdb7970cd82a5cff697134c1775  logs/f31/atlantic_seed14_e45_eval_h9.log
  - f6d896b64b046e41cf6f595bc59a7184d4ddb04fb3b17698ba46c7e808945031  logs/f31/atlantic_seed15.log
  - bce1b2b846cc504317ce052a2e5ea877f188231b93ea54ad8ead56253c55b096  logs/f31/atlantic_seed15_e12_eval_h9.log
  - 0d9e6019c1b6919cb3c566297d0c5c3f7f53e066322de30fe933da348f31bf74  logs/f31/atlantic_seed15_e24_eval_h9.log
  - 429488028eb7cd605aec3680901b154570d13ca441f1ddaf58a3e95288c63d18  logs/f31/atlantic_seed15_e37_eval_h9.log
  - 873f0b16cf8120f536ad233fb36cbf5ee36d987581d8d5bbf87eee17be3d9159  logs/f31/atlantic_seed15_e47_eval_h9.log
  - 7d18076f55025a085a3d6d3861f15800dcfd264d2914776f85d8a911171db6e0  logs/f31/atlantic_seed16.log
  - a9da64acdd461513d1584fc05dbc7650278621e4798db273ba7f34583d8f50b9  logs/f31/atlantic_seed16_e13_eval_h9.log
  - 34b389fb28f937f2752b472d0384ac2794859836aff277c560ea1eb1af0d1198  logs/f31/atlantic_seed16_e20_eval_h9.log
  - 9ea5d9734469cd1b3310f30797fa8f3cbe9736d475c9c01b48a7d6afdf30dd14  logs/f31/atlantic_seed16_e41_eval_h9.log
  - d2e30622a9fc59f250638500d35100f1f348bebe3901a29c13b4255833d1a9a4  logs/f31/atlantic_seed16_e49_eval_h9.log
  - fa566fc7ff4727ec630658f83bfc3b134c6a519c8e4371c86da63eb1610be37b  logs/f31/atlantic_seed17.log
  - b7484dca07564e13518c6c235b40146024a30a5aae12f48c43b8d527aa0e58f7  logs/f31/atlantic_seed17_e11_eval_h9.log
  - 78a7ddfcd8d862edc34bee57fb2e2f60bd0644cca3c17f5615b5717bacf723d8  logs/f31/atlantic_seed17_e23_eval_h9.log
  - 7d6f2c90d30f88dd9a6872e935a988cae6c66be972d5638316d5186a3f94c47b  logs/f31/atlantic_seed17_e37_eval_h9.log
  - f0fa99b2be00f8cd23b6868ba54c76abcde540660648ce34023bfd1339fbbc1b  logs/f31/atlantic_seed17_e39_eval_h9.log
  - 9fbbcaa0ffa8475c24bd565ed7943bfa85b3f14ee90861a049adb6074149a448  logs/f31/atlantic_seed18.log
  - d18eb8435816ed953dd89065e614f6614ebadd2d358a1b7a33a57e3d1b222721  logs/f31/atlantic_seed18_e09_eval_h9.log
  - 726e797e25eda10baf654990c83bb00c8b364687285b2d35fd36126bd1a53d65  logs/f31/atlantic_seed18_e20_eval_h9.log
  - 57f1033fd07939c3d437f141044fe55e9d7ace6e6f89e4ca102b685f91224ff7  logs/f31/atlantic_seed18_e42_eval_h9.log
  - a69cbcc73701d263ad13496feb768ccd4638d6e0a19be6208e7646791ba46fc8  logs/f31/atlantic_seed18_e51_eval_h9.log
  - 6731660e9307ce38600819a67b5368022972b31213cf7e7273ecf6465558efcc  logs/f31/atlantic_seed19.log
  - b25fe8cfbcf2d3a23c216ce3442af6a6d3a48d888e3c087ddc9b5ea9bf583475  logs/f31/atlantic_seed19_e09_eval_h9.log
  - 834b5c1bd9fff721a2250112c261a026585da6e2a5d6b4504e040aba608bcf81  logs/f31/atlantic_seed19_e24_eval_h9.log
  - 56182bedbbc155ee2af05358ae3afb75bd57106e128ba092d4c8ea39d01f4508  logs/f31/atlantic_seed19_e46_eval_h9.log
  - 97649d8d32d38450f1adae2e985f948829dad1f611c16e8a1c339f7b3908eea8  logs/f31/atlantic_seed19_e48_eval_h9.log
  - 9294536ca1b91eb11b0fb1ca2a3b5e78679aafb13b3ace424f5496f91c87360e  logs/f31/atlantic_seed20.log
  - c2b02a4ffb4a729b9156774e9c738998b7e8443d18a12883b0b4fd77cca7c461  logs/f31/atlantic_seed20_e10_eval_h9.log
  - ec2347f595055a24081980b1217b979bf1da16b146bf15881c0605bd99552ce1  logs/f31/atlantic_seed20_e22_eval_h9.log
  - 8686c956ad3326802ee32a98a5c6dc3f01006ec36d67189356a686a0e10880fd  logs/f31/atlantic_seed20_e43_eval_h9.log
  - 897c4b79041f1d3797544180ea68d80bd92b4a0d3ce602d9b15812690608ad00  logs/f31/atlantic_seed20_e52_eval_h9.log
  - a9c457080c3a89400e9517a0d33e69360da4dc04742151542e009b2351eb9825  logs/f31/atlantic_seed21.log
  - cc01328ec07488802d1a46784d5af46259405b4dd92833ae7df13c3bf6994fac  logs/f31/atlantic_seed21_e10_eval_h9.log
  - e7694a29a24dfa02f747dd139874112c60f8489fa0c09d70ab729da0a8d523a2  logs/f31/atlantic_seed21_e25_eval_h9.log
  - bd53686a77e1dd4d7f85d709f6cd6bac251b952ef1af4fd52063003bea3eb57b  logs/f31/atlantic_seed21_e49_eval_h9.log
  - b921754e9d7106a734950cd1526e4ae602dcebd6e4d0325f9b0b1debd56717ce  logs/f31/atlantic_seed21_e51_eval_h9.log
  - fe54683f2b537708911c87164426819144a3dbdb8382d9d5675b09592906394f  logs/f31/atlantic_seed22.log
  - 57f90c54002271997868f53301c47cde0ba390201144d5ea0a5ff247f6539aa8  logs/f31/atlantic_seed22_e08_eval_h9.log
  - 8238062c7c44a7be8f8ad9c9e482aad98a40338959f92b0ddd9e54d76ca1f3ae  logs/f31/atlantic_seed22_e26_eval_h9.log
  - 647a295534e120dadac1472df53f52b77358dad05612ceb22f53e4c4b905b46c  logs/f31/atlantic_seed22_e39_eval_h9.log
  - 3e9b2162e4e76bba24a45c5e636a1917af839b516e8c4371970be040670d5a82  logs/f31/atlantic_seed22_e50_eval_h9.log
  - fb3d3bf7327443548c7400ac385a4f4b79ee56e9414c4ee44644989d3384a42e  logs/f31/atlantic_seed23.log
  - 977f60b59d97f65b0a148dd64d0106aa3c5389288d310c5d8bf1779dfe3b4f58  logs/f31/atlantic_seed23_e13_eval_h9.log
  - 48d9cc8b9db787fcd604cc493d1bba5396010247d2df8937a3f7894033e92087  logs/f31/atlantic_seed23_e23_eval_h9.log
  - b3c82451a6b85206dae4aa01f6e0ac1939b951e6e4930814cd678e0533292d93  logs/f31/atlantic_seed23_e51_eval_h9.log
  - 5e8f3e2d193a7883faec74220574518691791c006f683566d67673b8053396d4  logs/f31/atlantic_seed23_e52_eval_h9.log
  - 21148481e0d791344e9f02ccec29639727994cbd5b4a885d2dcd21bd4bb98f31  logs/f31/atlantic_seed24.log
  - 36ebbc7f626b491d793a43b975fcc58dbc09c6a4a78ddea9fcca5d0f13aeb1d4  logs/f31/atlantic_seed24_e10_eval_h9.log
  - 2b92f6c9abae00a30b79a7ca0cf83352f8a07caf6efb57b5f4459cfb3192f53d  logs/f31/atlantic_seed24_e25_eval_h9.log
  - aaf0985a13ee4ab4bdcd62e73c9bf6b3bef7a840e4a54634990c5fafd166b024  logs/f31/atlantic_seed24_e50_eval_h9.log
  - 10bd65a3c5e63d96f590d06a986f5d17a8e5fbede36e8d809c0c1983ece8f345  logs/f31/atlantic_seed24_e52_eval_h9.log
  - 8a642a17ba06a17ea7f9a67886dbc7513b26f88c5d8bc189611babb1850efd37  logs/f31/atlantic_seed25.log
  - 223cfcb92647100be67049cf00da99d46c6e72f69da44b9ee2caa97290bd0524  logs/f31/atlantic_seed25_e13_eval_h9.log
  - d42d55af0f712c5c6cfdfe7ccb5f6618fe35665a356c3497ba2dee2039746ea4  logs/f31/atlantic_seed25_e25_eval_h9.log
  - 487e669d05bee78830816af3d7cd4cdad869677d67df42d98ea4eb5ff51976ac  logs/f31/atlantic_seed25_e45_eval_h9.log
  - 1f59203c68da5d63493d1f6242fd110ac4651c973ff67a523f7602fa56ba63fb  logs/f31/atlantic_seed25_e46_eval_h9.log
  - 44d92e5cbd514c1dccc08c0d524a007053e175045c6d8a7154503be43bb1f98a  logs/f31/atlantic_seed26.log
  - 0e14a456884198732fa045536356226b8a8f34b197a60ec23fd655a8d4202285  logs/f31/atlantic_seed26_e10_eval_h9.log
  - b1b671d4cc269cc32ab43304c4f15cc9fdc8b0ed178eba3ee585a97047f9fc9e  logs/f31/atlantic_seed26_e17_eval_h9.log
  - 5ce15eb131738b6181241faeed03de0c3627a76f93e76402e8c261cd09759edc  logs/f31/atlantic_seed26_e44_eval_h9.log
  - 2a5f2a7fd400f472f6ae00f63bd2997c209f53082c54d02a8b2aa3b2927e20fb  logs/f31/atlantic_seed26_e50_eval_h9.log
  - f0855de9181631b66af71fe0b8896b1e4bdb52f5634489d67fd074d0531777a7  logs/f31/atlantic_seed27.log
  - a14b9cf84c51d5df4bf3977ec5115c933a29a7a53970a84a347eca9e6cbbde95  logs/f31/atlantic_seed27_e12_eval_h9.log
  - 04be440afb4f43c7afb91699b066f3f0ca13d0435f4fc5ee5810e8ea8f2ab12a  logs/f31/atlantic_seed27_e24_eval_h9.log
  - a02fd573785666e3d3aa8ddab939260f27dcc57dcdc28bbf835a53fed2e577eb  logs/f31/atlantic_seed27_e28_eval_h9.log
  - 6b4e5a89500f8e44fe941f578ef951cb76131b0d78690976072c68f6ffdfd2e7  logs/f31/atlantic_seed27_e30_eval_h9.log
  - b778eb88835f33b88658a2063943c6c88233d88d63d07a2b0ddbdae8fa1200f3  logs/f31/atlantic_seed28.log
  - f189a372f520fbb88a8894ee200130899f1819b35d216e42fbe2cba3cf45f96d  logs/f31/atlantic_seed28_e09_eval_h9.log
  - 21149a3fa8b470b4103efc6c9abfd20d33b3ea8639e10ea4be0a66f9cd61dfb2  logs/f31/atlantic_seed28_e18_eval_h9.log
  - 7d7eba2b1af3c35310d1090ebb47fde2f47b58f71da803c4b2ddc18a9a0da289  logs/f31/atlantic_seed28_e49_eval_h9.log
  - 33be8eec9772738b99de616d55890e557fac1808f37d19069fefdc97228f1abc  logs/f31/atlantic_seed28_e51_eval_h9.log
  - ab1956f7ce3a07ac278dd10b3b6a5da2de9f5b517d6978323d09f54844be1f97  logs/f31/atlantic_seed29.log
  - 7c39df337730265098df0e24a07b1d95fccee174393b856b1e9d0442c7939ccb  logs/f31/atlantic_seed29_e10_eval_h9.log
  - ab631b8c8a7f39426b8c0d2703be65403456d9a7d3927a37ce507b944d542fad  logs/f31/atlantic_seed29_e25_eval_h9.log
  - 39485f6142be6715ef756905842dad3955e2a18f50f2cb5ea837cc479a9c9923  logs/f31/atlantic_seed29_e46_eval_h9.log
  - 52a450b971cf1061d320147b20f2fb9770803534bcb762b598aed87cb9836636  logs/f31/atlantic_seed29_e51_eval_h9.log
  - 421bc2cdd4dc7b22af1c02856488c0ed09d5690ea2fb40d7cc7440d1167a6d98  logs/f31/control_seed10.log
  - 3f97df171911b9646851ab5696a2536308be368ba013c611dfa4b7da6f44c1b9  logs/f31/control_seed10_e15_eval_h9.log
  - c9b057e314230174544143c3984dbac14993638ef54324b299a70fe497c6a911  logs/f31/control_seed10_e45_eval_h9.log
  - 7620ec9d5ef49fa10a8d160f4b8fbfc6dd322ed20f5032a375c279ba152c16a6  logs/f31/control_seed10_e46_eval_h9.log
  - 32cc9768d19354e4035f3097595ef92bcfae8bb1eb55509818588aff98a38c70  logs/f31/control_seed10_e47_eval_h9.log
  - 92a495d11a9a0a975c5570eeef46abd47f8eacb946b6ea950919f6b44041cac4  logs/f31/control_seed10_e76_eval_h9.log
  - a5a38d49c35916e075e71977ef847bde250498c539d49678d265f14402d6b882  logs/f31/control_seed11.log
  - c9e7bbd1a619881281c43706ddf5c9fa11d2618c21cacf6c46ed9ec682f0fc40  logs/f31/control_seed11_e13_eval_h9.log
  - 7fd6329cde16d9c8b2c909566efa2922b783de98faa947c135d59927b09dfc53  logs/f31/control_seed11_e27_eval_h9.log
  - 63035640aaff733f6bee914f1885584bbf3d2b974499d876f81ab9eb853054ab  logs/f31/control_seed11_e71_eval_h9.log
  - 307fc6fd7816159253764bce4e21654d0c9248d525d8b4b61bea55e3ba2c25f6  logs/f31/control_seed11_e91_eval_h9.log
  - daec52d76a83b075fcd3200fd5dcb965a8d29d62e12ee4da890f94def48590c7  logs/f31/control_seed12.log
  - b3a130ea12cefbb992211be63bc2bef77f33d48cbb34ad0c2269b031ac51bb0b  logs/f31/control_seed12_e23_eval_h9.log
  - a13034607e21fd8996f31098f2d71bd88eb1c120628254836a7b840ac6651c21  logs/f31/control_seed12_e35_eval_h9.log
  - 4ce7b2526f745fe3365ea802702b6b78daa8bebccba174c5dbe2b34a78954a3c  logs/f31/control_seed12_e63_eval_h9.log
  - b830ce64114b1f4566c56904ea952c187192100ee9ae7c51406cc7ece39268ff  logs/f31/control_seed12_e86_eval_h9.log
  - 156bec18aadeb6d3018a22463d91dbc700c16c6cf6be0afca3015e70b0544edc  logs/f31/control_seed12_e88_eval_h9.log
  - 3c77dc7b7ebaa8fc8724d3477df81c4a0efa82a2279933617e9efdae68549e5e  logs/f31/control_seed13.log
  - bb7dd06bbea792e8c4822edc6aa41e074351c4c7d1db0939b496afe56e3c23fc  logs/f31/control_seed13_e22_eval_h9.log
  - 61074cbec6c00dc60b46e6d0dd0dc1eb89f3c33357494b3d52dd50c6080b41d8  logs/f31/control_seed13_e35_eval_h9.log
  - ee35dbaf4c59963e337575bfdfec0beaf7677ca8c93c901a341fb15c59c0d343  logs/f31/control_seed13_e50_eval_h9.log
  - 76b172b8140b2e4c24bf20dcd6fed6ad45cffa9a52584f228bee23e488a78f8a  logs/f31/control_seed14.log
  - 1a77e9b1c34ef1cd324179955bc4b51d11b8769e39e3f2f8af56fd0df59694a0  logs/f31/control_seed14_e25_eval_h9.log
  - 6c13fdab6f595c78f5c581a74ba6c40fcbda5da8ef1ec98ef3148aa5f27f9e37  logs/f31/control_seed14_e39_eval_h9.log
  - 4650bb71f77a9d08030320a1408c79aa398a4f5f93fe69d9b499167a3fcaa65c  logs/f31/control_seed14_e87_eval_h9.log
  - e2ebb938aa314d7138bf2f0ae664b1c22e6b71b28de20a085def25c87bb03848  logs/f31/control_seed14_e91_eval_h9.log
  - 417fc43b78277d74733b889168f8dada11ce5c0635687b81ac1182250677ead1  logs/f31/control_seed15.log
  - 19ddbba496c86ef906d050f62f74f9e9b810a47ac64a07a67f37c64cf7033e3d  logs/f31/control_seed15_e24_eval_h9.log
  - c4adbef580fe5a3e1646ea78f61116845521f12b7ed33685b24bbf4df0d2b645  logs/f31/control_seed15_e45_eval_h9.log
  - 81910769c9c7417bf39ff7ba1268ef57e13728f15d17b4fb8ebcdee936c3245a  logs/f31/control_seed15_e63_eval_h9.log
  - 6ead3a8bffb5f9b1496395d2b6836258a660b7f091d276b3da6d0c948de2747a  logs/f31/control_seed15_e80_eval_h9.log
  - 21693936dde6311c42b9729b277ae8c587c8a3c52ecbb8965920041f33382974  logs/f31/control_seed16.log
  - da68343f48f898846aea819ec99538a61b2cd35d0dbd54cd1622bfe5e45861fc  logs/f31/control_seed16_e22_eval_h9.log
  - 92d87f87a038a83da650b1b6ef7c35da919fb774e284936882f9be908721c77d  logs/f31/control_seed16_e49_eval_h9.log
  - 2f302aa59895fc01a34ae0be39c0e491e75e6a40486c2459d23e132a43abc779  logs/f31/control_seed16_e72_eval_h9.log
  - a5e65743fa375c42cb548feb8a4dc5d5a61bbf20e9ec4f85a7aa6e43ac0bb99b  logs/f31/control_seed16_e74_eval_h9.log
  - ddfab5d034fb6def6e5f3abc1fafb501065ea334f387ef3adcc7395d1450c4b2  logs/f31/control_seed16_e75_eval_h9.log
  - f7609b9b939f110731e59dbde6af4e6ba60ff0e9cad9dd75cf1e24338c03a096  logs/f31/control_seed17.log
  - 657543a8d5c6629d993d21b3c5652d6da45baa73054bb43bdb5e48302b4b378b  logs/f31/control_seed17_e24_eval_h9.log
  - 1004b2d538e64bd854907593d89a0ccd161e12660e6c74d6874cea915eb0a7ad  logs/f31/control_seed17_e34_eval_h9.log
  - 2bfa78174e769f16ab2040a77ca2474669afdd8f601f4ead0c2b5dc4aaf941d1  logs/f31/control_seed17_e77_eval_h9.log
  - 99977fee46e918f37f20b434a7a49e4f2c5d0924554b81fedc485f84089af07a  logs/f31/control_seed17_e85_eval_h9.log
  - 70878a1862c9edc7dbfbf3a516b79dd9672cbe2260e8e9bbf11ca2f8a57b9c83  logs/f31/control_seed18.log
  - 18435c695e8b0648241f91603b746f09431d91c98220b6cab8bd4dda90c75dd8  logs/f31/control_seed18_e22_eval_h9.log
  - c4abbe3b605e53c2d8d42174d8c4f89ef6c76bc34ae8dab260bf9bec11d4c2fa  logs/f31/control_seed18_e28_eval_h9.log
  - 183c6fb220bef4999a531a546600fa5bafd1afaaacc536782c4130505d530421  logs/f31/control_seed18_e89_eval_h9.log
  - 430231c0c692c5760b9f2820f27c00306a67ce4a767d46383c3103d1bf13e95c  logs/f31/control_seed18_e90_eval_h9.log
  - 29a10f30b8fcbd0b3033371af53dff3c655dbcddc8340bcd8dd7e7ec771f0526  logs/f31/control_seed18_e91_eval_h9.log
  - ca6cc95f277b3752e10b246e607d1722cb3b80b25cf39d0dae3364bf114d5034  logs/f31/control_seed19.log
  - 37147bde354f45ca492ae3117fb296d88f1b9c8d26de472f35ec51ad3618fa26  logs/f31/control_seed19_e19_eval_h9.log
  - 67baf2defa8e1e77da552a523427908816197ed3305ad82b3ba3a7641984a56a  logs/f31/control_seed19_e42_eval_h9.log
  - b82e21bd2ce4eef29751a72eeb4d5e18ed5ef4cede8c122fbeb25c021e66e118  logs/f31/control_seed19_e73_eval_h9.log
  - 7d97cce3927c1aed7280f1399e34671e65ca99f66db005987127cb947d2ae7d3  logs/f31/control_seed19_e96_eval_h9.log
  - fc16129179105a5716145e2036ca8943d6e54c4f9501fec45240a3e68d218d59  logs/f31/control_seed20.log
  - fcb22098f76a5ffcee536548354d3665f370c4de7ca4a7c8a7294471f66f1dce  logs/f31/control_seed20_e19_eval_h9.log
  - 292695c1001bbf206b74e9a1d2615f8a246d274e556aafa571255e03a06457dc  logs/f31/control_seed20_e32_eval_h9.log
  - 5f502e1491b11595abcbc1b4967c29e6bb062aa44cc11142c41d6d12c9eb5f94  logs/f31/control_seed20_e61_eval_h9.log
  - 34f96e9a324481e932a1f5cb9743f4c4b59f17737c3b6b33df14606df80ead8d  logs/f31/control_seed20_e80_eval_h9.log
  - f4b0875ff8f022332f8895b1fd71349349d0a7161e8fec48bfc9c2e54291d980  logs/f31/control_seed21.log
  - deeeff359766cec92ea0e2abe0af0ac8baeb13f6c83bf9bdecca426c98b2310d  logs/f31/control_seed21_e23_eval_h9.log
  - 50ff6357e0a345f0eb47ef5a9f5d9e1e63d0b6bf9c25266d2172f5871866ff8a  logs/f31/control_seed21_e43_eval_h9.log
  - 554265ed4ba283f80f4b786bc897699ac6c996ddbae76d13f6c838f02a71ccb1  logs/f31/control_seed21_e61_eval_h9.log
  - 4a1beba751c78e80a90343f16c081af5f5fe3bf343b1423e48360bbcb53be8d5  logs/f31/control_seed21_e76_eval_h9.log
  - a4ca138fa4855358323baa061432fa1616a2be701ca715481b091caf0113d897  logs/f31/control_seed21_e77_eval_h9.log
  - 94492d703b1ff0eada364b1f0adcd5de86d4dd31ad3eb7893cc1f1f2db1b39c0  logs/f31/control_seed22.log
  - 2612bbb6a1caad5de8e36af0bb4a6842306134307a8942b7f07c92e2f05361c6  logs/f31/control_seed22_e21_eval_h9.log
  - 3a2f9cb699ae1d5b18b4402a614026b9c642d4210360b6f41ba07df0f35e78ac  logs/f31/control_seed22_e46_eval_h9.log
  - 63d3d0f31c0dd9bd71d40e28a7507b7ca72ee2d7e653b54a5c498c646fe197af  logs/f31/control_seed22_e89_eval_h9.log
  - 5c693bb4b2f8f3483fc721ac76f640bda92c0c16a8f5ced6eb127891d8e539d2  logs/f31/control_seed22_e95_eval_h9.log
  - 296e7268c3b0998708b27194224d0d31c43fdfd6ab97918914ab4ae094d29e27  logs/f31/control_seed23.log
  - 9edb8a1f9f8ff48f8e749ec49f939e041b38083ff4f50236112d82011e00659a  logs/f31/control_seed23_e22_eval_h9.log
  - 91765696bec20f1c74bef173c608c4c76f82e74294cf18f80077c8daae453dbf  logs/f31/control_seed23_e41_eval_h9.log
  - 6d00d17991cbbe5c5bc98d8170df6671c6bdf3ea5b0aa981cab158a3878a2652  logs/f31/control_seed23_e83_eval_h9.log
  - bc8c9d1a944ac5e5cda53460ea762d17ea45beadb90a09ae6520e2e4869d6aec  logs/f31/control_seed23_e84_eval_h9.log
  - 3c4a79d669230770a2e39cd8efd16ecbbee862658912be822bf4aa133ce00f37  logs/f31/control_seed23_e85_eval_h9.log
  - 4c479adff1c1a018e6c7479cd87ead98475841bb643907595cedaea408087cef  logs/f31/control_seed24.log
  - 5f84cbc0ccc52e560511e3a2994fb97282eb67606ef2b895c961d1b83742d3a7  logs/f31/control_seed24_e19_eval_h9.log
  - 10f88240afe758406b1d7c479976ea37e9329465076f950a8eb89ed5a8f4c276  logs/f31/control_seed24_e41_eval_h9.log
  - 52839f605b1082ad86a9e8ef92a9da676f607d7fb8d32292f4a29737b39f573a  logs/f31/control_seed24_e51_eval_h9.log
  - 8d86ce6e7bb16acb4dc6964de1960373cb984667b79b8220f7a502f4ad1264cf  logs/f31/control_seed24_e66_eval_h9.log
  - 367420283aa5584421fdbe8520abddbb4111e6ef7f237feeee3a9f5bc20b98aa  logs/f31/control_seed24_e94_eval_h9.log
  - b9d2b8d8ed6a95015c57d587ec556d9fefe28d61303d1cb4fc7821af8421bac7  logs/f31/control_seed25.log
  - a0b7067eccf02545b0d2b0dcc0249b12c157ff8e3685fbdb7c2c6f00565cbe64  logs/f31/control_seed25_e25_eval_h9.log
  - 0162d2e583e446c6f16ddec8cb601fed08c145047ece42b083c315e89eaf7612  logs/f31/control_seed25_e48_eval_h9.log
  - fbb6039d9be386a77a20ec6c4bdd2fb775c6c5e8905c26c3cad652a0076aaa17  logs/f31/control_seed25_e67_eval_h9.log
  - cdf6e68aed0bb9d45938abb2fa90ef2f926b944eec6cf51bb530afc6a74b5fe7  logs/f31/control_seed25_e68_eval_h9.log
  - ad0e36ac3c5c74300b902e2ff12d05ca091157e07638f06c1c34911c975857bd  logs/f31/control_seed25_e70_eval_h9.log
  - ab47452ead7a251a2cd27cfa24cc54886db3f1b845bb2b027eb19eb0fa5fa370  logs/f31/control_seed26.log
  - 7daf589cd7d6d1adf9d8eea9cf975981616dbc012cb7bf6704de45fe2dbf445d  logs/f31/control_seed26_e21_eval_h9.log
  - 8f34fdada7d23a30273126b7fcd5dbd932e54f88690561e65fb9c9142a6edd23  logs/f31/control_seed26_e45_eval_h9.log
  - b8ea7696b4ea549b66396ede376448747f152a7f857317964b378e2d8143ceb5  logs/f31/control_seed26_e66_eval_h9.log
  - 1b515781f61402736a0a045a0ad26b5d41a51d2d0a40b7bd550008f423b1c4da  logs/f31/control_seed26_e81_eval_h9.log
  - 2d8d8c81532d7224372e1e0164df2f122ff7c06a55d58849bb0bbee065bc9f1b  logs/f31/control_seed27.log
  - 64fa48989b2e2b7fdaa04bdca4f3ee3cc2f842b8410748952d1ae2dfc67dab4e  logs/f31/control_seed27_e25_eval_h9.log
  - 40daffb397ce93c0e9376359e1c45867a6feb5b609b5261552b25ef7427b2aea  logs/f31/control_seed27_e48_eval_h9.log
  - 1080a560030cf1d94f779f1b43d001c80e46432815967a62f1a5851e2e66191f  logs/f31/control_seed27_e62_eval_h9.log
  - cf8c74e5759837fa2f0f8a4f9d633ba33c04fbe8acdd570464865c461a88ba6b  logs/f31/control_seed27_e81_eval_h9.log
  - 6795388ca3bb0d6b7374fe56b28a9b4ee881e7c0f7a745e03516998469c31ae4  logs/f31/control_seed28.log
  - b25f32d35d44ce24409933d0087431088b81b1927c6c6859367a2967df125473  logs/f31/control_seed28_e24_eval_h9.log
  - c819076f57327b4e00f1b370202abc63378f9d118f4a07ae05e0e11470c354e2  logs/f31/control_seed28_e55_eval_h9.log
  - 9c52989d483d5dff3a60545c45790c4d8f55045dc0125331970e69c2c11d0eba  logs/f31/control_seed28_e84_eval_h9.log
  - 230f9b9a991919f1ca24a2a7e4f3e3b899cd61a87b6bfddd2a27e2eb42191c83  logs/f31/control_seed29.log
  - 2e60cd1a4320a8076cb71e3b0bde4beba9e9fa9f171b434f5126a84315503369  logs/f31/control_seed29_e21_eval_h9.log
  - e8b72ed3b5db0c952315b4055bc1c0e5a60aa0e402f4849873029f28a3d58f5c  logs/f31/control_seed29_e42_eval_h9.log
  - 67d34373d9f505bddbd0ba2cab7873df256ea79248476131cbc2c213a581f6a0  logs/f31/control_seed29_e79_eval_h9.log
  - f40779b6772c852f7d33642cd71dcae1ae6ab5b9706f5075d8dec87dd3a18df2  logs/f31/control_seed29_e81_eval_h9.log
  - aca1f925fefcc19e976bdce4dc1bdb020039c40eab7c7b2366fa33614da553c2  logs/f31/phase2_started.json
  - cbd9b5c5cbb313b40fb8ccf12fc7edea951d2f476ae698e03bdd9643d7f01fc8  logs/f31/runs/atlantic_seed10.json
  - 5178f0d757a23a0698b85002b64b27dfb30fcac6bc593bc17d6cc096b284c1ab  logs/f31/runs/atlantic_seed11.json
  - e453fcd46b5a6158f4861c987b488c8ee4133ddac965ba19bfa3159175eddf69  logs/f31/runs/atlantic_seed12.json
  - 9c1bdf6f3bccb141d91edb0e48e27005ceb258a88c915022fe13b9f113ce8c8c  logs/f31/runs/atlantic_seed13.json
  - e97858bcf64257306effa36e54f442679d4ee1d30a4ae0e24dd084448a4c5322  logs/f31/runs/atlantic_seed14.json
  - e78beac3e3ddfe689bc2aa5ecd15f6e29a78c2d74aa65f023468ea7a0e4491b7  logs/f31/runs/atlantic_seed15.json
  - 5ab28a6f9327bc6a62909ce9399d776668739a3ca131e5710cd7c2059a30b7a9  logs/f31/runs/atlantic_seed16.json
  - fba759e46d661534f434b68c2f083a027a4eec8b286b99ecc9db5b56481d3911  logs/f31/runs/atlantic_seed17.json
  - a3aa26c6bf57f16d4a9f0c9ff3bdcb3c9745288a745580ff5a81de4b3a67b13c  logs/f31/runs/atlantic_seed18.json
  - 7f5061d6f045c13cc04537069639036b67aad310c7fc0e5d38d576c13a756aa8  logs/f31/runs/atlantic_seed19.json
  - 1c1f11f5a703c666c1ce57aea229f179716d89c47a0d6807029529abcf6d26e6  logs/f31/runs/atlantic_seed20.json
  - f73d28d7cc6a63a6af979e3dc0ef3ffdf05229c138471d39baecb2279a2a18dd  logs/f31/runs/atlantic_seed21.json
  - 15c8808c0ed12502d5eee0ea149c04d7244a63d2f59a656de9833b62daf752c3  logs/f31/runs/atlantic_seed22.json
  - 59913696992beca4c802ae8c7178e7c43cd426812918c3f1bd1c0f3fc63b8d98  logs/f31/runs/atlantic_seed23.json
  - 17f94847112474a6cc8e7475a9b05425c740725fad5b7ac5e1c09740490eeb3c  logs/f31/runs/atlantic_seed24.json
  - 03c6fca1509a82ecbfb4537a02949348f89116419e6984dab77804fae54fb3db  logs/f31/runs/atlantic_seed25.json
  - 24bdad144f086ca0b72593537bee82338c9ff5af4dbb88987bbaa188de314c39  logs/f31/runs/atlantic_seed26.json
  - 6fdc8e5bdc29ab401aef9a96f03b5a14db2d5375bcfbb0c62ea190f701ce87b9  logs/f31/runs/atlantic_seed27.json
  - 2263c0d0cbd1037a96e8157356cff0e51b2303eb93a465b7191ee0f09872ad70  logs/f31/runs/atlantic_seed28.json
  - bd438277ff9ce753b4193c6ebc787c94d9e8af341dc5147a659726d1bc2cfc41  logs/f31/runs/atlantic_seed29.json
  - 66b24fd4cd72c4d988263ffde10edaacc2058f97e2f19d98e4835fcbd7533ab0  logs/f31/runs/control_seed10.json
  - db03f79c871c3a92c0b80981bc81f55d744ac7c3a0f7508f633eb79177f9b131  logs/f31/runs/control_seed11.json
  - bd44d6d3d72c2e340ada335ec656bc692a73f20c30d6c190badf754fea676061  logs/f31/runs/control_seed12.json
  - e56e2afe74af85dbad638191c7ec6282d7f5df3a9aebf2932171d5cf4d848557  logs/f31/runs/control_seed13.json
  - 5caaebaba300bd16387044ba03c81c9ada11a6b0f18f8b26e86f1013445c1e98  logs/f31/runs/control_seed14.json
  - f87e511d278a4fd6ef40471b46ed0771bbc51197d8b967f78e6eda5c7d7428f3  logs/f31/runs/control_seed15.json
  - b6d5d703d26c35ce4b7f2761159bdfdd772df869a2515824df3adadfcbd293c4  logs/f31/runs/control_seed16.json
  - e73e471c768e711e77ab45bf42790c0d28b540d0c00b5c5cfe698b3b1c81c348  logs/f31/runs/control_seed17.json
  - 3c826a922f6de10a62551a0c37b15633c15e388681fda3905ce5d17aa35e8921  logs/f31/runs/control_seed18.json
  - b5cfef0b5654e17da2b888c66f96d8ec6b7738a60504d0eedde6b11eb905fbc5  logs/f31/runs/control_seed19.json
  - 19dc52aad092db25faf4cec3a389a1ca456018b82de13e28c95bac2e25289e81  logs/f31/runs/control_seed20.json
  - a58f80671559ac0dcd71aaa2b560eceebf507eb435d939478076d6eac3abf6b8  logs/f31/runs/control_seed21.json
  - f14fd1957fc616e9b99ed44969ae0b66b41f75922d1144d453f25d3b964c421a  logs/f31/runs/control_seed22.json
  - 715de288281ca8a4482eb87170db8127319dae6f11dc72c828075d6e3f94c294  logs/f31/runs/control_seed23.json
  - 18895ecb91bc1e97576d47fd78816c24335afe37c85861e23696add8b80ccdeb  logs/f31/runs/control_seed24.json
  - e2ca2a7fecec9c14503fc5a73ba7407d4cb9c5e84490326b27d06fd2d06f0983  logs/f31/runs/control_seed25.json
  - 10fbbd19115410ee3e08a8a6a518ad71ad9821b3c39b11e844a2921cc109f659  logs/f31/runs/control_seed26.json
  - 5c107475096aaa771bc22890ea508d7e4f5a4c91c6a082539c0a62db122ff5bc  logs/f31/runs/control_seed27.json
  - 580d52d0274d562f00eb61c16fe4fe5f34c054ead7cfa6452367e9b257b49336  logs/f31/runs/control_seed28.json
  - 1d34bf58b6f24e782596a441273139661ff2b42177b023a41511eeceafd1319b  logs/f31/runs/control_seed29.json
  - 37b6ff2d4f361cbe193a606001409f85010d946eb26b3fbbeb71c4eb376be48b  reports/decisions/F31-result.md
  - ad8737146aa60320edc228ed52041c6576ec0b7a4fb6b9e3494254d32284c7fd  results/f31/analysis/budget_curve.csv
  - 96f9dcf1146ebe7dbdf86bf32bc98150665eca84da43b85a38e017a8b8c92cd0  results/f31/analysis/ensemble.csv
  - 170b1a9525d1994917e8370473a3dd52d7e45c00d74a65fcb64a87c4ee1295f5  results/f31/analysis/per_run.csv
  - 5875c4a39e47ef4c06c8192a0325a5736362e1281dd4188db4be1ce5331948f5  results/f31/analysis/per_storm.csv
  - 729a36d01b6ddd70bdcdbdb49654c795814bc393dbd570a47d26abdfc0f891c2  results/f31/analysis/primary_tests.csv
  - 3fbc930478fbdaa1c346011d0bdcb97ef8ee8125bdedb7bc5777dd129754073d  results/f31/analysis/report.txt
  - 076bd1b3098a8816879104d2b58aa1ee19ba3235ae401eec93b03461c1348ac7  results/f31/analysis/secondary.csv
  - 52854ec8842d75480ce07bc8eeb2c1052e68abb6fb02630183dd1ed40c915fd2  results/f31/analysis/selection.csv
  - f1fdc54fa6ff2f2edbebb182431552e741d8a0a47d4f902126dbd8ab87c56cb2  results/f31/analysis/sensitivity.csv
  - 7efec6c15291c6a85737784b050bda922c363a918da14c35f0eefc73408aa11c  results/f31/analysis/val_trace.csv
  - 46210c2e8c84770b69b667c6fd09510535d210e8252c19b1932061317ee4d401  results/f31/atlantic_seed10_e12_h9.csv
  - 0f2334914d0e506b97517c8a5a81aa468fbad8b6cb2c1689494b34a3d5b259df  results/f31/atlantic_seed10_e25_h9.csv
  - d3ebc6f420d49d357f5c7f14ba0695e16ff96f5ce10fcb0713de9ba11eac3efd  results/f31/atlantic_seed10_e30_h9.csv
  - f906eb7704aeb80037b3dfc7dc14b56aefb3a715ff4cf892055e434ad97df2f8  results/f31/atlantic_seed10_e47_h9.csv
  - 8ead22e8bfde2a879f7ac5357e65a7dc558ba46a01e860c11f536cd053654248  results/f31/atlantic_seed11_e10_h9.csv
  - 92bee723947ffbd47cd7f7895d1e334ff7c4100d2465d84adc2f2d949c3d7b86  results/f31/atlantic_seed11_e28_h9.csv
  - 6c1f1e746d14acd43aafa26ef7a1c5a8f808b0919616745de602e48f77c1e698  results/f31/atlantic_seed11_e43_h9.csv
  - 9ebf43fccd898e2c9efd3bedfafb91e8c9cff5e9ce866dff6a1461b8134128ae  results/f31/atlantic_seed12_e13_h9.csv
  - 00c3d91fbb661531e3319512f2eaf60b3280944a99b129a63fd8c010d5a77be3  results/f31/atlantic_seed12_e24_h9.csv
  - 062d6b7591357448fdb10ca08689b4421d6d39dcd799b2a8c90c8a1d9469245c  results/f31/atlantic_seed12_e38_h9.csv
  - 68fe2ed3895a2c7ea3505bd348dc9c2e1b6388ed8d2de73e5c4e8ed47633252b  results/f31/atlantic_seed12_e44_h9.csv
  - 01c8b0d87272df33b220b974c3a6db6495cda0c1c5c04b992aca2741c16fc173  results/f31/atlantic_seed13_e13_h9.csv
  - 73b6c129768bf790ef75ad1af74ddf2898459eb7ce64d0b1e67378ea955f1a11  results/f31/atlantic_seed13_e18_h9.csv
  - e40fdd6f62ab4cdcd09a8c9f0204880b130c6b8b026940bc79a742ca13db8675  results/f31/atlantic_seed13_e46_h9.csv
  - 14aea2bebdcf323ac26425d8c81c151002cbddc81a273ed8ba2fd0aa401c8000  results/f31/atlantic_seed13_e48_h9.csv
  - f1efb2a5c71301b99794655b84984250b0f7ff1a09af16c55d2fe4f4ee6b3a79  results/f31/atlantic_seed14_e10_h9.csv
  - 7b3d98112b8b1930d9f6cfaa3fbba416212d9c806b5f0be7cc762a500f666f14  results/f31/atlantic_seed14_e24_h9.csv
  - c0e1785fcaa176c7887b0f64f8a107c3d9d7cb835dc8d51af63ea725927ac2c4  results/f31/atlantic_seed14_e31_h9.csv
  - e670c82443aa5aa23010648b025867568f3214f3821d8ca9348e753819980edd  results/f31/atlantic_seed14_e45_h9.csv
  - c1c261a8ca0e5567a4bd6a4479958b8ec3eba3470fd990b162670a14544ca0ec  results/f31/atlantic_seed15_e12_h9.csv
  - 4757f83f17a5eead36b9e332b3f396c89d78fde2b827626c7747c82e766b80ef  results/f31/atlantic_seed15_e24_h9.csv
  - 8baf035f6918c86ddf1ed81a5060cbf12ff5db91a11f20c74aa0324a5807de8c  results/f31/atlantic_seed15_e37_h9.csv
  - 434f85a51ca5e9a5cfe76693a814667527f58bca0f14bad6373e3a89fd764ea6  results/f31/atlantic_seed15_e47_h9.csv
  - d14e7cc244035d00c2dc04cb9be086fcb8f933f955f76e6ca4c2f4355101ab7e  results/f31/atlantic_seed16_e13_h9.csv
  - aef92f662773c0fe847909161c80a3839e0865de246b368b7e1aba92608839ff  results/f31/atlantic_seed16_e20_h9.csv
  - fe37ffccb4def963bd9498f4b24cf95d54b99226739b6148f29bebde9c5e71bc  results/f31/atlantic_seed16_e41_h9.csv
  - cd8cda5708c6d1e02d041b8e0e3e1b5f47947a658d7b1dbda864890d3b823894  results/f31/atlantic_seed16_e49_h9.csv
  - 15ce1c8201df9a9ee3acda008e41094e4da669ac1b17e47d5c4e602f3fe269af  results/f31/atlantic_seed17_e11_h9.csv
  - de25a26917d7681f1976294941fc0bb83f33ffbdc9ec3aab56c6e222b52cf670  results/f31/atlantic_seed17_e23_h9.csv
  - 976296a50dcb2361cbf04005fcedbb06d94655211ff32e4086b19b906487204d  results/f31/atlantic_seed17_e37_h9.csv
  - 2753089295fce304ea71924722bfdc736ea1976a53e8351d600a00f168de1492  results/f31/atlantic_seed17_e39_h9.csv
  - f4f90541ea0f6d0f54a6c7e5a4908886e09569416bb440fe9f47e36de3dbb6f0  results/f31/atlantic_seed18_e09_h9.csv
  - 550a85d3e1c309ee287e6226e3171dc5f75cf527ce37f11f699cabf99ee1151b  results/f31/atlantic_seed18_e20_h9.csv
  - 17f5c6bd54935961281e2e6dfb89fc38de0eaf2c98a47d044eba6594f7d4a822  results/f31/atlantic_seed18_e42_h9.csv
  - 5dfff2f6f900b64cbbdc4bd936e1784b35a4c525500e0abe483772243d07d42c  results/f31/atlantic_seed18_e51_h9.csv
  - 0a9a639f2b6b8a06e6afe031fcaee519b08427de14f52e9ef1cffa1f5bf00d70  results/f31/atlantic_seed19_e09_h9.csv
  - d1d3b335dfabc9bdb51af5f64f492b20c8080d4ae96ca84a1d224c43123f353d  results/f31/atlantic_seed19_e24_h9.csv
  - b4a46393ec7007713af8bdd3c90fca272823b421afe47f1e1913134d74e2c9f8  results/f31/atlantic_seed19_e46_h9.csv
  - 6f4e40779b9eb4b157db69ffa01b8f48e902758386c729ed385a4cfaf2017328  results/f31/atlantic_seed19_e48_h9.csv
  - 49a398b751a9325bfcf83dd67a947b22f5fcd813c37f83822b6f31bbc4db3f52  results/f31/atlantic_seed20_e10_h9.csv
  - dbc1e443a0756ae766b13b5711c7e7b28ba46241747c5145ec30e9b09bced00b  results/f31/atlantic_seed20_e22_h9.csv
  - 6988ed9b1aa7c5028835f39b56e54433a27dbf34a67b253ea80b833b713c91d8  results/f31/atlantic_seed20_e43_h9.csv
  - 77b9ebf56ec1eda8c90d2727c062ffa43d0e9f74c1ca8ccf66538d639c17b63a  results/f31/atlantic_seed20_e52_h9.csv
  - 3b5f7c00cba80d52a0fc93e17a1960ac60ce7966a8ab610632a36f2bc38f592f  results/f31/atlantic_seed21_e10_h9.csv
  - 18a8c4518ef237553a1f8629d3e3eb5a2a75325092764a7994643cb0389c7feb  results/f31/atlantic_seed21_e25_h9.csv
  - 1de3625a5fcd7b9a3c83b3e00872a15dd6966cffae0a3c5d1046e59b98be5f2b  results/f31/atlantic_seed21_e49_h9.csv
  - 3613bfbdbad074bb3d7cb0abf2c2e198b750f3faef6f040a0e4246713f685470  results/f31/atlantic_seed21_e51_h9.csv
  - 10855a79e07538676ece51dbbda07d0df6b08020cdd3c2667de4d16f5e4a8669  results/f31/atlantic_seed22_e08_h9.csv
  - 098ab18462890d37bb94e8d75b376c0cdf79275a9fd7524d8bf6d863a7270c20  results/f31/atlantic_seed22_e26_h9.csv
  - 3a68db5ac39b73d1df6c406ba02803144cdcf76ee3319360985badcec4572344  results/f31/atlantic_seed22_e39_h9.csv
  - 2177d5558c809f92efe421ccd12bc6a31599c147c24d91c70d560a22db0b0d40  results/f31/atlantic_seed22_e50_h9.csv
  - 18e2062d5f0388d9272184ab287e5e25c894e62e7511221aa12011f643040af9  results/f31/atlantic_seed23_e13_h9.csv
  - 6f7b5c2cdd913b998ec2347c5b4a6aad2d5f757be480b2cadeee50e783238ded  results/f31/atlantic_seed23_e23_h9.csv
  - 9d395829662c7da2ad1cd3c89218a9b71efb2522c1e3f09c9c516faa667ccbf6  results/f31/atlantic_seed23_e51_h9.csv
  - 9deb9febf6271c87b4e0f8f075599d08b77ae08e147c0d6b650a325c8d123b7e  results/f31/atlantic_seed23_e52_h9.csv
  - 28130817afd2653a5e4b142fcc5e8fd891c60376dceddad2a9a5b88620e86114  results/f31/atlantic_seed24_e10_h9.csv
  - 1202b64d30d701578a38d11b24b1e314169036a095b28dbbaf710501b1adbe58  results/f31/atlantic_seed24_e25_h9.csv
  - 1aae05377413902117aea2a2a02f4773881068ff2eff3b8798b0299855e2c0bc  results/f31/atlantic_seed24_e50_h9.csv
  - d938415d23b0e539f6d2eacce5a435db2fc6b975b87cae3ecc3ff94156df9e67  results/f31/atlantic_seed24_e52_h9.csv
  - e4eecdb3289a0833d59af72736867d686559616ee87ab3b1a3dd09e90a23ceaf  results/f31/atlantic_seed25_e13_h9.csv
  - 9091c2c4f9273bfd4e9838c6573545c774ff6b74f0adeed1853676fc3a0f8846  results/f31/atlantic_seed25_e25_h9.csv
  - a7c0597db40bb9712c622718284752d244681b9f6fe85a87cd13cf220e24b87f  results/f31/atlantic_seed25_e45_h9.csv
  - bfcf902280ed920cd0b9a253710f8076a795839ba4e361cf95f51775449a1a7c  results/f31/atlantic_seed25_e46_h9.csv
  - b8887644e4c3eee68e575b5de92a35448725a72f2472b7bfe11a3aaf4a5faa0d  results/f31/atlantic_seed26_e10_h9.csv
  - e21a9a53b0db6f616ed3bb980a4d8a2b4e0be2f5f556554d79afe74069affcba  results/f31/atlantic_seed26_e17_h9.csv
  - e23fbb8c783af551e494cc9b6633ea1f7d2181a03db507e552cca6adce18e174  results/f31/atlantic_seed26_e44_h9.csv
  - 6bcc3b2ff199a8c495042bc41061622952fbf61d7b891dc7bfa68d6b2c27ffd3  results/f31/atlantic_seed26_e50_h9.csv
  - 26d2b513ea13c05af131770cea44e5788d54473c4feb16733d7c0e188deb4b71  results/f31/atlantic_seed27_e12_h9.csv
  - afbdebd90e9815efae6ae1c38a3b7a4379ec488fc14c7736f111ff23c6aa3393  results/f31/atlantic_seed27_e24_h9.csv
  - fe2c62a4dc78001da739c563132222e5b70bcbdac89913b4976b56d9a0e5d61a  results/f31/atlantic_seed27_e28_h9.csv
  - dcef36dac015a521bd5cda85b2b7b4bfde3ce83f9fca66db3e5b389e28e554e0  results/f31/atlantic_seed27_e30_h9.csv
  - c57d6dd0f9a0d5721296d4cd279cf2812158a8abcd8412d6751e4a8df2f7a4ec  results/f31/atlantic_seed28_e09_h9.csv
  - d9c9eb7ab659755d73a561803964a9994f6adfa1795ab735c24d55f522616789  results/f31/atlantic_seed28_e18_h9.csv
  - 9f208f1b43a494aeb624ac7ea4e3f1f3e04bfa8b880bcd8142a8b15326105564  results/f31/atlantic_seed28_e49_h9.csv
  - 27095a87c716462ef11766e67d7d211cedca2144d32c57fd1d8f3cce3eb18215  results/f31/atlantic_seed28_e51_h9.csv
  - 5e977d75b9ac820efd10cf2d596d77c43eab98fb47dd7adb5e19da119248030e  results/f31/atlantic_seed29_e10_h9.csv
  - 9f0809f97f59512ba479bf2ee1135e1e955fd109840028524f13ff8382734917  results/f31/atlantic_seed29_e25_h9.csv
  - 662780823c85ef9c1c3e74b113112187eea2396e6da2ec47feecd7e1dfe34e4a  results/f31/atlantic_seed29_e46_h9.csv
  - e39ba0979e26bb193b70de20b1427cbf6a23314d5bab3eca1ea0d00e62032327  results/f31/atlantic_seed29_e51_h9.csv
  - 81152973bdcf680e35e068008447a04dec3b976a59227b1437026ce044d8b70d  results/f31/control_seed10_e15_h9.csv
  - f07a9b2bbefc7f18366b322f10d4028a839a1884844f76167ec2f81b503fe64c  results/f31/control_seed10_e45_h9.csv
  - 839acf1dc113a369dabcf4ce895fb76c6a2bb71dced2efb0b2e43ccadbdd1f20  results/f31/control_seed10_e46_h9.csv
  - a734300dd24c30cdaff20d4cbfab44867e944dd3bac98207a4d2f90b9ef03f08  results/f31/control_seed10_e47_h9.csv
  - bd7c5929a0141d2eaf13372ef6a25f89210b413e213836be3de36f1d1811fce4  results/f31/control_seed10_e76_h9.csv
  - 8741a02993ddcd4a4d281e22877eec0f1bde847023f36b8b2e37e7755afa27f1  results/f31/control_seed11_e13_h9.csv
  - a6a62215d06e977a6c32666c8d614a81b6add3b397c4af9c3f6a1a3368debbb4  results/f31/control_seed11_e27_h9.csv
  - 557688392dec7cb670af3fe00f2d0107369246eadb784ef9aa33b64e594df0e6  results/f31/control_seed11_e71_h9.csv
  - c3a8aec3070808698a98afcb713f553cbfb3fb6772cb569a516cead386b78653  results/f31/control_seed11_e91_h9.csv
  - 8fc5cb404e58d505d1ac8b7920e1e62f83b581eaa2dad0f0623c06f10673bac7  results/f31/control_seed12_e23_h9.csv
  - dfb048411c6cc0336803cf05b4b11b76f81d182940519871d14138464da738bb  results/f31/control_seed12_e35_h9.csv
  - 4d64b8922a6253d40bd9b85d28c7d9bda8ac64b2eff69cf98d4e20ab1ed49322  results/f31/control_seed12_e63_h9.csv
  - c4d427d2eb7dd6071cb6cfbe3c294c3265ca2cee344d39a4c3ea6ccc4572bb2e  results/f31/control_seed12_e86_h9.csv
  - 1c067dee207c96d0fef00440f973862d3e491d15dd1f18badea2fef3a37743f5  results/f31/control_seed12_e88_h9.csv
  - 04e332f72114f02a76a7ae9cfed6ac6ac5f5ad52a65dcc9e4157caf2460e698e  results/f31/control_seed13_e22_h9.csv
  - 0f2eb8085a76f75fd6594eaeedc905518060fceaab1e44128a7d970c57981ea7  results/f31/control_seed13_e35_h9.csv
  - 3c45b5e4a4d3da0af74416ba23d62ed52daebe64bb3a55565871edc86062dd72  results/f31/control_seed13_e50_h9.csv
  - 3a022e3a640080b48593eadf3c76dac195162e251e48ec56a31ba6f43f3f95ff  results/f31/control_seed14_e25_h9.csv
  - 9089d7c6c82ea0e972e3870290822737b2801db5741136dfa064a0753b3adc8e  results/f31/control_seed14_e39_h9.csv
  - df0bada6b7483ae2d7c123c78afeef5342c494c5690af0139bb8dbb404679a11  results/f31/control_seed14_e87_h9.csv
  - 1db139d232134bb9cea218f7b18923efce9a999b3871a34fa8839b6879ede853  results/f31/control_seed14_e91_h9.csv
  - 9faf1150939d61200b11a136353e877e6949eed74a30a555f9c7a673b8f40918  results/f31/control_seed15_e24_h9.csv
  - 436d1f86c26f9395c3e0d89ee40975cf65807fd30827ac34401597d99c303186  results/f31/control_seed15_e45_h9.csv
  - 246a71877f3507626e77d442b7355e11ad8990880c475f7caad0faed18e2963e  results/f31/control_seed15_e63_h9.csv
  - 8075609359c47c199f7ef0568c0f6576c44971f64c4b09a8ebed6d6f49a63422  results/f31/control_seed15_e80_h9.csv
  - 0447cf96dd76342c30d9fd96bc6e789440c89094e6b9ee9183525559e3618f05  results/f31/control_seed16_e22_h9.csv
  - 21ae307be9f252d0094b88bdc520f31ab3352df41ef95ad8aba6c80bea3247da  results/f31/control_seed16_e49_h9.csv
  - e4ec07ed883b98044a39b85abacb432f0824c2808373ce89383fa133c0089d77  results/f31/control_seed16_e72_h9.csv
  - 3dae572ace7a8248cf155d892976968d35555f709e2c59e23c9c27b7d067ebd1  results/f31/control_seed16_e74_h9.csv
  - ff4e5080700eb531e900abcf31541b6dfae6b3a3b76af19252f05898c08f4d10  results/f31/control_seed16_e75_h9.csv
  - 35fe78656e75a9a79f6a4b0ec6eb2e5fdeb92569513d52e4334e7e2d07fae69c  results/f31/control_seed17_e24_h9.csv
  - f8cae78d6932629e117370e78b3ab9ece7527e420547b2a948ab61319d474f87  results/f31/control_seed17_e34_h9.csv
  - 0cabd747bacd70efa8b87b7b08531175ce2cc227208437eb12a4450dfd106d6e  results/f31/control_seed17_e77_h9.csv
  - cea5d4cc8f7a3065902594bb0ce36458a419c9d11f5e64302ad34aed87f145ae  results/f31/control_seed17_e85_h9.csv
  - 003b07ccb4dc6eb9a8b40cfb4c1317512132b5bd30fbcb20b3b80144b23ae9e6  results/f31/control_seed18_e22_h9.csv
  - 66df38100bb04e67449b55e5d3913749b340c32327d7c1bb6b6aae00e191946f  results/f31/control_seed18_e28_h9.csv
  - b542438986f46ee13ca3433e6b4f3e32a4b9dace9a68106a187cbf0941693506  results/f31/control_seed18_e89_h9.csv
  - e1b6c61f1d245b00f3470b6d6d9fe582c9fb6934003e5e394060c211a197be7a  results/f31/control_seed18_e90_h9.csv
  - c3c51977f9d77180ddc1a78d8ef263f507fc5fae51742ea7e17e63c7785f5d89  results/f31/control_seed18_e91_h9.csv
  - e8497b81347508d7ef1064154e95449d68eb83bf583786604de08151d8b03a77  results/f31/control_seed19_e19_h9.csv
  - 2987f2e91bfa8d391319e17ee99f08a493ec68a7031b6e5adede2b4fd1bafa3a  results/f31/control_seed19_e42_h9.csv
  - 5555966153065e61ad3b5d037b3a52741f953fadf9cc3808b85fa36cb7946cb2  results/f31/control_seed19_e73_h9.csv
  - 185ebf29396e54bc8362f07eee74b5a8ba83824e0e9c52a38ea21688df360184  results/f31/control_seed19_e96_h9.csv
  - b64e2dcbb129f11666b4454172740c6e93345c27e32f52a40bc644be56b5043a  results/f31/control_seed20_e19_h9.csv
  - 45f614a1412701a4f1d8d45c7865b29e4079f0c26a3ebb7b351b2e7cb14280e8  results/f31/control_seed20_e32_h9.csv
  - e4205c6419ef2b211ed4253dd82bb0b5c044719fcdaac43fee0d1b48788940c0  results/f31/control_seed20_e61_h9.csv
  - e9beafb581638081d6539e71d9f7b284bc4ca880fce5fad74185c0932ea6b56a  results/f31/control_seed20_e80_h9.csv
  - c99ee22423dd09d24daedf57acf72b713d375b5816c897c332b23f8cbef3fe8c  results/f31/control_seed21_e23_h9.csv
  - 73e776a44c48f39ca26f909b10b1b2926e11a79d6e00c88bfd6b9118b614ae4c  results/f31/control_seed21_e43_h9.csv
  - 170ba451f379d7ee1a40aa1ac17f929e097aad422ad7da4661591a894e64bf1b  results/f31/control_seed21_e61_h9.csv
  - d4c3280a4763d5f0f867a6389e2182cefc8b86d1635d107c9562fb05f40512ce  results/f31/control_seed21_e76_h9.csv
  - f1c08a2fc52f2c2daf90c77d731cf0ba705138e1d37d4037af01f95eda839f7f  results/f31/control_seed21_e77_h9.csv
  - 0d45b35cdf43d399e20c6fddcacdd627ceea145315329635b132d0722088eb22  results/f31/control_seed22_e21_h9.csv
  - 6036e8c3613799571f379bffd4cf03ef910f7a038e46b4ed8b08b1e0b63c0229  results/f31/control_seed22_e46_h9.csv
  - 97a772764217c6ed4c3e80d6f6589d3bf97787f9887340509556537fe9f9839e  results/f31/control_seed22_e89_h9.csv
  - 70c02bf6b1f14d961d87fea30cfd9f43f10aaa7c5c713c45a97bbb8b72eb30e4  results/f31/control_seed22_e95_h9.csv
  - 7e0f6767fb98ab4f1e2f07e79115a97483ea854d4423bd5ece070b53a3159020  results/f31/control_seed23_e22_h9.csv
  - 498772de8410c0b76a245c3f91ad948606033d00b590b7ae83cf862c752d4f7b  results/f31/control_seed23_e41_h9.csv
  - 2f7a2c36f9b22961ee05ccc2403426d39713056030382e1151075fd2093fbc1f  results/f31/control_seed23_e83_h9.csv
  - 05ac5adef8eb3df2ce9bf77505cdff9931667a5df584409dcf47e48ee97e4a12  results/f31/control_seed23_e84_h9.csv
  - 04ea101f8cd14dbdee4429e7cc98f704372db962671ad28738c8ce47b0878ef4  results/f31/control_seed23_e85_h9.csv
  - 8b6c27804d9183bed6d87ded7ad9d00f455d723e6ba07e724dcae1efc9f5c11d  results/f31/control_seed24_e19_h9.csv
  - eadef7218d30339fbe55f60f1d292d8a14f3902916d2a86c6c43bca4e261433c  results/f31/control_seed24_e41_h9.csv
  - 1c9eef0157c31b16df96460223d83bdd00f200bf5fb61cc3a7846da51647b027  results/f31/control_seed24_e51_h9.csv
  - 701e7f04e2f2e9f723ab59f194269cfb2127f57206d09d3015c4a6896b62a25b  results/f31/control_seed24_e66_h9.csv
  - 6352cbcf7d06bbaf1106b14939c1f7b0f8306f01ac923a4899ec5d0d4d53b75f  results/f31/control_seed24_e94_h9.csv
  - 6e362df8177aad5a916927bfc0ab02dcc767e791e1a5de3718838025f364cdec  results/f31/control_seed25_e25_h9.csv
  - e86f2fd493e1b2888fb50aeb2f847264ccb2907cab991712bf9f5ea09357578e  results/f31/control_seed25_e48_h9.csv
  - 3c0a7b3c061a5fd837f02f85f781444d4962f73286ccd27d2be6beec1835f7d5  results/f31/control_seed25_e67_h9.csv
  - c7cc7b497886a0780f76ac82ba8bab880aed0fe144a7d71e181d04ec86a44a7b  results/f31/control_seed25_e68_h9.csv
  - 5f6eb2d42882d0ca45ee0a1fd61d1fac48c8d9606589b366f2d3c2ff05127bc3  results/f31/control_seed25_e70_h9.csv
  - 34906e98ed4448d58abc85dd1d4f02575a7d263514590af3126a3f4407521b41  results/f31/control_seed26_e21_h9.csv
  - 649f1eafd4865fadde74c6af5129631fa19065ef85a8bc1bb8fd9b974f17b37f  results/f31/control_seed26_e45_h9.csv
  - aec537d9c051994cffb9607ff61480d5912ddb62a31d66231f1c16e2d56854c5  results/f31/control_seed26_e66_h9.csv
  - b0f115894f1a30b7fa491bb1184d3e1357be1b329482160f0534cfb52acfe3cd  results/f31/control_seed26_e81_h9.csv
  - d0d25f6fa5c8c083efb7efb9493f73eb052a612cafd9fb261b5e7ad319e94764  results/f31/control_seed27_e25_h9.csv
  - 56428274c38f6e8df5d846c44119d98f3ab9c7c3eb36c981d4cfd41afacd0266  results/f31/control_seed27_e48_h9.csv
  - 15646721094dfcd455fdcc4e49a406ad43d20ab2258cc9dfd2f4ec844ced3836  results/f31/control_seed27_e62_h9.csv
  - 5788af064d643ba8e768754d65b0497dcfa9bbc5cc53f1151253e5b43acf102a  results/f31/control_seed27_e81_h9.csv
  - 5e16cc2dc1d180606c9e688a28fc77bcdfb9151d758a31db35ed9daeff70ba8d  results/f31/control_seed28_e24_h9.csv
  - ff6d3df4a32bfefa0981a9626767c0e87dfbc86721c55ed8082930cd5a23a09f  results/f31/control_seed28_e55_h9.csv
  - 6a0b4767405cc4972e0891c2ada48b7eaa1c4f2dde4bb2a3025ce4f86cfa440d  results/f31/control_seed28_e84_h9.csv
  - 138693c79ec89b385ed83736b8b47928e0b1f7e0f1c132fac6c8b1c58bc2fadd  results/f31/control_seed29_e21_h9.csv
  - 3517be23ffc26901962cb6670b0b354b9fb2ce0ff19f7ef9f5470b7c846b1169  results/f31/control_seed29_e42_h9.csv
  - 0316aa6c55ec2a564882b4851f29d26bfbd988502973413cfe6e8dc0aae5c572  results/f31/control_seed29_e79_h9.csv
  - 93fdc5031ce35eedcca741821b794f7504cf1398cb18612eb48923a9122f951b  results/f31/control_seed29_e81_h9.csv

2026-10-21, 16:23. v2 after F-31: John's four decisions on the
coordinator's note. No training, evaluation, H9 or H-EP read, and no
compute, by this entry.
- Source: reports/v2_survey/f31/v2_implications.md, section 5 (the
  coordinator's draft note; untracked). SHA-256 of the note as John
  read it: 82381a1f1fac16aa8a872b69549906a253c05894cbaf536f22cbcf5c7be08f63.
- John's decisions, verbatim (relayed by the coordinator):
  "§5 decisions:
  1. Yes: the 4× control recipe (99 epochs, native selection, 10
  members) is the v2 Atlantic candidate, confirmed only on the 2026
  clean test.
  2. Yes: hc-survey drafts its design, hc-adversary reviews it (max two
  rounds unless there's a concrete defect), then it comes to me. No
  compute before my yes.
  3. Yes: E3 uses the same fixed budget, with an Atlantic-only
  comparator at that budget (the v2 ensemble may serve).
  4. Yes: start the 2026 clean-test protocol draft; the
  no-2026-imagery rule is unchanged. The draft must lay out the label
  choice for me: wait for the post-season HURDAT2 update vs. use
  operational best track with a stated caveat, with the expected date
  for each and how many 2026 storms each would give at ≥96 kt."
- What this does not change: the frozen ensemble stays the reported
  model; the 2025 Atlantic test set stays spent; I4 (no 2026 imagery)
  stands; H-EP stays sealed under protocol.md 3901f4cf.
- Next (design only):
  - hc-survey drafts the v2 Atlantic recipe design; then hc-adversary;
    then John.
  - The 2026 clean-test protocol draft, with the label choice laid
    out as John asks; then hc-adversary; then John.
  - E3: the rule above applies when its design is drafted.

2026-10-21, 18:37. PR #31 approved and PR #32's eight decisions: John's
reply to the coordinator's packages. No training, evaluation, H9 or H-EP
read, and no compute, by this entry.
- Sources: reports/decisions/PR31.md and reports/decisions/PR32.md (the
  coordinator's packages; untracked), and the PR #32 draft
  reports/v2_survey/2026_clean_test/protocol_draft.md at PR head
  13091026a59d19e5ace41cb1cf70c4a06b038b25, section 10.
- John's reply, verbatim (relayed by the coordinator):
  "PR #31: yes, merge pinned to 2b05386 on PASS FULL total.
  PR #32, my answers:
  1. Option 3: pool 2026 and 2027 in one scoring run. Decision 1 now
  reads "confirmed on the 2026–2027 clean test".
  2. Labels: 2026 scored on the HURDAT2 update (it arrives before
  scoring under option 3). 2027 on operational best track from a hashed
  December 2027 snapshot, caveat stated. Yes to the hashed December 2026
  operational snapshot as backup.
  3. Smallest effect worth acting on: 1.0 kt. State in the protocol that
  F-31's H9 gain (0.48 kt) is below it, so a null is the expected
  outcome and does not mean the recipe failed.
  4–6, 8: as recommended.
  7: noted; ask me at the list step.
  F-40: the 2026–2027 protocol must state its storm selection rule
  explicitly and say the 2025 rule was inferred (HURDAT2 peak ≥64 kt).
  Track v2_implications.md: yes, as a docs-only PR you may merge."
- Items 4–6 and 8 "as recommended" (PR #32 draft section 10): replace
  the frozen ensemble only on BETTER, as its own committed decision with
  new pins; 2026 East Pacific storms excluded; the clean-test pass holds
  until E3 is settled; no subtropical-only storms and no re-scoring
  against final HURDAT2.
- Amendment: decision 1 of the 16:23 entry now reads "confirmed on the
  2026–2027 clean test". The 16:23 entry and the quotation in
  reports/v2_survey/v2_recipe/design.md stay as written; this entry
  supersedes them on that point.
- PR #31 (the v2 Atlantic recipe design) merged at 18:37 pinned to
  2b05386425450b304f5006ce8ccfb71dfcfef5a7, after check.py --full
  passed on the merge tree. The design approves no compute; training
  waits for the launch PR and John's yes on it.
- reports/v2_survey/f31/v2_implications.md is tracked by this entry's
  PR, byte-for-byte (-text): SHA-256
  82381a1f1fac16aa8a872b69549906a253c05894cbaf536f22cbcf5c7be08f63, the
  note as John read it (16:23 entry).
- What this does not change: the frozen ensemble stays the reported
  model; the 2025 Atlantic test set stays spent; I4 (no 2026 imagery)
  stands, and its amendment for the test list is asked at the list step;
  H-EP stays sealed under protocol.md 3901f4cf. The snapshots are
  label files only, taken in December and hashed; nothing is downloaded
  by this entry.
- Next: hc-survey folds these answers into the PR #32 draft (the storm
  selection rule stated explicitly, the 2025 rule marked as inferred,
  the 0.48 kt sentence); then the v2 launch PR (driver and tests), to
  hc-adversary and then John.

2026-10-21, 20:38. PR #34 approved with a fail-closed condition, v2
training approved, and PR #32's three decisions: John's reply to the
coordinator's packages (the time is the first clock read after the
reply arrived). No training, evaluation, H9 or H-EP read, and no
compute, by this entry.
- Sources: reports/decisions/PR34.md and reports/decisions/PR32-r3.md
  (the coordinator's packages; untracked; each pre-checked by
  hc-adversary).
- John's reply, verbatim (relayed by the coordinator):
  "PR #34:
  (a) Yes, merge pinned to 01c5a3c, provided the GitHub-only freeze
  check refuses when GitHub can't be reached or read (fail closed).
  Confirm that with a test before merging; if it's missing, add it as
  a tightening commit.
  (b) Yes, train v2 (seeds 30–39) under the conditions you listed.
  Send one status when training and verification finish.
  PR #32: merge pinned to 294d5af; confirm "latest 2026 HURDAT2 file
  on NHC's list at the list step" with the December 2026 backup only if
  there is none; BETTER rule (a) plus your note.
  Run the full check on whichever merges second, as planned."
- PR #32 merged at 20:38 pinned to
  294d5af4ec74a437b00e4ce23e0c7607344e3008, after check.py --full
  passed on its merge tree. This entry's PR records in
  protocol_draft.md: the version and trigger rule as confirmed (§10
  item 2), the decision rule (a) as confirmed (§10 item 4), and the
  coordinator's note beside John's SESOI sentence (§6), as adopted.
- The conditions of (b), from reports/decisions/PR34.md: the launch
  commit is the PR #34 merge plus the pre-launch entry's docs-only
  merge, with nothing else landing in between; every pin in the
  pre-launch entry equals the package's (train.py afceb4b1, the nine
  fc71b98 src blobs, excluded_crops cc9dfcff, lists 12a20a14 /
  ead8004f, run_n5 2ebd8cea, analyze_n5 679e2169); FAST and FULL pass
  on the launch commit. The yes covers retraining a run that fails a
  training-only check (abort-run, design §2), recorded with its failed
  attempt and reported in the freeze PR; any other refusal stops and
  goes back to John.
- PR #34's merge waits for the fail-closed test (a tightening commit)
  and check.py --full on its new merge tree.
- What this does not change: the frozen ensemble stays the reported
  model; the 2025 Atlantic test set stays spent; I4 (no 2026 imagery)
  stands; H-EP stays sealed under protocol.md 3901f4cf; no H9
  evaluation of v2.

2026-10-21, 21:03. v2 recipe pre-launch: the pins the v2 Atlantic training
runs under (reports/v2_survey/v2_recipe/design.md §9 step 3). Nothing is
trained by this entry; training is `run_v2_recipe.py launch` in the main
checkout, at the merge of this entry's docs-only PR, under John's yes and
reports/decisions/PR34.md's conditions.
- Every hash below was read from git on the PR #34 merge commit be2a6f473776876b176e9fa40160518be463207a
  (`git rev-parse be2a6f473776876b176e9fa40160518be463207a:<path>` for the blob id; SHA-256 of the
  `git show be2a6f473776876b176e9fa40160518be463207a:<path>` bytes), not copied from the design. Each
  equals the design's and the driver's pin.
- Design: reports/v2_survey/v2_recipe/design.md, as merged with its
  amendment cc3d537: blob 11089a489764fcff414799c0654c5e03b590a513,
  SHA-256 48aff23cd75d068e404ba3ef0de6a448c9256b58700916f5b2c946afc82b5315.
- Driver: scripts/run_v2_recipe.py (as at f6c039e): blob
  6ae848ea788e5d99d4ca17f76d48e0318e0c6195, SHA-256
  d51d9511c71fe8f648a914409be43cefb33307820df8aa55d11b4659eebb5fb7.
- Imported, hashed before they run:
  - scripts/run_n5.py: blob 2ebd8cea65b996aee6f4acf8ef262f4a65a1c229,
    SHA-256 e8f13e06b80c29101ef1a4efaeac7aaa17e5922f0a8bebdc26a19f270d6d3ed7;
  - scripts/analyze_n5.py: blob 679e21696b6592e36a0ff12ee7d7acb4622dd565,
    SHA-256 0130641583cf09c8abb5791b9ea9467fd6410b953bc8a57c3a2ae68fb3bb1288.
- scripts/train.py: blob 500bbc1e2ff6465dc645138e1904f0819f5b3e96, SHA-256
  afceb4b19ea7d8c732c6f26b7d52e4b3b6d313d58153e2d0518887a3a6db231f.
- src/hurricane_intensity/*.py, the nine blobs at fc71b98 (blob, SHA-256):
  - __init__.py 7a6fe6399245c5e368ffd7f648d8aff589527f21,
    928e490ccd10efccf941662dec33551e54a39233e60a229e758bf91b2cbe93e3;
  - config.py 305d3c4947141fe97fcb3634b3b22d614894a889,
    642caef3efc7564639a20600cbf21062538b4c1c35d3dd37934ef161b949a81a;
  - crop.py 101be8fcc4be79ec9d9523cf2d8107f7eb1075a7,
    25e93d1ab045c7545d939dbf633f9a89365a2fde9cc1c4fe50d74beecd58f8f9;
  - dataset.py 458ed363d48b50b672b69792827cb5b82c6a316b,
    f31f25f01392d4546cdb0b6bdd37317e1cff138e962ff24c0a39663726b2ba13;
  - goes_index.py bba3b98f65838b68028bf545d833e6e70fd78815,
    350a0bb714129928c769dc05413eb5f412fbc3c006021c5cd99cf0b3891b49df;
  - hurdat2.py 11dee801471a6ed5b86a6872bbdf840f6e447dfa,
    a91bd2ab02dc2d05fcb0c4d64018de293516fd7c7d5a4fd7169bf982ecde80ed;
  - pipeline.py 58746c606bd3b3beb3ad5c6d5d271ee4abd11305,
    0db33a3e885367dea65021cfdcf5b6cf595c97446bbf50bb329065ce35a863ac;
  - projection.py 29cc9b6c08ecd93d58c773165b606b8688fb5b6d,
    d4faa06d2a8824582c1c67fa427a28cf3c0736f2c573f160a83204603c8e1d75;
  - splits.py 355f5f63e37b3f31c504c71dce7e745560c7a0f9,
    399b2c1e47f2a5d57eda2cd00ffb68af7a17a5403cba36ff31e0bda0c34c192a.
  No file is added or removed (the folder holds exactly these nine).
- excluded_crops.txt: blob cc9dfcff4036fcbefe5d2ec1a6488fa2feef8f5f, SHA-256
  102009773f843410149f8f8cadeb1b3c981252fcb8d7c53cc8d43e018df00cea.
- Lists: storms_train_expanded.txt blob
  12a20a1499cafa50a753a6f5f7da3f4439c8fa8f (SHA-256
  b94fb5e2f117d56c7f30dca526587a139ecc1a608a0c85920816cb0baf5502ca);
  storms_val_expanded.txt blob ead8004f39712f43f9f787e1851de6092c35abab
  (SHA-256 75b449e0324dbc67dd58e80df3fc43ed561fe6e0d49edfa101bd01df00ec3b90).
- `git diff fc71b98 be2a6f473776876b176e9fa40160518be463207a -- src scripts/train.py excluded_crops.txt
  storms_train_expanded.txt storms_val_expanded.txt` is empty.
- Runs: seeds 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, in that order
  (order 1-10); 99 epochs x 66 batches = 6534 steps each
  (epochs_for(2081, 4 x 1640)); native selection; 2081 train and 940
  validation frames.
- Disclosure (design §4), by line only: while drafting the design's r1,
  hc-survey's search of reports/v2_survey/hep/protocol.md (SHA-256
  3901f4cf41856c40e4c08a533011106ae78067a7301c7e27fbd3e8cd30624c7f) also
  displayed its l.84-88 and l.101. No v2 choice depends on them; every
  recipe item was fixed by F-31 and John. From r2 on, protocol.md was read
  by line range only.
- What this does not change: nothing is read from H9, H-EP or 2026; the
  frozen ensemble stays the reported model; v2 is frozen only by its
  freeze PR (design §4).

2026-10-21, 23:25. v2 Atlantic ensemble frozen
(reports/v2_survey/v2_recipe/design.md §4-§5): ten members, seeds 30-39,
trained at one launch commit and verified. v2 is frozen only when this
freeze PR merges into phase2-experiments; until then it is not frozen.
- Launch commit L: 2b4948e54b001747c9fedbfc26a7e2e6885536b8, the PR #36
  merge (the PR #34 merge be2a6f4 plus the pre-launch entry, nothing else
  in between). Training ran 21:27 to 23:01 in the main checkout; every
  record's commit is L. `run_v2_recipe.py verify` passed: 10 records, no
  tie, no abort. Freeze ran at 23:22.
- Members file: results/v2_recipe/members_sha256.txt, blob SHA-256
  558b781bc87e892b640994cdefc309059fcc786ac861e9ac34850ebb7815ad29
  (SHA-256 of the `git show 4b91baa:results/v2_recipe/members_sha256.txt`
  bytes; the file is LF, so its file hash is the same).
- The 10 members, models/v2_recipe/seed<s>.pt (SHA-256):
  - seed30 b1f819777e40840d9ddbaeeca62f2328611287e8bf47dc70f0476e59784ef2b2
  - seed31 efd5e13808aa22cf490f12530f244044d145d5369c1aa40c22340447a0d95457
  - seed32 e2eaeb571775214c6185324414f1897ebe9663a754ac45c184ffcdf3fd55655c
  - seed33 0f9b429ffde4abd4fe44fb32480d50de3b1b3b89ba53c55b364d9418d4ce7233
  - seed34 01230f9f1630e063e8443b1756ebf6850179251c8978fe1e716b8cf7f9849594
  - seed35 09d79a00230abe8393fb0675169f403600a412b755b42d612cd6a6001ed31841
  - seed36 7a41b2b4d27c613e635c85e07df96ffb6b921698ae647cbff0f015832d87b692
  - seed37 9df9ac91475d88d34a60267d70d1d95d363b8490b2300a43df121dcfdda21f6b
  - seed38 ba87bc056ccdf5debe43d95fa669087d950a65b59b34e907ce9166f952dd305a
  - seed39 fc7301049ef1ea6739660f865778a9e5c0c7e9697a5f3ef693975e976c3b2911
- Train list: storms_train_expanded.txt, blob
  12a20a1499cafa50a753a6f5f7da3f4439c8fa8f (at L). Selection used
  storms_val_expanded.txt, blob ead8004f39712f43f9f787e1851de6092c35abab.
- Aborted attempts: no aborted attempts. The PR commits no
  logs/v2_recipe/aborted/*/abort.json, and every record's
  aborted_attempts is empty.
- Committed with this entry (byte-for-byte, -text): the 10 records
  logs/v2_recipe/runs/seed30-39.train.json, the 10 train logs
  logs/v2_recipe/seed30-39.log and the members file.
- Validation, the selection set, not an estimate (design §3), from the
  verify output: pick epoch / validation MAE at the pick (kt): seed 30
  95 / 8.0034; 31 69 / 8.1579; 32 78 / 8.1747; 33 67 / 7.7087; 34 89 /
  8.0453; 35 88 / 7.7990; 36 84 / 7.9814; 37 67 / 7.8400; 38 89 / 7.9914;
  39 68 / 8.0488. Under-training flag (pick > 0.85 x 99): 4/10, seeds 30,
  34, 35 and 38.
- Every later use passes exactly `--checkpoints models/v2_recipe/seed30.pt
  ... models/v2_recipe/seed39.pt` (ascending) and `--expect-sha
  results/v2_recipe/members_sha256.txt`, and never the members file as
  `--sha-out`.
- What still blocks unfreeze (design §4): while this PR is unmerged,
  `run_v2_recipe.py unfreeze --reason` may still withdraw the members file.
  It refuses, after a `git fetch origin`, once origin/phase2-experiments
  holds the members file or this entry (so for good once this PR merges),
  if the fetch fails, or if any H-EP or clean-test output exists. After the
  merge the members are read-only and are never retrained, fine-tuned,
  re-selected or replaced; a defect found later is reported here, and any
  fix is a new ensemble under a new design.
- What this does not change: nothing was read from H9, H-EP or 2026, and
  no member was evaluated on any held-out set; the frozen ensemble stays
  the reported model. Once this PR merges, v2 is eligible as E3's
  comparator (design §4); E3's design declares it.

2026-10-22, 00:03. PR #37 approved: John's yes to the v2 freeze, and a
gate on the freeze-PR check script. No training, evaluation, H9 or H-EP
read, and no compute, by this entry.
- Source: reports/decisions/PR37.md (the coordinator's package;
  untracked; pre-checked by hc-adversary).
- John's reply, verbatim (relayed by the coordinator):
  "PR #37: yes, merge pinned to 2db0ccc. v2 is frozen. After the merge,
  move the main checkout back to phase2-experiments without touching
  files and confirm the members file hash 558b781b… and all ten
  checkpoint hashes still match. Log the six freeze-check-script gaps as
  a gate: fixed and adversary-reviewed before that script is reused or
  any re-freeze."
- PR #37 merged at 00:03 pinned to
  2db0ccca09c9838079ce10f8fe9274faec77b963, as 8514451 (parents 2b4948e
  and 2db0ccc), after check.py --full passed on its tree. v2 is frozen:
  the "2026-10-21, 23:25. v2 Atlantic ensemble frozen" entry and
  results/v2_recipe/members_sha256.txt are on phase2-experiments, so
  run_v2_recipe.py unfreeze refuses from now on.
- After the merge: the main checkout was moved from v2-freeze to
  phase2-experiments by `git fetch origin
  phase2-experiments:phase2-experiments` and `git switch
  phase2-experiments`. The SHA-256, modification time and mode of all 31
  v2 files (21 committed outputs, 10 checkpoints) were identical before
  and after. The members file's SHA-256 on disk and in git is
  558b781bc87e892b640994cdefc309059fcc786ac861e9ac34850ebb7815ad29, and
  `sha256sum -c` of it passes for all ten models/v2_recipe/seed30-39.pt,
  which are read-only.
- Gate (John): the freeze-PR check script (uncommitted; SHA-256
  05261b043f17e37cdb739117dcd86a7a62517a866260fb97ca52dd01d5d057d1) is
  not reused, and no re-freeze happens, until its six gaps are fixed and
  hc-adversary has reviewed the fix. The gaps (hc-adversary item 37;
  none is present in PR #37, each checked directly):
  - M1: the allowed-path pattern admits any path under logs/v2_recipe/
    and results/v2_recipe/; anchor it to the design §5 forms.
  - M3: the entry-title match allows text before the title; require the
    title right after the timestamp.
  - M4: the ten member SHA-256s are checked as substrings, so swapped
    seed-hash pairs pass; compare parsed pairs with the members file.
  - M5: an aborted/ directory without an abort.json passes; require
    one in every aborted directory, committed or on disk.
  - M6: it checks only that the launch commit is an ancestor of the
    merge-base; require the merge-base to be the launch commit, or the
    difference to be decisions docs only.
  - M7: check-attr reads the working tree's .gitattributes, and no
    clean tracked tree is asserted; use the index or assert a clean
    tree.
- What this does not change: the frozen ensemble
  (models/storms/expanded/seed0-9.pt) stays the reported model; main
  stays at fbb39c4; no v2 member has been evaluated on any held-out set;
  v2 meets held-out data only in the 2026–2027 clean test, under its
  protocol, committed before any clean-test build and before the H-EP
  pass.

2026-10-26, 18:15. Preprocessing pinned for evaluation, and the v2
pin test: John's decisions. No training, evaluation or compute by this
entry.
- John, verbatim (relayed by the coordinator): "Agreed on the pin-test
  change. Add to the 2026–2027 protocol (docs-only, you may merge): v2
  and the frozen ensemble are always evaluated with preprocessing from
  their pinned commits, or the run must first prove byte-identical crops
  between pinned and current code for every evaluated storm. Same rule
  for E3's comparator. Tell me when B-v2-2 merges so I can rebase
  fix-antimeridian."
- The pin-test change (B-v2-2's PR): run_v2_recipe.py's SRC_BLOBS stay
  the fc71b98 pins; test_pins_match_the_committed_blobs asserts them
  against fc71b98, not HEAD. With src changed, launch still refuses
  (guard 2; guard 8, since v2 is frozen).
- The preprocessing rule is recorded in
  reports/v2_survey/2026_clean_test/protocol_draft.md §1 and §10 by
  this entry's PR.
