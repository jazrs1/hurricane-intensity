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