# H-EP evaluation protocol (DRAFT r2, not approved, nothing run)

r1 drafted 2026-10-18 by hc-survey. r2 is the same day, after hc-adversary item 8 (14 findings) and the coordinator's four corrections. It is modelled on the 2025 test's pinned design (decisions.md 2026-10-15 22:57 / 22:59 / 23:05; commits f9861bf, 115fcea). It goes to hc-adversary and then John.

**Nothing has been run:** no download, build, list, model or git write.

**What has been computed** (`reference_geometry.py`, read-only, from manifests and HURDAT2 positions): the reference viewing-angle tables, and H-EP frame and concurrency estimates.

**Changed in r2:**
- §1: "once" enforced outside the driver; the blob hash; an E3 condition; `best_model.pt` recommended out.
- §2: per-evaluation VZA references with the p95; all pooled figures exclude ≥ 137 kt, and the leak test is defined.
- §3: the bug-rerun clause constrained.
- §4: gates reordered and added (PR #4, PR #5, gate 3 before the N5 launch, spend in code as gate 9, adversary review before gate 5); geometry and domain shift moved to immediately before the pass.

**Changed in r3** (adversary 8b):
- §2d: "no pooled table alone" wording.
- §4: gates renumbered into execution order (the pass is gate 8; spend in code is gate 9).
- §5 and gate 7: cite PR #4's `--processed-root` and carry-forward work.
- §6: the wrote-line regex allows ensemble_eval's ` and <sha_out>` suffix.
- §5: West rule fixed; sealed crops.
- §6: linkage fix; pinned paths.
- §8: write-up checklist.

**H-EP (N1, Option A, John):** EP052025 Erick, EP062025 Flossie, EP112025 Kiko, EP162025 Priscilla, EP112023 Jova, EP182023 Otis. All six are **excluded from any training list** (I5).

**Estimates** from the pinned nepac HURDAT2 (`hurdat2-nepac-1949-2025-092926.txt`, SHA-256 `caece626…a0507`), via `parse_storm` + `build_sample_grid(1h)` (`hep_estimates.csv`):

| ATCF | Name | Peak kt | Est. hourly frames | > 96 kt | ≥ 137 kt |
|---|---|---|---|---|---|
| EP052025 | Erick | 120 | 74 | 20 | 0 |
| EP062025 | Flossie | 105 | 157 | 15 | 0 |
| EP112025 | Kiko | 125 | 265 | 100 | 0 |
| EP162025 | Priscilla | 100 | 181 | 7 | 0 |
| EP112023 | Jova | 140 | 229 | 42 | 10 |
| EP182023 | Otis | 145 | 116 | 18 | 7 |

**This table is H-EP label information** (peaks, intensity counts), like the 2025 record's (decisions.md:1401 vs 1894). **E3's design may not use it,** nor any other H-EP label or pixel information (§4 gate 6).

- **"Spent 2025" means only the six Atlantic test IDs** (AL042025, AL052025, AL072025, AL082025, AL092025, AL132025).
- **Not in H-EP:** CP012025 Iona (CP ID); Dora EP052023 (the GOES-18 regression scan; training-only).

**The question.** How do frozen Atlantic-trained models, and the later Atlantic + East Pacific arm, transfer to East Pacific storms that no model has seen, imaged by GOES-18? H-EP is evaluated **once, in one declared pass**. It is then spent, **in code** (gate 9).

## 1. Which checkpoints may ever touch H-EP

Only the evaluations declared below. Nothing else is ever evaluated on H-EP: no other checkpoint, no rerun except under §3, no later addition.

| ID | Checkpoints | Frozen by (hashes) | Train list (refusal) | Condition |
|---|---|---|---|---|
| **E1** | Frozen Atlantic ensemble, `models/storms/expanded/seed0-9.pt`, 10 members (the transfer baseline) | `results/ensemble/members_sha256.txt`. **Blob** SHA-256 `3a2dac2303bf…623d` (980 bytes, LF, committed in d22fc48), computed in Python from the bytes of `git show`. The working-copy hash `f27e2e9d…b5ce` (990 bytes, CRLF under autocrlf) is checkout-dependent and recorded for reference only. Member hashes as in the COORDINATOR.md protected block (e.g. seed0 `316b5118…bdfa`). | `storms_train_expanded.txt` | frozen |
| **E2c** | N5 control: the 10 matched-grid selected checkpoints | `results/n5_atlantic/members_control_sha256.txt`, written from the N5 run manifests' `selected_checkpoint_sha256`; blob hash recorded at commit | `storms_train_expanded.txt` | **only if N5 is closed with a results entry** before the pass; otherwise the slot closes |
| **E2a** | N5 atlantic arm: the 10 selected checkpoints | `results/n5_atlantic/members_atlantic_sha256.txt`, the same way | `storms_train_n5_atlantic.txt` | same as E2c |
| **E3** | The later Atlantic + East Pacific arm, plus its comparator if its design declares one | its design entry: checkpoint hashes (blob-hashed members file) and train list | its list | **its design entry must be committed, and its checkpoints frozen, by a date John sets when this protocol is committed. If that date passes, the slot closes.** |

Rules:
- **One pass** (recommended; the coordinator and the adversary agree). All declared evaluations run in **a single `run_hep.py --stage pass` invocation**, once every slot that will run is frozen.
  - An early E1 would let H-EP results shape E3's design.
  - E1 and E2c are the same configuration (frozen vs retrained expanded), so an early E1 would also preview E2c.
  - Any slot not frozen at the pass is **closed for good**; H-EP is spent afterwards.
- **Ensembles:** each evaluation is the unweighted mean of its 10 members per frame (`ensemble_eval.py`), with the members' own predictions kept. There is no recalibration, weighting, member selection or test-time augmentation. Preprocessing is `dataset.py`'s, unchanged.
- **Hashes are rechecked:** `ensemble_eval.py --expect-sha` refuses any other checkpoints. Its comparison reads with universal newlines, so it is unaffected by autocrlf. `--sha-out` records the hashes at run time.
- **`best_model.pt`: recommended OUT.**
  - It adds nothing E1's members don't: the member mean, sd and range already show single-model spread.
  - It is the 9.00 kt "good draw" (decisions.md:305-314), a misleading single reference.
  - If John wants continuity with the 2025 table, it must be declared before gate 5, never after.
- **How "once" is enforced, not just promised** (item 8, finding 1):
  - **(a) The pass driver refuses any existing output directory.**
  - **(b) Sealed crops.** Between the build and the pass, the H-EP crops live outside `data/processed`, in `data/hep_sealed/`. They are reachable only through `--processed-root data/hep_sealed` (`ensemble_eval.py:27`; `domain_shift.py`, and the geometry step, also take a root). An accidental `TCDataset("data/processed", ...)` finds nothing.
  - **(c) Results pinned.** `results/hep/` is committed at the pass. `check.py` pins it like `results/final_2025/`: a clean `git status`, and an empty `git diff <pass commit> HEAD -- results/hep`. There is a new FEATURES.md row, **I3b**.
  - **(d) Spent in code** (gate 9). After the pass, the six H-EP IDs join the spent set in TCDataset through a **registry role** (e.g. `spent`), not a second hard-coded list. The tests are the same as PR #3's / PR #5's, including the item 6 aliases.

## 2. Metrics, fixed now

There is no inferential test. As in 2025: six storms of autocorrelated hourly frames, and one ensemble per evaluation, give no sampling distribution. Everything is descriptive.

**2a. Viewing angle.**

**Reference per evaluation** (finding 8b). Each evaluation's VZA note uses **its own training set's** distribution, reporting both the **p95 and the max** at > 96 kt (finding 8a). Computed now (`vza_reference.csv`; spherical earth, manifest lat/lon, lon_0 −75.0):

| Training set | Frames | Median | p5–p95 | Max | Share > 45° |
|---|---|---|---|---|---|
| E1 / E2c: expanded, minus val | all 2081 | 34.7° | 23.3–49.6° | 67.0° | 8% |
| E1 / E2c | > 96 kt, 394 | 33.0° | 29.3–**42.7°** | **45.4°** | 0% |
| E2a: expanded + 31 N5 storms (the new storms estimated from HURDAT2 3h positions) | all about 4083 | 37.6° | 22.6–58.5° | 75.6° | 24% |
| E2a | > 96 kt, about 402 | 33.1° | 29.3–**42.6°** | **45.4°** | 0% |
| E3 | computed from E3's training manifests immediately before the pass | | | | |

For comparison: validation all frames median 34.4° (max 51.5°); H9 all frames 32.0° (max 63.2°), H9 > 96 kt 31.0° (max 46.2°).

**H-EP, estimated from best-track positions** (GOES-West lon_0 −137.0; already public, not pixel data):

| Storm | Median VZA, West (all / > 96) | Max West |
|---|---|---|
| Erick 2025 | 49.7° / **48.6°** | 52.6° |
| Flossie 2025 | 39.0° / 39.8° | 46.0° |
| Kiko 2025 | 20.6° / 17.4° | 38.0° |
| Priscilla 2025 | 38.9° / 38.0° | 40.0° |
| Jova 2023 | 30.3° / 32.2° | 41.9° |
| Otis 2023 | 47.5° / **46.4°** | 48.6° |

Erick 2025's and Otis's > 96 kt frames lie beyond both the p95 (42.7°) and the max (45.4°) of E1's/E2's high-end training angles. For those evaluations, the two storms test **viewing-angle extrapolation at the high end** as well as basin transfer. E3's reference will differ: EP GOES-West training frames reach about 48–52° (John 2024, Dora). Each per-storm row words its note per evaluation.

**Method.** The VZA uses spherical earth, `atan2(sin g, cos g − R/(R+H))` with R = 6378.137 and H = 35786. The adversary checked it: ellipsoid effects are a few tenths of a degree.

**The built-geometry table** (per storm, from the built manifests' lat/lon, the platform in `source_key`, and the files' lon_0) is computed **immediately before the pass** (gate 8), not after the build.

**2b. Per storm, for each evaluation** (all frames of that storm, including its own ≥ 137 kt frames):
- frames, MAE, fixes-only MAE, bias;
- VZA median/min/max and satellite;
- the share of frames above that evaluation's > 96 kt training p95 and max;
- the members' mean/sd/range of storm MAE.
- **The 2025 rule:** > 96 kt MAE, bias and Spearman only for storms with **≥ 10 frames > 96 kt**. Priscilla has an estimated 7, so it gets none, and that is stated rather than filled in.

**2c. Jova and Otis at the high end, individually, never averaged (John):**
- each storm's own > 96 kt row and its own ≥ 137 kt row (MAE, bias, n; estimated Jova 10, Otis 7);
- every ≥ 137 kt frame listed. Columns: time, best track, ensemble prediction, members' min/max, VZA, fix or interpolated.

**2d. Pooled (all six storms): EVERY pooled figure is computed over frames < 137 kt only**, and labelled "**excl. ≥ 137 kt**" (finding 6, option a):
- MAE, fixes-only MAE, bias, < 64 kt MAE and bias;
- per-band MAE/bias/n for TD, TS and Cat 1–4;
- MAE over TD–Cat 4 reweighted to H9's band shares;
- the members' mean, sd and range on each.

**What is never computed or published by `hep_report.py`:**
- **the pooled MAE or bias over all frames;**
- a pooled > 96 kt figure;
- a pooled Cat 5 figure.

**Notes on what can still be derived:**
- **No Cat 5 residual from the pooled tables alone:** every pooled figure excludes ≥ 137 kt frames, so **no pooled table alone** lets a Jova + Otis Cat 5 average be recovered by subtraction. `hep_report.py`'s tests check exactly that: the residual (N·MAE − Σ n_b·MAE_b over the published pooled bands) must come out to zero frames, or not be computable.
- **What this does not claim:** that the Cat 5 average cannot be derived at all. 2b publishes each storm's all-frames MAE, bias and n. So the pooled all-frames figure (Σ n_s·MAE_s / N), and from it the Cat 5 residual, can be reconstructed from pooled plus per-storm rows. Likewise the two individual Cat 5 rows can be averaged by hand (below). The rule is about what is computed, tabulated or quoted.
- **Cat 3 + Cat 4:** combining the pooled rows gives a 96–137 kt figure, which excludes all Cat 5 frames. It is not computed or quoted as "> 96".
- **Stated plainly:** Jova's and Otis's Cat 5 rows are published individually (John's requirement), so anyone can average the two by hand. The rule is that no such average is computed, tabulated or quoted (§8).
- **Raw records are not reports:** the raw per-frame CSVs and `ensemble_eval.py`'s stdout contain all frames. Its stdout prints an all-frames "ensemble overall MAE" (ensemble_eval.py:80). They are kept as records for the linkage checks, never quoted.

**2e. Domain shift.** Run `scripts/domain_shift.py` (frozen; working-file SHA-256 `d439e736…c7c3b4` as recorded 10-15, and its blob hash recorded in the protocol PR), H-EP vs H9. It computes per-band mean BT, sd, p5 and share < 200 K, read against the H3-vs-H9 reference (about 8 K), as in 2025.
- **Timing:** immediately before the pass, after every declared slot is frozen (finding 9), never earlier.
- **Bands:** its band rows stop at Cat 4 under the 2d rule; Jova's and Otis's Cat 5 rows are per storm.

**2f. Between evaluations (descriptive).**
- **Pooled, excl. ≥ 137 kt:** MAE, fixes-only MAE, bias, < 64 kt MAE/bias, and per-band MAE/bias for TD–Cat 4. Compared as E2a − E2c, as E3 against its comparator, and as each evaluation's transfer gap (H-EP − its own H9 values over the same < 137 kt bands; E1 from `results/ensemble/expanded10_h9.csv`, E2 from the N5 outputs).
- **Per storm:** the 2b rows side by side, and the 2c rows side by side per storm, never averaged.

## 3. Reading, fixed now

- **Degradation is expected:** a new basin, a new satellite, smaller storms, and VZA beyond training for two storms. The protocol reports how much and where (storm, band, VZA), not pass or fail.
- **Otis is a near-coastal rapid intensifier;** Kiko is a near-nadir open-ocean storm. Neither is called representative.
- **Validation is blind at Cat 5,** and the Cat 5 frames are Jova's and Otis's alone (2c).
- **If the result is poor, it is still the result:** no retraining or tuning on H-EP, no second evaluation, no change of members, averaging, preprocessing, build, lists or bucket. H-EP is never used for model or design selection afterwards.
- **A bug found after the pass** (finding 2):
  - only **evaluation or report code** may change;
  - the fix is reviewed by hc-adversary **before** the rerun;
  - predictions the bug did not touch must reproduce **bit for bit**;
  - the first run stays in the record, and both runs are reported with the reason.

## 4. Gates, in order (each committed before the next; none done)

1. **B-v2-1: antimeridian fix.** `build_sample_grid` unwraps longitude, or tracks are trimmed at ±180 (Dora). No H-EP storm crosses (−159.5 to −92.4), but every EP build waits for it.
2. **B-v2-2: East Pacific HURDAT2 cache.** It reads the pinned nepac file (SHA-256 `caece626…a0507`) from a basin-specific path. It can never write or refresh `data/interim/hurdat2_atlantic.txt`, which is protected (`ee35819c…0bcb5c`) and hashed before and after. `--force-hurdat2-refresh` is never used.
3. **H-EP list and guards, before or INSIDE the N5 launch PR** (finding 10). `storms_train_n5_atlantic.txt` is the first v2 train list, so all of the following land with it at the latest:
   - **the list:** `storms_holdout_hep.txt` (six IDs);
   - **its I5 forbidden-set entry** (`tests/test_list_invariants.py`) and **its registry role**;
   - **the F-10 alias/concurrency check** run against it. Today it finds Priscilla's 650 km box overlapping EP152025 Octave (6 h) and EP172025 Raymond (16 h) (`hep_concurrency.csv`). Those two 2025 EP storms are **excluded from any training list** (N1 keeps 2025 EP out of training). Flossie and AL022025 Barry come no closer than 844 km and never overlap.
4. **PR #5: TCDataset 2025 block** (John, 23:57). A worker writes it after PR #3 merges, by ATCF ID only, for the six Atlantic IDs (AL042025, AL052025, AL072025, AL082025, AL092025, AL132025). It does not use `*2025` or a year rule. Its test asserts that EP052025 / EP062025 / EP112025 / EP162025 are **not** blocked.
   - **Merge order:** PR #2 → PR #4 (the check_nan-safe exclusion step) → PR #3 → PR #5.
   - Checked read-only (r1): `n6-required-args` had no 2025 block.
5. **Adversary review of the protocol PR** (finding 11): `run_hep.py` and `hep_report.py` (with its synthetic tests, including the 2d residual test) are reviewed by hc-adversary before gate 6.
6. **This protocol committed** as a decisions.md entry. It includes:
   - the members files' blob hashes (E1 now, E2 at N5 close);
   - `run_hep.py`, `hep_report.py` and `domain_shift.py` blob hashes;
   - `vza_reference.csv` and its SHA-256;
   - the West rule (§5);
   - E3's deadline;
   - the statement that **E3's design may not use any H-EP label or pixel information**, including this file's estimates table.
7. **Download and build into `data/hep_sealed/`** (§5), plus the exclusion step for the six storms only. It runs through PR #4's check_nan step extended with `--processed-root` and **carry-forward of entries for storms absent from the scanned root** (assigned to hc-reviewer, adversary 8b finding 15). Without the carry-forward, every later `check_nan.py` run over `data/processed` would treat the H-EP entries as removed and exit 1. **No model, no geometry and no domain shift yet.**
8. **The pass, when John calls it,** with every declared slot frozen, E2 conditional on N5's results entry and E3 by its deadline. In order:
   - the built geometry table and the per-evaluation VZA references (2a);
   - the domain shift (2e);
   - every declared evaluation;
   - `hep_report.py`;
   - commit `results/hep/`;
   - then gate 9.
9. **Spend in code, after the pass:** the six IDs join TCDataset's spent set via the registry role, with the PR #3 / #5 tests and the item 6 aliases. PR #4's carry-forward then takes the spent set from the same registry role (adversary 8b, 15c). The FEATURES.md I3b row and the `check.py` pin of `results/hep/` are committed in the same step.

## 5. Build, exactly

- **Storms:** the six H-EP IDs, from `storms_holdout_hep.txt`.
- **Bucket: `noaa-goes18` for all six.** This is a fixed rule: John approved N2 at 22:47, GOES-West consistently for every boundary storm. Erick 2025 and Otis come from GOES-18 like the rest. All six have GOES-18 operational coverage: Jova and Otis in Sep–Oct 2023, after the 2023-01-04 18Z handover; the 2025 storms on GOES-18.
- **Settings identical to H9 except the bucket, the HURDAT2 source and the root:** hourly grid plus every exact fix; ABI-L2-CMIPF band 13 full disk; 600 km; 10 min tolerance; statuses LO,TD,SD,TS,SS,HU; **`--out-root data/hep_sealed`**.
- **Every scan key** must be dated ≤ 2025 (I4), and is recorded in `source_key`.
- **Missing scans stay missing.** Nothing is re-cropped or rebuilt after any prediction is seen.
- **Exclusions:** PR #4's check_nan step, run with `--processed-root data/hep_sealed` for the six H-EP storms only. It carries forward entries for storms absent from the scanned root, so later runs over `data/processed` keep the H-EP entries. Both are being added to PR #4 by hc-reviewer; they are not existing modes.
  - It writes to a temp file, then `os.replace`.
  - The diff may only add entries under the six H-EP IDs.
  - The file is committed. The rule is unchanged: any NaN pixel excludes the frame.

## 6. Paths and commands

**Never written:**
- `best_model.pt`, `results/heldout_predictions.csv`;
- anything under `models/`, `data/processed/`, `results/storms/`, `results/ensemble/`, `results/final_2025/`, `results/n5_atlantic/` (read only), or `site/`.

**New:**
- `data/hep_sealed/<ID>/`;
- `results/hep/geometry/`, `results/hep/domain_shift/`;
- `results/hep/E1_frozen_expanded/`, `results/hep/E2c_n5_control/`, `results/hep/E2a_n5_atlantic/`, `results/hep/E3_<name>/` (only if declared);
- `logs/hep/` and `logs/hep/pass_manifest.json`.

**One Python driver, `scripts/run_hep.py`** (specification only here; written in the protocol PR and reviewed at gate 5). Behaviour as in N5 §8, **including N5 r2's A1 linkage fix** (finding 5):
- **Calls:** `subprocess.run` with list arguments; stdout/stderr written as UTF-8 bytes without BOM; `RUN` / `START` / `END` / `FAILED` lines.
- **Refusals:** any existing output directory is refused.
- **Commit:** a single commit for the whole pass.
- **Protected hashes:** the COORDINATOR.md protected block, hashed before and after.
- **`logs/hep/pass_manifest.json`** records:
  - the commit;
  - the blob hashes of `run_hep.py`, `hep_report.py`, `ensemble_eval.py` and `domain_shift.py`;
  - every output's path and SHA-256.
- **Pinned output paths per evaluation,** as listed below. `hep_report.py` refuses any other path.
- **Each CSV is content-linked to its eval log.** `ensemble_eval.py` prints every member's MAE to 3 decimals (line 74), the sample count (line 81), and `wrote <out> (<n> rows) and <sha_out>` (line 95). The regex allows that trailing ` and <sha_out>`: `^wrote (\S+) \((\d+) rows\)(?: and (\S+))?$`, and the sha path must equal the pinned `members_sha256_at_run.txt`. `hep_report.py` recomputes each member's MAE from the CSV's `pred_<member>` columns, and requires every member line, the sample count and the wrote-path and rows to match.

There is no shell redirection and no brace patterns. `PY` = `sys.executable`; paths are relative to the repo root.

**Build stage** (`--stage build`, gate 7). Per ID:
```
[PY, scripts/build_dataset.py, --atcf-id, <ID>, --bucket, noaa-goes18, --out-root, data/hep_sealed,
 --hurdat2-path, <B-v2-2 nepac cache path>]
    -> logs/hep/build_<ID>.log
```
Then PR #4's exclusion step for the six IDs over `data/hep_sealed`, and both HURDAT2 hashes.

**Pre-pass stage** (`--stage pre`, the first steps of gate 8, after every slot is frozen):
```
[PY, scripts/hep_report.py, --geometry-only, --processed-root, data/hep_sealed, --holdout-list, storms_holdout_hep.txt,
 --training-lists, <each declared evaluation's train list>, --out, results/hep/geometry/hep_vza.csv,
 --reference-out, results/hep/geometry/vza_reference_by_evaluation.csv]
[PY, scripts/domain_shift.py, --processed-root, data/hep_sealed, --set, HEP, storms_holdout_hep.txt,
 --out, results/hep/domain_shift/domain_shift_hep.csv]
[PY, scripts/domain_shift.py, --processed-root, data/processed, --set, H9, storms_holdout_h9.txt,
 --out, results/hep/domain_shift/domain_shift_h9.csv]
```
`domain_shift.py` takes one root per call, so it runs twice; `hep_report.py` joins the two and applies the < 137 kt rule.

**The pass** (`--stage pass --evaluations <the declared, frozen set>`). Per evaluation `<E>`, with members `<m0> … <m9>`:
```
[PY, scripts/ensemble_eval.py, --checkpoints, <m0>, ..., <m9>, --expect-sha, <members file>, --train-list, <train list>,
 --holdout-list, storms_holdout_hep.txt, --processed-root, data/hep_sealed,
 --out, results/hep/<E>/ensemble_hep.csv, --sha-out, results/hep/<E>/members_sha256_at_run.txt]
    -> logs/hep/<E>_eval.log
[PY, scripts/hep_report.py, --csv, results/hep/<E>/ensemble_hep.csv, --eval-log, logs/hep/<E>_eval.log,
 --geometry, results/hep/geometry/hep_vza.csv, --h9, <E's H9 per-frame CSV>, --out, results/hep/<E>/hep_metrics.csv,
 --frames-out, results/hep/<E>/jova_otis_ge137_frames.csv]
```
Then `hep_report.py --compare` over the declared evaluations writes `results/hep/comparison.csv` (2f).

**Train lists for E2:** `storms_train_expanded.txt` (E2c) and `storms_train_n5_atlantic.txt` (E2a).

**`scripts/hep_report.py`** (new; specification only here; written, tested on synthetic data, reviewed at gate 5 and blob-hashed at gate 6; no model, no torch). It computes exactly §2b–2d and 2f. It **refuses**:
- any storm outside H-EP, and any Atlantic 2025 ID;
- a CSV whose frame set differs from the geometry table;
- a path that is not the pinned one;
- an eval log that does not content-match its CSV.

**It never computes a pooled figure over frames ≥ 137 kt.** Its tests include the 2d residual test: from every published pooled table, the recovered "remaining frames" count must be 0.

## 7. Open items for John

- **Pass timing:** a single pass (recommended; the coordinator and the adversary agree).
- **`best_model.pt`:** recommended **OUT**. If John wants it for continuity with the 2025 table, it must be declared before gate 5.
- **E3's deadline** (§1): a date set when the protocol is committed.

## 8. Write-up checklist (hc-adversary reviews it against the draft results entry before it is committed)

- [ ] No prose, table or caption gives a pooled, averaged or combined high-end figure for H-EP. Jova and Otis appear at the high end only individually.
- [ ] Every pooled figure says "excl. ≥ 137 kt". The all-frames pooled MAE or bias is not quoted, including from `ensemble_eval.py`'s stdout.
- [ ] Each evaluation's VZA note uses that evaluation's own training reference (p95 and max).
- [ ] Priscilla's missing > 96 kt statistics are stated, not filled in.
- [ ] Every evaluation that ran is reported, and every slot that closed is named as closed.
- [ ] If the bug clause was used, both runs are reported, with the review and the bit-for-bit check.
- [ ] Gate 9 (spent in code), I3b and the `check.py` pin are committed.

## Files in this draft

- `protocol.md`: this file.
- `reference_geometry.py`: read-only. Run with `.venv\Scripts\python.exe -I -B reports/v2_survey/hep/reference_geometry.py . reports/v2_survey/_scratch/hurdat2-nepac-1949-2025-092926.txt`. It reads manifests (lat/lon/source_key; no crop is opened), the pinned Atlantic HURDAT2 and the scratch nepac file, and asserts both hashes. r2 adds the E2a training reference.
- `vza_reference.csv`, `hep_estimates.csv`, `hep_concurrency.csv`.
