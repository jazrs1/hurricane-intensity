# 2026 clean test: protocol (DRAFT r2, after hc-adversary item 32; not approved, nothing built or run)

**Status:** a draft for John (his decision 4, 2026-10-21). I4 is unchanged now: no 2026 imagery is downloaded, listed or cropped, and no 2026 label file is downloaded. §8 states the one I4 amendment the test would need later, at step 4, with John's yes.

**Inputs:**
- **2026 counts:** from hc-reviewer's fact sheet, `label_facts.md` (this folder). The season counts are **as of 2026-10-21 20:27 UTC**, from the operational b-decks.
- **That sheet is 2026 label information** (peaks and > 96 kt counts), so this draft is written by someone who has seen the 2026 peaks (§3, disclosure).
- **Route names follow the sheet:** **A** = operational ATCF b-deck, **B** = per-storm TCRs, **C** = annual HURDAT2 update.

## Read first: the 2026 Atlantic season is small, and how much that matters depends on an unknown spread

**As of 2026-10-21, the Atlantic has 9 storms** (AL01–AL09 2026).
- All 9 have tropical-storm status rows.
- Only **Isaias AL092026** went above 96 kt (operational peak 105 kt). Every other storm peaked at ≤ 60 kt (`label_facts.md` §3).
- The season runs to Nov 30.

**What each list rule gives** (operational labels, unbuilt; scan availability can't be checked under I4):

| List rule | 2026 Atlantic storms | > 96 kt storms | The primary (≥ 6 storms needed, §5) |
|---|---|---|---|
| **The 2025 test's inferred rule:** peak ≥ 64 kt in any row (sheet §4; never written down in 2025) | **1** (Isaias) | 1 | **not testable:** descriptive only |
| **This draft's rule (since r0, §3):** at least one TS or HU fix, ≥ 12 built frames | **up to 9.** Each has ≥ 24 h of included-status fixes, so about 25 or more hourly label times before scan and NaN drops (item 32) | 1 | testable if ≥ 6 survive the build |
| Either rule: the > 96 kt secondaries | — | 1 | descriptive only (they need ≥ 6 storms with ≥ 10 frames > 96 kt) |

**How much the small n matters depends on the between-storm sd of the paired difference, which is unknown** (§6).
- **The only anchor:** H9's per-storm spread, 0.47 kt (95% interval about 0.32–0.90; H9-derived; likely an underestimate for 2026).
- **If the sd is near 0.5 kt:** 2026 alone (n ≈ 8–9) is **well powered** for Δ = 1.0 kt: 0.87 at n = 6, 1.00 at n = 8.
- **If the sd is 1 kt or more:** 2026 alone is underpowered (0.35–0.80 at n = 6–10 for Δ = 1.0). A NULL (inconclusive) reading is then the most likely result unless v2's gain is large.
- **Power here is P(p < 0.05).** The chance of reading BETTER is at most that.
- **The high end:** under any option with 2026 alone, the > 96 kt questions, where the frozen ensemble's known weakness lies, can only be descriptive. There is one such storm, and it sits one 5 kt step above the line.

**What this means for John's decision 1** ("confirmed only on the 2026 clean test"):
- Under the 2025-style rule, 2026 cannot confirm anything.
- Under the all-storm rule, 2026 can confirm or fail to confirm v2 on mean per-storm MAE over mostly weak storms. Whether the test is decisive depends on the unknown sd.

**Options** (presented neutrally):

| Option | What it is | For | Against |
|---|---|---|---|
| **1. All-storm rule, 2026 only** | §3's rule; the primary on up to about 9 Atlantic storms | keeps decision 1's wording and timing (pass about Feb–Jun 2027, §8); well powered if sd ≈ 0.5 | underpowered if sd ≥ 1; the population is mostly weak storms; the high end is descriptive only |
| **2. 2026 descriptive only; confirmation deferred** | 2026 reported without a test or replacement decision; confirmation on 2027 storms under this protocol | no test spent on a small set | 2026 is used without testing; v2 is unconfirmed until 2027 labels |
| **3. Pooled 2026 + 2027, one pass** | the all-storm rule over both seasons, one pass after the 2027 labels | the n is robust to the unknown sd (below); more hurricanes in the population | changes decision 1 to "confirmed on the 2026–2027 clean test"; about one year's delay; needs 2027 protections at commit (§8) |
| **4. Add 2026 East Pacific storms** | the 9 EP-ID storms > 96 kt join the test | many high-end storms | mixes basin and GOES-18 transfer into a recipe comparison, and spends E3's natural confirmation set (§4) |

**Pooled n, with the basis stated:**
- 2026 has 9 candidates.
- The NOAA 1991–2020 normal Atlantic season has about 14 named storms, 7 hurricanes and 3 major hurricanes.
- So option 3 would plausibly reach n ≥ 16 before build drops. At n = 16, power for Δ = 1.0 is 0.96 at sd 1.0 and 0.69 at sd 1.5.
- For the > 96 kt secondaries, about 3 major hurricanes in a normal 2027 plus Isaias would still likely fall short of 6 qualifying storms. **Option 3 does not reliably make the high end testable either.**

**Recommendation, label-informed and model-blind: option 3.**
- **The reason is robustness to the unknown sd.** Option 1 is decisive only if the sd is near its one H9-derived anchor. Option 3 keeps power at about 0.7–1.0 for Δ = 1.0 up to sd 1.5 kt, which is this draft's own grid cap, not a known bound; at sd 2.0 and n = 16 it is 0.47. Its population also has more hurricanes, though it is still mostly sub-hurricane: about 9 + 14 named storms, with about 8 hurricanes.
- **For comparison at sd 1.0 and Δ = 1.0:** option 1 at n ≈ 9 has power of about 0.74, against 0.96 for option 3 at n = 16. The 0.74 is interpolated between `power_output.txt`'s n = 8 (0.67) and n = 10 (0.80) rows, since n = 9 is not in the grid; the 0.96 is its n = 16 row.
- **Its cost is a year, and changing decision 1's wording.**
- **Option 1 is the better choice** if John weighs 2026 timing above robustness, or judges the sd likely near 0.5. If the sd is in fact about 0.5, option 3 buys little power for its year.
- **This recommendation was formed after seeing the 2026 counts.** The counts are model-blind (labels, not predictions), and John asked for them. Whichever option John picks, the test stays valid for its pre-registered population, because the list rule is applied mechanically and nothing is tuned to any storm (§3).

**What carries over under any option:** the list rule, the label route and snapshot (§2), the SESOI and the EP decision.

## 0. What the test is

**The question:** on Atlantic storms that no model has seen, does the v2 Atlantic ensemble's **mean per-storm MAE** differ from the frozen ensemble's?
- v2 Atlantic is the 4× control recipe, with 10 new members (`reports/v2_survey/v2_recipe/design.md`).
- The frozen ensemble is 1×: `models/storms/expanded/seed0-9.pt`.

**Why only an unspent season can answer it:**
- The v2 recipe was chosen on H9, through F-31's P1.
- The 2025 set is spent.
- H-EP is an East Pacific transfer test with its own protocol (`reports/v2_survey/hep/protocol.md`, 3901f4cf).

**How it runs:**
- **One declared pass,** and the set is then spent in code (§8).
- **What the result decides:** whether v2 replaces the frozen ensemble, by the rule in §5.
- **Nothing is fitted to test data.** Nothing is retrained, re-selected or tuned on it.

## 1. The two models (both frozen before any test build)

| ID | Members | Frozen by | Train list |
|---|---|---|---|
| **F** (frozen) | `models/storms/expanded/seed0-9.pt` | `results/ensemble/members_sha256.txt`, blob `3a2dac23…623d` (as in H-EP protocol §1) | `storms_train_expanded.txt` |
| **V** (v2) | `models/v2_recipe/seed30-39.pt` | `results/v2_recipe/members_sha256.txt`, blob hash in the "v2 Atlantic ensemble frozen" entry (v2 design §4) | `storms_train_expanded.txt` |

- **Evaluation:** both are evaluated the same way: the unweighted mean of 10 members per frame (`ensemble_eval.py --expect-sha <members file>`, with `--checkpoints` exactly as listed in each members file). There is no recalibration, weighting, member selection or test-time augmentation.
- **Code pinned at commit:** `ensemble_eval.py`, `src/hurricane_intensity/*.py` (including `dataset.py` preprocessing) and the pass driver and report scripts are pinned by blob hash at the protocol commit (§8 step 2). Each is required equal at the pass. A year can pass under option 3, and nothing else would stop drift.
- **`best_model.pt`:** out (as in H-EP §1).
- **V must be frozen before the protocol is committed.** V's freeze entry is committed first, and the protocol is never committed with an open slot.

## 2. Labels: John's choice, route A (operational) vs route C (HURDAT2)

John asked for "wait for the post-season HURDAT2 update vs. use operational best track with a stated caveat, with the expected date for each and how many 2026 storms each would give at ≥96 kt". All figures are from `label_facts.md`, as of 2026-10-21 20:27 UTC.

| | **A. Operational best track (ATCF b-deck), with a caveat** | **C. Post-season HURDAT2 update** |
|---|---|---|
| Source | `bal{nn}2026.dat`, NHC's working best track (6-hourly, 5 kt steps) | the annual HURDAT2 file containing 2026, pinned by filename and SHA-256 |
| Expected date | **now** for finished storms; the season count is final only after **Nov 30, 2026** | **late Feb to mid-June 2027, most years April–May.** Over 2017–2025, posting dates ran Mar 5 to Jun 10; the most recent (2025's) was the earliest. |
| 2026 Atlantic storms (all) | 9 so far (the season is open) | not knowable yet; likely the same storms |
| …of which > 96 kt | **1:** Isaias AL092026, 105 kt, about 15 hourly frames > 96 kt (estimated, unbuilt). ≥ 96 is the same set, since all peaks are multiples of 5 (sheet §5). | not knowable yet; operationally 1 |
| Caveat | **Operational values are revised in post-analysis.** NHC: the best track "may differ from values contained in system advisories". Isaias's 105 kt is one 5 kt step above the Cat 3 line, at two fixes. A −10 kt revision at its 18Z fix alone takes it from 15 to 9 frames > 96 kt, below the ≥ 10-frame per-storm rule (sheet §3). At both fixes, it would become a Cat 2. | none beyond the best track's own uncertainty |
| How often post-analysis moves a peak | **No citable rate.** The repo's own 2024–25 check (31 storms) found 9 peaks moved (mostly 5–10 kt), 3 changed category, and **none crossed 96 kt** (sheet §3). It is not a rate. | — |
| Stability | **Not stable.** NHC overwrites archived b-decks with the final best track. All 2025 archives were rewritten on 2026-03-06, and Dexter's 50 kt became 70. Needs a hashed snapshot (below). | stable per versioned filename |

**Route B (per-storm TCRs)** is the intermediate source.
- For a hurricane, by 2025 precedent, a TCR appears "about 2.5-5 months after the storm, sometimes later" (sheet §2). So Isaias's is due roughly late Dec 2026 to late Mar 2027.
- It is final storm by storm, but not a pinned season file, so it is not offered as a route of its own.

**Per-season route (option 3 only; John's choice).** Each season's route is chosen at the commit:

| Choice | 2026 | 2027 | Pass |
|---|---|---|---|
| A for both | Dec 2026 snapshot | Dec 2027 snapshot | about Dec 2027 |
| **C for 2026, A for 2027** | the 2026 HURDAT2 (spring 2027), which exists before the pass | Dec 2027 snapshot | about Dec 2027 |
| C for both | spring 2027 | spring 2028 | spring 2028 |

- Under "A for both", the 2026 storms would be scored against operational values although HURDAT2 with 2026 already exists. The hybrid avoids that without waiting for 2028.

**The route A snapshot (John's choice; not done):**
- **What it is:** a copy of the season's Atlantic b-decks (`bal01–bal{nn}<yr>.dat`), each with its SHA-256 and retrieval time, at a basin- and year-specific path.
  - It is never `data/interim/hurdat2_atlantic.txt`: that file is protected (`ee35819c…0bcb5c`) and never refreshed, and `--force-hurdat2-refresh` is never used.
- **Taken alongside it:**
  - **the a-decks** (`aal{nn}<yr>`) for the same storms;
  - **a per-storm record of whether that storm's TCR existed at snapshot time.** A live b-deck may already carry post-analysis values for a storm whose TCR is out. The sheet shows only the bulk archive rewrite.
- **Who and when:** a worker named by the coordinator, in its own commit, in the **first half of December** of the season. That is after Nov 30 and well before the archive rewrite (2025's was 2026-03-06).
- **The snapshot may come before this protocol's commit (§8 step 2).** It is hashed and committed but **not read** until §8 step 4.
- **A-deck fallback:** if the snapshot is missed, the operational track is rebuilt from the a-deck's **OFCL** τ0 rows (the advisory intensity), named now. CARQ is not used; the two differ (sheet: Helene 120/115, Karen 40/45). The known limitation: intensities given only in special or intermediate advisories are missed.
- **It needs John's explicit yes,** because it brings label files into the repo.
- **Route C needs no snapshot:** the HURDAT2 file is pinned at retrieval, as in 2025.

**Fixed whatever the choice:**
- The label route is chosen once per season, in the commit of this protocol.
- A later HURDAT2 release does not trigger a second pass.
- Under A, a descriptive re-scoring against final HURDAT2 is possible only if it is declared here. **Open item:** recommend none.

## 3. The list rule (stated without imagery, applied mechanically)

**The rule (unchanged since r0):** the test list = every **Atlantic** ATCF ID of the test season(s) (`AL<nn>2026`, plus `AL<nn>2027` under option 3) in the chosen label source that meets all of these:
1. **Status:** at least one fix with status TS or HU.
2. **No overlap:** it is in no other list (checked by `load_splits`, the I5 forbidden sets, and the F-10 alias/concurrency check).
   - **A test-season storm that another list names is never silently dropped.** The list commit refuses and the conflict goes to John, because it means a protection was breached (§8, the 2027 protections).
3. **Frames:** ≥ 12 frames after the build survive the unchanged NaN rule. A storm with fewer is dropped and reported before any model runs.

**Why all storms, on meaning:**
- The deployed ensemble predicts every intensity, and the primary is mean per-storm MAE.
- A hurricane-only rule would test "skill on hurricanes", which the primary doesn't ask.
- The > 96 kt questions are secondaries on whichever storms qualify.

**What is read to apply it:** header lines and the status column only. No imagery.

**The build:** as for the 2025 test, except the root and bucket:
- hourly grid plus every exact fix;
- ABI-L2-CMIPF band 13, full disk;
- 600 km box, 10 min tolerance;
- statuses LO, TD, SD, TS, SS, HU;
- `--bucket noaa-goes19`.

**GOES-19 contingency:**
- GOES-19 has been GOES-East since 2025-04-07 (decisions.md:1402).
- A frame whose matched scan is not GOES-19 is **dropped, never substituted**. This is checked on the manifest's `source_key` platform.
- If that leaves a storm under 12 frames, rule 3 drops it.

**Fixed after the build:**
- A storm is never added, dropped or re-cropped for any other reason.
- Nothing is rebuilt after any prediction is seen.

**Disclosure:**
- **The rule was in r0** (01d107b, 16:35:53 EDT). It was committed before the fact sheet was committed (80ed2bb, 16:37:40) or merged here (d2baf56, 16:41:59).
  - Git shows the commit order, not what the author had seen. The sheet's counts were taken at 16:27 EDT.
- **Any option John chooses now** (Read first) is chosen after the 2026 counts were seen. The counts are model-blind, and John asked for them.
- **No threshold was tuned to Isaias** or to any 2026 storm. The ≥ 12-frame floor is r0's; the ≥ 10-frame > 96 kt rule is 2025's.

**Subtropical-only storms** (SS but never TS or HU) are out under rule 1. **Open item:** include them.

## 4. East Pacific storms: excluded (recommended; John's choice)

- **What 2026 EP holds** (operational, sheet §3):
  - 9 EP-ID storms > 96 kt: Genevieve, Karina, Lowell, Nolo, Odalys, Polo, Rachel, Simon and Trudy;
  - plus Lala, a CP ID, excluded as in H-EP.
  - Lowell, Nolo and Lala are ≥ 75° of longitude from GOES-East's sub-point at > 96 kt. Nolo reaches −179.2, at the B-v2-1 antimeridian issue.
- **Why exclude them:**
  - F and V are Atlantic-only, and EP storms are imaged by GOES-18 at different viewing angles. So V − F there mixes the recipe with transfer, which H-EP measures.
  - 2026 EP is also E3's natural unspent confirmation set.
- **If John includes them:**
  - they form a separate, declared secondary, never pooled;
  - the hold in §8 step 3 becomes required;
  - 2026 EP is spent for E3 afterwards.

## 5. The comparison, the test and the reading

**Primary (one, α = 0.05):**
- **Statistic:** d̄ = the mean over storms of d_s = MAE_V(s) − MAE_F(s), the difference in **mean per-storm MAE**. Each storm's MAE is over its frames, and each storm is weighted equally. d < 0 means v2 is better.
- **Test:** two-sided exact sign-flip over storms, with a 95% CI by test inversion (`analyze_n5`'s `signflip_p` / `signflip_ci`, with n passed explicitly).
- **Why storms, not frames:** the models are fixed, so the only replication is over storms. Equal weights stop long storms dominating (item 32 agreed).
- **Floor:** p ≥ 2^(1−n). **At least 6 storms are needed for p < 0.05.** With fewer than 6, the comparison is descriptive only.

**Reading, fixed now** (N5 / F-31 categories, SESOI Δ, §6):

| Result | Condition | Reading |
|---|---|---|
| **BETTER** | p < 0.05, d̄ < 0, CI reaches beyond −Δ | v2 is better on these Atlantic storms |
| **WORSE** | p < 0.05, d̄ > 0, CI reaches beyond +Δ | v2 is worse |
| **SIGNIFICANT BUT BELOW SESOI** | p < 0.05, CI inside ±Δ | real but smaller than what matters |
| **NULL (informative)** | p ≥ 0.05, CI inside ±Δ | no difference of Δ either way |
| **NULL (inconclusive)** | p ≥ 0.05, CI reaches ±Δ | no conclusion |

- **A note for John:** at these n the CIs are wide. BETTER does **not** require d̄ near Δ, only that the CI reach beyond −Δ. For example, d̄ = −0.6 kt with a CI of [−1.2, −0.1] reads BETTER at Δ = 1.0.
- **Conversely,** with an effect near the H9-derived 0.48 kt, a significant result is often SIGNIFICANT BUT BELOW SESOI.

**Decision rule** (open item for John; recommendation):
- v2 replaces the frozen ensemble **only on BETTER**.
- On every other reading, the frozen ensemble stays.
- A replacement is its own committed decision, with new I3/I6 pins.

**Secondaries (α = 0.05 each, uncorrected, labelled):**
- **S1:** > 96 kt MAE, V − F, over storms with ≥ 10 frames > 96 kt. It is tested only with ≥ 6 such storms; otherwise it is descriptive.
- **S2:** > 96 kt bias, on the same storms and rule.

**Descriptive (not tests):**
- the pooled-over-frames MAE, fixes-only MAE and bias for each model, and the difference;
- per band, and per storm by ATCF ID;
- the members' mean, sd and range;
- the reweighted-to-H9-band-shares figure;
- the Cat 5 breakout;
- F's figures beside its 2025 and H9 figures;
- under option 3, each season's d̄ separately (descriptive; the test is pooled).

## 6. SESOI and power

**SESOI, grounded in the decision it serves** (not in F-31, and not in label noise):
- **What Δ is:** the smallest gain in mean per-storm MAE that would justify replacing a frozen, tested reference. A replacement costs new I3/I6 pins, a new site build and a new reference figure, and it breaks the 2025 test's comparability for the deployed model.
- **Recommended Δ = 1.0 kt** (open item for John). That is a judgment of what gain is worth a switch; no external figure fixes it. The alternative is Δ = 0.5 kt.
- **Label noise is not the argument.** It is shared by both models in a paired comparison and averages over frames. So best-track uncertainty does not set Δ.

**Power over storms** (`power.py`; output `power_output.txt`):
- **Method:** d_s ~ Normal(−δ, sd); exact sign-flip at α 0.05; 2000 reps; seed 20261021.
- **Power is P(p < 0.05). P(BETTER) is at most that,** and close to it only when δ = Δ.
- **The planning sd is unknown.**
  - The only paired per-storm spread available is H9's. F-31's control 4× − 1× per-storm MAE, averaged over 20 members, has sd **0.47 kt** over 9 storms (95% interval about 0.32–0.90).
  - It is **H9-derived and likely an underestimate** for 2026, not a bound:
    - the 20-member average removes seed noise that two fixed 10-member ensembles keep;
    - V and F also differ in code version;
    - the storms are on GOES-19, and mostly short and weak, where H9's were long hurricanes. That last point cuts both ways.

**Power at Δ = 1.0 kt:**

| n | sd 0.5 | sd 1.0 | sd 1.5 | sd 2.0 |
|---|---|---|---|---|
| 6 | 0.87 | 0.35 | 0.18 | 0.11 |
| 8 | 1.00 | 0.67 | 0.35 | 0.22 |
| 10 | 1.00 | 0.80 | 0.45 | 0.29 |
| 12 | 1.00 | 0.87 | 0.55 | 0.36 |
| 14 | 1.00 | 0.93 | 0.62 | 0.41 |
| 16 | 1.00 | 0.96 | 0.69 | 0.47 |

**Power at Δ = 0.5 kt:**

| n | sd 0.5 | sd 1.0 | sd 1.5 | sd 2.0 |
|---|---|---|---|---|
| 6 | 0.36 | 0.11 | 0.06 | 0.05 |
| 8 | 0.65 | 0.22 | 0.12 | 0.09 |
| 10 | 0.79 | 0.30 | 0.15 | 0.10 |
| 12 | 0.88 | 0.36 | 0.19 | 0.13 |
| 14 | 0.94 | 0.40 | 0.23 | 0.14 |
| 16 | 0.96 | 0.44 | 0.24 | 0.16 |

**Power at the H9-derived planning effect, 0.48 kt** (F-31's 20-member control ensemble gain, 4× vs 1×; **a planning effect only, not an expectation for 2026**):

| n | sd 0.5 | sd 1.0 | sd 1.5 | sd 2.0 |
|---|---|---|---|---|
| 6 | 0.33 | 0.10 | 0.07 | 0.05 |
| 8 | 0.63 | 0.22 | 0.12 | 0.10 |
| 10 | 0.78 | 0.26 | 0.14 | 0.09 |
| 12 | 0.86 | 0.35 | 0.17 | 0.11 |
| 14 | 0.91 | 0.39 | 0.21 | 0.13 |
| 16 | 0.95 | 0.43 | 0.22 | 0.15 |

**Reading the tables:**
- **2026 alone** (n up to about 9) sits in the 8–10 rows.
- **At sd ≈ 0.5,** it is well powered for Δ = 1.0 (0.87 at n = 6, 1.00 at n = 8), and even for the H9-derived 0.48 kt (0.63 at n = 8).
- **At sd ≥ 1,** it is not. "NULL (inconclusive) most likely" and option 3's power case both hold **only if sd ≥ 1**.
- **Option 3's pooled n** (≥ 16 plausible; basis in Read first) keeps power at Δ = 1.0 at 0.69 or more for sd up to 1.5.

## 7. What gets written (later, by the pass driver; nothing now)

- **Driver and report:** `scripts/run_2026.py` and `scripts/report_2026.py`, specification only here. They are written in the protocol PR, tested on synthetic data and reviewed by hc-adversary before the protocol is committed.
- **Label input (new code; the build has none today).** The build reads fixes only from `config.HURDAT2_CACHE_PATH` = `data/interim/hurdat2_atlantic.txt` (`config.py:35`). `download_hurdat2` uses that cache on a hit and ignores `--hurdat2-url` (`hurdat2.py:103-113`). That file holds no test-season storm, and it is protected and never refreshed. No b-deck or a-deck reader exists. So the protocol PR also writes:
  - for route C, a pinned, season-specific HURDAT2 path (like H-EP's B-v2-2 cache);
  - for route A, a snapshot reader;
  - the OFCL τ0 a-deck fallback reader.
  - Like the driver, this code is tested on synthetic data and reviewed with it. Its blobs are pinned at §8 step 2 with the rest, before the commit, so it is neither unpinned nor in conflict with the `src` pin.
- **Sealed root:** builds go to `data/2026_sealed/` (as H-EP's `data/hep_sealed/`), reachable only through `--processed-root`.
- **Outputs:**
  - `results/final_2026/{F,V}/ensemble_2026.csv`;
  - `members_sha256_at_run.txt`;
  - the metrics, `primary.csv` and the report;
  - `logs/final_2026/` and `pass_manifest.json`.
- **`.gitattributes`:**
  - `-text` rows for `results/final_2026/**`, `logs/final_2026/**` and the snapshot paths;
  - a coverage test that enumerates the PR's committed files, with `git check-attr text` = unset on each (the PR #29 gap).
- **Refusals:**
  - any existing output directory;
  - any storm outside the list;
  - members not matching their file;
  - any path not pinned;
  - any pinned script or `src` blob that differs from its commit-time hash;
  - any frame not from GOES-19.

## 8. Order of operations (each committed before the next)

1. **V frozen:** its freeze entry is committed (v2 design §4; target mid-December 2026). Route A snapshots (§2) may be taken and committed at any time, unread.
2. **This protocol committed:**
   - **When:** after V's freeze and after hc-adversary's review of the driver and report scripts, and **before the H-EP pass, unconditionally.** F is E1 in H-EP whatever happens, so H-EP results on F always precede the clean pass.
   - **It can't be amended after the H-EP pass.** A defect found later is reported, not edited.
   - **The decisions.md entry records:**
     - John's option (Read first) and the label route per season, with the snapshot choice;
     - Δ, the decision rule and the EP choice;
     - the list rule;
     - the members files' blob hashes;
     - the blob hashes of the driver, the report, the label-input code (§7), `ensemble_eval.py` and `src/hurricane_intensity/*.py`.
   - **Under option 3, the same commit adds the 2027 protections:**
     - an I4-style row for 2027 ("no 2027 storm imagery downloaded or cropped");
     - a `test_no_2027_id_in_any_list`;
     - a `data/processed/*2027*` check, as for 2026.
3. **E3 settled (John's choice; recommended):** the pass waits until E3's design and checkpoints are frozen, or its slot has closed at its deadline (about end-January 2027).
   - **Reason:** no test result can then shape E3's design, its budget or its comparator.
   - **Cost:** under option 1 with route A, the pass moves from about December 2026 to about February 2027. There is no cost under route C or option 3.
   - **Alternative:** no hold.
   - **If 2026 EP storms are included (§4),** the hold is required.
4. **Labels available, and the list:**
   - The route's labels become available, or the snapshot is read for the first time.
   - The list rule is applied, and `storms_test_2026.txt` is committed (or `storms_test_2026_2027.txt` under option 3; header and status reads only).
   - **I4 amendment, with John's explicit yes, in the same commit:**
     - `tests/test_list_invariants.py::test_no_2026_id_in_any_list` exempts exactly that list file by name, as `test_no_2025_test_storm_in_any_other_list` does for `storms_test_2025.txt`;
     - the list gets its `LIST_ROLES` entry;
     - check.py's I4 check and the FEATURES.md I4 row are reworded: no `*2026` ID in any `storms_*.txt` **except** the named test list, and no `data/processed/*2026*`, with `data/2026_sealed/` the only place 2026 crops may exist;
     - under option 3, the same applies for 2027.
   - **Nothing is changed in any test now;** this draft only states the amendment.
5. **Build** into `data/2026_sealed/` (I4's lift for this build, under step 4's yes). The build reads labels only through the pinned label input (§7). `data/interim/hurdat2_atlantic.txt` is neither refreshed nor used for test seasons.
   - Then the NaN exclusion step, with carry-forward (as H-EP gate 7), and the GOES-19 check (§3).
   - Frames are counted per storm, and rule 3 drops are committed. Still no model runs.
6. **Domain shift:** the test set vs H9, before any model runs.
7. **The pass:**
   - pinned blobs re-checked;
   - F and V in one driver invocation;
   - the report;
   - a commit of `results/final_2026/`.
8. **Spend in code:** the test IDs join TCDataset's spent set through the registry role, with a `check.py` pin of the results (as I3 / I3b).

**Timeline:**

| Option and route | Pass |
|---|---|
| Option 1, route A | about Feb 2027 with the hold, about Dec 2026 without it |
| Option 1, route C | after the 2027 HURDAT2 file (late Feb to mid-June 2027) |
| Option 3, A for both or C/A | about Dec 2027 |
| Option 3, C for both | spring 2028 |

## 9. If the result is poor, or a bug is found

- **The result stands:**
  - no retraining or tuning on test data;
  - no second pass;
  - no change of members, averaging or preprocessing.
- **A bug in the evaluation code** found afterwards is fixed and rerun, and both runs are reported with the reason.
- **No later selection:** the test set is never used to select a model.

## 10. Open items for John

1. **The option:** 1, 2, 3 (recommended; label-informed and model-blind) or 4 (Read first). Options 2 and 3 change decision 1's "confirmed only on the 2026 clean test".
2. **Labels:** route A (operational, with a caveat) or C (HURDAT2), per season under option 3 (§2). Under A, a yes is needed to the December snapshot (with a-decks and the per-storm TCR record), or to the OFCL a-deck rebuild instead.
3. **SESOI:** Δ = 1.0 kt recommended, or 0.5 kt (§6).
4. **Decision rule:** replace only on BETTER (recommended) (§5).
5. **2026 EP storms:** excluded (recommended) (§4).
6. **The hold:** hold the pass until E3 is settled (recommended), or no hold (§8 step 3).
7. **The I4 amendment at step 4** (§8). It will need John's yes when the list is committed. It is stated here so it is not a surprise.
8. **Subtropical-only storms** (§3), and under A, any declared re-scoring against final HURDAT2 (§2). Recommend no to both.

## Files in this draft

- `protocol_draft.md`: this file.
- `label_facts.md`: hc-reviewer's fact sheet (2026 label information, as of 2026-10-21 20:27 UTC), unedited here.
- `power.py`: the §6 tables. CPU only; it reads nothing. Run with `.venv\Scripts\python.exe -I -B reports/v2_survey/2026_clean_test/power.py` (about 10 s).
- `power_output.txt`: its output.
