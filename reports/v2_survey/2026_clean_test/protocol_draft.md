# 2026 clean test: protocol (DRAFT r1, not approved, nothing built or run)

**Status:** a draft for John (his decision 4, 2026-10-21). I4 is unchanged: no 2026 imagery is downloaded, listed or cropped, and no 2026 label file is downloaded.

**Inputs:**
- **The 2026 counts** come from hc-reviewer's fact sheet, `label_facts.md` (this folder). Its season counts are **as of 2026-10-21 20:27 UTC**, from the operational b-decks. That sheet is 2026 label information (peaks and > 96 kt counts), so this draft is written by someone who has seen the 2026 peaks (§3, disclosure).
- **Route names** follow the sheet:
  - **A** = the operational ATCF b-deck;
  - **B** = the per-storm TCRs;
  - **C** = the annual HURDAT2 update.

## Read first: the 2026 Atlantic season is too small for the planned test

**As of 2026-10-21, the 2026 Atlantic season has 9 storms** (AL01–AL09 2026).
- Only **Isaias AL092026** went above 96 kt (operational peak 105 kt). Every other storm peaked at ≤ 60 kt (`label_facts.md` §3).
- The season runs to Nov 30, so these counts can still grow.

**What each list rule gives** (operational labels, unbuilt; scan availability can't be checked under I4):

| List rule | 2026 Atlantic storms | > 96 kt storms | The primary (≥ 6 storms needed, §5) |
|---|---|---|---|
| **The 2025 test's inferred rule:** peak ≥ 64 kt in any row (sheet §4; never written down in 2025) | **1** (Isaias) | 1 | **not testable**, descriptive only |
| **This draft's rule (r0, §3):** at least one TS or HU fix, ≥ 12 built frames | **up to 9**, before the frame and scan checks | 1 | testable only if ≥ 6 survive the build; power at n = 8–10 in §6 |
| Either rule, > 96 kt secondaries | — | 1 | descriptive only (≥ 6 storms with ≥ 10 frames > 96 kt needed) |

**What this means for John's decision 1** ("confirmed only on the 2026 clean test"):
- **Under the 2025-style rule,** 2026 cannot confirm anything: one storm is an anecdote.
- **Under the all-storm rule,** the primary may be testable, but on at most about 9 storms, 8 of them ≤ 60 kt. Its power is low (§6): about 0.67 for a 1-kt gain with a between-storm sd of 1 kt at n = 8, and about 0.22 at the H9-derived planning effect.
  - **So NULL (inconclusive) is the most likely reading,** unless v2's gain on weak storms is large.
- **The high-end question can't be tested on 2026 Atlantic storms at all:** the > 96 kt MAE and bias, where the frozen ensemble's known weakness lies. The data hold one storm, and that storm sits one 5 kt step above the line.

**Options** (neutral; the recommendation follows):

| Option | What it is | For | Against |
|---|---|---|---|
| **1. All-storm rule, 2026 only** | §3's rule as written in r0, with the primary on up to about 9 Atlantic storms | keeps decision 1's wording and timing; the meaning fits (§3) | low power; mostly weak storms; the high end is descriptive only |
| **2. 2026 descriptive only; confirmation deferred** | 2026 is reported descriptively, with no test and no replacement decision. Confirmation comes from 2027 Atlantic storms under this protocol, committed now | no underpowered test spends the set | v2 stays unconfirmed until 2027 labels (2028 under C); 2026 is used without testing |
| **3. Pooled 2026 + 2027, one pass** | the all-storm rule over both seasons. This protocol is committed now, before any 2026 build, and one pass runs after the 2027 labels | roughly doubles n; one pre-registered look; the high end has a better chance of ≥ 6 storms | changes decision 1 to "confirmed on the 2026–2027 clean test"; delay of about a year; v2 and the frozen ensemble must stay frozen until then |
| **4. Add 2026 East Pacific storms** | the 9 EP-ID storms > 96 kt (`label_facts.md` §3) join the test | many high-end storms | not recommended (§4): it mixes basin and GOES-18 transfer into a recipe comparison, and spends E3's natural confirmation set |

**Recommendation: option 3**, with the all-storm rule. The reason is meaning and power, not Isaias.
- v2 is a candidate to replace a deployed ensemble. A test that will most likely end NULL (inconclusive) on about 9 mostly weak storms would spend the 2026 season for little.
- Pooling two seasons under one protocol committed now keeps the single-look discipline.
- It is John's change to decision 1. Option 1 is the choice if keeping the 2026 timing matters more.
- **The other choices carry over unchanged under any option:** the list rule, the label route and the snapshot (§2), the SESOI, and the EP decision.

## 0. What the test is

**The question.** On Atlantic storms that no model has seen, is the v2 Atlantic ensemble's overall MAE different from the frozen ensemble's?
- v2 Atlantic: the 4× control recipe, 10 new members (`reports/v2_survey/v2_recipe/design.md`).
- The frozen ensemble: 1×, `models/storms/expanded/seed0-9.pt`.

**Why only an unspent season can answer it:**
- The v2 recipe was chosen on H9, through F-31's P1.
- The 2025 set is spent.
- H-EP is an East Pacific transfer test under its own protocol (`reports/v2_survey/hep/protocol.md`, 3901f4cf).

**How the test runs:** **once, in one declared pass**, and the set is then spent in code (§8). The result decides whether v2 replaces the frozen ensemble, by the rule in §5. Nothing is retrained, re-selected or tuned on test data.

## 1. The two models (both frozen before any 2026 build)

| ID | Members | Frozen by | Train list |
|---|---|---|---|
| **F** (frozen) | `models/storms/expanded/seed0-9.pt` | `results/ensemble/members_sha256.txt`, blob `3a2dac23…623d` (as in H-EP protocol §1) | `storms_train_expanded.txt` |
| **V** (v2) | `models/v2_recipe/seed30-39.pt` | `results/v2_recipe/members_sha256.txt`, blob hash in the "v2 Atlantic ensemble frozen" entry (v2 design §4) | `storms_train_expanded.txt` |

- **Evaluation:** both ensembles are evaluated the same way: the unweighted mean of 10 members per frame (`ensemble_eval.py --expect-sha <members file>`), with no recalibration, weighting, member selection or test-time augmentation. `dataset.py` preprocessing is unchanged.
- **`best_model.pt`: out** (as in H-EP §1).
- **Freeze before commit:** V's freeze entry is committed before this protocol is. The protocol is never committed with an open slot.

## 2. Labels: John's choice, route A (operational) vs route C (HURDAT2)

John asked for "wait for the post-season HURDAT2 update vs. use operational best track with a stated caveat, with the expected date for each and how many 2026 storms each would give at ≥96 kt". All figures are from `label_facts.md`, as of 2026-10-21 20:27 UTC.

| | **A. Operational best track (ATCF b-deck), with a caveat** | **C. Post-season HURDAT2 update** |
|---|---|---|
| Source | `bal{nn}2026.dat`, NHC's working best track: 6-hourly, 5 kt steps | the annual HURDAT2 file containing 2026, pinned by filename and SHA-256 |
| Expected date | **now** for finished storms; the season count is final only after **Nov 30, 2026** | **late Feb to mid-June 2027, most years April–May** (2017–2025 posting dates: Mar 5 to Jun 10; the most recent, 2025's, was the earliest) |
| 2026 Atlantic storms (all) | 9 so far (season open) | not knowable yet; likely the same storms |
| …of which > 96 kt (≥ 96 is the same set: all peaks are multiples of 5, sheet §5) | **1:** Isaias AL092026, 105 kt, about 15 hourly frames > 96 kt (estimated, unbuilt) | not knowable yet; operationally 1 |
| Caveat | **Operational values are revised in post-analysis.** NHC: the best track "may differ from values contained in system advisories". Isaias's 105 kt is one 5 kt step above the Cat 3 line, at two fixes. A −10 kt revision at its 18Z fix alone takes it from 15 to 9 frames > 96 kt, below the ≥ 10-frame per-storm rule (sheet §3 table). A −10 kt revision at both fixes makes it a Cat 2. | none beyond best-track uncertainty: typically about ±10% in intensity (Michael TCR, cited in sheet §1) |
| How often post-analysis moves a peak | **No citable rate.** In the repo's own 2024–25 check (31 storms), peaks moved in 9 of 31 storms (mostly 5–10 kt), 3 changed category, and **none crossed 96 kt** (sheet §3). Not a rate. | — |
| Stability | **Not stable:** NHC overwrites archived b-decks with the final best track. All 2025 archives were rewritten on 2026-03-06, and Dexter's 50 kt became 70. Route A needs a hashed snapshot (below). | stable per versioned filename; later versions in the same year usually revise only old seasons |

**Route B (per-storm TCRs)** is the intermediate source. Each storm's post-analysis best track appears storm by storm. For a hurricane that is, by 2025 precedent, "about 2.5-5 months after the storm, sometimes later" (sheet §2), so Isaias's TCR is roughly late Dec 2026 to late Mar 2027. It is final storm by storm but not a pinned season file, so it is not offered as a route of its own.

**The route A snapshot (John's choice, not done):**
- **What:** route A has no fixed version unless the protocol takes one. That means a copy of the 2026 Atlantic b-decks (`bal01–bal{nn}2026.dat`), with each file's SHA-256 and the retrieval time, kept at a basin-and-year-specific path. It is never `data/interim/hurdat2_atlantic.txt`, which is protected (`ee35819c…0bcb5c`) and never refreshed; `--force-hurdat2-refresh` is never used.
- **Who and when:** a worker named by the coordinator, in a commit of its own. It should run after the season ends (Nov 30, 2026) and well before NHC rewrites the archive (the 2025 rewrite was 2026-03-06): **in the first half of December 2026**.
- **Alternative:** rebuild the operational track later from the a-deck `OFCL` / `CARQ` τ0 rows. The known limitation is that intensities given only in special or intermediate advisories are missed (sheet §3 caveats).
- **It needs John's explicit yes,** because taking the copy brings 2026 label files into the repo. Until then nothing is downloaded.
- Under route C, no snapshot is needed: the HURDAT2 file is pinned at retrieval, as in 2025.

**Fixed whatever the choice:**
- The label route is chosen once, in the commit of this protocol.
- A later HURDAT2 release does not trigger a second pass.
- Under A, a descriptive re-scoring against final HURDAT2 is possible only if it is declared here, before the pass. **Open item:** recommend none.

## 3. The list rule (stated without 2026 imagery, applied mechanically)

**The rule (unchanged from r0):** `storms_test_2026.txt` = every **Atlantic** ATCF ID of the test season(s) (`AL<nn>2026`, and `AL<nn>2027` under option 3) in the chosen label source that meets all of these:
1. **Status:** at least one fix with status TS or HU.
2. **No overlap:** in no other list. Checked by `load_splits`, the I5 forbidden sets and the F-10 alias/concurrency check.
3. **Frames:** ≥ 12 frames after the build survive the unchanged NaN rule. A storm with fewer is dropped and reported before any model runs.

**Why all storms, on meaning:**
- The deployed ensemble predicts every intensity, and the primary is overall MAE.
- A rule restricted to hurricanes would test a different question, "skill on hurricanes", that the primary doesn't ask.
- The > 96 kt questions are secondaries on whichever storms qualify.

**What is read, and the build:**
- Only header lines and the status column are read to apply the rule. No imagery.
- The build is as for the 2025 test, except the root and bucket:
  - hourly grid plus every exact fix;
  - ABI-L2-CMIPF band 13, full disk;
  - 600 km box, 10 min tolerance;
  - statuses LO, TD, SD, TS, SS, HU;
  - `--bucket noaa-goes19`.
- A storm is never added, dropped or re-cropped for any other reason. Nothing is rebuilt after any prediction is seen.

**Disclosure:**
- **The rule predates the counts.** This rule was written and committed in r0 (01d107b) before the fact sheet was merged (this PR's merge of 41223fb), so before this author saw any 2026 count.
- **Any rule John chooses now is chosen after the 2026 counts were seen.** The counts are model-blind (labels, not predictions), and John asked for them.
- **Nothing was tuned to Isaias.** No threshold here was tuned to Isaias's frame counts or to any 2026 storm. The ≥ 12-frame floor and the ≥ 10-frame > 96 kt rule are r0's and 2025's.
- **The 2025 rule** (peak ≥ 64 kt, inferred in sheet §4) is shown in the table above for comparison only.

**Subtropical-only storms** (SS but never TS or HU) are out under rule 1. **Open item:** include them.

## 4. East Pacific 2026 storms: excluded (recommended; John's choice)

- **What 2026 EP holds** (operational, sheet §3):
  - 9 EP-ID storms > 96 kt: Genevieve, Karina, Lowell, Nolo, Odalys, Polo, Rachel, Simon, Trudy;
  - plus Lala, a CP ID, which is excluded as in H-EP.
  - Three are far west: Lowell, Nolo and Lala are ≥ 75° of longitude from GOES-East's sub-point at > 96 kt, and Nolo reaches −179.2, at the B-v2-1 antimeridian issue.
- **Why exclude them:**
  - **They would mix questions.** F and V are Atlantic-only, and EP storms are imaged by GOES-18 at different viewing angles. V − F there mixes the recipe with basin and viewing-angle transfer, which H-EP measures.
  - **They are E3's confirmation set.** 2026 EP is E3's (Atlantic + East Pacific) natural unspent confirmation set, under its own protocol.
  - **Keeping them out keeps them unseen.** No 2026 EP result of any model then exists before E3 is frozen.
- **If John includes them:**
  - they form a separate, declared secondary, never pooled with the Atlantic primary;
  - the hold in §8 step 3 becomes required;
  - 2026 EP is spent for E3 afterwards.

## 5. The comparison, the test and the reading

**Primary (one, α = 0.05):**
- **Statistic:** d̄ = mean over storms of d_s = MAE_V(s) − MAE_F(s), with overall MAE per storm over its frames and each storm weighted equally. d < 0 means v2 is better.
- **Test:** two-sided exact sign-flip over storms, with a 95% CI by test inversion (`analyze_n5`'s `signflip_p` / `signflip_ci`, n passed explicitly).
- **Why storms:** the two models are fixed, so the only replication is over storms, and frames within a storm are dependent. **Open item for the adversary:** frame-weighted storm-level statistic as the alternative.
- **Floor:** p ≥ 2^(1−n). **At least 6 storms are needed for p < 0.05.** With fewer than 6 after §3, the comparison is descriptive only, and the reading says so.

**Reading, fixed now** (N5 / F-31 categories, SESOI Δ, §6):

| Result | Condition | Reading |
|---|---|---|
| **BETTER** | p < 0.05, d̄ < 0, CI reaches beyond −Δ | v2 is better on these Atlantic storms |
| **WORSE** | p < 0.05, d̄ > 0, CI reaches beyond +Δ | v2 is worse |
| **SIGNIFICANT BUT BELOW SESOI** | p < 0.05, CI inside ±Δ | real but smaller than what matters |
| **NULL (informative)** | p ≥ 0.05, CI inside ±Δ | no difference of Δ either way |
| **NULL (inconclusive)** | p ≥ 0.05, CI reaches ±Δ | no conclusion |

**Decision rule** (open item for John; recommendation):
- v2 replaces the frozen ensemble **only on BETTER**.
- On every other reading, the frozen ensemble stays.
- A replacement is its own committed decision with new I3/I6 pins.

**Secondaries (α = 0.05 each, uncorrected, labelled):**
- **S1:** > 96 kt MAE, V − F, over storms with ≥ 10 frames > 96 kt. It is tested only with ≥ 6 such storms; otherwise it is descriptive. **On 2026 Atlantic alone it is descriptive (one storm at most).**
- **S2:** > 96 kt bias, V − F, on the same storms and rule.

**Descriptive (not tests):**
- pooled-over-frames MAE, fixes-only MAE and bias for each model, and the difference;
- per band;
- per storm by ATCF ID;
- the members' mean, sd and range;
- the reweighted-to-H9-band-shares figure;
- the Cat 5 breakout;
- F's figures beside its 2025 and H9 figures.

## 6. SESOI and power

**SESOI, fixed on what would matter, not taken from F-31:**
- **The decision it serves:** replacing a frozen, tested ensemble, with new pins and a new reference.
- **Recommended Δ = 1.0 kt overall MAE** (open item for John): about 12% of the frozen ensemble's overall MAE level (8.6 kt on H9), and a fifth of the 5-kt label resolution.
- **For scale:** best-track intensity itself is uncertain by about ±10% (Michael TCR, sheet §1), about ±5 kt at 50 kt and ±10 kt at 100 kt. A model difference much smaller than that is hard to call meaningful.
- **Alternative:** Δ = 0.5 kt.

**Power over storms** (`power.py`; output in `power_output.txt`):
- **Model:** d_s ~ Normal(−δ, sd); exact sign-flip at α 0.05; 2000 reps; seed 20261021.
- **Planning sd:**
  - The only paired per-storm spread available is H9's. F-31's control 4× − 1× member-mean per-storm MAE has sd **0.47 kt** over 9 storms (**H9-derived**: same code, seeds and satellite).
  - It is a lower bound for 2026, where V and F differ in code version and seeds, and the storms are on GOES-19.
  - So the grid runs from 0.5 to 2.0 kt.

| n storms | Δ 1.0, sd 1.0 | Δ 1.0, sd 1.5 | Δ 0.5, sd 1.0 | **H9-derived 0.48, sd 1.0** | **H9-derived 0.48, sd 1.5** |
|---|---|---|---|---|---|
| 6 | 0.35 | 0.18 | 0.11 | 0.10 | 0.07 |
| 8 | 0.67 | 0.35 | 0.22 | 0.22 | 0.12 |
| 10 | 0.80 | 0.45 | 0.30 | 0.26 | 0.14 |
| 12 | 0.87 | 0.55 | 0.36 | 0.35 | 0.17 |
| 14 | 0.93 | 0.62 | 0.40 | 0.39 | 0.21 |
| 16 | 0.96 | 0.69 | 0.44 | 0.43 | 0.22 |

- **The H9 effect is for planning only.** 0.48 kt is F-31's H9 ensemble gain (20-member control, 4× vs 1×): **H9-derived, a planning effect only**.
- **2026 alone** (n up to about 9) sits in the 8–10 rows.
- **Pooling 2026 + 2027** (option 3) would plausibly reach the 14–16 rows. That is not a forecast of 2027's count.
- **At the H9-derived effect, no row reaches 0.5.**

## 7. What gets written (later, by the pass driver; nothing now)

- **Scripts:** `scripts/run_2026.py` and `scripts/report_2026.py`, specification only here. They are written in the protocol PR, tested on synthetic data, and reviewed by hc-adversary before the protocol is committed.
- **Sealed root:** builds go to `data/2026_sealed/` (as H-EP's `data/hep_sealed/`), reachable only through `--processed-root`.
- **Outputs:**
  - `results/final_2026/{F,V}/ensemble_2026.csv`;
  - `members_sha256_at_run.txt`;
  - metrics, `primary.csv` and the report;
  - `logs/final_2026/` and `pass_manifest.json`.
- **`.gitattributes`:** `results/final_2026/** -text`, `logs/final_2026/** -text`, and the route A snapshot's path if taken, plus a coverage test (`git check-attr text` = unset on every committed output; the PR #29 gap).
- **Refusals:** any existing output directory, any storm outside the list, members not matching their file, and any path that is not pinned.

## 8. Order of operations (each committed before the next)

1. **V frozen:** its freeze entry is committed (v2 design §4).
2. **This protocol committed**, before any 2026 build and before the H-EP pass if V is E3's comparator. The decisions.md entry records:
   - John's option (§ "Read first"), the label route and the snapshot choice;
   - Δ, the decision rule and the EP choice;
   - the list rule;
   - the members files' blob hashes, and the driver and report blob hashes.
3. **E3 settled (John's choice; recommended):** the pass waits until E3's design and checkpoints are frozen, or its slot has closed at its deadline (about end-January 2027).
   - **Reason:** no test result can shape E3's design, its budget or its comparator.
   - **Cost:** under route A with 2026 only, the pass moves from label-ready (about December 2026) to about February 2027. There is no cost under route C, or under option 3.
   - **Alternative:** no hold. Only step 2 orders the two tests, so a v2-vs-frozen result could be seen while E3's design is still open.
   - If 2026 EP storms are included (§4), the hold is required.
4. **Labels available** under the chosen route. Under A, that is the snapshot of §2. The list rule is applied, and the storm list is committed (header and status reads only).
5. **Build** into `data/2026_sealed/`, with John's explicit yes, since I4 is lifted only for this build. Then the NaN exclusion step with carry-forward (as H-EP gate 7). Frames are counted per storm and rule 3 drops are committed. Still no model.
6. **Domain shift:** the test set vs H9, before any model runs.
7. **The pass:** F and V in one driver invocation, then the report, then a commit of `results/final_2026/`.
8. **Spend in code:** the test IDs join TCDataset's spent set through the registry role, with a `check.py` pin of the results (as I3 / I3b).

**Timeline:**
- **Option 1, route A:** about February 2027 with the hold, December 2026 without it.
- **Option 1, route C:** after the 2027 HURDAT2 file (late Feb to mid-June 2027).
- **Option 3:** after the 2027 season's labels; about December 2027 under A, spring 2028 under C.

## 9. If the result is poor, or a bug is found

- **The result stands:**
  - no retraining or tuning on test data;
  - no second pass;
  - no change of members, averaging or preprocessing.
- **A bug in the evaluation code** found afterwards is fixed and rerun, and both runs are reported with the reason.
- **No later selection:** the test set is never used to select a model.

## 10. Open items for John

1. **The feasibility option:** 1, 2, 3 (recommended) or 4 ("Read first"). Options 2 and 3 change decision 1's "confirmed only on the 2026 clean test".
2. **Labels:** route A (operational, with a caveat) or C (HURDAT2) (§2). Under A, a yes is also needed to the December 2026 b-deck snapshot, or the a-deck rebuild instead.
3. **SESOI:** Δ = 1.0 kt recommended, or 0.5 kt (§6).
4. **Decision rule:** replace only on BETTER (recommended) (§5).
5. **2026 EP storms:** excluded (recommended) (§4).
6. **The hold:** hold the pass until E3 is settled (recommended), or rely on the protocol commit before the H-EP pass alone (§8 step 3).
7. **Subtropical-only storms** (§3), and under A, any declared re-scoring against final HURDAT2 (§2). Recommend no to both.

## Files in this draft

- `protocol_draft.md`: this file.
- `label_facts.md`: hc-reviewer's fact sheet (2026 label information, as of 2026-10-21 20:27 UTC). Unedited here.
- `power.py`: the §6 table. CPU only; it reads nothing. Run it with `.venv\Scripts\python.exe -I -B reports/v2_survey/2026_clean_test/power.py` (about 10 s).
- `power_output.txt`: its output.
