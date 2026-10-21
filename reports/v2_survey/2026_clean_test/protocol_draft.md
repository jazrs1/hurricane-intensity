# 2026 clean test: protocol (DRAFT r0, not approved, nothing built or run)

**Status:** a draft for John (his decision 4, 2026-10-21). I4 is unchanged:
- no 2026 imagery is downloaded, listed or cropped;
- nothing in this draft read any 2026 storm, label or image.

The label figures in §2 come from hc-reviewer's fact sheet. They are marked **[fact sheet]** until it is merged in.

## 0. What the test is

**The question.** On Atlantic storms of the 2026 season, which no model has seen, is the v2 Atlantic ensemble's overall MAE different from the frozen ensemble's?
- v2 Atlantic: the 4× control recipe, 10 new members (`reports/v2_survey/v2_recipe/design.md`).
- The frozen ensemble: 1×, `models/storms/expanded/seed0-9.pt`.

**Why only this set can answer it.**
- The v2 recipe was chosen on H9, through F-31's P1, so H9 is optimistic for it.
- The 2025 set is spent.
- H-EP is an East Pacific transfer test with its own protocol (`reports/v2_survey/hep/protocol.md`, 3901f4cf). It is not a confirmation of an Atlantic recipe.

**How it runs.** The 2026 set is used **once, in one declared pass**, and is then spent in code (§8).

**What the result decides.** Whether v2 replaces the frozen ensemble, by the rule fixed in §5. Nothing else is retrained, re-selected or tuned on 2026 data, whatever the result.

## 1. The two models (both frozen before any 2026 build)

| ID | Members | Frozen by | Train list |
|---|---|---|---|
| **F** (frozen) | `models/storms/expanded/seed0-9.pt` | `results/ensemble/members_sha256.txt`, blob `3a2dac23…623d` (as in H-EP protocol §1) | `storms_train_expanded.txt` |
| **V** (v2) | `models/v2_recipe/seed30-39.pt` | `results/v2_recipe/members_sha256.txt`, blob hash in the "v2 Atlantic ensemble frozen" entry (v2 design §4) | `storms_train_expanded.txt` |

- **Both ensembles are evaluated the same way:**
  - the unweighted mean of 10 members per frame (`ensemble_eval.py --expect-sha <members file>`);
  - no recalibration, weighting, member selection or test-time augmentation;
  - `dataset.py` preprocessing unchanged.
- **`best_model.pt`: out.** It is a single "good draw", as H-EP §1 argues. Including it would need a declaration before this protocol is committed.
- **Requirement:** V's freeze entry is committed before this protocol is. If V is not frozen when this protocol would be committed, the protocol waits; it is never committed with an open slot.

## 2. Labels: John's choice

John asked for this choice to be laid out. Post-season HURDAT2 vs operational best track with a stated caveat, with the expected date and the number of 2026 storms at ≥ 96 kt for each.

| | **(A) Wait for the post-season HURDAT2 update** | **(B) Operational best track, with a stated caveat** |
|---|---|---|
| Source | the reanalysed HURDAT2 release including 2026 | **[fact sheet]**: the operational best-track file (e.g. ATCF b-deck) as of a fixed, declared retrieval date |
| Expected availability | **[fact sheet]** | **[fact sheet]** |
| 2026 Atlantic storms the list rule (§3) would give | **[fact sheet]** | **[fact sheet]** |
| …of which reach ≥ 96 kt | **[fact sheet]** | **[fact sheet]** |
| …with ≥ 10 frames > 96 kt (the per-storm > 96 kt rule) | **[fact sheet]** (estimated from fixes; frames are known only after the build) | **[fact sheet]** |
| Caveat | none beyond best-track uncertainty | operational intensities are revised after the season; the size and direction of revisions is **[fact sheet]**. Results read "against operational best track". |
| Earliest pass | after availability (and, under §8 choice 3, after E3's freeze or deadline) | when the labels are ready; under §8 choice 3, not before E3's freeze or deadline (about end-January 2027) |

**What does not change with the choice:** the list rule, the metrics, the SESOI, the reading and the single pass.

**How the source is fixed:**
- **(A):** the HURDAT2 file is pinned by SHA-256 at retrieval and kept at a basin-and-year-specific path. It is never `data/interim/hurdat2_atlantic.txt`: that file is protected (`ee35819c…0bcb5c`), is never refreshed, and `--force-hurdat2-refresh` is never used.
- **(B):** the files are pinned the same way.
- **The label source is chosen once,** in the commit of this protocol. A later HURDAT2 release does not trigger a second pass. Under (B), a descriptive re-scoring against final HURDAT2 is possible only if it is declared here, before the pass. **Open item:** recommend none.

## 3. The list rule (stated without 2026 imagery, applied mechanically)

**The rule:** `storms_test_2026.txt` = every **2026 Atlantic** ATCF ID (`AL<nn>2026`) in the chosen label source (§2) that meets all of these:
1. **Statuses:** at least one fix with status TS or HU. These are named-storm or hurricane strength, so TDs that never strengthened are out.
2. **Not listed anywhere else:** the storm is in no other list. Training, validation, H9, the 2025 test and H-EP are all earlier seasons or East Pacific; this is checked anyway by `load_splits`, the I5 forbidden sets, and the F-10 alias/concurrency check.
3. **Enough frames:** after the build (§8), the storm has ≥ 12 frames that survive the unchanged NaN rule. A storm with fewer is dropped and reported, before any model runs.

**What is read to apply it:**
- header lines and the status column only;
- no intensities beyond what rule 1 needs;
- no imagery.

**Build settings:**
- as the 2025 test's, except the root: hourly grid plus every exact fix, ABI-L2-CMIPF band 13, full disk;
- 600 km box, 10 min tolerance, statuses LO, TD, SD, TS, SS, HU;
- `--bucket noaa-goes19`.

**What is not allowed:**
- A storm is never added, dropped or re-cropped for any other reason.
- Nothing is rebuilt after any prediction is seen.
- The 2025 entry recorded its list but no selection rule; this rule is explicit so the 2026 list can't be chosen by hand.

**Subtropical-only storms** (SS but never TS/HU) are out under rule 1. **Open item:** include them.

## 4. East Pacific 2026 storms: excluded (recommended; John's choice)

- **Recommendation:** 2026 EP storms are **not** part of this test.
- **Why:**
  - **They would mix two questions.** Both F and V are Atlantic-only, and EP storms are imaged by GOES-18 (West) at different viewing angles. A v2-vs-frozen difference there would mix the recipe with basin and viewing-angle transfer, which H-EP already measures.
  - **They should stay unspent for E3.** E3 (Atlantic + East Pacific) is evaluated on H-EP under its own protocol. 2026 EP storms are the natural unspent confirmation set for E3 later, under a protocol of its own.
  - **No EP result should reach E3's design.** Leaving them out also means no 2026 EP result of any model exists before E3's design and checkpoints are frozen.
- **If John includes them:**
  - they are a separate, declared secondary (V − F on 2026 EP), never pooled with the Atlantic primary;
  - the pass then also waits for E3's freeze (§8);
  - and the protocol states that 2026 EP is spent for E3's confirmation afterwards.

## 5. The comparison, the test and the reading

**Primary (one, α = 0.05):**
- **The statistic:** d̄ = mean over storms of d_s = MAE_V(s) − MAE_F(s), overall MAE per storm over its frames, each storm weighted equally. d < 0 means v2 is better.
- **The test:** two-sided exact sign-flip over storms (all 2^n sign assignments), with a 95% CI by test inversion. It is `analyze_n5`'s `signflip_p` / `signflip_ci`, called with n passed explicitly.
- **Why storms, not frames:**
  - The two models are fixed, so the only replication is over storms, and frames within a storm are dependent.
  - A frame-level test would treat a long storm as hundreds of independent samples.
  - Equal storm weights stop one long storm from dominating. **Open item for the adversary:** a frame-weighted storm-level statistic is the alternative.
- **The floor:** with n storms the smallest possible p is 2^(1−n). **At least 6 storms are needed for p < 0.05 to be reachable.** With fewer than 6 after §3, the test is reported as descriptive only, and the reading says so.

**Reading, fixed now:** N5 / F-31 categories, with SESOI Δ (§6).

| Result | Condition | Reading |
|---|---|---|
| **BETTER** | p < 0.05, d̄ < 0, CI reaches beyond −Δ | v2 is better on 2026 Atlantic storms |
| **WORSE** | p < 0.05, d̄ > 0, CI reaches beyond +Δ | v2 is worse |
| **SIGNIFICANT BUT BELOW SESOI** | p < 0.05, CI inside ±Δ | real but smaller than what matters |
| **NULL (informative)** | p ≥ 0.05, CI inside ±Δ | no difference of Δ either way |
| **NULL (inconclusive)** | p ≥ 0.05, CI reaches ±Δ | no conclusion |

**Decision rule** (open item for John; recommendation):
- v2 replaces the frozen ensemble **only on BETTER**.
- On every other reading, the frozen ensemble stays, and v2 is reported as tested.
- A replacement is its own committed decision, with new I3/I6 pins. It is not automatic.

**Secondaries (α = 0.05 each, uncorrected, labelled; never replace the primary):**
- **S1:** > 96 kt MAE, V − F, over storms with ≥ 10 frames > 96 kt. It is tested only if there are ≥ 6 such storms; otherwise it is descriptive.
- **S2:** > 96 kt bias, V − F, on the same storms and the same rule. The frozen ensemble's high-end low bias (H9 −7.7 kt) is the known weakness.

**Descriptive (not tests):**
- the pooled-over-frames MAE, fixes-only MAE and bias for each model, and their difference;
- per-band MAE and bias;
- per storm by ATCF ID;
- each ensemble's members' mean, sd and range;
- the reweighted-to-H9-band-shares figure, as in 2025;
- a Cat 5 (≥ 137 kt) breakout, reported separately, since validation is blind there;
- F's 2026 figures beside its 2025 and H9 figures (descriptive drift).

## 6. SESOI and power

**SESOI, fixed on what would matter.** It is not taken from F-31.
- **What it is for:** the decision the test serves is whether to replace a frozen, tested ensemble, with new pins, a new site build and a new reference. A difference worth that should be visible against what the labels can resolve. Best-track intensities are given to 5 kt and are themselves uncertain by several knots.
- **Recommended Δ = 1.0 kt overall MAE** (open item for John). That is about 12% of the frozen ensemble's overall MAE level (8.6 kt on H9), and a fifth of the label resolution.
- **Alternative: Δ = 0.5 kt.** The power below shows what each costs.

**Power over storms** (`power.py`, output in `power_output.txt`):
- **Model:** d_s ~ Normal(−δ, sd); exact sign-flip at α 0.05; 2000 reps; seed 20261021.
- **The planning sd is not known.**
  - The only paired per-storm spread available is H9's: F-31's control 4× − 1× member-mean per-storm MAE has sd **0.47 kt** over 9 storms.
  - That figure is **H9-derived**, from the same code, seeds and satellite (GOES-16). It is a lower bound for 2026, where V and F also differ in code version and seeds and the storms are on GOES-19.
  - So the grid runs from 0.5 to 2.0 kt.

| n storms | Δ 1.0, sd 1.0 | Δ 1.0, sd 1.5 | Δ 0.5, sd 1.0 | **H9-derived 0.48, sd 1.0** | **H9-derived 0.48, sd 1.5** |
|---|---|---|---|---|---|
| 6 | 0.35 | 0.18 | 0.11 | 0.10 | 0.07 |
| 8 | 0.67 | 0.35 | 0.22 | 0.22 | 0.12 |
| 10 | 0.80 | 0.45 | 0.30 | 0.26 | 0.14 |
| 12 | 0.87 | 0.55 | 0.36 | 0.35 | 0.17 |
| 14 | 0.93 | 0.62 | 0.40 | 0.39 | 0.21 |
| 16 | 0.96 | 0.69 | 0.44 | 0.43 | 0.22 |

- **The planning effect:** 0.48 kt is F-31's H9 ensemble gain (20-member control ensemble, 4× vs 1×). It is **H9-derived, a planning effect only, not an expectation for 2026**.
- **What to expect, stated before the labels are in:**
  - If v2's true 2026 gain is near the H9-derived 0.48 kt, the most likely outcome at any plausible storm count is **NULL (inconclusive)**.
  - Power reaches about 0.8 only for a 1-kt gain, with a between-storm sd near 1 kt and about 10 or more storms.
  - John should weigh this before committing. An inconclusive 2026 result is still the result (§9), and the frozen ensemble stays.

## 7. What gets written (later, by the pass driver; nothing now)

- **Driver:** `scripts/run_2026.py` and `scripts/report_2026.py`, specification only. They are written in the protocol PR, tested on synthetic data, and reviewed by hc-adversary before the protocol is committed.
- **Sealed root:** builds go to `data/2026_sealed/`, as H-EP's `data/hep_sealed/` do. The crops are reachable only through `--processed-root`.
- **Outputs:**
  - `results/final_2026/{F,V}/ensemble_2026.csv`, `members_sha256_at_run.txt` and the metrics;
  - `results/final_2026/primary.csv` and the report;
  - `logs/final_2026/` and `pass_manifest.json`.
- **`.gitattributes`:** `results/final_2026/** -text` and `logs/final_2026/** -text`, plus a coverage test (`git check-attr text` = unset on every committed output; the PR #29 gap).
- **Refusals:**
  - any existing output directory;
  - any storm outside `storms_test_2026.txt`;
  - members not matching their file;
  - any path that is not pinned.

## 8. Order of operations (each committed before the next)

1. **V frozen:** its freeze entry is committed (v2 design §4). F is already frozen.
2. **This protocol committed** as a decisions.md entry, **before any 2026 build, and before the H-EP pass if V is E3's comparator** (v2 design §4). It includes:
   - John's label choice;
   - Δ, the decision rule and the EP choice;
   - the list rule;
   - the members files' blob hashes;
   - the driver and report blob hashes.
3. **E3 settled (John's choice; recommended):** the 2026 pass waits until E3's design and checkpoints are frozen, or its slot has closed at its deadline (about end-January 2027).
   - **Reason:** then no 2026 result can shape E3's design, its budget or its comparator.
   - **Cost:** under option (B) the pass moves from when the labels are ready (about December 2026) to about February 2027. Under option (A) there is no cost, since the HURDAT2 update comes later anyway.
   - **Alternative:** no hold. Only step 2 orders the two tests: the protocol is committed before the H-EP pass. The 2026 pass may then run before E3 is frozen, and a 2026 v2-vs-frozen result could be seen while E3's design is still open.
   - If 2026 EP storms are included (§4), the hold is required, not a choice.
4. **Labels available** under the chosen option (§2). The list rule is applied and `storms_test_2026.txt` is committed, with only header and status reads.
5. **Build** into `data/2026_sealed/` with `--bucket noaa-goes19`. Then the NaN exclusion step, with carry-forward (as H-EP gate 7). Frames per storm are counted, and rule 3 drops are committed. Still no model.
6. **Domain shift:** 2026 vs H9, before any model runs.
7. **The pass:** F and V in one driver invocation, then the report, then a commit of `results/final_2026/`.
8. **Spend in code:** the 2026 IDs join TCDataset's spent set through the registry role, with a `check.py` pin of `results/final_2026/` (as I3 / I3b).

**Timeline:**
- **Option (B):** about February 2027 with the hold in step 3; as soon as the labels are ready without it.
- **Option (A):** after the HURDAT2 update (**[fact sheet]**).
- No step before 5 touches 2026 imagery. Step 5 needs John's explicit yes, since I4 is lifted only for this build.

## 9. If the result is poor, or a bug is found

- **The result stands:**
  - no retraining or tuning on 2026 data;
  - no second pass;
  - no change of members, averaging or preprocessing.
- **A bug in the evaluation code** found afterwards is fixed and rerun, and both runs are reported with the reason.
- **No later model selection:** the 2026 set is never used to select a model.

## 10. Open items for John

1. **Labels:** (A) or (B) (§2).
2. **SESOI:** Δ = 1.0 kt recommended, or 0.5 kt (§6).
3. **The decision rule:** replace only on BETTER (recommended) (§5).
4. **2026 EP storms:** excluded (recommended) (§4).
5. **Hold the 2026 pass until E3 is settled** (recommended), or rely on the protocol commit before the H-EP pass alone (§8 step 3).
6. **Subtropical-only storms** (§3), and under (B), any declared re-scoring against final HURDAT2 (§2). Recommend no to both.

## Files in this draft

- `protocol_draft.md`: this file.
- `power.py`: the §6 table. CPU only; it reads nothing. Run it with `.venv\Scripts\python.exe -I -B reports/v2_survey/2026_clean_test/power.py` (about 10 s).
- `power_output.txt`: its output.
