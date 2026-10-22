# 2026 clean test: label-source facts

Fact sheet for the 2026 clean-test protocol draft (hc-survey). It covers the label choice John asked to have laid out: wait for the post-season HURDAT2 update, or use operational best track with a stated caveat.

- Written by hc-reviewer, 2026-10-21. The season counts are as of **2026-10-21 20:27 UTC**.
- Research only:
  - no 2026 imagery and no imagery listings (I4);
  - nothing downloaded into the repo;
  - no train.py or evaluate.py run.
- Web files (b-decks, a-decks, HURDAT2 tails, HTTP headers) were streamed to stdout and never saved. The commands are in the appendix.
- **This file is 2026 label information** (peaks and >96 kt counts). It plays the same role for 2026 that decisions.md:1401 played for 2025: anyone designing the test with it has seen the peaks.

## Summary

| route | label source | expected date | Atlantic storms > 96 kt | East Pacific storms > 96 kt |
|---|---|---|---|---|
| A. Operational | ATCF b-deck (operational working best track), with a caveat | now (season open to Nov 30) | **1**: Isaias AL092026, 105 kt | **9** EP-ID storms, plus 1 CP-ID storm (Lala) |
| B. Per-storm TCR | TCR post-analysis best track | storm by storm. For a hurricane, by precedent about 2.5-5 months after the storm, sometimes later. For Isaias: roughly late Dec 2026 to late Mar 2027 | not knowable yet; operationally 1 | not knowable yet |
| C. HURDAT2 | annual HURDAT2 update | late Feb to mid-June 2027, most years April-May (precedent below) | not knowable yet; operationally 1 | not knowable yet |

- The Atlantic route A count is 1 storm with 15 hourly frames > 96 kt.
- The Atlantic set is thin on any route.
- Across both basins, the three storms one 5 kt step above the lowest Cat 3 value all peaked at 105 kt: Isaias (AL), Rachel and Trudy (EP).
- No published rate was found for how often post-analysis moves a storm across the Cat 3 line. A two-season check in this repo's terms (2024-25, 31 Atlantic storms) found that peaks moved in 9-10 storms, but **none crossed 96 kt**. It is not a citable rate (see §3).

## 1. The three label sources

### A. Real-time advisories and operational best track (ATCF b-deck)

**What it is:**
- NHC's real-time best track, kept in the ATCF "b-deck" (`https://ftp.nhc.noaa.gov/atcf/btk/bal092026.dat` etc.).
- It holds 6-hourly positions and intensities in 5 kt steps, with status codes. Each synoptic time can have several rows, one per wind-radii threshold.
- The advisories' intensities come from the same working analysis.

**When it's available:**
- During and right after each storm.
- Isaias's b-deck was last modified 2026-10-11 14:59 GMT. Its last fix is 2026-10-11 12Z.

**How it differs from the final values:**
- NHC's glossary says the best track is "based on a post-storm assessment of all available data" and "may differ from values contained in system advisories" ([NHC glossary](https://www.nhc.noaa.gov/aboutgloss.shtml), accessed 2026-10-21).
- The final best track uses data not available in real time.
- The Michael (2018) TCR is the textbook case: 135 kt operationally, 140 kt (Cat 5) at landfall in post-analysis. The same TCR gives a typical best-track intensity uncertainty of about ±10% ([AL142018 TCR](https://www.nhc.noaa.gov/data/tcr/AL142018_Michael.pdf), accessed 2026-10-21).

**Not stable: NHC overwrites the b-deck with the final best track after post-analysis.**
- The archived 2025 b-decks (`https://ftp.nhc.noaa.gov/atcf/archive/2025/bal{nn}2025.dat.gz`) are all Last-Modified 2026-03-06 18:58 GMT.
- For all six 2025 test storms, their peaks equal HURDAT2's. That includes Dexter at 70 kt, which was 50 kt operationally (§3).
- So "operational best track" has no fixed version unless the protocol records a copy and its SHA-256 at a stated time. The live b-deck file today is not the same file it will be in March.
- The operational values stay recoverable from the a-decks:
  - the official forecast `OFCL` at τ = 0 is the advisory intensity;
  - `CARQ` at τ = 0 is the pre-advisory analysis.
  - Both live in `https://ftp.nhc.noaa.gov/atcf/archive/<yr>/aal{nn}<yr>.dat.gz`, which is what §3 uses.

### B. Per-storm Tropical Cyclone Reports (TCRs)

**What it is:**
- NHC's post-storm report for each storm, with the post-analysis best track (six-hourly) and the reasoning behind it ([NHC data archive](https://www.nhc.noaa.gov/data/), TCR section, accessed 2026-10-21).
- It is the first release of the final values for that storm.

**When it's available:**
- Storm by storm, through winter and spring.
- The [2026 Atlantic TCR index](https://www.nhc.noaa.gov/data/tcr/index.php?season=2026&basin=atl) (accessed 2026-10-21) lists 9 storms and no TCRs yet. It still marks Isaias "Active TC", although its b-deck ends 2026-10-11 12Z. **The b-deck is the authority for the counts here.**

**Precedent:** server `Last-Modified` of the 2025 Atlantic TCR PDFs, from HEAD requests on 2026-10-21.

| storm | TCR Last-Modified (GMT) | peak (HURDAT2) |
|---|---|---|
| Andrea AL012025 | 2025-08-12 | 35 |
| Dexter AL042025 | 2025-10-27 | 70 |
| Barry AL022025 | 2025-11-13 | 40 |
| Fernand AL062025 | 2025-11-14 | 50 |
| Lorenzo AL122025 | 2025-11-20 | 50 |
| Karen AL112025 | 2025-12-04 | 50 |
| Humberto AL082025 | 2025-12-11 | 140 |
| Jerry AL102025 | 2025-12-17 | 55 |
| Gabrielle AL072025 | 2025-12-17 | 120 |
| Imelda AL092025 | 2026-01-27 | 80 |
| Erin AL052025 | 2026-01-30 | 140 |
| Melissa AL132025 | 2026-05-19 (index page: "updated") | 165 |
| Chantal AL032025 | 2026-06-29 (index page: "updated") | 50 |

- `Last-Modified` is the latest revision, not the first release. It is an upper bound on when each TCR was first available.
  - Two 2024 examples: Helene's TCR shows 2026-02-17 and Milton's 2026-07-30. Both are long after HURDAT2 2024 (April 2025).
  - True first-release dates aren't recoverable from the headers.
- A TCR revision doesn't necessarily move the best track. Melissa's block (`AL132025` to end of file) is byte-identical (md5 `fb7dda8d…`) in `hurdat2-1851-2025-02272026.txt` and `hurdat2-1851-2025-092326.txt`, despite the May 19 TCR update.
- **For Isaias:** the 2025 hurricanes other than Melissa were available about 2.5 months (Dexter, Gabrielle, Humberto) to 5 months (Erin, last HURDAT2 fix Aug 27, TCR Jan 30) after the storm. For Isaias (ended Oct 11), that suggests roughly late Dec 2026 to late Mar 2027. It is precedent, not a schedule.

### C. The annual HURDAT2 update

**What it is:**
- The season's TCR best tracks, assembled into the HURDAT2 text file, sometimes with reanalysis revisions to old seasons.
- It is versioned by filename (`hurdat2-1851-<lastseason>-<MMDDYY or MMDDYYYY>.txt`).
- This repo pins by filename and SHA-256 (decisions.md:9).

**When it's available:** the precedent table below comes from the [HURDAT2 directory listing](https://www.nhc.noaa.gov/data/hurdat/) (accessed 2026-10-21). "Filename date" is the version date in the name. "Posted" is the server's Last-Modified.

| season | Atlantic file | filename date | posted | East Pacific file | filename date | posted |
|---|---|---|---|---|---|---|
| 2017 | hurdat2-1851-2017-050118 | 2018-05-01 | 2018-05-01 | hurdat2-nepac-1949-2017-050418 | 2018-05-04 | 2018-05-11 |
| 2018 | hurdat2-1851-2018-051019 | 2019-05-10 | 2019-05-17 | hurdat2-nepac-1949-2018-070119 | 2019-07-01 | 2019-07-09 |
| 2019 | hurdat2-1851-2019-042820 | 2020-04-28 | 2020-04-28 | hurdat2-nepac-1949-2019-040820 | 2020-04-08 | 2020-04-09 |
| 2020 | hurdat2-1851-2020-052921 | 2021-05-29 | 2021-06-10 | hurdat2-nepac-1949-2020-043021a | 2021-04-30 | 2021-05-03 |
| 2021 | hurdat2-1851-2021-040822 | 2022-04-08 | 2022-04-11 | hurdat2-nepac-1949-2021-040622 | 2022-04-06 | 2022-04-07 |
| 2022 | hurdat2-1851-2022-04042023 | 2023-04-04 | 2023-04-04 | hurdat2-nepac-1949-2022-04042023 | 2023-04-04 | 2023-04-04 |
| 2023 | hurdat2-atl-1851-2023-042624 | 2024-04-26 | 2024-04-30 | hurdat2-nepac-1949-2023-042624 | 2024-04-26 | 2024-04-30 |
| 2024 | hurdat2-1851-2024-040225 | 2025-04-02 | 2025-04-03 | hurdat2-nepac-1949-2024-031725 | 2025-03-17 | 2025-03-19 |
| 2025 | hurdat2-1851-2025-02272026 | 2026-02-27 | 2026-03-05 | hurdat2-nepac-1949-2025-02272026 | 2026-02-27 | 2026-03-05 |

- **`hurdat2-atl-02052024.txt` is not a 2023 release.** It is posted 2024-02-07 and its last rows are 2022-11-11 (tail read on 2026-10-21). The first file with the 2023 season is the 2024-04-26 one.
- **Expected date for a file containing 2026:**
  - Atlantic: **late February to mid-June 2027**, most years April to May. Over 2017-2025, Atlantic files were posted between Mar 5 and Jun 10, and the earliest was the most recent (2025: Mar 5).
  - East Pacific: the same window, plus the one July case (2018).
- Later versions in the same year revise old seasons:
  - [The data archive page](https://www.nhc.noaa.gov/data/) (accessed 2026-10-21) now points to `hurdat2-1851-2025-092326.txt`: "updated on September 23rd, 2026 to include the revisions of the 1971 through 1975 hurricane seasons".
  - For the East Pacific it points to `hurdat2-nepac-1949-2025-092926.txt`: "updated on September 29th, 2026 to include revisions of Hurricane Olivia in 1971".

**How it differs from TCRs:** HURDAT2 normally carries the TCR best track. Revisions after the annual release are possible but usually concern old seasons. No 2024-25 instance of a post-release change to a recent storm was found here (Melissa above being the one checked).

**Local copy (read only; not refreshed; no `--force-hurdat2-refresh`):**
- `data/interim/hurdat2_atlantic.txt`:
  - `head -1` is `AL011851,            UNNAMED,     14,`;
  - mtime 2026-08-18 17:08:23 −0400;
  - SHA-256 `ee35819c…` (the 2025 test's pinned hash, decisions.md:1896);
  - last storm in the file is Melissa AL132025; no 2026 storm.
- `hurdat2_atlantic.txt.source.json`: `https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt`, fetched 2026-08-18T21:08:23Z.
- So the 2025 test (built 2026-10-15/16, decisions.md:2077) was labelled from the Feb 27, 2026 HURDAT2.
  - That file came about 3 months after the season ended (Nov 30, 2025). The build came about 7.5 months after the file.
  - The file predates two current TCR revisions: Melissa (May 19) and Chantal (Jun 29). Both storms' blocks are unchanged between the Feb 27 file and the Sep 23 one: Melissa by the md5 match above, Chantal by `diff` of its block (2026-10-21).
- The resolver scrapes the index page, which now points to the Sep 23 file. A refresh today would change the SHA away from `ee35819c…` without adding any 2026 storm.
- The nepac file used by H-EP is pinned in `reports/v2_survey/hep/protocol.md`: `hurdat2-nepac-1949-2025-092926.txt`, SHA-256 `caece626…`. Its B-v2-2 cache is a pending gate there.

## 2. Expected date for each route

| route | expected date | basis |
|---|---|---|
| A. Operational b-deck | available now for finished storms; the season count is final only after Nov 30, 2026 | live b-decks; season end |
| B. TCR | storm by storm. A hurricane is typically about 2.5-5 months after the storm, sometimes later (upper bounds above). Isaias: roughly late Dec 2026 to late Mar 2027 | 2025 TCR Last-Modified dates |
| C. HURDAT2 | late Feb to mid-June 2027; most years April-May | 2017-2025 release table |

## 3. 2026 storms at the line, and how often post-analysis moves a peak

### Atlantic (as of 2026-10-21 20:27 UTC)

- There are 9 storms, AL01-AL09 2026, plus invest AL93 (not a storm).
- The operational b-deck peaks are:
  - Arthur 40, Bertha 50, Cristobal 40, Dolly 35, Edouard 50, Fay 60, Gonzalo 45 and Hanna 40 kt;
  - **Isaias AL092026, 105 kt**.
- Source: `https://ftp.nhc.noaa.gov/atcf/btk/bal{01..09}2026.dat`, accessed 2026-10-21. The peak is the maximum over unique synoptic times.

| ATCF | name | operational peak | 6-hourly fixes ≥ 100 kt | est. hourly frames > 96 kt | lon at > 96 kt |
|---|---|---|---|---|---|
| AL092026 | Isaias | 105 kt (2026-10-09 12Z and 18Z) | 2 | 15 | −87.6 to −87.1 |

- Isaias's sequence is 85, 90, **105, 105**, 95 kt (09/00Z to 10/00Z). It is a Cat 3 for two synoptic times only.
- Isaias is **one 5 kt step above the lowest Cat 3 value** (100 kt). A −10 kt post-analysis change at both of its 105 kt fixes would make it a Cat 2.
- All other 2026 Atlantic storms are ≤ 60 kt, so none is near the line.
- Hourly frames are estimated through the repo's `build_sample_grid(fixes, "1h")` on the b-deck fixes with statuses LO,TD,SD,TS,SS,HU. They are an estimate of the label count only: 1 h samples with no scan matching, and no imagery seen.
- Isaias has 15 frames > 96 kt, so the per-storm statistic threshold (≥ 10 frames > 96 kt, decisions.md:1963; `analyze_n5.py:91`) would apply operationally. That threshold is sensitive to post-analysis. The est. frames > 96 kt, computed the same way with the 105 kt fixes revised:

| revision | frames > 96 kt |
|---|---|
| none | 15 |
| 12Z → 100 | 14 |
| 12Z → 95 | 11 |
| 18Z → 100 | 14 |
| **18Z → 95** | **9** |
| both → 100 | 13 |
| both → 95 | 0 |

- A −10 kt revision at the 18Z fix alone takes Isaias below the per-storm threshold; the same revision at 12Z does not.
- It stops being a Cat 3 only if both fixes drop to ≤ 95.

**Name reuse:** the 2026 Atlantic names are the 2020 list.
- Isaias AL092020 and Hanna AL082020 are in `storms_n5_new.txt` / `storms_train_n5_atlantic.txt`.
- Fay AL062020 is in `storms_train.txt` and the other training lists.
- As with Humberto 2025, everything must key on ATCF ID.

### East Pacific and Central Pacific (as of 2026-10-21 20:27 UTC)

- There are 22 EP-ID systems; Twenty-Two-E is still active at 35 kt. There are 2 CP-ID systems.
- Source: `https://ftp.nhc.noaa.gov/atcf/btk/bep{01..22}2026.dat` and `bcp{01,02}2026.dat`, accessed 2026-10-21.

| ATCF | name | operational peak | 6-hourly fixes ≥ 100 kt | est. hourly frames > 96 kt | ≥ 137 kt frames | lon at > 96 kt |
|---|---|---|---|---|---|---|
| EP072026 | Genevieve | 140 | 13 | 76 | 5 | −118.7 to −108.7 |
| EP112026 | Karina | 125 | 15 | 89 | 0 | −136.2 to −123.3 |
| EP122026 | Lowell | 140 | 23 | 139 | 8 | −163.8 to −150.8 |
| EP152026 | Nolo | 135 | 11 | 67 | 0 | −179.2 to −158.8 |
| EP162026 | Odalys | 115 | 5 | 31 | 0 | −124.1 to −123.7 |
| EP172026 | Polo | 155 | 28 | 169 | 60 | −114.1 to −101.3 |
| EP182026 | Rachel | 105 | 3 | 20 | 0 | −110.1 to −109.4 |
| EP202026 | Simon | 135 | 9 | 56 | 0 | −105.8 to −104.6 |
| EP212026 | Trudy | 105 | 1 | 7 | 0 | −102.4 |
| CP012026 | Lala | 115 | 5 | 30 | 0 | −171.3 to −168.5 |

- The other storms peaked at or below 90 kt:
  - Fausto EP062026 at 90 and Marie EP132026 at 85 are the only ones within 15 kt of the line;
  - every other storm is ≤ 60 kt.
- **Near the line, by the 2024-25 spread below (changes of 5-10 kt, one of 20):**
  - Isaias, Rachel and Trudy (105 kt) could drop below. Trudy reaches 105 at a single fix (90, 105, 90), and it has 7 hourly frames > 96 kt even operationally, below the per-storm threshold of 10.
  - Fausto (90 kt) could rise across with a +10 kt change.
- **Coverage is a protocol question, not settled here.** Longitudes are given so the protocol can decide.
  - The 2025 test's included fixes spanned −78.6 to −23.1 (GOES-19, lon_0 −75.0).
  - H-EP uses GOES-18 for every EP storm (protocol.md:188) and excludes CP IDs (protocol.md:40; Iona CP012025).
  - Lowell, Nolo and Lala are ≥ 75° of longitude from GOES-East's sub-point at > 96 kt.
  - Nolo reaches −179.2, which touches the B-v2-1 antimeridian work. None of the three b-decks has a fix at an east longitude.
  - No viewing-angle figure is asserted here; H-EP's reference-geometry method is the model for that.
- The nepac HURDAT2 is a separate file with its own version string. The repo's resolver ignores basin-tagged files (`hurdat2.py:41`), and the nepac cache is H-EP gate B-v2-2, not yet in place.

### How often post-analysis moves a peak

**Citable:**
- the NHC glossary ("may differ");
- the Michael TCR (135 to 140 kt, Cat 4 to 5; uncertainty about ±10%).

**No published rate** was found for how often post-analysis moves a peak across a category line. The searches were NHC verification reports, the glossary, TCRs and a WMO best-track chapter (all 2026-10-21). **The historical rate is unknown** as a citable figure.

**This repo's own check (reproducible, not a citation):** each 2024-25 Atlantic storm's operational peak was compared with its peak in the local HURDAT2 copy (`ee35819c…`).
- Operational peak: the maximum `OFCL` τ0 intensity in the archived a-deck, i.e. what each 6-hourly advisory gave. The maximum `CARQ` τ0 is in brackets where it differs.
- 31 storms in total.

| storm | operational (OFCL τ0) [CARQ τ0] | HURDAT2 | change | category at peak |
|---|---|---|---|---|
| Chris AL032024 | 35 | 40 | +5 | TS, same |
| Francine AL062024 | 80 | 90 | +10 | **Cat 1 to Cat 2** |
| Helene AL092024 | 120 [115] | 120 | 0 [+5] | Cat 4, same |
| Kirk AL122024 | 125 | 130 | +5 | Cat 4, same |
| Milton AL142024 | 150 | 155 | +5 | Cat 5, same |
| Nadine AL152024 | 45 | 50 | +5 | TS, same |
| Dexter AL042025 | 50 | 70 | +20 | **TS to hurricane-force, at EX fixes only** |
| Imelda AL092025 | 85 | 80 | −5 | **Cat 2 to Cat 1** |
| Karen AL112025 | 40 [45] | 50 | +10 | TS, same |
| Melissa AL132025 | 155 | 165 | +10 | Cat 5, same |
| the other 21 | equal | equal | 0 | same |

- **Result:** the peak moved in 9 of 31 storms by OFCL (10 by CARQ), 8 up and 1 down, mostly by 5-10 kt. Three changed category at peak. **None crossed 96 kt in either direction.**
- 31 storms in two seasons is too few for a rate.
- **Caveats:**
  - OFCL τ0 misses intensities given only in intermediate or special advisories and updates between cycles. Some "operational" peaks here may be 5 kt low; Melissa's 155 is one to check before relying on it.
  - The comparison uses peaks only. Frame-level labels move with every changed fix, not only at the peak.

## 4. The 2025 test set's inclusion rule, applied to 2026

**No written rule.**
- decisions.md:1401 (2026-10-14, a9d07f4) names the six storms ("2025 / GOES-19 storms: Dexter, Erin (140 kt), Gabrielle (120), Humberto (140, …), Imelda and Melissa (165 kt, …)").
- The 2026-10-15 22:57 entry (decisions.md:1882) lists them. Neither says why these six.

**Inferred rule:** the only simple rule that reproduces the six from the cached HURDAT2 is:

> every 2025 Atlantic storm whose **HURDAT2 peak is ≥ 64 kt in any row, status-blind** (EX included).

**Check on all 13 2025 storms:**
- Peaks are Erin 140, Gabrielle 120, Humberto 140, Imelda 80, Melissa 165 and Dexter 70 kt. The other seven are ≤ 55 kt.
- "Reached HU status" gives five, not six. Dexter has no HU row; its 65-70 kt rows are all EX.
- The build's status filter (LO,TD,SD,TS,SS,HU) then dropped Dexter's hurricane-force rows. It contributed 13 usable fixes, all ≤ 50 kt (logs/final_2025/build_AL042025.log).
- **Dexter qualified only through post-analysis.** Its operational peak was 50 kt (OFCL τ0, §3). Under operational labels the 2025 rule would have selected five storms.

**Other fixed parts of the 2025 build** (decisions.md:2065-2069):
- 1 h grid plus every exact fix;
- ABI-L2-CMIPF band 13 full disk;
- 600 km box, 10 min scan tolerance;
- statuses LO,TD,SD,TS,SS,HU;
- GOES-East bucket.
- Scan availability can't be checked without imagery listings (I4), so it is not applied here.

**Per-storm statistic rule:** > 96 kt MAE, bias and Spearman only for storms with ≥ 10 frames > 96 kt (decisions.md:1963, `analyze_n5.py:91`).

**Applied to 2026 (operational, as of 2026-10-21):**
- **Atlantic:** the rule selects Isaias only (every other storm ≤ 60 kt in every row). The > 96 kt count stays 1. Isaias has 15 est. frames > 96 kt, so it would get per-storm > 96 kt statistics.
- **East Pacific (EP IDs):** the rule selects Fausto, Genevieve, Karina, Lowell, Marie, Nolo, Odalys, Polo, Rachel, Simon and Trudy (11). It adds Fausto (90) and Marie (85) to the set, but changes no > 96 kt count.
  - If CP IDs are excluded as in H-EP, Lala drops: 9 EP storms > 96 kt, not 10.
  - Trudy (7 est. frames) would get no per-storm > 96 kt statistics.
- **What the rule changes:** total storm count, not the > 96 kt count. The coverage question (§3) could change the EP count; the inclusion rule as written does not.
- **On route C,** the rule is applied to HURDAT2 peaks, as in 2025. A storm can then enter late, as Dexter did, or drop out (an Imelda-type −5 kt at 65 kt).

## 5. The threshold: ">96" and "≥96"

- **Same storms by peak.** Best-track intensities are multiples of 5 kt.
  - The local HURDAT2 has 55,605 fix rows and 0 non-multiples of 5.
  - The 2026 b-decks above have 0.
  - So no peak is 96, and "> 96" and "≥ 96" both mean peak ≥ 100 kt, i.e. Category 3+ (SSHWS Cat 3 = 96-112 kt).
  - The code agrees: `analyze_n5.py:90` (`HIGH = 96`, `t > HIGH`, imported by `analyze_f31.py`), `ensemble_report.py:43` (`t > 96`), and its bands put Cat 3 at [96, 113).
- **Frame level.** Labels interpolated to image times are not multiples of 5. Exact-fix times off the 6 h grid could in principle put a frame exactly on 96.0. Checked empirically:
  - all 174 Atlantic storms of 2017-2025 in the local HURDAT2, through `parse_storm` + `build_sample_grid(1h)`;
  - 25,209 hourly frames, 2,257 > 96 and 2,257 ≥ 96;
  - **0 within 1e-6 of 96.0**.
  - Interpolation never exceeds its bracketing fixes, so a storm with peak ≤ 95 has no frame > 96. The two thresholds select the same frames on this data.
  - A different interpolation or label time grid would need rechecking.

## Appendix: commands

All commands were run on 2026-10-21 (`date -u`: Wed Oct 21 20:27 UTC). They were run from Git Bash, streaming web content to stdout; nothing was saved.

```sh
# HURDAT2 versions: directory listing (WebFetch) https://www.nhc.noaa.gov/data/hurdat/ ; current version text: https://www.nhc.noaa.gov/data/
curl -s -r -400 https://www.nhc.noaa.gov/data/hurdat/hurdat2-atl-02052024.txt | tail -2      # ends 2022-11-11
curl -s -r -40000 https://www.nhc.noaa.gov/data/hurdat/<file> | tr -d '\r' | awk '/^AL132025/{p=1} p' | md5sum

# 2026 operational peaks (unique synoptic times; field 9 = VMAX, 11 = TY, 28 = name)
curl -s https://ftp.nhc.noaa.gov/atcf/btk/bal092026.dat | awk -F', *' '{t=$3; if(t in s)next; s[t]=1; if($9+0>mx){mx=$9+0;tm=t}} END{print mx, tm}'

# operational vs HURDAT2 peaks, 2024-25 (a-deck τ0)
curl -s https://ftp.nhc.noaa.gov/atcf/archive/2025/aal042025.dat.gz | gunzip | awk -F', *' '$6+0==0 && $5=="OFCL"{v=$9+0; if(v>o)o=v} END{print o}'

# TCR dates
curl -sI https://www.nhc.noaa.gov/data/tcr/AL052025_Erin.pdf | grep -i last-modified
```

- **Hourly frame estimates:** the b-deck rows go through `hurricane_intensity.pipeline.build_sample_grid(fixes, "1h")` after the status filter `config.DEFAULT_INCLUDE_STATUSES`. The frame-level threshold check uses `hurdat2.parse_storm` over the local cache.
- **Isaias revision table (§3):** the same `build_sample_grid` b-deck script, with the 12Z/18Z (2026-10-09) `wind_kt` overridden to 100 or 95, one fix at a time and then both.
- Both were run with `.venv/Scripts/python.exe -B` and `PYTHONPATH=src`, from scripts in the session scratchpad, not added to the repo.

**Sources** (all accessed 2026-10-21):
- [NHC data archive](https://www.nhc.noaa.gov/data/)
- [HURDAT2 directory](https://www.nhc.noaa.gov/data/hurdat/)
- [2026 Atlantic TCR index](https://www.nhc.noaa.gov/data/tcr/index.php?season=2026&basin=atl)
- [2025 Atlantic TCR index](https://www.nhc.noaa.gov/data/tcr/index.php?season=2025&basin=atl)
- [Michael TCR](https://www.nhc.noaa.gov/data/tcr/AL142018_Michael.pdf)
- [NHC glossary](https://www.nhc.noaa.gov/aboutgloss.shtml)
- [ATCF b-decks](https://ftp.nhc.noaa.gov/atcf/btk/)
- [ATCF archive](https://ftp.nhc.noaa.gov/atcf/archive/)
