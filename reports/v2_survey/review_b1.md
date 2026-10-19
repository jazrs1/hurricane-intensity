# B1: leakage review of the v2 survey (reports/v2_survey/survey.md)

Written 2026-10-18 by hc-reviewer. Read-only, except for this file. No downloads, no model loads, nothing switched in the main checkout.

**Verdict: ISSUES (minor, none blocking).** The proposal has no ID or alias leakage into H9, `storms_val_expanded.txt`, `storms_test_2025.txt` or the proposed H-EP. Every factual claim I recomputed about our lists holds. There is one weak leakage path the report does not cover: a proposed training storm (Marco 2020) shares crop area with a validation storm (Laura 2020) for 8 hours. There are also four clarifications for John before the lists are written.

## Method

- Inputs, hashed first:
  - `data/interim/hurdat2_atlantic.txt`: `ee35819c...0bcb5c`, matches.
  - `reports/v2_survey/_scratch/hurdat2-nepac-1949-2025-092926.txt`: `caece626...aa0507`, matches.
- Both files were parsed by my own HURDAT2 parser (not `hurricane_intensity.hurdat2`). Lists were read from the main checkout with `hurricane_intensity.splits.load_storm_list`.
- Scripts are in the hc-reviewer session scratchpad (`b1/b1_check.py`, `b1/b1_overlap.py`), run with `.venv\Scripts\python.exe -I`. Their full output is quoted where used.
- **The proposed v2 training set**, as I read it from §5:
  - Atlantic: every 2018–2024 Atlantic storm in no list with peak >= 64 kt, minus AL022022. That is 33 IDs, with Pablo (optional) and Rebekah included, to be conservative.
  - East Pacific: the 22 named EP storms, 13 GOES-17 and 9 GOES-18. All 22 IDs resolve to the right names in the nepac file.
  - The effective training set of `storms_train_expanded.txt` is carried along, since v2 extends it.
- **H-EP:** EP052025, EP062025, EP112025, EP162025, EP112023 and EP182023, plus CP012025 (optional), treated as held-out.

## 1. §1 "in no split": holds

```
12 storms_*.txt files, 39 distinct IDs
total 110; >=64 kt overall 34; >=64 tropical-only 33; TS 69; <34 7
only-overall >=64: [('AL192019', 'REBEKAH')]
majors (>=96 overall): [('AL022022', 'BONNIE', 100), ('AL072021', 'GRACE', 105), ('AL272020', 'EPSILON', 100), ('AL282020', 'ZETA', 100)]
```

All the counts match the report: 110 storms, 34 hurricanes (33 counting the tropical phase only), 69 tropical storms and 7 weaker. The three unused Atlantic majors are Grace, Epsilon and Zeta. Bonnie is the fourth, correctly routed to EP042022.

## 2. Disjointness: clean by ID and by alias

```
v2 additions & H9 / val_expanded / test_2025 / H-EP: [] [] [] []
v2 additions + expanded training & H9 / val_expanded / test_2025 / H-EP: [] [] [] []
v2 additions already in some list: []
```

**Aliases.** I searched every 2017–2025 storm in both files for any two IDs with fixes at the same time and centres within 75 km. The search finds exactly the two pairs the report names, and no others:

```
AL022022 BONNIE == ['EP042022']
AL132022 JULIA == ['EP182022']
alias-closed: train & H9 / val_expanded / test_2025 / H-EP: [] [] [] []
alias not itself listed: AL022022 BONNIE, EP182022 JULIA
```

**Same-name pairs.** These are pairs across all listed and proposed IDs. The IDs differ in every case:

```
BERYL      AL022018 [v2-train]; AL022024 [gradcam, holdout_h6, holdout_h9, storms_train]
ERICK      EP052025 [H-EP]; EP062019 [v2-train]
HELENE     AL082018 [v2-train]; AL092024 [train_expanded]
HUMBERTO   AL082025 [test_2025]; AL092019 [train_expanded]
ISAAC      AL092018 [v2-train]; AL102024 [v2-train]
KIKO       EP112025 [H-EP]; EP132019 [v2-train]
LESLIE     AL132018 [v2-train]; AL132024 [v2-train]
OSCAR      AL162018 [v2-train]; AL162024 [v2-train]
```

Beryl, Erick, Kiko and Humberto each pair a training storm with a held-out storm under the same name. They are safe only while everything is keyed on ATCF ID, which `load_storm_list` and `check.py` already do.

## 3. 2026 / 2025: holds

```
2026 IDs anywhere: []
2025 Atlantic in v2 training: []
any 2025 in v2 training: []
```

## 4. Dora EP052023: training only

```
['v2-train'] | in forbidden: []
```

## 5. Spot checks against the nepac file: all match the report

```
EP182023 OTIS      peak 145 at 10-25 0300/0600; track 10-21 - 10-25; >=96: 10-24 18 - 10-25 06, 3 synoptic (18 h)
EP112023 JOVA      peak 140 at 09-07 0000/0600; track 09-04 - 09-14; >=96: 09-06 18 - 09-08 06, 7 synoptic (42 h)
EP122024 KRISTY    peak 140 at 10-24 1800/10-25 0000; track 10-21 - 10-29; >=96: 10-23 12 - 10-25 18, 10 synoptic (60 h)
EP052023 DORA      peak 130 at 08-06 0000/0600; track 07-31 - 08-17; >=96: 08-03 00 - 08-12 00, 33 synoptic (198 h)
EP112025 KIKO      peak 125 at 09-04 0000/0600; track 08-31 - 09-11; >=96: 09-03 18 - 09-07 12, 16 synoptic (96 h)
(also Bonnie EP042022, Priscilla EP162025, Erick EP052025: match)
>=137 kt synoptic fixes: Jova 2 (12 h), Otis 1 (6 h), Kristy 2 (12 h): match
EP/CP >=96 kt 2019-2025: 29, including CP012025: matches
```

**Satellite handover dates.** I could not check S1, S4 or S5 against their primary sources, because this task allowed no downloads. None of the split choices depends on the exact dates:

```
last EP major fix before 2023-01-04 18Z: EP192022 ROSLYN 2022-10-23 12Z
first EP major fix after:                EP032023 CALVIN 2023-07-14 12Z
```

- **GOES-17 to GOES-18:** any handover date between 2022-10-23 and 2023-07-14 gives the same assignment of every candidate.
- **GOES-17 start:** 2019-02-12 precedes the first candidate major, Barbara on 2019-07-02.
- **GOES-19:** 2025-04-07 matches `logs/decisions.md:1403`. That is an internal record, not a primary source. It precedes Erick 2025, in June.

**Unverified items that the splits depend on: none.** The others affect how storms are built or filtered, not which split they go in:
- **GOES-17 LHP season windows and per-band DQF thresholds:** these only affect which training frames are filtered. GOES-17 never enters val or test, so the split doesn't depend on them.
- **GOES-18 pre-operational lon_0 (Darby, Orlene, Roslyn):** this affects the build only.

## 6. Issues

### I-1 (leakage, minor): Marco AL142020 (proposed training) shares crop area with Laura AL132020 (validation)

```
AL142020 MARCO vs AL132020 LAURA [storms_val.txt, storms_val_expanded.txt, ...]:
closest 733 km (dx 477, dy 556) at 2020-08-25 18Z; hours with overlapping 600 km boxes: 8;
other centre inside crop (dx,dy<300): 0
```

This is the only pair, among all proposed and existing training storms against every held-out or validation storm, whose centres come within 1000 km at the same hour. Positions are from hourly linear interpolation of tropical fixes, on a flat-earth box test, so the numbers are approximate.

For about 8 hours on 2020-08-25, the corner of a Marco training crop would contain outer cloud from Laura, which drives checkpoint selection. Neither centre is inside the other's crop, so the effect is small. Still, `load_splits` and the I5 test cannot see it, and the report doesn't mention it.

Proposed fix: either drop the Marco frames whose box overlaps a val, H9 or H-EP storm's box, or keep them and record the exception. Either way, v2 should add a concurrency check, meaning any training box that overlaps a held-out box at the same time, next to the alias check.

### I-2 (clarity): Rebekah AL192019 is inside "all remaining hurricanes in §1"

Its 70 kt peak is extratropical. The pipeline trims EX fixes, so its kept frames peak at 45 kt (SS). §1 says it is "not really a hurricane candidate", but §5 doesn't exclude it. Recommend listing it as excluded explicitly. This is not leakage.

### I-3 (checks): ID-only checks can't see aliases, and the report's I5 status is stale

- **Stale status:** §5 says the I5 wording is "still open with John". John approved it on 2026-10-18.
- **Alias test:** the approved I5 test in PR #1 is keyed on ID. v2 should add a test that the two alias pairs are each listed under one ID only: EP042022 (never AL022022) and AL132022 (never EP182022). The concurrency check from I-1 belongs in the same place.
- **New lists:** each new v2 `storms_train*.txt` needs a `VAL_LIST_FOR` entry in `tests/test_list_invariants.py`. Until it has one, the test fails closed, by design.

### I-4 (comparability): notes for John, not leakage

- **Two changes at once:** v2 changes the training set along two axes, about +33 Atlantic and +22 EP storms. A v2 H9 result against the ensemble can't separate the effect of storm count from the effect of basin. An Atlantic-only v2 arm would separate them. H9 and val_expanded themselves stay unchanged, as proposed.
- **HURDAT2 re-pin:** re-pinning the Atlantic file (§5 gap d) would change labels for existing storms and break comparability with every earlier number. The report already flags this.
- **H-EP satellite mix:** under the "better view" rule, Erick 2025 and Otis come from GOES-East, so H-EP would mix three satellites. The rule should be fixed before H-EP is built.
- **Bonnie EP042022:** its nepac track has 9 tropical fixes east of 90W, in the Caribbean from 78.5W. A single GOES-West bucket for the whole storm would view those at a high angle. The boundary-storm rule needs to cover this case within one storm.

## Checks that passed

§1 counts and majors; ID disjointness; alias disjointness; same-name IDs distinct; no 2026 IDs; no 2025 storm in training; Dora training-only; five or more EP peaks, dates and >= 96 / >= 137 kt hours; 29 EP/CP majors; both input hashes.
