# v2 data and split survey (F4)

Written 2026-10-18 by hc-survey on branch `phase2-experiments`. Read-only: nothing outside `reports/v2_survey/` was written, no list file was created or edited, nothing was trained or evaluated and no model was loaded. Every scan read is dated 2025 or earlier.

This report proposes. The split decisions belong to John.

## Inputs

| Input | Provenance |
|---|---|
| Atlantic best track | `data/interim/hurdat2_atlantic.txt`, read-only, SHA-256 `ee35819cba2b61ab0c2175601084def682813727ba209c2dab7d3af71d0bcb5c` (re-hashed: unchanged). Source per `.source.json`: `hurdat2-1851-2025-02272026.txt`. |
| East Pacific best track | `reports/v2_survey/_scratch/hurdat2-nepac-1949-2025-092926.txt`, from https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2025-092926.txt, 4,087,516 bytes, SHA-256 `caece626f7038db64a803a73f5985b563693f0130f3ff16b4b0a0a93c7aa0507`, fetched 2026-10-18. |
| Split union | Every `storms_*.txt` at the repo root (12 files), read through `hurricane_intensity.splits.load_storm_list`: 39 distinct IDs. |

The scripts are in the session scratchpad, not the repo. All ran under `.venv\Scripts\python.exe -I`:
- `candidates.py`: candidate tables and name collisions.
- `geom.py`: viewing angles and GOES-17 season overlap.
- `g18_read.py`: the GOES-18 read.

Hours of major phase below are approximate: (number of 6-hourly synoptic fixes at >= 96 kt) x 6.

"Peak kt" is the maximum over all HURDAT2 records, EX included. The pipeline keeps only LO/TD/SD/TS/SS/HU (`config.DEFAULT_INCLUDE_STATUSES`). For every storm >= 50 kt here, the two peaks agree except these:
- Rebekah 2019: 70 overall vs 45 tropical, so it is not really a hurricane candidate.
- Arthur 2020, Alpha 2020, Ana 2021, Odette 2021 and Wanda 2021 (all tropical storms).

---

## 1. Remaining Atlantic GOES-16-era storms (2018–2024), in no split

**Scope.** 110 Atlantic storms from 2018–2024 are in no list:
- 34 reached hurricane strength (>= 64 kt), and 4 of those are majors (>= 96 kt). That is 33 counting the tropical phase only, since Rebekah 2019's 70 kt was extratropical.
- 69 peaked as tropical storms (34–63 kt), and 7 more as depressions or lows (< 34 kt).

**Coverage.** GOES-16 was GOES-East from 2017-12-18 until GOES-19 replaced it on 2025-04-07 [S5], so every storm below is GOES-16 only.

### Hurricanes (>= 64 kt), by peak

The "Other storms with this name" column lists same-name storms from 2017–2025 in either HURDAT2 file. The "Westmost lon" column is the westmost longitude at >= 64 kt.

| ATCF | Name | Year | Peak kt | Dates (HURDAT2) | ~h as HU | ~h >= 96 kt | Westmost lon | Other storms with this name |
|---|---|---|---|---|---|---|---|---|
| AL072021 | Grace | 2021 | 105 | 08-13 – 08-21 | 48 | 12 | 97.2W | – |
| AL022022 | Bonnie | 2022 | 100 | 07-01 – 07-11 | 114 | 6 | 119.3W | **EP042022 (same storm, see §2)** |
| AL272020 | Epsilon | 2020 | 100 | 10-16 – 10-26 | 114 | 12 | 62.1W | – |
| AL282020 | Zeta | 2020 | 100 | 10-24 – 10-29 | 54 | 0 (one 100 kt landfall fix at 21:00Z, not synoptic) | 91.7W | – |
| AL082018 | Helene | 2018 | 95 | 09-07 – 09-16 | 90 | 0 | 37.3W | **AL092024 Helene (training, expanded)** |
| AL162018 | Oscar | 2018 | 95 | 10-26 – 10-31 | 72 | 0 | 58.5W | AL162024 Oscar (unlisted) |
| AL032018 | Chris | 2018 | 90 | 07-05 – 07-12 | 48 | 0 | 73.5W | AL032024 Chris (unlisted) |
| AL062024 | Francine | 2024 | 90 | 09-08 – 09-12 | 30 | 0 | 94.7W | – |
| AL102019 | Jerry | 2019 | 90 | 09-17 – 09-28 | 36 | 0 | 61.2W | AL102025 Jerry (2025, unlisted) |
| AL102024 | Isaac | 2024 | 90 | 09-25 – 09-30 | 54 | 0 | 47.7W | AL092018 Isaac (unlisted) |
| AL132024 | Leslie | 2024 | 90 | 10-01 – 10-12 | 126 | 0 | 50.8W | AL132018 Leslie (unlisted) |
| AL172020 | Paulette | 2020 | 90 | 09-07 – 09-28 | 84 | 0 | 64.9W | – |
| AL052024 | Ernesto | 2024 | 85 | 08-12 – 08-20 | 126 | 0 | 69.3W | AL052018 Ernesto (unlisted) |
| AL152023 | Nigel | 2023 | 85 | 09-15 – 09-22 | 96 | 0 | 55.2W | – |
| AL082020 | Hanna | 2020 | 80 | 07-23 – 07-26 | 18 | 0 | 97.6W | – |
| AL092020 | Isaias | 2020 | 80 | 07-30 – 08-04 | 54 | 0 | 79.7W | – |
| AL132018 | Leslie | 2018 | 80 | 09-23 – 10-13 | 126 | 0 | 57.2W | AL132024 Leslie (unlisted) |
| AL142023 | Margot | 2023 | 80 | 09-07 – 09-18 | 84 | 0 | 40.7W | – |
| AL152022 | Lisa | 2022 | 80 | 10-30 – 11-05 | 18 | 0 | 88.6W | – |
| AL052021 | Elsa | 2021 | 75 | 06-30 – 07-09 | 30 | 0 | 83.2W | – |
| AL052022 | Danielle | 2022 | 75 | 08-31 – 09-08 | 120 | 0 | 45.4W | – |
| AL132022 | Julia | 2022 | 75 | 10-07 – 10-10 | 18 | 0 | 84.7W | **EP182022 (same storm, see §2)** |
| AL162022 | Martin | 2022 | 75 | 11-01 – 11-03 | 24 | 0 | 50.9W | – |
| AL162024 | Oscar | 2024 | 75 | 10-19 – 10-22 | 36 | 0 | 74.5W | AL162018 Oscar (unlisted) |
| AL022018 | Beryl | 2018 | 70 | 07-04 – 07-17 | 30 | 0 | 49.3W | **AL022024 Beryl (H9 held-out; also in legacy storms_train.txt)** |
| AL182019 | Pablo | 2019 | 70 | 10-25 – 10-28 | 12 | 0 | 18.8W | – |
| AL192019 | Rebekah | 2019 | 70 **(extratropical; tropical-phase peak 45 kt as SS)** | 10-30 – 11-01 | 0 (never HU) | 0 | 43.1W | – |
| AL022019 | Barry | 2019 | 65 | 07-11 – 07-16 | 6 | 0 | 92.2W | AL022025 Barry (2025, unlisted) |
| AL052023 | Don | 2023 | 65 | 07-11 – 07-25 | 12 | 0 | 50.1W | AL052017 Don |
| AL082021 | Henri | 2021 | 65 | 08-15 – 08-24 | 24 | 0 | 72.7W | – |
| AL092018 | Isaac | 2018 | 65 | 09-07 – 09-15 | 24 | 0 | 44.8W | AL102024 Isaac (unlisted) |
| AL142020 | Marco | 2020 | 65 | 08-20 – 08-25 | 12 | 0 | 87.4W | – |
| AL162020 | Nana | 2020 | 65 | 09-01 – 09-04 | 6 | 0 | 88.3W | – |
| AL252020 | Gamma | 2020 | 65 | 10-02 – 10-06 | 0 (HU only at an off-synoptic fix) | 0 | 87.4W | – |

Notes:
- **Only three Atlantic majors are still unused:** Grace 2021, Epsilon 2020 and Zeta 2020. Bonnie 2022's major phase is in the East Pacific (see §2). decisions.md (2026-10-14) already named Grace, Epsilon and Zeta as alternates "short major phases", which the table confirms: 12, 12 and 0 synoptic hours at >= 96 kt.
- **Long Cat 1–2 storms:** Leslie 2018 (126 h as HU), Leslie 2024, Ernesto 2024, Danielle 2022, Bonnie and Epsilon each add 100+ hourly Cat 1–2 frames. decisions.md dropped Leslie 2018 and Paulette for exactly that reason. 3-hourly sampling (as in `storms_train_expanded.txt`) keeps them from dominating.
- **Pablo 2019 (18.8W, about 34–45N)** is far east of GOES-16's 75.2W sub-point, at a high viewing angle. Expect smaller native crops.

### Tropical storms and depressions (peak < 64 kt): 76 storms, IDs only

AL012018 Alberto 55; AL042018 Debby 45; AL052018 Ernesto 40; AL072018 Gordon 60; AL102018 Joyce 45; AL112018 Eleven 30; AL122018 Kirk 55; AL152018 Nadine 55; AL012019 Andrea 35; AL032019 Three 30; AL042019 Chantal 35; AL062019 Erin 35; AL072019 Fernand 45; AL082019 Gabrielle 55; AL112019 Imelda 40; AL122019 Karen 40; AL142019 Melissa 55; AL152019 Fifteen 30; AL162019 Nestor 50; AL172019 Olga 45; AL202019 Sebastien 60; AL012020 Arthur 55; AL022020 Bertha 45; AL032020 Cristobal 50; AL042020 Dolly 40; AL052020 Edouard 40; AL072020 Gonzalo 55; AL102020 Ten 30; AL112020 Josephine 40; AL122020 Kyle 45; AL152020 Omar 35; AL182020 Rene 40; AL212020 Vicky 45; AL222020 Beta 55; AL232020 Wilfred 35; AL242020 Alpha 60; AL302020 Theta 60; AL012021 Ana 50; AL022021 Bill 55; AL042021 Danny 40; AL062021 Fred 55; AL102021 Kate 40; AL112021 Julian 50; AL132021 Mindy 50; AL152021 Odette 55; AL162021 Peter 45; AL172021 Rose 50; AL192021 Teresa 40; AL202021 Victor 55; AL212021 Wanda 60; AL012022 Alex 60; AL032022 Colin 35; AL082022 Gaston 55; AL102022 Hermine 35; AL112022 Eleven 30; AL122022 Twelve 30; AL142022 Karl 50; AL012023 Unnamed 60; AL022023 Arlene 35; AL032023 Bret 60; AL042023 Cindy 50; AL062023 Gert 50; AL072023 Emily 45; AL092023 Harold 50; AL112023 Jose 55; AL122023 Katia 50; AL172023 Philippe 50; AL182023 Rina 45; AL192023 Sean 40; AL212023 Twenty-One 25; AL032024 Chris 40; AL072024 Gordon 40; AL112024 Joyce 45; AL152024 Nadine 50; AL172024 Patty 55; AL192024 Sara 45

### Name collisions involving storms already in a split

**Rule: key everything on the ATCF ID, never the name.** These collisions are where a name-based lookup would go wrong.

| Listed storm (role) | Same-name storm | Risk |
|---|---|---|
| AL082025 Humberto (spent 2025 test) | AL092019 Humberto (training, expanded) | Known (FEATURES.md I3). |
| AL052025 Erin (2025 test) | AL062019 Erin (TS, unlisted) | Adding "Erin" by name would pull a test storm. |
| AL072025 Gabrielle (2025 test) | AL082019 Gabrielle (TS, unlisted) | Same. |
| AL092025 Imelda (2025 test) | AL112019 Imelda (TS, unlisted) | Same. |
| AL132025 Melissa (2025 test) | AL142019 Melissa (TS, unlisted) | Same. |
| AL022024 Beryl (H9) | AL022018 Beryl (Cat 1, unlisted candidate) | A v2 training candidate shares a name with an H9 storm. |
| AL122024 Kirk (H9) | AL122018 Kirk (TS, unlisted) | Same. |
| AL092024 Helene (training) | AL082018 Helene (95 kt, unlisted candidate) | Harmless if keyed on ID. |
| AL132019 Lorenzo (training) | AL122025 Lorenzo (2025, unlisted) | No 2025 storm should enter v2 training without a decision (see §5). |
| AL012024 Alberto, AL042024 Debby (training) | AL012018 Alberto, AL042018 Debby (unlisted) | Harmless if keyed on ID. |
| AL162023 Ophelia, AL132023 Lee, AL082023 Franklin | 2017 namesakes (AL172017, AL142017, AL072017) | Pre-2018, out of scope. |

Cross-basin name repeats also matter for §2:
- **EP 2019 vs EP 2025 (different storms, IDs differ):** Erick EP062019 and EP052025; Kiko EP132019 and EP112025; Barbara EP022019 and EP022025; Juliette EP112019 and EP102025; Flossie EP072019 and EP062025; Priscilla EP192019 and EP162025.
- **Atlantic vs EP:** Erin, Gabrielle, Melissa, Kiko and others recur across basins.

---

## 2. East Pacific Cat 3+ (>= 96 kt) candidates, 2019–2025 (HURDAT2 nepac)

### Satellite timeline [S1–S5]

| Period | GOES-West | GOES-East |
|---|---|---|
| 2018-11 – 2019-02-12 | GOES-15 (not ABI) | GOES-16 |
| 2019-02-12 18:00Z – 2023-01-04 18:00Z | GOES-17 (137.2W) | GOES-16 |
| 2023-01-04 18:00Z onward | GOES-18 (137.0W per S2 and the file's lon_0; GOES-17 was at 137.2W per S4. The VZA columns use 137.2W for both, a 0.2 deg difference that does not matter here) | GOES-16 until 2025-04-07, then GOES-19 (75.2W) |

The handover date and time come from NWS SCN 22-113 [S1] ("On or near January 4, 2023, at approximately 18:00 UTC"), the NOAA GOES-18 timeline [S2] ("January 4, 2023, 1800 UTC: declared operational as GOES-West") and the TOWR-S January news item [S3]. The GOES-17 start is from the OSPO message of 2019-02-12 [S4].

The `noaa-goes18` bucket also holds **pre-operational** GOES-18 full-disk Band 13 data in 2022. I listed keys only, no download, for 2022-07-12 12Z, 2022-10-02 12Z and 2022-10-22 12Z. GOES-18 finished drifting to 137.0W on 2022-07-21 and was certified Provisional on 2022-07-28 [S2]. Its data before the handover was labelled "GOES-18 Preliminary, Non-Operational Data" [S2]. The `noaa-goes17` bucket continues past the handover (listed for 2023-01-04 and 2023-01-05).

### Candidates

29 storms reached >= 96 kt in 2019–2025: 28 are EP IDs and one is a Central Pacific ID (CP012025 Iona).

- **VZA** is the viewing zenith angle over the 6-hourly major-phase fixes, from GOES-West (137.2W) and GOES-East (75.2W), spherical-earth approximation.
- **Better view** is whichever satellite has the lower mean VZA.

| ATCF | Name | Peak kt | Track (HURDAT2) | Major phase (>= 96 kt), UTC | ~h >= 96 | ~h >= 137 | Major-phase lat / lon | VZA West | VZA East | Better view | Satellite for major phase | Landfall flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EP022019 | Barbara | 135 | 06-30 – 07-08 | 07-02 12 – 07-04 12 | 54 | 0 | 12.3–15.7N / 121.6–129.6W | 20–23 | 55–64 | West | G17 | – |
| EP062019 | Erick | 115 | 07-27 – 08-08 | 07-30 12 – 07-31 18 | 36 | 0 | 13.2–14.6N / 142.2–148.1W | 17–21 | 76–82 | West | G17 | – |
| EP112019 | Juliette | 110 | 09-01 – 09-09 | 09-03 06 – 09-04 00 | 24 | 0 | 17.9–18.8N / 114.2–115.8W | 33–34 | 49–51 | West | G17 (LHP season) | – |
| EP132019 | Kiko | 115 | 09-12 – 09-26 | 09-15 06 – 09-16 06 | 30 | 0 | 16.9–17.3N / 120.3–122.9W | 26–28 | 55–57 | West | G17 (LHP season) | – |
| EP082020 | Douglas | 115 | 07-20 – 07-30 | 07-23 06 – 07-25 00 | 48 | 0 | 12.8–17.5N / 133.3–144.3W | 16–22 | 67–79 | West | G17 | – |
| EP122020 | Genevieve | 115 | 08-16 – 08-22 | 08-18 12 – 08-19 00 | 18 | 0 | 17.7–19.1N / 107.6–108.8W | 39–40 | 42–44 | West (boundary) | G17 (LHP season) | – |
| EP182020 | Marie | 120 | 09-27 – 10-10 | 10-02 00 – 10-04 00 | 54 | 0 | 15.4–19.0N / 121.5–127.6W | 24–26 | 55–62 | West | G17 (LHP season) | – |
| EP062021 | Felicia | 125 | 07-14 – 07-22 | 07-16 06 – 07-18 18 | 66 | 0 | 14.5–15.9N / 120.5–128.5W | 21–26 | 54–63 | West | G17 | – |
| EP122021 | Linda | 115 | 08-09 – 08-24 | 08-13 18 – 08-15 12 | 48 | 0 | 17.2–19.2N / 112.3–119.7W | 30–35 | 47–55 | West | G17 (LHP season) | – |
| EP042022 | Bonnie | 100 | 06-27 – 07-11 | 07-05 12 (one fix) | 6 | 0 | 15.1N / 103.8W | 42 | 37 | **East** | G17 (G18 pre-ops drifting) | yes (Atlantic phase) |
| EP052022 | Darby | 120 | 07-09 – 07-17 | 07-11 12 – 07-14 00 | 48 | 0 | 14.5–15.6N / 122.8–137.5W | 18–24 | 56–72 | West | G17 (G18 pre-ops, drifting until 07-21) | – |
| EP162022 | Orlene | 115 | 09-28 – 10-03 | 10-02 06 – 10-02 18 | 18 | 0 | 18.6–20.0N / 106.9–107.0W | 41 | 42–43 | boundary | G17 (LHP season); G18 provisional also on S3 | yes |
| EP192022 | Roslyn | 115 | 10-20 – 10-24 | 10-22 06 – 10-23 12 | 36 | 0 | 17.0–22.1N / 105.5–106.7W | 41–44 | 40–43 | boundary | G17 (LHP season); G18 provisional also on S3 | yes |
| EP032023 | Calvin | 110 | 07-10 – 07-19 | 07-14 12 – 07-15 00 | 18 | 0 | 13.4–14.0N / 122.6–125.3W | 21–23 | 56–59 | West | G18 | – |
| EP052023 | Dora | 130 | 07-31 – 08-17 | 08-03 00 – 08-12 00 | 198 | 0 | 11.4–15.6N / 116.2–179.8W | 15–52 | 50–112 | West | G18 | – (**crosses the antimeridian; GOES-18 e2e storm**) |
| EP072023 | Fernanda | 115 | 08-12 – 08-24 | 08-14 06 – 08-15 06 | 30 | 0 | 15.5–16.5N / 117.4–120.2W | 28–29 | 51–54 | West | G18 | – |
| EP092023 | Hilary | 120 | 08-16 – 08-20 | 08-18 00 – 08-19 12 | 42 | 0 | 16.4–21.5N / 109.5–113.5W | 36–37 | 44–50 | West | G18 | yes |
| EP112023 | Jova | 140 | 09-04 – 09-14 | 09-06 18 – 09-08 06 | 42 | 12 | 14.6–18.8N / 111.1–119.1W | 30–35 | 44–54 | West | G18 | – |
| EP152023 | Lidia | 120 | 10-03 – 10-11 | 10-10 18 – 10-11 00 | 12 | 0 | 19.3–20.1N / 105.5–107.1W | 41–43 | 42–43 | boundary | G18 | yes |
| EP172023 | Norma | 115 | 10-17 – 10-23 | 10-19 06 – 10-21 06 | 54 | 0 | 16.4–21.4N / 107.7–110.1W | 39–40 | 42–47 | West (boundary) | G18 | yes |
| EP182023 | Otis | 145 | 10-21 – 10-25 | 10-24 18 – 10-25 06 | 18 | 6 | 14.9–16.8N / 99.4–99.9W | 47 | 33–34 | **East** | G18 | yes |
| EP072024 | Gilma | 115 | 08-18 – 08-29 | 08-22 00 – 08-25 18 | 72 | 0 | 16.4–17.9N / 123.6–134.1W | 21–25 | 58–69 | West | G18 | – |
| EP102024 | John | 105 | 09-22 – 09-27 | 09-24 00 – 09-24 03 | 6 | 0 | 15.9–16.6N / 98.6–98.9W | 48 | 33 | **East** | G18 | yes |
| EP122024 | Kristy | 140 | 10-21 – 10-29 | 10-23 12 – 10-25 18 | 60 | 12 | 14.0–16.2N / 113.0–125.4W | 23–33 | 46–60 | West | G18 | – |
| CP012025 | Iona | 115 | 07-26 – 08-01 | 07-29 06 – 07-30 00 | 24 | 0 | 10.8–11.0N / 152.6–156.4W | 22–26 | 86–90 | West | G18 | – (**CP ID, not EPnnYYYY**) |
| EP052025 | Erick | 120 | 06-16 – 06-20 | 06-19 00 – 06-19 12 | 18 | 0 | 14.9–16.4N / 96.8–98.4W | 48–49 | 30–33 | **East** | G18 (East = G19) | yes |
| EP062025 | Flossie | 105 | 06-29 – 07-05 | 07-02 00 – 07-02 06 | 12 | 0 | 18.0–18.4N / 107.1–107.9W | 40 | 42–43 | boundary | G18 | – |
| EP112025 | Kiko | 125 | 08-31 – 09-11 | 09-03 18 – 09-07 12 | 96 | 0 | 13.7–17.2N / 131.1–143.9W | 16–22 | 65–78 | West | G18 | – |
| EP162025 | Priscilla | 100 | 10-04 – 10-12 | 10-07 18 (one fix) | 6 | 0 | 19.7N / 110.5W | 38 | 46 | West | G18 | – |

"Landfall flag" means a HURDAT2 `L` record anywhere on the track, not necessarily during the major phase. "LHP season" means GOES-17 major-phase fixes fall within about 50 days of an equinox (see §4).

### Boundary storms and basin crossers

**GOES-East/West boundary.** About 106W is where the two viewing angles are equal.
- **Better seen from GOES-East:** Otis 2023, John 2024 and Erick 2025 sit at 97–100W, about 47–49 deg from GOES-West and 33 deg from GOES-East. Bonnie 2022 at 103.8W is 42 vs 37 deg.
- **Near-equal angles (40–44 deg from either):** Genevieve 2020, Orlene 2022, Roslyn 2022, Lidia 2023, Norma 2023 and Flossie 2025.
- **Decision for v2:** either take boundary storms from GOES-West consistently, or from whichever satellite views them better. Mixing within one storm would be worse than either. The pipeline picks the satellite through `bucket` per storm, so either rule is a single argument.

**Basin crossers.** Both crossers below appear with their full track in both HURDAT2 files, under two different IDs:
- **Bonnie 2022:** AL022022 / EP042022. Both files run from 2022-06-27/07-01 in the Caribbean to 2022-07-11 near 137W. Its only >= 96 kt fix (07-05 12Z, 103.8W) is in the East Pacific.
- **Julia 2022:** AL132022 / EP182022. Its 75 kt hurricane peak was in the Caribbean; the EP portion was weak.

The ID-based disjointness checks (`load_splits`, I5) cannot see that two IDs are the same storm. **Rule: list each crosser under exactly one ID** (Bonnie as EP042022, Julia as AL132022), never list the other ID, and add both pairs to whatever overlap check v2 uses.

Also seen: Nana AL162020 ended at 91.4W over Central America without a separate EP ID, and Terry EP182021 began at 84.9W (35 kt, not a candidate).

**Central Pacific.**
- **Iona 2025** has a CP ID (CP012025). If v2 includes it, the "EPnnYYYY" convention needs widening to EP/CP. `hurdat2.parse_storm` already accepts CP headers.
- **Dora 2023** kept its EP ID after crossing 140W and then 180. The nepac file carries it into the West Pacific: 20 fixes in East longitudes, ending at 168.0E.

---

## 3. GOES-18 Band 13 read (done; reused by F3)

| Item | Value |
|---|---|
| Storm / fix | **EP052023 Dora**. HURDAT2 nepac fix 2023-08-06 06:00Z, 13.1N 137.4W, 130 kt (Dora's peak). |
| Bucket / key | `noaa-goes18`, `ABI-L2-CMIPF/2023/218/06/OR_ABI-L2-CMIPF-M6C13_G18_s20232180600203_e20232180609523_c20232180609587.nc` |
| Scan | Start 2023-08-06T06:00:20Z, offset +0.333 min. `platform_ID` G18, Full Disk, ABI Mode 6, `longitude_of_projection_origin` -137.0. |
| Code path | `goes_index.find_nearest_scan(bucket="noaa-goes18")` -> `crop.process_one_scan(s3, key, 13.1, -137.4, config.DEFAULT_BOX_KM (600), scratch_path=reports/v2_survey/_scratch/g18_scratch.nc, bucket="noaa-goes18")`. No source edits, no `run_pipeline`. `_KEY_RE` matches G18 keys (6 keys listed for the hour). |
| Crop | Shape **(285, 301)**, float32, **0 NaN**. DQF all 0 (85,785 px). No pixel < 180 K or > 320 K. |
| BT (K) | min **197.96**, max **295.37**, mean 261.39. Percentiles: p1 200.73, p5 205.71, p10 218.30, p25 247.19, p50 265.56, p75 283.38, p90 290.63, p95 292.66, p99 294.26. Centre 21x21: min 198.64, max 289.65 (warm eye). |
| Comparison | GOES-16/19 Atlantic crops on disk are 251–285 x 300–312. Examples: Ian AL092022 mostly 283x301, Dorian 251x301, Melissa (G19) 279x301. |
| Downloads | This key twice: a metadata-only pass, then `process_one_scan`. Both copies are deleted; `_scratch/` holds only the nepac text. Coordinator note (2026-10-18): further reads download each key once only. |

**Why Dora is spent.** The storm is now **unusable for v2 validation and test**. It can still be used for training, and that is where it is most valuable:
- **Weak test candidate already:** its track crosses the antimeridian, which breaks the pipeline's longitude interpolation (see the pipeline gaps below), so Dora was a poor validation/test candidate anyway.
- **Most major-phase data:** it has the longest major phase of the 29 candidates, about 198 h at >= 96 kt.

**Caveat.** This fix is at GOES-18 nadir, the cleanest possible geometry. Typical EP majors sit at 105–125W, 20–45 deg off nadir (see the VZA column). If F3 should also cover an off-nadir case, that needs a second read; the coordinator has said not to take one now.

**How the file encodes fill, and how the pipeline handles it:**
- **CMI** is int16, packed with scale 0.06145332 and offset 89.62, `_FillValue` -1. xarray decodes the fill to NaN.
- **DQF** is int8 with `_FillValue` -1, decoded to float32 with NaN. `flag_values` 0–4, `flag_meanings` = `good_pixel_qf conditionally_usable_pixel_qf out_of_range_pixel_qf no_value_pixel_qf focal_plane_temperature_threshold_exceeded_qf`.
- **Space pixels:** the full disk had 6,373,440 NaN CMI pixels (21.7%), all off-earth. At those pixels DQF is also fill (NaN), **not** 3. Every on-earth pixel was DQF 0 in this scan.
- **Pipeline:** `extract_crop` saves `dqf` into the npz, but `dataset.py` never reads it. `np.clip` keeps NaN, so a NaN pixel reaches the network. The only guard is `check_nan.py`, which regenerates `excluded_crops.txt` and drops any frame with any NaN pixel. A frame whose pixels are finite but flagged 1, 2 or 4 passes straight through.

---

## 4. GOES-17 ABI loop-heat-pipe (LHP) degradation, 2019–2022

### What and when

**The defect.** The GOES-17 ABI LHP did not maintain the design focal-plane temperature [S6, S7]:
- **Daytime:** for 13–14 h a day, the MWIR/LWIR focal plane module (FPM) sits stable at about 81 K, warmer than GOES-16. In that state, radiometric calibration of channels 1–15 is unaffected, though noise in channels 8–16 is higher than on GOES-16 [S6].
- **Night:** for the other 10–11 h, the FPM warms. During "roughly 100 days centered on the vernal and autumnal equinoxes", calibration of **channels 8–16** is "further degraded and sometimes unusable" [S6].
- **Seasonal peaks:** about 20 days before and after each equinox. The dips between them come from Earth eclipse, and the small steps from yaw flips [S6].
- **Duration:** the elevated period runs about 3 h on the coolest peak days and about 9 h on the hottest [S6].

**Dated example (2020 spring) [S8]:** warming began in late January and peaked about 2020-02-26. It eased near the vernal equinox because of eclipse, worsened again until about mid-April, then recovered gradually until all channels were fine. "Channels 8-16 ... become saturated at certain times of the year"; the water-vapour bands 8–10 saturate first and "can be lost for up to six hours per day".

**Time of day.** The worst hours are late night into morning [S8]. Spacecraft midnight at 137.2W is about 09:00 UTC. The Mode 3 cooling timeline runs 06:00–12:00 UTC (spring 2022 example, 2022-04-08 to 05-11) [S9].

**Band 13 (10.3 um).** Band 13 is inside the affected channel range 8–16, but it is the least affected band:
- NOAA: "The 'Clean IR' band (channel 13) ... will never fully saturate, however there will be some striping evident when the sensor reaches its peak heating" [S8].
- CIMSS: Band 13 "is relatively unaffected by warming", "data integrity is preserved" [S10].
- So Band 13 is not lost, but striping and calibration error during the night-time warm periods of the two equinox seasons are plausible.

### How NOAA flagged and mitigated it

**Flags:**
- **Provisional maturity (2018-11-28):** "there are no data quality flags based on focal plane temperatures" [S6].
- **Since 2019-04-08:** the L1b and CMI files carry focal-plane-temperature DQF. That is **DQF value 4, `focal_plane_temperature_threshold_exceeded_qf`**, plus the summary attribute `percent_focal_plane_temperature_threshold_exceeded_qf` [S6, S11]. My GOES-18 file carries the same flag definition (§3).
- **Threshold updates:** 2019-08-08 19:45 UTC and 2019-10-02 17:04 UTC [S11]. The per-band values for Band 13 are in table images I could not read: **unverified**.

**Predictive calibration (pCal):** in operations from 2019-07-25 17:21 UTC. It corrects warm/cold bias between calibration looks while the FPM temperature changes, but "will not mitigate the issue of missing or striped data due to saturated sensors" [S6, S12].

**Other mitigations:**
- **Cooling timeline / Mode 3:** 6 h centred on spacecraft midnight, one full disk per 15 min, used in the peak seasons [S9].
- **GOES-15 supplement:** in some periods (spring 2022, 2022-04-12 to 04-18) [S9].
- **CIMSS data fusion:** fills missing bands using Band 13 as the predictor [S10].

**Unverified:** the year-by-year exact start and end dates of each season's degradation for 2019, 2021 and 2022. I found the 2020 and spring-2022 notices only. A claimed continuing FPM warming trend of about 1 K/yr appeared in a search summary only; I did not open a source page for it.

### Exposure of the EP candidates

Using the ±50-day-of-equinox window and synoptic major-phase fixes:

| Storm | G17 major fixes in season | Of those, 04–14 UTC |
|---|---|---|
| Barbara 2019, Erick 2019, Douglas 2020, Felicia 2021, Darby 2022 | 0 (Jun–Jul) | – |
| Juliette 2019 | 4/4 | 2 |
| Kiko 2019 | 5/5 | 3 |
| Genevieve 2020 | 3/3 | 1 |
| Marie 2020 | 9/9 | 4 |
| Linda 2021 | 8/8 | 4 |
| Orlene 2022 | 3/3 | 2 |
| Roslyn 2022 | 6/6 | 4 |
| Bonnie 2022 | 0/1 | 0 |

Barbara 2019 (major phase 07-02 to 07-04) predates pCal (2019-07-25) but is outside the LHP season.

### How to detect it in our crops

The repo already has a precedent: decisions.md:26 records that 6 GOES-16 scans near 05:00Z, a cooldown/calibration window, NaN'd the whole network and were excluded. Because Band 13 doesn't saturate, GOES-17 degradation will mostly **not** show up as NaN, so `check_nan.py` alone will miss it. Proposed checks, none implemented, all would need code:

1. **DQF:** count finite pixels with DQF in {1, 2, 4} per crop, from the `dqf` array already saved in every npz. Also read the file-level `percent_focal_plane_temperature_threshold_exceeded_qf` at crop time. That requires a pipeline change, because the attribute is not saved today.
2. **Striping:** the std of row means minus a smoothed version, compared against the same storm's daytime frames.
3. **Calibration drift against GOES-16 overlap:** for boundary storms (Genevieve, Linda, Orlene, Roslyn), crop the same time from `noaa-goes16` and compare BT histograms (p5/p50/p95) after accounting for the viewing-angle difference. Night-minus-day differences in that comparison point to LHP bias. This is an extra download, so it is not done here.
4. **Timing:** flag every 2019–2022 GOES-17 frame within ±50 days of an equinox **and** between 04 and 14 UTC.

### Recommendation

- **Include** GOES-17 EP storms (2019-02-12 – 2023-01-04) in **training only**.
- **Exclude** any frame with **any** finite DQF = 4 pixel, or any DQF 1/2 pixel, inside the crop. Until a DQF check exists, also exclude frames inside both windows below:
  - Season: **01-29 to 05-09** or **08-03 to 11-11** of 2019–2022 (±50 days of 03-20 and 09-22).
  - Time of day: **04:00–14:00 UTC**.
- **Never use GOES-17 storms for validation or held-out testing.** This is a mild cost: none of the G17 majors reaches >= 137 kt.
- **Orlene 2022 and Roslyn 2022** both fall in the LHP season and also have provisional GOES-18 data on S3. Whether to build them from `noaa-goes18` (pre-operational, but no LHP) is a decision for John. My lean is GOES-18, recorded as such.

---

## 5. Proposed v2 splits

**Frozen, unchanged, nothing added, never in training:**
- `storms_holdout_h9.txt` (9)
- `storms_val_expanded.txt` (5)
- `storms_test_2025.txt` (6, spent)

Every v2 training list must be disjoint from all three, as an *effective* training set (FEATURES.md I5 wording is still open with John; this proposal satisfies either reading). No list files were written.

### Recommendation

**Atlantic additions (training only):**
- **Majors:** AL072021 Grace, AL272020 Epsilon, AL282020 Zeta.
- **Hurricanes:** all remaining hurricanes in §1 except AL022022 (Bonnie goes in as EP042022) and the GOES-16 edge case AL182019 Pablo (optional). Build them at **3h** like the expanded arm.
- This roughly doubles the number of distinct Atlantic storms without making Cat 1–2 dominate.
- Tropical storms: leave them out of the first v2 arm. They add low-intensity frames, which is not where the error is.

**East Pacific: training plus its own held-out set; no EP validation in the first v2 run.**

- **EP training:**
  - All GOES-17 majors (13: Barbara, Erick 2019, Juliette, Kiko 2019, Douglas, Genevieve, Marie, Felicia, Linda, Bonnie (EP042022), Darby, Orlene, Roslyn), with the §4 filter.
  - GOES-18 majors 2023–2024 not held out: Calvin, **Dora** (e2e, training-only), Fernanda, Hilary, Lidia, Norma, Gilma, John, Kristy.
  - Kristy keeps 12 h of >= 137 kt in training.
- **EP held-out "H-EP"** (new list; evaluated once, like H9):
  - **EP052025 Erick, EP062025 Flossie, EP112025 Kiko, EP162025 Priscilla** (the untouched 2025 GOES-18 majors).
  - **EP112023 Jova** and **EP182023 Otis**, so that H-EP contains >= 137 kt frames: Jova 12 h, Otis 6 h. Otis is also the canonical rapid-intensification case and is viewed off-nadir from GOES-West.
  - CP012025 Iona optional, if the ID convention is widened.
  - All are GOES-18 if boundary storms are taken from GOES-West consistently, so H-EP measures transfer to a new basin on a clean satellite, not the LHP. Under the "better view" rule, Erick 2025 and Otis would come from GOES-East instead (noaa-goes19 and noaa-goes16 respectively).
- **Why no EP validation now:**
  - Keeping `storms_val_expanded.txt` as the only checkpoint-selection set keeps v2 directly comparable with the ensemble.
  - Atlantic validation has 0 frames >= 137 kt (decisions.md, 2026-10-14). Kristy and Jova would be the natural EP validation storms if John later wants checkpoint selection to see the top band. That would be a second arm, not a change to val.
- **Pacific 2025 storms are not "the 2025 test".** That set is Atlantic-only (AL IDs). H-EP uses 2025 EP storms, which have never been downloaded or evaluated. If John wants 2025 to stay entirely untouched, swap the four 2025 storms for Hilary 2023 and Gilma 2024. H-EP would then lose its time-forward property.
- **Not used anywhere:** 2026 storms (I4). Also no 2025 Atlantic non-test storm (e.g. AL122025 Lorenzo, AL102025 Jerry) without a separate decision.
- **Overlap checks v2 needs beyond `load_splits`:** the Bonnie and Julia ID pairs; name collisions resolved on ID only; CP/EP prefixes.

### Basin differences to track (stratify or report per basin)

- **Satellite:**
  - Atlantic = G16, plus G19 (test).
  - EP = G17 (2019–22, LHP), G18 (2023+), and G16/G19 if boundary storms are taken from the East.
  - Report per satellite as well as per basin.
- **Viewing angle and latitude:**
  - EP majors sit at 11–22N, 15–52 deg from GOES-West, and 30–49 deg for East-better storms.
  - The Atlantic training majors are mostly 15–30N at 10–40 deg from GOES-16. **Unverified:** I did not compute this for the training list.
  - A higher VZA means coarser, more stretched pixels and more parallax of cold tops.
- **Native crop size:** 600 km boxes give about 250–285 x 300–312 px on the Atlantic and 285x301 at G18 nadir. Off-nadir EP crops will be smaller (not measured). All are resampled to 224, but the effective resolution differs.
- **Storm size:** EP majors are on average smaller and more compact, and include annular-type cases (Dora). **Unverified**, domain knowledge, no source fetched.
- **SST and cold-tail climatology:** EP majors often decay fast over cold water west of about 120W and near Baja, and cloud-top temperature distributions may differ. **Unverified.** Compare the BT p1/p5 distributions per basin from the crops once built.
- **Rapid intensification near the coast:** Otis, John, Erick 2025 and Hilary. Coastal and landfall frames include land in the 600 km box.
- **HURDAT2 conventions:**
  - Separate files: nepac covers EP and CP.
  - Cross-basin storms appear in both files under two IDs.
  - nepac carries Dora into East longitudes.
  - Same 8-field data lines and status codes (checked by parsing), with -999 padding in the wind-radii fields. Format specs [S13, S14].
  - Name lists repeat on a 6-year cycle in both basins, so key on ID.

### Pipeline gaps for v2 (code changes needed before any EP build; none made)

- **(a) DQF is ignored.** `dataset.py` never reads `dqf`, so finite pixels flagged 1/2/4 pass through. That is why the §4 recommendation needs either a DQF-based exclusion step (alongside `check_nan.py`) or the time/season fallback.
- **(b) The antimeridian breaks longitude interpolation.** `pipeline.build_sample_grid` interpolates longitude linearly, so a track that crosses ±180 (Dora) gets nonsense positions between the last West-longitude and first East-longitude fixes. Dora must be trimmed at 180, or longitude interpolated on the unwrapped angle. Beyond about 175W it is also near the GOES-18 limb anyway.
- **(c) An EP build cannot run as the code stands.** `run_pipeline` always calls `hurdat2.download_hurdat2(url=..., force=...)` (`src/hurricane_intensity/pipeline.py:111`), which returns the Atlantic cache `data/interim/hurdat2_atlantic.txt` whenever it exists. `--hurdat2-url` is ignored on a cache hit. **The obvious workaround, `--force-hurdat2-refresh --hurdat2-url <nepac URL>`, would overwrite the protected Atlantic cache (I1/I7).** It must not be used. v2 needs a code change: a basin-specific cache path (e.g. `--hurdat2-path`, or a `hurdat2_nepac.txt` cache) chosen by ID prefix. Until then, no EP storm can be built.
- **(d) HURDAT2 versioning.** On 2026-10-18 the NHC data page links a newer Atlantic file, `hurdat2-1851-2025-092326.txt`, than the pinned cache's source (`hurdat2-1851-2025-02272026.txt`). I did not fetch it. v2 should decide whether to stay on the pinned file (keeps comparability) or re-pin. Re-pinning means a new file and hash, never an overwrite. The nepac file is `hurdat2-nepac-1949-2025-092926.txt` (hash above), and it should be pinned the same way.
- **(e) GOES-17 caveat.** Pre-operational GOES-18 data in 2022 and GOES-17 standby data after 2023-01-04 both exist on S3. The bucket choice per storm should be recorded in the manifest, which it already is, through `source_key`.

---

## 6. Sources

| # | Claim | URL |
|---|---|---|
| S1 | GOES-18 replaces GOES-17 as GOES-West on or near 2023-01-04 at about 18:00 UTC | https://www.weather.gov/media/notification/pdf2/scn22-113_goes-18_to_goes-west_operational_transition.pdf |
| S2 | GOES-18 timeline: at 137.0W after the 2022-07-21 nudge; Provisional 2022-07-28; operational 2023-01-04 1800 UTC; pre-ops data labelled "Preliminary, Non-Operational" | https://vlab.noaa.gov/web/towr-s/goes-18-timeline/-/web-browser/f0Ex0AwtGJYW/article/all-events |
| S3 | GOES-18 operational GOES-West, replacing GOES-17 on January 4; the LHP anomaly is not present on GOES-18 | https://vlab.noaa.gov/web/towr-s/satellite-operations-updates/-/web-browser/M3TwGEECzuCj/article/january-news |
| S4 | GOES-17 operational GOES-West as of 1800 UTC 2019-02-12, at 137.2W | https://ospo.noaa.gov/data/messages/2019/02/MSG_20190212_1804.html (opened: "As of 1800 UTC on February 12, 2019, GOES-17 is the official GOES-West satellite stationed at 137.2 degrees west") |
| S5 | GOES-19 replaces GOES-16 as GOES-East on 2025-04-07, at 75.2W | https://www.nesdis.noaa.gov/news/noaas-goes-19-now-operational-goes-east-providing-critical-new-data-forecasters (search result, not opened; the same date is in logs/decisions.md:1403) |
| S6 | GOES-17 L1b/CMI Provisional ReadMe (updated 2019-11-22): LHP details, 100-day equinox windows, channels 8–16, pCal 2019-07-25 1721 UTC, FPT DQF since 2019-04-08 | https://www.ospo.noaa.gov/operations/goes/product-quality-overview/ps-pvr/goes-17/ABI/Radiances/Provisional/GOES-17_ABI_L1b-CMI_Provisional_ReadMe_v2.pdf |
| S7 | Original Provisional ReadMe (2018-11-28): "no data quality flags based on focal plane temperatures" | https://www.ospo.noaa.gov/operations/goes/product-quality-overview/ps-pvr/goes-17/ABI/Cloud%20and%20Moisture%20Imagery/Provisional/GOES-17_ABI_L1b-CMI_Provisional_ReadMe.pdf |
| S8 | Feb 2020 update: channels 8–16 saturate seasonally; Band 13 never fully saturates but stripes; dates for the 2020 spring season | https://vlab.noaa.gov/web/towr-s/satellite-operations-updates/-/web-browser/M3TwGEECzuCj/article/goes-17-cooling-system-impacts-update-february-2020- |
| S9 | Spring 2022 update (the URL slug says "spring-2021", but the page is titled "Spring 2022" and dated 2022-04-05): Mode 3 cooling timeline 0600–1200 UTC, 2022-04-08 to 05-11; GOES-15 supplement 04-12 to 04-18 | https://vlab.noaa.gov/web/towr-s/satellite-operations-updates/-/web-browser/M3TwGEECzuCj/article/goes-17-abi-loop-heat-pipe-anomaly-spring-2021-update-1 |
| S10 | CIMSS, 2019-02-13: Band 13 relatively unaffected, data integrity preserved; data-fusion method | https://cimss.ssec.wisc.edu/goes/blog/archives/31740 |
| S11 | CIMSS: GOES-17 TDQF thresholds updated 2019-08-08 19:45 UTC and 2019-10-02 17:04 UTC. The DQF value 4 name and percent attribute are confirmed from S6 and my GOES-18 file; per-band thresholds unverified. | https://cimss.ssec.wisc.edu/satellite-blog/archives/33832 |
| S12 | CIMSS: pCal operational 2019-07-25 1721 UTC; it does not fix saturation or striping | https://cimss.ssec.wisc.edu/satellite-blog/archives/33789 |
| S13 | HURDAT2 Atlantic format | https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atl-1851-2021.pdf (linked from the data page, not opened) |
| S14 | HURDAT2 NE/NC Pacific format, and both data files | https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-nencpac-1949-2021.pdf (linked, not opened); https://www.nhc.noaa.gov/data/ |
| – | GOES-18 file attributes (DQF flag meanings, fill, packing, lon_0) | Read from the S3 key in §3 |

**Unverified items, collected:**
- GOES-17 season dates for each year other than 2020 and spring 2022.
- Per-band DQF thresholds, including Band 13.
- The about 1 K/yr FPM trend.
- EP storm-size and cold-tail climatology claims.
- Atlantic training-set viewing angles.
- Off-nadir EP crop sizes.
- Whether 2022 pre-operational GOES-18 files use lon_0 = 137.0 during the drift (Darby, 07-11 to 07-14).
- S5, S13 and S14 were taken from search results or links and not opened.

---

## 7. HURDAT2 label diff (pinned 02272026 vs 092326)

Task A7, 2026-10-18. Read-only. The decision is to stay on the pinned file; this section measures what that costs.

### Files

| File | Size | SHA-256 |
|---|---|---|
| Pinned: `data/interim/hurdat2_atlantic.txt` (source `hurdat2-1851-2025-02272026.txt`) | – | `ee35819cba2b61ab0c2175601084def682813727ba209c2dab7d3af71d0bcb5c`, asserted before and after the fetch and the diff |
| New: `reports/v2_survey/_scratch/hurdat2-1851-2025-092326.txt`, from https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt, fetched once | 7,071,568 bytes | `75202780ce56c8458f16fbef5fb8a026b99b00878dbb95c458211b3e6ef125b4` |

### Result: no labels change for any storm we use or propose

Two kinds of comparison:
- **Fix-level:** for each fix time, wind (kt), lat/lon (> 0.05 deg), status and record identifier; fixes added or removed; peak over all records and over tropical statuses only.
- **Byte-level:** the full text of each storm block.

| Role | Storms compared | Storms with any difference | Changed peak | Changed fixes at crop times |
|---|---|---|---|---|
| Training: union of `storms_train_expanded.txt`, `storms_train_control2.txt`, `storms_train.txt` and `storms_train_baseline.txt`, minus val_expanded and H9 | 19 | 0 | 0 | 0 |
| `storms_val_expanded.txt` | 5 | 0 | 0 | 0 |
| `storms_holdout_h9.txt` | 9 | 0 | 0 | 0 |
| `storms_test_2025.txt` | 6 | 0 | 0 | 0 |
| Proposed v2 Atlantic training: the 34 unlisted 2018–2024 storms >= 64 kt from §1, minus AL022022 Bonnie (listed as EP042022 instead) | 33 | 0 | 0 | 0 |

- **Every listed ID is covered.** The 39 listed IDs fall entirely within the first four roles (no other listed storms exist), and the legacy `storms_train.txt` H9 overlap (Beryl, Lee) is counted under H9.
- **All storms from 1976 onward are byte-identical.** That is 818 storm blocks in each file, including the pressure and wind-radii fields.
- **No manifests needed checking.** Since nothing changed, there are no affected fix times to match against `data/processed/<id>/manifest.csv`. The script reads the manifests but found nothing to check.

### What the new release changes

Only **1971–1975** differ. This looks like a reanalysis of those seasons: fix-level differences in 17 (1971), 17 (1972), 11 (1973), 15 (1974) and 20 (1975) storms.
- **5 IDs exist only in the new file:** AL181973, AL191973, AL201973, AL211974, AL241975.
- **21 IDs exist only in the pinned file:** AL011973, AL011975, AL021971, AL021973, AL021974, AL031973, AL031974, AL051971, AL051973, AL071972, AL071973, AL081972, AL101971, AL121975, AL151971, AL151973, AL151974, AL161971, AL171975, AL191974, AL201974.
- **Storm counts:** 2004 pinned vs 1988 new.

None of this is in the GOES era. **The cost of staying pinned is zero for every current and proposed Atlantic label.**

### Vintage mismatch to record

- **Atlantic labels** come from the **Feb 2026** release (`02272026`), pinned.
- **East Pacific labels** would come from **nepac `092926` (Sept 2026)**, SHA `caece626…a0507` (§Inputs).

The two basins' best tracks would therefore come from releases about seven months apart. For 2018–2025 the Atlantic Feb and Sept releases are identical (above), so this only matters if NHC revises recent EP seasons later. If v2 pins nepac 092926, record both release names and hashes next to every v2 result.

### Commands

Run from the repo root.

1. Fetch once, with the pinned hash asserted before and after:
   ```
   .venv\Scripts\python.exe -I -c "<inline: assert sha256(data/interim/hurdat2_atlantic.txt)==ee35819c…; requests.get(<092326 URL>) -> reports/v2_survey/_scratch/hurdat2-1851-2025-092326.txt (refuses to overwrite an existing file); print size and sha256; re-assert>"
   ```
2. Fix-level diff by role, plus crop-time matching against the manifests:
   ```
   .venv\Scripts\python.exe -I <scratchpad>\hurdat_diff.py . reports/v2_survey/_scratch/hurdat2-1851-2025-092326.txt
   ```
3. Byte-level comparison of every storm block from 1976 onward:
   ```
   .venv\Scripts\python.exe -I -c "<inline: split both files into header+N-line blocks keyed by ATCF ID and compare>"
   ```

The scripts are in the session scratchpad: `C:\Users\johnb\AppData\Local\Temp\claude\C--Users-johnb-Projects-hurricane-intensity\506a3b5b-9503-4971-9fd2-939a3bbd35e7\scratchpad\hurdat_diff.py` and `candidates_lib.py`. Nothing was written to `data/` and `--force-hurdat2-refresh` was not used. The new file stays in `_scratch/` pending a decision on whether to keep it or delete it, like the nepac text.
