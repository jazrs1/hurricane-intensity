# DECIDED (John, 2026-10-19)

- **Protocol approved:** protocol.md, SHA-256 3901f4cf41856c40e4c08a533011106ae78067a7301c7e27fbd3e8cd30624c7f.
- **Single pass.**
- **best_model.pt is out.**
- **E3's deadline** is set when the protocol is committed (planning on about the end of January 2027).
- **Note on the approved protocol.md:** its lines 18, 175 and 192 say PR #4's `--processed-root` and carry-forward are "being added". They have since been merged (8a1beb5 in ce251b9). The approved file is not edited, so its hash stays valid. The decisions.md entry written at gate 6 states the merged commit.
- **Follow-up PR, F-16/F-17: DONE.** Merged as PR #6 (474de21, 2026-10-19 00:46 EDT, git time), after hc-adversary item 11 OK. It covered:
  - F-16: refuse a missing excluded_crops.txt unless git reports it untracked (rc 1).
  - F-17: no carry-forward of an entry whose storm has a folder under another known root.
  - hc-adversary reviews the removal-refusal logic before John approves.

# H-EP: East Pacific held-out evaluation protocol, approval

**Needs John:**
1. Approve the protocol.
2. Pass timing (recommendation: **single pass**).
3. Include `best_model.pt` as a reference (recommendation: **no**; this changed from my earlier "yes").
4. Set **E3's deadline**: the date by which the later Atlantic + East Pacific arm's design must be committed and its checkpoints frozen, or its slot closes. A date is needed when the protocol is committed, not now.

This approval is NOT a download. Nothing East Pacific is downloaded until gates 1–6 below are done, and each gate is a separate reviewed step.

Protocol: `reports/v2_survey/hep/protocol.md` (r3). Drafted by hc-survey. Reviewed by hc-adversary in items 8, 8b and 8c.

## The set (N1 = A, your decision)

Erick EP052025, Flossie EP062025, Kiko EP112025, Priscilla EP162025, Jova EP112023, Otis EP182023. All are imaged by GOES-18 (the GOES-West rule, your N2 approval).

## Which models may ever be scored on it (§1)

| Slot | Models | Condition |
|---|---|---|
| E1 | The frozen Atlantic ensemble (models/storms/expanded/seed0-9.pt): the transfer baseline | Frozen; blob hash 3a2dac23… verified |
| E2c / E2a | The N5 control and atlantic arms | Only if N5 has closed with a results entry before the pass |
| E3 | The later Atlantic + East Pacific arm | Design committed and checkpoints frozen by the deadline you set; otherwise the slot closes |

- **One pass:** all declared slots are scored in a single run. Scoring E1 early would let these storms shape E3's design, and would preview E2c, which is the same configuration.
- **Ensembles:** an unweighted ensemble mean. No recalibration or member selection.

**"Once" is enforced, not just promised:**
1. the driver refuses existing output folders;
2. the built crops stay sealed in data/hep_sealed/ until the pass, so a standard script pointed at data/processed finds nothing;
3. results/hep/ is committed at the pass and pinned by check.py, like the 2025 results;
4. after the pass, the six IDs are blocked in TCDataset through a registry role, with the PR #3/#5 alias tests.

## Metrics, fixed now (§2)

- **Viewing angle first:** a VZA reference per evaluated model's training set (p95 and max). Erick 2025 and Otis at >96 kt are at about 49° / 46°, beyond the Atlantic training max (45.4°, p95 42.7°).
- **Per storm:**
  - MAE, fixes-only MAE and bias, with VZA and the share of frames beyond the training max;
  - >96 statistics only where a storm has ≥ 10 frames above 96 kt, which leaves Priscilla (7) without them, stated as such.
- **Jova and Otis at the high end:** each has its own >96 and Cat 5 rows, and every ≥137 kt frame is listed. **There is never a pooled >96 or Cat 5 figure** (your instruction).
- **Pooled figures:** all over frames below 137 kt, labelled "excl. ≥137 kt". The all-frames pooled figure is never computed, because it would leak the Cat 5 average by subtraction (hc-adversary finding 6).
- **Descriptive only, as in 2025:** H9-band reweighting, member spread, and transfer gaps against each model's own H9 numbers. There is no inferential test.
- **Diagnostics just before the pass:** domain shift and built geometry run only immediately before the pass, after every slot is frozen.
- **E3's design may not use any H-EP label or pixel information,** including the estimates table in the protocol.

## Gates, in execution order (§4; none done)

1. **B-v2-1:** the antimeridian fix.
2. **B-v2-2:** a basin-specific East Pacific HURDAT2 cache. It can never touch the protected Atlantic cache.
3. **The H-EP list and its guards:** the I5 forbidden set, the registry role and the concurrency check, before or inside the N5 launch PR. Priscilla's concurrent storms Octave and Raymond 2025 are excluded from any training list.
4. **PR #5,** the TCDataset 2025 block. It names the six Atlantic IDs only, so the East Pacific 2025 storms are not blocked.
5. **hc-adversary review** of run_hep.py and hep_report.py.
6. **Protocol committed** as a decisions.md entry, with blob hashes, the VZA reference, the West rule and E3's deadline.
7. **Download and build into data/hep_sealed/,** with exclusions via the merged check_nan.py: `--processed-root data/hep_sealed`, with carry-forward of entries for storms not under the scanned root. Both are in PR #4 commit 8a1beb5, merged as ce251b9. No model, geometry or domain shift yet.
   - **Required before gate 7 (John, 2026-10-19):**
     - (a) PR #6 (F-16/F-17): **merged** 474de21.
     - (b) **The F-18 + F-19 PR: merged** as PR #7, 34676b2 (git time 01:25 EDT), after hc-adversary items 12/12b OK. It covered:
       - F-18, dataset.py: `self.storm_ids = ids`; resolve the root once and memoize the per-crop check; a docstring line saying the block is path-based and does not detect copied or hard-linked spent crops.
       - F-19, check_nan.py:
         - refuse a known image folder that is a link or junction (a `data/hep_sealed` → `data/processed` junction would un-seal H-EP);
         - a refusal message, not a traceback, for a listed storm whose folder has no manifest;
         - a header note that a folder's new NaN frames appear only when that folder is scanned.
       - It goes through hc-adversary and John's yes.
     - (c) **F-35 (John, 2026-10-20): a gate before gate 7. CLOSED (John, 2026-10-20):** PR #22 merged as 31833ad (tests/test_crop_antimeridian.py, 54 centres, 109 tests). The crop code passes as it is, so no crop fix was needed. Add a test that catches a crop box clipped at 180° for centres a few degrees from 180, which the current crop pins miss at about 177.9°. If the test fails, fix the crop code too.
       - Why it is a gate: 4 v2 candidate storms have included-status fixes within 6° of 180° (coordinator's check against the cached nepac file hurdat2-nepac-1949-2025-092926.txt):
         - EP062019 Erick: 0.0°;
         - EP052023 Dora: 0.2° (it crosses);
         - EP082020 Douglas: 1.9°;
         - CP012025 Iona: 4.5°.
       - The six H-EP storms are all at least 20.5° from 180°.
     - (d) **B-v2-2b (John, 2026-10-26): a gate before gate 7. OPEN; assigned only after E3's design is approved.**
       - The East Pacific HURDAT2 reader refuses the six H-EP storm IDs, and Octave and Raymond 2025, unless it is called from the sealed H-EP pass. Tests show the refusal, and show the sealed-pass call is the only exception.
       - Timing of the download (John): not tied to the H-EP pass. The pinned nepac file (hurdat2-nepac-1949-2025-092926.txt, caece626…0507) is downloaded before the first East Pacific training build (E3's), after this gate's reader refusal has merged.
       - E3's design stage must not read the file.

8. **The pass, when you call it.**
9. **Spend in code,** plus the check.py pin.

## hc-adversary's findings, in summary

- **Item 8 (14 findings), all closed.** The four MED ones:
  - "once" was procedural only, now enforced four ways;
  - the all-frames pooled MAE leaked the pooled Cat 5 figure by subtraction, now pooled below 137 kt only;
  - the diagnostics came too early for E3, now just before the pass;
  - the gate order conflicted with the N5 launch, now H-EP guards land first.
  - Plus: the members-file hash recorded from the CRLF working copy (verified by me: 990 B vs the 980 B blob), now the blob hash.
- **Item 8b:**
  - PR #4 would refuse forever once sealed-storm exclusions exist. Fixed by adding carry-forward and `--processed-root` to PR #4 (commit 8a1beb5). Reviewed by hc-adversary in item 9b: the removal-refusal logic is correct in both two-folder directions. Merged as ce251b9 on John's yes.
  - A reword and a gate renumber, done in r3.
  - On best_model.pt: recommends **out**, because it is the 9.00 kt "good draw", and E1 already shows single-model spread.
- **Item 8c (the r3 delta): OK.**
  - The reword, the gate order and the output-line pattern all check out.
  - The carry-forward is safe in both directions: a scan of the sealed folder keeps the existing entries, and a later data/processed scan keeps the H-EP ones.
  - It notes that PR #4's new head must come back to it before merge, since the change touches the removal-refusal logic.


## Frozen version under review (SHA-256, LF file)

```
3901f4cf41856c40e4c08a533011106ae78067a7301c7e27fbd3e8cd30624c7f  reports/v2_survey/hep/protocol.md
```

## Timestamp correction (2026-10-19 00:47 EDT, from `date`)

The approved files cite "John, 23:57" for the change that a worker writes the TCDataset block: protocol.md line 164 and design.md §12.4. That time was a coordinator estimate, not read from a clock. John made that change before the PR #2 merge, which git records at 23:38 EDT. The approved files are not edited, so their hashes stay valid. The decisions.md entries written at commit will use git-anchored times.

**Hash definition:** the approved hashes above are blob hashes: SHA-256 of `git show <commit>:<path>` bytes, computed in Python. After reports/ is committed, Windows working copies are CRLF and hash differently, by design.
