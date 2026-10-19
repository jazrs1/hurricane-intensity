# PR #2: decisions.md entry on held-out storms in the epoch-budget choice

**Needs John: merge yes.** Tier 1, docs only.
- PR: https://github.com/jazrs1/hurricane-intensity/pull/2
- Head: 7e2afb81fcdbe44f7ec75b1a1120a7926789cc72, branch decisions-h9-selection.
- Base: phase2-experiments at 2e55087 (the PR #1 merge).

**Recommendation: yes.**

## What it does

It appends one dated entry (2026-10-18, 23:21) to logs/decisions.md and changes nothing else. The entry:
- records finding 7: the 40-epoch budget was chosen partly from per-epoch held-out MAE on Ian, Nicole and Claudette;
- lists every exposed model and result set;
- bounds the effect near zero using validation MAE from the same runs, with the window means in the entry;
- states that the 2025 GOES-19 result is unaffected, with the caveat that every H9 comparison in that entry is H9-referenced;
- notes the record's own contradiction about the 2025 peak intensities;
- sets the protocol from now on:
  - H9 is reported pooled and as original-3 vs added-6;
  - per-epoch traces are on validation only;
  - the existing held-out traces are not used for any further choice;
  - the rule is procedural, because two trace scripts still default to held-out lists.

## Evidence

- The coordinator byte-compared the committed text with the adversary-approved draft (SHA-256 cd86d75d…0ca131, with HH:MM filled as 23:21): IDENTICAL. The first 2153 lines of decisions.md are byte-identical to 2e55087.
- `git diff --stat`: 1 file, +58 / −0. `git diff --check` is clean. The committed blob has no CR characters.
- The validation/held-out window means were recomputed by the coordinator from results/epoch_traces_60/epoch_trace.csv and match the entry.

## hc-adversary

- **Item 4, first pass: CONCERN (wording).** "Unaffected" holds. All of it was applied:
  - the validation bound replaces "can't be measured";
  - the caveat is widened to every H9-referenced comparison;
  - the exposed-model scope and direction are added;
  - the Sam per-epoch line is added;
  - the 2025 intensity contradiction is added;
  - the held-out trace files and scripts are added.
- **Re-check: OK**, with two text fixes, both applied: "survey" became "the 10-14 more-distinct-storms design", and the counterfactual was softened. A third nit was left for commit time: the HH:MM placeholder, filled as 23:21.

## Not in this PR

The N5 design and the H-EP protocol each get their own decisions.md entry later, before their launches.
