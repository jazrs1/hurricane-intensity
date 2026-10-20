# F-31 runbook: the launch sequence and the abort routes

For the operator (the coordinator, in the main checkout). It adds no rule: `scripts/run_f31.py` enforces everything
below, and its docstring is the reference. Design: `reports/v2_survey/f31/design.md` (approved 2026-10-19).

## The sequence

0. **Before anything runs:**
   - design, analysis (PR #20) and driver (PR #25) merged;
   - a decisions.md entry with the blob hashes of `scripts/analyze_f31.py`, `scripts/run_f31.py`, `scripts/analyze_n5.py`, `scripts/run_n5.py` and `scripts/train.py`;
   - `check.py` PASS at the merge commit **L**;
   - John's yes to phase 1.
1. **Review the commands:** `python scripts/run_f31.py plan`. It prints every command of both phases and runs nothing.
2. **Phase 1:** `python scripts/run_f31.py launch`, in the main checkout at L.
   - 40 trainings, about 5.8 h, no H9.
   - It is resumable: a re-run skips the runs whose record verifies.
   - It stops and prints the 40 record paths.
3. **The records PR:**
   - On a branch from L, commit exactly the 40 `logs/f31/runs/<arm>_seed<s>.train.json` files (stored `-text`).
   - Merge the PR as a merge commit **H**, and bring the main checkout to H.
   - Nothing else may change between L and H, except `logs/decisions.md` and `reports/decisions/*.md` (guard 5e).
4. **Before the look:**
   - Record in decisions.md the BUDGET-LIMITED flags and qualifiers, and the zero-seed counts, from the 40 records (design §3).
   - John's separate yes to phase 2.
5. **Phase 2:** `python scripts/run_f31.py evaluate`, at H or a later docs-only commit. At most 180 H9 evaluations, about 20 min.
   - Before the first H9 step it re-runs every training-side check of the analysis on all 40 records.
   - It then writes `logs/f31/phase2_started.json`. From that moment H9 has been looked at.
6. **Analysis:** `python scripts/analyze_f31.py --manifest-dir logs/f31/runs --out-dir results/f31/analysis`. Add `--deviations logs/f31/deviations.json` only under rule (A).
7. **The results PR:** the manifests, logs, H9 CSVs, phase-2 marker and analysis outputs (all `-text`), with the results entry.

## Aborts: which route, when

**Before the records are committed (phase 1, nothing looked at):**
- A run that failed or was interrupted: `run_f31.py abort-run ARM SEED --reason ...`, then `launch` again.
- It restarts in its order position with its seed, and its record lists the attempt.
- A finished run is not aborted. To restart everything from a fix commit, use `abort-all`.

**After the records are committed (step 3 done), but still before any look.** A record that stops verifying (a kept file changed) can't simply be aborted and relaunched at H:
- `abort-run` moves the tracked record aside, so the tracked tree is dirty, and guard 1 refuses `launch`.
- Even with that committed, `launch` at a later commit finds the other 39 records at commit L, not HEAD, and refuses them (guard 4: records at another commit).

There are two routes:
- **(i) abort-all.** Move every run aside (`run_f31.py abort-all --reason ...`). Commit the records' removal through a PR. Then relaunch all 40 from that merge commit. Every run retrains. This is N5's fix-commit route (John's decision 2, 2026-10-19 15:48).
- **(ii) Restore the records at L by hand:**
  1. Check out L itself in the main checkout (it is on the first-parent chain, and the records are not tracked there).
  2. Run `abort-run` on the bad run and `launch` again. Only that run retrains, at commit L, so the other 39 records still match.
  3. Commit the 40 records again through a new records PR (step 3).
  4. Note the extra attempt in decisions.md.

**After a look** (any H9 output, any eval abort, or the phase-2 marker):
- No run may be aborted or retrained. `abort-run`, `abort-all` and `launch` all refuse.
- **A crashed or refused H9 evaluation:** `run_f31.py abort-eval ARM SEED --reason ...`, then `evaluate` again. The same slot epochs are re-evaluated, and the manifest lists the attempt.
- **A record that stops verifying:** restore the exact bytes of the changed files (their SHA-256s are in the record). Only if that is impossible does rule (A) apply:
  1. commit `logs/f31/deviations.json` naming the run;
  2. run `evaluate --deviations logs/f31/deviations.json`;
  3. run the analysis with the same flag.
  Every verdict is then NON-CONFIRMATORY, with the reduced n and its power stated.

## Practical notes

- **One stage at a time:** `logs/f31/run_f31.lock` serialises the writing stages. A lock left by a killed process is removed by hand, only after checking that no `run_f31.py` is running.
- **Interrupted aborts:** an abort that was interrupted leaves its `abort.json` "pending". Every other stage refuses until the same abort command is run again.
- **Windows paths:** abort paths are long (`logs/f31/aborted/<arm>_seed<s>_attempt1/models/f31/...`). Keep the checkout path short enough that they stay under MAX_PATH.
