# PR #3: N6 guards (required path args; tests for John's TCDataset 2025 block)

**Needs John: merge yes, AFTER PR #4.** Tier 2. **Recommendation: yes.**
- Head: **4877e3d6fd9b8e23f0952e25c02a7dc029ba14fb** (2 commits: 1028532 and 4877e3d).
- hc-adversary: items 6 and 6b, now **OK to merge**.
- Merge order (yours): PR #4 → PR #3 → PR #5. Once PR #4 is merged, I merge this pinned to 4877e3d and rerun check.py on the merged tip.

Tier 2.
- PR: https://github.com/jazrs1/hurricane-intensity/pull/3
- Branch: n6-required-args. Base: phase2-experiments. Branched from 2e55087; no overlap with PR #2 or PR #4.
- Size: 5 files, tests and two scripts. src/, check_nan.py and check.py are untouched.

## John's questions

**1. Who wrote the TCDataset 2025 block?**
Nobody. No block exists in PR #3 or on any branch.
- `git diff 2e55087 1028532b -- src` is empty (0 bytes).
- `src/hurricane_intensity/dataset.py` contains no "2025" at all.
- PR #3 has one commit, 1028532, and it touches only scripts/train.py, scripts/evaluate.py and three files in tests/.

The PR contains only the tests for the block: tests/test_dataset_2025_block.py, marked strict xfail. **Change (John, before the PR #2 merge at 23:38):** a worker writes the block as PR #5, after PR #3 merges. PR #5 removes the xfail marks, blocks the six Atlantic IDs by ATCF ID only with no override, and keeps the EP 2025 not-blocked test. hc-adversary reviews it, and John reviews the block diff before merging.

To check that those tests can tell a right block from a wrong one, hc-reviewer patched throwaway blocks into a scratch clone that was never committed or pushed:
- a correct block;
- an "any 2025 ID" block;
- a block placed inside the manifest loop.
Its proof notes are proof_n6_block.txt in its scratchpad. None of that code is in the PR. If you want to see those throwaway variants, say so.

**2. Does PR #3 fix check_nan.py (adversary finding 15)?**
No. check_nan.py and excluded_crops.txt are untouched. Per your ruling, PR #3 waits.
- The fix is assigned as **PR #4** (hc-reviewer, branch check-nan-safe, tier 2):
  1. never unlink first; write a temp file, then os.replace, so an exception leaves the file byte-identical;
  2. never load the six spent Atlantic 2025 IDs, and carry their entries forward (today they have none); East Pacific 2025 storms are still scanned (N1);
  3. refuse any change that would drop an existing entry;
  4. a main() guard so the tests can import it.
- Tests run on a tmp fixture only. Each fix gets a fail-then-pass proof.
- check_nan.py is not run on the real repo, because it rewrites a tracked file; that's your call later.
- **Merge order (approved by John before the PR #2 merge at 23:38; then updated):** PR #2, then PR #4, then PR #3, then PR #5 (the block).

## What PR #3 does

- **train.py:** --train-list, --val-list and --out are required, with no default. The old --out default was the protected best_model.pt.
- **evaluate.py:** --checkpoint, --holdout-list, --train-list and --out are required, with no default. The old --out default was the protected results/heldout_predictions.csv.
- **tests/test_required_args.py:** an ast-only check that the scripts are never imported or run. Both scripts parse arguments at module level, so importing one would run it. --epoch-dir and --frame-list still default to None.
- **tests/test_dataset_2025_block.py:** tests for the block (PR #5).
  - Strict xfail, with raises limited to "DID NOT RAISE", for each of the six Atlantic 2025 IDs, mixed lists, lowercase, and whitespace. The block must raise before any manifest is read.
  - Non-xfail controls:
    - the signature is unchanged (no override parameter);
    - AL142018 loads;
    - EP052025 loads. This is requirement 1 on the block: name the six IDs, not the year.
- **tests/test_list_invariants.py:** Fiona corrected to AL072022 (AL092022 is Ian), plus an explicit Fiona-in-validation test.

## Evidence (coordinator, own detached worktree at 1028532b)

- check.py: 6/6 PASS in 9.7 s; pytest 31 passed, 13 xfailed; 14/14 protected hashes before and after the smoke run. check.py itself is unchanged from base.
- --e2e: PASS in 20.1 s. Michael 88.58 vs 90 kt, Dora 124.23 vs 130 kt. 14/14 hashes after each case.
- Runtime argument check without running either script: only the argparse nodes, extracted with ast, were executed.
  - train.py with no args: exit 2, "required: --train-list, --val-list, --out".
  - evaluate.py with no args: exit 2, "required: --checkpoint, --holdout-list, --train-list, --out".
  - Full arguments parse.
- The dropped TRAIN_LIST/VAL_LIST/HOLDOUT_LIST imports are no longer referenced in either script.
- hc-reviewer's proofs:
  - Required-args test: base 7 failed / 2 passed; PR 9 passed.
  - Simulated blocks: correct block with markers → 13 XPASS(strict) failures; markers removed → 22 passed.
  - An "any 2025 ID" block fails only the EP control.
  - A check inside the manifest loop fails 4 tests.

## Findings for John

- **Historical runners now stop with an error.** run_order.sh, imbalance.sh, loss_experiment.sh and rotation_experiment.sh relied on the removed defaults; run_order.sh wrote best_model.pt that way. They now fail loudly. Decision (coordinator): leave them unchanged as records.
- **The choke-point claim was too broad.** TCDataset is the only path from a crop into a model. Other scripts read crops directly with np.load and no model: domain_shift.py (reads 2025 crops by design), audit/*, make_previews.py, throwaway_crop_stats.py. The guarantee is "every model input", not "every crop read".
- **This PR edits tests/, so check.py can't vouch for it.** That's why it carries an adversary diff review and the coordinator's independent re-hash.

## hc-adversary (item 6, @ 1028532b): CONCERN

The required-args change is right, but its test misses two ways of giving a default. The block tests accept a block that can be bypassed. Fixes are assigned to hc-reviewer as new commits on PR #3, after PR #4.

1. **MED, verified by the coordinator.** A plausible block passes all 22 tests and is still bypassable on Windows:
   - "AL042025.", "./AL042025", "../processed/AL042025", an absolute path, or a decoy dir whose manifest points into AL042025/ all reach the spent crops (NTFS);
   - gradcam.py:523 passes a raw --storm-id.
   Fix: alias and decoy cases; refuse malformed IDs (^[A-Z]{2}\d{6}$, no normalization) before any read; a manifest-row check; a __getitem__ guard.
2. **LOW-MED.** An env-var or module-flag override keeps the signature and passes. Fix: a source check (no os.environ/getenv), a frozenset of exactly the six IDs, and no toggle attribute.
3. **LOW.** A block on any AL*2025 ID passes. Fix: an AL012025 positive control.
4. **LOW.** Construction-only guard: ds.df reassigned later serves spent crops. Fix: a __getitem__ guard test.
5. OK: the obvious wrong blocks are caught (after-read, exact-case, list-file, **kwargs, getitem-only, warn, wrong exception type).
6. OK: the xfail markers can't hide a regression today. Nit: a skipif for case-sensitive filesystems.
7. **LOW-MED.** The ast test misses `const`/`nargs="?"` and `type=lambda v: v or default`. Fix: ban default/const/nargs/type on the required flags, plus a runtime parser test (construction nodes only; the scripts are never run).
8. Choke point: true today for model inputs. check_nan.py is the one TCDataset caller the block will break. Covered by PR #4, which lands before PR #3 and PR #5 in the approved order.
9. No objection to leaving the historical runners. Add a sentence to the PR body about explicit new-directory paths for reproductions.
10. OK: the Fiona fix, exactly 7 flags changed, and the constants are gone.

Status: PR #3 comes back to hc-adversary after the fix commits, then to John.

## Update at 4877e3d (the item-6 fixes, tests only, +219/−25)

**Block tests, still strict xfail until PR #5:**
- path aliases: trailing dot, ./, ../processed/, an absolute path, and a decoy folder with a copied manifest;
- malformed IDs refused before any read;
- manifest-row checks;
- a check when each crop loads (__getitem__);
- no environment variable or module flag can switch the block off;
- the spent set is one module-level frozenset.

**Non-xfail controls:**
- AL012025 (an unspent Atlantic 2025 storm) loads;
- EP052025 loads;
- AL142018 loads;
- the signature is unchanged;
- dataset.py reads no environment variables.

**Required args:** the default, const, nargs and type keywords are banned on the 7 flags. A runtime parser test (construction statements only; neither script is ever run) requires exit code 2 for missing and bare flags.

**Coordinator (own worktree at 4877e3d):** check.py 6/6 PASS in 10.9 s (35 passed, 41 xfailed), 14/14 hashes, and src/, check_nan.py and check.py untouched. The reviewer's --e2e also passes (Michael 88.6/90, Dora 124.2/130).

**hc-reviewer's proofs:**
- a reference block passes all 52 tests with the markers removed;
- each bypass variant fails: naive, env, flag, any-AL-2025, a set instead of a frozenset, no getitem check, no manifest check, and a "$" regex;
- each required-args variant fails: const/nargs, type, default, not required.

## hc-adversary item 6b (@ 4877e3d): OK to merge

- Item 6 findings 1–4, the 6 nit and 7 are closed.
- It wrote its own reference block, independent of the reviewer's: 52 passed. Its env-var and module-flag hatches now fail.
- **Three optional low follow-ups, to fold into PR #5 when the block lands:**
  - a mutable-container flag (`_BLOCK = [True]`) still passes. Fix: assert dataset.py has no module-level list, dict, set or bytearray (it has none today).
  - one legitimate design is over-constrained: a module-level pattern string fails the flip test, though it crashes rather than disabling the block. Fix: accept any exception in the flip loop, or require a compiled pattern.
  - a frozenset built at import from an absolute-path file read passes. Fix: an ast check that the frozenset is a literal of the six IDs.
- Residual, low: post-parse fallbacks or an injected argv aren't covered by the args tests.
