# PR #6: check_nan.py follow-ups F-16 and F-17

**Needs John: merge yes.** Tier 2. **Recommendation: yes.**
- PR: https://github.com/jazrs1/hurricane-intensity/pull/6
- Head: eedfdbe9742e56fd2d3f058a1ad62d0e22bb25c9, branch check-nan-followups.
- Base: phase2-experiments, now 8a522b9 (PR #5). The PR branched from f8d2c70. It touches no file PR #5 touched, and hc-adversary's trial merge of the two passes 114 tests.
- Files: check_nan.py and tests/test_check_nan.py, +119/−24.
- This is the "new PR with hc-adversary reviewing the removal-refusal logic" you asked for with the N5/H-EP approvals (after the PR #3 merge at 00:11, before the PR #5 merge at 00:38). Merging it before H-EP gate 7 (the East Pacific build) is required.

## What it does

**F-16: a missing excluded_crops.txt.** The script starts from an empty list only when git confirms the file is untracked inside a repo (exit code 1). It refuses, writing nothing, in every other case:
- tracked (0);
- not a repo (128, which includes Windows "dubious ownership");
- any other code;
- git missing.

The old rule was fail-open outside a repo.

**F-17: entries across the two image folders.**
- The two folders (`data/processed` and the future `data/hep_sealed`) are listed in code.
- **An entry for a storm under the other folder is re-checked by scanning that storm in its own folder during the same run.** Previously it was carried forward unchecked, which let a hand-added bogus entry survive.
- **Only two kinds of entry are carried forward unchanged:**
  - the six spent storms, which are never loaded;
  - storms with no folder anywhere, which now get a stderr warning (e.g. a mistyped ID).
- **A storm with a folder under both image folders is refused.**

**One change from my instructions.** I asked that such entries be "re-derived ... or the run is refused". hc-reviewer made it always re-derive, so runs that alternate between the two folders keep working. That is strictly stronger than carrying forward. hc-adversary judged it the better choice: refusing would force an ordering on runs for no safety gain.

## Evidence

**Coordinator (own detached worktree at eedfdbe):**
- check.py 6/6 PASS in 24.4 s (71 passed, 41 xfailed on the pre-PR #5 base);
- the real excluded_crops.txt unchanged;
- the full check_nan.py diff read;
- check_nan.py has not been run on the real repo.

**hc-reviewer:**
- 36 check_nan tests, up from 27. The fixtures are git-initialized temp dirs with GIT_CEILING_DIRECTORIES, so a parent repo can't mask the rc 128 case.
- 8 deliberately broken variants each fail, then pass once reverted.
- --e2e PASS (Michael 88.58 with a byte-identical crop, Dora 124.23).

## hc-adversary (item 11): OK

Every case was run both without and with PR #5's block; the results are identical.
1. **Smuggling closed.** Its earlier demonstration (a bogus AL142018 entry carried through a sealed-folder run) is now refused with the file unchanged. No string can exclude a real frame without being verified.
2. **The two-folder alternation works both ways.** Each folder's entries are re-derived under their own folder.
3. **Case and alias variants across folders are refused:** a lowercase directory with a manifest, and one without.
4. **Exit-code logic demonstrated:** 128 refused, 1 proceeds, 0 and OSError refused.
5. **Composes with PR #5;** merge order doesn't matter.

**Optional low follow-ups, logged as F-19, to do before H-EP gate 7:**
- Refuse a known image folder that is itself a junction. A junction from `data/hep_sealed` to `data/processed` would silently un-seal H-EP. This matters for the H-EP seal, so it should be done before the build.
- Turn the FileNotFoundError for a listed storm whose folder has no manifest into a normal refusal message. It already fails closed, but as a traceback.
- A header-comment line: a folder's new NaN frames appear only when that folder is scanned.

**Coordinator's recommendation:** merge eedfdbe now. Do F-19 together with F-18 (the PR #5 dataset.py cleanups) as one small follow-up PR, before the H-EP build.
