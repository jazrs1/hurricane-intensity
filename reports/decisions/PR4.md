# PR #4: make check_nan.py safe

**Needs John: merge yes.** Tier 2. **Recommendation: yes.**
- PR: https://github.com/jazrs1/hurricane-intensity/pull/4
- Head: 8a1beb5816859a266e94e05c8188387526bad3a4, branch check-nan-safe.
- Base: phase2-experiments, now 1f558da. The PR branched from 2e55087; the only change since then is decisions.md, so there is no overlap.
- Files: 2 (check_nan.py, tests/test_check_nan.py), 2 commits, +~500.
- Merge order (yours): PR #4 → PR #3 → PR #5.

## Why

The old check_nan.py did two dangerous things:
- it deleted the tracked excluded_crops.txt first (line 8);
- it then loaded every storm on disk, including the six spent 2025 storms.

Once the 2025 block lands it would crash after the delete, and every training run would then fail, because NaN validation frames make the validation MAE NaN. That was hc-adversary item 5, finding 15. You ruled that PR #3 waits for this fix.

## What it does

- **Never deletes the file.** It scans from an empty temp folder, so current exclusions don't apply and every frame is still scanned. It writes a temp file, then `os.replace`. Any failure leaves the file byte-identical.
- **Never loads the six spent Atlantic 2025 storms** (an in-code frozenset). Their existing entries are carried forward. East Pacific 2025 storms are still scanned (N1).
- **Refuses aliases before any crop loads:**
  - folder names that aren't exact ATCF IDs;
  - junctions and links;
  - manifest rows that belong to another storm;
  - stray rows after construction.
- **Refuses to drop any existing entry** for a storm it scans. Trailing spaces are normalized, and duplicates are reported.
- **Carries forward entries for storms not under the scanned folder,** and takes `--processed-root`. This lets the future sealed East Pacific folder (H-EP protocol) be scanned without blocking later runs. Both directions were checked.
- **Refuses a run if excluded_crops.txt is missing but git still tracks it.** A cleanup failure no longer hides the original error.
- **Output bytes match the old script:** same order, CRLF on Windows, no trailing newline. The only difference is a smaller "of N" count.

## Evidence

**Coordinator (own detached worktree at 8a1beb5):**
- check.py 6/6 PASS in 21.2 s (pytest 39 passed);
- --e2e PASS (Michael 88.6/90, Dora 124.2/130);
- 14/14 protected hashes throughout;
- the real excluded_crops.txt is unchanged (9df219d6…);
- check_nan.py has not been run on the real repo.

**hc-reviewer's proofs (proof_n4.txt, proof_n4_v2.txt):** 21 deliberately broken variants each fail the tests, then pass once reverted. Examples:
- deleting the file first;
- an in-place write;
- a case-sensitive spent filter;
- skipping all 2025 storms;
- no removal refusal;
- each alias check removed;
- the carry-forward removed;
- the root option ignored.

**Real data, read-only (reviewer, confirmed by hc-adversary):**
- 39 storms with manifests, 5839 rows;
- no malformed names, no links, 0 mismatched rows;
- all 21 current entries belong to scanned storms and are re-derived, as before.

## hc-adversary

- **Item 9, at 9004ada: CONCERN (low-med).** It demonstrated two alias bypasses: a copied manifest and a junction. Fixed in 8a1beb5.
- **Item 9b, at 8a1beb5: OK to merge.**
  - Both bypasses are now refused, plus a new variant: a junction named like a spent storm.
  - The two-folder carry-forward is correct and byte-stable, and it never carries an entry for a scanned storm.
- **Two optional low follow-ups, not blocking:**
  - **F-16:** outside any git repo (e.g. an exported copy), a missing excluded_crops.txt lets the run start from an empty list. Spent storms' entries would then be silently absent. Fix: proceed only when git says "untracked" (rc 1); refuse on every other result. The real checkout is a repo, so it's unaffected today.
  - **F-17:** a hand-added bogus entry for a clean data/processed crop survives a run over the sealed folder, and is caught only by the next default run. A mistyped storm ID is carried forever (harmless noise). Fix: don't carry an entry whose storm has a folder under any known root other than the one being scanned. This matters only once data/hep_sealed exists.

**Coordinator's recommendation:** merge now, and do F-16 and F-17 as one small PR before the H-EP build (gate 7), which is the first time a second folder exists.
