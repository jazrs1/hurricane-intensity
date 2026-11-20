# F-31 launch (phase 1, training only)

**Needs John: the F-31 phase-1 launch yes.** Recommendation: **yes**.

It covers, in order:
1. Merge docs-only **PR #27** (the design §10 pre-launch decisions entry, with every launch blob hash). Its merge commit becomes the launch commit **L**.
2. In the main checkout at L:
   - `check.py --full` must show "PASS FULL total";
   - `run_f31.py looked` must print "H9 LOOK EVIDENCE: none".
3. Run `.venv\Scripts\python.exe -B scripts/run_f31.py launch` in the background. It prints HEAD and the analysis blob, which I check against PR #27's entry.

- **Phase 1 only:** 40 trainings and validation-only selection; **no H9.** It stops after training.
- **Then:**
  - the 40 training records go through a records PR (hc-adversary, then you);
  - the BUDGET-LIMITED counts are known before any look, and I bring them to you, as with N5;
  - **phase 2 (H9) needs your separate yes.**

## Your question: is the single-run retrain (runbook route (ii)) possible only before any H9 output exists?

**Yes, now enforced in code.**

The route needs `abort-run` and `launch`. Both refuse whenever `looked()` finds any of:
- **on disk:** an H9 output, a manifest, an aborted evaluation, or the "phase 2 started" marker. evaluate writes the marker before its first H9 step, and nothing ever moves it;
- **committed on `origin/phase2-experiments` or the local `phase2-experiments`:** the same evidence;
- **anywhere in git history, on any ref** (unmerged branches, remote branches, stash): any such file ever added, even if later removed.

`looked()` also refuses outright if the history or both refs can't be read. A failed history read can never count as "no look".

**The gap this closes, and how it was closed:**
- I found that `looked()` was disk-only. The route's first step, a detached checkout of L, removes the committed H9 results from disk. After the results were committed, the retrain would have gone through.
- PR #26 added the branch check. hc-adversary item 27 then found results on an **unmerged** results branch were still missed, so the history-wide scan was added.
- **Coordinator check with real git:**
  - results committed on an unmerged branch, then checkout of L: the disk is clean, and `looked()` reports the marker and the H9 CSV from history;
  - with no look: `[]`.
- PR #26 merged at 15:02 as 9fd119e, **on my review, as a tightening only**, under your rule. hc-adversary judged the one rule-(A) side effect harmless.

**Remaining limit (deliberate only):** evidence that was never committed anywhere *and* was deleted from disk by hand, or that sits only on a branch deleted and garbage-collected. Both require deliberately destroying evidence.

**The runbook's step 0 for route (ii)** is `run_f31.py looked` printing "none". Running the route still needs your yes at the time.

## Facts for the launch

- **Plan:** 40 runs.
  - Seeds 10–29: control 99 epochs, Atlantic 52, in the interleaved order.
  - Slots are the control's 5 and the Atlantic arm's 4.
  - `train.py` is pinned at afceb4b1….
- **Estimated time:** phase 1 about 5.8 h (5.5–7 h). **Please keep the desktop awake.** A sleep stops the driver, and the runbook covers resuming.
- **Disk:** 787 GB free. About 4.4 GB is used per run during processing, and about 10 GB is kept.
- **GPU:** RTX 5070, idle (2.5 GB used by the desktop).
- **Not yet present:** `models/f31`, `logs/f31` and `results/f31`.
- **Current state:**
  - check.py `--full` on 9fd119e: PASS FULL total (568 passed, 20 xfailed);
  - `run_f31.py looked`: "H9 LOOK EVIDENCE: none".
