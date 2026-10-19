# PR #5: TCDataset 2025 block (worker-written; John reviews the diff)

**Needs John:** review the dataset.py diff below, then give a merge yes. Tier 2. **Recommendation: yes.**
- PR: https://github.com/jazrs1/hurricane-intensity/pull/5
- Head: efc120dba9e5d63aedf7c9e69dc3a8666e1ab6be, branch tcdataset-2025-block. Base: phase2-experiments f8d2c70, so no rebase is needed.
- Files: src/hurricane_intensity/dataset.py (+37/−2) and tests/test_dataset_2025_block.py (+34/−34). The xfail markers are removed, and hc-adversary's three 6b test gaps are added.
- Written by hc-reviewer per your change (made before the PR #2 merge at 23:38). The tests came first, in PR #3.

## The block, verbatim (`git diff f8d2c70 efc120d -- src/hurricane_intensity/dataset.py`)

```diff
diff --git a/src/hurricane_intensity/dataset.py b/src/hurricane_intensity/dataset.py
index de451c0..177d30e 100644
--- a/src/hurricane_intensity/dataset.py
+++ b/src/hurricane_intensity/dataset.py
@@ -1,3 +1,4 @@
+import re
 import pandas as pd
 import numpy as np
 import torch
@@ -9,6 +10,23 @@ BT_MIN = 180.0
 BT_MAX = 320.0
 SIZE = 224
 
+# The six 2025 Atlantic test storms, spent by the final test (logs/decisions.md, 2026-10-16): TCDataset never
+# loads them. Kept in code, never read from a file, the environment or the cwd (check_nan.py builds TCDataset
+# from an empty temporary cwd). Other 2025 storms (AL012025, the East Pacific H-EP set) load as usual.
+SPENT_2025 = frozenset({"AL042025", "AL052025", "AL072025", "AL082025", "AL092025", "AL132025"})
+# A storm ID is exactly two capital letters and six ASCII digits, with no normalisation.
+STORM_ID = re.compile(r"[A-Z]{2}[0-9]{6}")
+
+
+def crop_storm(processed_root, crop_path):
+    """The storm directory crop_path resolves to under processed_root, upper-cased; None if it is outside it."""
+    root = Path(processed_root).resolve()
+    try:
+        parts = (root / crop_path).resolve().relative_to(root).parts
+    except ValueError:
+        return None
+    return parts[0].rstrip(". ").upper() if len(parts) > 1 else None
+
 # Circular mask: everything outside the circle inscribed in the 224 x 224 frame is set to 300 K, which reads
 # as clear tropical ocean (0 would be 180 K, the coldest cloud top the scale allows).
 MASK_FILL = (300.0 - BT_MIN) / (BT_MAX - BT_MIN)
@@ -37,7 +55,16 @@ class TCDataset(Dataset):
     def __init__(self, processed_root, storm_ids, circular_mask=False, rotate=False, generator=None):
         """circular_mask: fill outside the inscribed circle with MASK_FILL. rotate: rotate each sample by a
         uniform random angle (training augmentation; requires circular_mask, so no frame corner sampled from
-        outside the crop is ever visible). With both off, preprocessing is exactly as before."""
+        outside the crop is ever visible). With both off, preprocessing is exactly as before.
+
+        Raises ValueError, before anything is read, for a storm ID that is malformed or spent (SPENT_2025),
+        and after reading a manifest whose rows belong to another storm or point outside its directory."""
+        ids = list(storm_ids)
+        for storm_id in ids:
+            if not isinstance(storm_id, str) or not STORM_ID.fullmatch(storm_id):
+                raise ValueError(f"malformed storm ID {storm_id!r}: expected two capital letters and six digits")
+            if storm_id in SPENT_2025:
+                raise ValueError(f"{storm_id} is a spent 2025 test storm and is never loaded")
         if rotate and not circular_mask:
             raise ValueError("rotate requires circular_mask")
         self.circular_mask = circular_mask
@@ -46,9 +73,14 @@ class TCDataset(Dataset):
         self.processed_root = processed_root
         self.storm_ids = storm_ids
         frames = []
-        for storm_id in storm_ids:
+        for storm_id in ids:
             manifest_path = Path(processed_root) / storm_id / "manifest.csv"
             df = pd.read_csv(manifest_path)
+            if "atcf_id" not in df or (df["atcf_id"].astype(str) != storm_id).any():
+                raise ValueError(f"{manifest_path}: atcf_id missing or not {storm_id} in some rows")
+            stray = [p for p in df["crop_path"].dropna() if crop_storm(processed_root, p) != storm_id]
+            if stray:
+                raise ValueError(f"{manifest_path}: crop_path outside {storm_id}/ in {len(stray)} rows, e.g. {stray[0]}")
             frames.append(df)
         df = pd.concat(frames, ignore_index=True)
         df = df[df["crop_path"].notna()]
@@ -70,6 +102,9 @@ class TCDataset(Dataset):
 
     def __getitem__(self, idx):
         row = self.df.iloc[idx]
+        storm = crop_storm(self.processed_root, row["crop_path"])
+        if storm is None or storm in SPENT_2025:
+            raise ValueError(f"{row['crop_path']} is a spent 2025 crop or outside {self.processed_root}")
         crop_path = Path(self.processed_root) / row["crop_path"]
         data = np.load(crop_path)
         bt = data["bt"]
```

## What it guarantees

1. **Spent IDs refused before any read.** AL042025, AL052025, AL072025, AL082025, AL092025 and AL132025 are a frozenset literal in code. Nothing comes from a file, the environment or the cwd, so check_nan's empty-cwd scan is safe. There is no override and no toggle.
2. **Malformed IDs refused before any read.** An ID must be exactly two capital letters and six ASCII digits, with no normalization. That closes the Windows aliases "AL042025.", "al042025", "./AL042025" and absolute paths, plus whitespace and other Unicode digits.
3. **Per-manifest-row check.** Every row's atcf_id must equal the storm, and every crop_path must resolve inside that storm's own folder. That closes decoy folders with copied manifests, and junctions.
4. **Per-sample check in `__getitem__`.** It survives df reassignment (train.py:75) and DataLoader worker processes.
5. **Not blocked:** AL012025 (an unspent Atlantic 2025 storm) and EP052025 (H-EP, per N1).
6. **No behaviour change for legal storms.** Preprocessing, exclusions and the printout are unchanged.

## Evidence

**Coordinator (own detached worktree at efc120d):**
- check.py 6/6 PASS in 22.5 s, pytest 105 passed, imported from the PR's src.
- Side-by-side old (f8d2c70) vs new dataset.py, loaded independently:
  - identical data tables for all 9 H9 storms plus Michael AL142018;
  - 400 random samples bit-identical.
- New refuses AL042025, "al142018", "AL142018.", "./AL142018" and "AL13202". AL012025 and EP052025 pass the block.

**hc-reviewer:**
- 9710 samples over all 33 non-spent storms (plain, masked, rotated): bit-identical to the old code.
- All 11 non-2025 storm lists construct. storms_test_2025.txt is refused after 0 manifest reads.
- Layer proofs: removing each layer fails specific tests:
  - old code: 42 failed;
  - no pre-read check: 26;
  - no spent check: 10;
  - no manifest-row check: 4;
  - no getitem check: 5;
  - a computed frozenset, a mutable set, `$`-match and `\d`: 1–3 each.
- Callers: all 29 TCDataset calls take legal IDs. The free-form inputs now refuse spent and malformed IDs: gradcam --storm-id/--storms-file, storm_fraction --storms-list, and ensemble_eval --holdout-list (including the old final-test command, as intended).
- --e2e passes, but on the main install, so it doesn't exercise the block. The side-by-side comparisons cover equivalence.

## hc-adversary (item 10): OK to merge, nothing fails open

**Attacks tried, all refused or correct:**
- the normalization asymmetry between the regex and crop_storm;
- NaN crop_path rows;
- a junction inside a legal storm folder that points at spent crops (refused at construction);
- DataLoader spawn workers with a spent row injected after construction (refused inside the worker);
- numpy-array, single-string and generator inputs (a single string is now refused instead of iterating its characters).

**The reviewer's unsure points:**
- **Cost:** acceptable. Construction goes from 0.10 to 0.55 s; per sample about +30%, roughly +20 s per 25-epoch N5 run.
- **Strict atcf_id rule:** 0 mismatches and 0 NaN over 5839 real rows.
- **Backslashes:** the data was already Windows-only. Before this PR, Linux silently produced an EMPTY dataset; now it refuses loudly. An improvement.

**Inherent limit (low):** a byte copy or hard link of a spent crop placed under a legal storm's folder and path still loads. No path-based guard can see that; it needs deliberate data copying. It stays a process rule, and the docstring should say so.

## Optional follow-ups (not blocking; logged as F-18)

- Set `self.storm_ids = ids`. Cosmetic: a generator input leaves it empty, and nothing reads it.
- Resolve the root once in `__init__` and memoize crop_storm, halving the per-sample cost. The per-item check still runs.
- Add one docstring line: the block is path-based and does not detect copied or hard-linked crop content.

**Recommendation:** merge efc120d as is, and do F-18 as a small follow-up PR. A change to this diff now would need a fresh adversary pass and your re-review.
