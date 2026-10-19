# PR #7: follow-ups F-18 (dataset.py) and F-19 (check_nan.py)

Written 2026-10-19 01:08 EDT (from `date`).

**Needs John:** review the dataset.py diff below, then give a merge yes. Tier 2. **Recommendation: yes.**
- **Why it matters:** your HEP.md gate list makes this PR a required step before gate 7 (the East Pacific download and build).
- PR: https://github.com/jazrs1/hurricane-intensity/pull/7
- Head: **ad395448738395de792cf389f564242629818e74**, branch followups-f18-f19. 2 commits: 5dba68c, then ad39544, which carries the memo fix from hc-adversary item 12.
- Base: phase2-experiments 474de21 (PR #6), so no rebase is needed.
- Files: dataset.py, check_nan.py and their two test files; 123 tests in total.

## dataset.py, verbatim (`git diff 474de21 ad39544 -- src/hurricane_intensity/dataset.py`)

```diff
diff --git a/src/hurricane_intensity/dataset.py b/src/hurricane_intensity/dataset.py
index 177d30e..458ed36 100644
--- a/src/hurricane_intensity/dataset.py
+++ b/src/hurricane_intensity/dataset.py
@@ -20,7 +20,11 @@ STORM_ID = re.compile(r"[A-Z]{2}[0-9]{6}")
 
 def crop_storm(processed_root, crop_path):
     """The storm directory crop_path resolves to under processed_root, upper-cased; None if it is outside it."""
-    root = Path(processed_root).resolve()
+    return resolved_crop_storm(Path(processed_root).resolve(), crop_path)
+
+
+def resolved_crop_storm(root, crop_path):
+    """crop_storm for a processed_root that is already resolved."""
     try:
         parts = (root / crop_path).resolve().relative_to(root).parts
     except ValueError:
@@ -58,7 +62,9 @@ class TCDataset(Dataset):
         outside the crop is ever visible). With both off, preprocessing is exactly as before.
 
         Raises ValueError, before anything is read, for a storm ID that is malformed or spent (SPENT_2025),
-        and after reading a manifest whose rows belong to another storm or point outside its directory."""
+        and after reading a manifest whose rows belong to another storm or point outside its directory.
+        The block is path-based: it does not detect spent crop content copied or hard-linked under another
+        storm's directory. Keeping spent crops out of other storms' directories is a process rule."""
         ids = list(storm_ids)
         for storm_id in ids:
             if not isinstance(storm_id, str) or not STORM_ID.fullmatch(storm_id):
@@ -71,14 +77,16 @@ class TCDataset(Dataset):
         self.rotate = rotate
         self.generator = generator
         self.processed_root = processed_root
-        self.storm_ids = storm_ids
+        self.storm_ids = ids
+        self._resolved_root = (str(processed_root), Path(processed_root).resolve())
+        self._crop_storms = {}
         frames = []
         for storm_id in ids:
             manifest_path = Path(processed_root) / storm_id / "manifest.csv"
             df = pd.read_csv(manifest_path)
             if "atcf_id" not in df or (df["atcf_id"].astype(str) != storm_id).any():
                 raise ValueError(f"{manifest_path}: atcf_id missing or not {storm_id} in some rows")
-            stray = [p for p in df["crop_path"].dropna() if crop_storm(processed_root, p) != storm_id]
+            stray = [p for p in df["crop_path"].dropna() if self._crop_storm(p) != storm_id]
             if stray:
                 raise ValueError(f"{manifest_path}: crop_path outside {storm_id}/ in {len(stray)} rows, e.g. {stray[0]}")
             frames.append(df)
@@ -100,9 +108,24 @@ class TCDataset(Dataset):
     def __len__(self):
         return len(self.df)
 
+    def _crop_storm(self, crop_path):
+        """crop_storm under self.processed_root, memoised per (processed_root, crop_path) so each crop is resolved
+        once. Only the resolution is cached: __getitem__ still checks the result on every item, whatever df holds.
+        A crop that does not exist yet is not memoised, so a directory swapped in later is resolved afresh."""
+        root = str(self.processed_root)
+        if self._resolved_root[0] != root:
+            self._resolved_root = (root, Path(root).resolve())
+        key = (root, crop_path)
+        if key in self._crop_storms:
+            return self._crop_storms[key]
+        storm = resolved_crop_storm(self._resolved_root[1], crop_path)
+        if (Path(root) / crop_path).is_file():
+            self._crop_storms[key] = storm
+        return storm
+
     def __getitem__(self, idx):
         row = self.df.iloc[idx]
-        storm = crop_storm(self.processed_root, row["crop_path"])
+        storm = self._crop_storm(row["crop_path"])
         if storm is None or storm in SPENT_2025:
             raise ValueError(f"{row['crop_path']} is a spent 2025 crop or outside {self.processed_root}")
         crop_path = Path(self.processed_root) / row["crop_path"]
```

**What changes in dataset.py (F-18):**
- `self.storm_ids` stores the list it was given. A generator input used to leave it empty; nothing reads it.
- **Per-crop folder resolution is cached, but only for crops that exist.** __getitem__ still checks the cached storm against the spent set and against "outside the root" on every access, whatever df holds.
- **Cost:** per sample is back to the pre-block level (about 0.73 ms, vs 0.93–1.00 ms with PR #5 alone). Construction is 0.45 s (0.55 s with PR #5 alone; 0.11 s before any block).
- A docstring line now states the limit: the block is path-based and doesn't detect spent crop content that has been copied or hard-linked under another storm's folder. That's a process rule.

## check_nan.py (F-19), verbatim (`git diff 474de21 ad39544 -- check_nan.py`)

```diff
diff --git a/check_nan.py b/check_nan.py
index 10c051c..e7fff25 100644
--- a/check_nan.py
+++ b/check_nan.py
@@ -22,6 +22,8 @@ KNOWN_ROOTS = (Path("data/processed"), Path("data/hep_sealed"))
 # root (reported as a warning). Every other entry is re-derived by scanning its storm under its own root, and
 # is never dropped: if its crop or manifest row is gone, every run is refused until the entry is removed by a
 # reviewed, committed manual edit of excluded_crops.txt.
+# A root's new NaN frames appear only when that root is scanned (--processed-root): a run over one root
+# re-derives only the other roots' storms that already have entries, not every storm under them.
 
 
 def refuse(reason, lines=()):
@@ -35,6 +37,20 @@ def storm_of(crop_path):
     return crop_path.replace("\\", "/").split("/")[0].strip().upper()
 
 
+def check_roots(processed):
+    """No root is a link or junction (data/hep_sealed -> data/processed would un-seal H-EP), and no two known
+    roots are the same directory."""
+    roots = [r for r in dict.fromkeys([processed, *KNOWN_ROOTS]) if r.exists()]
+    linked = [r for r in roots if r.is_symlink() or r.is_junction()
+              or os.path.normcase(r.resolve()) != os.path.normcase(r.parent.resolve() / r.name)]
+    if linked:
+        refuse("roots that are links or junctions", linked)
+    known = [r for r in KNOWN_ROOTS if r.exists()]
+    same = {os.path.normcase(r.resolve()) for r in known}
+    if len(same) != len(known):
+        refuse("known roots that are the same directory", known)
+
+
 def manifest_dirs(root):
     return sorted(p.parent.name for p in root.glob("*/manifest.csv"))
 
@@ -123,6 +139,7 @@ def main(argv=None):
     processed = Path(parser.parse_args(argv).processed_root)
 
     old = read_old()
+    check_roots(processed)
     roots = [processed] + [r for r in KNOWN_ROOTS if r.is_dir() and r.resolve() != processed.resolve()]
     where = {}
     for root in roots:
@@ -141,6 +158,9 @@ def main(argv=None):
     for s in listed:
         if s in where:
             plan.setdefault(where[s][0], []).append(s)
+    no_manifest = [root / s for root, storms in plan.items() for s in storms if not (root / s / "manifest.csv").is_file()]
+    if no_manifest:
+        refuse("listed storms whose directory has no manifest.csv", no_manifest)
     found, n = [], 0
     for root, storms in plan.items():
         check_manifests(root, storms)
```

**What changes in check_nan.py:**
- **It refuses a known image folder** (`data/processed`, `data/hep_sealed`) that is a link or junction, or two known folders that resolve to the same place. Before this, a junction from `data/hep_sealed` to `data/processed` was silently skipped, which would have un-sealed H-EP.
- **A listed storm whose folder has no manifest** gets a refusal message (exit 1, nothing written) under either folder, instead of a traceback.
- **A header note:** a folder's new NaN frames appear only when that folder is scanned.

## Evidence

**Coordinator (own detached worktree):**
- At 5dba68c: check.py 6/6 (122 passed); 160 random samples bit-identical, old 474de21 vs new; generator storm_ids OK; spent IDs refused.
- At ad39544: check.py 6/6 PASS in 23.4 s (123 passed); 120 more random samples bit-identical.
- The real excluded_crops.txt is unchanged; check_nan.py has not been run on the real repo.

**hc-reviewer:**
- 9710 samples (33 storms, plain, masked, rotated) bit-identical, old vs new, at both heads.
- No reader of .storm_ids anywhere.
- Fail-then-pass proofs for each change: storm_ids, the memo, the memo key, root re-resolution, getitem trusting the memo, the docstring, the root check, the missing-manifest refusal, and the not-memoised missing crop.
- --e2e PASS (Michael 88.58 with a byte-identical crop, Dora 124.23). That ran on the main install; the side-by-side samples are the evidence for this branch's dataset.py.

## hc-adversary (items 12 and 12b): OK

1. **Memo (item 12).** It demonstrated one deliberate sequence that loaded spent bytes through a stale cache:
   - inject a row whose file doesn't exist yet;
   - access it once;
   - swap a junction in;
   - access it again.
   The fix in ad39544 (cache only existing crops) refuses it; verified in 12b. The sequence is now a test.
2. **The remaining case is not a new hole (12b).** A cached crop that exists could be served after a folder swap only if a file with that crop's legal name sat at the new location. That means spent bytes copied or renamed under a legal name, which already defeats any path-based check without a junction or cache. It's the documented process-rule limit, and closing it would cost back about 0.2 ms per sample for nothing.
3. **H-EP can't be un-sealed by links.**
   - A junction at either known folder is refused.
   - A per-storm junction into the sealed folder is refused three ways, including by the TCDataset block itself during training.
   - A junctioned parent (`data/`) moves both folders together, so they stay distinct.
4. **The rest checks out:**
   - processed_root reassignment is handled;
   - the memo pickles fine into spawn workers (about 0.6 MB per 5,000 frames);
   - the getitem-memo test catches a check skipped on cache hits.
