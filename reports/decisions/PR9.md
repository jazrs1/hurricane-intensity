# PR #9: commit reports/ (dry-run records)

**Needs John: merge yes.** Tier 1, docs. **Recommendation: yes.**
- PR: https://github.com/jazrs1/hurricane-intensity/pull/9
- Branch: reports-dry-run. Base: phase2-experiments aa32f8e.
- Built by the coordinator in its own worktree.

## Contents

- **reports/decisions/:** the decision packages (N1, N5, HEP, PR2–PR7, and this file).
- **reports/v2_survey/:**
  - survey.md and review_b1.md;
  - the approved N5 design and its scripts and tables;
  - the approved H-EP protocol and its geometry tables.
- **.gitignore:** `/reports/v2_survey/_scratch/` (the pinned HURDAT2 downloads, 7.1 MB and 4.1 MB, are never committed) and `/reports/v2_survey/n5/synthetic/` (test output).
- **pyproject.toml:** `[tool.pytest.ini_options] testpaths = ["tests"]`, so a bare `pytest` doesn't collect the draft test under reports/. check.py already runs `pytest tests`.

## Your requirements

- **_scratch excluded:** 0 _scratch paths in the PR, and the new rule ignores the folder (`git check-ignore` confirms).
- **No COORDINATOR.md or FEATURES.md under any spelling:** a casefold path check over every file in the PR finds 0, and check.py's trust-file guard passes.

## Evidence

- check.py 6/6 on the branch (123 passed).
- **The four approved files are byte-identical to what you approved,** checked against their committed bytes:
  - design.md c0831235…;
  - analyze_n5.py cc9083ca…;
  - test_analyze_n5.py d4d1d56b…;
  - protocol.md 3901f4cf….
- No secrets, tokens or emails. The one local path containing the username (survey.md:446) is replaced with `<scratchpad>`.

## hc-adversary (item 13): OK

- **Secrets and personal data:** none. Only the public repo handle appears, in PR links.
- **Embedded hashes:** analyze_n5.py carries its own copy of the protected-hash block. That's by design (it's in the approved file) and isn't secret, but it can drift. The N5 launch PR A must test that it equals the COORDINATOR.md block.
- **Merging is safe with care.** The same files sit untracked in the main checkout, and git refuses to overwrite them. Procedure:
  1. move the copies aside;
  2. fast-forward;
  3. compare each committed file with its moved copy;
  4. only then delete the moved copies.
- **Approved hashes are blob hashes.** After the merge, Windows checks these files out with CRLF line endings, so on-disk hashes change. The approvals refer to the committed bytes, which don't change. This is recorded in COORDINATOR.md, N5.md and HEP.md.
- **Low items, all applied:** the path redaction, the synthetic/ ignore line and pytest testpaths.
