#!/usr/bin/env python
"""One-command checks for the invariants in FEATURES.md (I1-I7).

    .venv\\Scripts\\python.exe scripts\\check.py            # F1: under 3 minutes, no network
    .venv\\Scripts\\python.exe scripts\\check.py --e2e      # F2/F3: downloads one GOES-16 and one GOES-18 scan
    .venv\\Scripts\\python.exe scripts\\check.py --gitignore-target <repo>   # only the gitignore section

Protected files, data/ and models/ are gitignored and exist only in the main checkout, so they are
resolved against it (the parent of git's common dir, or --repo-root), even when this runs from a
worktree. Storm lists and tests come from the checkout this script lives in. A missing protected file
is a failure, never a skip. Every section runs even if an earlier one fails; the exit code is 1 if
any failed.

Expected protected hashes are never read from a file this checkout (a PR) can change. They come from
the fenced ```sha256 block under "## Protected hashes (check.py source of truth)" in the main
checkout's COORDINATOR.md (gitignored, coordinator-owned), or --coordinator-md. The block must list
exactly PROTECTED_PATHS; anything else, or a missing file or block, is a failure.

Which code is tested: pytest and the evaluate.py smoke run import hurricane_intensity from this
checkout's src/ (PYTHONPATH), not the venv's editable install, which points at the main checkout; a PR
that changes src/ is tested with its own code. --e2e stays on the main install, because
config.PROJECT_ROOT ties the package location to data/ (the HURDAT2 cache, data/processed) and
check_e2e refuses to run if the two differ. Each section prints the src it imported.

Nothing here writes outside a tempfile.mkdtemp() directory, which is removed at the end.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_SRC = CHECKOUT_ROOT / "src"
MAIN_SHA = "fbb39c4"  # I6: main == origin/main == this commit, and site/ is unchanged since it

# I1: the protected files (FEATURES.md). Their expected hashes come from COORDINATOR.md, never from this repo.
PROTECTED_PATHS = [
    "models/baseline_9kt.pt",
    "best_model.pt",
    "results/heldout_predictions.csv",
    *[f"models/storms/expanded/seed{s}.pt" for s in range(10)],
    "data/interim/hurdat2_atlantic.txt",
]
HASH_HEADING = "## Protected hashes (check.py source of truth)"
# Coordinator-owned, gitignored files that must never enter git (see check_trust_files_untracked).
TRUST_FILES = ("COORDINATOR.md", "FEATURES.md")
BASE_REF = "phase2-experiments"  # PRs target this branch; their diff is base...HEAD

# Must be ignored in the main checkout: a root .pt (the existing one and a new one), an ensemble member,
# and anything under models/ or data/. A missing trailing newline once turned `*.pt` into
# `*.ptCOORDINATOR.md` and un-ignored every root .pt.
MUST_BE_IGNORED = [
    "best_model.pt",
    "new_model.pt",
    "models/storms/expanded/seed0.pt",
    "models/x",
    "data/processed/x",
]

# --e2e cases: one sample time each, built through run_pipeline into a temp dir. Each one is compared
# either byte for byte with a crop on disk (`crop`) or with independently recorded numbers (`expect`).
E2E_CASES = [
    # F2: Michael, GOES-16. A training storm in every train list and in no val/H9/2025 list; rebuilt
    # byte for byte the same way on 2026-10-15 (logs/decisions.md, 23:05).
    dict(name="e2e GOES-16 Michael", storm="AL142018", time="2018-10-09T12:00:00Z", bucket="noaa-goes16",
         hurdat2=None, crop="data/processed/AL142018/crops/AL142018_20181009T1200Z.npz"),
    # F3: GOES-18 read/crop regression check. One exact fix of Dora EP052023, near GOES-18 nadir. It is NOT
    # East Pacific readiness: real EP builds are still blocked by the HURDAT2 cache path (pipeline.py:111)
    # and the linear longitude interpolation across the antimeridian. Its expected numbers come from
    # hc-survey's read of the same scan through the same crop code path, so this catches regressions, not
    # errors shared with that path. n=1; the ensemble result is not evidence of EP skill.
    # HURDAT2 comes from the nepac file, fetched into the temp dir: the pipeline's default cache path is
    # the protected Atlantic file, so force_hurdat2_refresh or a nepac hurdat2_url would overwrite it (I7).
    # Only the one exact fix is built, which avoids the antimeridian interpolation.
    dict(name="e2e GOES-18 read Dora", storm="EP052023", time="2023-08-06T06:00:00Z", bucket="noaa-goes18",
         hurdat2=("https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2025-092926.txt",
                  "caece626f7038db64a803a73f5985b563693f0130f3ff16b4b0a0a93c7aa0507"),
         key="ABI-L2-CMIPF/2023/218/06/OR_ABI-L2-CMIPF-M6C13_G18_s20232180600203_e20232180609523_c20232180609587.nc",
         expect=dict(shape=(285, 301), bt_min=197.96, bt_max=295.37, label=130.0)),
]
E2E_BT_MIN_RANGE = (180.0, 220.0)  # K: the coldest pixel of a hurricane crop
E2E_BT_MAX_BELOW = 320.0  # K: the top of the training clamp
E2E_BT_TOLERANCE_K = 0.01  # against recorded numbers, which are rounded to 0.01 K
E2E_TOLERANCE_KT = 30.0
MEMBERS_SHA = "results/ensemble/members_sha256.txt"


def git(*args, cwd=CHECKOUT_ROOT, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def find_main_root(explicit):
    if explicit:
        return Path(explicit).resolve()
    common = git("rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()
    return Path(common).parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_expected(coordinator_md):
    """(path, sha256) pairs from the first ```sha256 block after HASH_HEADING in COORDINATOR.md."""
    coordinator_md = Path(coordinator_md)
    if not coordinator_md.is_file():
        raise FileNotFoundError(f"{coordinator_md}: the source of truth for protected hashes is missing")
    lines = coordinator_md.read_text(encoding="utf-8-sig").splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == HASH_HEADING)
        fence = next(i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("```"))
        if lines[fence].strip() != "```sha256":
            raise StopIteration
        end = next(i for i in range(fence + 1, len(lines)) if lines[i].strip() == "```")
    except StopIteration:
        raise AssertionError(f"{coordinator_md} has no ```sha256 block under {HASH_HEADING!r}") from None
    expected = []
    for line in lines[fence + 1:end]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdef" for c in parts[0]):
            raise AssertionError(f"{coordinator_md}: malformed hash line {line!r}")
        expected.append((parts[1].lstrip("*").replace("\\", "/"), parts[0]))
    paths = [p for p, _ in expected]
    if sorted(paths) != sorted(PROTECTED_PATHS) or len(set(paths)) != len(paths):
        raise AssertionError(f"{coordinator_md} lists {sorted(paths)}, expected exactly {sorted(PROTECTED_PATHS)}")
    return expected


def resolve_base(root):
    for ref in (BASE_REF, f"origin/{BASE_REF}"):
        if git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", cwd=root, check=False).returncode == 0:
            return ref
    raise AssertionError(f"neither {BASE_REF} nor origin/{BASE_REF} resolves in {root}")


def is_trust_file(path):
    """A repo-root path that Windows (case-insensitive NTFS, trailing dots and spaces dropped) would open as one of
    TRUST_FILES. Root only: the trusted copies live at the root; a nested x/coordinator.md replaces nothing."""
    if "/" in path:
        return False
    return path.rstrip(". ").casefold() in {f.casefold() for f in TRUST_FILES}


def git_paths(*args, cwd):
    """Raw (-z, unquoted) paths from a git listing command."""
    return [p for p in git(*args, "-z", cwd=cwd).stdout.split("\0") if p.strip()]


def check_trust_files_untracked(main_root):
    """COORDINATOR.md and FEATURES.md must never be in git, under any spelling Windows treats as the same file. A PR
    that force-adds one (git add -f, e.g. as coordinator.md) would replace the main checkout's trusted, gitignored
    copy on merge, pull or checkout, without a "would be overwritten" error.

    Every path is listed in full and filtered with is_trust_file, rather than with a git pathspec, so case, trailing
    dots and spaces are all caught by one rule."""
    problems = []
    for root in dict.fromkeys([CHECKOUT_ROOT, main_root]):
        tracked = sorted({p for p in git_paths("ls-files", cwd=root) if is_trust_file(p)})
        if tracked:
            problems.append(f"{tracked} tracked in {root}")
        history = sorted({p for p in git_paths("log", "--all", "--format=", "--name-only", cwd=root) if is_trust_file(p)})
        if history:
            problems.append(f"{history} appear in git history of {root} (git log --all)")
    base = resolve_base(CHECKOUT_ROOT)
    in_diff = sorted({p for p in git_paths("diff", "--name-only", f"{base}...HEAD", cwd=CHECKOUT_ROOT)
                      if is_trust_file(p)})
    if in_diff:
        problems.append(f"{in_diff} changed in {base}...HEAD of {CHECKOUT_ROOT}")
    print(f"trust files {', '.join(TRUST_FILES)} (root level, case-insensitive, trailing dots/spaces ignored): "
          f"not tracked (ls-files), not in {base}...HEAD, not in git log --all; checked in {CHECKOUT_ROOT} and "
          f"{main_root}" if not problems else "")
    if problems:
        raise AssertionError("trust file in git:\n" + "\n".join(problems))


def check_hashes(main_root, coordinator_md):
    check_trust_files_untracked(main_root)
    print(f"expected hashes from: {coordinator_md}")
    expected = load_expected(coordinator_md)
    bad = []
    for rel, digest in expected:
        path = main_root / rel
        if not path.is_file():
            bad.append(f"MISSING {path}")
        elif sha256(path) != digest:
            bad.append(f"CHANGED {path}")
    if bad:
        raise AssertionError("\n".join(bad))
    return f"{len(expected)} protected files match"


def checkout_env(**extra):
    """Environment that makes a subprocess import hurricane_intensity from this checkout's src/."""
    env = dict(os.environ, PYTHONPATH=str(CHECKOUT_SRC), **extra)
    r = subprocess.run([sys.executable, "-c", "import hurricane_intensity; print(hurricane_intensity.__file__)"],
                       env=env, capture_output=True, text=True, check=True)
    imported = Path(r.stdout.strip()).resolve()
    print(f"hurricane_intensity imported from: {imported}")
    if not imported.is_relative_to(CHECKOUT_SRC):
        raise AssertionError(f"hurricane_intensity resolves to {imported}, not under {CHECKOUT_SRC}")
    return env


def check_pytest(main_root):
    env = checkout_env(HC_MAIN_ROOT=str(main_root))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                       cwd=CHECKOUT_ROOT, env=env, capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.stderr.strip():
        print(r.stderr.rstrip())
    if r.returncode != 0:
        raise AssertionError(f"pytest exited {r.returncode}")
    return r.stdout.strip().splitlines()[-1]


def make_fixture(tmp):
    """A tiny synthetic storm in evaluate.py's layout: <tmp>/data/processed/<id>/manifest.csv + crops."""
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    from hurricane_intensity import splits

    if not Path(splits.__file__).resolve().is_relative_to(CHECKOUT_SRC):
        raise AssertionError(f"fixture imported splits from {splits.__file__}, not {CHECKOUT_SRC}")
    fake = "XX991900"
    crops = tmp / "data" / "processed" / fake / "crops"
    crops.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    # (hour, wind_kt, is_interpolated): one Cat 1 frame for evaluate.py's per-storm table, two true fixes
    for hour, wind, interp in [(0, 70.0, False), (1, 72.5, True), (6, 100.0, False)]:
        name = f"{fake}_19000101T{hour:02d}00Z.npz"
        bt = rng.uniform(190.0, 300.0, size=(40, 44)).astype(np.float32)
        np.savez(
            crops / name, bt=bt, dqf=np.zeros_like(bt), wind_kt=np.float32(wind), lat=np.float32(20.0),
            lon=np.float32(-60.0), sample_time_unix=np.int64(0), scan_time_unix=np.int64(0),
            offset_min=np.float32(0.0), is_interpolated=np.bool_(interp), atcf_id=np.str_(fake),
            storm_name=np.str_("FIXTURE"), status=np.str_("" if interp else "HU"),
        )
        rows.append({
            "atcf_id": fake, "storm_name": "FIXTURE", "sample_time": f"1900-01-01T{hour:02d}:00:00+00:00",
            "lat": 20.0, "lon": -60.0, "wind_kt": wind, "is_interpolated": interp,
            "status": "" if interp else "HU", "match_status": "ok", "scan_time": "", "offset_min": 0.0,
            "source_key": "synthetic", "crop_path": str(Path(fake) / "crops" / name), "bt_shape": "40x44",
        })
    pd.DataFrame(rows).to_csv(tmp / "data" / "processed" / fake / "manifest.csv", index=False)

    holdout = tmp / "holdout.txt"
    holdout.write_text(fake + "\n")
    # evaluate.py has no --val-list: load_splits uses splits.VAL_LIST and refuses a train list that
    # doesn't name those storms. They are names only; evaluate.py builds no dataset for them.
    train = tmp / "train.txt"
    val_ids = splits.load_storm_list(splits.VAL_LIST)
    train.write_text("XX981900\n" + "\n".join(val_ids) + "\n")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    ckpt = tmp / "random_resnet18.pt"
    torch.save(model.state_dict(), ckpt)
    return ckpt, holdout, train, len(rows)


def check_smoke(main_root):
    protected_csvs = [main_root / "results" / "heldout_predictions.csv",
                      CHECKOUT_ROOT / "results" / "heldout_predictions.csv"]
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in protected_csvs if p.exists()}
    env = checkout_env()
    tmp = Path(tempfile.mkdtemp(prefix="hc_check_"))
    try:
        ckpt, holdout, train, n = make_fixture(tmp)
        out = tmp / "preds.csv"
        cmd = [sys.executable, str(CHECKOUT_ROOT / "scripts" / "evaluate.py"),
               "--checkpoint", str(ckpt), "--holdout-list", str(holdout),
               "--train-list", str(train), "--out", str(out)]
        print("cwd:", tmp)
        print("cmd:", subprocess.list2cmdline(cmd))
        print(f"PYTHONPATH={env['PYTHONPATH']}")
        r = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True)
        print(r.stdout.rstrip())
        if r.returncode != 0:
            print(r.stderr.rstrip())
            raise AssertionError(f"evaluate.py exited {r.returncode}")
        import pandas as pd
        rows = len(pd.read_csv(out))
        assert rows == n, f"{out} has {rows} rows, expected {n}"
        csvs = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*.csv"))
        assert csvs == ["data/processed/XX991900/manifest.csv", "preds.csv"], f"unexpected CSVs in tmp: {csvs}"
        assert not (tmp / "results").exists(), "evaluate.py created results/ in the temp dir"
        after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in protected_csvs if p.exists()}
        assert after == before, f"results/heldout_predictions.csv changed: {before} -> {after}"
        return f"evaluate.py wrote {rows} rows to the temp dir only"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_git(main_root):
    problems = []
    status = git("status", "--porcelain", "--", "results/final_2025", cwd=main_root).stdout
    if status.strip():
        problems.append(f"results/final_2025 has changes:\n{status}")
    want = git("rev-parse", f"{MAIN_SHA}^{{commit}}", cwd=main_root).stdout.strip()
    for ref in ("main", "origin/main"):
        got = git("rev-parse", ref, cwd=main_root).stdout.strip()
        if got != want:
            problems.append(f"{ref} is {got[:7]}, expected {MAIN_SHA}")
    for root in sorted({main_root, CHECKOUT_ROOT}):
        diff = git("diff", "--stat", MAIN_SHA, "HEAD", "--", "site", cwd=root).stdout
        if diff.strip():
            problems.append(f"site/ differs from {MAIN_SHA} at HEAD of {root}:\n{diff}")
    if problems:
        raise AssertionError("\n".join(problems))
    return f"results/final_2025 clean; main == origin/main == {MAIN_SHA}; site/ unchanged"


def check_gitignore(target):
    target = Path(target)
    print(f"{target / '.gitignore'}:")
    raw = (target / ".gitignore").read_bytes()
    if not raw.endswith(b"\n"):
        missing = ["(file does not end with a newline; the next appended line joins the last one)"]
    else:
        missing = []
    for rel in MUST_BE_IGNORED:
        r = git("check-ignore", "--no-index", "-q", rel, cwd=target, check=False)
        if r.returncode == 1:
            missing.append(rel)
        elif r.returncode != 0:
            raise RuntimeError(f"git check-ignore {rel} in {target} exited {r.returncode}: {r.stderr.strip()}")
        print(f"  {'ignored    ' if r.returncode == 0 else 'NOT IGNORED'}  {rel}")
    if missing:
        raise AssertionError(f"{target / '.gitignore'} fails: {missing}")
    return f"{len(MUST_BE_IGNORED)} paths ignored, ends with a newline: {target}"


def check_gitignores(main_root):
    """The PR's own .gitignore and the main checkout's; both must pass."""
    failed = []
    for root in dict.fromkeys([CHECKOUT_ROOT, main_root]):
        try:
            check_gitignore(root)
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append(str(root))
    if failed:
        raise AssertionError(f".gitignore fails in {failed}")
    return f"{len(MUST_BE_IGNORED)} paths ignored and trailing newline, in the PR checkout and the main checkout"


def remove_tree(path):
    """Remove a temp dir, never ignoring errors. Returns None once it is gone, or why it is still there (a file
    held open on Windows, e.g. an unclosed np.load, makes rmtree fail; ignore_errors would hide that)."""
    try:
        shutil.rmtree(path)
    except OSError as e:
        return f"could not remove {path}: {e!r}"
    return f"{path} still exists after rmtree" if Path(path).exists() else None


def snapshot(root):
    """path -> (size, mtime_ns) for every file under root."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            st = os.stat(p)
            out[p] = (st.st_size, st.st_mtime_ns)
    return out


def fetch_hurdat2(url, digest, dest):
    """Download a HURDAT2 file into the temp dir (never into data/) and check its SHA-256."""
    import requests

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    got = sha256(dest)
    if got != digest:
        raise AssertionError(f"{url} has SHA-256 {got}, expected {digest}")
    return dest


def check_e2e(main_root, case):
    import logging

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    from hurricane_intensity import config, crop, hurdat2, pipeline
    from hurricane_intensity.dataset import TCDataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    storm, bucket = case["storm"], case["bucket"]
    target = pd.Timestamp(case["time"])
    on_disk = main_root / case["crop"] if case.get("crop") else None
    if on_disk is not None and not on_disk.is_file():
        raise FileNotFoundError(on_disk)
    if not config.HURDAT2_CACHE_PATH.is_file():
        raise FileNotFoundError(f"{config.HURDAT2_CACHE_PATH}: the pipeline would download it")
    if config.PROJECT_ROOT.resolve() != main_root:
        raise AssertionError(f"hurricane_intensity is installed from {config.PROJECT_ROOT}, not {main_root}")

    data_before = snapshot(main_root / "data")
    tmp = Path(tempfile.mkdtemp(prefix="hc_e2e_"))
    scratch = tmp / "_scratch.nc"
    out_root = tmp / "processed"
    calls = {"keys": []}

    orig_grid, orig_scan, orig_h2 = pipeline.build_sample_grid, crop.process_one_scan, hurdat2.download_hurdat2

    def one_time_grid(fixes, interval):
        grid = orig_grid(fixes, interval)
        grid = grid[grid["time"] == target].reset_index(drop=True)
        assert len(grid) == 1, f"{case['time']} is not on the sample grid"
        return grid

    def scan_to_tmp(s3_client, key, *a, **kw):
        assert "scratch_path" not in kw
        calls["keys"].append(key)
        return functools.partial(orig_scan, scratch_path=scratch)(s3_client, key, *a, **kw)

    t0 = time.perf_counter()
    error = None
    try:
        # The pipeline is never given hurdat2_url or force_hurdat2_refresh: either would write the
        # protected Atlantic cache. The stub serves the cache read-only, or a temp copy of another file.
        if case["hurdat2"] is None:
            hurdat2_path = config.HURDAT2_CACHE_PATH
            def stub_hurdat2(url=None, force=False, **kw):
                assert url is None and not force and not kw, "e2e must read the HURDAT2 cache, never refresh it"
                return orig_h2()
        else:
            hurdat2_path = fetch_hurdat2(*case["hurdat2"], tmp / "hurdat2.txt")
            def stub_hurdat2(url=None, force=False, **kw):
                assert url is None and not force and not kw, "e2e must never refresh the HURDAT2 cache"
                return hurdat2_path

        pipeline.build_sample_grid, crop.process_one_scan, hurdat2.download_hurdat2 = (
            one_time_grid, scan_to_tmp, stub_hurdat2)
        try:
            manifest_path = pipeline.run_pipeline(atcf_id=storm, out_root=out_root, bucket=bucket)
        finally:
            pipeline.build_sample_grid, crop.process_one_scan, hurdat2.download_hurdat2 = (
                orig_grid, orig_scan, orig_h2)
        build_s = time.perf_counter() - t0

        assert not scratch.exists(), f"scratch file left behind: {scratch}"
        manifest = pd.read_csv(manifest_path)
        assert len(manifest) == 1 and manifest.loc[0, "match_status"] == "ok", manifest.to_string()
        row = manifest.iloc[0]
        key = row["source_key"]
        assert calls["keys"] == [key]
        built = out_root / row["crop_path"]

        _, fixes = hurdat2.parse_storm(storm, hurdat2_path)
        fix = fixes[fixes["time"] == target]
        assert len(fix) == 1, "no exact HURDAT2 fix at the target time"
        label = float(fix["wind_kt"].iloc[0])
        assert float(row["wind_kt"]) == label and not bool(row["is_interpolated"])

        with np.load(built) as d:  # closed at once: an open NpzFile locks the crop and the temp dir on Windows
            bt = d["bt"]
            dqf_nonzero = int(np.count_nonzero(d["dqf"]))
        bt_min, bt_max = float(np.min(bt)), float(np.max(bt))
        nans = int(np.isnan(bt).sum())

        # Training preprocessing: TCDataset on the manifest run_pipeline wrote (clip 180-320, scale,
        # bilinear to 224, 3 channels; the expanded members were trained unmasked and unrotated).
        ds = TCDataset(str(out_root), [storm])
        assert len(ds) == 1
        image, ds_label = ds[0]
        assert abs(float(ds_label) * 185.0 - label) < 1e-3

        sha_lines = [l for l in (main_root / MEMBERS_SHA).read_text().splitlines() if l.strip()]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model = model.to(device).eval()
        preds = []
        for line in sha_lines:
            digest, rel = line.split(maxsplit=1)
            path = main_root / rel.lstrip("*")
            if sha256(path) != digest:
                raise AssertionError(f"{path} does not match {MEMBERS_SHA}")
            model.load_state_dict(torch.load(path, map_location=device))
            with torch.no_grad():
                preds.append(float(model(image.unsqueeze(0).to(device)).squeeze()) * 185.0)
        preds = np.array(preds)
        total_s = time.perf_counter() - t0

        print(f"  S3 key:        s3://{bucket}/{key}")
        print(f"  scan time:     {row['scan_time']}  (offset {row['offset_min']} min)")
        print(f"  HURDAT2 label: {label:.1f} kt (exact fix, status {row['status']}, from {hurdat2_path.name})")
        print(f"  crop shape:    {bt.shape[0]}x{bt.shape[1]}  ({built.stat().st_size} bytes)")
        print(f"  BT min/max:    {bt_min:.2f} / {bt_max:.2f} K  NaNs: {nans}  nonzero DQF: {dqf_nonzero}")

        problems = []
        if on_disk is not None:
            same_bytes = built.read_bytes() == on_disk.read_bytes()
            with np.load(on_disk) as z:
                same_bt = bool(np.array_equal(bt, z["bt"]))
            print(f"  byte-identical to {case['crop']}: {same_bytes}  (bt arrays equal: {same_bt})")
            if not same_bytes:
                problems.append(f"temp crop differs from {on_disk} (bt arrays equal: {same_bt})")
        if "key" in case and key != case["key"]:
            problems.append(f"matched {key}, expected {case['key']}")
        if "expect" in case:
            e = case["expect"]
            print(f"  expected:      shape {e['shape'][0]}x{e['shape'][1]}, BT {e['bt_min']:.2f} / {e['bt_max']:.2f} K, "
                  f"label {e['label']:.1f} kt (within {E2E_BT_TOLERANCE_K} K)")
            if tuple(bt.shape) != tuple(e["shape"]):
                problems.append(f"crop shape {bt.shape}, expected {e['shape']}")
            for got, want, what in ((bt_min, e["bt_min"], "BT min"), (bt_max, e["bt_max"], "BT max")):
                if abs(got - want) > E2E_BT_TOLERANCE_K:
                    problems.append(f"{what} {got:.3f} K, expected {want} +- {E2E_BT_TOLERANCE_K}")
            if label != e["label"]:
                problems.append(f"label {label} kt, expected {e['label']}")
            if dqf_nonzero:
                problems.append(f"{dqf_nonzero} nonzero DQF pixels, expected none")
        print(f"  members:       {' '.join(f'{p:.1f}' for p in preds)}")
        print(f"  ensemble mean: {preds.mean():.2f} kt  error {preds.mean() - label:+.2f} kt")
        print(f"  member spread: sd {preds.std(ddof=1):.2f} kt, range {preds.min():.1f}-{preds.max():.1f} kt")
        print(f"  runtime:       build {build_s:.1f} s, total {total_s:.1f} s")

        if nans:
            problems.append(f"{nans} NaNs in the crop")
        if not E2E_BT_MIN_RANGE[0] <= bt_min <= E2E_BT_MIN_RANGE[1]:
            problems.append(f"BT min {bt_min:.2f} K outside {E2E_BT_MIN_RANGE}")
        if not bt_max < E2E_BT_MAX_BELOW:
            problems.append(f"BT max {bt_max:.2f} K not below {E2E_BT_MAX_BELOW}")
        if abs(preds.mean() - label) > E2E_TOLERANCE_KT:
            problems.append(f"ensemble mean {preds.mean():.2f} kt is more than {E2E_TOLERANCE_KT} kt from {label}")
        if problems:
            raise AssertionError("\n".join(problems))
    except BaseException as e:
        error = e
        raise
    finally:
        # Runs on success and on any exception partway. A cleanup failure fails the case loudly; it never replaces
        # the case's own exception, which propagates with the cleanup failures attached and printed.
        failures = []
        leak = remove_tree(tmp)
        if leak:
            failures.append(f"temp dir left behind: {leak}")
        data_after = snapshot(main_root / "data")
        if data_after != data_before:
            changed = sorted(set(data_after.items()) ^ set(data_before.items()))
            failures.append(f"data/ changed during e2e: {changed[:10]}")
        if failures:
            if error is None:
                raise AssertionError("\n".join(failures))
            for f in failures:
                print(f"  ALSO: {f}", flush=True)
                error.add_note(f"also: {f}")
    return f"{storm} {case['time']}: ensemble {preds.mean():.1f} kt vs label {label:.0f} kt; data/ unchanged"


def run_sections(sections):
    failed = []
    results = []
    for name, fn in sections:
        print(f"\n=== {name} ===", flush=True)
        t = time.perf_counter()
        try:
            msg = fn()
            ok = True
        except Exception as e:
            sys.stdout.flush()
            traceback.print_exc()
            sys.stderr.flush()
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            ok = False
        dt = time.perf_counter() - t
        results.append((name, ok, dt, msg))
        if not ok:
            failed.append(name)
    print("\n=== summary ===")
    for name, ok, dt, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<22} {dt:6.1f} s  {msg}")
    return failed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", help="main checkout holding data/, models/ and results/ (default: from git)")
    p.add_argument("--e2e", action="store_true", help="instead of F1, run the E2E_CASES (Michael GOES-16, Dora GOES-18): download, build, predict")
    p.add_argument("--gitignore-target", help="run only the gitignore section, against this repo")
    p.add_argument("--coordinator-md", help="file holding the protected-hash block (default: <main root>/COORDINATOR.md)")
    args = p.parse_args()

    t0 = time.perf_counter()
    if args.gitignore_target:
        print(f"gitignore target: {Path(args.gitignore_target).resolve()}")
        failed = run_sections([("gitignore", lambda: check_gitignore(args.gitignore_target))])
    else:
        main_root = find_main_root(args.repo_root)
        coordinator_md = Path(args.coordinator_md).resolve() if args.coordinator_md else main_root / "COORDINATOR.md"
        print(f"checkout:  {CHECKOUT_ROOT}")
        print(f"main root: {main_root}")
        print(f"python:    {sys.executable}")
        hashes = ("hashes", lambda: check_hashes(main_root, coordinator_md))
        if args.e2e:
            # --e2e uses the main install: config.PROJECT_ROOT must be the main checkout (see the docstring).
            import hurricane_intensity
            print(f"hurricane_intensity imported from: {Path(hurricane_intensity.__file__).resolve()} (main install)")
            sections = [hashes]
            for case in E2E_CASES:
                sections.append((case["name"], functools.partial(check_e2e, main_root, case)))
                sections.append((f"hashes after {case['name'].split()[-1]}",
                                 lambda: check_hashes(main_root, coordinator_md)))
        else:
            # F1 tests this checkout's code, in this process too (the smoke fixture reads splits.VAL_LIST).
            sys.path.insert(0, str(CHECKOUT_SRC))
            sections = [
                hashes,
                ("pytest", lambda: check_pytest(main_root)),
                ("evaluate.py smoke", lambda: check_smoke(main_root)),
                ("hashes after smoke", lambda: check_hashes(main_root, coordinator_md)),
                ("git I3/I6", lambda: check_git(main_root)),
                ("gitignore", lambda: check_gitignores(main_root)),
            ]
        failed = run_sections(sections)
    print(f"\n{'FAIL' if failed else 'PASS'}  total {time.perf_counter() - t0:.1f} s"
          + (f"  failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
