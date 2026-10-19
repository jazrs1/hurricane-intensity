#!/usr/bin/env python
"""One-command checks for the invariants in FEATURES.md (I1-I7).

    .venv\\Scripts\\python.exe scripts\\check.py            # F1: under 3 minutes, no network
    .venv\\Scripts\\python.exe scripts\\check.py --e2e      # F2: downloads one GOES-16 scan
    .venv\\Scripts\\python.exe scripts\\check.py --gitignore-target <repo>   # only the gitignore section

Protected files, data/ and models/ are gitignored and exist only in the main checkout, so they are
resolved against it (the parent of git's common dir, or --repo-root), even when this runs from a
worktree. Storm lists and tests come from the checkout this script lives in. A missing protected file
is a failure, never a skip. Every section runs even if an earlier one fails; the exit code is 1 if
any failed.

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
HASH_FILE = CHECKOUT_ROOT / "scripts" / "protected_sha256.txt"
MAIN_SHA = "fbb39c4"  # I6: main == origin/main == this commit, and site/ is unchanged since it

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

# F2: Michael AL142018, 2018-10-09 12:00 UTC. A training storm in every train list and in no
# val/H9/2025 list; rebuilt byte for byte the same way on 2026-10-15 (logs/decisions.md, 23:05).
E2E_STORM = "AL142018"
E2E_TIME = "2018-10-09T12:00:00Z"
E2E_CROP = "data/processed/AL142018/crops/AL142018_20181009T1200Z.npz"
E2E_BUCKET = "noaa-goes16"
E2E_BT_MIN_RANGE = (180.0, 220.0)  # K: the coldest pixel of a hurricane crop
E2E_BT_MAX_BELOW = 320.0  # K: the top of the training clamp
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


def load_expected():
    expected = []
    for line in HASH_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            expected.append((rel.strip().lstrip("*"), digest))
    return expected


def check_hashes(main_root):
    expected = load_expected()
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


def check_pytest(main_root):
    env = dict(os.environ, HC_MAIN_ROOT=str(main_root))
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
    tmp = Path(tempfile.mkdtemp(prefix="hc_check_"))
    try:
        ckpt, holdout, train, n = make_fixture(tmp)
        out = tmp / "preds.csv"
        cmd = [sys.executable, str(CHECKOUT_ROOT / "scripts" / "evaluate.py"),
               "--checkpoint", str(ckpt), "--holdout-list", str(holdout),
               "--train-list", str(train), "--out", str(out)]
        print("cwd:", tmp)
        print("cmd:", subprocess.list2cmdline(cmd))
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
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
    missing = []
    for rel in MUST_BE_IGNORED:
        r = git("check-ignore", "--no-index", "-q", rel, cwd=target, check=False)
        if r.returncode == 1:
            missing.append(rel)
        elif r.returncode != 0:
            raise RuntimeError(f"git check-ignore {rel} in {target} exited {r.returncode}: {r.stderr.strip()}")
        print(f"  {'ignored    ' if r.returncode == 0 else 'NOT IGNORED'}  {rel}")
    if missing:
        raise AssertionError(f"{target / '.gitignore'} does not ignore: {missing}")
    return f"{len(MUST_BE_IGNORED)} paths ignored in {target}"


def snapshot(root):
    """path -> (size, mtime_ns) for every file under root."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            st = os.stat(p)
            out[p] = (st.st_size, st.st_mtime_ns)
    return out


def check_e2e(main_root):
    import logging

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    from hurricane_intensity import config, crop, hurdat2, pipeline
    from hurricane_intensity.dataset import TCDataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    target = pd.Timestamp(E2E_TIME)
    on_disk = main_root / E2E_CROP
    if not on_disk.is_file():
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
        assert len(grid) == 1, f"{E2E_TIME} is not on the sample grid"
        return grid

    def scan_to_tmp(s3_client, key, *a, **kw):
        assert "scratch_path" not in kw
        calls["keys"].append(key)
        return functools.partial(orig_scan, scratch_path=scratch)(s3_client, key, *a, **kw)

    def cached_hurdat2(url=None, force=False, **kw):
        assert url is None and not force and not kw, "e2e must read the HURDAT2 cache, never refresh it"
        return orig_h2()

    t0 = time.perf_counter()
    try:
        pipeline.build_sample_grid, crop.process_one_scan, hurdat2.download_hurdat2 = (
            one_time_grid, scan_to_tmp, cached_hurdat2)
        try:
            manifest_path = pipeline.run_pipeline(atcf_id=E2E_STORM, out_root=out_root, bucket=E2E_BUCKET)
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

        _, fixes = hurdat2.parse_storm(E2E_STORM, config.HURDAT2_CACHE_PATH)
        fix = fixes[fixes["time"] == target]
        assert len(fix) == 1, "no exact HURDAT2 fix at the target time"
        label = float(fix["wind_kt"].iloc[0])
        assert float(row["wind_kt"]) == label and not bool(row["is_interpolated"])

        d = np.load(built)
        bt = d["bt"]
        bt_min, bt_max = float(np.min(bt)), float(np.max(bt))
        nans = int(np.isnan(bt).sum())

        same_bytes = built.read_bytes() == on_disk.read_bytes()
        same_bt = bool(np.array_equal(bt, np.load(on_disk)["bt"]))

        # Training preprocessing: TCDataset on the manifest run_pipeline wrote (clip 180-320, scale,
        # bilinear to 224, 3 channels; the expanded members were trained unmasked and unrotated).
        ds = TCDataset(str(out_root), [E2E_STORM])
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

        print(f"  S3 key:        s3://{E2E_BUCKET}/{key}")
        print(f"  scan time:     {row['scan_time']}  (offset {row['offset_min']} min)")
        print(f"  HURDAT2 label: {label:.1f} kt (exact fix, status {row['status']})")
        print(f"  crop shape:    {bt.shape[0]}x{bt.shape[1]}  ({built.stat().st_size} bytes)")
        print(f"  BT min/max:    {bt_min:.2f} / {bt_max:.2f} K  NaNs: {nans}")
        print(f"  byte-identical to {E2E_CROP}: {same_bytes}  (bt arrays equal: {same_bt})")
        print(f"  members:       {' '.join(f'{p:.1f}' for p in preds)}")
        print(f"  ensemble mean: {preds.mean():.2f} kt  error {preds.mean() - label:+.2f} kt")
        print(f"  member spread: sd {preds.std(ddof=1):.2f} kt, range {preds.min():.1f}-{preds.max():.1f} kt")
        print(f"  runtime:       build {build_s:.1f} s, total {total_s:.1f} s")

        problems = []
        if nans:
            problems.append(f"{nans} NaNs in the crop")
        if not E2E_BT_MIN_RANGE[0] <= bt_min <= E2E_BT_MIN_RANGE[1]:
            problems.append(f"BT min {bt_min:.2f} K outside {E2E_BT_MIN_RANGE}")
        if not bt_max < E2E_BT_MAX_BELOW:
            problems.append(f"BT max {bt_max:.2f} K not below {E2E_BT_MAX_BELOW}")
        if not same_bytes:
            problems.append(f"temp crop differs from {on_disk} (bt arrays equal: {same_bt})")
        if abs(preds.mean() - label) > E2E_TOLERANCE_KT:
            problems.append(f"ensemble mean {preds.mean():.2f} kt is more than {E2E_TOLERANCE_KT} kt from {label}")
        if problems:
            raise AssertionError("\n".join(problems))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        data_after = snapshot(main_root / "data")
        if data_after != data_before:
            changed = sorted(set(data_after.items()) ^ set(data_before.items()))
            raise AssertionError(f"data/ changed during e2e: {changed[:10]}")
    return f"Michael {E2E_TIME}: ensemble {preds.mean():.1f} kt vs label {label:.0f} kt; data/ unchanged"


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
    p.add_argument("--e2e", action="store_true", help="F2 instead of F1: build Michael 2018-10-09 12:00 from S3")
    p.add_argument("--gitignore-target", help="run only the gitignore section, against this repo")
    args = p.parse_args()

    t0 = time.perf_counter()
    if args.gitignore_target:
        print(f"gitignore target: {Path(args.gitignore_target).resolve()}")
        failed = run_sections([("gitignore", lambda: check_gitignore(args.gitignore_target))])
    else:
        main_root = find_main_root(args.repo_root)
        print(f"checkout:  {CHECKOUT_ROOT}")
        print(f"main root: {main_root}")
        print(f"python:    {sys.executable}")
        hashes = ("hashes", lambda: check_hashes(main_root))
        if args.e2e:
            sections = [hashes, ("e2e GOES-16 Michael", lambda: check_e2e(main_root)),
                        ("hashes after e2e", lambda: check_hashes(main_root))]
        else:
            sections = [
                hashes,
                ("pytest", lambda: check_pytest(main_root)),
                ("evaluate.py smoke", lambda: check_smoke(main_root)),
                ("hashes after smoke", lambda: check_hashes(main_root)),
                ("git I3/I6", lambda: check_git(main_root)),
                ("gitignore", lambda: check_gitignore(main_root)),
            ]
        failed = run_sections(sections)
    print(f"\n{'FAIL' if failed else 'PASS'}  total {time.perf_counter() - t0:.1f} s"
          + (f"  failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
