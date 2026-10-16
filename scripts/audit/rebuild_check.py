"""Rebuild three existing GOES-16 crops through the real run_pipeline with no bucket given, into a scratch
out_root, and compare them with the crops on disk. Only the sample grid is narrowed (to one sample time per
storm); everything else is the pipeline's own code and defaults."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(REPO / "src"))
import os

os.chdir(REPO)
from hurricane_intensity import pipeline  # noqa: E402

CASES = [("AL142018", "2018-10-09T12:00:00+00:00"),   # Michael, scan mode 3
         ("AL092022", "2022-09-27T06:00:00+00:00"),   # Ian
         ("AL122024", "2024-10-05T12:00:00+00:00")]   # Kirk
orig_grid = pipeline.build_sample_grid

for atcf, when in CASES:
    target = pd.Timestamp(when)
    pipeline.build_sample_grid = lambda fixes, interval, _t=target: (
        lambda g: g[g.time == _t].reset_index(drop=True))(orig_grid(fixes, interval))
    pipeline.run_pipeline(atcf, out_root=OUT)  # sample interval, box, tolerance, bucket: all defaults
    pipeline.build_sample_grid = orig_grid

    new_manifest = pd.read_csv(OUT / atcf / "manifest.csv")
    assert len(new_manifest) == 1 and new_manifest.match_status[0] == "ok", new_manifest
    rel = new_manifest.crop_path[0]
    old_row = pd.read_csv(REPO / "data/processed" / atcf / "manifest.csv")
    old_row = old_row[old_row.sample_time == when].iloc[0]
    a, b = (REPO / "data/processed" / rel).read_bytes(), (OUT / rel).read_bytes()
    same_key = old_row.source_key == new_manifest.source_key[0]
    print(f"{atcf} {when}: source_key same: {same_key}; file bytes identical: {a == b} ({len(a)} vs {len(b)} bytes)")
    if a != b:
        za, zb = zipfile.ZipFile(io.BytesIO(a)), zipfile.ZipFile(io.BytesIO(b))
        for n in za.namelist():
            ia, ib = za.getinfo(n), zb.getinfo(n)
            print(f"   {n}: member bytes identical {za.read(n) == zb.read(n)}; zip date_time {ia.date_time} vs {ib.date_time}")
