#!/usr/bin/env python
"""Crop the analog-forecasting storms (configs/analog_storms.csv, usable == 1) at 3-hourly sampling.

    .venv\\Scripts\\python.exe scripts\\expand_analog.py --workers 4            # every usable storm not built
    .venv\\Scripts\\python.exe scripts\\expand_analog.py --ids AL012017 EP012024 # a pilot

Each storm goes through hurricane_intensity.pipeline.run_pipeline (the build_dataset code path) with the
bucket chosen in analog_storms.csv, and for EP storms the pinned nepac HURDAT2 (hurdat2_path). Raw full-disk
files are deleted right after each crop by crop.process_one_scan; each worker process gets its own scratch
file (data/raw/_scratch_w<i>.nc) so parallel workers never share one.

Resumable: a storm with a manifest.csv under data/processed/<ID>/ is skipped (run_pipeline writes the
manifest last). Progress is appended to data/exports/expand_log.txt. A network error on any S3 call is
retried (RETRY_DELAYS_S); if the network stays down the storm is abandoned unsaved, the worker waits for the
network (up to NETWORK_WAIT_S) and retries the storm, so an outage never saves a storm with gaps. No sealed or spent storm is in the CSV;
run_pipeline and parse_storm refuse them anyway.
"""
from __future__ import annotations

import argparse
import functools
import logging
import multiprocessing as mp
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hurricane_intensity import config

STORMS_CSV = config.PROJECT_ROOT / "configs" / "analog_storms.csv"
LOG = config.PROJECT_ROOT / "data" / "exports" / "expand_log.txt"


def log(msg):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def storm_bytes(atcf_id):
    d = config.PROCESSED_DIR / atcf_id
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


RETRY_DELAYS_S = (15, 30, 60, 120, 240)  # per S3 call: ~8 min of retries before the network counts as down
NETWORK_WAIT_S = 6 * 3600                 # then wait up to 6 h for it to come back, probing every minute
STORM_ATTEMPTS = 3


class NetworkDown(BaseException):
    """Raised past run_pipeline's `except Exception` (which would record the scan as missing and save the storm
    with a gap): the storm is abandoned unsaved and retried once the network is back."""


def network_errors():
    import botocore.exceptions as be
    return (be.ConnectionError, be.HTTPClientError, ConnectionError, TimeoutError, socket.gaierror)


def with_retry(fn, i, what):
    errors = network_errors()

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        for attempt, delay in enumerate((*RETRY_DELAYS_S, None), start=1):
            try:
                return fn(*args, **kwargs)
            except errors as e:
                if delay is None:
                    raise NetworkDown(f"{what}: {type(e).__name__}: {e}") from e
                log(f"w{i} network error in {what} (attempt {attempt}), retrying in {delay} s: {type(e).__name__}")
                time.sleep(delay)
    return wrapped


def wait_for_network(i, s3, bucket):
    t0 = time.time()
    while time.time() - t0 < NETWORK_WAIT_S:
        try:
            s3.list_objects_v2(Bucket=bucket, Prefix=config.ABI_PRODUCT + "/", MaxKeys=1)
            log(f"w{i} network back after {time.time() - t0:.0f} s")
            return True
        except network_errors():
            time.sleep(60)
    return False


def worker(i, jobs, interval):
    import shutil

    import tqdm
    from hurricane_intensity import crop, goes_index, pipeline

    logging.basicConfig(level=logging.WARNING)
    tqdm.tqdm.__init__ = functools.partialmethod(tqdm.tqdm.__init__, disable=True)
    crop.process_one_scan = with_retry(
        functools.partial(crop.process_one_scan, scratch_path=config.RAW_DIR / f"_scratch_w{i}.nc"),
        i, "download")
    goes_index.find_nearest_scan = with_retry(goes_index.find_nearest_scan, i, "S3 listing")
    probe = goes_index.make_anonymous_s3_client()
    for atcf_id, basin, bucket in jobs:
        if (config.PROCESSED_DIR / atcf_id / "manifest.csv").is_file():
            log(f"w{i} skip {atcf_id}: already built")
            continue
        for attempt in range(1, STORM_ATTEMPTS + 1):
            t0 = time.time()
            try:
                manifest = pipeline.run_pipeline(
                    atcf_id, sample_interval=interval, bucket=bucket,
                    hurdat2_path=config.HURDAT2_EP_CACHE_PATH if basin == "EP" else None)
                df = pd.read_csv(manifest)
                ok = int((df["match_status"] == "ok").sum())
                log(f"w{i} done {atcf_id} ({bucket}): {ok}/{len(df)} cropped, "
                    f"{storm_bytes(atcf_id) / 1e6:.1f} MB, {time.time() - t0:.0f} s")
                break
            except NetworkDown as e:
                shutil.rmtree(config.PROCESSED_DIR / atcf_id, ignore_errors=True)  # no manifest: never half-saved
                log(f"w{i} NETWORK DOWN {atcf_id} attempt {attempt}/{STORM_ATTEMPTS}, partial removed: {e}")
                if attempt == STORM_ATTEMPTS or not wait_for_network(i, probe, bucket):
                    log(f"w{i} FAIL {atcf_id} ({bucket}): network down; worker stops (rerun to resume)")
                    return
            except Exception as e:  # logged and skipped, as build_batch.py does
                log(f"w{i} FAIL {atcf_id} ({bucket}) after {time.time() - t0:.0f} s: {type(e).__name__}: {e}")
                break


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ids", nargs="*", help="only these storms (default: every usable storm not yet built)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--interval", default="3h")
    args = p.parse_args()

    df = pd.read_csv(STORMS_CSV)
    df = df[df["usable"] == 1]
    if args.ids:
        unknown = set(args.ids) - set(df["atcf_id"])
        if unknown:
            raise SystemExit(f"not usable storms in {STORMS_CSV.name}: {sorted(unknown)}")
        df = df[df["atcf_id"].isin(args.ids)]
    todo = [(r.atcf_id, r.basin, r.bucket) for r in df.itertuples()
            if not (config.PROCESSED_DIR / r.atcf_id / "manifest.csv").is_file()]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log(f"start: {len(todo)} storms to build, {args.workers} workers, interval {args.interval}")
    t0 = time.time()
    shards = [todo[i::args.workers] for i in range(args.workers)]
    procs = [mp.Process(target=worker, args=(i, shards[i], args.interval)) for i in range(args.workers)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    log(f"finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
