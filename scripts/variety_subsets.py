"""Frame lists for the variety-against-volume experiment. CPU only; trains nothing. Run from the repo root.

Two pools of training frames, both already on disk:
  dense   the original nine training storms at hourly (as built)
  sparse  all 19 training storms at 3-hourly: the original nine thinned to their 3-hour grid points (a subset of
          the hourly crops) plus the ten new storms (built at 3-hourly)
Matching, per seed, so both arms get identical intensity distributions:
  1. 5 kt bins nested inside the Saffir-Simpson bands (edges: every 5 kt plus 34/64/83/96/113/137). In every bin
     where both pools have frames, each arm draws min(dense, sparse) frames from its pool.
  2. Fallback: bins where one pool has no frames cannot be matched exactly. Within each band, those orphan bins are
     pooled and each arm draws min(dense orphans, sparse orphans) frames from them. Reported per band; orphan frames
     beyond that are dropped and reported.
Draws are without replacement and deterministic from the seed (numpy default_rng([seed, arm])); a fresh draw per
seed. Writes results/variety/frame_lists/<arm>_seed<s>.txt (crop paths as in the manifests), plus
results/variety/frame_lists/summary.csv (per arm, seed and band: count, mean and sd of label), matching.csv (per bin
and band: pool counts, target, fallback) and sha256.txt. Re-running reproduces the lists byte for byte.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import os  # noqa: E402

os.chdir(REPO)
from hurricane_intensity import config, hurdat2  # noqa: E402
from hurricane_intensity.dataset import TCDataset  # noqa: E402
from hurricane_intensity.pipeline import build_sample_grid  # noqa: E402
from hurricane_intensity.splits import load_splits  # noqa: E402

SEEDS = range(10)
ARMS = {"dense": 0, "sparse": 1}
OUT = REPO / "results" / "variety" / "frame_lists"
H = REPO / "data" / "interim" / "hurdat2_atlantic.txt"
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
EDGES = sorted(set(range(0, 200, 5)) | {34, 64, 83, 96, 113, 137}) + [np.inf]

nine, val, hold = load_splits(train_list=REPO / "storms_train_control2.txt", val_list=REPO / "storms_val_expanded.txt", holdout_list=REPO / "storms_holdout_h9.txt")
nineteen = load_splits(train_list=REPO / "storms_train_expanded.txt", val_list=REPO / "storms_val_expanded.txt", holdout_list=REPO / "storms_holdout_h9.txt")[0]
new = [s for s in nineteen if s not in nine]

dense = TCDataset("data/processed", nine).df.copy()
keep = []
for s in nine:
    grid3 = set(build_sample_grid(hurdat2.parse_storm(s, H, config.DEFAULT_INCLUDE_STATUSES)[1], "3h").time.dt.strftime("%Y-%m-%dT%H:%M"))
    d = dense[dense.atcf_id == s]
    keep.append(d[pd.to_datetime(d.sample_time).dt.strftime("%Y-%m-%dT%H:%M").isin(grid3)])
sparse = pd.concat(keep + [TCDataset("data/processed", new).df], ignore_index=True)
pools = {"dense": dense.reset_index(drop=True), "sparse": sparse.reset_index(drop=True)}
for p in pools.values():
    p["bin"] = pd.cut(p.wind_kt, EDGES, right=False).astype(str)
    p["band"] = pd.cut(p.wind_kt, [b[0] for b in BANDS] + [np.inf], right=False, labels=[b[2] for b in BANDS]).astype(str)

# matching plan (counts only; identical for every seed)
bins = sorted(set(pools["dense"].bin) | set(pools["sparse"].bin), key=lambda b: float(b.split(",")[0][1:]))
plan = []
for b in bins:
    nd, ns = int((pools["dense"].bin == b).sum()), int((pools["sparse"].bin == b).sum())
    band = (pools["dense"][pools["dense"].bin == b].band.tolist() or pools["sparse"][pools["sparse"].bin == b].band.tolist())[0]
    plan.append(dict(bin=b, band=band, dense_pool=nd, sparse_pool=ns, matched_in_bin=min(nd, ns), orphan=(nd == 0) != (ns == 0)))
plan = pd.DataFrame(plan)
fallback = {}
for band, g in plan.groupby("band", sort=False):
    od, os_ = int(g[g.orphan].dense_pool.sum()), int(g[g.orphan].sparse_pool.sum())
    fallback[band] = dict(dense_orphans=od, sparse_orphans=os_, matched_at_band=min(od, os_), dropped=od + os_ - 2 * min(od, os_))
plan["band_fallback_matched"] = plan.band.map({k: v["matched_at_band"] for k, v in fallback.items()})
OUT.mkdir(parents=True, exist_ok=True)
plan.to_csv(OUT / "matching.csv", index=False)

rows, hashes = [], []
for seed in SEEDS:
    for arm, aid in ARMS.items():
        pool = pools[arm]
        rng = np.random.default_rng([seed, aid])
        chosen = []
        for _, r in plan[~plan.orphan & (plan.matched_in_bin > 0)].iterrows():
            idx = pool.index[pool.bin == r.bin].to_numpy()
            chosen += list(rng.choice(idx, size=r.matched_in_bin, replace=False))
        for band, f in fallback.items():
            if f["matched_at_band"]:
                orphan_bins = plan[(plan.band == band) & plan.orphan].bin
                idx = pool.index[pool.bin.isin(orphan_bins)].to_numpy()
                chosen += list(rng.choice(idx, size=f["matched_at_band"], replace=False))
        sel = pool.loc[sorted(chosen)]
        assert sel.crop_path.is_unique
        assert not sel.atcf_id.isin(val + hold).any(), "validation or held-out frame in a training list"
        text = "\n".join(sel.crop_path) + "\n"
        path = OUT / f"{arm}_seed{seed}.txt"
        path.write_text(text, encoding="utf-8", newline="\n")
        hashes.append(f"{hashlib.sha256(text.encode()).hexdigest()}  {path.name}")
        for band, g in sel.groupby("band", sort=False):
            rows.append(dict(arm=arm, seed=seed, band=band, n=len(g), mean_kt=g.wind_kt.mean(), sd_kt=g.wind_kt.std(), storms=g.atcf_id.nunique()))
        rows.append(dict(arm=arm, seed=seed, band="all", n=len(sel), mean_kt=sel.wind_kt.mean(), sd_kt=sel.wind_kt.std(), storms=sel.atcf_id.nunique(),
                         ge96=int((sel.wind_kt >= 96).sum()), ge113=int((sel.wind_kt >= 113).sum()), ge137=int((sel.wind_kt >= 137).sum()),
                         largest_storm_share=sel.atcf_id.value_counts(normalize=True).max(), dorian_share=(sel.atcf_id == "AL052019").mean()))
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "summary.csv", index=False)
(OUT / "sha256.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8", newline="\n")

# checks: identical counts per bin (matched bins) and per band, every seed
for seed in SEEDS:
    a = {arm: pd.read_csv(OUT / f"{arm}_seed{seed}.txt", header=None)[0] for arm in ARMS}
    sel = {arm: pools[arm].set_index("crop_path").loc[a[arm]] for arm in ARMS}
    for col in ("band",):
        assert (sel["dense"][col].value_counts().sort_index() == sel["sparse"][col].value_counts().sort_index()).all(), f"band counts differ, seed {seed}"
    mb = plan[~plan.orphan].bin
    cd, cs = sel["dense"].bin.value_counts(), sel["sparse"].bin.value_counts()
    assert all(cd.get(b, 0) == cs.get(b, 0) for b in mb), f"bin counts differ, seed {seed}"
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)
print(f"pools: dense {len(pools['dense'])} frames / {pools['dense'].atcf_id.nunique()} storms; sparse {len(pools['sparse'])} / {pools['sparse'].atcf_id.nunique()}")
print(f"bins: {len(plan)}; matched exactly: {int((~plan.orphan & (plan.matched_in_bin > 0)).sum())}; orphan (one pool empty): {int(plan.orphan.sum())}")
print("orphan bins:", plan[plan.orphan][["bin", "band", "dense_pool", "sparse_pool"]].to_dict("records"))
print("band fallback:", {k: v for k, v in fallback.items() if v["dense_orphans"] or v["sparse_orphans"]})
by_band = summ[summ.band != "all"].groupby(["band", "arm"]).agg(n=("n", "first"), mean_kt=("mean_kt", "mean"), mean_kt_sd_over_seeds=("mean_kt", "std"), storms=("storms", "mean")).unstack("arm")
print(by_band.reindex([b[2] for b in BANDS]).to_string(float_format=lambda v: f"{v:.2f}"))
print(summ[summ.band == "all"].groupby("arm")[["n", "mean_kt", "storms", "ge96", "ge113", "ge137", "largest_storm_share", "dorian_share"]].agg(["min", "max"]).T.to_string(float_format=lambda v: f"{v:.3f}"))
print("checks passed: per-band counts identical in every seed; matched-bin counts identical; no validation or H9 frame")
