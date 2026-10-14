"""Read-only audit of the storm splits. Writes nothing into the repo."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1])
ROOT = REPO / "data" / "processed"

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def consts(path, names):
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def storm_list(path):
    ids = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line.upper())
    return ids


tr = consts(REPO / "scripts" / "train.py", {"TRAIN_STORMS", "VAL_STORMS"})
ev = consts(REPO / "scripts" / "evaluate.py", {"HOLDOUT_STORMS"})
TRAIN, VAL, HOLD = tr["TRAIN_STORMS"], tr["VAL_STORMS"], ev["HOLDOUT_STORMS"]
GRADCAM = storm_list(REPO / "storms_gradcam.txt")
f_train = storm_list(REPO / "storms_train.txt")
f_hold = storm_list(REPO / "storms_holdout.txt")

print("== split sources ==")
print("train.py TRAIN_STORMS :", TRAIN)
print("train.py VAL_STORMS   :", VAL)
print("evaluate.py HOLDOUT   :", HOLD)
print("storms_train.txt      :", f_train)
print("storms_holdout.txt    :", f_hold)
print("storms_gradcam.txt    :", GRADCAM)
print("storms_train.txt == TRAIN+VAL:", set(f_train) == set(TRAIN) | set(VAL), " dupes:", len(f_train) - len(set(f_train)))
print("storms_holdout.txt == HOLDOUT:", set(f_hold) == set(HOLD))

excluded = set((REPO / "excluded_crops.txt").read_text().split())
print("\nexcluded_crops.txt entries:", len(excluded))

# Same filtering as TCDataset: crop_path present, not excluded, file exists.
frames = []
for d in sorted(p for p in ROOT.iterdir() if (p / "manifest.csv").exists()):
    df = pd.read_csv(d / "manifest.csv")
    df["dir_id"] = d.name
    df["n_manifest"] = 1
    frames.append(df)
allm = pd.concat(frames, ignore_index=True)
allm["has_crop"] = allm["crop_path"].notna()
allm["excluded"] = allm["crop_path"].isin(excluded)
allm["exists"] = allm["crop_path"].apply(lambda p: isinstance(p, str) and (ROOT / p).exists())
allm["loaded"] = allm["has_crop"] & ~allm["excluded"] & allm["exists"]

print("\n== per-storm manifest accounting ==")
acc = allm.groupby(["dir_id", "storm_name"]).agg(
    manifest_rows=("n_manifest", "sum"), has_crop=("has_crop", "sum"), excluded=("excluded", "sum"),
    exists=("exists", "sum"), loaded=("loaded", "sum"), max_kt=("wind_kt", "max"),
    wind_nan=("wind_kt", lambda s: int(s.isna().sum())),
)
print(acc)
print("atcf_id column matches directory for every row:", bool((allm["atcf_id"] == allm["dir_id"]).all()))
print("crop_path prefix matches directory for every row with a crop:",
      bool(allm[allm.has_crop].apply(lambda r: r["crop_path"].startswith(r["dir_id"] + "\\"), axis=1).all()))

ds = allm[allm["loaded"]].copy()

EDGES = [-np.inf, 34, 64, 83, 96, 113, 137, np.inf]
NAMES = ["TD <34", "TS 34-63", "Cat1 64-82", "Cat2 83-95", "Cat3 96-112", "Cat4 113-136", "Cat5 137+"]
ds["band"] = pd.cut(ds["wind_kt"], EDGES, labels=NAMES, right=False)

# evaluate.py's BANDS are closed integer ranges with gaps (33<w<34, 63<w<64 ...): count frames it would not bin.
EV = [(0, 33), (34, 63), (64, 82), (83, 95), (96, 112), (113, 136), (137, 999)]
ds["in_eval_band"] = np.any([(ds["wind_kt"] >= lo) & (ds["wind_kt"] <= hi) for lo, hi in EV], axis=0)


def table(assign, title):
    print(f"\n== {title} ==")
    d = ds.copy()
    d["split"] = d["dir_id"].map(assign)
    d = d[d["split"].notna()]
    order = [s for s in ["train", "val", "holdout", "gradcam-only (unassigned)"] if s in set(d["split"])]
    t = pd.crosstab(d["split"], d["band"], dropna=False).reindex(order).reindex(columns=NAMES)
    t["total"] = t.sum(axis=1)
    t[">96"] = d[d.wind_kt > 96].groupby("split").size().reindex(order).fillna(0).astype(int)
    t[">=96"] = d[d.wind_kt >= 96].groupby("split").size().reindex(order).fillna(0).astype(int)
    t["fixes"] = d[~d.is_interpolated.astype(bool)].groupby("split").size().reindex(order).fillna(0).astype(int)
    print(t)
    pct = (t[NAMES].div(t["total"], axis=0) * 100).round(1)
    print("\nrow %:")
    print(pct)
    print("\nframes evaluate.py's BANDS would leave unbinned (non-integer kt in a gap):")
    print(d[~d.in_eval_band].groupby("split").size().reindex(order).fillna(0).astype(int).to_string())
    return d


def per_storm(d, split):
    s = d[d.split == split]
    t = pd.crosstab([s["dir_id"], s["storm_name"]], s["band"], dropna=False).reindex(columns=NAMES)
    t["total"] = t.sum(axis=1)
    t[">96"] = s[s.wind_kt > 96].groupby(["dir_id", "storm_name"]).size().reindex(t.index).fillna(0).astype(int)
    t[">96 fixes"] = s[(s.wind_kt > 96) & ~s.is_interpolated.astype(bool)].groupby(["dir_id", "storm_name"]).size().reindex(t.index).fillna(0).astype(int)
    t["max_kt"] = s.groupby(["dir_id", "storm_name"])["wind_kt"].max().reindex(t.index)
    print(f"\n-- {split}, per storm --")
    print(t)


before = {s: "train" for s in TRAIN} | {s: "val" for s in VAL} | {s: "holdout" for s in HOLD} | {s: "gradcam-only (unassigned)" for s in GRADCAM}
after = dict(before) | {"AL022024": "train", "AL132023": "train", "AL182021": "holdout"}

d0 = table(before, "BEFORE (as train.py / evaluate.py load them)")
for sp in ["train", "val", "holdout", "gradcam-only (unassigned)"]:
    per_storm(d0, sp)
d1 = table(after, "AFTER reassignment (Beryl, Lee -> train; Sam -> holdout)")
for sp in ["train", "holdout"]:
    per_storm(d1, sp)

unassigned = sorted(set(ds.dir_id) - set(before))
print("\nprocessed storms in no list:", unassigned)

print("\n== leakage checks (AFTER assignment) ==")
d1all = allm.copy()
d1all["split"] = d1all["dir_id"].map(after)
d1all = d1all[d1all.split.notna()]
storm_splits = {}
for s, sp in after.items():
    storm_splits.setdefault(s, set()).add(sp)
lists = {"train": set(TRAIN) | {"AL022024", "AL132023"}, "val": set(VAL), "holdout": set(HOLD) | {"AL182021"}}
for a in lists:
    for b in lists:
        if a < b:
            print(f"storm ids {a} & {b}:", sorted(lists[a] & lists[b]))
cp = d1all[d1all.has_crop]
print("duplicate crop_path rows overall:", int(cp["crop_path"].duplicated().sum()))
print("crop_path in >1 split:", int((cp.groupby("crop_path")["split"].nunique() > 1).sum()))
sk = cp.groupby("source_key")["split"].agg(lambda s: sorted(set(s)))
multi = sk[sk.apply(len) > 1]
print("GOES source_key shared by >1 split:", len(multi))
if len(multi):
    print(cp[cp.source_key.isin(multi.index)][["dir_id", "storm_name", "sample_time", "split", "source_key"]].sort_values("source_key").to_string())
skst = cp.groupby("source_key")["dir_id"].nunique()
print("GOES source_key shared by >1 storm (any split):", int((skst > 1).sum()))
for s in ["AL022024", "AL132023", "AL182021"]:
    sub = cp[cp.dir_id == s]
    print(f"{s}: split(s)={sorted(set(sub.split))} rows={len(sub)} unique crop_path={sub.crop_path.nunique()} "
          f"unique sample_time={sub.sample_time.nunique()} time range {sub.sample_time.min()} .. {sub.sample_time.max()}")

print("\n== NaN scan of every crop on disk (raw bt, the array TCDataset clips and resizes) ==")
rows = []
for r in allm[allm.exists].itertuples():
    bt = np.load(ROOT / r.crop_path)["bt"]
    n = int(np.isnan(bt).sum())
    inf = int(np.isinf(bt).sum())
    if n or inf or pd.isna(r.wind_kt):
        rows.append(dict(storm=r.dir_id, name=r.storm_name, sample_time=r.sample_time, wind_kt=r.wind_kt,
                         nan_px=n, nan_pct=round(100 * n / bt.size, 2), inf_px=inf, shape=bt.shape,
                         in_excluded=r.excluded, split_after=after.get(r.dir_id, "-"), crop_path=r.crop_path))
bad = pd.DataFrame(rows)
print(f"crops scanned: {int(allm.exists.sum())}   with NaN/inf pixels or NaN label: {len(bad)}")
if len(bad):
    print(bad.drop(columns=["crop_path"]).to_string())
    print("\nNOT covered by excluded_crops.txt:")
    for p in bad[~bad.in_excluded].crop_path:
        print("  ", p)
print("\nexcluded_crops.txt entries with no NaN on disk / not in any manifest:")
badset = set(bad.crop_path) if len(bad) else set()
for p in sorted(excluded):
    print("  ", p, "| nan:", p in badset, "| in manifest:", p in set(allm.crop_path.dropna()), "| exists:", (ROOT / p).exists())

beryl = allm[(allm.dir_id == "AL022024") & allm.sample_time.str.startswith("2024-07-02T1")]
print("\nBeryl 2024-07-02 10:00-19:00 rows:")
print(beryl[["sample_time", "wind_kt", "is_interpolated", "match_status", "crop_path", "excluded", "exists", "bt_shape"]].to_string())
