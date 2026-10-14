"""Read-only final audit, binned with evaluate.py's own BANDS. Writes nothing into the repo."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1])
ROOT = REPO / "data" / "processed"
os.chdir(REPO)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def consts(path, names):
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def storm_list(path):
    return [ln.split("#", 1)[0].strip().upper() for ln in path.read_text().splitlines() if ln.split("#", 1)[0].strip()]


tr = consts(REPO / "scripts" / "train.py", {"TRAIN_STORMS", "VAL_STORMS"})
ev = consts(REPO / "scripts" / "evaluate.py", {"HOLDOUT_STORMS", "BANDS"})
TRAIN, VAL, HOLD, BANDS = tr["TRAIN_STORMS"], tr["VAL_STORMS"], ev["HOLDOUT_STORMS"], ev["BANDS"]
NAMES = [n for _, _, n in BANDS]

excl_now = set((REPO / "excluded_crops.txt").read_text().split())
excl_head = set(subprocess.run(["git", "show", "HEAD:excluded_crops.txt"], capture_output=True, text=True, check=True).stdout.split())
print(f"excluded_crops.txt: HEAD {len(excl_head)} entries, working tree {len(excl_now)} entries; added: {sorted(excl_now - excl_head)}")

allm = pd.concat(
    [pd.read_csv(d / "manifest.csv").assign(dir_id=d.name) for d in sorted(ROOT.iterdir()) if (d / "manifest.csv").exists()],
    ignore_index=True,
)
allm["exists"] = allm["crop_path"].apply(lambda p: isinstance(p, str) and (ROOT / p).exists())


def band(w):
    for lo, hi, name in BANDS:
        if lo <= w <= hi:
            return name
    return "unbinned"


allm["band"] = allm["wind_kt"].apply(band)
allm["storm"] = allm["dir_id"] + " " + allm["storm_name"].str.title()
COLS = NAMES + ["unbinned"]


def summarize(d, by):
    t = d.groupby([by, "band"]).size().unstack(fill_value=0).reindex(columns=COLS, fill_value=0)
    t["total"] = t.sum(axis=1)
    g = d.groupby(by)
    t[">96"] = g.apply(lambda s: int((s.wind_kt > 96).sum()), include_groups=False)
    t[">96 fix"] = g.apply(lambda s: int(((s.wind_kt > 96) & ~s.is_interpolated.astype(bool)).sum()), include_groups=False)
    t["<64"] = g.apply(lambda s: int((s.wind_kt < 64).sum()), include_groups=False)
    return t


def report(title, assign, excluded):
    d = allm[allm.crop_path.notna() & ~allm.crop_path.isin(excluded) & allm.exists].copy()
    d["split"] = d["dir_id"].map(assign)
    d = d[d.split.notna()]
    order = [s for s in ["train", "val", "holdout", "holdout (orig 3)", "unassigned (gradcam list)"] if s in set(d.split)]
    print(f"\n===== {title} =====")
    t = summarize(d, "split").reindex(order)
    print(t.to_string())
    print("\nrow % of total:")
    print((t[COLS].div(t["total"], axis=0) * 100).round(1).to_string())
    for sp in order:
        print(f"\n-- {sp}: per storm --")
        print(summarize(d[d.split == sp], "storm").to_string())
    return d


before = {s: "train" for s in TRAIN} | {s: "val" for s in VAL} | {s: "holdout" for s in HOLD} | {
    s: "unassigned (gradcam list)" for s in ["AL022024", "AL132023", "AL182021"]}
after = dict(before) | {"AL022024": "train", "AL132023": "train", "AL182021": "holdout"}

report("BEFORE (train.py / evaluate.py constants, excluded_crops.txt at HEAD)", before, excl_head)
d1 = report("AFTER (Beryl, Lee -> train; Sam -> holdout; excluded_crops.txt as now on disk)", after, excl_now)

print("\n===== list files vs code =====")
f_train, f_hold, f_grad = (storm_list(REPO / n) for n in ("storms_train.txt", "storms_holdout.txt", "storms_gradcam.txt"))
want_train = set(TRAIN) | set(VAL) | {"AL022024", "AL132023"}
want_hold = set(HOLD) | {"AL182021"}
print("storms_train.txt  :", f_train)
print("storms_holdout.txt:", f_hold)
print("storms_gradcam.txt:", f_grad)
print("storms_train.txt == TRAIN+VAL+Beryl+Lee:", set(f_train) == want_train, "| dupes:", len(f_train) - len(set(f_train)))
print("storms_holdout.txt == HOLDOUT+Sam:", set(f_hold) == want_hold, "| dupes:", len(f_hold) - len(set(f_hold)))
print("storms_train.txt & storms_holdout.txt:", sorted(set(f_train) & set(f_hold)))
print("VAL storms still in storms_train.txt:", all(s in f_train for s in VAL))
print("train.py TRAIN_STORMS contains Beryl/Lee (i.e. code already updated?):", {"AL022024", "AL132023"} <= set(TRAIN))
print("evaluate.py HOLDOUT_STORMS contains Sam:", "AL182021" in HOLD)

print("\n===== leakage (AFTER) =====")
cp = allm[allm.crop_path.notna()].copy()
cp["split"] = cp["dir_id"].map(after)
print("storms mapped to a split:", cp.dir_id.nunique(), "of", allm.dir_id.nunique(), "| unmapped:", sorted(set(allm.dir_id) - set(after)))
print("duplicate crop_path rows:", int(cp.crop_path.duplicated().sum()))
print("crop_path in >1 split:", int((cp.groupby("crop_path").split.nunique() > 1).sum()))
print("GOES source_key in >1 split:", int((cp.groupby("source_key").split.nunique() > 1).sum()))
print("GOES source_key in >1 storm:", int((cp.groupby("source_key").dir_id.nunique() > 1).sum()))
print("(storm, sample_time) duplicates:", int(cp.duplicated(["dir_id", "sample_time"]).sum()))
for s in ["AL022024", "AL132023", "AL182021"]:
    sub = cp[cp.dir_id == s]
    print(f"  {s} {sub.storm_name.iloc[0]}: split={sorted(set(sub.split))} manifest_rows={int((allm.dir_id == s).sum())} crops={len(sub)} "
          f"unique_paths={sub.crop_path.nunique()} excluded={int(sub.crop_path.isin(excl_now).sum())}")

print("\n===== NaN coverage (every crop on disk) =====")
bad = []
for r in allm[allm.exists].itertuples():
    bt = np.load(ROOT / r.crop_path)["bt"]
    if not np.isfinite(bt).all():
        bad.append(r.crop_path)
labels_bad = int(allm.loc[allm.exists, "wind_kt"].isna().sum())
print(f"crops scanned: {int(allm.exists.sum())} | non-finite crops: {len(bad)} | NaN labels: {labels_bad}")
print("non-finite crops not in excluded_crops.txt:", sorted(set(bad) - excl_now))
print("excluded_crops.txt entries that are finite / unknown:", sorted(excl_now - set(bad)))
print("excluded entries all match a manifest crop_path exactly:", excl_now <= set(allm.crop_path.dropna()))

print("\n===== through the real TCDataset (after-assignment lists) =====")
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402
from hurricane_intensity.dataset import TCDataset  # noqa: E402

for name, ids in [("train", TRAIN + ["AL022024", "AL132023"]), ("val", VAL), ("holdout x4", HOLD + ["AL182021"]), ("holdout x3", HOLD)]:
    ds = TCDataset(str(ROOT), ids)
    nan = 0
    for i in range(len(ds)):
        img, lab = ds[i]
        if torch.isnan(img).any() or torch.isnan(lab).any():
            nan += 1
    print(f"  {name}: {len(ds)} frames, {nan} with NaN in the tensor the model sees")
