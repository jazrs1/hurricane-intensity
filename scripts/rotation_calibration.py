"""Calibration check for the rotation experiment. Inference only; trains nothing. Run from the repo root.

For each of the 20 runs: predict the validation frames (masked, unrotated) with the run's selected checkpoint,
fit OLS truth = a + b * pred on those validation predictions only, apply it to the run's saved H6 predictions,
and repeat the three primary comparisons (rotation - control, two-sided exact permutation, threshold 0.0167) on
the corrected predictions. Design and decision rule: logs/decisions.md, 2026-10-14 "Rotation calibration check".
Writes results/rotation/<arm>_seed<s>_val.csv, calibration_per_run.csv and calibration_tests.csv.
"""
import os
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torchvision.models import resnet18

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
from hurricane_intensity.dataset import TCDataset  # noqa: E402
from hurricane_intensity.splits import load_splits  # noqa: E402

RES = REPO / "results" / "rotation"
ARMS, SEEDS = ["control", "rotation"], range(10)
THRESHOLD = 0.0167
PRIMARY = [">96 bias", ">96 MAE", "overall MAE"]
HIGH = ["Ian", "Beryl", "Lee", "Sam"]
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

_, VAL, _ = load_splits(train_list=REPO / "storms_train_baseline.txt")
val_ds = TCDataset("data/processed", VAL, circular_mask=True)
val_x = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
val_y = val_ds.df.wind_kt.to_numpy(float)
print(f"validation: {VAL}, {len(val_y)} frames; above 96 kt: {(val_y > 96).sum()} "
      f"({val_ds.df[val_y > 96].storm_name.str.title().value_counts().to_dict()}); >= 113 kt: {(val_y >= 113).sum()}; max {val_y.max():.0f} kt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device).eval()


def predict(x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            out.append(model(x[i:i + 32].to(device)).squeeze(1).cpu())
    return torch.cat(out).numpy() * 185.0


def metrics(t, p, storm):
    e = p - t
    hi, lo = t > 96, t < 64
    m = {"overall MAE": np.abs(e).mean(), ">96 bias": e[hi].mean(), ">96 MAE": np.abs(e)[hi].mean(),
         "<64 MAE": np.abs(e)[lo].mean(), "<64 bias": e[lo].mean(), "overall bias": e.mean()}
    for s in HIGH:
        k = (storm == s) & hi
        m[f">96 bias {s}"], m[f">96 MAE {s}"] = e[k].mean(), np.abs(e)[k].mean()
    return m


rows = []
for arm in ARMS:
    for seed in SEEDS:
        model.load_state_dict(torch.load(REPO / "models" / "rotation" / arm / f"seed{seed}.pt"))
        pv = predict(val_x)
        val_ds.df.assign(truth=val_y, pred=pv)[["atcf_id", "storm_name", "sample_time", "truth", "pred"]].to_csv(RES / f"{arm}_seed{seed}_val.csv", index=False)
        log = (REPO / "logs" / "rotation" / f"{arm}_seed{seed}.log").read_text()
        logged = min(float(v) for v in re.findall(r"epoch \d+: train [\d.]+ kt\s+val ([\d.]+) kt", log))
        b, a = np.polyfit(pv, val_y, 1)  # truth = a + b * pred
        h6 = pd.read_csv(RES / f"{arm}_seed{seed}_h6.csv")
        t, p, storm = h6.truth.to_numpy(), h6.pred.to_numpy(), h6.storm_name.str.title().to_numpy()
        pc = a + b * p
        hi = t > 96
        r = dict(arm=arm, seed=seed, a=a, b=b, val_mae_recomputed=np.abs(pv - val_y).mean(), val_mae_logged=logged,
                 val_mae_corrected=np.abs(a + b * pv - val_y).mean(), val_bias=(pv - val_y).mean(),
                 pearson_all=stats.pearsonr(p, t)[0], spearman_all=stats.spearmanr(p, t)[0],
                 pearson_gt96=stats.pearsonr(p[hi], t[hi])[0], spearman_gt96=stats.spearmanr(p[hi], t[hi])[0])
        r.update({f"raw {k}": v for k, v in metrics(t, p, storm).items()})
        r.update({f"corrected {k}": v for k, v in metrics(t, pc, storm).items()})
        rows.append(r)
        print(f"{arm} seed {seed}: a {a:+7.2f}  b {b:.3f}  val MAE recomputed {r['val_mae_recomputed']:.2f} (logged {logged:.2f})", flush=True)
runs = pd.DataFrame(rows)
runs.to_csv(RES / "calibration_per_run.csv", index=False)

combs = np.array(list(combinations(range(20), 10)))
SPLIT = np.zeros((len(combs), 20), bool)
SPLIT[np.arange(len(combs))[:, None], combs] = True


def compare(col):
    x, y = runs[runs.arm == "control"][col].to_numpy(), runs[runs.arm == "rotation"][col].to_numpy()
    pool = np.concatenate([x, y])
    mb = (SPLIT * pool).sum(1) / 10
    ma = (pool.sum() - mb * 10) / 10
    obs = y.mean() - x.mean()
    return {"metric": col, "control": x.mean(), "control sd": x.std(ddof=1), "rotation": y.mean(), "rotation sd": y.std(ddof=1),
            "difference": obs, "exact p": float((np.abs(mb - ma) >= abs(obs) - 1e-12).mean())}


print(f"\nintegrity: max |recomputed - logged validation MAE| {(runs.val_mae_recomputed - runs.val_mae_logged).abs().max():.3f} kt")
print("\n== correction coefficients per arm (truth = a + b * pred, fitted on validation) ==")
print(runs.groupby("arm")[["a", "b", "val_bias", "val_mae_recomputed", "val_mae_corrected"]].agg(["mean", "std", "min", "max"]).T.to_string())

print(f"\n== the three primary comparisons, raw and corrected (rotation - control, exact permutation, threshold {THRESHOLD}) ==")
tests = []
for m in PRIMARY:
    for kind in ("raw", "corrected"):
        r = compare(f"{kind} {m}")
        r["survives"] = r["exact p"] < THRESHOLD
        tests.append(r)
tests = pd.DataFrame(tests)
tests.to_csv(RES / "calibration_tests.csv", index=False)
print(tests.to_string(index=False, formatters={"exact p": "{:.5f}".format}))
for m in [">96 bias", ">96 MAE"]:
    raw = tests[tests.metric == f"raw {m}"].difference.iloc[0]
    cor = tests[tests.metric == f"corrected {m}"].difference.iloc[0]
    print(f"  {m}: corrected difference is {cor / raw:.0%} of the raw difference")

print("\n== exploratory: calibration-free discrimination (a linear correction cannot change these) ==")
print(pd.DataFrame([compare(c) for c in ["pearson_all", "spearman_all", "pearson_gt96", "spearman_gt96"]]).to_string(index=False, formatters={"exact p": "{:.4f}".format}))
print("\n== exploratory: other corrected metrics ==")
cols = ["overall bias", "<64 MAE", "<64 bias"] + [f">96 {k} {s}" for s in HIGH for k in ("bias", "MAE")]
print(pd.DataFrame([compare(f"{kind} {c}") for c in cols for kind in ("raw", "corrected")]).to_string(index=False, formatters={"exact p": "{:.4f}".format}))
