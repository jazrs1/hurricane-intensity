"""Is the high-end underestimation a representation problem or a head problem? Inference only; trains nothing.

Takes one L1 checkpoint from models/loss/ (the run whose H6 >96 kt bias is closest to the L1 arm mean),
extracts the 512-d penultimate features (avgpool output, before fc) for every training frame and every H6
frame, and for each held-out frame finds its 20 nearest training frames by Euclidean distance. Reports the
spread of the neighbours' labels and splits each frame's error into
    pred - truth = (pred - neighbour mean) + (neighbour mean - truth)
                     head term                  representation term
Decision rule fixed in logs/decisions.md (2026-10-14, "Embedding-neighbour test ... design") before this ran.
Writes results/embedding_neighbours/. Run from the repo root.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
from hurricane_intensity.dataset import TCDataset  # noqa: E402
from hurricane_intensity.splits import load_splits  # noqa: E402

K = 20
STRONG = 113  # fraction of neighbours at or above this: Cat 4 or stronger
OUT = REPO / "results" / "embedding_neighbours"
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")

# checkpoint rule: L1 run whose H6 >96 kt bias is closest to the L1 arm mean
runs = pd.read_csv(REPO / "results" / "loss" / "summary_per_run.csv")
l1 = runs[(runs.arm == "l1") & (runs.set == "h6")].set_index("seed")[">96 bias"]
SEED = int((l1 - l1.mean()).abs().idxmin())
CKPT = REPO / "models" / "loss" / "l1" / f"seed{SEED}.pt"
print(f"checkpoint: {CKPT.relative_to(REPO)} (H6 >96 bias {l1[SEED]:.2f}; L1 arm mean {l1.mean():.2f})")

train_ids, _, h6_ids = load_splits(train_list=REPO / "storms_train_baseline.txt", holdout_list=REPO / "storms_holdout_h6.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(CKPT))
fc = model.fc
model.fc = nn.Identity()
model = model.to(device).eval()
fc = fc.to(device).eval()


def embed(ids):
    ds = TCDataset("data/processed", ids)
    feats, preds = [], []
    with torch.no_grad():
        for i in range(0, len(ds), 32):  # same batch size as evaluate.py
            x = torch.stack([ds[j][0] for j in range(i, min(i + 32, len(ds)))]).to(device)
            f = model(x)
            feats.append(f.cpu())
            preds.append(fc(f).squeeze(1).cpu())
    df = ds.df.copy().reset_index(drop=True)
    df["truth"] = df.wind_kt.astype(float)
    df["pred"] = torch.cat(preds).numpy() * 185.0
    df["storm"] = df.storm_name.str.title()
    t = pd.to_datetime(df.sample_time)
    df["fix6h"] = df.atcf_id + "_" + t.dt.strftime("%Y%m%d") + "_" + (t.dt.hour // 6).astype(str)
    df["band"] = pd.cut(df.truth, [b[0] for b in BANDS] + [np.inf], right=False, labels=[b[2] for b in BANDS]).astype(str)
    return df, torch.cat(feats).numpy()


tr, Ftr = embed(train_ids)
ho, Fho = embed(h6_ids)
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / f"embeddings_l1_seed{SEED}.npz", train=Ftr, heldout=Fho,
                    train_key=(tr.atcf_id + "|" + tr.sample_time).to_numpy(), heldout_key=(ho.atcf_id + "|" + ho.sample_time).to_numpy())

ref = pd.read_csv(REPO / "results" / "loss" / f"l1_seed{SEED}_h6.csv")
m = ho.merge(ref[["atcf_id", "sample_time", "pred"]], on=["atcf_id", "sample_time"], suffixes=("", "_ref"))
print(f"train frames {len(tr)} ({(tr.truth >= STRONG).sum()} at >= {STRONG} kt) | H6 frames {len(ho)} | "
      f"fc(embedding) vs evaluate.py predictions: max |diff| {np.abs(m.pred - m.pred_ref).max():.4f} kt over {len(m)} frames")


def neighbours(Fq, Fbase, metric):
    if metric == "cosine":
        Fq = Fq / np.linalg.norm(Fq, axis=1, keepdims=True)
        Fbase = Fbase / np.linalg.norm(Fbase, axis=1, keepdims=True)
    d = (Fq ** 2).sum(1)[:, None] + (Fbase ** 2).sum(1)[None, :] - 2 * Fq @ Fbase.T
    return d


def describe(q, Fq, base, Fbase, metric="euclidean", exclude_same_storm=False):
    d = neighbours(Fq, Fbase, metric)
    if exclude_same_storm:
        d[q.atcf_id.to_numpy()[:, None] == base.atcf_id.to_numpy()[None, :]] = np.inf
    idx = np.argsort(d, axis=1)[:, :K]
    lab = base.truth.to_numpy()[idx]
    out = q[["atcf_id", "storm", "sample_time", "is_interpolated", "truth", "pred", "band"]].copy().reset_index(drop=True)
    out["nbr_mean"], out["nbr_median"], out["nbr_sd"] = lab.mean(1), np.median(lab, 1), lab.std(1, ddof=1)
    out["nbr_min"], out["nbr_max"] = lab.min(1), lab.max(1)
    out["nbr_frac_ge113"] = (lab >= STRONG).mean(1)
    out["nbr_storms"] = [len(set(r)) for r in base.atcf_id.to_numpy()[idx]]
    out["nbr_fixes"] = [len(set(r)) for r in base.fix6h.to_numpy()[idx]]
    out["nbr_top_storm_share"] = [pd.Series(r).value_counts().iloc[0] / K for r in base.storm.to_numpy()[idx]]
    out["head_term"] = out.pred - out.nbr_mean
    out["repr_term"] = out.nbr_mean - out.truth
    out["error"] = out.pred - out.truth
    out["nbr_median_minus_mean"] = out.nbr_median - out.nbr_mean
    return out


def summary(df, by):
    g = df.groupby(by, sort=False)
    s = g.agg(n=("truth", "size"), truth=("truth", "mean"), pred=("pred", "mean"), nbr_mean=("nbr_mean", "mean"), nbr_sd=("nbr_sd", "mean"),
              frac_ge113=("nbr_frac_ge113", "mean"), nbr_min=("nbr_min", "mean"), nbr_max=("nbr_max", "mean"),
              storms=("nbr_storms", "mean"), fixes=("nbr_fixes", "mean"), top_storm_share=("nbr_top_storm_share", "mean"),
              error=("error", "mean"), head_term=("head_term", "mean"), head_sd=("head_term", "std"),
              repr_term=("repr_term", "mean"), repr_sd=("repr_term", "std"), median_minus_mean=("nbr_median_minus_mean", "mean"))
    return s


hi = ho[ho.truth > 96]
held = describe(ho, Fho, tr, Ftr)
held.to_csv(OUT / f"heldout_neighbours_l1_seed{SEED}.csv", index=False)
held_hi = held[held.truth > 96]

ctrl_q = tr[tr.truth >= STRONG]
ctrl = describe(ctrl_q, Ftr[ctrl_q.index.to_numpy()], tr, Ftr, exclude_same_storm=True)
ctrl.to_csv(OUT / f"train_cat45_neighbours_other_storms_l1_seed{SEED}.csv", index=False)

rows = []
for name, d in [("held-out Cat 3", held_hi[held_hi.band == "Cat 3"]), ("held-out Cat 4", held_hi[held_hi.band == "Cat 4"]),
                ("held-out Cat 5", held_hi[held_hi.band == "Cat 5"]), ("held-out all >96", held_hi),
                ("REF held-out < 64 kt", held[held.truth < 64]),
                ("REF training Cat 4/5, other storms' neighbours", ctrl)]:
    rows.append(summary(d.assign(group=name), "group"))
main = pd.concat(rows)
main.to_csv(OUT / f"summary_l1_seed{SEED}.csv")
print(f"\n== k={K} nearest training frames (Euclidean, raw 512-d features): label spread and error split ==")
print("head_term = pred - neighbour mean; repr_term = neighbour mean - truth; error = their sum")
print(main.to_string())

print("\n== by storm and band, held-out >96 kt ==")
bs = summary(held_hi, ["band", "storm"])
print(bs.to_string())
bs.to_csv(OUT / f"summary_by_storm_l1_seed{SEED}.csv")

print("\n== does the prediction follow the neighbourhood? held-out >96 kt frames ==")
for b in ["Cat 3", "Cat 4", "Cat 5"]:
    d = held_hi[held_hi.band == b]
    print(f"  {b}: corr(pred, nbr_mean) {d.pred.corr(d.nbr_mean):+.2f} | corr(truth, nbr_mean) {d.truth.corr(d.nbr_mean):+.2f} | "
          f"frames with >= half their neighbours >= 113 kt: {(d.nbr_frac_ge113 >= 0.5).mean():.0%}; of those, mean error {d[d.nbr_frac_ge113 >= 0.5].error.mean():+.1f} kt")
d = held_hi
print(f"  all >96: corr(pred, nbr_mean) {d.pred.corr(d.nbr_mean):+.2f}, corr(truth, nbr_mean) {d.truth.corr(d.nbr_mean):+.2f}")

print("\n== which training storms supply the neighbours of held-out Cat 4/5 frames ==")
idx = np.argsort(neighbours(Fho[held.index[held.truth >= STRONG].to_numpy()], Ftr, "euclidean"), axis=1)[:, :K]
src = pd.DataFrame({"storm": tr.storm.to_numpy()[idx].ravel(), "label": tr.truth.to_numpy()[idx].ravel()})
print(src.groupby("storm").label.agg(["size", "mean", "min", "max"]).sort_values("size", ascending=False).to_string())

print("\n== sensitivity: cosine distance instead of Euclidean ==")
held_c = describe(ho, Fho, tr, Ftr, metric="cosine")
hc = held_c[held_c.truth > 96]
print(summary(hc.assign(group=hc.band), "group").reindex(["Cat 3", "Cat 4", "Cat 5"])[["n", "nbr_mean", "nbr_sd", "frac_ge113", "head_term", "repr_term"]].to_string())
