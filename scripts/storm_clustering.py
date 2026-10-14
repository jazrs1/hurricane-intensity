"""Is the embedding organized by storm identity? CPU only; reads saved embeddings, trains nothing.

Uses the training-frame embeddings saved by embedding_neighbours.py (models/loss/l1/seed2.pt, chosen there by
rule). For every training frame, takes its 20 nearest training frames by Euclidean distance, excluding itself
and every frame of the same storm within 6 hours of it, and measures the fraction from the same storm.
Chance levels per query frame:
  naive         the storm's share of the training set, n_storm / N
  adjusted      the same, after removing the frames the query cannot pick (itself, its +-6 h window)
  intensity     the storm's share among eligible frames within +-10 kt of the query's label: what
                same-storm fraction to expect if neighbours were picked by intensity alone
Writes results/embedding_neighbours/storm_clustering_l1_seed2.csv. Run from the repo root.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
EMB = sys.argv[2] if len(sys.argv) > 2 else "results/embedding_neighbours/embeddings_l1_seed2.npz"
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
from hurricane_intensity.dataset import TCDataset  # noqa: E402
from hurricane_intensity.splits import load_splits  # noqa: E402

K, WINDOW_H, LABEL_TOL = 20, 6, 10
BANDS = [(0, 34, "TD"), (34, 64, "TS"), (64, 83, "Cat 1"), (83, 96, "Cat 2"), (96, 113, "Cat 3"), (113, 137, "Cat 4"), (137, np.inf, "Cat 5")]
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

z = np.load(REPO / EMB, allow_pickle=True)
F = z["train"]
train_ids = load_splits(train_list=REPO / "storms_train_baseline.txt")[0]
df = TCDataset("data/processed", train_ids).df.reset_index(drop=True)
assert ((df.atcf_id + "|" + df.sample_time).to_numpy() == z["train_key"]).all(), "embedding order does not match the dataset"
N = len(df)
storm = df.atcf_id.to_numpy()
name = df.storm_name.str.title().to_numpy()
y = df.wind_kt.to_numpy(float)
ts = pd.to_datetime(df.sample_time)
t = ((ts - ts.min()).dt.total_seconds() / 3600).to_numpy()  # hours; resolution-independent
assert np.isclose(np.diff(t[:2])[0], 1.0), "consecutive hourly frames should be 1 h apart"
band = pd.cut(y, [b[0] for b in BANDS] + [np.inf], right=False, labels=[b[2] for b in BANDS]).astype(str)
D = (F ** 2).sum(1)[:, None] + (F ** 2).sum(1)[None, :] - 2 * F @ F.T
same = storm[:, None] == storm[None, :]
n_storm = pd.Series(storm).map(pd.Series(storm).value_counts()).to_numpy()


def run(window_h):
    blocked = same & (np.abs(t[:, None] - t[None, :]) <= window_h)  # includes the frame itself
    d = np.where(blocked, np.inf, D)
    idx = np.argsort(d, axis=1)[:, :K]
    obs = same[np.arange(N)[:, None], idx].mean(1)
    w = blocked.sum(1)  # frames removed from the candidate pool, self included
    adjusted = (n_storm - w) / (N - w)
    eligible = ~blocked & (np.abs(y[:, None] - y[None, :]) <= LABEL_TOL)
    intensity = (eligible & same).sum(1) / np.maximum(eligible.sum(1), 1)
    return pd.DataFrame(dict(atcf_id=storm, storm=name, sample_time=df.sample_time, wind_kt=y, band=band, same_storm_frac=obs,
                             chance_naive=n_storm / N, chance_adjusted=adjusted, chance_intensity=intensity,
                             all_same=(obs == 1.0), eligible_within_tol=eligible.sum(1)))


def table(r, by):
    g = r.groupby(by, sort=False)
    s = g.agg(n=("same_storm_frac", "size"), observed=("same_storm_frac", "mean"), chance_naive=("chance_naive", "mean"),
              chance_adjusted=("chance_adjusted", "mean"), chance_intensity=("chance_intensity", "mean"), all_20_same=("all_same", "mean"))
    s["obs / naive"] = s.observed / s.chance_naive
    s["obs / intensity"] = s.observed / s.chance_intensity
    return s


r = run(WINDOW_H)
r.to_csv(REPO / "results" / "embedding_neighbours" / "storm_clustering_l1_seed2.csv", index=False)
print(f"{N} training frames, {len(set(storm))} storms; k={K}; excluded: self and same-storm frames within +-{WINDOW_H} h "
      f"(mean {(r.chance_naive * N).mean():.0f} frames per storm)")
order = [b for *_, b in BANDS]
print("\n== same-storm fraction among the 20 nearest training frames, against chance ==")
print(table(r.assign(all="all frames"), "all").to_string())
print("\nby intensity band of the query frame:")
print(table(r, "band").reindex(order).to_string())
print("\nby storm:")
print(table(r, "storm").sort_values("n", ascending=False).to_string())
print(f"\nframes with no candidate within +-{LABEL_TOL} kt outside their window: {(r.eligible_within_tol == 0).sum()}")

r24 = run(24)
print(f"\n== sensitivity: exclude same-storm frames within +-24 h instead of +-{WINDOW_H} h ==")
print(table(r24.assign(all="all frames"), "all").to_string())
print(table(r24, "band").reindex(order)[["n", "observed", "chance_naive", "chance_intensity", "obs / naive", "obs / intensity"]].to_string())
