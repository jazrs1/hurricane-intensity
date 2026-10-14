"""Robustness check for the more-distinct-storms experiment (not a primary test). Read-only, CPU only.

The control arm's H9 overall MAE rose with seed (Spearman +0.72; seed and run order are confounded within an
arm). This repeats the three primary comparisons using only control seeds 0-4, the earlier and better runs,
against all ten expanded runs, with the same two-sided exact permutation test (all C(15, 5) = 3003 splits) and
the same 0.0167 threshold. Reads results/storms/summary_per_run.csv; writes results/storms/robustness_control_seeds0-4.csv.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RES = REPO / "results" / "storms"
PRIMARY = ["spearman >96", ">96 bias", "overall MAE"]
THRESHOLD = 0.0167


def exact_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool, n, k = np.concatenate([a, b]), len(a) + len(b), len(b)
    obs = b.mean() - a.mean()
    hits = total = 0
    for idx in combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        d = pool[m].mean() - pool[~m].mean()
        hits += abs(d) >= abs(obs) - 1e-12
        total += 1
    return hits / total, total


runs = pd.read_csv(RES / "summary_per_run.csv")
d = runs[runs.set == "h9"]
ctrl_all, ctrl_early, exp = d[d.arm == "control"], d[(d.arm == "control") & (d.seed <= 4)], d[d.arm == "expanded"]
rows = []
for m in PRIMARY:
    p, n = exact_p(ctrl_early[m], exp[m])
    rows.append({"metric": m, "control seeds 0-4": ctrl_early[m].mean(), "control all ten": ctrl_all[m].mean(), "expanded": exp[m].mean(),
                 "difference vs seeds 0-4": exp[m].mean() - ctrl_early[m].mean(), "difference vs all ten (primary)": exp[m].mean() - ctrl_all[m].mean(),
                 "exact p (5 vs 10)": p, "splits": n, "passes 0.0167": p < THRESHOLD,
                 "expanded runs beyond seeds 0-4 range": int(((exp[m] < ctrl_early[m].min()) | (exp[m] > ctrl_early[m].max())).sum())})
out = pd.DataFrame(rows)
out.to_csv(RES / "robustness_control_seeds0-4.csv", index=False)
pd.set_option("display.width", 250)
print("ROBUSTNESS CHECK (not a primary test): control seeds 0-4 vs all ten expanded runs, H9 pooled")
print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("control seeds 0-4 run order:", ctrl_early.sort_values("seed").order.tolist())
