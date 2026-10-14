import numpy as np, pandas as pd
d = pd.read_csv("results/heldout_predictions.csv")
t, p = d.truth.to_numpy(), d.pred.to_numpy()
old = [(0, 33, "TD"), (34, 63, "TS"), (64, 82, "Cat 1"), (83, 95, "Cat 2"), (96, 112, "Cat 3"), (113, 136, "Cat 4"), (137, 999, "Cat 5")]
new = [(0, 34), (34, 64), (64, 83), (83, 96), (96, 113), (113, 137), (137, np.inf)]
print("band    old n  old MAE old bias |  new n  new MAE new bias")
tot_old = 0
for (lo, hi, name), (nlo, nhi) in zip(old, new):
    mo = (t >= lo) & (t <= hi); mn = (t >= nlo) & (t < nhi); tot_old += mo.sum()
    print(f"{name:6s} {mo.sum():5d} {abs(p-t)[mo].mean():8.2f} {(p-t)[mo].mean():+8.2f} | {mn.sum():5d} {abs(p-t)[mn].mean():8.2f} {(p-t)[mn].mean():+8.2f}")
print("old binned:", tot_old, "of", len(d), "| overall MAE", round(abs(p-t).mean(), 4), "| fixes-only", round(abs(p-t)[~d.is_interpolated.astype(bool)].mean(), 4))
c1 = d[(t >= 64) & (t < 83)].groupby("storm_name").abs_error.agg(["count", "mean"]); print(c1)
