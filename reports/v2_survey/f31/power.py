"""F-31: power of the pre-registered tests at n = 10 and n = 20 seeds (deterministic Monte Carlo).

Same method as N5 design section 5 (reports/v2_survey/n5/power.py): normal model for the per-seed paired
differences, two-sided exact sign-flip test over all 2^n sign assignments, seeded generator, 2000 reps per
power cell, the 80% MDE from 1000 reps per grid step.

One change, for speed at n = 20: the "informative null" (CI by test inversion inside +-SESOI) is decided by
testing the two edges, p(d + SESOI) < alpha and p(d - SESOI) < alpha, instead of scanning a grid. For the
sign-flip test of a mean shift the acceptance set is an interval, so the two are the same.

    .venv\\Scripts\\python.exe -I -B reports/v2_survey/f31/power.py

sd inputs (all observed, no held-out trace used):
  within-run, two selections of one run: 0.54 kt overall MAE (N5 control, native - matched, seeds 0-9,
    results/n5_atlantic/control_seed*_h9.csv vs *_native_h9.csv; picks 0-5 epochs apart);
  N5 planning case: 0.85 kt (design section 5);
  N5 observed, atlantic - control: 1.12 kt overall MAE (matched), 4.79 kt >96 bias (report.txt).
"""
import itertools
from math import comb

import numpy as np

SEED = 20261019
REPS = 2000
MDE_REPS = 1000
_SIGNS = {}


def signs(n):
    if n not in _SIGNS:
        _SIGNS[n] = np.array(list(itertools.product((1.0, -1.0), repeat=n)), dtype=np.float32)
    return _SIGNS[n]


def signflip_p(d):
    n = len(d)
    stat = np.abs(signs(n) @ d.astype(np.float32)) / n
    return float((stat >= abs(float(d.mean())) - 1e-6).mean())


def power(n, delta, sd, alpha, reps, rng):
    return float(np.mean([signflip_p(rng.normal(delta, sd, n)) < alpha for _ in range(reps)]))


def mde(n, sd, alpha, grid, rng):
    return next((g for g in grid if power(n, g, sd, alpha, MDE_REPS, rng) >= 0.80), float("nan"))


def informative_null(n, sd, sesoi, alpha, reps, rng):
    hits = 0
    for _ in range(reps):
        d = rng.normal(0.0, sd, n)
        hits += signflip_p(d + sesoi) < alpha and signflip_p(d - sesoi) < alpha
    return hits / reps


def table(title, ns, sds, deltas, alpha, grid, sesoi, rng):
    print(f"\n{title}: two-sided exact sign-flip, alpha {alpha}, {REPS} reps per cell")
    print("n | sd_diff | " + " | ".join(f"{d} kt" for d in deltas) + f" | 80% MDE | informative null at 0 (SESOI {sesoi})")
    for n in ns:
        for sd in sds:
            cells = [power(n, d, sd, alpha, REPS, rng) for d in deltas]
            m = mde(n, sd, alpha, grid, rng)
            nul = informative_null(n, sd, sesoi, alpha, REPS, rng)
            print(f"{n} | {sd:.2f} | " + " | ".join(f"{c:.2f}" for c in cells) + f" | {m:.2f} | {nul:.2f}")


# Native rule, flagged epochs > 0.85 x E_k: (arm, budget, flagged epochs, E_k)
FLAG_CELLS = (("control", "1x", 4, 25), ("control", "2x", 8, 50), ("control", "4x", 15, 99),
              ("atlantic", "1x", 2, 13), ("atlantic", "2x", 4, 26), ("atlantic", "4x", 8, 52))


def tail(n, k, p):
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


def flag_reference():
    print("\nUnder-training flag on a flat validation curve (pick uniform over the E_k epochs of the budget):")
    for arm, budget, flagged, e in FLAG_CELLS:
        p = flagged / e
        print(f"  {arm} {budget}: P(flag) = {flagged}/{e} = {p:.4f}; P(>= 10/20) = {tail(20, 10, p):.5f}; "
              f"P(>= 5/10) = {tail(10, 5, p):.4f}")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    table("P1/P2, overall MAE, budget 4x - 1x within one configuration", (10, 20), (0.54, 0.85, 1.12),
          (0.5, 0.7, 0.9, 1.2), 0.025, np.arange(0.2, 2.51, 0.05), 0.9, rng)
    table("S1, overall MAE, atlantic - control at 4x (secondary)", (10, 20), (0.85, 1.12, 1.40),
          (0.5, 0.9, 1.2), 0.05, np.arange(0.2, 2.51, 0.05), 0.9, rng)
    table("S2, >96 kt bias, atlantic - control at 4x (secondary)", (10, 20), (2.84, 4.79, 6.00),
          (2.0, 3.0, 4.0, 5.0), 0.05, np.arange(1.0, 9.01, 0.1), 3.0, rng)
    flag_reference()
