"""N5-D: power of the two pre-registered primaries at n = 10 seeds per arm (deterministic Monte Carlo).

Primaries (Bonferroni, alpha 0.025 each): H9 pooled overall MAE and H9 >96 kt bias, each a two-sided exact paired
sign-flip test on the 10 per-seed differences (atlantic - control). Also: the rate of an "informative null"
(97.5% CI by test inversion inside +-SESOI) when the true difference is 0. The unpaired exact permutation test of
the 10-14 designs is shown at alpha 0.05 for comparison with the first draft.

    .venv\\Scripts\\python.exe -I reports/v2_survey/n5/power.py

sd inputs (observed, results/storms/*_h9.csv and primary_tests.csv):
  overall MAE: control2 sd 0.633, expanded 0.286, sd of per-seed differences 0.694, correlation +0.001;
  >96 bias:    control2 sd 2.01,  expanded 1.43,  sd of per-seed differences 2.44,  correlation +0.025.
"""
import itertools

import numpy as np

N = 10
ALPHA = 0.025
SIGNS = np.array(list(itertools.product((1.0, -1.0), repeat=N)))  # 1024 x 10


def signflip_p(d):
    return float((np.abs(SIGNS @ d / N) >= abs(d.mean()) - 1e-12).mean())


def ci_inside(d, sesoi, n_grid=801):
    grid = np.linspace(d.min(), d.max(), n_grid)
    keep = [g for g in grid if signflip_p(d - g) >= ALPHA]
    return bool(keep) and -sesoi < min(keep) and max(keep) < sesoi


def power(delta, sd_d, reps, rng):
    return float(np.mean([signflip_p(rng.normal(delta, sd_d, N)) < ALPHA for _ in range(reps)]))


def mde(sd_d, grid, rng):
    return next(d for d in grid if power(d, sd_d, 1000, rng) >= 0.80)


def null_rate(sd_d, sesoi, reps, rng):
    return float(np.mean([ci_inside(rng.normal(0, sd_d, N), sesoi) for _ in range(reps)]))


def table(name, sds, deltas, grid, sesois, rng):
    print(f"\n{name}: paired sign-flip, two-sided, alpha {ALPHA} (Bonferroni over 2), n = 10, 2000 reps per cell")
    print("sd_diff | " + " | ".join(f"delta {d}" for d in deltas) + " | MDE80 | " +
          " | ".join(f"informative null at 0, SESOI {s}" for s in sesois))
    for sd_d in sds:
        cells = [power(d, sd_d, 2000, rng) for d in deltas]
        nulls = [null_rate(sd_d, s, 500, rng) for s in sesois]
        print(f"{sd_d:.2f} | " + " | ".join(f"{c:.2f}" for c in cells) + f" | {mde(sd_d, grid, rng):.2f} | " +
              " | ".join(f"{v:.2f}" for v in nulls))


if __name__ == "__main__":
    rng = np.random.default_rng(20261018)
    table("H9 overall MAE", (0.40, 0.69, 0.85), (0.5, 0.6, 0.9, 1.2), np.arange(0.2, 2.01, 0.05), (0.9,), rng)
    table("H9 >96 kt bias", (2.03, 2.44, 2.84), (2.0, 2.5, 3.0, 3.5, 4.0), np.arange(1.0, 6.01, 0.1), (3.0, 2.5), rng)
