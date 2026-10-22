"""Power of the 2026 clean test's primary (protocol_draft.md §6). CPU only; reads nothing.

The primary is the mean over storms of d_s = MAE_v2(s) - MAE_frozen(s), each storm one replicate, tested with a
two-sided exact sign-flip test over storms at alpha 0.05. Power is simulated with d_s ~ Normal(delta, sd) for the
planning grid below. The H9-derived planning effect (0.48 kt, F-31's 20-member control ensemble, 4x vs 1x) is labelled
as such; the SESOI candidates are not taken from F-31.

Run: .venv\\Scripts\\python.exe -I -B reports/v2_survey/2026_clean_test/power.py
"""
import itertools

import numpy as np

ALPHA = 0.05
SEED = 20261021
REPS = 2000
NS = range(6, 17, 2)
SDS = (0.5, 1.0, 1.5, 2.0)
DELTAS = (("SESOI 1.0 kt", 1.0), ("SESOI 0.5 kt", 0.5), ("H9-derived planning 0.48 kt", 0.48))


def signs(n):
    return np.array(list(itertools.product((1.0, -1.0), repeat=n)))


def reject(d, S):
    """Two-sided exact sign-flip p < ALPHA for each row of d (reps x n), on |mean|."""
    obs = np.abs(d.mean(axis=1))
    null = np.abs(d @ S.T) / d.shape[1]  # reps x 2^n
    p = (null >= obs[:, None] - 1e-12).mean(axis=1)
    return p < ALPHA


def main():
    print(f"two-sided exact sign-flip over storms, alpha {ALPHA}; reps {REPS}; seed {SEED}")
    print("p floor 2^(1-n): n = 5 gives 0.0625, so at least 6 storms are needed for p < 0.05 to be possible")
    for label, delta in DELTAS:
        print(f"\n{label}: power by n storms (rows) and between-storm sd of d_s in kt (columns)")
        print("n    " + "  ".join(f"sd {s:>3}" for s in SDS))
        for n in NS:
            S = signs(n)
            row = []
            for sd in SDS:
                rng = np.random.default_rng([SEED, n, int(sd * 10), int(delta * 100)])
                d = rng.normal(-delta, sd, size=(REPS, n))
                hits = np.concatenate([reject(d[i:i + 200], S) for i in range(0, REPS, 200)])
                row.append(f"{hits.mean():6.2f}")
            print(f"{n:<4} " + "  ".join(row))


if __name__ == "__main__":
    main()
