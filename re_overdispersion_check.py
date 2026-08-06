"""Diagnostic: can a per-observation random effect on log r explain the
overdispersion? (Separate analysis; reads the fitted Model 2 posterior, writes
nothing that other scripts use.)

Takes the fitted Model 2 posterior, adds u ~ N(0, sigma) to log r for each
observation, simulates prey-eaten counts, and reports the three discrepancy
statistics as a function of sigma against the observed values. If some moderate
sigma recovers the observed full-consumption / zero / overdispersion, the OLRE is
the right fix.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import run_dream as R
import mechanistic_model as mm

NE, N0 = R.NE, R.N0
N0_LEVELS = sorted(set(int(n) for n in N0))
SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def _sim(logr_base, loga_base, sigma, target, seed):
    """Simulate an eaten count per fish. target in {'r','a','both'} chooses which
    rate carries the OLRE u ~ N(0, sigma)."""
    rng = np.random.RandomState(seed)
    ur = rng.normal(0, sigma, len(NE)) if target in ('r', 'both') else 0.0
    ua = rng.normal(0, sigma, len(NE)) if target in ('a', 'both') else 0.0
    r = np.exp(logr_base + ur)
    a = np.exp(loga_base + ua)
    out = np.empty(len(NE), dtype=int)
    for k in range(len(NE)):
        p = np.clip(mm.solve_system_simple_h_simplex(int(N0[k]), a[k], r[k]), 0, None)
        out[k] = rng.choice(len(p), p=p / p.sum())
    return out


def stats(d):
    full = np.mean(d == N0)
    zero = np.mean(d == 0)
    over = float(np.sum([d[N0 == n].var() for n in N0_LEVELS]))
    return full, zero, over


if __name__ == "__main__":
    arr, _ = R.load_posterior(2)
    X_a, X_r, _ = R.design(2)
    flat = arr.reshape(-1, arr.shape[-1])
    rng = np.random.RandomState(0)
    draws = flat[rng.choice(len(flat), 120, replace=False)]
    na = X_a.shape[1]
    logr = draws[:, na:] @ X_r.T                       # (n_draws, n_fish)
    loga = np.repeat(draws[:, [0]], len(NE), axis=1)   # a0 const, broadcast per fish

    obs = stats(NE)
    print(f"OBSERVED:  full={obs[0]:.3f}  zero={obs[1]:.3f}  overdisp={obs[2]:.1f}\n")
    for target in ('r', 'a', 'both'):
        print(f"--- OLRE on {target} ---")
        print(f"{'sigma':>6} {'full(0.29)':>12} {'zero(0.021)':>12} {'overdisp(281)':>14}")
        for sig in SIGMAS:
            sims = Parallel(n_jobs=-1, max_nbytes=None)(
                delayed(_sim)(logr[i], loga[i], sig, target, 1000 + i)
                for i in range(len(draws)))
            s = np.array([stats(d) for d in sims]).mean(0)
            print(f"{sig:6.2f} {s[0]:12.3f} {s[1]:12.3f} {s[2]:14.1f}", flush=True)
        print()
