"""Compute posterior-predictive discrepancy statistics and two-sided Bayesian
p-values for ALL four models (two-state 1 & 2, mixtures 1a & 2a), on a common set
of statistics, and write outputs/all_models_bayes_pvalues.csv.

Same statistics as ppc_mixture.py: full-consumption P(eaten=n0), zero-consumption
P(eaten=0), dispersion sum_n0 Var(eaten|n0), mean eaten, and full-consumption
within light/moderate/heavy weight terciles. Deterministic (fixed posterior
thinning + fixed simulation seeds). Run under __main__ (joblib/macOS spawn).

Usage: python make_ppc_table_all.py
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import run_dream as R
import run_dream_mixture as M
import ppc_checks as PC

NE, N0 = R.NE, R.N0
N0_LEVELS = sorted(set(int(n) for n in N0))
TERT = np.asarray(pd.qcut(R.DF['weight'].to_numpy(), 3, labels=['light', 'moderate', 'heavy']))
NDRAW = 250                                     # posterior draws per model (deterministic thinning)


def _thin(arr):
    flat = arr.reshape(-1, arr.shape[-1])
    idx = np.linspace(0, len(flat) - 1, NDRAW).astype(int)
    return flat[idx]


def base_sims(model_id):                        # two-state CME model
    draws = _thin(R.load_posterior(model_id)[0])
    X_a, X_r, _ = R.design(model_id)
    res = Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(PC._loglik_and_sim)(draws[i], X_a, X_r, 7000 + i) for i in range(len(draws)))
    return np.array([r[1] for r in res])        # (NDRAW, n_obs) simulated counts


def mix_sims(model_id):                         # voracious-mixture model
    draws = _thin(M.load_posterior_mix(model_id)[0])
    return M.simulate(model_id, draws)


def full(d):  return np.mean(d == N0)
def zero(d):  return np.mean(d == 0)
def over(d):  return float(np.sum([d[N0 == n].var() for n in N0_LEVELS]))
def meanE(d): return float(d.mean())

STATS = {'full-consumption P(eaten=n0)': full, 'zero-consumption P(eaten=0)': zero,
         'overdispersion sum Var(eaten|n0)': over, 'mean eaten': meanE}
for _t in ['light', 'moderate', 'heavy']:
    STATS[f'full-consumption ({_t} weight)'] = (lambda d, t=_t: np.mean(d[TERT == t] == N0[TERT == t]))

MODELS = [('model_1', lambda: base_sims(1)), ('model_2', lambda: base_sims(2)),
          ('model_1a', lambda: mix_sims(1)), ('model_2a', lambda: mix_sims(2))]


def main():
    sims = {}
    for tag, fn in MODELS:
        sims[tag] = fn()
        print(f"[{tag}] simulated {sims[tag].shape[0]} draws x {sims[tag].shape[1]} obs", flush=True)
    rows, means = [], []
    for name, T in STATS.items():
        obs = float(T(NE))
        row = {'statistic': name, 'observed': round(obs, 4)}
        mrow = {'statistic': name, 'observed': round(obs, 4)}
        for tag, _ in MODELS:
            rep = np.array([T(s) for s in sims[tag]])
            p = float(np.mean(rep >= obs))
            row[f'{tag}_p'] = round(2 * min(p, 1 - p), 3)      # two-sided
            mrow[f'{tag}_mean'] = round(float(rep.mean()), 4)  # model's expected value of the statistic
        rows.append(row); means.append(mrow)
    pd.DataFrame(rows).to_csv('outputs/all_models_bayes_pvalues.csv', index=False)
    pd.DataFrame(means).to_csv('outputs/all_models_stat_means.csv', index=False)
    print("saved outputs/all_models_bayes_pvalues.csv and outputs/all_models_stat_means.csv")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
