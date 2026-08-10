"""Quantitative posterior-predictive checks + calibration for a fitted model.

(1) Discrepancy PPCs with Bayesian p-values, p = P(T_rep >= T_obs), for test
    statistics that target the known misfit:
      - full-consumption rate  P(eaten == n0)
      - zero-consumption rate  P(eaten == 0)
      - overdispersion         sum over n0 of Var(eaten | n0)
    plus the per-n0 full-consumption rate (observed vs replicated 95% band).
(2) LOO-PIT calibration (ArviZ): uniform => calibrated; U / inverted-U => over /
    under-dispersion.

Simulated replicate datasets and the pointwise log-likelihood are computed from
the pooled posterior in one pass. Usage:  python ppc_checks.py --model 1  (or 2)
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import arviz as az
import run_dream as R
import mechanistic_model as mm

NE, N0 = R.NE, R.N0
N0_LEVELS = sorted(set(int(n) for n in N0))


def _loglik_and_sim(x, X_a, X_r, seed):
    """One posterior draw -> (pointwise loglik vector, simulated-count vector)."""
    n_a = X_a.shape[1]
    ba, br = x[:n_a], x[n_a:]
    rng = np.random.RandomState(seed)
    ll = np.empty(len(NE)); sim = np.empty(len(NE), dtype=int)
    for k in range(len(NE)):
        a = np.exp(ba @ X_a[k]); r = np.exp(br @ X_r[k])
        p = np.clip(mm.solve_system_simple_h_simplex(int(N0[k]), a, r), 1e-300, None)
        p = p / p.sum()
        ll[k] = np.log(p[int(NE[k])])
        sim[k] = rng.choice(len(p), p=p)
    return ll, sim


# --- test statistics on a length-668 count vector -------------------------
def T_full(d):  return np.mean(d == N0)
def T_zero(d):  return np.mean(d == 0)
def T_over(d):  return float(np.sum([d[N0 == n].var() for n in N0_LEVELS]))


def run(model_id, n_per_chain=40):
    arr, _ = R.load_posterior(model_id)
    C, N, P = arr.shape
    idx = np.unique(np.linspace(0, N - 1, n_per_chain).astype(int))
    sub = arr[:, idx, :]
    flat = sub.reshape(-1, P)
    X_a, X_r, _ = R.design(model_id)
    res = Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(_loglik_and_sim)(flat[i], X_a, X_r, i) for i in range(len(flat)))
    ll = np.array([r[0] for r in res])         # (n_draws, obs)
    pp = np.array([r[1] for r in res])         # (n_draws, obs)

    # (1) discrepancy p-values
    print(f"\n===== MODEL {model_id}: discrepancy PPCs  (p = P(T_rep >= T_obs)) =====")
    stats = [("full-consumption rate P(eaten=n0)", T_full),
             ("zero-consumption rate P(eaten=0)", T_zero),
             ("overdispersion  sum_n0 Var(eaten|n0)", T_over)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for ax, (name, T) in zip(axes, stats):
        rep = np.array([T(pp[i]) for i in range(len(pp))])
        obs = T(NE)
        p = float(np.mean(rep >= obs))
        p2 = 2 * min(p, 1 - p)                 # two-sided
        print(f"  {name:40s} obs={obs:8.3f}  p={p:.3f}  (2-sided {p2:.3f})")
        ax.hist(rep, bins=30, color='#D2691E', alpha=0.8)
        ax.axvline(obs, color='k', lw=2, label=f'observed (p={p:.3f})')
        ax.set_title(name, fontsize=10); ax.legend(fontsize=9)
        ax.set_xlabel('test statistic'); ax.set_ylabel('replicate count')

    # per-n0 full-consumption: observed vs replicated band
    ax = axes[3]
    obs_by = [np.mean(NE[N0 == n] == n) for n in N0_LEVELS]
    rep_by = np.array([[np.mean(pp[i][N0 == n] == n) for n in N0_LEVELS]
                       for i in range(len(pp))])
    lo, hi = np.percentile(rep_by, [2.5, 97.5], axis=0)
    ax.fill_between(N0_LEVELS, lo, hi, alpha=0.3, color='#D2691E',
                    label='replicated 95%')
    ax.plot(N0_LEVELS, rep_by.mean(0), color='#D2691E', label='replicated mean')
    ax.plot(N0_LEVELS, obs_by, 'ko-', label='observed')
    ax.set_title('full-consumption rate by $N_0$', fontsize=10)
    ax.set_xlabel('initial prey, $N_0$'); ax.set_ylabel('$P(\mathrm{eaten}=N_0)$'); ax.legend(fontsize=8)
    fig.suptitle(f'Model {model_id}: discrepancy PPCs')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'outputs/model_{model_id}_ppc_discrepancy.png', dpi=115)
    plt.close(fig)

    # (2) LOO-PIT
    ll3 = ll.reshape(C, len(idx), -1)
    pp3 = pp.reshape(C, len(idx), -1)
    idata = az.from_dict(posterior={'theta': sub},
                         log_likelihood={'obs': ll3},
                         posterior_predictive={'obs': pp3},
                         observed_data={'obs': NE})
    figp, axp = plt.subplots(figsize=(6, 4))
    az.plot_loo_pit(idata, y='obs', y_hat='obs', ax=axp)
    axp.set_title(f'Model {model_id}: LOO-PIT (uniform = calibrated)')
    figp.tight_layout()
    figp.savefig(f'outputs/model_{model_id}_loo_pit.png', dpi=115)
    plt.close(figp)
    print(f"  saved outputs/model_{model_id}_ppc_discrepancy.png and _loo_pit.png",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="1,2")
    a = ap.parse_args()
    for mid in [int(s) for s in a.models.split(",")]:
        run(mid)
