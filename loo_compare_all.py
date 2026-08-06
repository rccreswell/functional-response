"""PSIS-LOO comparison of the base models (1, 2) and the voracious-mixture models
(1a, 2a) on the same 668 observations, via ArviZ.

Each model contributes a pointwise log-likelihood matrix (chain, draw, fish);
az.compare ranks them by expected log predictive density (elpd_loo). The base
pointwise loglik is the CME probability of the observed count; the mixture one is
the marginal mixture probability. Reads the consolidated posteriors
(outputs/model_<id>[a]_posterior.csv), so run after --consolidate.

  python loo_compare_all.py
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import arviz as az
from joblib import Parallel, delayed
import run_dream as R
import run_dream_mixture as M
from mixture_model import mixture_loglik


def _mix_pointwise(x, model_id):
    X_a, X_r, _ = R.design(model_id)
    na = X_a.shape[1]; nbase = na + X_r.shape[1]
    ba, br = x[:na], x[na:nbase]
    g0, gw = x[nbase], x[nbase + 1]
    _, lls = mixture_loglik(R.NE, R.N0, ba, br, g0, gw, X_a, X_r, R.WZ, n_jobs=1)
    return np.array(lls)


def _idata(model_id, is_mixture, n_per_chain=40):
    arr, _ = (M.load_posterior_mix(model_id) if is_mixture
              else R.load_posterior(model_id))
    C, N, P = arr.shape
    idx = np.unique(np.linspace(0, N - 1, n_per_chain).astype(int))
    sub = arr[:, idx, :]
    flat = sub.reshape(-1, P)
    if is_mixture:
        rows = Parallel(n_jobs=-1, max_nbytes=None)(
            delayed(_mix_pointwise)(x, model_id) for x in flat)
    else:
        X_a, X_r, _ = R.design(model_id)
        rows = Parallel(n_jobs=-1, max_nbytes=None)(
            delayed(R._pointwise_loglik)(x, X_a, X_r) for x in flat)
    ll = np.array(rows).reshape(C, len(idx), -1)
    return az.from_dict(posterior={'theta': sub}, log_likelihood={'obs': ll})


if __name__ == "__main__":
    idatas = {}
    for mid in (1, 2):
        idatas[f'model_{mid}'] = _idata(mid, is_mixture=False)
    for mid in (1, 2):
        idatas[f'model_{mid}a'] = _idata(mid, is_mixture=True)

    def pareto_bad(loo):
        for acc in (lambda: loo["pareto_k"], lambda: loo.pareto_k):
            try:
                return int((np.asarray(acc()) > 0.7).sum())
            except Exception:
                continue
        return np.nan

    print("=== per-model PSIS-LOO ===")
    bad = {}
    for k, d in idatas.items():
        loo = az.loo(d, pointwise=True)
        bad[k] = pareto_bad(loo)
        print(f"\n----- {k} -----"); print(loo)

    cmp = az.compare(idatas, ic='loo').rename(columns={'loo': 'elpd_loo'})
    cmp['bad_pareto_k(>0.7)'] = [bad.get(ix, np.nan) for ix in cmp.index]
    cmp.to_csv('outputs/loo_comparison.csv')
    print("\n=== az.compare (PSIS-LOO) ===")
    print(cmp.to_string())
    print("\nsaved outputs/loo_comparison.csv")
