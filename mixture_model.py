"""Two-component 'voracious' mixture likelihood for the functional-response model.

Each fish is, with probability pi_i, a *voracious* forager; otherwise an ordinary
one. Ordinary fish follow the usual chemical-master-equation (CME) dynamics with
rates a_i, r_i. Voracious fish capture prey essentially instantly (the a -> infinity
limit of the same dynamics), so they are limited only by digestion and their eaten
count is  min(n0, 1 + Poisson(r_i))  -- a closed form, no matrix exponential. The
latent type is marginalised out:

    L_i = (1 - pi_i) * P_CME(y_i | a_i, r_i)  +  pi_i * P_voracious(y_i | r_i)

with logit(pi_i) = gamma0 + gamma_w * weight_z_i (heavier -> more likely voracious).
This adds two parameters (gamma0, gamma_w) and costs ONE CME solve per fish (the
voracious term is a cheap Poisson evaluation). The instant-capture form was
validated against the full CME at large attack rate (they agree as a -> infinity).
"""
import math
import numpy as np
from mechanistic_model import solve_system_simple_h_simplex

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False


def _pois_pmf(k, r):
    """Poisson(k; r) via math (much faster than scipy.stats for scalars)."""
    if r <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(r) - r - math.lgamma(k + 1))


def voracious_pmf(n0, r):
    """Full pmf of the instant-capture (a->infinity) voracious component:
    eaten = min(n0, 1 + Poisson(r))."""
    p = np.zeros(n0 + 1)
    for j in range(1, n0):                      # eaten = 1 .. n0-1  <->  N = j-1
        p[j] = _pois_pmf(j - 1, r)
    if n0 >= 1:
        p[n0] = max(0.0, 1.0 - sum(_pois_pmf(k, r) for k in range(n0 - 1)))
    return p


def _voracious_prob(yi, n0, r):
    """P(voracious fish eats exactly yi) -- scalar."""
    if yi <= 0:
        return 0.0
    if yi < n0:                                 # eaten = yi  <->  N = yi-1
        return _pois_pmf(yi - 1, r)
    # yi == n0:  P(N >= n0-1) = 1 - sum_{k=0}^{n0-2} pmf(k)
    return max(0.0, 1.0 - sum(_pois_pmf(k, r) for k in range(n0 - 1)))


def _chunk_mixture_ll(eaten, n0, a, r, pi):
    out = []
    for i in range(len(eaten)):
        yi = int(eaten[i])
        p_ord = solve_system_simple_h_simplex(int(n0[i]), a[i], r[i])
        po = max(p_ord[yi], 0.0)
        pv = max(_voracious_prob(yi, int(n0[i]), r[i]), 0.0)
        L = (1.0 - pi[i]) * po + pi[i] * pv
        out.append(np.log(max(L, 1e-300)))
    return out


def mixture_loglik(eaten, n0, beta_a, beta_r, gamma0, gamma_w,
                   X_a, X_r, weight_z, n_jobs=1):
    """Total marginal log-likelihood and per-fish contributions for the
    instant-capture voracious mixture."""
    eaten = np.asarray(eaten)
    n0 = np.asarray(n0)
    a = np.exp(X_a @ beta_a)
    r = np.exp(X_r @ beta_r)
    pi = 1.0 / (1.0 + np.exp(-(gamma0 + gamma_w * weight_z)))
    n = len(eaten)

    if not _HAVE_JOBLIB or n_jobs == 1:
        lls = _chunk_mixture_ll(eaten, n0, a, r, pi)
    else:
        import os
        nw = os.cpu_count() or 1 if n_jobs < 0 else n_jobs
        nc = max(1, min(nw, n))
        bnd = np.linspace(0, n, nc + 1).astype(int)
        res = Parallel(n_jobs=n_jobs, max_nbytes=None)(
            delayed(_chunk_mixture_ll)(eaten[b0:b1], n0[b0:b1], a[b0:b1],
                                       r[b0:b1], pi[b0:b1])
            for b0, b1 in zip(bnd[:-1], bnd[1:]))
        lls = [v for c in res for v in c]

    total = 0.0
    for v in lls:
        total += v
    return total, lls
