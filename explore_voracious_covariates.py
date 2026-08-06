"""Does the voracious (instant-capture) probability depend on covariates beyond
weight? Fit the mixture with logit(pi) = gamma . X_pi for two designs -- weight
only vs weight + all condition/week covariates -- and compare the maximised
log-likelihood and the coefficients. (Separate exploratory script; writes nothing.)
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import pints
from scipy.special import expit
import run_dream as R
from run_dream_mixture import base_mean_var
from mixture_model import _chunk_mixture_ll
from joblib import Parallel, delayed

NE, N0, WZ = R.NE, R.N0, R.WZ
DUMMIES = ["f.acc_condH", "f.acc_condM", "f.acc_condT", "f.exp_condH",
           "f.exp_condM", "f.exp_condT", "f.week6", "f.acc_condH:f.week6",
           "f.acc_condM:f.week6", "f.acc_condT:f.week6"]
DUM = R.DF[DUMMIES].to_numpy()
ONES = np.ones((len(NE), 1))


def pi_design(kind):
    if kind == 'weight':
        return np.column_stack([ONES, WZ]), ['1', 'weight_z']
    return (np.column_stack([ONES, WZ, WZ ** 2, DUM]),
            ['1', 'weight_z', 'weight_z2'] + DUMMIES)


class GenMixLL(pints.LogPDF):
    def __init__(self, model_id, X_pi):
        self.X_a, self.X_r, _ = R.design(model_id)
        self.na = self.X_a.shape[1]; self.nr = self.X_r.shape[1]
        self.X_pi = X_pi; self.npi = X_pi.shape[1]

    def __call__(self, p):
        ba = p[:self.na]; br = p[self.na:self.na + self.nr]; g = p[self.na + self.nr:]
        a = np.exp(self.X_a @ ba); r = np.exp(self.X_r @ br)
        pi = expit(self.X_pi @ g)
        n = len(NE); nc = 10; b = np.linspace(0, n, nc + 1).astype(int)
        res = Parallel(n_jobs=-1, max_nbytes=None)(
            delayed(_chunk_mixture_ll)(NE[i:j], N0[i:j], a[i:j], r[i:j], pi[i:j])
            for i, j in zip(b[:-1], b[1:]))
        return float(sum(v for c in res for v in c))

    def n_parameters(self):
        return self.na + self.nr + self.npi


def build_prior(na, nr, npi):
    m, v = base_mean_var(na, nr)
    gm = np.zeros(npi); gm[0] = 1.0            # intercept ~ pi high
    gv = np.full(npi, 1.5 ** 2)
    return pints.MultivariateGaussianLogPrior(np.concatenate([m, gm]),
                                              np.diag(np.concatenate([v, gv])))


def fit(model_id, kind, opt_iters):
    X_pi, names = pi_design(kind)
    ll = GenMixLL(model_id, X_pi)
    prior = build_prior(ll.na, ll.nr, ll.npi)
    post = pints.LogPosterior(ll, prior)
    np.random.seed(0)
    x0 = next(c for c in (prior.sample()[0] for _ in range(300)) if np.isfinite(post(c)))
    opt = pints.OptimisationController(post, x0, method=pints.CMAES)
    opt.set_max_iterations(opt_iters); opt.set_log_to_screen(False)
    xhat, _ = opt.run()
    return ll(xhat), xhat[ll.na + ll.nr:], names


if __name__ == "__main__":
    for mid in (1, 2):
        llw, gw, _ = fit(mid, 'weight', 120)
        llf, gf, names = fit(mid, 'full', 250)
        d = 2 * (llf - llw); ddf = len(gf) - len(gw)
        print(f"\n===== MODEL {mid}a: does voracious prob depend on covariates? =====")
        print(f"  max loglik: weight-only={llw:.1f}   full={llf:.1f}")
        print(f"  2*dLL={d:.1f} over {ddf} extra df  (chi2_0.05({ddf})~{ddf+2*np.sqrt(2*ddf):.0f})")
        order = np.argsort(-np.abs(gf))
        print("  voracious-logit coefficients (|largest| first):")
        for k in order:
            flag = '  <-- notable' if abs(gf[k]) > 0.8 and names[k] != '1' else ''
            print(f"    {names[k]:22s} {gf[k]:+.2f}{flag}")
