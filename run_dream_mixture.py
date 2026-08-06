"""DREAM fit for the 'voracious mixture' models 1a / 2a (separate from the base
models 1 / 2; writes outputs/model_<id>a_*). Adds 3 parameters to the base model:
gamma0, gamma_w (logit voracious probability vs weight_z) and log_delta
(voracious speed-up delta = exp(log_delta)).

  python run_dream_mixture.py --map-check --models 1,2   # fast: CMA-ES MAP + PPC
  python run_dream_mixture.py --models 1,2 [--n-iter N] [--resume]
  python run_dream_mixture.py --consolidate --models 1,2
  python run_dream_mixture.py --discrepancy --models 1,2
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import pints
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import run_dream as R
import mechanistic_model as mm
from mixture_model import mixture_loglik

NE, N0, WZ = R.NE, R.N0, R.WZ
N0_LEVELS = sorted(set(int(n) for n in N0))
MIX_LABELS = ['gamma0', 'gamma_w']   # voracious = instant-capture (a->inf)


def design_mix(model_id):
    X_a, X_r, base_labels = R.design(model_id)
    return X_a, X_r, base_labels + MIX_LABELS


def base_mean_var(n_a, n_r, log_a0=np.log(4.0), sd_a=1.0, log_r0=np.log(20.0),
                  sd_r=0.5, slope=0.75):
    m = np.zeros(n_a + n_r); v = np.zeros(n_a + n_r)
    m[0] = log_a0; v[0] = sd_a ** 2
    if n_a - 1 > 0: v[1:n_a] = (slope / np.sqrt(n_a - 1)) ** 2
    m[n_a] = log_r0; v[n_a] = sd_r ** 2
    if n_r - 1 > 0: v[n_a + 1:] = (slope / np.sqrt(n_r - 1)) ** 2
    return m, v


def build_prior_mix(model_id):
    X_a, X_r, _ = design_mix(model_id)
    m, v = base_mean_var(X_a.shape[1], X_r.shape[1])
    # gamma0 ~ N(-1, 1.5)   gamma_w ~ N(0, 1.5)
    m = np.concatenate([m, [-1.0, 0.0]])
    v = np.concatenate([v, [1.5 ** 2, 1.5 ** 2]])
    return pints.MultivariateGaussianLogPrior(m, np.diag(v))


class MixLikelihood(pints.LogPDF):
    def __init__(self, model_id, n_jobs=1):
        self.X_a, self.X_r, _ = R.design(model_id)
        self.na = self.X_a.shape[1]; self.nr = self.X_r.shape[1]
        self.nbase = self.na + self.nr
        self.n_jobs = n_jobs

    def __call__(self, x):
        ba = x[:self.na]; br = x[self.na:self.nbase]
        g0, gw = x[self.nbase], x[self.nbase + 1]
        ll, _ = mixture_loglik(NE, N0, ba, br, g0, gw,
                               self.X_a, self.X_r, WZ, n_jobs=self.n_jobs)
        return ll

    def n_parameters(self):
        return self.nbase + 2


# ---- simulation from the mixture (for PPC / discrepancy) ------------------
def _sim_mix_one(x, X_a, X_r, na, nbase, seed):
    ba = x[:na]; br = x[na:nbase]
    g0, gw = x[nbase], x[nbase + 1]
    a = np.exp(X_a @ ba); r = np.exp(X_r @ br)
    pi = 1.0 / (1.0 + np.exp(-(g0 + gw * WZ)))
    rng = np.random.RandomState(seed)
    vor = rng.random(len(NE)) < pi
    out = np.empty(len(NE), dtype=int)
    for k in range(len(NE)):
        n0k = int(N0[k])
        if vor[k]:                                   # instant capture: min(n0, 1+Poisson(r))
            out[k] = min(n0k, 1 + rng.poisson(r[k]))
        else:
            p = np.clip(mm.solve_system_simple_h_simplex(n0k, a[k], r[k]), 0, None)
            out[k] = rng.choice(len(p), p=p / p.sum())
    return out


def simulate(model_id, draws):
    from joblib import Parallel, delayed
    X_a, X_r, _ = R.design(model_id)
    na = X_a.shape[1]; nbase = na + X_r.shape[1]
    return np.array(Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(_sim_mix_one)(draws[i], X_a, X_r, na, nbase, 7000 + i)
        for i in range(len(draws))))


def discrepancy(model_id, sims):
    tert = np.asarray(pd.qcut(R.DF['weight'].to_numpy(), 3,
                              labels=['light', 'moderate', 'heavy']))

    def full(d): return np.mean(d == N0)
    def zero(d): return np.mean(d == 0)
    def over(d): return float(np.sum([d[N0 == n].var() for n in N0_LEVELS]))
    print(f"\n[model {model_id}a] discrepancy  (p = P(T_rep >= T_obs))")
    for name, T, o in [("full-consumption", full, full(NE)),
                       ("zero-consumption", zero, zero(NE)),
                       ("overdispersion", over, over(NE))]:
        rep = np.array([T(s) for s in sims]); p = float(np.mean(rep >= o))
        print(f"  {name:18s} obs={o:8.3f}  model={rep.mean():8.3f}  p={p:.3f}")
    print("  full-consumption by weight tertile (obs vs model):")
    for t in ['light', 'moderate', 'heavy']:
        m = tert == t
        print(f"    {t:9s} obs={np.mean(NE[m]==N0[m]):.3f}  "
              f"model={np.mean(sims[:, m]==N0[None, m]):.3f}")


# ---- checkpointed DREAM fit (mirrors run_dream.run_model) -----------------
def load_posterior_mix(model_id):
    tag = f'model_{model_id}a'
    df = pd.read_csv(f'outputs/{tag}_posterior.csv')
    _, _, labels = design_mix(model_id)
    C = df['chain'].nunique(); Nn = df['iter'].nunique()
    return df[labels].to_numpy().reshape(C, Nn, len(labels)), labels


def write_csv_mix(tag, labels, chains):
    C, Nn, P = chains.shape
    df = pd.DataFrame(chains.reshape(C * Nn, P), columns=labels)
    df.insert(0, 'iter', np.tile(np.arange(Nn), C))
    df.insert(0, 'chain', np.repeat(np.arange(C), Nn))
    df.to_csv(f'outputs/{tag}_posterior.csv', index=False)
    print(f"[{tag}] wrote outputs/{tag}_posterior.csv ({C*Nn} draws)", flush=True)


def run_model_mix(model_id, n_iter, n_chains, resume, opt_iters, init_spread, seed):
    tag = f'model_{model_id}a'
    _, _, labels = design_mix(model_id)
    prior = build_prior_mix(model_id)
    opt_post = pints.LogPosterior(MixLikelihood(model_id, n_jobs=-1), prior)
    post = pints.LogPosterior(MixLikelihood(model_id, n_jobs=1), prior)
    np.random.seed(seed)
    prev = R.latest_chunk(tag)
    if resume and prev >= 0:
        inits = list(R.load_chunk(tag, prev)[:, -1, :]); n_chains = len(inits)
        source = f'chunk {prev}'
    else:
        x0 = next(c for c in (prior.sample()[0] for _ in range(300))
                  if np.isfinite(opt_post(c)))
        opt = pints.OptimisationController(opt_post, x0, method=pints.CMAES)
        opt.set_max_iterations(opt_iters); opt.set_log_to_screen(False)
        xhat, _ = opt.run()
        np_ = post.n_parameters()
        inits = []
        while len(inits) < n_chains:
            c = xhat + np.random.normal(0, init_spread, np_)
            if np.isfinite(opt_post(c)): inits.append(c)
        source = 'CMA-ES cold start'
    k = prev + 1
    print(f"[{tag}] {source}; {n_chains} chains x {n_iter} iters -> chunk {k}", flush=True)
    mcmc = pints.MCMCController(post, n_chains, inits, method=pints.DreamMCMC)
    mcmc.set_parallel(True)
    mcmc.set_max_iterations(n_iter); mcmc.set_log_to_screen(False)
    mcmc.set_chain_filename(f'outputs/{tag}_dream_c{k}.csv')
    mcmc.run()
    pooled = R.pooled_posterior(tag)
    np.save(f'outputs/{tag}_dream.npy', pooled)
    R.report(tag, pooled, labels, warmup_frac=0.0)
    write_csv_mix(tag, labels, pooled)


def prior_predictive_mix(model_id, n_draws=100, seed=0):
    """Prior-predictive check for the mixture: draw params from the prior, simulate
    prey-eaten counts, and compare the per-n0 distribution to the data."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    tag = f'model_{model_id}a'
    prior = build_prior_mix(model_id)
    np.random.seed(seed)
    draws = prior.sample(n_draws)
    sims = simulate(model_id, draws)                     # (n_draws, nfish) counts
    levels = sorted(set(int(n) for n in N0))
    ncol = 3; nrow = int(np.ceil(len(levels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow)); axs = axs.ravel()
    covered = total = 0
    for i, n in enumerate(levels):
        counts = np.arange(n + 1)
        # per-draw predictive distribution over counts for fish with this n0
        fishmask = N0 == n
        perdraw = np.array([np.bincount(sims[d, fishmask], minlength=n + 1)[:n + 1] /
                            max(1, fishmask.sum()) for d in range(len(sims))])
        data = np.zeros(n + 1)
        for y in NE[fishmask]:
            data[int(y)] += 1
        data = data / max(1, fishmask.sum())
        lo, hi = np.percentile(perdraw, 2.5, 0), np.percentile(perdraw, 97.5, 0)
        axs[i].fill_between(counts, lo, hi, alpha=.3, color='C0')
        axs[i].plot(counts, perdraw.mean(0), color='C0', label='prior pred.')
        axs[i].plot(counts, data, 'k-', lw=1.2, label='data')
        axs[i].set_title(f'n0 = {n}', fontsize=10); axs[i].set_xlabel('num prey eaten'); axs[i].set_ylabel('prob')
        if i == 0:
            axs[i].legend(fontsize=8)
        covered += int(((data >= lo) & (data <= hi)).sum()); total += len(counts)
    for j in range(len(levels), len(axs)):
        axs[j].axis('off')
    fig.tight_layout(); fig.savefig(f'outputs/{tag}_prior_predictive.png', dpi=115)
    print(f"[{tag}] prior-predictive: data in 95% band {covered}/{total} bins; "
          f"saved outputs/{tag}_prior_predictive.png", flush=True)


def map_check(model_id, opt_iters=120):
    """Fast validation: CMA-ES to the MAP, simulate, report discrepancy."""
    prior = build_prior_mix(model_id)
    post = pints.LogPosterior(MixLikelihood(model_id, n_jobs=-1), prior)
    np.random.seed(0)
    x0 = next(c for c in (prior.sample()[0] for _ in range(300))
              if np.isfinite(post(c)))
    opt = pints.OptimisationController(post, x0, method=pints.CMAES)
    opt.set_max_iterations(opt_iters); opt.set_log_to_screen(False)
    xhat, ll = opt.run()
    nbase = MixLikelihood(model_id).nbase
    from scipy.special import expit
    print(f"[model {model_id}a] MAP logpost={ll:.1f}  "
          f"gamma0={xhat[nbase]:.2f} gamma_w={xhat[nbase+1]:.2f}  "
          f"(voracious prob: light~{expit(xhat[nbase]-1.4*xhat[nbase+1]):.2f} "
          f"heavy~{expit(xhat[nbase]+1.4*xhat[nbase+1]):.2f})")
    sims = simulate(model_id, np.repeat(xhat[None, :], 150, axis=0))
    discrepancy(model_id, sims)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="1,2")
    ap.add_argument("--n-iter", type=int, default=2000)
    ap.add_argument("--n-chains", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--discrepancy", action="store_true")
    ap.add_argument("--prior-predictive", action="store_true")
    ap.add_argument("--map-check", action="store_true")
    ap.add_argument("--opt-iters", type=int, default=120)
    ap.add_argument("--init-spread", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mids = [int(s) for s in a.models.split(",")]
    for mid in mids:
        if a.map_check:
            map_check(mid, a.opt_iters)
        elif a.prior_predictive:
            prior_predictive_mix(mid)
        elif a.consolidate:
            _, _, labels = design_mix(mid)
            write_csv_mix(f'model_{mid}a', labels, R.pooled_posterior(f'model_{mid}a'))
        elif a.discrepancy:
            arr, _ = load_posterior_mix(mid)
            flat = arr.reshape(-1, arr.shape[-1])
            draws = flat[np.random.RandomState(0).choice(len(flat), 150, False)]
            discrepancy(mid, simulate(mid, draws))
        else:
            run_model_mix(mid, a.n_iter, a.n_chains, a.resume,
                          a.opt_iters, a.init_spread, a.seed)
