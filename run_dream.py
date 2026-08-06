"""Run DREAM inference for both functional-response models, writing chains to CSV.

Standalone (not the notebook) on purpose: the fast, robust parallelism used here
-- a *serial* likelihood plus ``pints`` ``set_parallel`` across the chains (one
persistent process pool) -- relies on ``spawn``, which is unreliable inside
Jupyter on macOS. Runs at ~3.5 s/iteration for 30 chains on ~10 cores.

The model/prior setup mirrors run_models.ipynb exactly: standardized weight,
recentred+calibrated Gaussian priors (log a0 ~ log 4, log r0 ~ log 20, p-aware
slopes). Model 1 = covariates on the attack rate a; Model 2 = a constant, all
covariates on the digestion rate r (this is the configuration that converged).

Usage (from the repo root):
    python run_dream.py                      # both models, 4000 iters each
    python run_dream.py --models 2 --n-iter 2000
    python run_dream.py --resume             # continue from the latest chunk

Chunked checkpointing (kill-resistant): each launch writes a new chunk,
    outputs/<tag>_dream_c<k>_<chain>.csv     # incremental, one file per chain
and --resume warm-starts the chains from the last saved positions of the latest
chunk (no warmup redone). At the end all chunks are pooled into
    outputs/<tag>_dream.npy
and convergence diagnostics (R-hat, ESS, 95% CI) are printed.
"""
import argparse
import glob
import re
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pints
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mechanistic_model as mm

# --------------------------------------------------------------------------
# Data + design matrices (identical to the notebook)
# --------------------------------------------------------------------------
DF = pd.read_csv('data/df_covariates_better_date.csv')
NE = DF['n_prey_eaten'].to_numpy()
N0 = DF['n_prey_initial'].to_numpy()
_W = DF['weight']
WZ = ((_W - _W.mean()) / _W.std()).to_numpy()
DUMMIES = ["f.acc_condH", "f.acc_condM", "f.acc_condT", "f.exp_condH",
           "f.exp_condM", "f.exp_condT", "f.week6", "f.acc_condH:f.week6",
           "f.acc_condM:f.week6", "f.acc_condT:f.week6"]


def design(model_id):
    """Return (X_a, X_r, labels) for model 1 or 2, each with a leading Ones col."""
    ones = np.ones((len(DF), 1))
    dummies = DF[DUMMIES].to_numpy()
    weight = np.column_stack([WZ, WZ ** 2])
    if model_id == 1:                      # covariates on a; weight on r
        X_a = np.column_stack([ones, dummies])
        X_r = np.column_stack([ones, weight])
        labels = (['a0'] + [f'a:{d}' for d in DUMMIES]
                  + ['r0', 'r:weight_z', 'r:weight_z2'])
    else:                                  # a constant; everything on r
        X_a = ones
        X_r = np.column_stack([ones, dummies, weight])
        labels = (['a0(const)', 'r0'] + [f'r:{d}' for d in DUMMIES]
                  + ['r:weight_z', 'r:weight_z2'])
    return X_a, X_r, labels


def build_prior(n_a, n_r, log_a0=np.log(4.0), sd_a=1.0,
                log_r0=np.log(20.0), sd_r=0.5, slope_marginal_sd=0.75):
    """Calibrated, p-aware Gaussian prior on [beta_a, beta_r] (see notebook)."""
    mean = np.zeros(n_a + n_r)
    var = np.zeros(n_a + n_r)
    mean[0] = log_a0
    var[0] = sd_a ** 2
    if n_a - 1 > 0:
        var[1:n_a] = (slope_marginal_sd / np.sqrt(n_a - 1)) ** 2
    mean[n_a] = log_r0
    var[n_a] = sd_r ** 2
    if n_r - 1 > 0:
        var[n_a + 1:] = (slope_marginal_sd / np.sqrt(n_r - 1)) ** 2
    return pints.MultivariateGaussianLogPrior(mean, np.diag(var))


class Likelihood(pints.LogPDF):
    """n_jobs controls parallelism of the 668 per-fish solves.

    Use n_jobs=-1 (joblib across fish) for single-point work like CMA-ES/init.
    Use n_jobs=1 (serial) for the DREAM run itself, where pints ``set_parallel``
    parallelises across the 30 chains -- nesting joblib inside those workers
    would reintroduce the loky memmap churn that kills long runs.
    """

    def __init__(self, X_a, X_r, n_jobs=1):
        self.X_a = X_a
        self.X_r = X_r
        self.n_a = X_a.shape[1]
        self.n_r = X_r.shape[1]
        self.n_jobs = n_jobs

    def __call__(self, x):
        ll, _ = mm.log_likelihood_all_fish_parallel(
            NE, N0, x[:self.n_a], x[self.n_a:], self.X_a, self.X_r, n_jobs=self.n_jobs)
        return ll

    def n_parameters(self):
        return self.n_a + self.n_r


# --------------------------------------------------------------------------
# Checkpointing helpers
# --------------------------------------------------------------------------
def latest_chunk(tag):
    """Highest existing chunk index for `tag`, or -1 if none."""
    idxs = [int(re.search(r'_c(\d+)_0\.csv$', f).group(1))
            for f in glob.glob(f'outputs/{tag}_dream_c*_0.csv')]
    return max(idxs) if idxs else -1


def load_chunk(tag, k):
    """Stack a chunk's per-chain CSVs into (chains, iters, params)."""
    files = sorted(glob.glob(f'outputs/{tag}_dream_c{k}_*.csv'),
                   key=lambda f: int(re.search(r'_(\d+)\.csv$', f).group(1)))
    n = min(len(pd.read_csv(f)) for f in files)
    return np.stack([pd.read_csv(f).to_numpy()[:n] for f in files])


def pooled_posterior(tag, first_warm_discard=500):
    """Pool the usable post-warmup posterior across chunks.

    Chunk 0 is the cold (CMA-ES-started) run; chunks 1+ are warm-started
    continuations. So we drop chunk 0, discard the first `first_warm_discard`
    iterations of the first warm chunk, and keep everything after. If only the
    cold chunk exists, fall back to discarding its first half.
    """
    k = latest_chunk(tag)
    if k < 0:
        raise FileNotFoundError(f'no chunks found for {tag}')
    if k == 0:
        c = load_chunk(tag, 0)
        return c[:, c.shape[1] // 2:, :]
    parts = [load_chunk(tag, 1)[:, first_warm_discard:, :]]
    parts += [load_chunk(tag, j) for j in range(2, k + 1)]
    return np.concatenate(parts, axis=1)


def load_posterior(model_id):
    """Load the consolidated posterior CSV -> (chains, iters, params), labels.

    This is the durable artifact; analyses (LOO, figures) read it so the raw
    per-chunk checkpoint CSVs can be deleted after consolidation.
    """
    df = pd.read_csv(f'outputs/model_{model_id}_posterior.csv')
    _, _, labels = design(model_id)
    C = df['chain'].nunique()
    N = df['iter'].nunique()
    arr = df[labels].to_numpy().reshape(C, N, len(labels))
    return arr, labels


def write_single_csv(tag, model_id, chains):
    """Write the pooled posterior to ONE tidy CSV: columns chain, iter, <params>."""
    _, _, labels = design(model_id)
    C, N, P = chains.shape
    df = pd.DataFrame(chains.reshape(C * N, P), columns=labels)
    df.insert(0, 'iter', np.tile(np.arange(N), C))
    df.insert(0, 'chain', np.repeat(np.arange(C), N))
    path = f'outputs/{tag}_posterior.csv'
    df.to_csv(path, index=False)
    print(f"[{tag}] wrote {path}: {C * N} draws x {P} params "
          f"({C} chains x {N} post-warmup iters)", flush=True)
    return path


# --------------------------------------------------------------------------
# Prior predictive check
# --------------------------------------------------------------------------
def _pvecs_one_draw(x, X_a, X_r):
    """Return the eaten-count probability vector p(eaten | n0) for every fish
    under one prior draw x (list of arrays, in fish order)."""
    n_a = X_a.shape[1]
    ba, br = x[:n_a], x[n_a:]
    pvecs = []
    for k in range(len(NE)):
        a = np.exp(ba @ X_a[k])
        r = np.exp(br @ X_r[k])
        p = np.clip(mm.solve_system_simple_h_simplex(int(N0[k]), a, r), 0, None)
        pvecs.append(p / p.sum())
    return pvecs


def prior_predictive(model_id, n_draws=100, seed=0):
    """Prior-predictive check as a grid of panels, one per initial prey count n0.

    For each n0, plots the prior-predictive distribution over the number of prey
    eaten (mean + 95% band across prior draws x fish with that n0) against the
    empirical data distribution. A well-calibrated prior should cover the data in
    every panel. Saves outputs/model_<id>_prior_predictive.png.
    """
    from joblib import Parallel, delayed
    tag = f'model_{model_id}'
    X_a, X_r, _ = design(model_id)
    prior = build_prior(X_a.shape[1], X_r.shape[1])
    np.random.seed(seed)
    draws = prior.sample(n_draws)
    per_draw = Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(_pvecs_one_draw)(draws[i], X_a, X_r) for i in range(n_draws))

    uniq = sorted(set(int(n) for n in N0))
    # prior-predictive p-vectors grouped by n0 (over draws x fish)
    prior_by_n0 = {n: [] for n in uniq}
    for pvecs in per_draw:
        for k, p in enumerate(pvecs):
            prior_by_n0[int(N0[k])].append(p)
    # empirical data distribution per n0 (one-hot of observed eaten, over fish)
    data_by_n0 = {n: [] for n in uniq}
    for k in range(len(NE)):
        n = int(N0[k])
        oh = np.zeros(n + 1)
        oh[int(NE[k])] = 1.0
        data_by_n0[n].append(oh)

    ncol = 3
    nrow = int(np.ceil(len(uniq) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow))
    axs = np.atleast_1d(axs).ravel()
    covered = total = 0
    for i, n in enumerate(uniq):
        ax = axs[i]
        m = np.array(prior_by_n0[n])            # (draws*fish, n+1)
        d = np.array(data_by_n0[n]).mean(0)     # (n+1,)
        counts = np.arange(n + 1)
        lo = np.percentile(m, 2.5, axis=0)
        hi = np.percentile(m, 97.5, axis=0)
        ax.fill_between(counts, lo, hi, alpha=0.3, color='C0')
        ax.plot(counts, m.mean(0), color='C0', label='prior pred.')
        ax.plot(counts, d, 'k-', lw=1.2, label='data')
        ax.set_title(f'n0 = {n}', fontsize=10)
        ax.set_xlabel('num prey eaten'); ax.set_ylabel('probability')
        if i == 0:
            ax.legend(fontsize=8)
        covered += int(((d >= lo) & (d <= hi)).sum())
        total += len(counts)
    for j in range(len(uniq), len(axs)):
        axs[j].axis('off')
    fig.suptitle(f'{tag} prior predictive: eaten distribution per n0 '
                 f'({n_draws} draws)')
    fig.tight_layout()
    fig.savefig(f'outputs/{tag}_prior_predictive.png', dpi=110)
    plt.close(fig)

    print(f"[{tag}] prior-predictive: data inside 95% band in {covered}/{total} "
          f"count bins ({covered / total:.0%}); "
          f"saved outputs/{tag}_prior_predictive.png", flush=True)


# --------------------------------------------------------------------------
# Model comparison: PSIS-LOO
# --------------------------------------------------------------------------
def _pointwise_loglik(x, X_a, X_r):
    """log p(y_i | x) for every fish under one posterior draw x (for LOO)."""
    n_a = X_a.shape[1]
    ba, br = x[:n_a], x[n_a:]
    out = np.empty(len(NE))
    for k in range(len(NE)):
        a = np.exp(ba @ X_a[k])
        r = np.exp(br @ X_r[k])
        p = np.clip(mm.solve_system_simple_h_simplex(int(N0[k]), a, r), 1e-300, None)
        p = p / p.sum()
        out[k] = np.log(p[int(NE[k])])
    return out


def loo_compare(model_ids, n_draws_per_chain=60, seed=0):
    """PSIS-LOO for each model + a compare() table ranking them.

    Builds the pointwise log-likelihood matrix (chain, draw, fish) from the pooled
    posterior, then uses ArviZ's PSIS-LOO (with Pareto-k reliability diagnostics).
    """
    import arviz as az
    from joblib import Parallel, delayed
    idatas = {}
    for mid in model_ids:
        tag = f'model_{mid}'
        X_a, X_r, _ = design(mid)
        chains, _ = load_posterior(mid)                # from consolidated CSV
        C, N, P = chains.shape
        idx = np.unique(np.linspace(0, N - 1, min(n_draws_per_chain, N)).astype(int))
        sub = chains[:, idx, :]                         # (C, n, P)
        flat = sub.reshape(-1, P)
        ll = np.array(Parallel(n_jobs=-1, max_nbytes=None)(
            delayed(_pointwise_loglik)(x, X_a, X_r) for x in flat))
        ll = ll.reshape(C, len(idx), -1)                # (chain, draw, fish)
        n_bad = int(np.sum(~np.isfinite(ll)))
        if n_bad:
            print(f"[{tag}] warning: {n_bad} non-finite pointwise loglik entries", flush=True)
        idata = az.from_dict(posterior={'theta': sub},
                             log_likelihood={'obs': ll})
        idatas[tag] = idata
        loo = az.loo(idata, pointwise=True)
        print(f"\n[{tag}] PSIS-LOO\n{loo}", flush=True)

    if len(idatas) > 1:
        print("\n===== az.compare (PSIS-LOO) =====", flush=True)
        print(az.compare(idatas, ic='loo'), flush=True)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def split_rhat(x):
    m = x.shape[1] // 2
    x = x[:, :2 * m]
    d = np.concatenate(np.array_split(x, 2, axis=1), axis=0)
    mm_ = d.shape[1]
    B = mm_ * d.mean(1).var(ddof=1)
    W = d.var(1, ddof=1).mean()
    return np.sqrt(((mm_ - 1) / mm_ * W + B / mm_) / W) if W > 0 else np.nan


def ess(x):
    def a1(y):
        y = y - y.mean()
        return (y[:-1] * y[1:]).sum() / (y * y).sum() if y.std() > 0 else 1.0
    r = np.mean([a1(x[c]) for c in range(x.shape[0])])
    tau = (1 + r) / (1 - r) if r < 1 else np.inf
    return x.size / tau if np.isfinite(tau) else 0.0


def report(tag, chains, labels, warmup_frac=0.5):
    post = chains[:, int(chains.shape[1] * warmup_frac):, :]
    print(f"\n=== {tag}: {chains.shape[0]} chains x {chains.shape[1]} iters "
          f"(discard first {warmup_frac:.0%}) ===")
    print(f"{'param':16s}{'mean':>8s}{'2.5%':>8s}{'97.5%':>8s}{'Rhat':>7s}{'ESS':>8s}")
    rmax = 0.0
    for k, lab in enumerate(labels):
        fl = post[:, :, k].ravel()
        rh = split_rhat(post[:, :, k])
        rmax = max(rmax, rh)
        print(f"{lab:16s}{fl.mean():8.2f}{np.percentile(fl, 2.5):8.2f}"
              f"{np.percentile(fl, 97.5):8.2f}{rh:7.2f}{ess(post[:, :, k]):8.0f}")
    print(f"max Rhat = {rmax:.2f}")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_model(model_id, n_iter, n_chains, resume, opt_iters, init_spread, seed):
    tag = f'model_{model_id}'
    X_a, X_r, labels = design(model_id)
    prior = build_prior(X_a.shape[1], X_r.shape[1])
    # Fast joblib-parallel posterior for single-point work (CMA-ES, init search);
    # serial posterior for DREAM, which parallelises across chains via set_parallel.
    opt_posterior = pints.LogPosterior(Likelihood(X_a, X_r, n_jobs=-1), prior)
    posterior = pints.LogPosterior(Likelihood(X_a, X_r, n_jobs=1), prior)
    np.random.seed(seed)

    prev = latest_chunk(tag)
    if resume and prev >= 0:
        inits = list(load_chunk(tag, prev)[:, -1, :])   # warm-start (already finite)
        n_chains = len(inits)
        source = f'chunk {prev}'
    else:
        x0 = next(c for c in (prior.sample()[0] for _ in range(200))
                  if np.isfinite(opt_posterior(c)))
        opt = pints.OptimisationController(opt_posterior, x0, method=pints.CMAES)
        opt.set_max_iterations(opt_iters)
        opt.set_log_to_screen(False)
        xhat, _ = opt.run()
        np_ = posterior.n_parameters()
        # Reject-resample so every chain starts with finite logpdf: dispersing
        # around xhat can land in the p_eaten-underflow (-inf) region, which
        # DREAM refuses to start from.
        inits = []
        while len(inits) < n_chains:
            cand = xhat + np.random.normal(0, init_spread, np_)
            if np.isfinite(opt_posterior(cand)):
                inits.append(cand)
        source = 'CMA-ES cold start'

    k = prev + 1
    print(f"[{tag}] {source}; {n_chains} chains x {n_iter} iters -> chunk {k}", flush=True)
    mcmc = pints.MCMCController(posterior, n_chains, inits, method=pints.DreamMCMC)
    mcmc.set_parallel(True)                    # one persistent pool, across chains
    mcmc.set_max_iterations(n_iter)
    mcmc.set_log_to_screen(False)
    mcmc.set_chain_filename(f'outputs/{tag}_dream_c{k}.csv')
    mcmc.run()

    # pool the usable post-warmup posterior (drop cold chunk 0, warmup of first warm)
    pooled = pooled_posterior(tag)
    np.save(f'outputs/{tag}_dream.npy', pooled)
    report(tag, pooled, labels, warmup_frac=0.0)     # already warmup-discarded
    write_single_csv(tag, model_id, pooled)          # one consolidated CSV


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='1,2', help='comma list, e.g. "2" or "1,2"')
    ap.add_argument('--n-iter', type=int, default=4000)
    ap.add_argument('--n-chains', type=int, default=30)
    ap.add_argument('--resume', action='store_true',
                    help='warm-start from the latest chunk instead of CMA-ES')
    ap.add_argument('--prior-predictive', action='store_true',
                    help='run prior-predictive checks (save figures) and exit; '
                         'no MCMC')
    ap.add_argument('--pp-draws', type=int, default=100,
                    help='number of prior draws for the prior-predictive check')
    ap.add_argument('--consolidate', action='store_true',
                    help='write the pooled posterior of each model to a single '
                         'CSV (outputs/model_<id>_posterior.csv) and exit; no MCMC')
    ap.add_argument('--loo', action='store_true',
                    help='PSIS-LOO model comparison from the pooled posteriors '
                         '(needs arviz); prints per-model LOO + compare table')
    ap.add_argument('--opt-iters', type=int, default=150)
    ap.add_argument('--init-spread', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    models = [int(s) for s in args.models.split(',')]
    if args.prior_predictive:
        for mid in models:
            prior_predictive(mid, n_draws=args.pp_draws, seed=args.seed)
    elif args.consolidate:
        for mid in models:
            pooled = pooled_posterior(f'model_{mid}')
            write_single_csv(f'model_{mid}', mid, pooled)
    elif args.loo:
        loo_compare(models)
    else:
        for mid in models:
            run_model(mid, args.n_iter, args.n_chains, args.resume,
                      args.opt_iters, args.init_spread, args.seed)
