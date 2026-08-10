"""Reproduce all posterior-predictive-check figures for a fitted mixture model
(2a / 1a) and write a CSV table of Bayesian p-values.

Produces, for model <id>a:
  model_<id>a_posterior_predictive.png   per-n0 eaten-count distribution vs data
  model_<id>a_ppc_violin[_acc|_exp].png  actual vs simulated, by weight/acc/exp
  model_<id>a_ppc_warmed_week.png        warmed-acclimation x week
  model_<id>a_ppc_discrepancy.png        discrepancy stats with Bayesian p-values
  model_<id>a_loo_pit.png                LOO-PIT calibration
  model_<id>a_bayes_pvalues.csv          table of test statistics + p-values

Usage: python ppc_mixture.py --model 2   (or 1)
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import arviz as az
import run_dream as R
import run_dream_mixture as M
import mechanistic_model as mm
from mixture_model import voracious_pmf, mixture_loglik

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
NE, N0, WZ = R.NE, R.N0, R.WZ
N0_LEVELS = sorted(set(int(n) for n in N0))
WEIGHT = R.DF['weight'].to_numpy()
ACC = R.DF['acc_cond'].to_numpy(); EXP = R.DF['exp_cond'].to_numpy(); WEEK = R.DF['week'].to_numpy()
GREEN, ORANGE = '#4DA37E', '#D2691E'
COND_MAP = {'C': 'Ambient', 'H': 'Warmed', 'T': 'Turbid', 'M': 'Mixed'}
COND_ORDER = ['Ambient', 'Warmed', 'Turbid', 'Mixed']


def _mix_pmf_one(x, model_id):
    X_a, X_r, _ = R.design(model_id); na = X_a.shape[1]; nbase = na + X_r.shape[1]
    ba, br = x[:na], x[na:nbase]; g0, gw = x[nbase], x[nbase + 1]
    a = np.exp(X_a @ ba); r = np.exp(X_r @ br); pi = 1 / (1 + np.exp(-(g0 + gw * WZ)))
    out = []
    for k in range(len(NE)):
        n0k = int(N0[k])
        po = np.clip(mm.solve_system_simple_h_simplex(n0k, a[k], r[k]), 0, None); po = po / po.sum()
        out.append((1 - pi[k]) * po + pi[k] * voracious_pmf(n0k, r[k]))
    return out


def _mix_loglik_one(x, model_id):
    X_a, X_r, _ = R.design(model_id); na = X_a.shape[1]; nbase = na + X_r.shape[1]
    _, lls = mixture_loglik(NE, N0, x[:na], x[na:nbase], x[nbase], x[nbase + 1],
                            X_a, X_r, WZ, n_jobs=1)
    return np.array(lls)


def _style(vp, c):
    for b in vp['bodies']:
        b.set_facecolor(c); b.set_alpha(0.85); b.set_edgecolor('0.3'); b.set_linewidth(0.7)


def violin_facets(tag, sims, labels_per_fish, levels, ftitle, fname, points=True):
    fig, axes = plt.subplots(1, len(levels), figsize=(4.8 * len(levels), 6), sharey=True)
    for ax, lev in zip(np.atleast_1d(axes), levels):
        lmask = labels_per_fish == lev
        for i, n in enumerate(N0_LEVELS):
            cell = lmask & (N0 == n)
            if cell.sum() == 0:
                continue
            actual = NE[cell]; sim = sims[:, cell].ravel()
            _style(ax.violinplot([sim], positions=[i + 0.2], widths=0.36, showextrema=False), ORANGE)
            if len(actual) >= 3 and len(set(actual.tolist())) >= 2:
                _style(ax.violinplot([actual], positions=[i - 0.2], widths=0.36, showextrema=False), GREEN)
            if points:
                ax.scatter(np.full(len(actual), i - 0.2), actual, s=6, color='0.25', alpha=0.45, zorder=3)
        ax.set_title(lev, fontsize=12)
        ax.set_xticks(range(len(N0_LEVELS))); ax.set_xticklabels(N0_LEVELS); ax.set_xlabel('initial prey, $N_0$')
    np.atleast_1d(axes)[0].set_ylabel('# prey eaten')
    h = [plt.Rectangle((0, 0), 1, 1, fc=GREEN, alpha=.85, label='actual'),
         plt.Rectangle((0, 0), 1, 1, fc=ORANGE, alpha=.85, label='simulated')]
    fig.tight_layout(rect=[0, 0, 0.88, 1])       # reserve right margin for the legend
    fig.legend(handles=h, title='Data', loc='center left', bbox_to_anchor=(0.89, 0.5), frameon=False)
    fig.savefig(fname, dpi=115, bbox_inches='tight'); plt.close(fig)


def run(model_id, n_draws=200):
    tag = f'model_{model_id}a'
    arr, _ = M.load_posterior_mix(model_id)
    flat = arr.reshape(-1, arr.shape[-1])
    rng = np.random.RandomState(0)
    draws = flat[rng.choice(len(flat), n_draws, replace=False)]
    sims = M.simulate(model_id, draws)                     # (n_draws, nfish) counts

    # ---- (1) per-n0 posterior-predictive grid (mixture pmf) ----
    per = Parallel(n_jobs=-1, max_nbytes=None)(delayed(_mix_pmf_one)(draws[i], model_id)
                                               for i in range(min(120, n_draws)))
    by = {n: [] for n in N0_LEVELS}
    for pv in per:
        for k, p in enumerate(pv):
            by[int(N0[k])].append(p)
    dat = {n: [] for n in N0_LEVELS}
    for k in range(len(NE)):
        n = int(N0[k]); oh = np.zeros(n + 1); oh[int(NE[k])] = 1; dat[n].append(oh)
    ncol = 3; nrow = int(np.ceil(len(N0_LEVELS) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow)); axs = axs.ravel()
    for i, n in enumerate(N0_LEVELS):
        m = np.array(by[n]); d = np.array(dat[n]).mean(0); c = np.arange(n + 1)
        axs[i].fill_between(c, np.percentile(m, 2.5, 0), np.percentile(m, 97.5, 0), alpha=.3, color='C1')
        axs[i].plot(c, m.mean(0), color='C1', label='post. pred.'); axs[i].plot(c, d, 'k-', lw=1.2, label='data')
        axs[i].set_title(f'$N_0 = {n}$', fontsize=10); axs[i].set_xlabel('prey eaten'); axs[i].set_ylabel('prob')
        if i == 0:
            axs[i].legend(fontsize=8)
    for j in range(len(N0_LEVELS), len(axs)):
        axs[j].axis('off')
    fig.tight_layout()
    fig.savefig(f'outputs/{tag}_posterior_predictive.png', dpi=115); plt.close(fig)

    # ---- (2) violin PPCs: weight tertile / acc / exp ----
    tert = np.asarray(pd.qcut(WEIGHT, 3, labels=['light', 'moderate', 'heavy']))
    violin_facets(tag, sims, tert, ['light', 'moderate', 'heavy'], 'weight tertile',
                  f'outputs/{tag}_ppc_violin.png', points=False)   # Figure 2: no data-point overlay
    violin_facets(tag, sims, np.array([COND_MAP[a] for a in ACC]), COND_ORDER,
                  'acclimation condition', f'outputs/{tag}_ppc_violin_acc.png')
    violin_facets(tag, sims, np.array([COND_MAP[e] for e in EXP]), COND_ORDER,
                  'experimental condition', f'outputs/{tag}_ppc_violin_exp.png')

    # ---- (3) warmed x week ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    Hmask = ACC == 'H'; hlev = sorted(set(int(n) for n in N0[Hmask]))
    for ax, wk in zip(axes, [1, 6]):
        base = Hmask & (WEEK == wk)
        for i, n in enumerate(hlev):
            cell = base & (N0 == n)
            if cell.sum() == 0:
                continue
            actual = NE[cell]; sim = sims[:, cell].ravel()
            _style(ax.violinplot([sim], positions=[i + .2], widths=.36, showextrema=False), ORANGE)
            if len(actual) >= 3 and len(set(actual.tolist())) >= 2:
                _style(ax.violinplot([actual], positions=[i - .2], widths=.36, showextrema=False), GREEN)
            ax.scatter(np.full(len(actual), i - .2), actual, s=10, color='0.2', alpha=.55, zorder=3)
        ax.set_title(f'Warmed acclimation — week {wk}', fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.3', fc='0.92', ec='0.6'))
        ax.set_xticks(range(len(hlev))); ax.set_xticklabels(hlev); ax.set_xlabel('initial prey, $N_0$')
    axes[0].set_ylabel('# prey eaten')
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=GREEN, alpha=.85, label='actual'),
                        plt.Rectangle((0, 0), 1, 1, fc=ORANGE, alpha=.85, label='simulated')],
               title='Data', loc='center right', frameon=False)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96]); fig.savefig(f'outputs/{tag}_ppc_warmed_week.png', dpi=115); plt.close(fig)

    # ---- (4) discrepancy stats + Bayesian p-values ----
    def full(d): return np.mean(d == N0)
    def zero(d): return np.mean(d == 0)
    def over(d): return float(np.sum([d[N0 == n].var() for n in N0_LEVELS]))
    def meanE(d): return float(d.mean())
    stats = {'full-consumption P(eaten=n0)': full, 'zero-consumption P(eaten=0)': zero,
             'overdispersion sum Var(eaten|n0)': over, 'mean eaten': meanE}
    tert_full = {t: (lambda d, t=t: np.mean(d[tert == t] == N0[tert == t])) for t in ['light', 'moderate', 'heavy']}
    stats.update({f'full-consumption ({t} weight)': f for t, f in tert_full.items()})
    rows = []
    for name, T in stats.items():
        rep = np.array([T(s) for s in sims]); obs = T(NE); p = float(np.mean(rep >= obs))
        rows.append({'statistic': name, 'observed': round(obs, 4), 'model_mean': round(rep.mean(), 4),
                     'model_2.5%': round(np.percentile(rep, 2.5), 4), 'model_97.5%': round(np.percentile(rep, 97.5), 4),
                     'bayes_p_ge': round(p, 3), 'bayes_p_2sided': round(2 * min(p, 1 - p), 3)})
    pd.DataFrame(rows).to_csv(f'outputs/{tag}_bayes_pvalues.csv', index=False)

    # discrepancy figure (first three stats + per-n0 full-consumption)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8)); axes = axes.ravel()
    for ax, (name, T) in zip(axes, list(stats.items())[:3]):
        rep = np.array([T(s) for s in sims]); obs = T(NE); p = float(np.mean(rep >= obs))
        ax.hist(rep, bins=30, color=ORANGE, alpha=0.8); ax.axvline(obs, color='k', lw=2, label=f'observed (p={p:.3f})')
        ax.set_title(name, fontsize=10); ax.legend(fontsize=9); ax.set_xlabel('test statistic')
    ax = axes[3]
    ob = [np.mean(NE[N0 == n] == n) for n in N0_LEVELS]
    rp = np.array([[np.mean(s[N0 == n] == n) for n in N0_LEVELS] for s in sims])
    ax.fill_between(N0_LEVELS, np.percentile(rp, 2.5, 0), np.percentile(rp, 97.5, 0), alpha=.3, color=ORANGE, label='rep 95%')
    ax.plot(N0_LEVELS, rp.mean(0), color=ORANGE, label='rep mean'); ax.plot(N0_LEVELS, ob, 'ko-', label='observed')
    ax.set_title('full-consumption by $N_0$', fontsize=10); ax.set_xlabel('initial prey, $N_0$'); ax.set_ylabel('$P(\mathrm{eaten}=N_0)$'); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'outputs/{tag}_ppc_discrepancy.png', dpi=115); plt.close(fig)

    # ---- (5) LOO-PIT ----
    C, N, P = arr.shape
    idx = np.unique(np.linspace(0, N - 1, 40).astype(int)); sub = arr[:, idx, :]
    fl = sub.reshape(-1, P)
    ll = np.array(Parallel(n_jobs=-1, max_nbytes=None)(delayed(_mix_loglik_one)(x, model_id) for x in fl))
    pp = M.simulate(model_id, fl)
    ll3 = ll.reshape(C, len(idx), -1); pp3 = pp.reshape(C, len(idx), -1)
    idata = az.from_dict(posterior={'theta': sub}, log_likelihood={'obs': ll3},
                         posterior_predictive={'obs': pp3}, observed_data={'obs': NE})
    figp, axp = plt.subplots(figsize=(6, 4)); az.plot_loo_pit(idata, y='obs', y_hat='obs', ax=axp)
    figp.tight_layout()
    figp.savefig(f'outputs/{tag}_loo_pit.png', dpi=115); plt.close(figp)

    print(f"[{tag}] wrote all PPC figures + outputs/{tag}_bayes_pvalues.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", type=int, required=True, choices=[1, 2])
    run(ap.parse_args().model)
