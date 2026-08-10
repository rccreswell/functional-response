"""Posterior-predictive check: actual vs simulated prey-eaten counts.

Paired violins (green = actual data, orange = model-simulated) of the number of
prey eaten, at each initial prey density, faceted by a grouping variable:
  --facet weight   body-weight tertile (light / moderate / heavy)   [default]
  --facet acc      acclimation condition (Ambient/Warmed/Turbid/Mixed)
  --facet exp      experimental condition (Ambient/Warmed/Turbid/Mixed)

Simulated counts are drawn from the fitted posterior of the chosen model.

Usage:  python make_ppc_violin.py --model 1 [--facet acc]
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import run_dream as R
import mechanistic_model as mm

plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})
NE, N0 = R.NE, R.N0
WEIGHT = R.DF['weight'].to_numpy()
N0_LEVELS = sorted(set(int(n) for n in N0))
GREEN, ORANGE = '#4DA37E', '#D2691E'
COND_MAP = {'C': 'Ambient', 'H': 'Warmed', 'T': 'Turbid', 'M': 'Mixed'}
COND_ORDER = ['Ambient', 'Warmed', 'Turbid', 'Mixed']


def group_spec(facet):
    """Return (per-fish label array, ordered facet levels, title)."""
    if facet == 'weight':
        lab = np.asarray(pd.qcut(WEIGHT, 3, labels=['light', 'moderate', 'heavy']))
        return lab, ['light', 'moderate', 'heavy'], 'weight tertile'
    col = 'acc_cond' if facet == 'acc' else 'exp_cond'
    lab = R.DF[col].map(COND_MAP).to_numpy()
    name = 'acclimation condition' if facet == 'acc' else 'experimental condition'
    return lab, COND_ORDER, name


def _sim_all_fish(x, X_a, X_r, seed):
    n_a = X_a.shape[1]
    ba, br = x[:n_a], x[n_a:]
    rng = np.random.RandomState(seed)
    out = np.empty(len(NE), dtype=int)
    for k in range(len(NE)):
        a = np.exp(ba @ X_a[k]); r = np.exp(br @ X_r[k])
        p = np.clip(mm.solve_system_simple_h_simplex(int(N0[k]), a, r), 0, None)
        out[k] = rng.choice(len(p), p=p / p.sum())
    return out


def make_ppc(model_id, facet='weight', n_draws=200):
    arr, _ = R.load_posterior(model_id)
    flat = arr.reshape(-1, arr.shape[-1])
    rng = np.random.RandomState(0)
    draws = flat[rng.choice(len(flat), n_draws, replace=False)]
    X_a, X_r, _ = R.design(model_id)
    sims = np.array(Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(_sim_all_fish)(draws[i], X_a, X_r, 1000 + i) for i in range(n_draws)))

    labels, levels, ftitle = group_spec(facet)

    def style(vp, color):
        for b in vp['bodies']:
            b.set_facecolor(color); b.set_alpha(0.85)
            b.set_edgecolor('0.3'); b.set_linewidth(0.7)

    fig, axes = plt.subplots(1, len(levels), figsize=(4.8 * len(levels), 6),
                             sharey=True)
    for ax, lev in zip(np.atleast_1d(axes), levels):
        lmask = labels == lev
        for i, n in enumerate(N0_LEVELS):
            cell = lmask & (N0 == n)
            if cell.sum() == 0:
                continue
            actual = NE[cell]
            sim = sims[:, cell].ravel()
            style(ax.violinplot([sim], positions=[i + 0.2], widths=0.36,
                                showextrema=False), ORANGE)
            if len(actual) >= 3 and len(set(actual.tolist())) >= 2:
                style(ax.violinplot([actual], positions=[i - 0.2], widths=0.36,
                                    showextrema=False), GREEN)
            ax.scatter(np.full(len(actual), i - 0.2), actual, s=5,
                       color='0.25', alpha=0.45, zorder=3)
        ax.set_title(lev, fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.3', fc='0.92', ec='0.6'))
        ax.set_xticks(range(len(N0_LEVELS)))
        ax.set_xticklabels(N0_LEVELS)
        ax.set_xlabel('initial prey, $N_0$')
    np.atleast_1d(axes)[0].set_ylabel('# prey eaten')

    handles = [plt.Rectangle((0, 0), 1, 1, fc=GREEN, alpha=0.85, label='actual'),
               plt.Rectangle((0, 0), 1, 1, fc=ORANGE, alpha=0.85, label='simulated')]
    fig.legend(handles=handles, title='Data', loc='center right', frameon=False)
    which = "conditions on r" if model_id == 2 else "conditions on a"
    fig.suptitle(f"Posterior predictive check — Model {model_id} ({which}), "
                 f"by {ftitle}", y=0.99)
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])
    suffix = '' if facet == 'weight' else f'_{facet}'
    out = f'outputs/model_{model_id}_ppc_violin{suffix}.png'
    fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True, choices=[1, 2])
    ap.add_argument("--facet", default='weight', choices=['weight', 'acc', 'exp'])
    a = ap.parse_args()
    make_ppc(a.model, a.facet)
