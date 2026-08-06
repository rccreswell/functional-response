"""Targeted PPC for the warmed-acclimation x week-6 interaction (base models 1, 2).

Restricts to fish acclimated in the Warmed condition (acc_cond == 'H') and shows
actual vs model-simulated prey-eaten counts, faceted by acclimation duration
(week 1 vs week 6), across initial prey. This is the cell whose interaction
coefficient is large in the base models, so it checks whether that effect is
supported by the data. Usage: python ppc_warmed_week.py --model 1  (or 2)
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from joblib import Parallel, delayed
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import run_dream as R
from make_ppc_violin import _sim_all_fish

plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})
NE, N0 = R.NE, R.N0
ACC = R.DF['acc_cond'].to_numpy()
WEEK = R.DF['week'].to_numpy()
GREEN, ORANGE = '#4DA37E', '#D2691E'


def make(model_id, n_draws=120):
    arr, _ = R.load_posterior(model_id)
    flat = arr.reshape(-1, arr.shape[-1])
    draws = flat[np.random.RandomState(0).choice(len(flat), n_draws, replace=False)]
    X_a, X_r, _ = R.design(model_id)
    sims = np.array(Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(_sim_all_fish)(draws[i], X_a, X_r, 5000 + i) for i in range(n_draws)))

    Hmask = ACC == 'H'
    n0_levels = sorted(set(int(n) for n in N0[Hmask]))

    def style(vp, c):
        for b in vp['bodies']:
            b.set_facecolor(c); b.set_alpha(0.85); b.set_edgecolor('0.3'); b.set_linewidth(0.7)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, wk in zip(axes, [1, 6]):
        base = Hmask & (WEEK == wk)
        for i, n in enumerate(n0_levels):
            cell = base & (N0 == n)
            if cell.sum() == 0:
                continue
            actual = NE[cell]; sim = sims[:, cell].ravel()
            style(ax.violinplot([sim], positions=[i + 0.2], widths=0.36,
                                showextrema=False), ORANGE)
            if len(actual) >= 3 and len(set(actual.tolist())) >= 2:
                style(ax.violinplot([actual], positions=[i - 0.2], widths=0.36,
                                    showextrema=False), GREEN)
            ax.scatter(np.full(len(actual), i - 0.2), actual, s=10,
                       color='0.2', alpha=0.55, zorder=3)
        ax.set_title(f'Warmed acclimation — week {wk}  (n={int((base).sum())} fish)',
                     fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.3', fc='0.92', ec='0.6'))
        ax.set_xticks(range(len(n0_levels))); ax.set_xticklabels(n0_levels)
        ax.set_xlabel('# initial prey')
    axes[0].set_ylabel('# prey eaten')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=GREEN, alpha=0.85, label='actual'),
               plt.Rectangle((0, 0), 1, 1, fc=ORANGE, alpha=0.85, label='simulated')]
    fig.legend(handles=handles, title='Data', loc='center right', frameon=False)
    which = "conditions on r" if model_id == 2 else "conditions on a"
    fig.suptitle(f"Model {model_id} ({which}): warmed-acclimation x week PPC", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    out = f'outputs/model_{model_id}_ppc_warmed_week.png'
    fig.savefig(out, dpi=120)
    # numeric summary: mean eaten obs vs sim, warmed fish, per week
    print(f"[model {model_id}] warmed acclimation — mean # eaten (obs vs sim):")
    for wk in [1, 6]:
        m = ACC == 'H'
        m = m & (WEEK == wk)
        print(f"  week {wk}: obs={NE[m].mean():.2f}  sim={sims[:, m].mean():.2f}  (n={m.sum()})")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True, choices=[1, 2])
    make(ap.parse_args().model)
