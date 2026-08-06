"""Figure 3 for a fitted voracious-mixture model (2a / 1a).

Same layout as make_fig3.py (A: time vs experimental condition; B: time vs
acclimation condition; C: model-predicted # prey eaten, n0=40, median weight; all
by acclimation duration), but Panel C uses the *mixture* predictive (ordinary CME
component + instant-capture voracious component). Time in A/B is handling time
(60/r) for Model 2a and preying time (60/(a*40)) for Model 1a, exactly as in the
base figure. Reads outputs/model_<id>a_posterior.csv.

Usage: python make_fig3_mixture.py --model 2   (or 1)
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import make_fig3 as F3
import mechanistic_model as mm
import run_dream as R

COND, DUR, N0_REF = F3.COND, F3.DUR, F3.N0_REF
WZ_MEDIAN = 0.0                                 # median (z-scored) weight


def _pi(post, w):
    return 1.0 / (1.0 + np.exp(-(post['gamma0'].to_numpy() + post['gamma_w'].to_numpy() * w)))


def _sim_panelC(args):
    """One (acc, week) cell: sample mixture-predicted eaten at n0=N0_REF per draw."""
    la, lr, pi, seed = args
    rng = np.random.RandomState(seed)
    a = np.exp(la); r = np.exp(lr); out = np.empty(len(a), dtype=int)
    for i in range(len(a)):
        if rng.random() < pi[i]:
            out[i] = min(N0_REF, 1 + rng.poisson(r[i]))
        else:
            p = np.clip(mm.solve_system_simple_h_simplex(N0_REF, a[i], r[i]), 0, None)
            out[i] = rng.choice(len(p), p=p / p.sum())
    return out


def make(model_id):
    tag = f'model_{model_id}a'
    post = pd.read_csv(f'outputs/{tag}_posterior.csv')
    postC = post.sample(1000, random_state=0)
    rate_char, time_name = ('r', 'Handling time, minutes') if model_id == 2 \
        else ('a', f'Preying time (n0={N0_REF}), minutes')

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10)); yoff = {0: 0.18, 1: -0.18}

    def panel_time(ax, vary):
        for yi, (code, lab) in enumerate(COND):
            y = len(COND) - 1 - yi
            for w, _, col in DUR:
                acc = code if vary == 'acc' else 'C'
                exp = code if vary == 'exp' else 'C'
                rate = np.exp(F3.log_rate(post, model_id, rate_char, acc, exp, w, WZ_MEDIAN))
                t = F3.rate_to_time(rate, model_id)
                med = np.median(t); lo, hi = np.percentile(t, [2.5, 97.5])
                ax.plot([lo, hi], [y + yoff[w]] * 2, color=col, lw=2.0, solid_capstyle='round', zorder=1)
                ax.plot(med, y + yoff[w], 'o', color=col, ms=8, markeredgecolor='white', markeredgewidth=0.8, zorder=2)
        ax.set_yticks(range(len(COND))); ax.set_yticklabels([l for _, l in COND][::-1])
        ax.set_ylim(-0.6, len(COND) - 0.4); ax.set_xlabel(time_name); ax.margins(x=0.05)

    panel_time(axes[0], 'exp'); axes[0].set_ylabel('Experimental\ncondition')
    panel_time(axes[1], 'acc'); axes[1].set_ylabel('Acclimation\ncondition')

    ax = axes[2]
    for yi, (code, lab) in enumerate(COND):
        y = len(COND) - 1 - yi
        for w, _, col in DUR:
            la = F3.log_rate(postC, model_id, 'a', code, 'C', w, WZ_MEDIAN)
            lr = F3.log_rate(postC, model_id, 'r', code, 'C', w, WZ_MEDIAN)
            pi = _pi(postC, WZ_MEDIAN) * np.ones(len(la))
            counts = _sim_panelC((la, lr, pi, 12345 + yi * 10 + w))
            parts = ax.violinplot([counts], positions=[y + yoff[w]], vert=False, widths=0.34, showextrema=False)
            for b in parts['bodies']:
                b.set_facecolor(col); b.set_alpha(0.85); b.set_edgecolor('0.25'); b.set_linewidth(0.9)
            ax.plot(np.median(counts), y + yoff[w], '^', color='k', ms=7, markeredgecolor='white', markeredgewidth=0.6, zorder=3)
    ax.set_yticks(range(len(COND))); ax.set_yticklabels([l for _, l in COND][::-1])
    ax.set_ylim(-0.6, len(COND) - 0.4); ax.set_xlabel('# prey eaten'); ax.set_ylabel('Acclimation\ncondition')

    for a, L in zip(axes, 'ABC'):
        a.text(-0.14, 1.02, L, transform=a.transAxes, fontsize=15, fontweight='bold')
    handles = [plt.Line2D([], [], color=c, marker='s', ls='', ms=9, label=l) for _, l, c in DUR]
    fig.legend(handles=handles, title='Acclimation duration', loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    mname = 'Model 2a (mixture; conditions on r)' if model_id == 2 else 'Model 1a (mixture; conditions on a)'
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(f'outputs/{tag}_fig3.png', dpi=120, bbox_inches='tight')
    print(f"saved outputs/{tag}_fig3.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", type=int, required=True, choices=[1, 2])
    make(ap.parse_args().model)
