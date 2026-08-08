"""Deterministically plot the estimated voracious-class probability against fish
weight for the mixture model: pi = logistic(gamma0 + gamma_w * weight_z), where
weight_z = (weight - mean)/sd uses the same standardisation as model fitting
(run_dream.py: mean/sd of the 'weight' column, pandas ddof=1). Posterior median
line + central 95% band, with a rug of the observed weights. No title.

Usage: python make_voracious_prob.py --model 2   ->  outputs/model_2a_voracious_prob.png
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

W = pd.read_csv('data/df_covariates_better_date.csv')['weight'].to_numpy()
WMEAN, WSD = W.mean(), W.std(ddof=1)                 # matches run_dream.py standardisation


def make(model_id):
    tag = f'model_{model_id}a'
    post = pd.read_csv(f'outputs/{tag}_posterior.csv')
    g0 = post['gamma0'].to_numpy()[:, None]
    gw = post['gamma_w'].to_numpy()[:, None]

    grid = np.linspace(W.min(), W.max(), 200)        # raw weight (g)
    z = (grid - WMEAN) / WSD
    pi = 1.0 / (1.0 + np.exp(-(g0 + gw * z[None, :])))   # (ndraw, ngrid)
    med = np.median(pi, axis=0)
    lo, hi = np.percentile(pi, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.fill_between(grid, lo, hi, color='tab:orange', alpha=0.25, label='95% credible interval')
    ax.plot(grid, med, color='tab:orange', lw=2.5, label='posterior median')
    # tercile boundaries (the light/moderate/heavy strata used in the PPCs)
    for q in np.percentile(W, [100/3, 200/3]):
        ax.axvline(q, ls=':', color='0.5', lw=1)
    ax.plot(W, np.full_like(W, -0.02), '|', color='0.3', ms=6, alpha=0.5)  # rug of observed weights
    ax.set_xlabel('fish weight (g)'); ax.set_ylabel('P(voracious)')
    ax.set_ylim(-0.04, 1.02); ax.set_xlim(W.min(), W.max())
    ax.legend(loc='lower right', frameon=True)
    fig.tight_layout()
    fig.savefig(f'outputs/{tag}_voracious_prob.png', dpi=120, bbox_inches='tight')
    print(f"saved outputs/{tag}_voracious_prob.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", type=int, default=2, choices=[1, 2])
    make(ap.parse_args().model)
