"""Prior sensitivity check for Model 2a via PSIS importance-sampling reweighting
(no refit). Each alternative prior only WIDENS or SHIFTS the primary prior. The
existing outputs/model_2a_posterior.csv draws are reweighted by the Gaussian prior
ratio w_i = p_new(theta_i)/p_old(theta_i), PSIS-smoothed (arviz), and quantities
are recomputed as weighted posterior summaries. Reliable only where the Pareto-k
diagnostic is < 0.7 (else that scenario needs a full refit).

Writes (all new; nothing overwritten):
  outputs/sensitivity/sensitivity_summary.csv       full weighted median [95% CI] table
  outputs/sensitivity/sensitivity_forest.png        headline quantities x scenario
  outputs/sensitivity/sensitivity_panelA.png        Fig-3A (exp condition) under each prior
  outputs/sensitivity/sensitivity_panelB.png        Fig-3B (acc condition) under each prior
  paper/tables/table_prior_sensitivity.tex          compact LaTeX table for the supplement

Usage: python prior_sensitivity.py
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, arviz as az
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import run_dream_mixture as M

OUT = 'outputs/sensitivity'; os.makedirs(OUT, exist_ok=True)
TAB = 'paper/tables'; os.makedirs(TAB, exist_ok=True)
POST = pd.read_csv('outputs/model_2a_posterior.csv')
X_a, X_r, LAB = M.design_mix(2)
N_A, N_R = X_a.shape[1], X_r.shape[1]
TH = POST[LAB].to_numpy()
IJ = {l: i for i, l in enumerate(LAB)}

M0, V0 = M.base_mean_var(N_A, N_R)
M0 = np.concatenate([M0, [-1.0, 0.0]]); V0 = np.concatenate([V0, [1.5 ** 2, 1.5 ** 2]])
SLOPE_IDX = [i for i in range(N_A + N_R) if i not in (0, N_A)]
GA0, GW = N_A + N_R, N_A + N_R + 1

W = pd.read_csv('data/df_covariates_better_date.csv')['weight'].to_numpy()
WMEAN, WSD = W.mean(), W.std(ddof=1)
Z_LIGHT = (np.percentile(W, 10) - WMEAN) / WSD
Z_HEAVY = (np.percentile(W, 90) - WMEAN) / WSD

COND = [('C', 'Ambient'), ('H', 'Warmed'), ('T', 'Turbid'), ('M', 'Mixed')]
WEEKS = [(0, '1 week', 'tab:green'), (1, '6 weeks', 'tab:orange')]

SCENARIOS = {
    'primary':     dict(),
    'gamma0_lo':   dict(gamma0_mean=-2.0),
    'gamma0_hi':   dict(gamma0_mean=0.0),
    'gamma_wide':  dict(gamma_sd=2.5),
    'slopes_wide': dict(slope_num=3.0),
    'cr_wide':     dict(sd_r=1.0),
    'cr_lo':       dict(log_r0=10.0),
    'cr_hi':       dict(log_r0=30.0),
    'vague_all':   dict(slope_abs=3.0, sd_r=1.0, sd_a=2.0, gamma_sd=2.5),
}
DESC = {'primary': 'Primary prior', 'gamma0_lo': r'$\gamma_0$ mean $-2$',
        'gamma0_hi': r'$\gamma_0$ mean $0$', 'gamma_wide': r'$\gamma$ sd $2.5$',
        'slopes_wide': r'Slope sd $3/\sqrt{p}$', 'cr_wide': r'$c_r$ sd $1.0$',
        'cr_lo': r'$c_r$ mean $\log 10$', 'cr_hi': r'$c_r$ mean $\log 30$',
        'vague_all': 'All widened'}


def make_prior(log_r0=None, sd_r=None, sd_a=None, slope_num=None, slope_abs=None,
               gamma0_mean=None, gamma_sd=None):
    m, v = M0.copy(), V0.copy()
    if sd_a is not None:        v[0] = sd_a ** 2
    if log_r0 is not None:      m[N_A] = np.log(log_r0)
    if sd_r is not None:        v[N_A] = sd_r ** 2
    if slope_num is not None:   v[SLOPE_IDX] = (slope_num / np.sqrt(N_R - 1)) ** 2
    if slope_abs is not None:   v[SLOPE_IDX] = slope_abs ** 2
    if gamma0_mean is not None: m[GA0] = gamma0_mean
    if gamma_sd is not None:    v[[GA0, GW]] = gamma_sd ** 2
    return m, v


def _glp(x, m, v): return -0.5 * np.log(2 * np.pi * v) - (x - m) ** 2 / (2 * v)


def _weights(over):
    if not over:
        return np.full(len(TH), 1.0 / len(TH)), 0.0, 1.0
    m1, v1 = make_prior(**over)
    lw = (_glp(TH, m1, v1) - _glp(TH, M0, V0)).sum(axis=1)
    lw_s, k = az.psislw(lw.copy())
    w = np.exp(lw_s - lw_s.max()); w /= w.sum()
    return w, float(np.asarray(k).ravel()[0]), 1.0 / np.sum(w ** 2) / len(TH)


WSCEN = {n: _weights(o) for n, o in SCENARIOS.items()}          # precompute once


def wq(x, w, q):
    o = np.argsort(x); xs, ws = x[o], w[o]; c = np.cumsum(ws) - 0.5 * ws
    return np.interp(q, c / c[-1], xs)


def logistic(z): return 1.0 / (1.0 + np.exp(-z))


def ht_draws(vary, code, wk):
    """Handling time (min) per draw for a scenario cell, median weight (z=0)."""
    val = TH[:, IJ['r0']].copy()
    if wk == 1:
        val = val + TH[:, IJ['r:f.week6']]
    if code != 'C':
        if vary == 'exp':
            val = val + TH[:, IJ[f'r:f.exp_cond{code}']]
        else:
            val = val + TH[:, IJ[f'r:f.acc_cond{code}']]
            if wk == 1:
                val = val + TH[:, IJ[f'r:f.acc_cond{code}:f.week6']]
    return 60.0 / np.exp(val)


# ---- (1) full summary table (all quantities) -----------------------------
def quantities():
    r0 = TH[:, IJ['r0']]; g0, gw = TH[:, GA0], TH[:, GW]
    return {'pi (median wt)': logistic(g0), 'pi (light wt)': logistic(g0 + gw * Z_LIGHT),
            'pi (heavy wt)': logistic(g0 + gw * Z_HEAVY),
            'handling warm 6wk': ht_draws('acc', 'H', 1), 'handling ambient 1wk': ht_draws('acc', 'C', 0),
            'warmed x wk6 coef': TH[:, IJ['r:f.acc_condH:f.week6']],
            '% turbid 1wk vs amb': 100.0 * (ht_draws('acc', 'T', 0) / ht_draws('acc', 'C', 0) - 1.0)}


Q = quantities()
rows = []
for name in SCENARIOS:
    w, k, essf = WSCEN[name]
    row = {'scenario': name, 'pareto_k': round(k, 2), 'ess_frac': round(essf, 3)}
    for qn, qv in Q.items():
        row[qn] = f"{wq(qv, w, 0.5):.2f} [{wq(qv, w, 0.025):.2f}, {wq(qv, w, 0.975):.2f}]"
    rows.append(row)
pd.DataFrame(rows).to_csv(f'{OUT}/sensitivity_summary.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))

# ---- (2) compact LaTeX table for the supplement --------------------------
def cell(qv, w): return f"{wq(qv, w, 0.5):.2f} [{wq(qv, w, 0.025):.2f}, {wq(qv, w, 0.975):.2f}]"
with open(f'{TAB}/table_prior_sensitivity.tex', 'w') as f:
    f.write("\\begin{tabular}{lrccc}\n\\toprule\n")
    f.write("Prior scenario & $\\hat{k}$ & $\\pi$ (median wt) & "
            "Handling, warm 6\\,wk (min) & Warmed$\\times$6\\,wk coef. \\\\\n\\midrule\n")
    for name in SCENARIOS:
        w, k, _ = WSCEN[name]
        f.write(f"{DESC[name]} & {k:.2f} & {cell(Q['pi (median wt)'], w)} & "
                f"{cell(Q['handling warm 6wk'], w)} & {cell(Q['warmed x wk6 coef'], w)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---- (3) headline forest -------------------------------------------------
qn = list(Q.keys()); names = list(SCENARIOS.keys())
fig, axes = plt.subplots(1, len(qn), figsize=(3.0 * len(qn), 4.2), sharey=True)
for ax, q in zip(axes, qn):
    for yi, nm in enumerate(names):
        w = WSCEN[nm][0]; y = len(names) - 1 - yi
        col = 'tab:red' if nm == 'primary' else '0.25'
        ax.plot([wq(Q[q], w, .025), wq(Q[q], w, .975)], [y, y], color=col, lw=2)
        ax.plot(wq(Q[q], w, .5), y, 'o', color=col, ms=5)
    ax.set_title(q, fontsize=9)
axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels(names[::-1], fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/sensitivity_forest.png', dpi=120, bbox_inches='tight')

# ---- (4) Panel A / B reproduced under each prior -------------------------
def panel_grid(vary, letter, fname):
    names = list(SCENARIOS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    for ax, nm in zip(axes.ravel(), names):
        w = WSCEN[nm][0]
        for yi, (code, lab) in enumerate(COND):
            y = len(COND) - 1 - yi
            for wk, wl, col in WEEKS:
                off = 0.18 if wk == 0 else -0.18
                ht = ht_draws(vary, code, wk)
                med, lo, hi = wq(ht, w, .5), wq(ht, w, .025), wq(ht, w, .975)
                ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.8, solid_capstyle='round')
                ax.plot(med, y + off, 'o', color=col, ms=5, mec='white', mew=0.6)
        ax.set_yticks(range(len(COND))); ax.set_yticklabels([l for _, l in COND][::-1], fontsize=8)
        ax.set_title(DESC[nm], fontsize=9); ax.set_ylim(-0.6, len(COND) - 0.4)
        if nm == 'primary':
            for s in ax.spines.values():
                s.set_edgecolor('tab:red'); s.set_linewidth(1.6)
    for ax in axes[-1]:
        ax.set_xlabel('Handling time (min)', fontsize=9)
    h = [plt.Line2D([], [], color=c, marker='o', ls='-', label=l) for _, l, c in WEEKS]
    fig.legend(handles=h, title='Acclimation duration', loc='upper center', ncol=2, frameon=False)
    ylab = 'Experimental condition' if vary == 'exp' else 'Acclimation condition'
    fig.supylabel(ylab, fontsize=10)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    print(f"saved {fname}")


panel_grid('exp', 'A', f'{OUT}/sensitivity_panelA.png')
panel_grid('acc', 'B', f'{OUT}/sensitivity_panelB.png')
print(f"saved {OUT}/sensitivity_summary.csv, forest, panels; {TAB}/table_prior_sensitivity.tex")
