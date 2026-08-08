"""Generate LaTeX tables from the committed result CSVs (deterministic).

Tables that the manuscript \\input's are written under paper/tables/ (so the
paper is self-contained); reference-only tables go to outputs/.
  - outputs/table_loo.tex             : 4-model PSIS-LOO comparison (reference; Table 1 is inlined)
  - paper/tables/table_bayes_p.tex    : posterior-predictive discrepancy p-values, all 4 models
                                        (article S4; needs outputs/all_models_bayes_pvalues.csv from
                                        make_ppc_table_all.py)
  - paper/tables/table_params_2a.tex  : Model 2a parameter estimates (article S3)
  - paper/tables/table_params_1a.tex  : Model 1a parameter estimates (article S3)
Usage: python make_supp_tables.py
"""
import os
import numpy as np, pandas as pd, arviz as az

TABLES = 'paper/tables'
os.makedirs(TABLES, exist_ok=True)


def _pretty(col):
    """Human-readable label for a posterior column name."""
    if col in ('a0', 'a0(const)'):
        return r'intercept ($\log a$)'
    if col == 'r0':
        return r'intercept ($\log r$)'
    if col == 'gamma0':
        return r'mixture intercept $\gamma_0$'
    if col == 'gamma_w':
        return r'mixture weight $\gamma_w$'
    rate = ''
    if col.startswith('a:'):
        rate, col = r' ($\log a$)', col[2:]
    elif col.startswith('r:'):
        rate, col = r' ($\log r$)', col[2:]
    term = {'f.acc_condH': 'acclim: warmed', 'f.acc_condM': 'acclim: mixed',
            'f.acc_condT': 'acclim: turbid', 'f.exp_condH': 'exp: warmed',
            'f.exp_condM': 'exp: mixed', 'f.exp_condT': 'exp: turbid',
            'f.week6': '6 weeks', 'weight_z': 'weight', 'weight_z2': r'weight$^2$',
            'f.acc_condH:f.week6': r'acclim warmed $\times$ 6 wk',
            'f.acc_condM:f.week6': r'acclim mixed $\times$ 6 wk',
            'f.acc_condT:f.week6': r'acclim turbid $\times$ 6 wk'}.get(col, col)
    return term + rate


def param_table(tag, outpath):
    df = pd.read_csv(f'outputs/{tag}_posterior.csv')
    params = [c for c in df.columns if c not in ('chain', 'iter')]
    chains = sorted(df['chain'].unique())
    n = min((df['chain'] == c).sum() for c in chains)          # equal-length for R-hat
    with open(outpath, 'w') as f:
        f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n")
        f.write("Parameter & Mean & SD & 2.5\\% & 25\\% & 50\\% & 75\\% & 97.5\\% & $\\hat{R}$ \\\\\n\\midrule\n")
        for p in params:
            v = df[p].to_numpy()
            arr = np.stack([df.loc[df.chain == c, p].to_numpy()[:n] for c in chains])
            rhat = float(az.rhat(arr))
            q = np.percentile(v, [2.5, 25, 50, 75, 97.5])
            f.write(f"{_pretty(p)} & {v.mean():.3f} & {v.std():.3f} & "
                    f"{q[0]:.3f} & {q[1]:.3f} & {q[2]:.3f} & {q[3]:.3f} & {q[4]:.3f} & {rhat:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved {outpath}")

# Per-model structure: covariates on the attack rate a, covariates on the handling
# rate r, and whether a voracious-mixture component is included. "Treatments" =
# acclimation x experimental condition x week (+ weight, weight$^2$).
CONST, TRT = 'constant', 'treatments + weight'
SPEC = {'model_2':  dict(a=CONST, r=TRT,   mix='No'),
        'model_1':  dict(a=TRT,   r=CONST, mix='No'),
        'model_2a': dict(a=CONST, r=TRT,   mix='Yes'),
        'model_1a': dict(a=TRT,   r=CONST, mix='Yes')}

loo = pd.read_csv('outputs/loo_comparison.csv', index_col=0).sort_values('rank')
with open('outputs/table_loo.tex', 'w') as f:
    f.write("\\begin{tabular}{lllrrrr}\n\\toprule\n")
    f.write("Covariates on $a$ & Covariates on $r$ & Mixture & "
            "$\\widehat{\\mathrm{elpd}}_{\\mathrm{LOO}}$ & SE & $\\Delta$elpd & SE($\\Delta$) \\\\\n\\midrule\n")
    for m, r in loo.iterrows():
        s = SPEC.get(m, dict(a='?', r='?', mix='?'))
        d = "--" if r['rank'] == 0 else f"{r['d_loo']:.0f}"
        dse = "--" if r['rank'] == 0 else f"{r['dse']:.0f}"
        f.write(f"{s['a']} & {s['r']} & {s['mix']} & "
                f"{r['elpd_loo']:.0f} & {r['se']:.0f} & {d} & {dse} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# All-four-model discrepancy p-value table, transposed so MODELS are rows and
# identified with the SAME descriptor columns as the LOO table (make_ppc_table_all.py).
bp = pd.read_csv('outputs/all_models_bayes_pvalues.csv').set_index('statistic')
OVERALL = [('full-consumption P(eaten=n0)', r'$P(\text{eaten}{=}N_0)$'),
           ('zero-consumption P(eaten=0)',  r'$P(\text{eaten}{=}0)$'),
           ('overdispersion sum Var(eaten|n0)', 'Disp.'),
           ('mean eaten', 'Mean')]
TERT = [('full-consumption (light weight)', 'Light'),
        ('full-consumption (moderate weight)', 'Moderate'),
        ('full-consumption (heavy weight)', 'Heavy')]
ALLSTAT = OVERALL + TERT
lead = 3 + len(OVERALL)                          # 3 descriptor cols + overall stat cols
with open(f'{TABLES}/table_bayes_p.tex', 'w') as f:
    f.write("\\begin{tabular}{lll" + "r" * len(ALLSTAT) + "}\n\\toprule\n")
    f.write("&" * lead + "\\multicolumn{%d}{c}{Full consumption by weight} \\\\\n" % len(TERT))
    f.write("\\cmidrule(l){%d-%d}\n" % (lead + 1, lead + len(TERT)))
    f.write("Covariates on $a$ & Covariates on $r$ & Mixture & "
            + " & ".join(h for _, h in ALLSTAT) + " \\\\\n\\midrule\n")
    obs = " & ".join((f"{bp.loc[s, 'observed']:.1f}" if bp.loc[s, 'observed'] >= 100
                      else f"{bp.loc[s, 'observed']:.3f}") for s, _ in ALLSTAT)
    f.write("\\multicolumn{3}{l}{\\textit{Observed value}} & " + obs + " \\\\\n\\midrule\n")
    for m in loo.index:                          # LOO rank order (best first)
        s = SPEC[m]
        pv = " & ".join(f"{bp.loc[stat, f'{m}_p']:.2f}" for stat, _ in ALLSTAT)
        f.write(f"{s['a']} & {s['r']} & {s['mix']} & {pv} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
    f.write("% rows below the observed line are two-sided posterior-predictive p-values\n")

param_table('model_2a', f'{TABLES}/table_params_2a.tex')
param_table('model_1a', f'{TABLES}/table_params_1a.tex')
print(f"saved outputs/table_loo.tex and {TABLES}/table_{{bayes_p,params_1a,params_2a}}.tex")
