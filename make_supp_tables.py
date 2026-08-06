"""Generate LaTeX tables from the committed result CSVs (deterministic):
  - outputs/table_loo.tex        : 4-model PSIS-LOO comparison (from loo_comparison.csv)
  - outputs/table_bayes_p.tex    : posterior-predictive discrepancy p-values (Model 2a)
Usage: python make_supp_tables.py
"""
import pandas as pd

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

bp = pd.read_csv('outputs/model_2a_bayes_pvalues.csv')
LAB = {'full-consumption P(eaten=n0)': 'Full consumption, $P(\\text{eaten}=N_0)$',
       'zero-consumption P(eaten=0)': 'Zero consumption, $P(\\text{eaten}=0)$',
       'overdispersion sum Var(eaten|n0)': 'Dispersion, $\\sum_{N_0}\\mathrm{Var}(\\text{eaten})$',
       'mean eaten': 'Mean eaten',
       'full-consumption (light weight)': 'Full consumption (light fish)',
       'full-consumption (moderate weight)': 'Full consumption (moderate fish)',
       'full-consumption (heavy weight)': 'Full consumption (heavy fish)'}
with open('outputs/table_bayes_p.tex', 'w') as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
    f.write("Discrepancy statistic & Observed & Model 95\\% interval & $p_B$ \\\\\n\\midrule\n")
    for _, r in bp.iterrows():
        f.write(f"{LAB.get(r['statistic'], r['statistic'])} & {r['observed']:.3f} & "
                f"[{r['model_2.5%']:.3f}, {r['model_97.5%']:.3f}] & {r['bayes_p_2sided']:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("saved outputs/table_loo.tex and outputs/table_bayes_p.tex")
