"""Deterministically compute the Model-2a summary quantities quoted in the
Results (handling time and predicted prey eaten, by condition x acclimation
duration) and write them to a reproducible table (CSV + LaTeX).

Mirrors Figure 3: Panel A = handling time by EXPERIMENTAL condition (acclimation
held at Ambient); Panel B = handling time by ACCLIMATION condition (experiment
held at Ambient), plus the predicted number of prey eaten at N0=40 (Panel C).
All for a fish of median (z-scored) weight. Handling time = 60/r minutes;
predicted eaten is the mixture expected count (ordinary CME + voracious).

Usage: python make_results_table.py   ->  outputs/model_2a_results_table.{csv,tex}
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import make_fig3 as F3
import mechanistic_model as mm
from mixture_model import voracious_pmf

MODEL, TAG, N0_PRED, WZ, N_SUB = 2, 'model_2a', 40, 0.0, 1000
COND = [('C', 'Ambient'), ('H', 'Warmed'), ('T', 'Turbid'), ('M', 'Mixed')]
DUR = [(0, '1 week'), (1, '6 weeks')]
post = pd.read_csv(f'outputs/{TAG}_posterior.csv')
IDX = np.linspace(0, len(post) - 1, N_SUB).astype(int)      # deterministic thinning


def med_ci(x):
    return np.median(x), np.percentile(x, 2.5), np.percentile(x, 97.5)


def handling_draws(acc, exp, sub=None):
    p = post if sub is None else post.iloc[IDX]
    return 60.0 / np.exp(F3.log_rate(p, MODEL, 'r', acc, exp, 0, WZ))   # minutes, per draw


def _Ecount(logr, loga, g0, gw):
    r, a = np.exp(logr), np.exp(loga)
    pi = 1.0 / (1.0 + np.exp(-(g0 + gw * WZ)))
    Ev = np.dot(np.arange(N0_PRED + 1), voracious_pmf(N0_PRED, r))
    p = np.clip(mm.solve_system_simple_h_simplex(N0_PRED, a, r), 0, None); p /= p.sum()
    Ec = np.dot(np.arange(N0_PRED + 1), p)
    return (1 - pi) * Ec + pi * Ev


def eaten_draws(acc, exp):
    sub = post.iloc[IDX]
    lr = F3.log_rate(sub, MODEL, 'r', acc, exp, 0, WZ)
    la = F3.log_rate(sub, MODEL, 'a', acc, exp, 0, WZ)
    g0, gw = sub['gamma0'].to_numpy(), sub['gamma_w'].to_numpy()
    return np.array(Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(_Ecount)(lr[i], la[i], g0[i], gw[i]) for i in range(len(sub))))


rows = []
# Panel B: acclimation condition (experiment = Ambient) -- handling time, eaten, % vs ambient
for code, lab in COND:
    for w, wlab in DUR:
        ht = 60.0 / np.exp(F3.log_rate(post, MODEL, 'r', code, 'C', w, WZ))
        ht_a = 60.0 / np.exp(F3.log_rate(post, MODEL, 'r', 'C', 'C', w, WZ))
        pct = 100.0 * (ht / ht_a - 1.0)          # per-draw % change vs ambient (same week)
        sub = post.iloc[IDX]
        lr = F3.log_rate(sub, MODEL, 'r', code, 'C', w, WZ)
        la = F3.log_rate(sub, MODEL, 'a', code, 'C', w, WZ)
        g0, gw = sub['gamma0'].to_numpy(), sub['gamma_w'].to_numpy()
        E = np.array(Parallel(n_jobs=-1, max_nbytes=None)(
            delayed(_Ecount)(lr[i], la[i], g0[i], gw[i]) for i in range(len(sub))))
        hm, hl, hh = med_ci(ht); pm, pl, ph = med_ci(pct); em, el, eh = med_ci(E)
        rows.append(dict(panel='B: acclimation (exp=Ambient)', condition=lab, duration=wlab,
                         handling_min=round(hm, 2), handling_lo=round(hl, 2), handling_hi=round(hh, 2),
                         pct_vs_ambient=round(pm, 1), pct_lo=round(pl, 1), pct_hi=round(ph, 1),
                         eaten_n0_40=round(em, 1), eaten_lo=round(el, 1), eaten_hi=round(eh, 1)))

# Panel A: experimental condition (acclimation = Ambient) -- handling time only
for code, lab in COND:
    for w, wlab in DUR:
        ht = 60.0 / np.exp(F3.log_rate(post, MODEL, 'r', 'C', code, w, WZ))
        hm, hl, hh = med_ci(ht)
        rows.append(dict(panel='A: experimental (acc=Ambient)', condition=lab, duration=wlab,
                         handling_min=round(hm, 2), handling_lo=round(hl, 2), handling_hi=round(hh, 2),
                         pct_vs_ambient=np.nan, pct_lo=np.nan, pct_hi=np.nan,
                         eaten_n0_40=np.nan, eaten_lo=np.nan, eaten_hi=np.nan))

df = pd.DataFrame(rows)
df.to_csv('outputs/model_2a_results_table.csv', index=False)

# LaTeX (Panel B, the main results table)
b = df[df.panel.str.startswith('B')]
with open('outputs/model_2a_results_table.tex', 'w') as f:
    f.write("\\begin{tabular}{llrrr}\n\\toprule\n")
    f.write("Acclimation & Duration & Handling time (min) & \\% vs ambient & Predicted eaten ($N_0{=}40$) \\\\\n\\midrule\n")
    for _, r in b.iterrows():
        pct = "--" if r.condition == 'Ambient' else f"{r.pct_vs_ambient:.0f} [{r.pct_lo:.0f}, {r.pct_hi:.0f}]"
        f.write(f"{r.condition} & {r.duration} & {r.handling_min:.1f} [{r.handling_lo:.1f}, {r.handling_hi:.1f}] "
                f"& {pct} & {r.eaten_n0_40:.0f} [{r.eaten_lo:.0f}, {r.eaten_hi:.0f}] \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print(df.to_string(index=False))
print("\nsaved outputs/model_2a_results_table.csv and .tex")
