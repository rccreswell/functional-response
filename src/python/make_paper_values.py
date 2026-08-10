"""Generate paper/generated_values.tex (\\newcommand macros) and
paper/tables/table_rope.tex, holding every headline number quoted in the Results
-- always WITH an uncertainty interval -- plus ROPE comparisons of handling-time
differences, computed reproducibly from the committed Model-2a artifacts.
article.tex \\input's the macros and uses them in prose. Deterministic.

ROPE: as defined in the Methods, we use a +/-1 minute region of practical
equivalence on handling time. For a difference Delta we report P(|Delta|<1)
(within ROPE) and the one-sided mass beyond it, concluding practical equivalence
if >=95% is within, or a meaningful difference if >=95% is beyond on one side.

Usage: python make_paper_values.py
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

POST = pd.read_csv('outputs/model_2a_posterior.csv')
RT = pd.read_csv('outputs/model_2a_results_table.csv')
BP = pd.read_csv('outputs/model_2a_bayes_pvalues.csv').set_index('statistic')
IJ = {c: i for i, c in enumerate(POST.columns)}
TH = POST.to_numpy()
CODE = {'Amb': 'C', 'Warm': 'H', 'Turb': 'T', 'Mix': 'M'}
WK = {'One': 0, 'Six': 1}
ROPE = 1.0                                                        # +/- 1 minute


def col(name): return TH[:, IJ[name]]


def ht_draws(cond, week, kind='acc'):
    """Handling time (min) per draw, median weight. kind='acc' varies the
    acclimation condition (experiment held at ambient); kind='exp' varies the
    experimental condition (acclimation held at ambient)."""
    v = col('r0').copy()
    if week == 1:
        v = v + col('r:f.week6')
    c = CODE[cond]
    if c != 'C':
        if kind == 'acc':
            v = v + col(f'r:f.acc_cond{c}')
            if week == 1:
                v = v + col(f'r:f.acc_cond{c}:f.week6')
        else:                                    # experimental condition (no week interaction)
            v = v + col(f'r:f.exp_cond{c}')
    return 60.0 / np.exp(v)


def ci(x): return np.percentile(x, [50, 2.5, 97.5])
def f1(x): return f"{x:.1f}"


M = {}
# --- handling time (all conditions x weeks), point + 95% CI ---------------
for cond in ['Amb', 'Warm', 'Turb', 'Mix']:
    for wk, w in WK.items():
        med, lo, hi = ci(ht_draws(cond, w))
        M[f'Ht{cond}{wk}'] = f1(med)
        M[f'Ht{cond}{wk}CI'] = f"{f1(lo)}--{f1(hi)}"

# --- reductions vs ambient (same week), turbid & mixed, point + CI --------
for cond in ['Turb', 'Mix']:
    for wk, w in WK.items():
        red = 100.0 * (1.0 - ht_draws(cond, w) / ht_draws('Amb', w))
        med, lo, hi = ci(red)
        M[f'Red{cond}{wk}'] = f"{med:.0f}"
        M[f'Red{cond}{wk}CI'] = f"{lo:.0f}--{hi:.0f}"

# --- prey eaten (n0=40), point + CI (from the deterministic results table) -
B = RT[RT.panel.str.startswith('B')]
for cond, lab in [('Amb', 'Ambient'), ('Warm', 'Warmed'), ('Turb', 'Turbid'), ('Mix', 'Mixed')]:
    for wk, dur in [('One', '1 week'), ('Six', '6 weeks')]:
        r = B[(B.condition == lab) & (B.duration == dur)].iloc[0]
        M[f'Eat{cond}{wk}'] = f"{r.eaten_n0_40:.0f}"
        M[f'Eat{cond}{wk}CI'] = f"{r.eaten_lo:.0f}--{r.eaten_hi:.0f}"

# --- voracious subpopulation ---------------------------------------------
idx = np.linspace(0, len(POST) - 1, 3000).astype(int)
W = pd.read_csv('data/df_covariates_better_date.csv')['weight'].to_numpy()
wz = (W - W.mean()) / W.std(ddof=1)
pi = 1.0 / (1.0 + np.exp(-(col('gamma0')[idx][:, None] + col('gamma_w')[idx][:, None] * wz[None, :])))
pf = np.percentile(pi.mean(1), [50, 2.5, 97.5])
zl, zh = (W.min() - W.mean()) / W.std(ddof=1), (W.max() - W.mean()) / W.std(ddof=1)
M['VoracFrac'] = f"{100 * pf[0]:.0f}\\%"
M['VoracFracCI'] = f"{100 * pf[1]:.0f}--{100 * pf[2]:.0f}\\%"
M['VoracLight'] = f"{100 * np.median(1 / (1 + np.exp(-(col('gamma0') + col('gamma_w') * zl)))):.0f}\\%"
M['VoracHeavy'] = f"{100 * np.median(1 / (1 + np.exp(-(col('gamma0') + col('gamma_w') * zh)))):.0f}\\%"
M['FullConsObs'] = f"{100 * BP.loc['full-consumption P(eaten=n0)', 'observed']:.0f}\\%"
# two-state model's predicted full-consumption rate (the misfit the mixture fixes)
SM = pd.read_csv('outputs/all_models_stat_means.csv').set_index('statistic')
M['FullConsPred'] = f"{100 * SM.loc['full-consumption P(eaten=n0)', 'model_2_mean']:.0f}\\%"
M['Nfish'] = f"{len(W)}"                                          # experimental trials analysed (data rows)

# --- ROPE comparisons on handling-time differences ------------------------
def contrast(d):
    p_in = float(np.mean(np.abs(d) < ROPE))
    p_hi = float(np.mean(d > ROPE)); p_lo = float(np.mean(d < -ROPE))
    if p_hi >= 0.95:   verdict, p = 'a meaningful increase', p_hi
    elif p_lo >= 0.95: verdict, p = 'a meaningful reduction', p_lo
    elif p_in >= 0.95: verdict, p = 'practical equivalence', p_in
    else:              verdict, p = 'no conclusive difference', max(p_in, p_hi, p_lo)
    med, lo, hi = np.percentile(d, [50, 2.5, 97.5])
    return dict(cell=f"${med:+.1f}$ [${lo:+.1f}$, ${hi:+.1f}$]", pin=p_in,
                verdict=verdict, pmac=("{>}0.99" if p > 0.995 else "{=}" + f"{p:.2f}"))

def stars(p_in):                                 # strength of evidence the difference is OUTSIDE the ROPE
    return '$^{***}$' if p_in <= 0.001 else '$^{**}$' if p_in <= 0.01 else '$^{*}$' if p_in <= 0.05 else ''

# Each contrast is condition-vs-ambient, evaluated as an experimental effect (vary the
# experimental condition, acclimation = ambient) and an acclimation effect (vary the
# acclimation condition, experiment = ambient), for both acclimation durations.
CONDS = [('Warm', 'Warmed'), ('Turb', 'Turbid'), ('Mix', 'Mixed')]
DURS = [('One', 0, '1 week'), ('Six', 1, '6 weeks')]

table_rows = []
for cc, clab in CONDS:
    for dk, w, wlab in DURS:
        exp = contrast(ht_draws(cc, w, 'exp') - ht_draws('Amb', w))
        acc = contrast(ht_draws(cc, w) - ht_draws('Amb', w))
        M[f'vRopeExp{cc}{dk}'] = exp['verdict']; M[f'pRopeExp{cc}{dk}'] = exp['pmac']
        M[f'vRope{cc}{dk}'] = acc['verdict']; M[f'pRope{cc}{dk}'] = acc['pmac']
        table_rows.append((f"{clab}, {wlab}", exp, acc))
dur = contrast(ht_draws('Warm', 1) - ht_draws('Warm', 0))   # acclimation duration contrast (no exp analogue)
M['vRopeWarmDur'] = dur['verdict']; M['pRopeWarmDur'] = dur['pmac']

with open('paper/generated_values.tex', 'w') as f:
    f.write("% AUTO-GENERATED by make_paper_values.py -- do not edit by hand.\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

with open('paper/tables/table_rope.tex', 'w') as f:
    f.write("\\begin{tabular}{lcc}\n\\toprule\n")
    f.write(" & \\multicolumn{2}{c}{$\\Delta$ handling time (min), median [95\\% CI]} \\\\\n")
    f.write("\\cmidrule(l){2-3}\n")
    f.write("Contrast (vs.\\ ambient) & Experimental effect & Acclimation effect \\\\\n\\midrule\n")
    for lab, exp, acc in table_rows:
        f.write(f"{lab} & {exp['cell']}{stars(exp['pin'])} & {acc['cell']}{stars(acc['pin'])} \\\\\n")
    f.write("\\midrule\n")
    f.write(f"Warmed acclimation, 6 vs.\\ 1 week & --- & {dur['cell']}{stars(dur['pin'])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("saved paper/generated_values.tex and paper/tables/table_rope.tex")
for k, v in M.items():
    print(f"  \\{k:14s} {v}")
