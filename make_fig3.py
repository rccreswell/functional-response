"""Reproduce Figure 3 (Newport et al.) from a fitted posterior.

Panels:
  A: time (min) vs EXPERIMENTAL condition, acclimation = Ambient, by duration
  B: time (min) vs ACCLIMATION condition, experiment = Ambient, by duration
  C: model-predicted # prey eaten (n0 = 40, median weight), by acclimation
     condition and duration (violin = predictive distribution, triangle = median)

The "time" quantity is whichever rate the conditions act on:
  Model 2 (a constant, conditions on r): handling time  = 60 / r  minutes
  Model 1 (conditions on a):             preying  time  = 60 / a  minutes
(rates are per one-hour experiment, matching the paper's panel-C assumption.)

Usage:  python make_fig3.py --model 1        (or 2)
"""
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import mechanistic_model as mm

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 12, "axes.labelsize": 13, "axes.linewidth": 1.1,
    "xtick.labelsize": 11, "ytick.labelsize": 12,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
})

N0_REF = 40                              # reference prey density (matches panel C)
DUMMY_COLS = ["f.acc_condH", "f.acc_condM", "f.acc_condT",
              "f.exp_condH", "f.exp_condM", "f.exp_condT", "f.week6",
              "f.acc_condH:f.week6", "f.acc_condM:f.week6", "f.acc_condT:f.week6"]
# condition code -> label, in figure order (top to bottom)
COND = [("C", "Ambient"), ("H", "Warmed"), ("T", "Turbid"), ("M", "Mixed")]
GREEN, ORANGE = "#4DA37E", "#D2691E"     # 1 week, 6 weeks
DUR = [(0, "1 week", GREEN), (1, "6 weeks", ORANGE)]


def rate_to_time(rate, model_id):
    """Minutes. Model 2: handling time 1/r. Model 1: realized time to the next
    capture at n0=N0_REF, i.e. 1/(a * N0_REF) (a is a per-available-prey rate)."""
    if model_id == 2:
        return 60.0 / rate
    return 60.0 / (rate * N0_REF)


def scenario_dummies(acc, exp, w):
    """Indicator dict for the 10 design dummies given condition codes + week6."""
    d = {c: 0 for c in DUMMY_COLS}
    for lvl, pre in [(acc, "f.acc_cond"), (exp, "f.exp_cond")]:
        if lvl in ("H", "M", "T"):
            d[f"{pre}{lvl}"] = 1
    d["f.week6"] = w
    for lvl in ("H", "M", "T"):
        if acc == lvl:
            d[f"f.acc_cond{lvl}:f.week6"] = w
    return d


def log_rate(post, model_id, rate, acc, exp, w, wz=0.0):
    """Vector (over posterior draws) of log a or log r for one scenario."""
    d = scenario_dummies(acc, exp, w)
    if rate == "a":
        val = post["a0"].to_numpy() if model_id == 1 else post["a0(const)"].to_numpy()
        if model_id == 1:                     # conditions load on a
            for c in DUMMY_COLS:
                val = val + post[f"a:{c}"].to_numpy() * d[c]
        return val                            # model 2: a constant
    val = post["r0"].to_numpy()
    if model_id == 2:                         # conditions load on r
        for c in DUMMY_COLS:
            val = val + post[f"r:{c}"].to_numpy() * d[c]
    val = val + post["r:weight_z"].to_numpy() * wz + post["r:weight_z2"].to_numpy() * wz ** 2
    return val


def make_figure(model_id):
    post = pd.read_csv(f"outputs/model_{model_id}_posterior.csv")
    # thin for panel C (CME solves); panels A/B use all draws (cheap)
    postC = post.sample(1200, random_state=0)
    if model_id == 2:
        rate_char, time_name = "r", "Handling time, minutes"
    else:
        rate_char, time_name = "a", f"Preying time (n0={N0_REF}), minutes"

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10))
    yoff = {0: 0.18, 1: -0.18}

    def panel_time(ax, vary):
        """vary='exp' -> Panel A (acc=Ambient); vary='acc' -> Panel B (exp=Ambient)."""
        for yi, (code, label) in enumerate(COND):
            y = len(COND) - 1 - yi
            for w, _, col in DUR:
                acc = code if vary == "acc" else "C"
                exp = code if vary == "exp" else "C"
                rate = np.exp(log_rate(post, model_id, rate_char, acc, exp, w))
                t = rate_to_time(rate, model_id)
                med = np.median(t); lo, hi = np.percentile(t, [2.5, 97.5])
                ax.plot([lo, hi], [y + yoff[w]] * 2, color=col, lw=2.0,
                        solid_capstyle='round', zorder=1)
                ax.plot(med, y + yoff[w], 'o', color=col, ms=8,
                        markeredgecolor='white', markeredgewidth=0.8, zorder=2)
        ax.set_yticks(range(len(COND)))
        ax.set_yticklabels([lab for _, lab in COND][::-1])
        ax.set_ylim(-0.6, len(COND) - 0.4)
        ax.set_xlabel(time_name)
        ax.margins(x=0.05)

    panel_time(axes[0], "exp")
    axes[0].set_ylabel("Experimental\ncondition")
    panel_time(axes[1], "acc")
    axes[1].set_ylabel("Acclimation\ncondition")

    # Panel C: predicted # eaten, n0=40, acclimation condition x duration, exp=Ambient
    ax = axes[2]
    n0 = 40
    for yi, (code, label) in enumerate(COND):
        y = len(COND) - 1 - yi
        for w, _, col in DUR:
            la = log_rate(postC, model_id, "a", code, "C", w)
            lr = log_rate(postC, model_id, "r", code, "C", w)
            counts = np.array(Parallel(n_jobs=-1, max_nbytes=None)(
                delayed(_sim_count)(np.exp(a), np.exp(r), n0, s)
                for s, (a, r) in enumerate(zip(la, lr))))
            parts = ax.violinplot([counts], positions=[y + yoff[w]], vert=False,
                                  widths=0.36, showextrema=False)
            for b in parts['bodies']:
                b.set_facecolor(col); b.set_alpha(0.85)
                b.set_edgecolor('0.25'); b.set_linewidth(0.9)
            ax.plot(np.median(counts), y + yoff[w], '^', color='k', ms=7,
                    markeredgecolor='white', markeredgewidth=0.6, zorder=3)
    ax.set_yticks(range(len(COND)))
    ax.set_yticklabels([lab for _, lab in COND][::-1])
    ax.set_ylim(-0.6, len(COND) - 0.4)
    ax.set_xlabel("# prey eaten")
    ax.set_ylabel("Acclimation\ncondition")

    # panel letters + legend
    for a, L in zip(axes, "ABC"):
        a.text(-0.14, 1.02, L, transform=a.transAxes, fontsize=15, fontweight='bold')
    handles = [plt.Line2D([], [], color=c, marker='s', ls='', ms=9, label=l)
               for _, l, c in DUR]
    fig.legend(handles=handles, title="Acclimation duration", loc='lower center',
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    mname = "Model 2 (conditions on digestion rate r)" if model_id == 2 \
        else "Model 1 (conditions on attack rate a)"
    fig.suptitle(f"Figure 3 equivalent — {mname}", y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    out = f"outputs/model_{model_id}_fig3.png"
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"saved {out}", flush=True)


def _sim_count(a, r, n0, seed):
    p = np.clip(mm.solve_system_simple_h_simplex(n0, a, r), 0, None)
    return np.random.RandomState(seed).choice(len(p), p=p / p.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True, choices=[1, 2])
    make_figure(ap.parse_args().model)
