# Functional response
This repository contains resources for rerunning functional response-type analyses underpinning Newport et al. (2026), "Heatwaves and turbidity influxes have interacting and lasting effects on foraging behaviour of three-spined sticklebacks _Gasterosteus aculeatus_".

## Reproducibility

The analysis is split into two tiers:

- **Tier A — fast & fully reproducible:** everything downstream of the fitted posteriors (all summaries, figures, tables, and the numbers quoted in the paper). This is driven by a `Snakefile` and reproduces bit-for-bit given the committed posterior CSVs (seeds/thinning are fixed).
- **Tier B — the DREAM fits (`data → posteriors`):** expensive (hours) and *not* bitwise-reproducible across machines (parallelism + BLAS ordering), so the posterior CSVs are treated as **versioned inputs**. See *Python: inference* for how to (re)fit.

Pinned environment (versions match the machine the results were produced on — Python 3.9.7):
```bash
conda env create -f environment.yml     # numpy 1.20.3, scipy 1.7.1, pandas 1.3.4,
conda activate functional-response       # matplotlib 3.4.3, joblib 1.1.0, pints 0.5.0, arviz 0.11.2
# optional exact cross-platform lock:  conda-lock -f environment.yml
```

Regenerate every paper artifact from the committed posteriors:
```bash
snakemake -c1 -n        # dry run: show the DAG
snakemake -c1           # build all tables/figures/values in paper/
snakemake -c1 --use-conda   # ... each rule in the pinned env
```
The pipeline writes `paper/figures/*.png`, `paper/tables/*.tex`, and `paper/generated_values.tex` — a set of `\newcommand` macros holding every number quoted in the Results, each with its uncertainty interval (plus the ROPE comparisons). `article.tex` `\input`s that file and uses the macros; rerun `make_paper_values.py` (or `snakemake`) and the paper's numbers update themselves. Build the PDF locally with your usual LaTeX toolchain (`latexmk -pdf paper/article.tex`).

**Tier B (refitting).** The DREAM fits live in a separate `Snakefile.fit` file. It overwrites the committed posteriors (commit/back up first) and is stochastic (reproduces only up to Monte Carlo error):
```bash
snakemake -s Snakefile.fit -c1 fit_all                    # refit all four models
snakemake -s Snakefile.fit -c1 --config n_iter=30000      # override run length
```
These rules do a single long run per model (simpler for the DAG); the production runs instead accumulated iterations with `--resume` (see below). Check the printed R-hat / ESS table and raise `n_iter` if not yet converged.

## Models
A stochastic reaction (chemical master equation) model of a predator that switches between a *preying* state (searching, at attack rate `a`) and a *digesting* state (at rate `r`, the inverse of handling time `h`). Both rates depend on covariates through a log link, `log a = a0 + beta_a . x_a` and `log r = r0 + beta_r . x_r`.

- **Model 1:** `r` is a function of `weight` and `weight^2`; `a` is a function of all other covariates.
- **Model 2:** `r` is a function of all covariates; `a` is a constant.

## Python: inference

**`run_dream.py`** is reponsible for inference (below).

The priors are the same in both (`build_prior` in `run_dream.py` mirrors the notebook): intercepts calibrated so the baseline predator eats a realistic fraction of prey (`a0 ~ Normal(log 4, 1)`, `r0 ~ Normal(log 20, 0.5)`), and weakly-informative, "p-aware" slope priors (`Normal(0, (0.75/sqrt(p))^2)`, `p` = number of covariates on that rate) so the total covariate-driven spread is stable regardless of covariate count. Weight is z-scored; condition/week terms are 0/1 dummies.

### Reproducing the inference

**Requirements:** Python 3.9+ with `pints`, `numpy`, `pandas`, `scipy`, `matplotlib`, `joblib`. Run every command from the repository root; `data/df_covariates_better_date.csv` must be present.

**1. Prior-predictive checks** (recommended before committing to MCMC):

```
python run_dream.py --prior-predictive
```

Writes `outputs/model_<id>_prior_predictive.png` — a grid of panels, one per initial prey count `n0`, comparing the prior-predictive distribution of prey eaten against the data. The data should sit inside the prior band in every panel.

**2. Run the fits** (DREAM = differential-evolution adaptive Metropolis, which mixes this correlated posterior where random-walk Metropolis stalls):

```
python run_dream.py --models 1,2 --n-iter 4000
```

For each model this runs a short CMA-ES optimisation to a high-posterior point, then starts 40 DREAM chains in a tight Gaussian ball (s.d. 0.3 per coordinate) around that point — note these are **not** over-dispersed starts, so `R-hat` is a weaker convergence check than under the usual over-dispersed-initialisation recommendation. Chains are written incrementally to per-chain CSV chunks `outputs/model_<id>_dream_c<k>_<chain>.csv`. On completion it saves the pooled posterior (`outputs/model_<id>_dream.npy`), prints an R-hat / ESS / 95%-CI table, and writes the single consolidated CSV (step 3).

**Pooling note.** The reported posterior (`pooled_posterior`) is assembled heuristically: it drops chunk 0 (the CMA-ES-seeded cold run) in full, discards the first 500 iterations of the first warm chunk, and concatenates the remainder. It does **not** re-discard a transient at each subsequent chunk boundary, so it assumes the warm-started chains stay at stationarity across resumes.

Long runs can be interrupted. `--resume` starts a **new** DREAM run using the final states of the previous run as its initial population, and appends a new chunk. This is *not* a checkpoint/restart of the sampler: a fresh `DreamMCMC` object is constructed each time, so DREAM's internal adaptation and differential-evolution history are **not** carried over across chunks. You accumulate iterations across launches (aim for **R-hat < 1.1**):

```
python run_dream.py --models 1,2 --resume --n-iter 2000
```

Notes:
- On macOS, prefix with `caffeinate -is` so the machine cannot sleep mid-run; for multi-hour runs, detach with `nohup ... &`.
- **Model 1 mixes more slowly than Model 2** because its covariates act on the weakly-identified attack rate `a`; it typically needs more `--resume` passes to reach R-hat < 1.1.

**3. Consolidate to a single CSV** — write the full pooled posterior of each model to one tidy CSV, `outputs/model_<id>_posterior.csv` (columns: `chain, iter,` then one column per parameter):

```
python run_dream.py --consolidate --models 1,2
```

(This also runs automatically at the end of a completed fit in step 2.) Coefficients are on the log scale; parameter columns are named by block, e.g. `a0`, `a:f.acc_condH`, ..., `r0`, `r:weight_z`, `r:weight_z2`.

**4. (Optional) posterior-predictive check** — the same per-`n0` grid as step 1 but drawing from the fitted posterior is produced by the analysis in `outputs/`; see `_final_report.py` for the pooling + plotting used to generate the trace grids and posterior-predictive figures.

**5. Figure 3 (handling / preying time + predicted prey eaten):**

```
python make_fig3.py --model 2      # handling time (1/r) by condition
python make_fig3.py --model 1      # preying time (1/a at n0=40) by condition
```

Reads `outputs/model_<id>_posterior.csv` and writes `outputs/model_<id>_fig3.png` — three panels: (A) time vs experimental condition (acclimation held at Ambient), (B) time vs acclimation condition (experiment held at Ambient), (C) model-predicted number of prey eaten (n0 = 40, median weight), all stratified by acclimation duration (1 vs 6 weeks). Because the conditions act on the digestion rate `r` in Model 2 and the attack rate `a` in Model 1, panels A/B show **handling time = 60/r** minutes for Model 2 and **preying time = 60/(a·40)** minutes (realised time to the next capture at n0 = 40) for Model 1.

**6. Posterior-predictive check (actual vs simulated):**

```
python make_ppc_violin.py --model 2      # or --model 1
```

Writes `outputs/model_<id>_ppc_violin.png` — paired violins of the number of prey eaten (green = actual data, orange = model-simulated from the fitted posterior) at each initial prey density, faceted by body-weight tertile (light / moderate / heavy). Add `--facet acc` or `--facet exp` to instead facet by acclimation or experimental condition.

Quantitative checks and calibration:

```
python ppc_checks.py --models 1,2
```

Writes, per model, `outputs/model_<id>_ppc_discrepancy.png` (discrepancy PPCs with Bayesian p-values for the full-consumption rate, zero-consumption rate, and within-`n0` overdispersion) and `outputs/model_<id>_loo_pit.png` (LOO-PIT calibration; uniform = calibrated, ∪-shape = under-dispersed). Requires `arviz`.

**7. Model comparison (PSIS-LOO):**

```
python run_dream.py --loo --models 1,2
```

Builds the pointwise log-likelihood from the pooled posteriors and prints per-model PSIS-LOO (`elpd_loo`, `p_loo`, Pareto-k diagnostics) plus an `az.compare` ranking. Requires `arviz`. Note that a high `p_loo` / bad Pareto-k values here flag influential "ate everything" observations, so the comparison should be read with care.

## Voracious-mixture models (1a / 2a)

Posterior-predictive checks on the base models showed they badly under-predict the number of fish that eat *all* their prey (the "full-consumption" spike). Models **1a** and **2a** add a two-component mixture: a fish is, with a weight-dependent probability, a *voracious* forager whose attack is effectively instantaneous (the `a→∞` limit, so eaten `= min(n0, 1 + Poisson(r))`); otherwise it follows the ordinary CME dynamics. The latent type is marginalised out (`mixture_model.py`). Model 1a puts the covariates on `a`, Model 2a on `r`, matching the base models. **Model 2a is the preferred model** (best PSIS-LOO; retains the treatment effects; reproduces the warmed×week-6 interaction).

All commands are run from the repo root and are deterministic (fixed seeds), so the figures/tables reproduce exactly.

**Fit** (DREAM, resumable exactly like the base models — each `--resume` pass seeds a fresh DREAM from the previous chunk's endpoints, see the resume/pooling caveats above; converge to R-hat < 1.1 with `--resume` passes, then it consolidates automatically):

```
python run_dream_mixture.py --map-check --models 1,2         # fast CMA-ES sanity check
python run_dream_mixture.py --models 1,2 [--n-iter N]        # cold start
python run_dream_mixture.py --models 1,2 --resume            # warm passes to convergence
python run_dream_mixture.py --consolidate --models 1,2       # -> outputs/model_<id>a_posterior.csv
```

**Checks, figures and tables** (each reads the consolidated posterior CSV):

```
python run_dream_mixture.py --prior-predictive --models 2    # outputs/model_2a_prior_predictive.png
python ppc_mixture.py --model 2                              # ALL PPC figures + outputs/model_2a_bayes_pvalues.csv
python make_fig3_mixture.py --model 2                        # outputs/model_2a_fig3.png
python loo_compare_all.py                                    # 4-way base-vs-mixture PSIS-LOO -> outputs/loo_comparison.csv
```

`ppc_mixture.py --model 2` writes: the per-`n0` posterior-predictive grid, the actual-vs-simulated violin PPCs (by weight tertile / acclimation / experimental condition), the warmed×week-6 PPC, the discrepancy-statistic figure, the LOO-PIT calibration plot, and `outputs/model_2a_bayes_pvalues.csv` (test statistic, observed value, model mean, 95% interval, and Bayesian p-values). `loo_compare_all.py` writes `outputs/loo_comparison.csv` (elpd_loo, p_loo, elpd difference, SE, model weight, and bad-Pareto-k count for all four models).

### Paper tables (deterministic)
```bash
python make_results_table.py   # outputs/model_2a_results_table.{csv,tex}: handling time (60/r, min)
                               # and predicted prey eaten (n0=40) per condition x week, median-weight fish,
                               # with 95% CIs and % change vs ambient. Fixed thinning -> fully reproducible.
python make_supp_tables.py     # outputs/table_loo.tex (4-model PSIS-LOO) and outputs/table_bayes_p.tex
                               # (discrepancy p-values), generated from the committed result CSVs.
```
```bash
python make_voracious_prob.py --model 2   # outputs/model_2a_voracious_prob.png: P(voracious) vs fish weight
                                          # (posterior median + 95% band, rug of observed weights; no title)
```
```bash
python prior_sensitivity.py   # Model-2a prior sensitivity via PSIS reweighting (no refit).
                              # widen/shift-only priors -> outputs/sensitivity/{sensitivity_summary.csv,
                              # sensitivity_forest.png, sensitivity_panelA.png, sensitivity_panelB.png}
                              # + paper/tables/table_prior_sensitivity.tex. All Pareto-k < 0.7.
```
Draft LaTeX for the manuscript lives in `paper/draft_attack_unlimited.tex` (voracious-subpopulation Results subsection), `paper/draft_supplementary_ppc.tex` (supplementary PPC section), and `paper/draft_prior_sensitivity.tex` (supplementary prior-sensitivity section). Model-2a PPC and sensitivity figures are copied into `paper/figures/`.

## R: plotting
This code takes the `csv` files from the inference and uses these to plot the data.
