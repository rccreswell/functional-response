# Functional response

Analyses underpinning Newport et al. (2026), "Heatwaves and turbidity influxes have interacting and lasting effects on foraging behaviour of three-spined sticklebacks _Gasterosteus aculeatus_".

## Models

A stochastic (chemical master equation) model of a predator that switches between a *preying* state (searching, attack rate `a`) and a *handling* state (rate `r`, so handling time `h = 1/r`). Covariates enter through a log link, `log a = a0 + beta_a·x_a`, `log r = r0 + beta_r·x_r`. Four candidates are fitted and compared:

| | Covariates on `a` | Covariates on `r` | Voracious mixture |
|---|---|---|---|
| **1** | treatments + weight | — | no |
| **2** | — | treatments + weight | no |
| **1a** | treatments + weight | — | yes |
| **2a** | — | treatments + weight | **yes** |

The mixture (`1a`/`2a`) adds a weight-dependent probability that a fish is a *voracious* forager whose attack is effectively instantaneous (the `a→∞` limit, so `eaten = min(n0, 1 + Poisson(r))`), fixing the base models' under-prediction of full-consumption; the latent type is marginalised out (`mixture_model.py`). **Model 2a is the selected model** — best PSIS-LOO, retains the treatment effects, and reproduces the warmed×6-week interaction. Priors are calibrated (`a0 ~ N(log 4, 1)`, `r0 ~ N(log 20, 0.5)`), with p-aware slope priors `N(0, (0.75/√p)²)` and standardised weight; the mixture logits use `N(-1, 1.5²)` / `N(0, 1.5²)`.

## Reproducing the results

Pinned environment (Python 3.9.7; numpy 1.20.3, scipy 1.7.1, pandas 1.3.4, matplotlib 3.4.3, joblib 1.1.0, pints 0.5.0, arviz 0.11.2):

```bash
conda env create -f environment.yml && conda activate functional-response
```

Everything downstream of the fitted posteriors — every figure, table, and number quoted in the paper — regenerates deterministically from the committed posterior CSVs via Snakemake:

```bash
snakemake -c1 -n     # dry run: show the DAG
snakemake -c1        # build all artifacts under paper/
```

Outputs land in `paper/figures/`, `paper/tables/`, and `paper/generated_values.tex` — `\newcommand` macros holding every quoted value with its uncertainty interval (and the ROPE comparisons), `\input` by `article.tex` so no result is hand-typed. Build the PDF with `latexmk -pdf paper/article.tex`. (Figure 1B is produced by the R `targets` pipeline, `src/r/explanatory_figure.R`.)

### Refitting the models (Tier B — expensive, optional)

The DREAM fits (`data → posteriors`) are stochastic — they reproduce the results only up to Monte Carlo error (parallelism + BLAS ordering) — so the posterior CSVs are treated as versioned inputs. Refitting lives in a separate Snakefile so a plain `snakemake` never triggers it; it **overwrites** the committed posteriors:

```bash
snakemake -s Snakefile.fit -c1 fit_all                 # all four models
snakemake -s Snakefile.fit -c1 --config n_iter=30000   # override run length
```

Convergence caveats (see `run_dream.py` / `run_dream_mixture.py`): chains start in a tight ball around a CMA-ES optimum (not over-dispersed, so `R-hat` is a weaker check); the pooled posterior uses a heuristic warm-up discard; and `--resume` seeds a *fresh* DREAM from the previous run's endpoints rather than checkpointing the sampler.

## Repository layout

Python analysis code lives in `src/python/`, the R `targets` pipeline in `src/r/` (with `_targets.R` at the root), exploratory notebooks in `notebooks/`, data in `data/`, generated artifacts in `outputs/`, and the manuscript in `paper/`.

## Key scripts (in `src/python/`)

- `run_dream.py`, `run_dream_mixture.py` — fits (base / mixture), prior-predictive checks, PSIS-LOO.
- `mixture_model.py` — voracious-mixture likelihood.
- `ppc_mixture.py`, `make_fig3_mixture.py`, `make_voracious_prob.py` — posterior-predictive checks and figures.
- `loo_compare_all.py`, `make_ppc_table_all.py` — 4-model LOO comparison and discrepancy p-values.
- `make_results_table.py`, `make_supp_tables.py`, `make_paper_values.py` — paper tables and the macro values.
- `prior_sensitivity.py` — Model-2a prior sensitivity via PSIS reweighting (no refit).
- `src/r/explanatory_figure.R` — Figure 1B (R `targets` pipeline).
