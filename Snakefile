# Reproducible analysis pipeline (Tier A): regenerates every paper artifact
# (tables, figures, macro values) from the committed posterior draws.
#
#   conda env create -f environment.yml && conda activate functional-response
#   snakemake -c1                 # build everything in `rule all`
#   snakemake -c1 --use-conda     # ... in an isolated pinned env per rule
#   snakemake -c1 -n              # dry run (show the DAG / what would run)
#
# Tier B (the expensive, stochastic DREAM fits data -> posteriors) is deliberately
# NOT wired here: it takes hours and is not bitwise-reproducible across machines.
# Treat the posterior CSVs as versioned inputs; see README. To (re)fit, run e.g.
#   python src/python/run_dream_mixture.py --models 2 --resume     (repeat to convergence)

POSTERIORS = ["outputs/model_1_posterior.csv", "outputs/model_2_posterior.csv",
              "outputs/model_1a_posterior.csv", "outputs/model_2a_posterior.csv"]
DATA = "data/df_covariates_better_date.csv"

PAPER_TABLES = ["paper/tables/table_bayes_p.tex", "paper/tables/table_params_1a.tex",
                "paper/tables/table_params_2a.tex", "paper/tables/table_prior_sensitivity.tex",
                "paper/tables/table_rope.tex"]
PAPER_FIGS = ["paper/figures/model_2a_fig3.png", "paper/figures/model_2a_voracious_prob.png",
              "paper/figures/model_2a_ppc_violin.png", "paper/figures/model_2a_ppc_violin_acc.png",
              "paper/figures/model_2a_ppc_violin_exp.png", "paper/figures/model_2a_ppc_warmed_week.png",
              "paper/figures/model_2a_loo_pit.png", "paper/figures/model_2a_posterior_predictive.png",
              "paper/figures/model_2a_prior_predictive.png",
              "paper/figures/sensitivity_panelA.png", "paper/figures/sensitivity_panelB.png"]


rule all:
    input:
        "paper/generated_values.tex",
        "outputs/model_2a_results_table.tex",
        "outputs/table_loo.tex",
        "paper/figures/plot_explanatory_figure_bc.pdf",
        PAPER_TABLES,
        PAPER_FIGS,


# Explanatory figure (Fig 1B) is produced by the R `targets` pipeline
# (src/r/explanatory_figure.R); we treat the committed PDF as an input and copy it in.
rule explanatory_fig:
    input: "outputs/plot_explanatory_figure_bc.pdf"
    output: "paper/figures/plot_explanatory_figure_bc.pdf"
    shell: "cp {input} {output}"


rule loo:
    input: POSTERIORS
    output: "outputs/loo_comparison.csv"
    conda: "environment.yml"
    shell: "python src/python/loo_compare_all.py"


rule ppc_all_models:
    input: POSTERIORS
    output: "outputs/all_models_bayes_pvalues.csv", "outputs/all_models_stat_means.csv"
    conda: "environment.yml"
    shell: "python src/python/make_ppc_table_all.py"


rule ppc_mixture:
    input: "outputs/model_2a_posterior.csv", DATA
    output:
        "outputs/model_2a_bayes_pvalues.csv",
        "outputs/model_2a_ppc_discrepancy.png",
        p1="paper/figures/model_2a_posterior_predictive.png",
        p2="paper/figures/model_2a_ppc_warmed_week.png",
        p3="paper/figures/model_2a_loo_pit.png",
        p4="paper/figures/model_2a_ppc_violin.png",
        p5="paper/figures/model_2a_ppc_violin_acc.png",
        p6="paper/figures/model_2a_ppc_violin_exp.png",
    conda: "environment.yml"
    shell:
        "python src/python/ppc_mixture.py --model 2 && "
        "cp outputs/model_2a_posterior_predictive.png outputs/model_2a_ppc_warmed_week.png "
        "outputs/model_2a_loo_pit.png outputs/model_2a_ppc_violin.png "
        "outputs/model_2a_ppc_violin_acc.png outputs/model_2a_ppc_violin_exp.png paper/figures/"


rule prior_predictive:
    input: "outputs/model_2a_posterior.csv", DATA
    output: "paper/figures/model_2a_prior_predictive.png"
    conda: "environment.yml"
    shell:
        "python src/python/run_dream_mixture.py --models 2 --prior-predictive && "
        "cp outputs/model_2a_prior_predictive.png paper/figures/"


rule results_table:
    input: "outputs/model_2a_posterior.csv", DATA
    output: "outputs/model_2a_results_table.csv", "outputs/model_2a_results_table.tex"
    conda: "environment.yml"
    shell: "python src/python/make_results_table.py"


rule fig3:
    input: "outputs/model_2a_posterior.csv"
    output: "paper/figures/model_2a_fig3.png"
    conda: "environment.yml"
    shell: "python src/python/make_fig3_mixture.py --model 2 && cp outputs/model_2a_fig3.png paper/figures/"


rule voracious_prob:
    input: "outputs/model_2a_posterior.csv", DATA
    output: "paper/figures/model_2a_voracious_prob.png"
    conda: "environment.yml"
    shell: "python src/python/make_voracious_prob.py --model 2 && cp outputs/model_2a_voracious_prob.png paper/figures/"


rule prior_sensitivity:
    input: "outputs/model_2a_posterior.csv", DATA
    output:
        "outputs/sensitivity/sensitivity_summary.csv",
        "paper/tables/table_prior_sensitivity.tex",
        p1="paper/figures/sensitivity_panelA.png",
        p2="paper/figures/sensitivity_panelB.png",
    conda: "environment.yml"
    shell:
        "python src/python/prior_sensitivity.py && "
        "cp outputs/sensitivity/sensitivity_panelA.png outputs/sensitivity/sensitivity_panelB.png paper/figures/"


rule supp_tables:
    input:
        "outputs/loo_comparison.csv", "outputs/all_models_bayes_pvalues.csv",
        "outputs/model_1a_posterior.csv", "outputs/model_2a_posterior.csv",
    output:
        "outputs/table_loo.tex", "paper/tables/table_bayes_p.tex",
        "paper/tables/table_params_1a.tex", "paper/tables/table_params_2a.tex",
    conda: "environment.yml"
    shell: "python src/python/make_supp_tables.py"


rule paper_values:
    input:
        "outputs/model_2a_results_table.csv", "outputs/model_2a_bayes_pvalues.csv",
        "outputs/all_models_stat_means.csv", "outputs/model_2a_posterior.csv", DATA,
    output: "paper/generated_values.tex", "paper/tables/table_rope.tex"
    conda: "environment.yml"
    shell: "python src/python/make_paper_values.py"
