# Pipeline definition for the explanatory figure. All simulation,
# CTMC-solving, and plotting logic lives in src/r/explanatory_figure.R;
# this file just describes the dependency graph.

library(targets)

tar_option_set(
  packages = c("tidyverse", "expm", "patchwork")
)

tar_source("src/r/explanatory_figure.R")

N_replicates <- 50

list(

  # ------------------------------------------------------------------
  # Parameters
  # ------------------------------------------------------------------

  tar_target(N_prey_lower,  10),
  tar_target(N_prey_upper,  50),
  tar_target(h_rate_lower,  10),
  tar_target(h_rate_upper,  10000),
  tar_target(a_rate_lower,  2),
  tar_target(a_rate_upper,  100),

  # ------------------------------------------------------------------
  # Simulated trajectories (left column)
  # ------------------------------------------------------------------

  tar_target(df_explanatory_figure_b_i,
             simulate_fish_ctmc_replicates(N_replicates, N_prey_lower, 1,
                                           a_rate_lower, h_rate_upper)),
  tar_target(df_explanatory_figure_b_ii,
             simulate_fish_ctmc_replicates(N_replicates, N_prey_upper, 1,
                                           a_rate_lower, h_rate_upper)),
  tar_target(df_explanatory_figure_b_iii,
             simulate_fish_ctmc_replicates(N_replicates, N_prey_lower, 1,
                                           a_rate_upper, h_rate_lower)),
  tar_target(df_explanatory_figure_b_iv,
             simulate_fish_ctmc_replicates(N_replicates, N_prey_upper, 1,
                                           a_rate_upper, h_rate_lower)),

  # ------------------------------------------------------------------
  # CTMC mean + quantile summaries (right column)
  # ------------------------------------------------------------------

  tar_target(df_explanatory_c_i,
             ctmc_eaten_summary_grid(
               N0_grid  = 1:50,
               T_end    = 1,
               alpha    = a_rate_lower,
               h_rate   = h_rate_upper,
               probs    = c(0.1, 0.9),
               scenario = "I. Search-limited"
             )),
  tar_target(df_explanatory_c_ii,
             ctmc_eaten_summary_grid(
               N0_grid  = 1:50,
               T_end    = 1,
               alpha    = a_rate_upper,
               h_rate   = h_rate_lower,
               probs    = c(0.1, 0.9),
               scenario = "II. Handling-limited"
             )),

  # ------------------------------------------------------------------
  # Figures
  # ------------------------------------------------------------------

  tar_target(plot_explanatory_figure_b,
             make_plot_explanatory_figure_b(
               df_search_low    = df_explanatory_figure_b_i,
               df_search_high   = df_explanatory_figure_b_ii,
               df_handling_low  = df_explanatory_figure_b_iii,
               df_handling_high = df_explanatory_figure_b_iv,
               n_prey_low       = N_prey_lower,
               n_prey_high      = N_prey_upper
             )),

  tar_target(file_plot_explanatory_figure_b, {
    path <- "outputs/plot_explanatory.pdf"
    ggsave(path, plot_explanatory_figure_b, width = 7, height = 6)
    path
  }, format = "file"),

  tar_target(plot_explanatory_figure_bc,
             make_plot_explanatory_figure_bc(
               df_search_low    = df_explanatory_figure_b_i,
               df_search_high   = df_explanatory_figure_b_ii,
               df_handling_low  = df_explanatory_figure_b_iii,
               df_handling_high = df_explanatory_figure_b_iv,
               df_c_search      = df_explanatory_c_i,
               df_c_handling    = df_explanatory_c_ii,
               n_prey_low       = N_prey_lower,
               n_prey_high      = N_prey_upper
             )),

  tar_target(file_plot_explanatory_figure_bc, {
    path <- "outputs/plot_explanatory_figure_bc.pdf"
    ggsave(path, plot_explanatory_figure_bc, width = 10, height = 8)
    path
  }, format = "file")
)
