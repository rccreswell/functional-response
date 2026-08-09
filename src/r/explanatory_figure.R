simulate_fish_ctmc <- function(
    N0,
    T_end,
    alpha,
    h_rate
) {

  t <- 0
  state <- "searching"
  prey_remaining <- N0
  prey_eaten <- 0

  eaten_events <- data.frame(
    time = numeric(),
    prey_eaten = integer(),
    prey_remaining = integer()
  )

  while (t < T_end && prey_remaining > 0) {

    if (state == "searching") {

      rate <- alpha * prey_remaining
      wait <- rexp(1, rate = rate)
      t_next <- t + wait

      if (t_next > T_end) break

      t <- t_next
      prey_remaining <- prey_remaining - 1
      prey_eaten <- prey_eaten + 1
      state <- "handling"

      eaten_events <- rbind(
        eaten_events,
        data.frame(
          time = t,
          prey_eaten = prey_eaten,
          prey_remaining = prey_remaining
        )
      )

    } else {

      wait <- rexp(1, rate = h_rate)
      t_next <- t + wait

      if (t_next > T_end) break

      t <- t_next
      state <- "searching"
    }
  }

  # Add final endpoint
  eaten_events <- rbind(
    eaten_events,
    data.frame(
      time = T_end,
      prey_eaten = prey_eaten,
      prey_remaining = prey_remaining
    )
  )

  eaten_events
}

simulate_fish_ctmc_replicates <- function(
    n_replicates,
    N0,
    T_end,
    alpha,
    h_rate,
    seed = NULL
) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  sims <- lapply(seq_len(n_replicates), function(i) {

    sim <- simulate_fish_ctmc(
      N0 = N0,
      T_end = T_end,
      alpha = alpha,
      h_rate = h_rate
    )

    sim$id <- i

    sim
  })

  do.call(rbind, sims)
}


build_Q <- function(N0, alpha, h_rate) {

  states <- expand.grid(
    state = c("S", "H"),
    k = 0:N0
  )

  n_states <- nrow(states)
  Q <- matrix(0, n_states, n_states)

  idx <- function(state, k) {
    which(states$state == state & states$k == k)
  }

  for (k in 0:N0) {

    # Searching -> Handling after eating prey
    if (k > 0) {
      from <- idx("S", k)
      to <- idx("H", k - 1)
      Q[from, to] <- alpha * k
    }

    # Handling -> Searching
    from <- idx("H", k)
    to <- idx("S", k)
    Q[from, to] <- h_rate
  }

  diag(Q) <- -rowSums(Q)

  list(Q = Q, states = states)
}


ctmc_mean_eaten <- function(N0, T_end, alpha, h_rate) {

  obj <- build_Q(N0, alpha, h_rate)
  Q <- obj$Q
  states <- obj$states

  # Initial condition: searching with N0 prey remaining
  p0 <- rep(0, nrow(states))
  p0[states$state == "S" & states$k == N0] <- 1

  # Transient distribution at T_end
  pT <- as.numeric(p0 %*% expm(Q * T_end))

  prey_eaten <- N0 - states$k

  sum(pT * prey_eaten)
}


# Probability mass function of the number of prey eaten by time T_end,
# obtained by marginalising the joint distribution over (state, k) onto n = N0 - k.
ctmc_eaten_pmf <- function(N0, T_end, alpha, h_rate) {

  obj <- build_Q(N0, alpha, h_rate)
  Q <- obj$Q
  states <- obj$states

  p0 <- rep(0, nrow(states))
  p0[states$state == "S" & states$k == N0] <- 1

  pT <- as.numeric(p0 %*% expm(Q * T_end))

  prey_eaten <- N0 - states$k

  pmf <- tapply(pT, prey_eaten, sum)

  data.frame(
    n = as.integer(names(pmf)),
    p = as.numeric(pmf)
  )
}


# Mean and quantiles of the number of prey eaten. Quantiles use the
# standard discrete definition Q(q) = inf{ n : F(n) >= q }.
ctmc_eaten_summary <- function(N0, T_end, alpha, h_rate, probs = c(0.1, 0.9)) {

  pmf_df <- ctmc_eaten_pmf(N0, T_end, alpha, h_rate)
  pmf_df <- pmf_df[order(pmf_df$n), ]

  cdf <- cumsum(pmf_df$p)

  mean_eaten <- sum(pmf_df$n * pmf_df$p)

  qs <- vapply(probs, function(q) {
    as.numeric(pmf_df$n[which(cdf >= q)[1]])
  }, numeric(1))

  out <- data.frame(mean_eaten = mean_eaten)
  for (i in seq_along(probs)) {
    out[[paste0("q", round(probs[i] * 100))]] <- qs[i]
  }

  out
}


# Apply ctmc_eaten_summary across a grid of initial prey counts and tag
# the resulting rows with a scenario label. Returns a single data frame
# with columns N0, mean_eaten, q10, q90 (plus any other quantiles requested)
# and scenario.
ctmc_eaten_summary_grid <- function(N0_grid, T_end, alpha, h_rate,
                                    probs = c(0.1, 0.9),
                                    scenario = NULL) {

  out <- purrr::map_dfr(N0_grid, function(N0) {
    cbind(
      N0 = N0,
      ctmc_eaten_summary(
        N0 = N0, T_end = T_end,
        alpha = alpha, h_rate = h_rate,
        probs = probs
      )
    )
  })

  if (!is.null(scenario)) out$scenario <- scenario

  out
}


# Analytical E[N_eaten(t)] for one (N0, alpha, h_rate), evaluated on a grid
# of times. Returns a data frame with columns time and mean_eaten.
ctmc_mean_curve <- function(N0, t_grid, alpha, h_rate) {

  obj <- build_Q(N0, alpha, h_rate)
  Q <- obj$Q
  states <- obj$states

  p0 <- rep(0, nrow(states))
  p0[states$state == "S" & states$k == N0] <- 1

  prey_eaten <- N0 - states$k

  means <- vapply(t_grid, function(t) {
    if (t == 0) return(0)
    pT <- as.numeric(p0 %*% expm(Q * t))
    sum(pT * prey_eaten)
  }, numeric(1))

  data.frame(time = t_grid, mean_eaten = means)
}


# Mean curves for one scenario (both low and high N0), bound into a single
# long-format frame with columns time, mean_eaten, n_prey, scenario.
ctmc_mean_curve_scenario <- function(t_grid, alpha, h_rate,
                                     n_prey_low, n_prey_high,
                                     scenario) {

  bind_rows(
    ctmc_mean_curve(n_prey_low,  t_grid, alpha, h_rate) %>%
      mutate(n_prey = n_prey_low),
    ctmc_mean_curve(n_prey_high, t_grid, alpha, h_rate) %>%
      mutate(n_prey = n_prey_high)
  ) %>%
    mutate(
      n_prey = factor(n_prey),
      scenario = scenario
    )
}


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------

# Combine the low- and high-prey simulation replicates for a single
# scenario into one long-format data frame, tagging each row with the
# initial prey count and the scenario label.
prepare_scenario_traces <- function(df_low, df_high,
                                    n_prey_low, n_prey_high,
                                    scenario) {

  bind_rows(
    df_low  %>% mutate(n_prey = n_prey_low),
    df_high %>% mutate(n_prey = n_prey_high)
  ) %>%
    mutate(
      scenario = scenario,
      n_prey = factor(n_prey)
    )
}


# Left-column panel: stochastic traces + smoothed mean, faceted by scenario.
plot_traces_facet <- function(df) {

  ggplot(df, aes(x = time * 60, y = prey_eaten, colour = n_prey)) +
    geom_step(alpha = 0.4, aes(group = interaction(scenario, id, n_prey))) +
    geom_smooth(se = FALSE) +
    scale_color_brewer("initial\n# prey", palette = "Dark2") +
    labs(x = "Time (minutes)", y = "Cumulative prey eaten") +
    facet_wrap(~ scenario, ncol = 1) +
    ylim(0, 51) +
    theme_classic()
}


# Single-scenario trace panel (no facet). Used by the bc figure where the
# scenario label sits in its own strip above the row.
#
# df_mean (optional) is a long-format frame with columns time, mean_eaten,
# n_prey, scenario -- overlaid as a thick coloured line per n_prey.
# show_legend = TRUE places the legend inside the bottom-right of the panel
# (used on the search-limited row only); otherwise the legend is hidden.
plot_traces_single <- function(df, df_mean = NULL, show_legend = FALSE) {

  p <- ggplot(df, aes(x = time * 60, y = prey_eaten, colour = n_prey)) +
    geom_step(alpha = 0.3, aes(group = interaction(id, n_prey)))

  if (!is.null(df_mean)) {
    p <- p + geom_line(
      data = df_mean,
      aes(y = mean_eaten),
      linewidth = 1.4
    )
  }

  p <- p +
    coord_cartesian(ylim = c(0, 51)) +
    scale_color_brewer("initial\n# prey", palette = "Dark2") +
    labs(x = "Time (minutes)", y = "Cumulative prey eaten") +
    theme_classic()

  if (show_legend) {
    p <- p + theme(
      legend.position      = c(0.97, 0.3),
      legend.justification = c(1, 0),
      legend.background    = element_rect(fill = scales::alpha("white", 0.7),
                                          colour = NA),
      legend.key           = element_rect(fill = "transparent", colour = NA),
      legend.box.background = element_blank()
    )
  } else {
    p <- p + theme(legend.position = "none")
  }

  p
}


# Right-column panel: mean number eaten vs initial prey, with a
# q10-q90 ribbon underneath. vlines (optional) is a data frame with
# columns xintercept and colour, drawn as dashed reference lines to tie
# the right column back to the n_prey values shown on the left column;
# coloured endpoint dots are drawn at the same xintercepts on the mean
# curve.
plot_mean_with_quantiles <- function(df, vlines = NULL) {

  vline_layers <- list()
  point_layers <- list()

  if (!is.null(vlines)) {
    vline_layers <- lapply(seq_len(nrow(vlines)), function(i) {
      geom_vline(
        xintercept = vlines$xintercept[i],
        colour     = vlines$colour[i],
        linetype   = "dashed",
        linewidth  = 0.6,
        alpha      = 0.85
      )
    })

    point_df <- merge(
      vlines,
      df[, c("N0", "mean_eaten")],
      by.x  = "xintercept",
      by.y  = "N0",
      all.x = TRUE
    )

    point_layers <- list(
      geom_point(
        data = point_df,
        aes(x = xintercept, y = mean_eaten, colour = colour),
        size  = 3.5,
        inherit.aes = FALSE
      ),
      scale_colour_identity()
    )
  }

  ggplot(df, aes(x = N0, y = mean_eaten)) +
    vline_layers +
    geom_ribbon(
      aes(ymin = q10, ymax = q90),
      fill = "grey60",
      alpha = 0.35
    ) +
    geom_line(linewidth = 1.2) +
    point_layers +
    coord_cartesian(ylim = c(0, 51)) +
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.04))) +
    labs(x = "Initial # prey", y = NULL) +
    theme_classic()
}


# Bold row-title strip used between the two scenarios in the bc figure.
row_title_strip <- function(label) {

  ggplot(data.frame(x = 0, y = 0.5)) +
    geom_text(
      aes(x, y),
      label = label,
      hjust = 0,
      vjust = 0.5,
      fontface = "bold",
      size = 5.5
    ) +
    coord_cartesian(
      xlim = c(0, 1),
      ylim = c(0, 1),
      clip = "off"
    ) +
    theme_void() +
    theme(plot.margin = margin(t = 5, r = 5, b = 5, l = 5))
}


# Column-title strip used at the very top of the bc figure to label the
# two columns ("Trajectories over time" vs "Endpoint vs initial prey").
column_title_strip <- function(label) {

  ggplot(data.frame(x = 0.5, y = 0.5)) +
    geom_text(
      aes(x, y),
      label = label,
      hjust = 0.5,
      vjust = 0.5,
      fontface = "bold",
      size = 5
    ) +
    coord_cartesian(
      xlim = c(0, 1),
      ylim = c(0, 1),
      clip = "off"
    ) +
    theme_void() +
    theme(plot.margin = margin(t = 5, r = 5, b = 5, l = 5))
}


# Top-level: figure b -- the faceted figure with two scenarios as facets.
make_plot_explanatory_figure_b <- function(df_search_low, df_search_high,
                                           df_handling_low, df_handling_high,
                                           n_prey_low, n_prey_high) {

  df <- bind_rows(
    prepare_scenario_traces(
      df_search_low, df_search_high,
      n_prey_low, n_prey_high,
      "I. Search-limited"
    ),
    prepare_scenario_traces(
      df_handling_low, df_handling_high,
      n_prey_low, n_prey_high,
      "II. Handling-limited"
    )
  )

  plot_traces_facet(df)
}


# Top-level: figure bc -- 2x2 grid of (traces | mean+quantiles) per scenario,
# with bold row-title strips between them and column titles at the top.
# df_mean_search / df_mean_handling carry the analytical E[N_eaten(t)]
# curves overlaid on the left column.
make_plot_explanatory_figure_bc <- function(df_search_low, df_search_high,
                                            df_handling_low, df_handling_high,
                                            df_c_search, df_c_handling,
                                            df_mean_search, df_mean_handling,
                                            n_prey_low, n_prey_high) {

  df_search <- prepare_scenario_traces(
    df_search_low, df_search_high,
    n_prey_low, n_prey_high,
    "I. Search-limited"
  )
  df_handling <- prepare_scenario_traces(
    df_handling_low, df_handling_high,
    n_prey_low, n_prey_high,
    "II. Handling-limited"
  )

  # Colours matching scale_color_brewer(palette = "Dark2") for the two
  # n_prey levels, used for the dashed reference lines on the right column.
  dark2_pal <- RColorBrewer::brewer.pal(3, "Dark2")[1:2]
  vlines_df <- data.frame(
    xintercept = c(n_prey_low, n_prey_high),
    colour     = dark2_pal
  )

  # Strips and panels
  col_title_left  <- column_title_strip("Trajectories over time")
  col_title_right <- column_title_strip("Endpoint vs. initial prey count")

  title_search    <- row_title_strip("I. Search-limited")
  title_handling  <- row_title_strip("II. Handling-limited")

  p_search_left    <- plot_traces_single(df_search,
                                         df_mean = df_mean_search,
                                         show_legend = TRUE)
  p_search_right   <- plot_mean_with_quantiles(df_c_search,
                                               vlines = vlines_df)
  p_handling_left  <- plot_traces_single(df_handling,
                                         df_mean = df_mean_handling,
                                         show_legend = FALSE)
  p_handling_right <- plot_mean_with_quantiles(df_c_handling,
                                               vlines = vlines_df)

  (col_title_left | col_title_right) /
    title_search /
    (p_search_left | p_search_right) /
    title_handling /
    (p_handling_left | p_handling_right) +
    plot_layout(
      heights = c(0.10, 0.12, 1, 0.12, 1)
    )
}
