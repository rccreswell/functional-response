# Functional response
This repository contains resources for rerunning functional response-type analyses underpinning Newport et al. (2026), "Heatwaves and turbidity influxes have interacting and lasting effects on foraging behaviour of three-spined sticklebacks _Gasterosteus aculeatus_".

## Python: inference
- `run_models.ipynb`: runs the inference for two considered models. Model 1 is when `r` (the inverse of `h`) is a function only of `weight` and `weight^2` and `a` is a function of all other covariates. Model 2 is when `r` (the inverse of `h`) is a function only of all covariates and `a` is a constant. This notebook outputs `csv` files containing the MCMC samples to the `outputs` directory.

## R: plotting
This code takes the `csv` files from the inference and uses these to plot the data.
