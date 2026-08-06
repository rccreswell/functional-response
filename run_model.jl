# using Pkg; Pkg.add("Pigeons")
# using Pkg; Pkg.add("Turing")
# using Pkg; Pkg.add("Distributions")
# using Pkg; Pkg.add("MCMCChains")
# using Pkg; Pkg.add("CSV")
# using Pkg; Pkg.add("DataFrames")
# using Pkg; Pkg.add("BenchmarkTools")
# using Pkg; Pkg.add("SparseArrays")
# using Pkg; Pkg.add("ExponentialUtilities")

using LinearAlgebra
using Statistics
using Turing
using Distributions
using Pigeons
using MCMCChains
using CSV, DataFrames
using BenchmarkTools
using Random
using SparseArrays
using ExponentialUtilities

function construct_matrix(n0::Int, a::T, r::T) where {T<:Real}
    m = n0 + 1
    a_ns = T[-a * (n0 - i + 1) for i in 1:m]
    rs = fill(r, m)
    rs[1] = zero(T)

    A = zeros(T, 2 * m, 2 * m)
    for i in 1:m
        A[i, i] = a_ns[i]
        A[i, m + i] = rs[i]
        A[m + i, m + i] = -rs[i]
    end
    for ri in 2:m
        A[m + ri, ri - 1] = -a_ns[ri - 1]
    end
    return A
end

function solve_system_p(t::Real, n0::Int, a::Real, r::Real)
    T = float(promote_type(typeof(a), typeof(r)))
    A = construct_matrix(n0, T(a), T(r))
    p0 = zeros(T, 2(n0 + 1))
    p0[1] = one(T)
    p = exp(A .* T(t)) * p0
    dp = n0 + 1
    return p[1:dp] .+ p[dp+1:end]
end

function eaten_probs(n0::Int, a::Real, r::Real)
    p = solve_system_p(1.0, n0, a, r)
    p = max.(p, zero(eltype(p)))
    return p ./ sum(p)
end

# X_a, X_r are assumed to have a leading column of ones; the intercept is handled
# separately from the covariate slopes so it can be recentred independently.
#
#   * Intercepts recentred to sit in the interior of the saturating count curve.
#     Over the unit window eaten ~= min(n0, r), so log_r0 ~ log(median eaten) puts
#     the baseline predator near half of a typical n0 rather than at r=1 (~1 eaten).
#     a is only weakly identified (fast rate -> flat likelihood), so its intercept
#     gets a modestly informative prior to keep the sampler off that plateau.
#   * Hierarchical slopes: tau_a, tau_r are the (half-Normal) scales of the
#     covariate effects, learned from the data. This resolves the tension between
#     "enough variation in the covariate effects" and "not too much spread in the
#     counts" -- tau shrinks when covariates are uninformative and expands when
#     they are not, instead of us hard-coding a single slope variance.
@model function functional_response(eaten, n_initial, X_a, X_r;
                                    log_a0_mean = log(4.0), log_a0_sd = 1.0,
                                    log_r0_mean = log(20.0), log_r0_sd = 0.5,
                                    tau_scale = 0.5)
    n_a = size(X_a, 2)
    n_r = size(X_r, 2)

    a0 ~ Normal(log_a0_mean, log_a0_sd)
    r0 ~ Normal(log_r0_mean, log_r0_sd)

    if n_a > 1
        tau_a ~ truncated(Normal(0.0, tau_scale), 0.0, Inf)
        beta_a ~ filldist(Normal(0.0, tau_a), n_a - 1)
        log_a = a0 .+ X_a[:, 2:end] * beta_a
    else
        log_a = fill(a0, length(eaten))
    end

    if n_r > 1
        tau_r ~ truncated(Normal(0.0, tau_scale), 0.0, Inf)
        beta_r ~ filldist(Normal(0.0, tau_r), n_r - 1)
        log_r = r0 .+ X_r[:, 2:end] * beta_r
    else
        log_r = fill(r0, length(eaten))
    end

    a_all = exp.(log_a)
    r_all = exp.(log_r)

    for i in eachindex(eaten)
        p = eaten_probs(n_initial[i], a_all[i], r_all[i])
        eaten[i] ~ Categorical(p)
    end
end



df = CSV.read("data/df_covariates_better_date.csv", DataFrame)
# Standardize weight (z-score) and use its square, so the linear and quadratic
# weight terms are on a comparable unit scale and are largely decorrelated.
df.weight_z = (df.weight .- mean(df.weight)) ./ std(df.weight)
df.weight_z2 = df.weight_z .^ 2
nums_prey_eaten = Int.(df.n_prey_eaten)
nums_prey_initial = Int.(df.n_prey_initial)

a_cols = ["f.acc_condH",
          "f.acc_condM",
          "f.acc_condT",
          "f.exp_condH",
          "f.exp_condM",
          "f.exp_condT",
          "f.week6",
          "f.acc_condH:f.week6",
          "f.acc_condM:f.week6",
          "f.acc_condT:f.week6"]
r_cols = ["weight_z", "weight_z2"]

X_a = hcat(ones(nrow(df)), Matrix{Float64}(df[:, a_cols]))
X_r = hcat(ones(nrow(df)), Matrix{Float64}(df[:, r_cols]))

num_fish = nrow(df)
num_a_covariates = size(X_a, 2)
num_r_covariates = size(X_r, 2)

model = functional_response(nums_prey_eaten .+ 1, nums_prey_initial, X_a, X_r)
lp = TuringLogPotential(model)
x = Pigeons.initialization(lp, Random.default_rng(), 1)
@btime $lp($x)

print(x)
print(lp(x))



pt = pigeons(target = TuringLogPotential(model), explorer = SliceSampler(), n_chains = 10, n_rounds = 2, record = [traces; round_trip; record_default()])
chn = Chains(pt)

# Alternative: gradient-based NUTS. Turing can autodiff through exp(A*t), and with
# the recentred intercepts + hierarchical slopes the geometry is far better
# conditioned than the old wide-Gaussian setup, so NUTS should mix well and give
# an honest R-hat/ESS. Run several chains from dispersed starts:
#   chn_nuts = sample(model, NUTS(0.8), MCMCThreads(), 1000, 4)
# If NUTS still struggles (e.g. funnel from the tau hierarchy), a non-centred
# parameterisation of beta_a/beta_r is the usual next step.
