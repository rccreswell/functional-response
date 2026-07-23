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

@model function functional_response(eaten, n_initial, X_a, X_r; prior_sd = 0.1)
    n_a = size(X_a, 2)
    n_r = size(X_r, 2)

    beta_a ~ filldist(Normal(0.0, prior_sd), n_a)
    beta_r ~ filldist(Normal(0.0, prior_sd), n_r)

    a_all = exp.(X_a * beta_a)
    r_all = exp.(X_r * beta_r)

    for i in eachindex(eaten)
        p = eaten_probs(n_initial[i], a_all[i], r_all[i])
        eaten[i] ~ Categorical(p)
    end
end



df = CSV.read("data/df_covariates_better_date.csv", DataFrame)
df.weight2 = df.weight .^ 2
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
r_cols = ["weight", "weight2"]

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
