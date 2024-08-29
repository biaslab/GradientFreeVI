using Pkg
Pkg.activate(".")
using Distributions
using Turing
using SliceSampling

using OrdinaryDiffEq
using SciMLSensitivity
using Random
using ProgressMeter
using DataFrames
using StatsPlots
using BenchmarkTools
using JSON

include("bayes_sir_turing_spec.jl")
include("rxinfer_ode_solve.jl")

results = Dict()
results["NUTS10"] = []
results["NUTS100"] = []
results["NUTS1000"] = []
results["Sliced10"] = []
results["Sliced100"] = []
results["Sliced1000"] = []
results["RxInfer10"] = []
results["RxInfer100"] = []
results["RxInfer1000"] = []

for _ in 1:10
    β = rand()
    i = 1
    while i > β
        i = rand()
    end
    @show i, β
    nuts_10 = run_ode_experiment(i, β, NUTS(0.65), 10)
    nuts_100 = run_ode_experiment(i, β, NUTS(0.65), 100)
    nuts_1000 = run_ode_experiment(i, β, NUTS(0.65), 1000)
    push!(results["NUTS10"], nuts_10)
    push!(results["NUTS100"], nuts_100)
    push!(results["NUTS1000"], nuts_1000)

    sliced_10 = run_ode_experiment(i, β, externalsampler(LatentSlice(1)), 10)
    sliced_100 = run_ode_experiment(i, β, externalsampler(LatentSlice(1)), 100)
    sliced_1000 = run_ode_experiment(i, β, externalsampler(LatentSlice(1)), 1000)
    push!(results["Sliced10"], sliced_10)
    push!(results["Sliced100"], sliced_100)
    push!(results["Sliced1000"], sliced_1000)

    rxi10 = run_experiments(i, β, sir_constraints(5, 2))
    rxi100 = run_experiments(i, β, sir_constraints(49, 3))
    rxi1000 = run_experiments(i, β, sir_constraints(49, 50))
    push!(results["RxInfer10"], rxi10)
    push!(results["RxInfer100"], rxi100)
    push!(results["RxInfer1000"], rxi1000)
end

open("nuts_sir.json", "w") do f
    JSON.print(f, results)
end
