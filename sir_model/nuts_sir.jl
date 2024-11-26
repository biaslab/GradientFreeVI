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
using StableRNGs
using BenchmarkTools
using JSON
using Printf

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

rng = StableRNG(42)

for _ in 1:10
    β = rand(rng)/10
    i = rand(rng)/100
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

    rxi10 = run_experiments(i, β, sir_constraints(3, 5), 2)
    rxi100 = run_experiments(i, β, sir_constraints(30, 5), 2)
    rxi1000 = run_experiments(i, β, sir_constraints(49, 49), 4)
    push!(results["RxInfer10"], rxi10)
    push!(results["RxInfer100"], rxi100)
    push!(results["RxInfer1000"], rxi1000)
end

open("nuts_sir.json", "w") do f
    JSON.print(f, results)
end

function calculate_statistics(results::Dict)
    stats = Dict()
    
    # Group results by method and sample size
    methods = ["RxInfer", "NUTS", "Sliced"]
    sizes = ["10", "100", "1000"]
    
    for size in sizes
        for method in methods
            key = method * size
            if !isempty(results[key])
                # Extract β coverage (index 1), i₀ coverage (index 2), 
                # β MSE (index 3), i₀ MSE (index 4)
                values = results[key]
                
                # Calculate means and standard errors
                n = length(values)
                means = zeros(4)
                stds = zeros(4)
                
                for i in 1:4
                    coords = [v[i] for v in values]
                    means[i] = mean(coords)
                    stds[i] = std(coords) / sqrt(n)
                end
                
                stats[key] = (means=means, stds=stds)
            end
        end
    end
    
    return stats
end

function format_value(mean::Float64, std::Float64)
    # For very small numbers (less than 0.0001), use 5 decimal places
    if mean < 0.0001
        return @sprintf("%.5f ± %.5f", mean, std)
    # For small numbers (less than 0.01), use 4 decimal places
    elseif mean < 0.01
        return @sprintf("%.4f ± %.4f", mean, std)
    # For small numbers (less than 0.1), use 3 decimal places
    elseif mean < 0.1
        return @sprintf("%.3f ± %.3f", mean, std)
    # For regular numbers, use 2 decimal places
    else
        return @sprintf("%.2f ± %.2f", mean, std)
    end
end

function generate_latex_table(results::Dict)
    stats = calculate_statistics(results)
    
    # Start LaTeX table
    table = """
    \\begin{tabular}{l|c|c|c|c|}
    Method & \$\\beta\$ Coverage & \$i_{0}\$ Coverage & MSE \$\\beta\$ & MSE \$i_{0}\$\\\\
    \\hline
    """
    
    # Add rows for each group (marked with *, ∘, ★)
    markers = ["*", "\\circ", "\\star"]
    sizes = ["10", "100", "1000"]
    methods = ["RxInfer", "NUTS", "Sliced"]
    
    for (size_idx, size) in enumerate(sizes)
        for method in methods
            key = method * size
            if haskey(stats, key)
                marker = markers[size_idx]
                if method == "RxInfer"
                    table *= "$(marker) RBBVI & "
                else
                    table *= (method == "NUTS" ? "NUTS & " : "LSS & ")
                end
                
                # Add values with error margins
                means = stats[key].means
                stds = stats[key].stds
                
                table *= join([format_value(means[i], stds[i]) for i in 1:4], " & ")
                table *= "\\\\\n"
            end
        end
        table *= "\\hline\n"
    end
    
    # Close table
    table *= "\\end{tabular}"
    
    return table
end

# Generate the table
latex_table = generate_latex_table(results)
println(latex_table)