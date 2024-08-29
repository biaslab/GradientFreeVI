using RxInfer, Random, OrdinaryDiffEq, ExponentialFamilyProjection, Optimisers, StableRNGs, StatsFuns, Plots, StaticArrays
using ProgressMeter
import BayesBase

include("bayes_sir_turing_spec.jl")

function simulate_data(l, i₀, β)
    prob = ODEProblem(sir_ode!, [990.0, 10.0, 0.0, 0.0], (0.0, l), [β, 10.0, 0.25])
    X = sir_ode_solve(prob, l, i₀, β)
    Y = rand.(Poisson.(X))
    return X, Y
end;

mutable struct SirOdeSolver
    counter::Int
end

function sir_ode(u, p, t)
    (S, I, R, C) = u
    (β, c, γ) = p
    N = S + I + R
    infection = β * c * I / N * S
    recovery = γ * I
    return [
        -infection,
        infection - recovery,
        recovery,
        infection
    ]
end

function (s::SirOdeSolver)(u, p, t)
    s.counter += 1
    return sir_ode(u, p, t)
end

function sir_ode_solve(problem, l, i₀, β)
    I = i₀ * 1000.0
    S = 1000.0 - I
    u0 = [S, I, 0.0, 0.0]
    p = [β, 10.0, 0.25]
    prob = remake(problem; u0=u0, p=p)
    sol = solve(prob, Tsit5(), saveat=1.0)
    sol_C = view(sol, 4, :) # Cumulative cases
    sol_X = Array{eltype(sol_C)}(undef, l)
    @inbounds @simd for i in 1:length(sol_X)
        sol_X[i] = sol_C[i+1] - sol_C[i]
    end
    return sol_X
end

struct ODEFused{I,B,L,F} <: DiscreteMultivariateDistribution
    i₀::I
    β::B
    l::L
    problem::F
end

function BayesBase.logpdf(ode::ODEFused, y::Vector)
    sol_X = sir_ode_solve(ode.problem, ode.l, ode.i₀, ode.β)
    # `sum` over individual entries of the result of the `ODE` solver
    sumlpdf = sum(zip(sol_X, y)) do (x_i, y_i)
        return logpdf(Poisson(abs(x_i)), y_i)
    end
    # `clamp` to avoid infinities in the beginning, where 
    # priors are completelly off
    return clamp(sumlpdf, -100000, Inf)
end

function BayesBase.insupport(ode::ODEFused, y::Vector)
    return true
end

function BayesBase.mean(p::PointMass{D}) where {D<:ODEProblem}
    return p.point
end

@node ODEFused Stochastic [y, i₀, β, l, problem]

@average_energy ODEFused (q_y::Any, q_i₀::Any, q_β::Any, q_l::Any, q_problem::Any) = begin

    r = logpdf(ODEFused(mean(q_i₀), mean(q_β), mean(q_l), mean(q_problem)), mean(q_y))
    @show r
end

RxInfer.@model function bayes_sir_rxinfer(y, s)
    l = length(y)
    problem = ODEProblem(s[1], [990.0, 10.0, 0.0, 0.0], (0.0, l), [0.05, 10.0, 0.25])

    i₀ ~ Beta(1.0, 1.0)
    β ~ Beta(1.0, 1.0)
    y ~ ODEFused(i₀, β, l, problem)
end

@initialization function sir_initialization()
    q(β) = Beta(1, 1)
    q(i₀) = Beta(1, 1)
end

@constraints function sir_constraints(nsamples, niterations)
    parameters = ProjectionParameters(
        strategy=ExponentialFamilyProjection.ControlVariateStrategy(nsamples=nsamples),
        niterations=niterations
    )

    # In principle different parameters can be used for different states
    q(i₀)::ProjectedTo(Beta; parameters=parameters)
    q(β)::ProjectedTo(Beta; parameters=parameters)

    # `MeanField` is required for `NodeFunctionRuleFallback`
    q(i₀, β) = MeanField()
end

function run_experiment(i₀_true, β_true, constraints, seed)

    l = 40
    solver = SirOdeSolver(0)
    Random.seed!(seed)
    _, Y = simulate_data(l, i₀_true, β_true)

    # @show mean(Y)

    current_marginals = Dict()
    FEs = []

    function on_marginal_update_callback(model, variable_name, posterior)
        current_marginals[variable_name] = posterior
    end

    function after_iteration_callback(model, iteration)
        problem = ODEProblem(sir_ode, [990.0, 10.0, 0.0, 0.0], (0.0, length(Y)), [0.05, 10.0, 0.25])
        rel_entropy = logpdf(ODEFused(mean(current_marginals[:i₀]), mean(current_marginals[:β]), length(Y), problem), Y)
        push!(FEs, -(rel_entropy - entropy(current_marginals[:i₀]) - entropy(current_marginals[:β])))
    end

    result = infer(
        model=bayes_sir_rxinfer(s=[solver]),
        data=(y=Y,),
        constraints=constraints,
        initialization=sir_initialization(),
        iterations=2,
        showprogress=true,
        # free_energy = true,
        # free_energy_diagnostics=RxInfer.ObjectiveDiagnosticCheckInfs(),
        options=(
            # Read `https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/undefinedrules/`
            rulefallback=NodeFunctionRuleFallback(),
        ),
        callbacks=(on_marginal_update=on_marginal_update_callback, after_iteration=after_iteration_callback,
        )
    )

    post_i = result.posteriors[:i₀][end]
    post_b = result.posteriors[:β][end]

    return result, solver.counter, in_credible_interval(cdf(post_i, i₀_true)), in_credible_interval(cdf(post_b, β_true)), (mean(post_i) - i₀_true)^2, (mean(post_b) - β_true)^2
end

function run_experiments(i₀_true, β_true, constraints)
    Random.seed!(1234)
    nsims = 100
    seeds = map(abs, rand(Int, nsims))


    i₀_coverage = Array{Float64}(undef, nsims)
    β_coverage = Array{Float64}(undef, nsims)
    i₀_mse = Array{Float64}(undef, nsims)
    β_mse = Array{Float64}(undef, nsims)
    ncalls = Array{Int}(undef, nsims)


    @showprogress for i in 1:nsims
        _, counter, crd_i, crd_b, m_i, m_b = run_experiment(i₀_true, β_true, constraints, seeds[i])
        i₀_coverage[i] = crd_i
        β_coverage[i] = crd_b
        ncalls[i] = counter
        i₀_mse[i] = m_i
        β_mse[i] = m_b
    end

    return mean(i₀_coverage), mean(β_coverage), mean(ncalls), mean(i₀_mse), mean(β_mse)
end

