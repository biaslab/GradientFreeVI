using Distributions
using Turing
using OrdinaryDiffEq

function sir_ode_solve(problem, l, i₀, β)
    I = i₀*1000.0
    S = 1000.0 - I
    u0 = [S, I, 0.0, 0.0]
    p = [β, 10.0, 0.25]
    prob = remake(problem; u0=u0, p=p)
    sol = solve(prob, Tsit5(), saveat = 1.0)
    sol_C = view(sol, 4, :) # Cumulative cases
    sol_X = Array{eltype(sol_C)}(undef, l)
    @inbounds @simd for i in 1:length(sol_X)
        sol_X[i] = sol_C[i + 1] - sol_C[i]
    end
    return sol_X
end;

function simulate_data(l, i₀, β)
    prob = ODEProblem(sir_ode!, [990.0, 10.0, 0.0, 0.0], (0.0, l), [β, 10.0, 0.25])
    X = sir_ode_solve(prob, l, i₀, β)
    Y = rand.(Poisson.(X))
    return X, Y
end;

mutable struct SirOdeSolver!
    counter::Int
end 

function sir_ode!(du,u,p,t)
    (S,I,R,C) = u
    (β,c,γ) = p
    N = S+I+R
    infection = β*c*I/N*S
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
        du[4] = infection
    end
    nothing
end;

function (solver::SirOdeSolver!)(du,u,p,t)
    solver.counter += 1
    sir_ode!(du,u,p,t)
end

Turing.@model function bayes_sir(y, solver)
    # Calculate number of timepoints
    l = length(y)
    i₀  ~ Uniform(0.0,1.0)
    β ~ Uniform(0.0,1.0)
    I = i₀*1000.0
    u0=[1000.0-I,I,0.0,0.0]
    p=[β,10.0,0.25]
    tspan = (0.0,float(l))
    prob = ODEProblem(solver,
            u0,
            tspan,
            p)
    sol = solve(prob,
                Tsit5(),
                saveat = 1.0)
    sol_C = Array(sol)[4,:] # Cumulative cases
    sol_X = sol_C[2:end] - sol_C[1:(end-1)]
    l = length(y)
    for i in 1:l
        y[i] ~ Poisson(abs(sol_X[i]))
    end
end;

# Convenience function to check if the true value is within the credible interval
function in_credible_interval(x, lwr=0.025, upr=0.975)
    return x >= lwr && x <= upr
end;

function run_ode_experiment(i₀_true, β_true, sampler, nsamples)
    nsims = 100
    
    Random.seed!(1234)
    seeds = map(abs, rand(Int, nsims))

    l = 40
    i₀_mean = Array{Float64}(undef, nsims)
    β_mean = Array{Float64}(undef, nsims)
    i₀_coverage = Array{Float64}(undef, nsims)
    β_coverage = Array{Float64}(undef, nsims)
    i₀_mse = Array{Float64}(undef, nsims)
    β_mse = Array{Float64}(undef, nsims)
    number_of_odes_solver = Array{Int}(undef, nsims)

    for i in 1:nsims
        Random.seed!(seeds[i])
        _, Y_sim = simulate_data(l, i₀_true, β_true)
        solver! = SirOdeSolver!(0)
        r = sample(bayes_sir(Y_sim, solver!), sampler, nsamples, verbose=false, progress=false)
        i₀_mean[i] = mean(r[:i₀])
        i0_cov = sum(r[:i₀] .<= i₀_true) / length(r[:i₀])
        β_mean[i] = mean(r[:β])
        b_cov = sum(r[:β] .<= β_true) / length(r[:β])
        i₀_coverage[i] = i0_cov
        β_coverage[i] = b_cov
        number_of_odes_solver[i] = solver!.counter
        i₀_mse[i] = (i₀_mean[i] - i₀_true)^2
        β_mse[i] = (β_mean[i] - β_true)^2
    end;

    in_cred_interval_i0 = map(in_credible_interval, i₀_coverage)
    in_cred_interval_β = map(in_credible_interval, β_coverage)

    return mean(in_cred_interval_i0), mean(in_cred_interval_β), mean(number_of_odes_solver), mean(i₀_mse), mean(β_mse)
end 