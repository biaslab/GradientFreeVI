using Pkg
Pkg.activate(".")
using Turing
using Base
using Flux, MLDatasets
using Flux: train!, onehotbatch
using ProgressMeter
using DataStructures
using RecursiveArrayTools
using JSON
using SliceSampling
using Distributions
using Bijectors
using Random
using JSON

Random.seed!(123)

struct NNFused{E,T,L} <: DiscreteMatrixDistribution
    ε::E
    X::T
    caller::L
end;

Base.length(nn::NNFused) = size(nn.X)[2]

mutable struct TrainNNCaller
    counter::Int
end

function train_nn(model, ε, n_iter, train_data, val_data, pl_min_dist=0.01)
    optim = Flux.setup(Flux.ADAM(ε), model)
    f = (accuracy) -> -accuracy
    # pl = Flux.plateau(f, 5; min_dist=pl_min_dist)
    @showprogress for j in 1:n_iter
        for (i, datapoint) in enumerate(train_data)
            input, labels = datapoint
            val, grads = Flux.withgradient(model) do m
                # Any code inside here is differentiated.
                # Evaluation of the model and loss must be inside!
                result = m(input)
                Flux.logitcrossentropy(result, labels)
            end

            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end
            Flux.update!(optim, model, grads[1])

        end
    end
    return model
end

function (caller::TrainNNCaller)(model, ε, n_iter, train_data, val_data, pl_min_dist=0.01)
    caller.counter += 1
    return train_nn(model, ε, n_iter, train_data, val_data, 0.01)
end

function Distributions.logpdf(fused_neural_net::NNFused, y::AbstractMatrix{<:Real})
    model = Chain(
        Dense(784, 256, relu),
        Dropout(0.45),
        Dense(256, 256, relu),
        Dropout(0.45),
        Dense(256, 10, relu),
        softmax
    )
    train_data = Flux.DataLoader((fused_neural_net.X, y), shuffle=true, batchsize=128)
    trained_nn = fused_neural_net.caller(model, fused_neural_net.ε, 20, train_data, train_data)
    ps = trained_nn(x_val_flat)

    sumlpdf = mean(zip(eachcol(y_val_flat), eachcol(ps))) do (sy, p)
        return clamp(logpdf(Categorical(p[1:10]), argmax(sy)), -1.5f1, 1.0f3,)
    end

    return sumlpdf
end;

@model function bayes_optim(y, X, caller)
    ε ~ Gamma(1.0, 1 / 300.0)
    y ~ NNFused(ε, X, caller)
end

slice_size = 3000
# Load training data (images, labels)
begin
    x_train, y_train = MNIST(split=:train)[:]
    x_test, y_test = MNIST(split=:test)[:]
    x_test = Flux.flatten(x_test)
    x_val = x_test[:, 1:1000]
    x_val_flat = Flux.flatten(x_val)
    y_val = y_test[1:1000]
    y_val_flat = Flux.onehotbatch(y_val, 0:9)
    x_test = x_test[:, 1001:end]
    y_test = y_test[1001:end]

    # cutted
    x_cutted = x_train[:, :, 1:slice_size]
    y_cutted = y_train[1:slice_size]
end

sampler = externalsampler(LatentSlice(10))
caller = TrainNNCaller(0)
result = sample(bayes_optim(Flux.onehotbatch(y_cutted, 0:9), Flux.flatten(x_cutted), caller), sampler, 5)


data = JSON.parsefile("rxinfer_results_lrs.json")

push!(data[1], vec(result[:ε].data))
data[2]["Turing"] = mean(result[:ε].data)

open("rxinfer_results_lrs.json", "w") do f
    JSON.print(f, data)
end