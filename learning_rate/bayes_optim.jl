using Pkg
Pkg.activate(".")
using RxInfer
using Flux, MLDatasets
using Flux: train!, onehotbatch
using ExponentialFamilyProjection
using ExponentialFamily
using ExponentialFamilyProjection.Manopt
using ProgressMeter
using DataStructures
using RecursiveArrayTools
using JSON
using Plots
using Random
using RxInfer.ReactiveMP
pgfplotsx()
Random.seed!(123)

struct NNFused{E,T} <: DiscreteMultivariateDistribution
    ε::E
    X::T
end;

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
        # accuracy = mean(Flux.onecold(model(val_data.data[1])) .== Flux.onecold(val_data.data[2]))
        # if j > 10 && accuracy < 0.75
        #     break
        # end
        # pl(accuracy) && break
    end
    return model
end

function BayesBase.logpdf(fused_neural_net::NNFused, ε, y::AbstractMatrix{<:Real})
    model = Chain(
        Dense(784, 256, relu),
        Dropout(0.45),
        Dense(256, 256, relu),
        Dropout(0.45),
        Dense(256, 10, relu),
        softmax
    )
    train_data = Flux.DataLoader((fused_neural_net.X, y), shuffle=true, batchsize=128)
    trained_nn = train_nn(model, fused_neural_net.ε, 20, train_data, train_data)
    ps = trained_nn(x_val_flat)

    sumlpdf = mean(zip(eachcol(y_val_flat), eachcol(ps))) do (sy, p)
        return clamp(logpdf(Categorical(p[1:10]), argmax(sy)), -1.5f1, 1.0f3,)
    end

    return sumlpdf
end;

BayesBase.logpdf(fused_neural_net::NNFused, y::AbstractMatrix{<:Real}) = BayesBase.logpdf(fused_neural_net, fused_neural_net.ε, y)

slice_size = 3000
# Load training data (images, labels)
x_train, y_train = MNIST(split=:train)[:];
x_test, y_test = MNIST(split=:test)[:];
x_test = Flux.flatten(x_test)
x_val = x_test[:, 1:1000]
x_val_flat = Flux.flatten(x_val)
y_val = y_test[1:1000]
y_val_flat = Flux.onehotbatch(y_val, 0:9)
x_test = x_test[:, 1001:end]
y_test = y_test[1001:end]

# cutted
x_cutted = x_train[:, :, 1:slice_size];
y_cutted = y_train[1:slice_size];

@node NNFused Stochastic [y, ε, X];

@node Exponential Stochastic [out, rate]
@model function bayes_optim(y, X, dist)
    if dist == Exponential
        ε ~ Exponential(0.00001)
    elseif dist == Gamma
        ε ~ Gamma(1.0, 1 / 300.0)
    elseif dist == InverseGamma
        ε ~ InverseGamma(1.0, 1 / 300.0)
    end
    y ~ NNFused(ε, X)
end
data = Dict()
resulting_lrs = Dict()
accuracies = Dict()
global record
for dist in [Exponential, Gamma, InverseGamma]
    if dist == Exponential
        global record = [RecordEntry(ArrayPartition{Float64,Tuple{Vector{Float64}}}, :p), RecordCost()]
    else
        global record = [RecordEntry(ArrayPartition{Float64,Tuple{Vector{Float64},Vector{Float64}}}, :p), RecordCost()]
    end

    @constraints function nn_constraints()
        parameters = ProjectionParameters(
            strategy=ExponentialFamilyProjection.ControlVariateStrategy(nsamples=30),
            niterations=100
        )
        q(ε)::ProjectedTo(dist; parameters=parameters, kwargs=(record=record,))
    end

    function ExponentialFamilyProjection.getinitialpoint(::ExponentialFamilyProjection.ControlVariateStrategy, M::ExponentialFamilyProjection.AbstractManifold, parameters::ProjectionParameters)
        if dist == Exponential
            return ArrayPartition([-300.0])
        elseif dist == Gamma
            return ArrayPartition([0.0], [-300.0])
        elseif dist == InverseGamma
            return ArrayPartition([-2.0], [-(1 / 300)])
        end
    end

    result = infer(
        model=bayes_optim(X=Flux.flatten(x_cutted), dist=dist),
        data=(y=Flux.onehotbatch(y_cutted, 0:9),),
        constraints=nn_constraints(),
        showprogress=false,
        options=(
            # Read `https://reactivebayes.github.io/RxInfer.jl/stable/manuals/inference/undefinedrules/`
            rulefallback=NodeFunctionRuleFallback(),
        )
    )
    distributions = ExponentialFamilyDistribution.(dist, record[1].recorded_values)
    distributions = convert.(dist, distributions)
    data[dist] = [distributions, record[2].recorded_values]
end

dist = NNFused(1.0f-3, Flux.flatten(x_cutted));
y = Flux.onehotbatch(y_cutted, 0:9);
culp = ContinuousUnivariateLogPdf(UnspecifiedDomain(), (e) -> logpdf(dist, e, y))
right = Gamma(1.0, 1 / 300.0)

open("rxinfer_results.json", "w") do f
    JSON.print(f, data)
end

moving_average(vs, n) = [sum(@view vs[i:(i+n-1)]) / n for i in 1:(length(vs)-(n-1))]
minimum_so_far(vs) = [minimum(@view vs[1:i]) for i in 1:length(vs)]
stopping_threshold = 0.029
indices = []
for dist in [Exponential, Gamma, InverseGamma]
    index = findfirst(x -> abs(x) < stopping_threshold, diff(data[dist][2]))
    if isnothing(index)
        index = length(data[dist][2])
    end
    push!(indices, index)
    resulting_lrs[dist] = data[dist][1][index]
end
@show indices
m, v = ReactiveMP.approximate_meancov(ghcubature(30 * minimum(indices)), x -> pdf(culp, x), right)
@show m, v  # This will still print to the console
shape = m^2 / v
scale = v / m

resulting_lrs[ghcubature] = Gamma(shape, scale)

open("rxinfer_results_lrs.json", "w") do f
    JSON.print(f, [[rand(resulting_lr, 50) for (_, resulting_lr) in resulting_lrs], Dict(name => mean(resulting_lr) for (name, resulting_lr) in resulting_lrs)])
end

@show resulting_lrs
@show [(name, mean(resulting_lr)) for (name, resulting_lr) in resulting_lrs]
@show [(name, median(resulting_lr)) for (name, resulting_lr) in resulting_lrs]

