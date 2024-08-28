using Pkg
Pkg.activate(".")

using Plots
using JSON

pgfplotsx()

minimum_so_far(vs) = [minimum(@view vs[1:i]) for i in 1:length(vs)]

rxinfer_data = JSON.parsefile("rxinfer_results.json")


p = plot(30:30:3000, rxinfer_data["Exponential"][2], yaxis=:log, label="Exponential", xlabel="Number of Neural Networks trained", ylabel="Variational Free Energy", legend=:topright, size=(350, 233))
scatter!(p, [30 * 46], [rxinfer_data["Exponential"][2][46]], color="blue", primary=false, markershape=:x)
plot!(p, 30:30:3000, rxinfer_data["Gamma"][2], label="Gamma")
scatter!(p, [30 * 7], [rxinfer_data["Gamma"][2][7]], color=:red, primary=false, markershape=:x, markercolor=:red)
plot!(p, 30:30:3000, rxinfer_data["InverseGamma"][2], label="Inverse Gamma")
scatter!(p, [30 * 19], [rxinfer_data["InverseGamma"][2][19]], color=:green, primary=false, markershape=:x)
savefig(p, "rxinfer_results.tikz")

bayesopt_results = JSON.parsefile("bayesopt_results.json")

losses = bayesopt_results["losses"]
p = plot(1:1000, minimum_so_far(losses["RBF(length_scale=1)"][3:end]), label="RBF", xlabel="Number of Neural Networks trained", ylabel="Validation loss", yaxis=:log, legend=:topright, size=(350, 233))
plot!(p, 1:1000, minimum_so_far(losses["Matern(length_scale=2.5, nu=1.5)"][3:end]), label="Matern(2.5)")
plot!(p, 1:1000, minimum_so_far(losses["Matern(length_scale=1, nu=1.5)"][3:end]), label="Matern(1)")

plot!(p, 1:1000, minimum_so_far(losses["TPE"])[1:1000], label="TPE", xlabel="Number of Neural Networks trained", ylabel="Validation loss", yaxis=:log)
savefig(p, "tpe_results.tikz")