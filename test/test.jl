using MCMCChains
using StatsPlots

# dimension of the parameters
D = 10
# initial position (random)
θ_init = rand(D)
# likelihood function
ℒ(θ) = -logpdf(MvNormal(zeros(D), ones(D)), θ)
n_samples = 3000
# step size
ϵ = .798
# length of the steps in leapfrog
L = 2
samples = Array{Array{Float64, 1}, 1}(undef, 0)
push!(samples, θ_init)
for i in 1:n_samples
    θ = hamiltonianMC(ℒ, ϵ, L, θ_init)
    if  θ !== θ_init
        push!(samples, θ)
    end
end

chn = Chains(samples[end-999:end])
plot(chn)
