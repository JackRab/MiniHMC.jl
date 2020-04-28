"""
Implementation of the (KISS) Hamiltonian Monte Carlo.
"""
module MiniHMC

export
    # hmc sampling
    hamiltonianMC

using Distributions, LinearAlgebra
import ForwardDiff

include("hamiltonianMC.jl")

end #module
