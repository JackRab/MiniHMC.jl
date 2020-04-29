"""
Implementation of the (KISS) Hamiltonian Monte Carlo.
"""
module MiniHMC

# hmc sampling
export hamiltonianMC

using Distributions, LinearAlgebra
using ArgCheck
import ForwardDiff

include("hamiltonianMC.jl")

end #module
