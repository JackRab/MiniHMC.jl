"""
Implementation of the (KISS) Hamiltonian Monte Carlo.
"""
module MiniHMC

export
    # hmc sampling
    hamiltonianMC

using Distributions
using Parameters: @with_kw, @unpack

include("hamiltonianMC.jl")

end #module
