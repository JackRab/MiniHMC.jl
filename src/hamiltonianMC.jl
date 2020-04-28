# implemente a Hamiltonian Monte carlo

"""
Perform Hamiltonian Monte Carlo with hand-tuning ϵ and L.
- `ℒ`: the negative log likelihood function to sample from
- `ϵ`: a scalar for stepsize
- `L`: the length of steps for leapfrog integrator
- `q⁰`: the initial position of the parameters
Return a new position of q.
"""
function hamiltonianMC(ℒ, ϵ, L, q⁰::T
    ) where {T<:AbstractVecOrMat{<:AbstractFloat}}

    q = copy(q⁰)
    # first draw a momentum from a Standard Multivariate Normal
    p = rand(MvNormal(size(q, 1), 1))
    # store current p
    p⁰ = copy(p)

    # then apply L leapfrog steps to get a proposal of the new position and momentum
    q, p = leapfrog(ℒ, ϵ, L, p, q)

    # evaluate potential and kinetic engeries at start and end of trajectory
    current_U = ℒ(q⁰)
    current_K = .5 * dot(p⁰, p⁰)
    proposed_U = ℒ(q)
    proposed_K = .5 * dot(p, p)

    # Last reject or accept the proposal according to the Metropolis algorithm
    p_accept = min(1, exp(current_U+current_K-proposed_U-proposed_K))
    if rand() < p_accept
        # accept the proposal point
        return q
    else
        # reject, return the initial point
        return q⁰
    end
end

"""
Perform leapfrog of L steps. Based on Neal-2011.
- `ℒ`: the negative log likelihood function to sample from
- `ϵ`: a scalar for stepsize
- `L`: the length of steps for a leapfrog integrator
--------------
Return a new position/momentum of q/p.
"""
function leapfrog(ℒ, ϵ, L, p, q)
    q, p = copy(q), copy(p)

    # first make a half step for momentum
    p = p - .5 * ϵ * ForwardDiff.gradient(ℒ, q)
    for _ in 1:L-1
        # a full step for the position
        q = q + ϵ * p
        # a full step for the momentum
        p = p - ϵ * ForwardDiff.gradient(ℒ, q)
    end

    # a full step for the position
    q = q + ϵ * p
    # a half step at the end for momentum
    p = p - .5 * ϵ * ForwardDiff.gradient(ℒ, q)

    # nagate the momentum at the tend to make the proposal symmetric
    return q, -p
end
