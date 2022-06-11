using Convex, SCS
using Combinatorics, LinearAlgebra

include("BosonicSymmetry.jl")

"""Maximum visibility (w.r.t. random noise) s.t. the DPS criterion is certified.
An objective value of 1 means feasibility (unconclusive), and < 1 means entanglement."""
function maximally_mixed_distance(state, local_dim, n::Integer=3; ppt::Bool=true)
    # Constants
    dim = size(state, 1)
    noise = eye(dim) / dim^2
    dims = repeat([local_dim], n + 1) # Alice's dim. + `n` copies of Bob.
    P = kron(eye(local_dim), symmetric_projection(local_dim, n)) # Bosonic subspace projector.
    Qdim = local_dim * binomial(n + local_dim - 1, local_dim - 1)  # Dim. extension w/ bosonic symmetries.

    # Program setup
    vis = Variable()
    Q = HermitianSemidefinite(Qdim)
    problem = maximize(vis)

    noisy_state = vis * state + (1 - vis) * noise
    problem.constraints += noisy_state == ptr(P * Q * P', 3:n+1, dims)
    if ppt
        ssys = Int.(1:ceil(n / 2) + 1)
        problem.constraints += ptransp(P * Q * P', ssys, dims) in :SDP
    end

    # Solution
    solve!(problem, SCS.Optimizer, silent_solver=false)
    problem, Q
end

#==========================
Helper functions
==========================#

eye(d) = Matrix{ComplexF64}(I(d))

"""Partial trace for multiple subsystems."""
function ptr(oper, syss, DIMS)
    dims = copy(DIMS)
    for sys in 1:length(syss)
        oper = Convex.partialtrace(oper, syss[sys] - sys + 1, dims)
        deleteat!(dims, syss[sys] - sys + 1)
    end
    oper
end

"""Partial transpose for multiple subsystems."""
function ptransp(oper, syss, dims)
    for sys in syss
        oper = partialtranspose(oper, sys, dims)
    end
    oper
end