using JuMP, Convex, ComplexOptInterface, SCS
using Combinatorics, LinearAlgebra

include("BosonicSymmetry.jl")

"""Maximum visibility (w.r.t. random noise) s.t. the DPS criterion is certified."""
function maximally_mixed_distance(state, local_dim, n::Integer=3; ppt::Bool=true)
    # Constants
    dim = size(state, 1)
    noise = eye(dim) / dim^2
    dims = repeat([local_dim], n + 1) # Alice's dim. + `n` copies of Bob.
    P = kron(eye(local_dim), symmetric_projection(local_dim, n)) # Bosonic subspace projector.
    Qdim = local_dim * binomial(n + local_dim - 1, local_dim - 1)  # Dim. extension w/ bosonic symmetries.

    problem = Model(SCS.Optimizer)
    COI = ComplexOptInterface # Adds complex number support do JuMP.
    COI.add_all_bridges(problem)

    # Optimization variables
    @variable(problem, 0 <= vis <= 1)
    Q = @variable(problem, [1:Qdim, 1:Qdim] in ComplexOptInterface.HermitianPSDCone())
    if ppt
        # Dummy vars. to enforce PPT (not possible directly in ComplexOptInterface).
        fulldim = prod(dims)
        PSD = @variable(problem, [1:fulldim, 1:fulldim] in ComplexOptInterface.HermitianPSDCone())
    end

    # Constraints
    noisy_state = vis * state + (1 - vis) * noise
    @constraint(problem, noisy_state .== ptr(P * Q * P', 3:n+1, dims))
    if ppt
        ssys = Int.(1:ceil(n / 2) + 1)
        @constraint(problem, PSD .== ptransp(P * Q * P', ssys, dims))
    end

    # Solution
    @objective(problem, Max, vis)
    @show problem
    optimize!(problem)
    @show solution_summary(problem, verbose=true)
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

"""Example for the GHZ state."""
function ghz_entanglement(dim, n; ppt=true)
    @show dim n ppt
    @time maximally_mixed_distance(ghz(dim), dim, n, ppt=ppt)
end

#==========================
Example
==========================#
"""GHZ state."""
function ghz(d::Integer=2, parties::Integer=2, ket::Bool=false)
    ghz = zeros(ComplexF64, d^parties)
    offset = 0
    for p in 0:parties-1
        offset += d^p
    end
    for p in 0:d-1
        ghz[p * offset + 1] = 1 / sqrt(d)
    end
    if ket
        return ghz
    end
    ghz * ghz'
end