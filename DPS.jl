using JuMP, Convex, ComplexOptInterface, Mosek, MosekTools, SCS
using Combinatorics, LinearAlgebra
using JLD2

include("MUBs.jl")
include("BosonicSymmetry.jl")

"""SDP to certify Schmidt number of a steering assemblage."""
function srk_assemblage_distance(ass, sn::Integer, n::Integer=3;
                                 solver::String="mosek", ppt::Bool=true)
    #==========================
    Constants
    ==========================#
    nx = length(ass)
    na = size(ass[1],1)
    dB = size(ass[1][1],1)
    dBBp = sn * dB  # Dimension of BB' subsystem.
    dims = [sn; repeat([dBBp], n)]
    Qdim = sn * binomial(n + dBBp - 1, dBBp - 1)  # Dim. extension w/ bosonic symm.
    noise = eye(dB) ./ (na * dB)
    entangling = sn .* kron(ghz(sn), eye(dB))
    P = kron(eye(sn), par_symmetric_projection(dBBp, n))  # Bosonic subspace projector.
    E = kron(eye(sn), kronpower(kron(eye(sn), hadamard_matrix(dB)), n))  # Unitary between meas.
    problem = setsolver(solver)
    print("Created constants.")
    #==========================
    Optimization variables
    ==========================#
    @variable(problem, 0 <= vis <= 1)
    Q = [@variable(problem, [1:Qdim, 1:Qdim]
         in ComplexOptInterface.HermitianPSDCone()) for a in 1:na]
    if ppt
        # These are dummy variables just to enforce PPT constraints because it is not possible
        # to write PPT constraints literally in ComplexOptInterface yet:
        fulldim = prod(dims)
        PSD = [@variable(problem, [1:fulldim, 1:fulldim]
               in ComplexOptInterface.HermitianPSDCone()) for a in 1:na]
    end
    #==========================
    Constraints
    ==========================#
    for a in 1:na
        # Symmetric extensions/assemblage simulation constraints:
        noisy_ass = vis * ass[1][a] + (1 - vis) * noise
        # TODO: Write directly in symm. subsace and project assemblage instead
        extension = ptr(P * Q[a] * P', 3:n+1, dims) * entangling
        @constraint(problem, noisy_ass .== ptr(extension, [1, 2], [sn, sn, dB]))
        # PPT constraints:
        if ppt
            ssys = Int.(1:ceil(n / 2) + 1)
            @constraint(problem, PSD[a] .== ptransp(P * Q[a] * P', ssys, dims))
        end
    end
    # Nonsignalling constraints:
    @expression(problem, sumQ, sum(Q))
    for x in 1:nx-1
        Ex = P' * (E^x) * P
        @constraint(problem, sumQ .== Ex * sumQ * Ex')
    end
    print("Set constraints.")
    #==========================
    Solution and output
    ==========================#
    @objective(problem, Max, vis)
    @show problem
    optimize!(problem)
    @show solution_summary(problem, verbose=true)
    @show objective_value(problem) - dual_objective_value(problem)
    problem, tovalue(Q, P, E, na, nx), P
end

"""Generates the GHZ-MUB assemblage, calls the SDP and save results."""
function main(dim, nmubs, sn, n; solver="scs", ppt=true, exportresult=false)
    print("Using MUB symmetries.\n")
    @show dim nmubs sn n
    ass = assemblage(ghz(dim), mubmeas(dim)[1:nmubs])
    res, Q, P = @time srk_assemblage_distance(ass, sn, n, solver=solver, ppt=ppt)
    optval = objective_value(res)
    if exportresult
        @save "$dim-$nmubs-$sn-$n.jld2" Q P
    end
    res, Q, P
end

#==========================
Helper functions
==========================#

eye(d) = Matrix{ComplexF64}(I(d))

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

"""Prepares an assemblage from state `st` and measurements `meas`."""
function assemblage(st, meas)
    dA = size(meas[1][1], 1)
    dB = div(size(st, 1), dA)
	ass = Vector{Vector{Array{ComplexF64}}}(undef, length(meas))
    for x in 1:length(meas)
        ass[x] = [Convex.partialtrace(kron(meas[x][a], eye(dB)) * st, 1, [dA,dB])
                  for a in 1:length(meas[x])]
    end
    ass
end

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

"""A^otimes_n"""
kronpower(A, n) = kron([A for _ in 1:n]...)

"""Set solver parameters."""
function setsolver(solver="mosek")
    if solver == "mosek"
        problem = Model(Mosek.Optimizer)
    elseif solver == "scs"
        # These params. accelerate convergence but increase duality gap of solution.
        EPS_ABS, EPS_REL = 1E-4, 2 * 1E-4
        @show EPS_ABS EPS_REL
        problem = Model(optimizer_with_attributes(SCS.Optimizer,
                                                  "eps_abs" => EPS_ABS,
                                                  "eps_rel" => EPS_REL,
                                                  "linear_solver" => SCS.IndirectSolver))
    end
    COI = ComplexOptInterface # Adds complex number support do JuMP.
    COI.add_all_bridges(problem)
    problem
end

"""Converts result to Julia datatypes."""
function tovalue(Q, P, E, na, nx)
    M = Vector{Vector{Matrix{ComplexF64}}}(undef, nx)
    for x in 1:nx
        M[x] = [(E^x) * (P * value.(Q[a]) * P') * (E^x)' for a in 1:na]
    end
    M
end

# """Check wether the extended operators `Q` satisfy the constraints."""
# function checkresult(Q, P, vis, dim, nmubs, sn, n; ATOL=1E-6)
#     check = true
#     ass = assemblage(ghz(dim), mubmeas(dim)[1:nmubs])
#     dB = size(ass[1][1],1)
#     dBBp = sn * dB  # Dimension of BB' subsystem.
#     nx = length(ass)
#     na = size(ass[1],1)
#     noise = eye(dB) ./ (na * dB)
#     entangling = sn .* kron(ghz(sn), eye(dB))
#     dims = [sn; repeat([dBBp], n)]

#     # Nonsignalling constraints:
#     for (x,xprime) in collect(combinations(1:nx, 2))
#         check = check && isapprox(sum(Q[x][:]), sum(Q[xprime][:]), atol=ATOL)
#         @show "Nonsignalling (x=$x, x'=$xprime)?" check
#     end
#     # Symmetric extensions/assemblage simulation constraints:
#     for x in 1:nx
#         for a in 1:na
#             noisy_ass = vis * ass[x][a] + (1 - vis) * noise
#             extension = ptr(P * Q[x][a] * P', 3:n+1, dims) * entangling
#             check = check && isapprox(noisy_ass, ptr(extension, [1, 2], [sn, sn, dB]), atol=ATOL)
#             # TODO: Add PPT constraints checking
#             @show "Assemblage simulation (x=$x, a=$a)?" check
#         end
#     end
#     check
# end


#========================================================================
To run from command line, the parameters are:

- `dim`::Int -> Dimension of the GHZ state.
- `nmubs`::Int -> Number of MUB measurements on the assemblage.
- `sn`::Int -> Schmidt number to test for infeasibility.
- `n`::Int -> Hierarchy level.
- `solver`::String -> Either "mosek" or "scs".
- `ppt`::Bool -> Whether to enforce PPT constraints on the extensions.

Results will be saved under a file named "dim-nmubs-sn-n.jdl2".
========================================================================#
if abspath(PROGRAM_FILE) == @__FILE__
    dim = parse(Int64, ARGS[1])
    nmubs = parse(Int64, ARGS[2])
    sn = parse(Int64, ARGS[3])
    n = parse(Int64, ARGS[4])
    solver = ARGS[5]
    ppt = parse(Bool, ARGS[6])
    main(dim, nmubs, sn, n, solver=solver, ppt=ppt, exportresult=true)
end