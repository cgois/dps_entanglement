using Combinatorics, LinearAlgebra, SparseArrays

"""Projection operator onto the symmetric subspace."""
function symmetric_projection(dim::Int, copies::Int) 
    # The symmetric projection is the average over all permutation operators.
    P = zeros(dim^copies, dim^copies)
    perm_list = permutations(1:copies)
    for perm in perm_list
        P += permutation_operator(dim, perm)
    end
    # Inverse of projector onto the symmetric subspace.
    orthonormal_range(P / length(perm_list))
end

"""Threaded version of `symmetric_projection`."""
function par_symmetric_projection(dim::Int, copies::Int) 
    # The symmetric projection is the average over all permutation operators.
    perm_list = collect(permutations(1:copies))
    P = zeros(dim^copies, dim^copies, length(perm_list))
    Threads.@threads for i in 1:length(perm_list)
        P[:,:,i] = permutation_operator(dim, perm_list[i])
    end
    P = sum(P, dims=3)[:,:,1]
    # Inverse of projector onto the symmetric subspace.
    orthonormal_range(P / length(perm_list))
end

"""Threaded version of `symmetric_projection`."""
function par_symmetric_projection(dim::Int, copies::Int) 
    # The symmetric projection is the average over all permutation operators.
    perm_list = collect(permutations(1:copies))
    P = zeros(dim^copies, dim^copies, length(perm_list))
    Threads.@threads for i in 1:length(perm_list)
        P[:,:,i] = permutation_operator(dim, perm_list[i])
    end
    P = sum(P, dims=3)[:,:,1]
    # Inverse of projector onto the symmetric subspace.
    orthonormal_range(P / length(perm_list))
end

"""Unitary to permute subsystems of dimension `dim` w.r.t. vector `perm`."""
function permutation_operator(dim::Int, perm::Vector)
    # This trick only works up to a 62x62 permutation matrix because of basis conversion.
    BASIS=vcat(0:9, 'A':'Z', 'a':'z')
    n = length(perm)
    # `dim`-ary basis with length `n`.
    basis = unique(permutations(vcat([repeat([BASIS[i]], n) for i in 1:dim]...), n))
    basis = map(x -> getindex(x, perm), basis) # Permute each basis element.
    dec_perm = map(x -> parse(Int, join(x), base=dim) + 1, basis) # Change of basis.
    getindex(I(dim^n), dec_perm, :) # Permute the identity rows.
end

"""Orthonormal basis for the range of `A`."""
function orthonormal_range(A)
    decomp = svd(A, alg=LinearAlgebra.DivideAndConquer())
    rk = count(map(x -> !isapprox(x, 0, atol=1E-11), decomp.S)) # Rank.
    chopzeros(decomp.U[:,1:rk]) # The first `rk` columns of U span A.
end

"""Put numbers smaller than `ZERO_TOL` to 0."""
function chopzeros(it; ZERO_TOL=1E-11)
    map(x -> abs(x) < ZERO_TOL ? 0.0 : x, it)
end

"""Check if input is a projector onto the symmetric subspace."""
function check_symm_projection(P)
    # TODO
end