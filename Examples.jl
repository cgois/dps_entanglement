include("DPS.jl")

#===========================
Helper functions
===========================#

"""GHZ state."""
function ghz(d::Integer=2; parties::Integer=2, ket::Bool=false)
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

"""Isotropic state."""
function isotropic(d::Integer=2, vis::Float64=1.0)
    vis * ghz(d, parties=2, ket=false) + (1 - vis) * I(d^2) / (d^2)
end

#===========================
Examples
===========================#

"""From arXiv:1310.3530, a bipartite state has a N = 2 symmetric extension
iff tr(rho_B)^2 >= tr(rho_AB^2) - 4 sqrt(det rho_AB)."""
rho = [1 0 0 -1; 0 1 1/2 0; 0 1/2 1 0; -1 0 0 1]
@time maximally_mixed_distance(rho, 2, 2, ppt=false)

"""Isotropic states are entangled for visibility above 1 / (d + 1)"""
# This should be separable (i.e., objective value = 1):
@time maximally_mixed_distance(isotropic(2, 1/3), 2, 2, ppt=true)
# This should be entangled (i.e., objective value < 1):
@time maximally_mixed_distance(isotropic(2, 1/3 + 0.01), 2, 2, ppt=true)