# Entanglement certification via symmetric extensions.
Implementation of the [symmetric extensions criterion](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.69.022308) with bosonic symmetry and PPT constraint.

Uses the `SCS` solver, `Convex` for modelling, and `Combinatorics` to generate the projection onto the symmetric subspace, so these should be installed.

Check `Examples.jl` for... examples.

## Files
- `DPS.jl` -- Implements symmetric extensions SDP with bosonic symmetries and PPT criterion.
- `BosonicSymmetry.jl` -- Functions to generate the projector onto/out from the symmetric subspace.
- `Examples.jl` -- Provides simple examples for isotropic and extendible states.
