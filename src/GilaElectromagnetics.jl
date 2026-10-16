"""
GilaElectromagnetics implements single (complex) frequency electromagnetic Green
functions between generalized source and target cuboid ``volumes''. Technical 
details are available in the supporting document files.

Author: Sean Molesky 
Distribution: The code distributed under GNU LGPL.
"""
module GilaElectromagnetics

using KernelAbstractions: CPU, get_backend
using Adapt: adapt
export CPU, get_backend, adapt

export GlaSlv, AbstractGlaOpr, AbstractGlaVacOpr
export GlaVol
export GlaCmpVol, CompositeVolume, refine, regions, nregions, coordinates, cellvolumes, finest
export GlaFld, MultiScaleField, discretize!, regionview, eachregion, regrid
export GMRESSolver, GCRODRSolver, RcySpc, BiCGStabSolver, MixPrcRfn, SlvLog, solve, isvarying
export MixedPrecisionRefinement
export GlaOprVac, InvSctOpr, SctOpr, GlaOpr, SusOpr
export VacuumGreenOperator, InverseScatteringOperator, ScatteringOperator, GreenOperator, SusceptibilityOperator
export isadjoint, isselfoperator, isexternaloperator, isoverlappingoperator, isquasistatic, adjoint!, vac, slv, sus, asym, sym, setSus!, setBlc!

"""
    dflPrc

Storage precision (`Float32`) used by every unparameterized constructor. Request
another precision with the type parameter (e.g. `GlaOpr{Float64}(...)`).
"""
const dflPrc = Float32

"""
    GlaSlv

Abstract base type for all solvers in the Gila package. All concrete solver types
must subtype this type.
"""
abstract type GlaSlv end

"""
    AbstractGlaOpr{T<:AbstractFloat}

Abstract base type for all operators in the Gila package. All concrete operator types
must subtype this type. The operators are matrix free and are not `AbstractArray`s:
they carry `size`, `mul!` and `\\` rather than an element at a time. `T` is the real
storage precision of the operator, so the operator acts on `Complex{T}` data.
"""
abstract type AbstractGlaOpr{T<:AbstractFloat} end

"""
    AbstractGlaVacOpr{T}

Abstract base type for the vacuum Green operators, the ones built from geometry
alone with no material in them. Scattering operators are not vacuum operators;
they subtype `AbstractGlaOpr` directly.

Traits shared by every vacuum operator are defined once on this type, so a
subtype only overrides the ones where it differs.

# Interface

A subtype has to implement:
- `Base.size(opr)`: The number of rows and columns of the operator as a matrix
- `mulAct!(opr, act::AbstractVector{Complex{T}})`: The matrix-vector product on a
  flat vector (mutating)
- `adjoint!(opr)`: The adjoint, with the value semantics of its docstring
- `KernelAbstractions.get_backend(opr)`: The backend the operator computes on
- `isadjoint(opr)`: Whether the operator is the adjoint of the Green operator
- `isselfoperator(opr)`: Whether the source and target geometry are the same
- `isexternaloperator(opr)`: Whether the source and target geometry are disjoint
- `glaSze(opr)`: The target and source sizes in tensor form

`isoverlappingoperator` defaults to `false` here, `eltype` to `Complex{T}` on
`AbstractGlaOpr`, and `slv` to a `GCRODRSolver`. Override any of the three when
it does not hold.
"""
abstract type AbstractGlaVacOpr{T} <: AbstractGlaOpr{T} end

include("glaVol.jl")
include("glaFld.jl")

# The G₀ kernel: memory, generation, FFT action
include("vacuum/glaVac.jl")

include("glaSlv.jl")

# The linear operators built on the kernel
include("operators/glaOpr.jl")

end
