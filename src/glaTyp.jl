"""
    GilaTypes

This module provides the core abstract types and interfaces used across the Gila package.
It helps break circular dependencies between modules.
"""
module GilaTypes

using LinearAlgebra

export GlaSlv, AbstractGlaOpr, AbstractGlaVacOpr, dflPrc

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

#= Declared here, with the methods in GilaOperators, so the solvers can query
the adjoint state of an operator they are handed. =#
function isadjoint end

# One labelled row of a multi-line show; a multi-line val re-indents to align.
# Shared by GlaVol (GilaVolumes) and the operator show (GilaOperators).
function _shwRow(io::IO, lbl::AbstractString, val::AbstractString)
    lines = split(val, '\n')
    print(io, "\n  ", rpad(lbl, 14), "  ", lines[1])
    for line in lines[2:end]
        print(io, "\n", " "^18, line)
    end
end

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

end # module
