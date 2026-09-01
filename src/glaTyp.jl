"""
    GilaTypes

This module provides the core abstract types and interfaces used across the Gila package.
It helps break circular dependencies between modules.
"""
module GilaTypes

using LinearAlgebra

export GlaSlv, AbstractGlaOpr, AbstractGlaVacOpr

"""
    GlaSlv

Abstract base type for all solvers in the Gila package. All concrete solver types
must subtype this type.
"""
abstract type GlaSlv end

"""
    AbstractGlaOpr

Abstract base type for all operators in the Gila package. All concrete operator types
must subtype this type and implement the AbstractMatrix interface.
"""
abstract type AbstractGlaOpr <: AbstractMatrix{ComplexF64} end

"""
    AbstractGlaVacOpr

Abstract base type for the vacuum Green operators, the ones built from geometry
alone with no material in them. Scattering operators are not vacuum operators;
they subtype `AbstractGlaOpr` directly.

Traits shared by every vacuum operator are defined once on this type, so a
subtype only overrides the ones where it differs.

# Interface

A subtype has to implement:
- `Base.size(opr)`: The number of rows and columns of the operator as a matrix
- `mulAct!(opr, act::AbstractVector{ComplexF64})`: The matrix-vector product on a
  flat vector (mutating)
- `adjoint!(opr)`: The adjoint, with the value semantics of its docstring
- `arrTyp(opr)`: `Array` or `CuArray`, whichever the operator computes with
- `isadjoint(opr)`: Whether the operator is the adjoint of the Green operator
- `isselfoperator(opr)`: Whether the source and target geometry are the same
- `isexternaloperator(opr)`: Whether the source and target geometry are disjoint
- `isgpu(opr)`: Whether the operator computes on the GPU
- `glaSze(opr)`: The target and source sizes in tensor form

`isoverlappingoperator` defaults to `false` here, `eltype` to `ComplexF64` on
`AbstractGlaOpr`, and `slv` to a `BiCGStabSolver`. Override any of the three when
it does not hold.
"""
abstract type AbstractGlaVacOpr <: AbstractGlaOpr end

end # module
