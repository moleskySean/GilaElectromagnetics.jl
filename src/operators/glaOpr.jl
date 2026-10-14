"""
    GilaOperators

The operators: vacuum `GlaOprVac` (G₀), susceptibility `SusOpr` (X), `InvSctOpr`
(I - XG₀), `SctOpr` ((I - XG₀)⁻¹, through a solver) and `GlaOpr` (G₀(I - XG₀)⁻¹).
"""
module GilaOperators

using ..GilaVolumes
using ..GilaFields
using ..GilaVacuum
using ..GilaTypes
using ..GilaSolvers
using Adapt
using KernelAbstractions
using LinearAlgebra
using Serialization
using StaticArrays: StaticArray, SMatrix

import LinearAlgebra: adjoint!

import ..GilaTypes: isadjoint, _shwRow
import ..GilaVacuum: egoCmpPos, setBlc!
import ..GilaVolumes: _lwrEdg, _uprEdg, _ovrLap, _dofOff, _volDesc, _sclStr, _ratStr
import ..GilaSolvers: isvarying

export GlaOprVac, AsyGlaOprVac, SymGlaOprVac, GlaCmpOprVac, AsyGlaCmpOprVac, SymGlaCmpOprVac, InvSctOpr, SctOpr, GlaOpr, SusOpr
export VacuumGreenOperator, AsymVacuumGreenOperator, SymVacuumGreenOperator, CompositeVacuumGreenOperator, AsymCompositeVacuumGreenOperator, SymCompositeVacuumGreenOperator, InverseScatteringOperator, ScatteringOperator, GreenOperator, SusceptibilityOperator
export isadjoint, isselfoperator, isexternaloperator, isoverlappingoperator, isquasistatic, adjoint!, glaSze, slv, sus, asym, sym, setSus!, setBlc!

include("glaVacOpr.jl")
include("glaSusOpr.jl")
include("glaSctOpr.jl")
include("glaLinAlg.jl")
include("glaShw.jl")

end # module
