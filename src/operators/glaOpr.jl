#= The operators: vacuum `GlaOprVac` (G₀), susceptibility `SusOpr` (X), `InvSctOpr`
(I - XG₀), `SctOpr` ((I - XG₀)⁻¹, through a solver) and `GlaOpr` (G₀(I - XG₀)⁻¹). =#
using Adapt
using KernelAbstractions
using LinearAlgebra
using Serialization
using StaticArrays: StaticArray, SMatrix

import LinearAlgebra: adjoint!

include("glaVacOpr.jl")
include("glaSusOpr.jl")
include("glaSctOpr.jl")
include("glaLinAlg.jl")
include("glaShw.jl")
