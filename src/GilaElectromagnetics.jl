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

include("glaTyp.jl")
using .GilaTypes
export GlaSlv, AbstractGlaOpr, AbstractGlaVacOpr

include("glaVol.jl")
using .GilaVolumes
export GlaVol
export GlaCmpVol, CompositeVolume, refine, regions, nregions, coordinates, cellvolumes, finest

include("glaFld.jl")
using .GilaFields
export GlaFld, MultiScaleField, zerofield, discretize!, regionview, eachregion, regrid

# The G₀ kernel: memory, generation, FFT action
include("vacuum/glaVac.jl")
using .GilaVacuum
export GlaKerOpt, GlaVacOprMem

include("glaSlv.jl")
using .GilaSolvers
export GMRESSolver, GCRODRSolver, RcySpc, BiCGStabSolver, MixPrcRfn, SlvLog, solve, isvarying
export MixedPrecisionRefinement

# The linear operators built on the kernel
include("operators/glaOpr.jl")
using .GilaOperators
export GlaOprVac, AsyGlaOprVac, SymGlaOprVac, GlaCmpOprVac, AsyGlaCmpOprVac, SymGlaCmpOprVac, InvSctOpr, SctOpr, GlaOpr, SusOpr
export VacuumGreenOperator, AsymVacuumGreenOperator, SymVacuumGreenOperator, CompositeVacuumGreenOperator, AsymCompositeVacuumGreenOperator, SymCompositeVacuumGreenOperator, InverseScatteringOperator, ScatteringOperator, GreenOperator, SusceptibilityOperator
export isadjoint, isselfoperator, isexternaloperator, isoverlappingoperator, isquasistatic, adjoint!, glaSze, slv, sus, asym, sym, setSus!, setBlc!

end
