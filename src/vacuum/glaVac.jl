module GilaVacuum
"""
    GilaVacuum

This module provides memory structures and operations for the vacuum Green function operator in Gila.
"""

# Holds computational information (CPU vs GPU, number of threads, etc.) for the Green function
include("glaVacCmp.jl")
export GlaKerOpt, CPUKerOpt, GPUKerOpt, frqPhz, intOrd, adjMod, bckEnd, arrTyp

# Defines GlaVacOprMem, the struct holding the info to compute the Green function
include("glaVacOprMem.jl")
export GlaVacOprMem, useCpu!, useGpu! 

include("glaVacAct.jl")
export egoOpr!

end
