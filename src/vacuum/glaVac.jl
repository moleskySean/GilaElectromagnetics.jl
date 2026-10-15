module GilaVacuum
"""
    GilaVacuum

This module provides memory structures and operations for the vacuum Green function operator in Gila.
"""

# The Bloch-periodic Green tensor of a periodic unit cell
include("bloch/bloch.jl")
using .Bloch

# Holds computational information (CPU vs GPU, etc.) for the Green function
include("glaVacCmp.jl")
export GlaKerOpt, dflPrc

# Defines GlaVacOprMem, the struct holding the info to compute the Green function
include("glaVacOprMem.jl")
export GlaVacOprMem

include("glaVacAct.jl")
export egoOpr!

end
