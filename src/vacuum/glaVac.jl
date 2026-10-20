# The Bloch-periodic Green tensor of a periodic unit cell
include("bloch/bloch.jl")
using .Bloch

# Holds computational information (CPU vs GPU, etc.) for the Green function
include("glaVacCmp.jl")

# Defines GlaVacOprMem, the struct holding the info to compute the Green function
include("glaVacOprMem.jl")

include("glaVacAct.jl")
