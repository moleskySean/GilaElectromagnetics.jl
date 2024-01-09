###UTILITY LOADING
include("./test/preamble.jl")
using GilaMem
###SETTINGS
# number of cells in each volume 
celB = [32,16,32]
celA = [16,32,16]
# size of cells relative to wavelength
sclB = (1//25, 1//100, 1//25)
sclA = (1//100, 1//25, 1//100)
# center position of volume
crdB = (3.00, 3.00, 3.00)
crdA = (0.00, 0.00, 0.00)
# gila volumes
trgVol  = GlaVol(celB, sclB, crdB)
srcVol  = GlaVol(celA, sclA, crdA)
# functions for testing functionality
include("extTrialFun.jl")
# Contents:
# extInfGen(trgVol, srcVol)---generate pair transfer information

extInf = extInfGen(trgVol, srcVol)