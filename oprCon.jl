###UTILITY LOADING
include("./test/preamble.jl")
###SETTINGS
# number of cells in each volume 
celB = [4,4,4]
celA = [16,16,16]
# size of cells relative to wavelength
sclB = (1//25, 1//25, 1//25)
sclA = (1//50, 1//50, 1//50)
# center position of volume
orgB = (3//1, 3//1, 3//1)
orgA = (0//1, 0//1, 0//1)
# gla volumes
trgVol  = GlaVol(celB, sclB, orgB)
srcVol  = GlaVol(celA, sclA, orgA)
## computation settings (pick one)
# host execution
cmpInf = GlaKerOpt(false)
# device execution
# cmpInf = GlaKerOpt(true)
###OPERATOR MEMORY
# generate from scratch
oprInf = GlaOprMem(cmpInf, trgVol, srcVol) 
# save for future use
serialize("./tmp/gFourExt2_4x3_64", oprInf.egoFur)
# load previously computed fourier information
# gFurX = deserialize("./tmp/gFourExt16x3_64")
# regenerate operator memory
# oprInf = GlaOprMem(cmpInf, trgVol, srcVol, gFurX)

actVec = ones(ComplexF64, oprInf.srcVol.cel...)
include("./src/glaWrk.jl")

# number of elements in a branch
brnSze = div.(oprInf.mixInf.trgCel .+ oprInf.mixInf.srcCel, 2)
# number of source partitions
parNum = prod(oprInf.mixInf.srcDiv)
# allocate branching vectors
if sum(oprInf.cmpInf.devMod) == true
	# even branch
	parVecEve = CuArray{eltype(actVec)}(undef, brnSze..., parNum, 3) 
	# odd branch
	parVecOdd = CuArray{eltype(actVec)}(undef, brnSze..., parNum, 3) 
else
	# even branch
	parVecEve = Array{eltype(actVec)}(undef, brnSze..., parNum, 3) 
	# odd branch
	parVecOdd = Array{eltype(actVec)}(undef, brnSze..., parNum, 3) 
end 

genPrt(oprInf, parVecEve, parVecOdd, actVec)
println("done!")
##TO DO!!
# slf Green function where grid scale is larger than cell scale -> remove some singular computations. 