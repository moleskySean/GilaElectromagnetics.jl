###UTILITY LOADING
include("./test/preamble.jl")
###SETTINGS
# number of cells in each volume 
# to run external operator test, celU should be the union of celA and celB
celB = (16, 16, 16)
celA = (16, 16, 16)
celU = (16, 32, 16)
# size of cells relative to wavelength
# to run external operator test, the scales of the three volumes should match
sclB = (1//50, 1//50, 1//50)
sclA = (1//50, 1//50, 1//50)
sclU = (1//50, 1//50, 1//50)
# center position of volumes
# to run external operator test, volA should touch (but not overlap!) volB
orgB = (0//1, 16//50, 0//1)
orgA = (0//1, 0//1, 0//1)
orgU = (0//1, 0//1, 0//1)
## compute settings
# use for host execution
cmpInfHst = GlaKerOpt(false)
# use for device execution
cmpInfDev = GlaKerOpt(true)
###PREP 
# build Gila volumes
volB = GlaVol(celB, sclB, orgB)
volA = GlaVol(celA, sclA, orgA)
volU = GlaVol(celU, sclU, orgU)
###OPERATOR MEMORY
println("Green function construction started.")
# generate from scratch---new circulant matrices
oprSlfHst = GlaOprMem(cmpInfHst, volA, setType = ComplexF32)
# oprExtHst = GlaOprMem(cmpInfHst, volB, volA, setType = ComplexF32) 
# same convention can be used to save and reuse previously computed Fourier info
# oprExtDev = GlaOprMem(cmpInfDev, volB, volA, egoFur = oprExtHst.egoFur, 
	# setType = ComplexF32) 
# oprMrgHst = GlaOprMem(cmpInfHst, volU, setType = ComplexF32) 
# oprMrgDev = GlaOprMem(cmpInfDev, volU, egoFur = oprMrgHst.egoFur, 
	# setType = ComplexF32) 
# serialize / deserialize to reuse Fourier information
println("Green function construction completed.")
###TESTS
## integral convergence 
# println("Integral convergence test started.")
# include("./test/intConTest.jl")
# println("Integral convergence test completed.")
## analytic agreement test on self operator
# println("Analytic test started.")
# include("./test/anaTest.jl")
# println("Analytic test completed.")
## positive semi-definite test on self operator
# test becomes very slow for domains larger than [16,16,16]
# println("Semi-definiteness test started.")
# include("./test/posDefTest.jl")
# println("Semi-definiteness test completed.")
## test external Green function using self Green function
# println("External operator test started.")
# include("./test/extSlfTest.jl")
# println("External operator test completed.")
# println("Testing complete.")
##TODOS
# complete in multiple GPU support