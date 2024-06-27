###UTILITY LOADING
using CUDA, AbstractFFTs, FFTW, Base.Threads, LinearAlgebra, LinearAlgebra.BLAS, 
Random, GilaElectromagnetics, Test, Serialization, Scratch
include("preamble.jl")
###SETTINGS
# type for tests
useType = ComplexF32
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
if CUDA.functional()
	cmpInfDev = GlaKerOpt(true)
end
###PREP 
# build Gila volumes
volB = GlaVol(celB, sclB, orgB)
volA = GlaVol(celA, sclA, orgA)
volU = GlaVol(celU, sclU, orgU)
###OPERATOR MEMORY
println("Green function construction started.")

function getFur(fname)
	preload_dir = @get_scratch!("preload")
	if isfile(joinpath(preload_dir, fname))
		return deserialize(joinpath(preload_dir, fname))
	end
	return nothing
end

function writeFur(fur, fname)
	preload_dir = @get_scratch!("preload")
	serialize(joinpath(preload_dir, fname), fur)
end

# generate from scratch---new circulant matrices
furSlfHst = getFur("slfHst.fur")
if isnothing(furSlfHst)
	oprSlfHst = GlaOprMem(cmpInfHst, volA, setType = useType)
	writeFur(oprSlfHst.egoFur, "slfHst.fur")
	furSlfHst = oprSlfHst.egoFur
else
	oprSlfHst = GlaOprMem(cmpInfHst, volA, egoFur = furSlfHst, setType = useType)
end
furExtHst = getFur("extHst.fur")
if isnothing(furExtHst)
	oprExtHst = GlaOprMem(cmpInfHst, volB, volA, setType = useType)
	furExtHst = oprExtHst.egoFur
	writeFur(furExtHst, "extHst.fur")
else
	oprExtHst = GlaOprMem(cmpInfHst, volB, volA, egoFur = furExtHst, setType = useType)
end
# merged domains to check validity of external operator construction
furMrgHst = getFur("mrgHst.fur")
if isnothing(furMrgHst)
	oprMrgHst = GlaOprMem(cmpInfHst, volU, setType = useType)
	writeFur(oprMrgHst.egoFur, "mrgHst.fur")
	furMrgHst = oprMrgHst.egoFur
else
	oprMrgHst = GlaOprMem(cmpInfHst, volU, egoFur = furMrgHst, setType = useType)
end
# run same test on device
if CUDA.functional()
	furExtDev = getFur("extDev.fur")
	if isnothing(furExtDev)
		oprExtDev = GlaOprMem(cmpInfDev, volB, volA, setType = useType)
		writeFur(oprExtDev.egoFur, "extDev.fur")
		furExtDev = oprExtDev.egoFur
	else
		oprExtDev = GlaOprMem(cmpInfDev, volB, volA, egoFur = furExtDev, 
			setType = useType)
	end
	furMrgDev = getFur("mrgDev.fur")
	if isnothing(furMrgDev)
		oprMrgDev = GlaOprMem(cmpInfDev, volU, setType = useType)
		writeFur(oprMrgDev.egoFur, "mrgDev.fur")
		furMrgDev = oprMrgDev.egoFur
	else
		oprMrgDev = GlaOprMem(cmpInfDev, volU, egoFur = furMrgDev, 
			setType = useType)
	end
end
# serialize / deserialize to reuse Fourier information
println("Green function construction completed.")
###TESTS
## integral convergence 
println("Integral convergence test started.")
include("intConTest.jl")
println("Integral convergence test completed.")
## analytic agreement test on self operator
println("Analytic test started.")
include("anaTest.jl")
println("Analytic test completed.")
## positive semi-definite test on self operator
# test becomes very slow for domains larger than [16,16,16]
println("Semi-definiteness test started.")
include("posDefTest.jl")
println("Semi-definiteness test completed.")
## test external Green function using self Green function
println("External operator test started.")
include("extSlfTest.jl")
println("External operator test completed.")
println("Testing complete.")
