using Base.Threads, CUDA, FFTW, Random, GilaMem
include("./src/egoAct2.jl")
###PROGRESS
#=
Standard Fourier transform has been opened up and tested on CPU.
=#
###VARIABLE DECLARATIONS
# number of cells in volume 
celInf = [16]
celDiv = 4
# computation settings, true / false switches between CPU and GPU operation
cmpInf = GlaKerOpt(false)
# memory type 
setTyp = ComplexF32
###MEMORY PREPARATION
#=
	memConTst(celInf::AbstractVector{<:Integer}, maxLvl::Integer, 
	setTyp::Union{ComplexF64,ComplexF32}, cmpInf::GlaKerOpt)::TrnVecMem

Memory structure for testing environment.
=#
function memConTst(celInf::AbstractVector{<:Integer}, maxLvl::Integer, 
	setTyp::Union{Type{ComplexF64},Type{ComplexF32}}, 
	cmpInf::GlaKerOpt)::TrnVecMem
	# number of even / odd branches
	numBrn = ^(2, maxLvl)
	phzInf = Array{Array{setTyp}}(undef, maxLvl)	
	## allocations
	if cmpInf.dev == false
		fftPlnFwd = Array{FFTW.cFFTWPlan}(undef, 1, maxLvl)
		fftPlnRev = Array{FFTW.ScaledPlan}(undef, 1, maxLvl)
		# Fourier transform planning area
		fftWrk = Array{setTyp}(undef, celInf...)
		# vector that will be Fourier transformed 
		actVec = Array{Array{setTyp,3}}(undef, numBrn)
		for bId ∈ 1:numBrn
			actVec[bId] = Array{setTyp}(undef, celInf...)
		end
	else
		celInfDev = CuArray{Int32}(undef, maxLvl)
		phzInfDev = Array{CuArray{setTyp}}(undef, maxLvl)
		fftPlnFwdDev = Array{CUDA.CUFFT.cCuFFTPlan}(undef, 1, maxLvl)
		fftPlnRevDev = Array{AbstractFFTs.ScaledPlan}(undef, 1, maxLvl)
		fftWrkDev = CuArray{setTyp}(undef, celInf...)
		# vector that will be Fourier transformed 
		actVecDev = Array{CuArray{setTyp,3}}(undef, numBrn)
		# act vector starts as all zeros
		for bId ∈ 1:numBrn
			actVecDev[bId] = CuArray{setTyp}(undef, celInf...)
			copyto!(actVecDev[bId], zeros(setTyp, celInf...))
		end
		# dimension information for convenient passing to kernel
		copyto!(celInfDev, Int32.(celInf))
	end
	## initialize Fourier transform plans
	if cmpInf.dev == false
		for dir ∈ 1:maxLvl	
			fftPlnFwd[dir] = plan_fft!(fftWrk, [dir]; flags = FFTW.MEASURE)
			fftPlnRev[dir] = plan_ifft!(fftWrk, [dir]; flags = FFTW.MEASURE)
		end
	else
		for dir ∈ 1:maxLvl
			@CUDA.sync fftPlnFwdDev[dir] =  plan_fft!(fftWrkDev, dir)
			@CUDA.sync fftPlnRevDev[dir] =  plan_ifft!(fftWrkDev, dir)
		end
	end
	## computation of phase transformation
	for itr ∈ 1:maxLvl
		# allows calculation odd coefficient numbers
		phzInf[itr] = [exp(-im * pi * k / celInf[itr]) for 
			k ∈ 0:(celInf[itr] - 1)]
		# active GPU
		if cmpInf.dev == true 		
			phzInfDev[itr] = CuArray{setTyp}(undef, celInf...)
			copyto!(selectdim(phzInfDev, 1, itr), 
				selectdim(phzInf, 1, itr))
		end
	end
	# wait for completion of GPU operation, create memory structure
	if cmpInf.dev == true 
		CUDA.synchronize(CUDA.stream())
		return TrnVecMem(cmpInf, celInf, celInfDev, actVecDev, fftPlnFwdDev, 
			fftPlnRevDev, phzInfDev)
	else
		return TrnVecMem(cmpInf, celInf, celInf, actVec, fftPlnFwd, fftPlnRev, 
			phzInf)
	end
end
# maxLvl from egoAct2
# trnMem = memConTst(celInf, maxLvl, setTyp, cmpInf);
# seed vector
inzVec = Array{ComplexF32}(undef, celInf...)
rand!(inzVec)
inzVecCpy = copy(inzVec)
hlfVecE = Array{ComplexF32}(undef, div.(celInf, 2)...)
# work vector, simulating decomposition
wrkVec = Array{ComplexF32}(undef, div.(celInf, celDiv)...)
# vector for saved work
wrkOut = Array{ComplexF32}(undef, div.(celInf, celDiv)..., celDiv...)

inzVec[1:8] .= 0.0 + im *0.0
hlfVecE[1:8] .= inzVec[9:end]
hlfVecO = copy(hlfVecE)
phz = [exp(-im * pi * k / 8) for k ∈ 0:(8 - 1)]
# # plant seed in memory structure
# copyto!(trnMem.actVec[1], inzVec)
# if cmpInf.dev == true 
# 	CUDA.synchronize(CUDA.stream())
# end
# sub-division of original vector
# embedded vector for comparison
fftPlnTot = zeros(ComplexF32, celInf...)
fftPlnHlf = zeros(ComplexF32, div.(celInf, 2)...)
fftPlnBig = zeros(ComplexF32, div.(celInf, celDiv)...)
fftPlnSml = zeros(ComplexF32, div.(celInf, celDiv)..., celDiv...)

fftFwdTot = plan_fft!(fftPlnTot; flags = FFTW.MEASURE)
fftFwdHlf = plan_fft!(fftPlnHlf; flags = FFTW.MEASURE)
fftFwdBig = plan_fft!(fftPlnBig; flags = FFTW.MEASURE)
fftFwdSml = plan_fft!(fftPlnSml, [2]; flags = FFTW.MEASURE)
fftFwdSml2 = plan_fft!(wrkVec; flags = FFTW.MEASURE)
###COMPUTATION
# standard form
# fftFwdTot * inzVec
# split form
# for div in 1:celDiv
# 	copy!(wrkVec, inzVecCpy[div:celDiv:end])
# 	# perform big Fourier transforms
# 	fftFwdBig * wrkVec
# 	# phase correction
# 	for itr in eachindex(wrkVec)
# 		wrkVec[itr] = exp(-2 * pi * im * (div - 1) * (itr - 1) / 16) * wrkVec[itr]
# 	end
# 	# copy out work
# 	copy!(selectdim(wrkOut, 2, div), wrkVec)
# end

# wItr = 1

# for itr in eachindex(inzVec)

# 	if mod(itr,4) == 2
# 		global wrkVec[wItr] = inzVec[itr]
# 		global wItr += 1
# 	else
# 		inzVec[itr] = 0.0 + im*0.0
# 	end
# end
hlfVecO .= phz .* hlfVecO

inzVec = fftFwdTot * inzVec
hlfVecE = fftFwdHlf * hlfVecE 
hlfVecO = fftFwdHlf * hlfVecO
# perform small transforms
# fftFwdSml * wrkOut
# fftFwdSml2 * wrkVec

a = 0.0
# egoExp!(trnMem, 0, 0)
