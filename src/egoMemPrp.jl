# GlaOprMemGenExt
"""

	function GlaOprMemGenExt(trgVol::GlaVol, srcVol::GlaVol)::GlaOprMem

Prepare memory for Green function operator between a pair of distinct domains. 
"""
# function GlaOprMemGenExt(cmpInf::GlaKerOpt, trgVol::GlaVol, 
# 	srcVol::GlaVol)::GlaOprMem
# 	# memory for circulant Green function vector
# 	egoCrc = Array{ComplexF32}(undef, 3, 3, srcVol.cells[1] + trgVol.cells[1],
# 		srcVol.cells[2] + trgVol.cells[2], srcVol.cells[3] + trgVol.cells[3])
# 	# generate circulant Green function
# 	genEgoExt!(greenCrc, trgVol, srcVol, cmpInf)
# 	return GlaOprPrpExt(egoCrc, trgVol, srcVol)
# end
"""

	function GlaOprMemGenSlf(cmpInf::GlaKerOpt, slfVol::GlaVol)::GlaOprMem

Prepare memory for Green function operator for a self domain. 
"""
function GlaOprMemGenSlf(cmpInf::GlaKerOpt, slfVol::GlaVol, 
	egoFur::Union{AbstractArray{<:AbstractArray{T}},
	Nothing}=nothing)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}
	
	if isnothing(egoFur)
		# memory for circulant green function vector
		egoCrc = Array{ComplexF64}(undef, 3, 3, 2 * slfVol.cells[1], 2 * 
			slfVol.cells[2], 2 * slfVol.cells[3])
		# generate circulant Green function, from GilaCrc module
		genEgoSlf!(egoCrc, slfVol, cmpInf)
		# verify that egoCrc contains numeric values
		if maximum(isnan.(egoCrc)) == 1 || maximum(isinf.(egoCrc)) == 1
			error("Computed circulant contains non-numeric values.")
		end
		# branching depth of multiplication
		lvl = 3
		# Fourier transform of circulant Green function
		egoFurPrp = Array{eltype(egoCrc)}(undef, (2 .* slfVol.cells)..., 
			2 * lvl)
		# plan Fourier transform
		fftCrcOut = plan_fft(egoCrc[1,1,:,:,:], (1, 2, 3))
		# Fourier transform of the Green function, making use of real space 
		# symmetry under transposition--entries are xx, yy, zz, xy, xz, yz
		for colItr ∈ 1:3, rowItr ∈ 1:colItr
			# blkEgoItr follows xx, yy, ...
			# vector direction moved to outermost---largest stride---index
			egoFurPrp[:,:,:,blkEgoItr(3 * (colItr - 1) + rowItr)] =  
				fftCrcOut * egoCrc[rowItr,colItr,:,:,:]
		end
		# number of multiplication branches		
		eoDim = ^(2, lvl)
		# number of unique Green function elements
		ddDim = 2 * lvl
		egoFur = Array{Array{eltype(egoCrc)}}(undef, eoDim)
		for eoItr ∈ 0:(eoDim - 1)
			# odd / even branch extraction
			egoFur[eoItr + 1] = Array{eltype(egoCrc)}(undef, slfVol.cells..., 
				ddDim)
			# first division is along smallest stride -> largest binary division
			egoFur[eoItr + 1][:,:,:,:] .= egoFurPrp[(1 + 
				mod(div(eoItr, 4), 2)):2:(end - 1 + mod(div(eoItr, 4), 2)), 
				(1 + mod(div(eoItr, 2), 2)):2:(end - 1 + mod(div(eoItr, 2), 2)),
				(1 + mod(eoItr, 2)):2:(end - 1 + mod(eoItr, 2)), :]
			# verify that all values are numeric.
			if maximum(isnan.(egoFur[eoItr + 1])) == 1 || 
				maximum(isinf.(egoFur[eoItr + 1])) == 1
				error("Fourier information contains non-numeric values.")
			end
		end
	end
	setTyp = eltype(eltype(egoFur))
	# verify that egoCrc contains numeric values
	for eoItr ∈ 1:8
		if maximum(isnan.(egoFur[eoItr])) == 1 || 
				maximum(isinf.(egoFur[eoItr])) == 1
			error("Provided Fourier information contains non-numeric values.")
		end
	end
	return GlaOprPrpSlf(egoFur, slfVol, cmpInf, setTyp)
end
#=
Memory preparation function.
=#
function GlaOprPrpSlf(egoFur::AbstractArray{<:AbstractArray{T}}, slfVol::GlaVol, 
	cmpInf::GlaKerOpt, setTyp::DataType)::GlaOprMem where 
	T<:Union{ComplexF64,ComplexF32}
	###MEMORY DECLARATION
	# number of embedding levels---dimensionality of ambient space.
	lvls = 3
	# operator dimensions
	celInf = slfVol.cells
	# active GPU
	if cmpInf.dev == true 
		celInfDev = CuArray{Int}(undef, lvls)
		copyto!(celInfDev, celInf)
	end
	# binary indexing system of even and odd coefficient extraction
	eoDim = ^(2, lvls)
	# phase transformations (internal for block Toeplitz transformations)
	phzInf = Array{Array{setTyp}}(undef, lvls)
	# Fourier transform plans
	if cmpInf.dev == false
		fftPlnFwd = Array{FFTW.cFFTWPlan}(undef, 1, lvls)
		fftPlnRev = Array{FFTW.ScaledPlan}(undef, 1, lvls)
		# Fourier transform planning area
		fftWrk = Array{setTyp}(undef, celInf..., lvls)
		# vector that will be transformed by action of the Green function
		# act vector starts as all zeros
		actVec = zeros(eltype(setTyp), celInf..., lvls)
	# active GPU 		
	else
		egoFurDev = Array{CuArray{setTyp}}(undef, eoDim)
		phzInfDev = Array{CuArray{setTyp}}(undef, lvls)
		fftPlnFwdDev = Array{CUDA.CUFFT.cCuFFTPlan}(undef, 1, lvls)
		fftPlnRevDev = Array{AbstractFFTs.ScaledPlan}(undef, 1, lvls)
		fftWrkDev = CuArray{setTyp}(undef, celInf..., lvls)
		actVecDev = CuArray{setTyp}(undef, celInf..., lvls)
		# act vector starts as all zeros
		copyto!(actVecDev,zeros(eltype(setTyp), celInf..., lvls))
	end
	###MEMORY PREPARATION
	# initialize Fourier transform plans
	if cmpInf.dev == false
		for dir ∈ 1:lvls	
			fftPlnFwd[dir] = plan_fft!(fftWrk, [dir]; flags = FFTW.MEASURE)
			fftPlnRev[dir] = plan_ifft!(fftWrk, [dir]; flags = FFTW.MEASURE)
		end
	else
		for dir ∈ 1:lvls
			@CUDA.sync fftPlnFwdDev[dir] =  plan_fft!(fftWrkDev, dir)
			@CUDA.sync fftPlnRevDev[dir] =  plan_ifft!(fftWrkDev, dir)
		end
	end
	# computation of phase transformation
	for itr ∈ 1:lvls
		# allows calculation odd coefficient numbers
		phzInf[itr] = [exp(-im * pi * k / celInf[itr]) for 
			k ∈ 0:(celInf[itr] - 1)]
		# active GPU
		if cmpInf.dev == true 		
			phzInfDev[itr] = CuArray{setTyp}(undef, celInf...)
			copyto!(selectdim(phzInfDev, 1, itr), selectdim(phzInf, 1, itr))
		end
	end
	# number of unique Green function elements
	ddDim = 2 * lvls
	# transfer Fourier coefficients to GPU if active
	if cmpInf.dev == true 
		# active GPU
		for eoItr ∈ 0:(eoDim - 1), ddItr ∈ 1:6
			egoFurDev[eoItr + 1] = CuArray{setTyp}(undef, celInf..., ddDim)
			copyto!(selectdim(egoFurDev, 1, eoItr + 1), 
				selectdim(egoFur, 1, eoItr + 1))
		end
	end
	# wait for completion of GPU operation, create memory structure
	if cmpInf.dev == true 
		CUDA.synchronize(CUDA.stream())
		GlaOprMem(cmpInf, slfVol, slfVol, celInf, celInfDev, actVecDev, 
			egoFurDev, fftPlnFwdDev, fftPlnRevDev, phzInfDev)
	else
		return GlaOprMem(cmpInf, slfVol, slfVol, celInf, celInf, actVec, 
			egoFur, fftPlnFwd, fftPlnRev, phzInf)
	end
end
#=
Block index for a given Cartesian index.
=#
@inline function blkEgoItr(crtInd::Integer)::Integer

	if crtInd == 1
		return 1
	elseif crtInd == 2 || crtInd == 4
		return 4
	elseif crtInd == 5 
		return 2	
	elseif crtInd == 7 || crtInd == 3
		return 5
	elseif crtInd == 8 || crtInd == 6
		return 6
	elseif crtInd == 9 
		return 3
	else
		error("Improper use case.")
		return 0
	end
end