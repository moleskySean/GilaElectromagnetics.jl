"""

	function GlaOprMemGen(cmpInf::GlaKerOpt, trgVol::GlaVol,
	srcVol::Union{GlaVol,Nothing}=nothing, 
	egoFur::Union{AbstractArray{<:AbstractArray{T}},
	Nothing}=nothing)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}

Prepare memory for Green function operator---when called with a single GlaVol, 
or identical source and target volumes, the function yields the self Green 
function construction. 
"""
function GlaOprMemGen(cmpInf::GlaKerOpt, trgVol::GlaVol,
	srcVol::Union{GlaVol,Nothing}=nothing, 
	egoFur::Union{AbstractArray{<:AbstractArray{T}, 3},
	Nothing}=nothing)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}
	
	# flag for self Green function case
	slfFlg = 0
	# self Green function case
	if isnothing(srcVol) || trgVol == srcVol
		slfFlg = 1
		srcVol = trgVol
		mixInf = extInfGen(trgVol, srcVol)
	# external Green function case
	else
		mixInf = extInfGen(trgVol, srcVol)
	end
	# generate circulant Green function
	if isnothing(egoFur)
		# generate circulant Green function, from GilaCrc module
		if slfFlg == 1
			# number of pair interactions based on division of volumes
			numEgoInt = 1
			# memory for circulant green function vector
			egoCrc = Array{ComplexF64}(undef, 3, 3, 
				trgVol.cel[1] + slfVol.cel[1], trgVol.cel[2] + slfVol.cel[2], 
				trgVol.cel[3] + slfVol.cel[3], 1, 1)
			genEgoSlf!(selectdim(selectdim(egoCrc, lastindex(egoCrc, 1), 
				lastindex(egoCrc) - 1, 1), slfVol, cmpInf), trgVol, cmpInf)
		else
			# number of pair interactions based on division of volumes
			numEgoInt = prod(mixInf.trgDiv) * prod(mixInf.srcDiv)
			# memory for circulant green function vector
			egoCrc = Array{ComplexF64}(undef, 3, 3, 
				mixInf.trgCel[1] + slfVol.cel[1], trgVol.cel[2] + slfVol.cel[2], 
				trgVol.cel[3] + slfVol.cel[3], mixInf.trgDiv, mixInf.srcDiv)

			

		end
		# verify that egoCrc contains numeric values
		if maximum(isnan.(egoCrc)) == 1 || maximum(isinf.(egoCrc)) == 1
			error("Computed circulant contains non-numeric values.")
		end
		# branching depth of multiplication
		lvl = 3
		# Fourier transform of circulant Green function
		egoFurPrp = Array{eltype(egoCrc)}(undef, (2 .* slfVol.cel)..., 
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
		# number of unique Green function blocks
		ddDim = 2 * lvl
		# unique Green function elements
		hlfRup = Integer.(ceil.(slfVol.cel ./ 2)) .+ 1
		cpyInd = CartesianIndices((1:hlfRup[1], 1:hlfRup[2], 1:hlfRup[3]))
		egoFur = Array{Array{eltype(egoCrc)}}(undef, eoDim)
		egoFurInt = Array{eltype(egoCrc)}(undef, slfVol.cel..., ddDim)
		for eoItr ∈ 0:(eoDim - 1)
			# odd / even branch extraction
			egoFur[eoItr + 1] = Array{eltype(egoCrc)}(undef, hlfRup..., ddDim)
			# first division is along smallest stride -> largest binary division
			egoFurInt[:,:,:,:] .= egoFurPrp[(1 + 
				mod(div(eoItr, 4), 2)):2:(end - 1 + mod(div(eoItr, 4), 2)), 
				(1 + mod(div(eoItr, 2), 2)):2:(end - 1 + mod(div(eoItr, 2), 2)),
				(1 + mod(eoItr, 2)):2:(end - 1 + mod(eoItr, 2)), :]
			# only one one eighth of the Green function is unique 
			egoFur[eoItr + 1][:,:,:,:] .= egoFurInt[1:hlfRup[1],1:hlfRup[2],
				1:hlfRup[3],:]
			# verify that all values are numeric.
			if maximum(isnan.(egoFur[eoItr + 1])) == 1 || 
				maximum(isinf.(egoFur[eoItr + 1])) == 1
				error("Fourier information contains non-numeric values.")
			end
		end
	end
	# verify that egoCrc contains numeric values
	for eoItr ∈ 1:8
		if maximum(isnan.(egoFur[eoItr])) == 1 || 
				maximum(isinf.(egoFur[eoItr])) == 1
			error("Provided Fourier information contains non-numeric values.")
		end
	end
	if cmpInf.dev == true
		return GlaOprPrpSlf(egoFur, slfVol, cmpInf, ComplexF32)
	else
		setTyp = eltype(eltype(egoFur))
		return GlaOprPrpSlf(egoFur, slfVol, cmpInf, setTyp)
	end
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
	celInf = slfVol.cel
	# active GPU
	if cmpInf.dev == true 
		celInfDev = CuArray{Int32}(undef, lvls)
		copyto!(celInfDev, Int32.(celInf))
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
		actVec = zeros(setTyp, celInf..., lvls)
	# active GPU 		
	else
		egoFurDev = Array{CuArray{setTyp}}(undef, eoDim)
		phzInfDev = Array{CuArray{setTyp}}(undef, lvls)
		fftPlnFwdDev = Array{CUDA.CUFFT.cCuFFTPlan}(undef, 1, lvls)
		fftPlnRevDev = Array{AbstractFFTs.ScaledPlan}(undef, 1, lvls)
		fftWrkDev = CuArray{setTyp}(undef, celInf..., lvls)
		actVecDev = CuArray{setTyp}(undef, celInf..., lvls)
		# act vector starts as all zeros
		copyto!(actVecDev, zeros(setTyp, celInf..., lvls))
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
			copyto!(selectdim(phzInfDev, 1, itr), 
				selectdim(phzInf, 1, itr))
		end
	end
	# number of unique Green function blocks
	ddDim = 2 * lvls
	# maximum number of unique elements in a Green function block
	hlfRup = Integer.(ceil.(celInf ./ 2)) .+ 1
	# transfer Fourier coefficients to GPU if active
	if cmpInf.dev == true 
		# active GPU
		for eoItr ∈ 0:(eoDim - 1), ddItr ∈ 1:6
			egoFurDev[eoItr + 1] = CuArray{setTyp}(undef, hlfRup..., ddDim)
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
External pair information for treating grid mismatch. 
=#
function extInfGen(trgVol::GlaVol, srcVol::GlaVol)::GlaExtInf
	
	# test that cell scales are compatible 
	if prod(isinteger.(srcVol.scl ./ trgVol.scl) .+ isinteger.(trgVol.scl ./ 
		srcVol.scl)) != 1 
		error("Volume pair must share a common scale grid in order to construct 
		Green function.")
	end
	# common scale 
	minScl = gcd.(srcVol.scl, trgVol.scl)
	# maximal scale
	maxScl = lcm.(srcVol.scl, trgVol.scl)
	# grid divisions for the source and target volumes
	trgDivGrd = [maxScl[itr] ./ trgVol.scl[itr] for itr ∈ 1:3]
	srcDivGrd = [maxScl[itr] ./ srcVol.scl[itr] for itr ∈ 1:3]
	# number of cells in each source (target) division 
	trgDivCel = trgVol.cel ./ grdDivTrg
	srcDivCel = srcVol.cel ./ grdDivSrc
	# create transfer information
	return GlaExtInf(minScl, trgDivGrd, srcDivGrd, trgDivCel, srcDivCel)
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