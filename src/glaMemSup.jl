"""

	function GlaOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol,
	srcVol::Union{GlaVol,Nothing}=nothing, 
	egoFur::Union{AbstractArray{<:AbstractArray{T}},
	Nothing}=nothing)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}

Prepare memory for green function operator---when called with a single GlaVol, 
or identical source and target volumes, the function yields the self green 
function construction. 
"""
function GlaOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol,
	srcVol::Union{GlaVol,Nothing}=nothing, 
	egoFur::Union{AbstractArray{<:AbstractArray{T}, 1},
	Nothing}=nothing)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}
	# flag for self green function case
	slfFlg = 0
	# ensure that number of cells is even, regenerate if not
	trgVol = glaVolEveGen(trgVol)
	# self green function case
	if isnothing(srcVol) || trgVol == srcVol
		slfFlg = 1
		srcVol = trgVol
		mixInf = GlaExtInf(trgVol, srcVol)
	# external green function case
	else
		# ensure that number of cells is even, regenerate if not
		srcVol = glaVolEveGen(srcVol)
		# useful information for aligning source and target volumes
		mixInf = GlaExtInf(trgVol, srcVol)
	end
	# total cells in circulant
	totCelCrc = mixInf.trgCel .+ mixInf.srcCel
	# total number of target and source partitions
	totParTrg = prod(mixInf.trgDiv)
	totParSrc = prod(mixInf.srcDiv)
	# generate circulant green function, from GilaCrc module
	if isnothing(egoFur)
		# memory for circulant green function vector
		egoCrc = Array{ComplexF64}(undef, 3, 3, totCelCrc..., totParTrg, 
			totParSrc)
		# self green function case
		if slfFlg == 1
			genEgoSlf!(selectdim(selectdim(egoCrc, 7, 1), 6, 1), trgVol, cmpInf)
		# external green function case
		else
			# partition source and target for consistent distance offsets
			for srcItr ∈ 1:totParSrc, 
				# relative grid position
				srcGrdPos = Tuple(mixInf.srcPar[srcItr]) .- (1, 1, 1)
				# offset of center of partition from center of volume
				srcOrgOff = Rational.((srcGrdPos .- (mixInf.srcDiv .- 1) 
					.// 2) .* srcVol.scl .+ srcVol.org)
				# grid scale of partition
				srcGrdScl = mixInf.srcDiv .* srcVol.scl
				# create target volume partition
				srcVolPar = GlaVol(mixInf.srcCel, srcVol.scl, srcOrgOff, 
					srcGrdScl)
				for trgItr ∈ 1:totParTrg
					# relative grid position
					trgGrdPos = Tuple(mixInf.trgPar[trgItr]) .- (1, 1, 1)
					# offset of center of partition from center of volume
					trgOrgOff = Rational.((trgGrdPos .- (mixInf.trgDiv .- 1) 
						.// 2) .* trgVol.scl .+ trgVol.org)
					# grid scale of partition
					trgGrdScl = mixInf.trgDiv .* trgVol.scl
					# create target volume partition
					trgVolPar = GlaVol(mixInf.trgCel, trgVol.scl, trgOrgOff, 
						trgGrdScl)
					# generate green function information for partition pair
					genEgoExt!(selectdim(selectdim(egoCrc, 7, srcItr), 6, 
						trgItr), trgVolPar, srcVolPar, cmpInf)
				end
			end	
		end
		# verify that egoCrc contains numeric values
		if maximum(isnan.(egoCrc)) == 1 || maximum(isinf.(egoCrc)) == 1
			error("Computed circulant contains non-numeric values.")
		end
		# branching depth of multiplication
		lvl = 3
		# Fourier transform of circulant green function
		egoFurPrp = Array{eltype(egoCrc)}(undef, totCelCrc..., 6, totParTrg, 
			totParSrc)
		# plan Fourier transform
		fftCrcOut = plan_fft(egoCrc[1,1,:,:,:,1,1], (1, 2, 3))
		# Fourier transform of the green function, making use of real space 
		# symmetry under transposition--entries are xx, yy, zz, xy, xz, yz
		for srcItr ∈ 1:totParSrc , trgItr ∈ 1:totParTrg, 
			colItr ∈ 1:3, rowItr ∈ 1:colItr
			# vector direction moved to outer volume index---largest stride
			egoFurPrp[:,:,:,blkEgoItr(3 * (colItr - 1) + rowItr),trgItr,
				srcItr] =  fftCrcOut * egoCrc[rowItr,colItr,:,:,:,trgItr,srcItr]
		end
		# verify integrity of Fourier transform data
		if maximum(isnan.(egoFurPrp)) == 1 || maximum(isinf.(egoFurPrp)) == 1
			error("Fourier transform of circulant contains non-numeric values.")
		end
		# number of multiplication branches		
		eoDim = ^(2, lvl)
		# number of unique green function blocks
		ddDim = 6
		# total number of cells in a branch of the multiplication operation
		# glaVolEveGen enforces that number of cells is even 
		mixInf = GlaExtInf(trgVol, srcVol)
		# determine whether source or target volume contains more cells
		srcDomDir = map(<, mixInf.trgCel, mixInf.srcCel)
		trgDomDir = map(!, srcDomDir)
		# number of unique elements in each cartesian index for a branch
		truInf = div.(max.(mixInf.trgCel, mixInf.srcCel), 2) .+ 1 
		# start, step, and end positions for information copy
		indBeg = (2 .* trgDomDir) .+ (div.(totCelCrc, 2) .* srcDomDir)
		indStp = trgDomDir .- srcDomDir
		indEnd = ((div.(totCelCrc, 2) .- truInf .+ 2) .* srcDomDir) .+ 
			(truInf .* trgDomDir)
		# copy ranges for intermediate Fourier values
		cpyColInt = collect(Iterators.product(Tuple(map(tuple, ones(Int,3), 
		 	map(StepRange, indBeg, indStp, indEnd)))...))
		# copy ranges for final Fourier values
		cpyColFur = collect(Iterators.product(Tuple(map(tuple, ones(Int,3), 
                                map(:, 2 .* ones(Int,3), truInf)))...))
		# final Fourier coefficients for a given branch
		egoFur = Array{Array{eltype(egoCrc)}}(undef, eoDim)
		# intermediate storage
		egoFurInt = Array{eltype(egoCrc)}(undef, div.(totCelCrc, 2)..., ddDim, 
			totParTrg, totParSrc)
		# only one one eighth of the green function is unique 
		for eoItr ∈ 0:(eoDim - 1)
			# odd / even branch extraction
			egoFur[eoItr + 1] = Array{eltype(egoCrc)}(undef, truInf..., ddDim, 
				totParTrg, totParSrc)
			# first division is along smallest stride -> largest binary division
			egoFurInt[:,:,:,:,:,:] .= egoFurPrp[(1 + 
				mod(div(eoItr, 4), 2)):2:(end - 1 + mod(div(eoItr, 4), 2)), 
				(1 + mod(div(eoItr, 2), 2)):2:(end - 1 + mod(div(eoItr, 2), 2)),
				(1 + mod(eoItr, 2)):2:(end - 1 + mod(eoItr, 2)),:,:,:]
			# number of copy combinations to consider
			cpyCom = 8
			# extract unique information---!flipped storage convention when 
			# number of source cells is larger than the number of target cells
			# edge information much be copied without reversing direction
			for cpyItr ∈ 1:cpyCom
				egoFur[eoItr + 1][CartesianIndices(cpyColFur[cpyItr]),:,:,:] .= 
				egoFurInt[CartesianIndices(cpyColInt[cpyItr]),:,:,:]
			end
		end
	end
	# verify that egoCrc contains numeric values
	for eoItr ∈ 1:8
		if maximum(isnan.(egoFur[eoItr])) == 1 || 
				maximum(isinf.(egoFur[eoItr])) == 1
			error("Fourier information contains non-numeric values.")
		end
	end
	if cmpInf.devMod == true
		return GlaOprPrp(egoFur, trgVol, srcVol, mixInf, cmpInf, ComplexF32)
	else
		setTyp = eltype(eltype(egoFur))
		return GlaOprPrp(egoFur, trgVol, srcVol, mixInf, cmpInf, setTyp)
	end
end
#=
Memory preparation sub-protocol.
=#
function GlaOprPrp(egoFur::AbstractArray{<:AbstractArray{T}}, trgVol::GlaVol,
	srcVol::GlaVol, mixInf::GlaExtInf, cmpInf::GlaKerOpt, 
	setTyp::DataType)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}
	###MEMORY DECLARATION
	# number of embedding levels---dimensionality of ambient space
	lvls = 3
	# operator dimensions---unique vector information does not typically 
	# match operator size for distinct source and target volumes
	# sum of source and target volumes being divisible by 2 is guaranteed by 
	# glaVolEveGen in GlaOprMemGen
	celInf = div.(mixInf.trgCel .+ mixInf.srcCel, 2)
	# active GPU
	if cmpInf.devMod == true 
		celInfDev = CuArray{Int32}(undef, lvls)
		copyto!(celInfDev, Int32.(celInf))
	end
	# binary indexing system of even and odd coefficient extraction
	eoDim = ^(2, lvls)
	# phase transformations (internal for block Toeplitz transformations)
	phzInf = Array{Array{setTyp}}(undef, lvls)
	# Fourier transform plans
	if cmpInf.devMod == false
		fftPlnFwd = Array{FFTW.cFFTWPlan}(undef, lvls)
		fftPlnRev = Array{FFTW.ScaledPlan}(undef, lvls)
		# Fourier transform planning area
		fftWrk = Array{setTyp}(undef, celInf..., lvls)
	# active GPU 		
	else
		egoFurDev = Array{CuArray{setTyp}}(undef, eoDim)
		phzInfDev = Array{CuArray{setTyp}}(undef, lvls)
		fftPlnFwdDev = Array{CUDA.CUFFT.cCuFFTPlan}(undef, lvls)
		fftPlnRevDev = Array{AbstractFFTs.ScaledPlan}(undef, lvls)
		fftWrkDev = CuArray{setTyp}(undef, celInf..., lvls)
	end
	###MEMORY PREPARATION
	# initialize Fourier transform plans
	if cmpInf.devMod == false
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
		if cmpInf.devMod == true 		
			phzInfDev[itr] = CuArray{setTyp}(undef, celInf...)
			copyto!(selectdim(phzInfDev, 1, itr), 
				selectdim(phzInf, 1, itr))
		end
	end
	# number of unique green function blocks
	ddDim = 2 * lvls
	# total number of target and source partitions
	totParTrg = prod(mixInf.trgDiv)
	totParSrc = prod(mixInf.srcDiv) 
	# transfer Fourier coefficients to GPU if active
	if cmpInf.devMod == true 
		# active GPU
		for eoItr ∈ 0:(eoDim - 1), ddItr ∈ 1:6
			egoFurDev[eoItr + 1] = CuArray{setTyp}(undef, hlfRup..., ddDim, 
				totParTrg, totParSrc)
			copyto!(selectdim(egoFurDev, 1, eoItr + 1), 
				selectdim(egoFur, 1, eoItr + 1))
		end
	end
	# wait for completion of GPU operation, create memory structure
	if cmpInf.devMod == true 
		CUDA.synchronize(CUDA.stream())
		GlaOprMem(cmpInf, trgVol, srcVol, mixInf, celInf, celInfDev, 
			egoFurDev, fftPlnFwdDev, fftPlnRevDev, phzInfDev)
	else
		return GlaOprMem(cmpInf, trgVol, srcVol, mixInf, celInf, celInf, 
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