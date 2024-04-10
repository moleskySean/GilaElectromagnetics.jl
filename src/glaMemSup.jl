"""

	function GlaOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol,
	srcVol::Union{GlaVol,Nothing}=nothing, 
	egoFur::Union{AbstractArray{<:AbstractArray{T}},
	Nothing}=nothing)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}

Prepare memory for green function operator---when called with a single GlaVol, 
or identical source and target volumes, yields the self construction. 
"""
function GlaOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol,
	srcVol::Union{GlaVol,Nothing}=nothing; 
	egoFur::Union{AbstractArray{<:AbstractArray{T}, 1},
	Nothing}=nothing, setType::DataType=ComplexF64)::GlaOprMem where 
	T<:Union{ComplexF64,ComplexF32}
	# flag for self green function case
	slfFlg = 0
	# self green function case
	if isnothing(srcVol) || trgVol == srcVol
		slfFlg = 1
		srcVol = trgVol
		mixInf = GlaExtInf(trgVol, srcVol)
	# external green function case
	else
		# ensure that sum number of cells is even, regenerate if not
		if sum(mod.(trgVol.cel .+ srcVol.cel, 2)) != 0	
			# regenerate target volume so that number of cells is even
			trgVol = glaVolEveGen(trgVol)
			# regenerate source volume so that number of cells is even
			srcVol = glaVolEveGen(srcVol)
		end
		# useful information for aligning source and target volumes
		mixInf = GlaExtInf(trgVol, srcVol)
	end
	# total cells in circulant
	totCelCrc = mixInf.trgCel .+ mixInf.srcCel
	# total number of target and source partitions
	totParTrg = prod(mixInf.trgDiv)
	totParSrc = prod(mixInf.srcDiv)
	# branching depth of multiplication
	lvl = 3
	# number of multiplication branches		
	eoDim = ^(2, lvl)
	# generate circulant green function, from GilaCrc module
	if isnothing(egoFur)
		# memory for circulant green function vector
		egoCrc = Array{ComplexF64}(undef, 3, 3, totCelCrc..., totParSrc, 
			totParTrg)
		# self green function case
		if slfFlg == 1
			genEgoSlf!(selectdim(selectdim(egoCrc, 7, 1), 6, 1), trgVol, cmpInf)
		# external green function case
		else
			# partition source and target for consistent distance offsets
			for trgItr ∈ eachindex(1:totParTrg)
				# target grid offset
				trgGrdOff = Tuple(mixInf.trgPar[trgItr]) 
				# offset of center of partition from center of volume
				trgOrgOff = Rational.((trgGrdOff .- (mixInf.trgDiv .- 1) 
					.// 2) .* trgVol.scl .+ trgVol.org)
				# grid scale of partition
				trgGrdScl = mixInf.trgDiv .* trgVol.scl
				# create target volume partition
				trgVolPar = GlaVol(mixInf.trgCel, trgVol.scl, trgOrgOff, 
					trgGrdScl)
				for srcItr ∈ eachindex(1:totParSrc) 
					# relative grid position
					srcGrdOff = Tuple(mixInf.srcPar[srcItr]) 
					# offset of center of partition from center of volume
					srcOrgOff = Rational.((srcGrdOff .- (mixInf.srcDiv .- 1) 
						.// 2) .* srcVol.scl .+ srcVol.org)
					# grid scale of partition
					srcGrdScl = mixInf.srcDiv .* srcVol.scl
					# create target volume partition
					srcVolPar = GlaVol(mixInf.srcCel, srcVol.scl, srcOrgOff, 
						srcGrdScl)
					# generate green function information for partition pair
					genEgoExt!(selectdim(selectdim(egoCrc, 7, trgItr), 6, 
						srcItr), trgVolPar, srcVolPar, cmpInf)
				end
			end	
		end
		# verify that egoCrc contains numeric values
		if maximum(isnan.(egoCrc)) == 1 || maximum(isinf.(egoCrc)) == 1
			error("Computed circulant contains non-numeric values.")
		end
		# Fourier transform of circulant green function
		egoFurPrp = Array{eltype(egoCrc)}(undef, totCelCrc..., 6, totParSrc, 
			totParTrg)
		# plan Fourier transform
		fftCrcOut = plan_fft(egoCrc[1,1,:,:,:,1,1], (1, 2, 3))
		# Fourier transform of the green function, making use of real space 
		# symmetry under transposition--entries are xx, yy, zz, xy, xz, yz
		for trgItr ∈ eachindex(1:totParTrg), srcItr ∈ eachindex(1:totParSrc),
			colItr ∈ eachindex(1:3), rowItr ∈ eachindex(1:colItr)
			# vector direction moved to outer volume index---largest stride
			egoFurPrp[:,:,:,blkEgoItr(3 * (colItr - 1) + rowItr), srcItr, 
				trgItr] =  fftCrcOut * egoCrc[rowItr,colItr,:,:,:,srcItr,trgItr]
		end
		# verify integrity of Fourier transform data
		if maximum(isnan.(egoFurPrp)) == 1 || maximum(isinf.(egoFurPrp)) == 1
			error("Fourier transform of circulant contains non-numeric values.")
		end
		# number of unique green function blocks
		ddDim = 6
		# Green function construction information
		mixInf = GlaExtInf(trgVol, srcVol)
		# determine whether source or target volume contains more cells
		srcDomDir = map(<, mixInf.trgCel, mixInf.srcCel)
		trgDomDir = map(!, srcDomDir)
		# number of unique elements in each cartesian index for a branch
		truInf = Array{Int}(undef,3)
		for dirItr ∈ eachindex(1:3)
			# row and column entries are symmetric or anti-symmetric
			if mixInf.trgCel[dirItr] == mixInf.srcCel[dirItr] && 
				prod(mixInf.srcDiv) == 1 && prod(mixInf.trgDiv) == 1 && 
				trgVol.org[dirItr] == srcVol.org[dirItr]
				# store only necessary information
				truInf[dirItr] = max(Integer(ceil(mixInf.trgCel[dirItr] / 2)) + 
					iseven(mixInf.trgCel[dirItr]), 2)
			# glaVolEveGen enforces that number of cells is even 
			else
				truInf[dirItr] = totCelCrc[dirItr] ÷ 2
			end
		end
		# information copy indicies
		cpyRng = tuple(map(UnitRange, ones(Int,3), truInf)...)
		# final Fourier coefficients for a given branch
		egoFur = Array{Array{setType}}(undef, eoDim)
		# intermediate storage
		egoFurInt = Array{setType}(undef, max.(div.(totCelCrc, 2), (2,2,2))..., 
			ddDim, totParSrc, totParTrg)
		# only one one eighth of the green function is unique 
		for eoItr ∈ 0:(eoDim - 1)
			# odd / even branch extraction
			egoFur[eoItr + 1] = Array{setType}(undef, truInf..., ddDim, 
				totParSrc, totParTrg)
			# first division is along smallest stride -> largest binary division
			egoFurInt[:,:,:,:,:,:] .= setType.(egoFurPrp[(1 + 
				mod(div(eoItr, 4), 2)):2:(end - 1 + mod(div(eoItr, 4), 2)), 
				(1 + mod(div(eoItr, 2), 2)):2:(end - 1 + mod(div(eoItr, 2), 2)),
				(1 + mod(eoItr, 2)):2:(end - 1 + mod(eoItr, 2)),:,:,:])
			# extract unique information
			@threads for cpyItr ∈ CartesianIndices(cpyRng)
				egoFur[eoItr + 1][cpyItr,:,:,:] .= egoFurInt[cpyItr,:,:,:]
			end
		end
	end
	# verify that egoCrc contains numeric values
	for eoItr ∈ eachindex(1:eoDim)
		if maximum(isnan.(egoFur[eoItr])) == 1 || 
				maximum(isinf.(egoFur[eoItr])) == 1
			error("Fourier information contains non-numeric values.")
		end
	end
	if cmpInf.devMod == true
		return GlaOprPrp(egoFur, trgVol, srcVol, mixInf, cmpInf, ComplexF32)
	else
		setType = eltype(eltype(egoFur))
		return GlaOprPrp(egoFur, trgVol, srcVol, mixInf, cmpInf, setType)
	end
end
#=
Memory preparation sub-protocol.
=#
function GlaOprPrp(egoFur::AbstractArray{<:AbstractArray{T}}, trgVol::GlaVol,
	srcVol::GlaVol, mixInf::GlaExtInf, cmpInf::GlaKerOpt, 
	setType::DataType)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}
	###MEMORY DECLARATION
	# number of embedding levels---dimensionality of ambient space
	lvls = 3
	# operator dimensions---unique vector information does not typically 
	# match operator size for distinct source and target volumes
	# sum of source and target volumes being divisible by 2 is guaranteed by 
	# glaVolEveGen in GlaOprMemGen
	brnSze = div.(mixInf.trgCel .+ mixInf.srcCel, 2)
	# binary indexing system of even and odd coefficient extraction
	eoDim = ^(2, lvls)
	# phase transformations (internal for block Toeplitz transformations)
	phzInf = Array{Array{setType}}(undef, lvls)
	# Fourier transform plans
	if cmpInf.devMod == false
		fftPlnFwd = Array{FFTW.cFFTWPlan}(undef, lvls)
		fftPlnRev = Array{FFTW.ScaledPlan}(undef, lvls)
	# active GPU 		
	else
		egoFurDev = Array{CuArray{setType}}(undef, eoDim)
		phzInfDev = Array{CuArray{setType}}(undef, lvls)
		fftPlnFwdDev = Array{CUDA.CUFFT.cCuFFTPlan}(undef, lvls)
		fftPlnRevDev = Array{AbstractFFTs.ScaledPlan}(undef, lvls)
	end
	###MEMORY PREPARATION
	# initialize Fourier transform plans
	if cmpInf.devMod == false
		for dir ∈ eachindex(1:lvls)
			# size of vector changes throughout application for external Green 
			vecSzeFwd = ntuple(x -> x <= dir ? brnSze[x] : mixInf.srcCel[x], 3)
			vecSzeRev = ntuple(x -> x > dir ? mixInf.trgCel[x] : brnSze[x], 3)
			# Fourier transform planning area
			fftWrkFwd = Array{setType}(undef, vecSzeFwd..., lvls, 
				prod(mixInf.srcDiv))
			fftWrkRev = Array{setType}(undef, vecSzeRev..., lvls, 
				prod(mixInf.trgDiv))
			# create Fourier transform plans
			fftPlnFwd[dir] = plan_fft!(fftWrkFwd, [dir]; flags = FFTW.MEASURE)
			fftPlnRev[dir] = plan_ifft!(fftWrkRev, [dir]; flags = FFTW.MEASURE)
		end
	else
		for dir ∈ eachindex(1:lvls)
			# size of vector changes throughout application for external Green 
			vecSzeFwd = ntuple(x -> x <= dir ? brnSze[x] : mixInf.srcCel[x], 3)
			vecSzeRev = ntuple(x -> x > dir ? mixInf.trgCel[x] : brnSze[x], 3)
			# Fourier transform planning area
			fftWrkFwdDev = CuArray{setType}(undef, vecSzeFwd..., lvls, 
				prod(mixInf.srcDiv))
			fftWrkRevDev = CuArray{setType}(undef, vecSzeRev..., lvls, 
				prod(mixInf.trgDiv))
			# create Fourier transform plans
			@CUDA.sync fftPlnFwdDev[dir] =  plan_fft!(fftWrkFwdDev, [dir])
			@CUDA.sync fftPlnRevDev[dir] =  plan_ifft!(fftWrkRevDev, [dir])
		end
	end
	# computation of phase transformation
	for itr ∈ eachindex(1:lvls)
		# allows calculation odd coefficient numbers
		phzInf[itr] = setType.([exp(-im * pi * k / brnSze[itr]) for 
			k ∈ 0:(brnSze[itr] - 1)])
		# active GPU
		if cmpInf.devMod == true 		
			phzInfDev[itr] = CuArray{setType}(undef, brnSze...)
			copyto!(selectdim(phzInfDev, 1, itr), 
				selectdim(phzInf, 1, itr))
		end
	end
	# number of unique green function blocks
	ddDim = 2 * lvls
	# total number of target and source partitions
	totParTrg = prod(mixInf.trgDiv)
	totParSrc = prod(mixInf.srcDiv) 
	# number of unique memory elements
	truInf = div.(max.(mixInf.trgCel, mixInf.srcCel), 2) .+ 1
	# transfer Fourier coefficients to GPU if active
	if cmpInf.devMod == true 
		# active GPU
		for eoItr ∈ 0:(eoDim - 1), ddItr ∈ eachindex(1:6)
			egoFurDev[eoItr + 1] = CuArray{setType}(undef, truInf..., ddDim, 
				totParSrc, totParTrg)
			copyto!(selectdim(egoFurDev, 1, eoItr + 1), 
				selectdim(egoFur, 1, eoItr + 1))
		end
	end
	# wait for completion of GPU operation, create memory structure
	if cmpInf.devMod == true 
		CUDA.synchronize(CUDA.stream())
		GlaOprMem(cmpInf, trgVol, srcVol, mixInf, brnSze, egoFurDev,
			fftPlnFwdDev, fftPlnRevDev, phzInfDev)
	else
		return GlaOprMem(cmpInf, trgVol, srcVol, mixInf, brnSze, egoFur,
			fftPlnFwd, fftPlnRev, phzInf)
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