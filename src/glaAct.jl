###MEMORY 
#=

	function GlaOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol,
	srcVol::Union{GlaVol,Nothing}=nothing, 
	egoFur::Union{AbstractArray{<:AbstractArray{T}},
	Nothing}=nothing)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}

Prepare memory for green function operator---when called with a single GlaVol, 
or identical source and target volumes, the function yields the self green 
function construction. 
=#
include("glaMemSup.jl")
###PROCEDURE
"""
	
	egoOpr!(egoMem::GlaOprMem)::Nothing

Act with the electric Green function on the memory location linked to egoMem. 
"""
#=
egoOpr! has no internal check for NaN entries---checks are done by GlaOprMem. 
=#
function egoOpr!(egoMem::GlaOprMem, 
	actVec::AbstractArray{T})::AbstractArray{T} where 
	T<:Union{ComplexF64,ComplexF32}
	
	return egoBrn!(egoMem, 0, 0, actVec)
end
# Support functions for operator action
include("glaActSup.jl")
"""
    egoBrn!(egoMem::GlaOprMem, lvl::Integer, bId::Integer, 
	actVec::AbstractArray{T})::AbstractArray{T} where 
	T<:Union{ComplexF64,ComplexF32}

Head branching function implementing Green function action.
"""
function egoBrn!(egoMem::GlaOprMem, lvl::Integer, bId::Integer, 
	actVec::AbstractArray{T})::AbstractArray{T} where 
	T<:Union{ComplexF64,ComplexF32}
	# generate branch pair to initiate operator
	# size of circulant vector
	brnSze = div.(egoMem.mixInf.trgCel .+ egoMem.mixInf.srcCel, 2)
	# number of source partitions
	parNumSrc = prod(egoMem.mixInf.srcDiv)
	# number of target partitions
	parNumTrg = prod(egoMem.mixInf.trgDiv)
	# if necessary, reshape source vector into partitions
	if lvl == 0 && sum(egoMem.mixInf.srcCel .!= size(actVec)[1:3]) > 0 
		# reshape current vector 
		# orgVec and actVec share the same underlying data
		orgVec = genPrt(egoMem.mixInf, egoMem.cmpInf, parNumSrc, actVec)
	elseif lvl == 0
		orgVec = reshape(actVec, egoMem.mixInf.srcCel..., 3, parNumSrc)
	else
		# partitions have been created, simply rename memory
		orgVec = actVec
	end
	# perform forward Fourier transform
	if lvl > 0
		# forward FFT
		# possibility of changing vector size accounted for in FFT plans
		egoMem.fftPlnFwd[lvl] * orgVec
		# GPU mode
		if sum(egoMem.cmpInf.devMod) == true
			CUDA.synchronize(CUDA.stream())
		end
	end 
	# split branch
	if lvl < length(egoMem.dimInf)	
		# check if size of current active vector dimension is compatible 
		if size(orgVec)[lvl + 1] == brnSze[lvl + 1]
			if sum(egoMem.cmpInf.devMod) == true
				prgVec = CuArray{eltype(orgVec)}(undef, size(orgVec)[1:3]...,
				3, parNumSrc)
				CUDA.synchronize(CUDA.stream())
			else 
				prgVec = similar(orgVec)
			end
			# split branch, includes phase operation and stream sync
			sptBrn!(size(orgVec)[1:3], prgVec, lvl + 1, egoMem.phzInf[lvl + 1], 
				parNumSrc, orgVec, egoMem.cmpInf)
			# rename vectors for consistent convention with general branching
			prgVecEve = orgVec 
			prgVecOdd = prgVec
		# size of progeny vectors must be changed
		else
			# current and progeny vector sizes
			curSze = size(orgVec)
			prgSze = ntuple(x -> x == (lvl + 1) ? brnSze[x] : curSze[x], 3)
			# allocate progeny vectors
			if sum(egoMem.cmpInf.devMod) == true
				prgVecEve = CuArray{eltype(orgVec)}(undef, prgSze..., 3,
					parNumSrc)
				prgVecOdd = CuArray{eltype(orgVec)}(undef, prgSze..., 3,
					parNumSrc)
				CUDA.synchronize(CUDA.stream())
			else
				prgVecEve = Array{eltype(orgVec)}(undef, prgSze..., 3, 
					parNumSrc) 
				prgVecOdd = Array{eltype(orgVec)}(undef, prgSze..., 3,
					parNumSrc) 
			end
			# memory previously associated with orgVec is freed internally
			sptBrn!(prgVecEve, prgVecOdd, lvl + 1, egoMem.phzInf[lvl + 1], 
				egoMem.mixInf, parNumSrc, orgVec, egoMem.cmpInf)
		end
	end
	# branch until depth of block structure
	if lvl < length(egoMem.dimInf)	
		# execute split branches 
		# !asynchronous GPU causes mysterious errors + minimal speed up!
		if sum(egoMem.cmpInf.devMod) == true
			# even branch
			eveVec = egoBrn!(egoMem, lvl + 1, bId, prgVecEve)	
			# !wait for even branch to return, functionality choice!
			if lvl < length(egoMem.dimInf)	- 1
				CUDA.synchronize(CUDA.stream())	
			end
			# phase modified branch
			oddVec = egoBrn!(egoMem, lvl + 1, 
				nxtBrnId(length(egoMem.dimInf), lvl, bId), prgVecOdd)
		# !asynchronous CPU is fine + some speed up!
		else
			@sync begin
				# origin branch
				Base.Threads.@spawn eveVec = 
					egoBrn!(egoMem, lvl + 1, bId, prgVecEve)
				# phase modified branch
				Base.Threads.@spawn oddVec = 
					egoBrn!(egoMem, lvl + 1, nxtBrnId(length(egoMem.dimInf), 
					lvl, bId), prgVecOdd)
			end
		end
		# check if size of vector must change on merge
		if size(eveVec)[lvl + 1] == egoMem.mixInf.trgCel[lvl + 1]
			# merge branches, includes phase operation and stream sync
			mrgBrn!(egoMem.mixInf, eveVec, lvl + 1, parNumTrg, 
				egoMem.phzInf[lvl + 1], oddVec, egoMem.cmpInf)
			# rename eveVec to match return name used elsewhere
			retVec = eveVec
		# merge and resize
		else
			# size of current vector
			curSze = size(eveVec)
			# size of cartesian indices of merged vector
			mrgSze = ntuple(x -> x == (lvl + 1) ? egoMem.mixInf.trgCel[x] : 
				curSze[x], 3)
			# allocate memory for merge
			if sum(egoMem.cmpInf.devMod) == true
				retVec = CuArray{eltype(eveVec)}(undef, mrgSze..., 3, 
					parNumTrg)
			else
				retVec = Array{eltype(eveVec)}(undef, mrgSze..., 3, parNumTrg)
			end
			# merge branches, even and odd vectors are freed, stream synced	
			mrgBrn!(egoMem.mixInf, retVec, lvl + 1, parNumTrg, 
				egoMem.phzInf[lvl + 1], eveVec, oddVec, egoMem.cmpInf)
		end
	else
		if sum(egoMem.cmpInf.devMod) == true
			retVec = CUDA.zeros(eltype(orgVec), brnSze..., 3, parNumTrg)
			CUDA.synchronize(CUDA.stream())
		else 
			retVec = zeros(eltype(orgVec), brnSze..., 3, parNumTrg)
		end
		# multiply by Toeplitz vector, avoid race condition DO NOT SPAWN!
		for trgItr in eachindex(1:parNumTrg) 
			for srcItr in eachindex(1:parNumSrc) 
				# CUDA stream is synchronized inside mulBrn!
				mulBrn!(egoMem.mixInf, bId, selectdim(retVec, 5, trgItr),
				view(egoMem.egoFur[bId + 1], :, :, :, :, srcItr, trgItr), 
				selectdim(orgVec, 5, srcItr), egoMem.cmpInf)
			end
		end
		# free orgVec
		if sum(egoMem.cmpInf.devMod) == true
			CUDA.synchronize(CUDA.stream())
			CUDA.unsafe_free!(orgVec)
			CUDA.synchronize(CUDA.stream())
		end
	end
	# perform reverse Fourier transform
	if lvl > 0
		# inverse FFT
		# possibility of changing vector size accounted for in FFT plans
		egoMem.fftPlnRev[lvl] * retVec
		# GPU mode
		if sum(egoMem.cmpInf.devMod) == true
			CUDA.synchronize(CUDA.stream())
		end
	end
	# terminate task and return control to previous level 
	if sum(egoMem.cmpInf.devMod) == true
		CUDA.synchronize(CUDA.stream())
	end
	# zero level return
	if lvl == 0
		# check if there are partitions that need to be merged
		if parNumTrg == 1
			return reshape(retVec, egoMem.mixInf.trgCel..., 3)
		else
			return mrgPrt(egoMem.mixInf, egoMem.cmpInf, parNumTrg, retVec)
		end
	end
	return retVec
end