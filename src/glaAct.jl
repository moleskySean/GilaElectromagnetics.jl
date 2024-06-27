###MEMORY 
#=

	GlaOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol, 
	srcVol::Union{GlaVol,Nothing}=nothing, 
	egoFur::Union{AbstractArray{<:AbstractArray{T}},
	Nothing}=nothing)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}

Prepare memory for green function operator---when called with a single GlaVol, 
or identical source and target volumes, the function yields the self green 
function construction. 

	GlaOpr(celSrc::NTuple{3, Int}, sclSrc::NTuple{3, Rational}, 
	orgSrc::NTuple{3, Rational}, celTrg::NTuple{3, Int}, 
	sclTrg::NTuple{3, Rational}, orgTrg::NTuple{3, Rational}; 
	useGpu::Bool=false, setType::DataType=ComplexF64)

Construct an external Green's operator.

=#
include("glaMemSup.jl")
###PROCEDURE
"""
	egoOpr!(egoMem::GlaOprMem, 
	actVec::AbstractArray{T})::AbstractArray{T} where 
	T<:Union{ComplexF64,ComplexF32}

Applies the electric Green function to actVec (current), returning the output 
field. actVec is internally modified by egoOpr! to reduce needed allocation. 
"""
#=
egoOpr! has no internal check for NaN entries---checks are done by GlaOprMem. 
=#
function egoOpr!(egoMem::GlaOprMem, 
	actVec::AbstractArray{T})::AbstractArray{T} where 
	T<:Union{ComplexF64,ComplexF32}
	# check that actVec is the correct size
	if (egoMem.mixInf.srcCel..., 3) == size(actVec)
		# device execution
		if prod(egoMem.cmpInf.devMod)
			return egoBrnDev!(egoMem, 0, 0, actVec)
		else
			return egoBrnHst!(egoMem, 0, 0, actVec)
		end
	else
		error("Size of input vector does not match size of source volume. 
		Required data layout is (srcCelX, srcCelY, srcCelZ, 3).")	
		return actVec
	end
end
# Support functions for operator action
include("glaActSup.jl")
"""
    egoBrnHst!(egoMem::GlaOprMem, lvl::Integer, bId::Integer, 
	actVec::AbstractArray{T})::AbstractArray{T} where 
	T<:Union{ComplexF64,ComplexF32}

Head branching function implementing Green function action on host.
"""
function egoBrnHst!(egoMem::GlaOprMem, lvl::Integer, bId::Integer, 
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
		orgVec = genPrtHst(egoMem.mixInf, egoMem.cmpInf, parNumSrc, actVec)
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
	end 
	# split branch
	if lvl < length(egoMem.dimInf)	
		# check if size of current active vector dimension is compatible 
		if size(orgVec)[lvl + 1] == brnSze[lvl + 1]
			prgVec = similar(orgVec)
			# split branch: perform phase shift
			sptBrnHst!(size(orgVec)[1:3], prgVec, lvl + 1, 
				egoMem.phzInf[lvl + 1], parNumSrc, orgVec, egoMem.cmpInf)
			# rename vectors for consistent convention with general branching
			prgVecEve = orgVec 
			prgVecOdd = prgVec
		# size of progeny vectors must be changed
		else
			# current and progeny vector sizes
			curSze = size(orgVec)
			prgSze = ntuple(x -> x == (lvl + 1) ? brnSze[x] : curSze[x], 3)
			# allocate progeny vectors
			prgVecEve = Array{eltype(orgVec)}(undef, prgSze..., 3, 
				parNumSrc) 
			prgVecOdd = Array{eltype(orgVec)}(undef, prgSze..., 3,
				parNumSrc) 
			# memory previously associated with orgVec is freed internally
			sptBrnHst!(prgVecEve, prgVecOdd, lvl + 1, egoMem.phzInf[lvl + 1], 
				egoMem.mixInf, parNumSrc, orgVec, egoMem.cmpInf)
		end
	end
	# branch until depth of block structure
	if lvl < length(egoMem.dimInf)	
		# !create external handles for sync---!not sure why this is needed!
		eveVec = prgVecEve
		oddVec = prgVecOdd
		# execute split branches---!asynchronous CPU is fine + some speed up!		
		@sync begin
			# origin branch
			Base.Threads.@spawn eveVec = egoBrnHst!(egoMem, lvl + 1, bId, 
				prgVecEve)
			# phase modified branch
			Base.Threads.@spawn oddVec = egoBrnHst!(egoMem, lvl + 1, 
				nxtBrnId(length(egoMem.dimInf), lvl, bId), prgVecOdd)
		end
		# check if size of vector must change on merge
		if size(eveVec)[lvl + 1] == egoMem.mixInf.trgCel[lvl + 1]
			# merge branches, includes phase operation and stream sync
			mrgBrnHst!(egoMem.mixInf, eveVec, lvl + 1, parNumTrg, 
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
			mrgBrnHst!(egoMem.mixInf, retVec, lvl + 1, parNumTrg, 
				egoMem.phzInf[lvl + 1], eveVec, oddVec, egoMem.cmpInf)
		end
	else
		retVec = zeros(eltype(orgVec), brnSze..., 3, parNumTrg)
		# multiply by Toeplitz vector, avoid race condition DO NOT SPAWN!
		for trgItr in eachindex(1:parNumTrg) 
			for srcItr in eachindex(1:parNumSrc) 
				mulBrnHst!(egoMem.mixInf, bId, selectdim(retVec, 5, trgItr),
				view(egoMem.egoFur[bId + 1], :, :, :, :, srcItr, trgItr), 
				selectdim(orgVec, 5, srcItr), egoMem.cmpInf)
			end
		end
	end
	# perform reverse Fourier transform
	if lvl > 0
		# inverse FFT
		# possibility of changing vector size accounted for in FFT plans
		egoMem.fftPlnRev[lvl] * retVec
	end
	# terminate task and return control to previous level 
	# zero level return
	if lvl == 0
		# check if there are partitions that need to be merged
		if parNumTrg == 1
			return reshape(retVec, egoMem.mixInf.trgCel..., 3)
		else
			return mrgPrtHst(egoMem.mixInf, egoMem.cmpInf, parNumTrg, retVec)
		end
	end
	return retVec
end
"""
    egoBrnDev!(egoMem::GlaOprMem, lvl::Integer, bId::Integer, 
	actVec::AbstractArray{T})::AbstractArray{T} where 
	T<:Union{ComplexF64,ComplexF32}

Head branching function implementing Green function action on device.
"""
function egoBrnDev!(egoMem::GlaOprMem, lvl::Integer, bId::Integer, 
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
		orgVec = genPrtDev(egoMem.mixInf, egoMem.cmpInf, parNumSrc, actVec)
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
		# wait for completion of FFT
		CUDA.synchronize(CUDA.stream())
	end 
	# split branch
	if lvl < length(egoMem.dimInf)	
		# check if size of current active vector dimension is compatible 
		if size(orgVec)[lvl + 1] == brnSze[lvl + 1]
			prgVec = CuArray{eltype(orgVec)}(undef, size(orgVec)[1:3]...,
				3, parNumSrc)
			# split branch, includes phase operation and stream sync
			sptBrnDev!(size(orgVec)[1:3], prgVec, lvl + 1, 
				egoMem.phzInf[lvl + 1], parNumSrc, orgVec, egoMem.cmpInf)
			# rename vectors for consistent convention with general branching
			prgVecEve = orgVec 
			prgVecOdd = prgVec
		# size of progeny vectors must be changed
		else
			# current and progeny vector sizes
			curSze = size(orgVec)
			prgSze = ntuple(x -> x == (lvl + 1) ? brnSze[x] : curSze[x], 3)
			# allocate progeny vectors
			prgVecEve = CuArray{eltype(orgVec)}(undef, prgSze..., 3,
				parNumSrc)
			prgVecOdd = CuArray{eltype(orgVec)}(undef, prgSze..., 3,
				parNumSrc)
			CUDA.synchronize(CUDA.stream())
			# memory previously associated with orgVec is freed internally
			sptBrnDev!(prgVecEve, prgVecOdd, lvl + 1, egoMem.phzInf[lvl + 1], 
				egoMem.mixInf, parNumSrc, orgVec, egoMem.cmpInf)
		end
	end
	# branch until depth of block structure
	if lvl < length(egoMem.dimInf)	
		# execute split branches---!async causes errors + no speed up!
		# even branch
		eveVec = egoBrnDev!(egoMem, lvl + 1, bId, prgVecEve)	
		# !wait for even branch to return, functionality choice!
		if lvl < length(egoMem.dimInf)	- 1
			CUDA.synchronize(CUDA.stream())	
		end
		# phase modified branch
		oddVec = egoBrnDev!(egoMem, lvl + 1, 
			nxtBrnId(length(egoMem.dimInf), lvl, bId), prgVecOdd)
		# check if size of vector must change on merge
		if size(eveVec)[lvl + 1] == egoMem.mixInf.trgCel[lvl + 1]
			# merge branches, includes phase operation and stream sync
			mrgBrnDev!(egoMem.mixInf, eveVec, lvl + 1, parNumTrg, 
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
			mrgBrnDev!(egoMem.mixInf, retVec, lvl + 1, parNumTrg, 
				egoMem.phzInf[lvl + 1], eveVec, oddVec, egoMem.cmpInf)
		end
	else
		retVec = CUDA.zeros(eltype(orgVec), brnSze..., 3, parNumTrg)
		CUDA.synchronize(CUDA.stream())
		# multiply by Toeplitz vector, avoid race condition DO NOT SPAWN!
		for trgItr in eachindex(1:parNumTrg) 
			for srcItr in eachindex(1:parNumSrc) 
				# CUDA stream is synchronized inside mulBrn!
				mulBrnDev!(egoMem.mixInf, bId, selectdim(retVec, 5, trgItr),
				view(egoMem.egoFur[bId + 1], :, :, :, :, srcItr, trgItr), 
				selectdim(orgVec, 5, srcItr), egoMem.cmpInf)
			end
		end
		# free orgVec
		CUDA.synchronize(CUDA.stream())
		CUDA.unsafe_free!(orgVec)
		CUDA.synchronize(CUDA.stream())
	end
	# perform reverse Fourier transform
	if lvl > 0
		# inverse FFT
		# possibility of changing vector size accounted for in FFT plans
		egoMem.fftPlnRev[lvl] * retVec
		CUDA.synchronize(CUDA.stream())
	end
	# terminate task and return control to previous level 
	CUDA.synchronize(CUDA.stream())
	# zero level return
	if lvl == 0
		# check if there are partitions that need to be merged
		if parNumTrg == 1
			return reshape(retVec, egoMem.mixInf.trgCel..., 3)
		else
			return mrgPrtDev(egoMem.mixInf, egoMem.cmpInf, parNumTrg, retVec)
		end
	end
	return retVec
end
###INTEGRATION
#=
Basic linear algebra definitions for egoOpr, allowing integration and simplified
use. 
=#
include("glaLinAlg.jl")