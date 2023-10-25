function egoBrn!(egoMem::GlaOprMem, lvl::Integer, bId::Integer, 
	actVec::AbstractArray{T,4})::Nothing where T<:Union{ComplexF64,ComplexF32}

	if lvl > 0
		# forward FFT
		egoMem.fftPlnFwd[lvl] * actVec
		# GPU mode
		if egoMem.cmpInf.dev == true
			CUDA.synchronize(CUDA.stream())
		end
	end 
	# branch until depth of block structure
	if lvl < length(egoMem.dimInfC)	
		# shallow copyto of actVec, inner structure of actVec must be a literal
		if egoMem.cmpInf.dev == true
			prgVec = CuArray{eltype(actVec)}(undef, egoMem.dimInfC..., 3)
			CUDA.synchronize(CUDA.stream())
		else 
			prgVec = similar(actVec)
		end
		# split branch, includes phase operation and stream sync
		sptBrn!(egoMem.dimInfD, prgVec, lvl + 1, egoMem.phzInf[lvl + 1], 
			actVec, egoMem.cmpInf)
		# execute split branches 
		# !asynchronous GPU causes mysterious errors + minimal speed up!
		if egoMem.cmpInf.dev == true
				# origin branch
				egoBrn!(egoMem, lvl + 1, bId, actVec)		
				# phase modified branch
				egoBrn!(egoMem, lvl + 1, 
					nxtBrnId(length(egoMem.dimInfC), lvl, bId), prgVec)
		# !asynchronous CPU is fine + some speed up!
		else
			@sync begin
				# origin branch
				Base.Threads.@spawn egoBrn!(egoMem, lvl + 1, bId, actVec)
				# phase modified branch
				Base.Threads.@spawn egoBrn!(egoMem, lvl + 1, 
					nxtBrnId(length(egoMem.dimInfC), lvl, bId), prgVec)
			end
		end
		# merge branches, includes phase operation and stream sync
		mrgBrn!(egoMem.dimInfD, actVec, lvl + 1, 
			egoMem.phzInf[lvl + 1], prgVec, egoMem.cmpInf)		
	else
		# shallow copy of actVec, inner structure of actVec must be a literal
		if egoMem.cmpInf.dev == true
			vecHld = CuArray{eltype(actVec)}(undef, egoMem.dimInfC..., 3)
			copyto!(vecHld, actVec)
			CUDA.synchronize(CUDA.stream())
		else 
			vecHld = similar(actVec)
			copyto!(vecHld, actVec)
		end
		# multiply by Toeplitz vector 
		mulBrn!(egoMem.dimInfD, actVec, vecHld, egoMem.egoFur[bId + 1], 
			egoMem.cmpInf)
	end
	# perform reverse Fourier transform
	if lvl > 0
		# inverse FFT
		egoMem.fftPlnRev[lvl] * actVec
		# GPU mode
		if egoMem.cmpInf.dev == true
			CUDA.synchronize(CUDA.stream())
		end
	end
	# terminate task and return control to previous level 
	if egoMem.cmpInf.dev == true
		CUDA.synchronize(CUDA.stream())
	end
	return nothing
end
# split a branch so that even and odd Fourier coefficients are independent
function sptBrn!(dimInf::AbstractVector{<:Integer}, 
	prgVec::AbstractArray{T,4}, phzDir::Integer, phzVec::AbstractVector{T}, 
	actVec::AbstractArray{T,4}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	if cmpInf.dev == true
		@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk sptKer!(dimInf, 
				prgVec, phzDir, phzVec, actVec)
		CUDA.synchronize(CUDA.stream())
	else 
		@inbounds @threads for itr ∈ CartesianIndices(actVec) 
						prgVec[itr] = phzVec[itr[phzDir]] * actVec[itr]
		end
	end
	return nothing
end
# device kernel for sptBrn!
function sptKer!(dimInf::AbstractVector{<:Integer}, 
	prgVec::AbstractArray{T,4}, phzDir::Integer, phzVec::AbstractVector{T}, 
	actVec::AbstractArray{T,4})::Nothing where T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 1) * blockDim().x 
	idY = threadIdx().y + (blockIdx().y - 1) * blockDim().y 
	idZ = threadIdx().z + (blockIdx().z - 1) * blockDim().z 
	# multiplication loop selected based on phase transformation direction
	if phzDir == 1
		# x phase---phzVec[itrX]
		@inbounds for itrZ = idZ:strZ:dimInf[3], 
			itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
			# x direction
			prgVec[itrX,itrY,itrZ,1] = phzVec[itrX] * 
				actVec[itrX,itrY,itrZ,1]
			# y direction
			prgVec[itrX,itrY,itrZ,2] = phzVec[itrX] * 
				actVec[itrX,itrY,itrZ,2]
			# z direction
			prgVec[itrX,itrY,itrZ,3] = phzVec[itrX] * 
				actVec[itrX,itrY,itrZ,3]
		end
	elseif phzDir == 2
		# y phase---phzVec[itrY]
		@inbounds for itrZ = idZ:strZ:dimInf[3], 
			itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
			# x direction
			prgVec[itrX,itrY,itrZ,1] = phzVec[itrY] * 
				actVec[itrX,itrY,itrZ,1]
			# y direction
			prgVec[itrX,itrY,itrZ,2] = phzVec[itrY] * 
				actVec[itrX,itrY,itrZ,2]
			# z direction
			prgVec[itrX,itrY,itrZ,3] = phzVec[itrY] * 
				actVec[itrX,itrY,itrZ,3]
		end
	else 
		# z phase---phzVec[itrZ]
		@inbounds for itrZ = idZ:strZ:dimInf[3], 
			itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
			# x direction
			prgVec[itrX,itrY,itrZ,1] = phzVec[itrZ] * 
				actVec[itrX,itrY,itrZ,1]
			# y direction
			prgVec[itrX,itrY,itrZ,2] = phzVec[itrZ] * 
				actVec[itrX,itrY,itrZ,2]
			# z direction
			prgVec[itrX,itrY,itrZ,3] = phzVec[itrZ] * 
				actVec[itrX,itrY,itrZ,3]
		end
	end
	return nothing
end
# merge two branches, eliminating unused coefficients
function mrgBrn!(dimInf::AbstractVector{<:Integer}, 
	actVec::AbstractArray{T,4}, phzDir::Integer, phzVec::AbstractVector{T}, 
	prgVec::AbstractArray{T,4}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	if cmpInf.dev == true
		for vecDir ∈ 1:3
			@cuda threads = cmpInf.numTrd blocks = cmpInf.numBlk mrgKer!(dimInf, 
				actVec, phzDir, phzVec, vecDir, prgVec)
		end
		CUDA.synchronize(CUDA.stream())
		CUDA.unsafe_free!(prgVec)
		CUDA.synchronize(CUDA.stream())
	else
		@inbounds @threads for itr ∈ CartesianIndices(actVec) 
						actVec[itr] = (actVec[itr] + 
				conj(phzVec[itr[phzDir]]) * prgVec[itr]) / 2
		end
	end
	return nothing
end
# device kernel for mrgBrn!
function mrgKer!(dimInf::AbstractVector{<:Integer}, 
	actVec::AbstractArray{T,4}, phzDir::Integer, phzVec::AbstractVector{T}, 
	vecDir::Integer, prgVec::AbstractArray{T,4})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 1) * blockDim().x 
	idY = threadIdx().y + (blockIdx().y - 1) * blockDim().y 
	idZ = threadIdx().z + (blockIdx().z - 1) * blockDim().z 
	# multiplication loop selected based on phase transformation direction
	if phzDir == 1
		# x phase---phzVec[itrX]
		@inbounds for itrZ = idZ:strZ:dimInf[3], 
			itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
			actVec[itrX,itrY,itrZ,vecDir] = 
				(actVec[itrX,itrY,itrZ,vecDir] + conj(phzVec[itrX]) * 
				prgVec[itrX,itrY,itrZ,vecDir]) / 2.0
		end
	elseif phzDir == 2 
		# y phase---phzVec[itrY]
		@inbounds for itrZ = idZ:strZ:dimInf[3], 
			itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
			actVec[itrX,itrY,itrZ,vecDir] = 
				(actVec[itrX,itrY,itrZ,vecDir] + conj(phzVec[itrY]) * 
				prgVec[itrX,itrY,itrZ,vecDir]) / 2.0
		end
	else
		# z phase---phzVec[itrZ]
		@inbounds for itrZ = idZ:strZ:dimInf[3], 
			itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
			actVec[itrX,itrY,itrZ,vecDir] = 
				(actVec[itrX,itrY,itrZ,vecDir] + conj(phzVec[itrZ]) * 
				prgVec[itrX,itrY,itrZ,vecDir]) / 2.0
		end
	end
	return nothing
end
# execute Hadamard product step
function mulBrn!(dimInf::AbstractVector{<:Integer}, 
	actVec::AbstractArray{T,4}, vecHld::AbstractArray{T,4}, 
	vecMod::AbstractArray{T,4}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	if cmpInf.dev == true
		@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk mulKer!(dimInf, 
			actVec, vecHld, vecMod)
		CUDA.synchronize(CUDA.stream())
		CUDA.unsafe_free!(vecHld)
		CUDA.synchronize(CUDA.stream())
	else 
		@inbounds @threads for itr ∈ CartesianIndices(selectdim(actVec, 4, 1)) 
			# vecMod follows ii, jj, kk, ij, ik, jk storage format
			actVec[itr,1] = vecMod[itr,1] * vecHld[itr,1] + 
				vecMod[itr,4] * vecHld[itr,2] + vecMod[itr,5] * vecHld[itr,3]
			# j vector direction
			actVec[itr,2] = vecMod[itr,4] * vecHld[itr,1] + 
				vecMod[itr,2] * vecHld[itr,2] + vecMod[itr,6] * vecHld[itr,3]
			# k vector direction
			actVec[itr,3] = vecMod[itr,5] * vecHld[itr,1] + 
				vecMod[itr,6] * vecHld[itr,2] + vecMod[itr,3] * vecHld[itr,3]
		end
	end
	return nothing
end
# device kernel for mulBrn!
function mulKer!(dimInf::AbstractVector{<:Integer}, 
	actVec::AbstractArray{T,4}, vecHld::AbstractArray{T,4}, 
	vecMod::AbstractArray{T,4})::Nothing where T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 1) * blockDim().x 
	idY = threadIdx().y + (blockIdx().y - 1) * blockDim().y 
	idZ = threadIdx().z + (blockIdx().z - 1) * blockDim().z 
	# multiplication loops
	@inbounds for itrZ = idZ:strZ:dimInf[3], 
		itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
		# vecMod follows ii, jj, kk, ij, ik, jk storage format
		actVec[itrX,itrY,itrZ,1] = vecMod[itrX,itrY,itrZ,1] * 
			vecHld[itrX,itrY,itrZ,1] + vecMod[itrX,itrY,itrZ,4] * 
			vecHld[itrX,itrY,itrZ,2] + vecMod[itrX,itrY,itrZ,5] * 
			vecHld[itrX,itrY,itrZ,3]
		# j vector direction
		actVec[itrX,itrY,itrZ,2] = vecMod[itrX,itrY,itrZ,4] * 
			vecHld[itrX,itrY,itrZ,1] + vecMod[itrX,itrY,itrZ,2] * 
			vecHld[itrX,itrY,itrZ,2] + vecMod[itrX,itrY,itrZ,6] * 
			vecHld[itrX,itrY,itrZ,3]
		# k vector direction
		actVec[itrX,itrY,itrZ,3] = vecMod[itrX,itrY,itrZ,6] * 
			vecHld[itrX,itrY,itrZ,1] + vecMod[itrX,itrY,itrZ,5] * 
			vecHld[itrX,itrY,itrZ,2] + vecMod[itrX,itrY,itrZ,3] * 
			vecHld[itrX,itrY,itrZ,3]
	end
	return nothing
end
# binary branch indexing.
@inline function nxtBrnId(maxLvl::Integer, lvl::Integer, bId::Integer)::Integer
	return bId + ^(2, maxLvl - (lvl + 1))
end