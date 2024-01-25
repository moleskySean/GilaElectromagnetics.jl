#= 
head branching function
=#
function egoBrn!(egoMem::GlaOprMem, lvl::Integer, bId::Integer, 
	actVec::AbstractArray{T})::AbstractArray{T} where 
	T<:Union{ComplexF64,ComplexF32}
	# size of circulant vector
	#WORK HERE
	# 	 
	if lvl > 0
		# forward FFT
		egoMem.fftPlnFwd[lvl] * actVec
		# GPU mode
		if sum(egoMem.cmpInf.devMod) == true
			CUDA.synchronize(CUDA.stream())
		end
	end 
	# generate branch pair
	# if necessary, reshape source vector to match required partitions
	if lvl == 0 && sum(egoMem.mixInf.srcCel .!= size(actVec)[1:3]) > 0 
		# reshape current vector to 
		# actVecPar and actVec share the same underlying data
	else
		# shallow copyto of actVec, inner structure of actVec must be a literal
		if sum(egoMem.cmpInf.devMod) == true
			prgVec = CuArray{eltype(actVec)}(undef, egoMem.dimInfC..., 3)
			CUDA.synchronize(CUDA.stream())
		else 
			prgVec = similar(actVec)
		end
		# split branch, includes phase operation and stream sync
		sptBrn!(egoMem.dimInfD, prgVec, lvl + 1, egoMem.phzInf[lvl + 1], 
			actVec, egoMem.cmpInf)
	end
	# branch until depth of block structure
	if lvl < length(egoMem.dimInfC)	
		# execute split branches 
		# !asynchronous GPU causes mysterious errors + minimal speed up!
		if sum(egoMem.cmpInf.devMod) == true
				# origin branch
				actVec = egoBrn!(egoMem, lvl + 1, bId, actVec)	
				# !wait for origin branch to return, functionality choice!
				if lvl < length(egoMem.dimInfC)	- 1
					CUDA.synchronize(CUDA.stream())	
				end
				# phase modified branch
				prgVec = egoBrn!(egoMem, lvl + 1, 
					nxtBrnId(length(egoMem.dimInfC), lvl, bId), prgVec)
		# !asynchronous CPU is fine + some speed up!
		else
			@sync begin
				# origin branch
				Base.Threads.@spawn actVec = 
					egoBrn!(egoMem, lvl + 1, bId, actVec)
				# phase modified branch
				Base.Threads.@spawn prgVec = 
					egoBrn!(egoMem, lvl + 1, nxtBrnId(length(egoMem.dimInfC), 
					lvl, bId), prgVec)
			end
		end
		# merge branches, includes phase operation and stream sync
		mrgBrn!(egoMem.dimInfD, actVec, lvl + 1, 
			egoMem.phzInf[lvl + 1], prgVec, egoMem.cmpInf)		
	else
		# shallow copy of actVec, inner structure of actVec must be a literal
		if sum(egoMem.cmpInf.devMod) == true
			vecHld = CuArray{eltype(actVec)}(undef, egoMem.dimInfC..., 3)
			copyto!(vecHld, actVec)
			CUDA.synchronize(CUDA.stream())
		else 
			vecHld = similar(actVec)
			copyto!(vecHld, actVec)
		end
		# multiply by Toeplitz vector 
		mulBrn!(egoMem.dimInfC, bId, actVec, vecHld, egoMem.egoFur[bId + 1], 
			egoMem.cmpInf)
	end
	# perform reverse Fourier transform
	if lvl > 0
		# inverse FFT
		egoMem.fftPlnRev[lvl] * actVec
		# GPU mode
		if sum(egoMem.cmpInf.devMod) == true
			CUDA.synchronize(CUDA.stream())
		end
	end
	# terminate task and return control to previous level 
	if sum(egoMem.cmpInf.devMod) == true
		CUDA.synchronize(CUDA.stream())
	end
	return actVec
end
#=
split a branch so that even and odd Fourier coefficients are independent
=#
function sptBrn!(dimInf::AbstractVector{<:Integer}, 
	prgVec::AbstractArray{T,4}, phzDir::Integer, phzVec::AbstractVector{T}, 
	actVec::AbstractArray{T,4}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	if cmpInf.devMod == true
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
#=
device kernel for sptBrn!
=#
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
#=
merge two branches, eliminating unused coefficients
=#
function mrgBrn!(dimInf::AbstractVector{<:Integer},  
	actVec::AbstractArray{T,4}, phzDir::Integer, phzVec::AbstractVector{T}, 
	prgVec::AbstractArray{T,4}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	if cmpInf.devMod == true
		for vecDir ∈ 1:3
			@cuda threads = cmpInf.numTrd blocks = cmpInf.numBlk mrgKer!(dimInf, 
				actVec, phzDir, phzVec, vecDir, prgVec)
		end
		CUDA.synchronize(CUDA.stream())
		CUDA.unsafe_free!(prgVec)
		CUDA.synchronize(CUDA.stream())
	else
		@inbounds @threads for itr ∈ CartesianIndices(actVec) 
						actVec[itr] = 0.5 * (actVec[itr] + 
				conj(phzVec[itr[phzDir]]) * prgVec[itr])
		end
	end
	return nothing
end
#=
device kernel for mrgBrn!
=#
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
				0.5 * (actVec[itrX,itrY,itrZ,vecDir] + conj(phzVec[itrX]) * 
				prgVec[itrX,itrY,itrZ,vecDir]) 
		end
	elseif phzDir == 2 
		# y phase---phzVec[itrY]
		@inbounds for itrZ = idZ:strZ:dimInf[3], 
			itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
			actVec[itrX,itrY,itrZ,vecDir] = 
				0.5 * (actVec[itrX,itrY,itrZ,vecDir] + conj(phzVec[itrY]) * 
				prgVec[itrX,itrY,itrZ,vecDir]) 
		end
	else
		# z phase---phzVec[itrZ]
		@inbounds for itrZ = idZ:strZ:dimInf[3], 
			itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
			actVec[itrX,itrY,itrZ,vecDir] = 
				0.5 * (actVec[itrX,itrY,itrZ,vecDir] + conj(phzVec[itrZ]) * 
				prgVec[itrX,itrY,itrZ,vecDir])
		end
	end
	return nothing
end
#=
execute Hadamard product step
=#
function mulBrn!(dimInfC::AbstractVector{<:Integer}, bId::Integer,
	actVec::AbstractArray{T,4}, vecHld::AbstractArray{T,4}, 
	vecMod::AbstractArray{T,4}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# branch symmetry properties (even / odd coefficients + size info)
	hlfRup = Integer.(ceil.(dimInfC ./ 2))
	hlfRdw = Integer.(floor.(dimInfC ./ 2))
	brnSym = [1 - mod(div(bId, ^(2, 3 - k)), 2) for k ∈ 1:3]
	# declare memory 
	if cmpInf.devMod == true
		dirNumDev = CuArray{Int32}(undef, 3, 8)
		dirSymDev = CuArray{Float32}(undef, 3, 8)
	else
		dirSymHst = Array{eltype(actVec)}(undef, 3, 8)
	end
	# division iteration references divisions of the Green function
	@sync begin
		for divItr ∈ 0:7
			# one for reverse direction, zero for forward
			revSwt = [mod(div(divItr, ^(2, k - 1)), 2) for k ∈ 1:3]
			# one for forward direction, zero for reverse
			fwdSwt = [1,1,1] .- revSwt
			# negative one for reverse direction, one for forward
			symSgn = [1,1,1] .- 2 .* revSwt
			# range of active vector
			frsActVec = [1,1,1] .+ (dimInfC .- 1) .* revSwt 	
			lstActVec = hlfRup .* fwdSwt .+ hlfRdw .* revSwt .+ brnSym .+ revSwt 
			# range of Green function
			frsItrMod = [1,1,1] .+ brnSym .* revSwt
			lstItrMod = hlfRup .* fwdSwt .+ hlfRdw .* revSwt .+ brnSym .* fwdSwt
			# size of multiplication
			mulNum = lstItrMod .- frsItrMod
			# active device	
			if cmpInf.devMod == true
				copyto!(view(dirNumDev, :, divItr + 1), mulNum)
				copyto!(view(dirSymDev, :, divItr + 1), 
					eltype(dirSymDev).([symSgn[1] * symSgn[2],
						symSgn[1] * symSgn[3],symSgn[2] * symSgn[3]]))
				# launch kernel
				@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk mulKerDev!(
					view(dirNumDev, :, divItr + 1), 
					view(dirSymDev, :, divItr + 1),
					view(actVec, frsActVec[1]:symSgn[1]:lstActVec[1], 
					frsActVec[2]:symSgn[2]:lstActVec[2], 
					frsActVec[3]:symSgn[3]:lstActVec[3], :), 
					view(vecHld, frsActVec[1]:symSgn[1]:lstActVec[1], 
					frsActVec[2]:symSgn[2]:lstActVec[2], 
					frsActVec[3]:symSgn[3]:lstActVec[3], :), 
					view(vecMod, frsItrMod[1]:1:lstItrMod[1], 
					frsItrMod[2]:1:lstItrMod[2], 
					frsItrMod[3]:1:lstItrMod[3], :)) 
			else 
				copyto!(view(dirSymHst, :, divItr + 1), 
					eltype(dirSymHst).([symSgn[1] * symSgn[2],
						symSgn[1] * symSgn[3],symSgn[2] * symSgn[3]]))
				Base.Threads.@spawn mulKerHst!(view(dirSymHst, :, divItr + 1), 
					view(actVec, frsActVec[1]:symSgn[1]:lstActVec[1], 
					frsActVec[2]:symSgn[2]:lstActVec[2], 
					frsActVec[3]:symSgn[3]:lstActVec[3], :), 
					view(vecHld, frsActVec[1]:symSgn[1]:lstActVec[1], 
					frsActVec[2]:symSgn[2]:lstActVec[2], 
					frsActVec[3]:symSgn[3]:lstActVec[3], :), 
					view(vecMod, frsItrMod[1]:1:lstItrMod[1], 
					frsItrMod[2]:1:lstItrMod[2], 
					frsItrMod[3]:1:lstItrMod[3], :))
			end
		end
	end
	# active device	
	if cmpInf.devMod == true	
		CUDA.synchronize(CUDA.stream())
		CUDA.unsafe_free!(dirNumDev)
		CUDA.unsafe_free!(dirSymDev)
		CUDA.unsafe_free!(vecHld)
		CUDA.synchronize(CUDA.stream())
	end
	return nothing
end
#= 
host kernel for mulBrn!
=#
function mulKerHst!(dirSym::AbstractVector{<:Real}, actVec::AbstractArray{T,4}, 
	vecHld::AbstractArray{T,4}, vecMod::AbstractArray{T,4})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}

	@inbounds @threads for itr ∈ CartesianIndices(selectdim(actVec, 4, 1)) 
		# vecMod follows ii, jj, kk, ij, ik, jk storage format
		actVec[itr,1] = vecMod[itr,1] * vecHld[itr,1] + 
			dirSym[1] * vecMod[itr,4] * vecHld[itr,2] + 
			dirSym[2] * vecMod[itr,5] * vecHld[itr,3]
		# j vector direction
		actVec[itr,2] = dirSym[1] * vecMod[itr,4] * vecHld[itr,1] + 
			vecMod[itr,2] * vecHld[itr,2] + 
			dirSym[3] * vecMod[itr,6] * vecHld[itr,3]
		# k vector direction
		actVec[itr,3] = dirSym[2] * vecMod[itr,5] * vecHld[itr,1] + 
			dirSym[3] * vecMod[itr,6] * vecHld[itr,2] + 
			vecMod[itr,3] * vecHld[itr,3]
	end
end
#=
device kernel for mulBrn!
=#
function mulKerDev!(dimInf::AbstractVector{<:Integer}, 
	dirSym::AbstractVector{<:Real}, actVec::AbstractArray{T,4}, 
	vecHld::AbstractArray{T,4}, vecMod::AbstractArray{T,4})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
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
			vecHld[itrX,itrY,itrZ,1] + 
			dirSym[1] * vecMod[itrX,itrY,itrZ,4] * vecHld[itrX,itrY,itrZ,2] + 
			dirSym[2] * vecMod[itrX,itrY,itrZ,5] * vecHld[itrX,itrY,itrZ,3]
		# j vector direction
		actVec[itrX,itrY,itrZ,2] = dirSym[1] * vecMod[itrX,itrY,itrZ,4] * 
			vecHld[itrX,itrY,itrZ,1] + 
			vecMod[itrX,itrY,itrZ,2] * vecHld[itrX,itrY,itrZ,2] + 
			dirSym[3] * vecMod[itrX,itrY,itrZ,6] * vecHld[itrX,itrY,itrZ,3]
		# k vector direction
		actVec[itrX,itrY,itrZ,3] = dirSym[3] * vecMod[itrX,itrY,itrZ,6] * 
			vecHld[itrX,itrY,itrZ,1] + 
			dirSym[2] * vecMod[itrX,itrY,itrZ,5] * vecHld[itrX,itrY,itrZ,2] + 
			vecMod[itrX,itrY,itrZ,3] * vecHld[itrX,itrY,itrZ,3]
	end
	return nothing
end
#=
binary branch indexing
=#
@inline function nxtBrnId(maxLvl::Integer, lvl::Integer, bId::Integer)::Integer
	return bId + ^(2, maxLvl - (lvl + 1))
end