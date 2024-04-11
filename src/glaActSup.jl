#=
restructure input vector to match partitioned Green function data
=#
function genPrtHst(mixInf::GlaExtInf, cmpInf::GlaKerOpt, parNum::Integer, 
	actVec::AbstractArray{T,4})::AbstractArray{T,5} where 
	T<:Union{ComplexF64,ComplexF32}
	# create partitioned vector
	orgVec = Array{eltype(actVec)}(undef, mixInf.srcCel..., 3, parNum) 
	# break into partitions 
	@threads for itr ∈ CartesianIndices(orgVec) 
		# copy partition data
		@inbounds orgVec[itr] = 
			actVec[CartesianIndex((Tuple(itr)[1:3] .- (1,1,1)) .* 
				mixInf.srcDiv) + CartesianIndex(1,1,1) +
				mixInf.srcPar[itr[5]], itr[4]]
	end
	return orgVec
end
#=
split into host and device functions to clarify type inference
=#
function genPrtDev(mixInf::GlaExtInf, cmpInf::GlaKerOpt, parNum::Integer, 
	actVec::AbstractArray{T,4})::AbstractArray{T,5} where 
	T<:Union{ComplexF64,ComplexF32}
	# memory for partitioned vector
	orgVec = CuArray{eltype(actVec)}(undef, mixInf.srcCel..., 3, parNum) 
	CUDA.synchronize(CUDA.stream())
	# perform partitioning
	for parItr ∈ eachindex(1:parNum) 
		for dirItr ∈ eachindex(1:3)
			@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk genPrtKer!(
				size(orgVec)[1:3]..., mixInf.srcDiv..., 
				Tuple(mixInf.srcPar[parItr])..., dirItr, parItr, orgVec, 
				actVec)
		end
	end
	# release active vector to reduce memory pressure
	CUDA.synchronize(CUDA.stream())
	CUDA.unsafe_free!(actVec)
	CUDA.synchronize(CUDA.stream())
	return orgVec
end
#=
device source vector partitioning kernel
=#
function genPrtKer!(maxX::Integer, maxY::Integer, maxZ::Integer, 
	stpX::Integer, stpY::Integer, stpZ::Integer,
	offX::Integer, offY::Integer, offZ::Integer, 
	dirItr::Integer, parItr::Integer,
	orgVec::AbstractArray{T,5}, actVec::AbstractArray{T,4})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 0x1) * blockDim().x 
	idY = threadIdx().y + (blockIdx().y - 0x1) * blockDim().y 
	idZ = threadIdx().z + (blockIdx().z - 0x1) * blockDim().z
	# copy partition data
	@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
		itrX = idX:strX:maxX
		orgVec[itrX,itrY,itrZ,dirItr,parItr] = actVec[(itrX - 1) * stpX + 
		offX + 1, (itrY - 1) * stpY + offY + 1, (itrZ -1) * stpZ + offZ + 1, 
		dirItr]
	end
	return nothing
end
#=
split a branch so that even and odd Fourier coefficients are independent
=#
function sptBrnHst!(vecSze::NTuple{3,Integer}, 
	prgVec::AbstractArray{T,5}, sptDir::Integer, phzVec::AbstractVector{T}, 
	parNum::Integer, orgVec::AbstractArray{T,5}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}

	@threads for itr ∈ CartesianIndices(orgVec) 
		@inbounds prgVec[itr] = phzVec[itr[sptDir]] * orgVec[itr]
	end
	return nothing
end
#=
host and device code split to clarify code operation
=#
function sptBrnDev!(vecSze::NTuple{3,Integer}, 
	prgVec::AbstractArray{T,5}, sptDir::Integer, phzVec::AbstractVector{T}, 
	parNum::Integer, orgVec::AbstractArray{T,5}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	for parItr ∈ eachindex(1:parNum)
		@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk sptKer!(vecSze..., 
				prgVec, sptDir, phzVec, parItr, orgVec)
	end
	CUDA.synchronize(CUDA.stream())
	return nothing
end
#=
device kernel for sptBrn!
=#
function sptKer!(maxX::Integer, maxY::Integer, maxZ::Integer,  
	prgVec::AbstractArray{T,5}, sptDir::Integer, phzVec::AbstractVector{T},
	parItr::Integer, orgVec::AbstractArray{T,5})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 0x1) * blockDim().x 
	idY = threadIdx().y + (blockIdx().y - 0x1) * blockDim().y 
	idZ = threadIdx().z + (blockIdx().z - 0x1) * blockDim().z 
	# multiplication loop selected based on phase transformation direction
	if sptDir == 1
		# x phase---phzVec[itrX]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			# x direction
			prgVec[itrX,itrY,itrZ,1,parItr] = phzVec[itrX] * 
				orgVec[itrX,itrY,itrZ,1,parItr]
			# y direction
			prgVec[itrX,itrY,itrZ,2,parItr] = phzVec[itrX] * 
				orgVec[itrX,itrY,itrZ,2,parItr]
			# z direction
			prgVec[itrX,itrY,itrZ,3,parItr] = phzVec[itrX] * 
				orgVec[itrX,itrY,itrZ,3,parItr]
		end
	elseif sptDir == 2
		# y phase---phzVec[itrY]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			# x direction
			prgVec[itrX,itrY,itrZ,1,parItr] = phzVec[itrY] * 
				orgVec[itrX,itrY,itrZ,1,parItr]
			# y direction
			prgVec[itrX,itrY,itrZ,2,parItr] = phzVec[itrY] * 
				orgVec[itrX,itrY,itrZ,2,parItr]
			# z direction
			prgVec[itrX,itrY,itrZ,3,parItr] = phzVec[itrY] * 
				orgVec[itrX,itrY,itrZ,3,parItr]
		end
	else 
		# z phase---phzVec[itrZ]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			# x direction
			prgVec[itrX,itrY,itrZ,1,parItr] = phzVec[itrZ] * 
				orgVec[itrX,itrY,itrZ,1,parItr]
			# y direction
			prgVec[itrX,itrY,itrZ,2,parItr] = phzVec[itrZ] * 
				orgVec[itrX,itrY,itrZ,2,parItr]
			# z direction
			prgVec[itrX,itrY,itrZ,3,parItr] = phzVec[itrZ] * 
				orgVec[itrX,itrY,itrZ,3,parItr]
		end
	end
	return nothing
end
#=
split branches for external Green function
=#
function sptBrnHst!(prgVecEve::AbstractArray{T,5}, prgVecOdd::AbstractArray{T,5}, 
	dirSpt::Integer, phzVec::AbstractVector{T}, mixInf::GlaExtInf,  
	parNum::Integer, orgVec::AbstractArray{T,5}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# final branch size
	brnSze = div.(mixInf.trgCel .+ mixInf.srcCel, 2)
	# current and progeny vector sizes
	curSze = size(orgVec)
	prgSze = ntuple(x -> x == dirSpt ? brnSze[x] : curSze[x], 3)
	# spill over of source vector into embedding extension
	ovrCpy = map(UnitRange, (1,1,1), mixInf.srcCel .- brnSze)
	# copy ranges
	ovrRng = ntuple(x -> x == dirSpt ? ovrCpy[x] : 1:prgSze[x], 3)
	# range for standard copy
	stdCpy = map(UnitRange, getproperty.(ovrRng, :stop) .+ (1,1,1), 
		min.(mixInf.srcCel, brnSze))
	stdRng = ntuple(x -> x == dirSpt ? stdCpy[x] : 1:prgSze[x], 3)
	# range for zero copy
	zroCpy = map(UnitRange, getproperty.(stdRng, :stop) .+ (1,1,1), brnSze)
	zroRng = ntuple(x -> x == dirSpt ? zroCpy[x] : 1:prgSze[x], 3)
	# offset for over range
	ovrOff = ntuple(x -> x == dirSpt ? brnSze[dirSpt] : 0, 3)
	# spill over indices
	splRng = CartesianIndices(ovrRng) .+ CartesianIndex(ovrOff)
	# perform split
	# copy range where source vector spills over brnSze
	@threads for crtItr ∈ CartesianIndices((ovrRng..., 1:3, 1:parNum))
		# even branch 
		@inbounds prgVecEve[crtItr] = orgVec[crtItr] + 
			orgVec[splRng[crtItr[1],crtItr[2],crtItr[3]],crtItr[4],
				crtItr[5]]
		# odd branch
		@inbounds prgVecOdd[crtItr] = phzVec[crtItr[dirSpt]] *
			(orgVec[crtItr] - 
				orgVec[splRng[crtItr[1],crtItr[2],crtItr[3]],crtItr[4],
				crtItr[5]])
	end
	# standard copy range 
	@threads for crtItr ∈ CartesianIndices((stdRng..., 1:3, 1:parNum))
		# even branch 
		@inbounds prgVecEve[crtItr] = orgVec[crtItr] 
		# odd branch
		@inbounds prgVecOdd[crtItr] = phzVec[crtItr[dirSpt]] * 
			orgVec[crtItr]
	end
	# zero copy range
	@threads for crtItr ∈ CartesianIndices((zroRng..., 1:3, 1:parNum))
		# even branch 
		@inbounds prgVecEve[crtItr] = 0.0 + 0.0 * im
		# odd branch
		@inbounds prgVecOdd[crtItr] = 0.0 + 0.0 * im
	end
	return nothing
end
#=
split host and device code to clarify functionality
=#
function sptBrnDev!(prgVecEve::AbstractArray{T,5}, prgVecOdd::AbstractArray{T,5}, 
	dirSpt::Integer, phzVec::AbstractVector{T}, mixInf::GlaExtInf,  
	parNum::Integer, orgVec::AbstractArray{T,5}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# final branch size
	brnSze = div.(mixInf.trgCel .+ mixInf.srcCel, 2)
	# current and progeny vector sizes
	curSze = size(orgVec)
	prgSze = ntuple(x -> x == dirSpt ? brnSze[x] : curSze[x], 3)
	# spill over of source vector into embedding extension
	ovrCpy = map(UnitRange, (1,1,1), mixInf.srcCel .- brnSze)
	# copy ranges
	ovrRng = ntuple(x -> x == dirSpt ? ovrCpy[x] : 1:prgSze[x], 3)
	# range for standard copy
	stdCpy = map(UnitRange, getproperty.(ovrRng, :stop) .+ (1,1,1), 
		min.(mixInf.srcCel, brnSze))
	stdRng = ntuple(x -> x == dirSpt ? stdCpy[x] : 1:prgSze[x], 3)
	# range for zero copy
	zroCpy = map(UnitRange, getproperty.(stdRng, :stop) .+ (1,1,1), brnSze)
	zroRng = ntuple(x -> x == dirSpt ? zroCpy[x] : 1:prgSze[x], 3)
	# offset for over range
	ovrOff = ntuple(x -> x == dirSpt ? brnSze[dirSpt] : 0, 3)
	# spill over indices
	splRng = CartesianIndices(ovrRng) .+ CartesianIndex(ovrOff)
	# perform split
	# copy range where source vector spills over brnSze
	if !isempty(CartesianIndices(ovrRng))
		for parItr ∈ eachindex(1:parNum) 
			for dirItr ∈ eachindex(1:3)
				@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk sptBrnOvrKer!(
					prgVecEve, prgVecOdd, getproperty.(ovrRng, :stop)..., 
					getproperty.(ovrRng, :start)..., sum(ovrOff), dirSpt, 
					dirItr, parItr, phzVec, orgVec)
			end
		end
	end
	# standard copy range 
	for parItr ∈ eachindex(1:parNum) 
		for dirItr ∈ eachindex(1:3)
			@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk sptBrnStdKer!(
			prgVecEve, prgVecOdd, getproperty.(stdRng, :stop)..., 
			getproperty.(stdRng, :start)..., dirSpt, dirItr, parItr,
			phzVec, orgVec)
		end
	end
	# zero copy range
	if !isempty(CartesianIndices(zroRng))
		for parItr ∈ eachindex(1:parNum)
			for dirItr ∈ eachindex(1:3)
				@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk sptBrnZroKer!(
					prgVecEve, prgVecOdd, getproperty.(zroRng, :stop)..., 
					getproperty.(zroRng, :start)..., dirItr, parItr, orgVec)
			end
		end
	end
	# release original vector to reduce memory pressure
	CUDA.synchronize(CUDA.stream())
	CUDA.unsafe_free!(orgVec)
	CUDA.synchronize(CUDA.stream()) 
	return nothing
end
#=
kernel for source spill over portion
=#
function sptBrnOvrKer!(prgVecEve::AbstractArray{T,5}, 
	prgVecOdd::AbstractArray{T,5}, endX::Integer, endY::Integer, endZ::Integer, 
	begX::Integer, begY::Integer, begZ::Integer, ovrOff::Integer, 
	dirSpt::Integer, dirItr::Integer, parItr::Integer, 
	phzVec::AbstractVector{T}, orgVec::AbstractArray{T,5})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (begX - 0x1) + (blockIdx().x - 0x1) * blockDim().x 
	idY = threadIdx().y + (begY - 0x1) + (blockIdx().y - 0x1) * blockDim().y 
	idZ = threadIdx().z + (begZ - 0x1) + (blockIdx().z - 0x1) * blockDim().z 
	# multiplication loop selected based on phase transformation direction
	if dirSpt == 1
		# x phase---phzVec[itrX]
		@inbounds for itrZ = idZ:strZ:endZ, itrY = idY:strY:endY, 
		itrX = idX:strX:endX 
			# even branch
			prgVecEve[itrX,itrY,itrZ,dirItr,parItr] = 
				orgVec[itrX,itrY,itrZ,dirItr,parItr] +
				orgVec[itrX + ovrOff,itrY,itrZ,dirItr,parItr]
			# odd branch
			prgVecOdd[itrX,itrY,itrZ,dirItr,parItr] = phzVec[itrX] * 
				(orgVec[itrX,itrY,itrZ,dirItr,parItr] - 
				orgVec[itrX + ovrOff,itrY,itrZ,dirItr,parItr])
		end
	elseif dirSpt == 2
		# y phase---phzVec[itrY]
		@inbounds for itrZ = idZ:strZ:endZ, itrY = idY:strY:endY, 
		itrX = idX:strX:endX 
			# even branch
			prgVecEve[itrX,itrY,itrZ,dirItr,parItr] =  
				orgVec[itrX,itrY,itrZ,dirItr,parItr] +
				orgVec[itrX,itrY + ovrOff,itrZ,dirItr,parItr]
			# odd branch
			prgVecOdd[itrX,itrY,itrZ,dirItr,parItr] = phzVec[itrY] * 
				(orgVec[itrX,itrY,itrZ,dirItr,parItr] - 
				orgVec[itrX,itrY + ovrOff,itrZ,dirItr,parItr])
		end
	else 
		# z phase---phzVec[itrZ]
		@inbounds for itrZ = idZ:strZ:endZ, itrY = idY:strY:endY, 
		itrX = idX:strX:endX 
			# even branch
			prgVecEve[itrX,itrY,itrZ,dirItr,parItr] =  
				orgVec[itrX,itrY,itrZ,dirItr,parItr] + 
				orgVec[itrX,itrY,itrZ + ovrOff,dirItr,parItr]
			# odd branch
			prgVecOdd[itrX,itrY,itrZ,dirItr,parItr] = phzVec[itrZ] * 
				(orgVec[itrX,itrY,itrZ,dirItr,parItr] - 
				orgVec[itrX,itrY,itrZ + ovrOff,dirItr,parItr])
		end
	end
	return nothing
end
#=
kernel for standard branching copy
=#
function sptBrnStdKer!(prgVecEve::AbstractArray{T,5}, 
	prgVecOdd::AbstractArray{T,5}, endX::Integer, endY::Integer, endZ::Integer, 
	begX::Integer, begY::Integer, begZ::Integer, dirSpt::Integer, 
	dirItr::Integer, parItr::Integer, phzVec::AbstractVector{T}, 
	orgVec::AbstractArray{T,5})::Nothing where T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (begX - 0x1) + (blockIdx().x - 0x1) * blockDim().x 
	idY = threadIdx().y + (begY - 0x1) + (blockIdx().y - 0x1) * blockDim().y 
	idZ = threadIdx().z + (begZ - 0x1) + (blockIdx().z - 0x1) * blockDim().z 
	# multiplication loop selected based on phase transformation direction
	if dirSpt == 1
		# x phase---phzVec[itrX]
		@inbounds for itrZ = idZ:strZ:endZ, itrY = idY:strY:endY, 
		itrX = idX:strX:endX 
			# even branch
			prgVecEve[itrX,itrY,itrZ,dirItr,parItr] = 
			orgVec[itrX,itrY,itrZ,dirItr,parItr]
			# odd branch
			prgVecOdd[itrX,itrY,itrZ,dirItr,parItr] = phzVec[itrX] * 
				orgVec[itrX,itrY,itrZ,dirItr,parItr]
		end
	elseif dirSpt == 2
		# y phase---phzVec[itrY]
		@inbounds for itrZ = idZ:strZ:endZ, itrY = idY:strY:endY, 
		itrX = idX:strX:endX 
			# even branch
			prgVecEve[itrX,itrY,itrZ,dirItr,parItr] =  
				orgVec[itrX,itrY,itrZ,dirItr,parItr]
			# odd branch
			prgVecOdd[itrX,itrY,itrZ,dirItr,parItr] = phzVec[itrY] * 
				orgVec[itrX,itrY,itrZ,dirItr,parItr]
		end
	else 
		# z phase---phzVec[itrZ]
		@inbounds for itrZ = idZ:strZ:endZ, itrY = idY:strY:endY, 
		itrX = idX:strX:endX 
			# even branch
			prgVecEve[itrX,itrY,itrZ,dirItr,parItr] =  
				orgVec[itrX,itrY,itrZ,dirItr,parItr]
			# odd branch
			prgVecOdd[itrX,itrY,itrZ,dirItr,parItr] = phzVec[itrZ] * 
				orgVec[itrX,itrY,itrZ,dirItr,parItr]
		end
	end
	return nothing
end
#=
kernel for additional embedding
=#
function sptBrnZroKer!(prgVecEve::AbstractArray{T,5}, 
	prgVecOdd::AbstractArray{T,5}, endX::Integer, endY::Integer, endZ::Integer, 
	begX::Integer, begY::Integer, begZ::Integer, dirItr::Integer, 
	parItr::Integer, orgVec::AbstractArray{T,5})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (begX - 0x1) + (blockIdx().x - 0x1) * blockDim().x 
	idY = threadIdx().y + (begY - 0x1) + (blockIdx().y - 0x1) * blockDim().y 
	idZ = threadIdx().z + (begZ - 0x1) + (blockIdx().z - 0x1) * blockDim().z 
	# zero appropriate elements
	@inbounds for itrZ = idZ:strZ:endZ, itrY = idY:strY:endY, 
		itrX = idX:strX:endX 
		# even branch
		prgVecEve[itrX,itrY,itrZ,dirItr,parItr] = 0.0 + 0.0 * im 
		# odd branch
		prgVecOdd[itrX,itrY,itrZ,dirItr,parItr] = 0.0 + 0.0 * im
	end
	return nothing
end
#=
merge two branches, eliminating unused coefficients
=#
function mrgBrnHst!(mixInf::GlaExtInf, orgVec::AbstractArray{T,5}, 
	mrgDir::Integer, parNumTrg::Integer, phzVec::AbstractVector{T}, 
	prgVec::AbstractArray{T,5}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	@threads for itr ∈ CartesianIndices(orgVec) 
		@inbounds orgVec[itr] = 0.5 * (orgVec[itr] + 
			conj(phzVec[itr[mrgDir]]) * prgVec[itr])
	end
	return nothing
end
#=
split host and device code to clarify functionality
=#
function mrgBrnDev!(mixInf::GlaExtInf, orgVec::AbstractArray{T,5},
	mrgDir::Integer, parNumTrg::Integer, phzVec::AbstractVector{T}, 
	prgVec::AbstractArray{T,5}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	for prtItr ∈ eachindex(1:parNumTrg) 
		for dirItr ∈ eachindex(1:3)
			@cuda threads = cmpInf.numTrd blocks = cmpInf.numBlk mrgKer!(
				size(orgVec)[1:3]..., orgVec, mrgDir, phzVec, dirItr, 
				prtItr, prgVec)
		end
	end
	CUDA.synchronize(CUDA.stream())
	CUDA.unsafe_free!(prgVec)
	CUDA.synchronize(CUDA.stream())
	return nothing
end
#=
device kernel for mrgBrn!
=#
function mrgKer!(maxX::Integer, maxY::Integer, maxZ::Integer, 
	orgVec::AbstractArray{T,5}, mrgDir::Integer, phzVec::AbstractVector{T}, 
	dirItr::Integer, prtItr::Integer, prgVec::AbstractArray{T,5})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 0x1) * blockDim().x 
	idY = threadIdx().y + (blockIdx().y - 0x1) * blockDim().y 
	idZ = threadIdx().z + (blockIdx().z - 0x1) * blockDim().z 
	# multiplication loop selected based on phase transformation direction
	if mrgDir == 1
		# x phase---phzVec[itrX]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			orgVec[itrX,itrY,itrZ,dirItr,prtItr] = 
				0.5 * (orgVec[itrX,itrY,itrZ,dirItr,prtItr] + 
				conj(phzVec[itrX]) * prgVec[itrX,itrY,itrZ,dirItr,prtItr]) 
		end
	elseif mrgDir == 2 
		# y phase---phzVec[itrY]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			orgVec[itrX,itrY,itrZ,dirItr,prtItr] = 
				0.5 * (orgVec[itrX,itrY,itrZ,dirItr,prtItr] + 
				conj(phzVec[itrY]) * prgVec[itrX,itrY,itrZ,dirItr,prtItr]) 
		end
	else
		# z phase---phzVec[itrZ]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			orgVec[itrX,itrY,itrZ,dirItr,prtItr] = 
				0.5 * (orgVec[itrX,itrY,itrZ,dirItr,prtItr] + 
				conj(phzVec[itrZ]) * prgVec[itrX,itrY,itrZ,dirItr,prtItr])
		end
	end
	return nothing
end
#=
generalized host merge allowing for different output size
=#
function mrgBrnHst!(mixInf::GlaExtInf, mrgVec::AbstractArray{T,5}, 
	mrgDir::Integer, parNumTrg::Integer, phzVec::AbstractVector{T}, 
	eveVec::AbstractArray{T,5}, oddVec::AbstractArray{T,5}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# size of current and merged vectors
	mrgSze = size(mrgVec)
	curSze = size(eveVec)
	# range for top half of merge vector
	topRng = ntuple(x -> x == mrgDir ? (1:min(mrgSze[x], curSze[x])) : 
		(1:mrgSze[x]), 3)
	# range for bottom half of merge vector
	botRng = ntuple(x -> x == mrgDir ? 
		(1:(mrgSze[x] - getproperty(topRng[x], :stop))) : (1:mrgSze[x]), 3)
	# offset for merge vector
	offSet = ntuple(x -> x == mrgDir ? 
		(getproperty(botRng[x], :stop) > 0 ? 
			getproperty(topRng[x], :stop) : 0) : 0, 5)
	# merge top range
	@threads for itr ∈ CartesianIndices((topRng..., 1:3, 1:mrgSze[5])) 
		@inbounds mrgVec[itr] = 0.5 * (eveVec[itr] + 
			conj(phzVec[itr[mrgDir]]) * oddVec[itr])
	end
	# check if merge vector is filled
	if getproperty(botRng[mrgDir], :stop) > 0
		@threads for itr ∈ CartesianIndices((botRng..., 1:3, 1:mrgSze[5])) 
			@inbounds mrgVec[itr + CartesianIndex(offSet)] = 0.5 * 
			(eveVec[itr] - conj(phzVec[itr[mrgDir]]) * oddVec[itr])
		end
	end
	return nothing
end
#=
device merge allowing for different output size
=#
function mrgBrnDev!(mixInf::GlaExtInf, mrgVec::AbstractArray{T,5}, 
	mrgDir::Integer, parNumTrg::Integer, phzVec::AbstractVector{T}, 
	eveVec::AbstractArray{T,5}, oddVec::AbstractArray{T,5}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# size of current and merged vectors
	mrgSze = size(mrgVec)
	curSze = size(eveVec)
	# range for top half of merge vector
	topRng = ntuple(x -> x == mrgDir ? (1:min(mrgSze[x], curSze[x])) : 
		(1:mrgSze[x]), 3)
	# range for bottom half of merge vector
	botRng = ntuple(x -> x == mrgDir ? 
		(1:(mrgSze[x] - getproperty(topRng[x], :stop))) : (1:mrgSze[x]), 3)
	# offset for merge vector
	offSet = ntuple(x -> x == mrgDir ? 
		(getproperty(botRng[x], :stop) > 0 ? 
			getproperty(topRng[x], :stop) : 0) : 0, 5)
	# merge top range
	for prtItr ∈ eachindex(1:parNumTrg)
		for dirItr ∈ eachindex(1:3)
			@cuda threads = cmpInf.numTrd blocks = cmpInf.numBlk mrgTopKer!(
				mrgVec, getproperty.(topRng, :stop)..., mrgDir, phzVec, 
				dirItr, prtItr, eveVec, oddVec)
		end
	end
	# check if merge vector is filled
	if getproperty(botRng[mrgDir], :stop) > 0
		for prtItr ∈ eachindex(1:parNumTrg)
			for dirItr ∈ eachindex(1:3)
				@cuda threads = cmpInf.numTrd blocks = cmpInf.numBlk mrgBotKer!(
					mrgVec, getproperty.(botRng, :stop)..., offSet[mrgDir], 
					mrgDir, phzVec, dirItr, prtItr, eveVec, oddVec)
			end
		end
	end
	# free liberated memory
	CUDA.synchronize(CUDA.stream())
	CUDA.unsafe_free!(eveVec)
	CUDA.unsafe_free!(oddVec)
	CUDA.synchronize(CUDA.stream())
	return nothing
end
#=
device kernel for top half of generalized mrgBrn!
=#
function mrgTopKer!(mrgVec::AbstractArray{T,5}, maxX::Integer, maxY::Integer, 
	maxZ::Integer, mrgDir::Integer, phzVec::AbstractVector{T}, dirItr::Integer, 
	prtItr::Integer, eveVec::AbstractArray{T,5}, 
	oddVec::AbstractArray{T,5})::Nothing where T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 0x1) * blockDim().x 
	idY = threadIdx().y + (blockIdx().y - 0x1) * blockDim().y 
	idZ = threadIdx().z + (blockIdx().z - 0x1) * blockDim().z 
	# multiplication loop selected based on phase transformation direction
	if mrgDir == 1
		# x phase---phzVec[itrX]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			mrgVec[itrX,itrY,itrZ,dirItr,prtItr] = 
				0.5 * (eveVec[itrX,itrY,itrZ,dirItr,prtItr] + 
				conj(phzVec[itrX]) * oddVec[itrX,itrY,itrZ,dirItr,prtItr]) 
		end
	elseif mrgDir == 2 
		# y phase---phzVec[itrY]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			mrgVec[itrX,itrY,itrZ,dirItr,prtItr] = 
				0.5 * (eveVec[itrX,itrY,itrZ,dirItr,prtItr] + 
				conj(phzVec[itrY]) * oddVec[itrX,itrY,itrZ,dirItr,prtItr]) 
		end
	else
		# z phase---phzVec[itrZ]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			mrgVec[itrX,itrY,itrZ,dirItr,prtItr] = 
				0.5 * (eveVec[itrX,itrY,itrZ,dirItr,prtItr] + 
				conj(phzVec[itrZ]) * oddVec[itrX,itrY,itrZ,dirItr,prtItr])
		end
	end
	return nothing
end
#=
device kernel for bottom half of generalized mrgBrn!
=#
function mrgBotKer!(mrgVec::AbstractArray{T,5}, maxX::Integer, maxY::Integer, 
	maxZ::Integer, mrgOff::Integer, mrgDir::Integer, phzVec::AbstractVector{T}, 
	dirItr::Integer, prtItr::Integer, eveVec::AbstractArray{T,5}, 
	oddVec::AbstractArray{T,5})::Nothing where T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 0x1) * blockDim().x 
	idY = threadIdx().y + (blockIdx().y - 0x1) * blockDim().y 
	idZ = threadIdx().z + (blockIdx().z - 0x1) * blockDim().z 
	# multiplication loop selected based on phase transformation direction
	if mrgDir == 1
		# x phase---phzVec[itrX]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			mrgVec[itrX + mrgOff,itrY,itrZ,dirItr,prtItr] = 
				0.5 * (eveVec[itrX,itrY,itrZ,dirItr,prtItr] - 
				conj(phzVec[itrX]) * oddVec[itrX,itrY,itrZ,dirItr,prtItr]) 
		end
	elseif mrgDir == 2 
		# y phase---phzVec[itrY]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			mrgVec[itrX,itrY + mrgOff,itrZ,dirItr,prtItr] = 
				0.5 * (eveVec[itrX,itrY,itrZ,dirItr,prtItr] - 
				conj(phzVec[itrY]) * oddVec[itrX,itrY,itrZ,dirItr,prtItr]) 
		end
	else
		# z phase---phzVec[itrZ]
		@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
			itrX = idX:strX:maxX 
			mrgVec[itrX,itrY,itrZ + mrgOff,dirItr,prtItr] = 
				0.5 * (eveVec[itrX,itrY,itrZ,dirItr,prtItr] - 
				conj(phzVec[itrZ]) * oddVec[itrX,itrY,itrZ,dirItr,prtItr])
		end
	end
	return nothing
end
#=
execute Hadamard product step
=#
function mulBrnHst!(mixInf::GlaExtInf, bId::Integer,
	prdVec::AbstractArray{T,4}, vecMod::AbstractArray{T,4}, 
	orgVec::AbstractArray{T,4}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# size of branch
	brnSze = (mixInf.trgCel .+ mixInf.srcCel) .÷ 2
	# unique information of Green function
	egoSze = size(vecMod)[1:3]
	# zero if branch is even in that direction, one if branch is odd
	brnSym = Bool.([mod(div(bId, ^(2, 3 - k)), 2) for k ∈ 1:3])
	# declare memory 
	dirSymHst = Array{Float32}(undef, 3, 8)
	# selection ranges for Green function and vector
	modRng = Array{StepRange}(undef, 3, 8)
	vecRng = Array{StepRange}(undef, 3, 8)
	# sign symmetry for Green function multiplication
	symSgn = Array{Float32}(undef, 3, 8)
	# perform Hadamard---forward and reverse operation in each direction
	@sync begin
		for divItr ∈ 0:7
			# one for reverse direction, zero for forward
			revSwt = [mod(div(divItr, ^(2, k - 1)), 2) for k ∈ 1:3]
			# set copy ranges
			for dirItr ∈ 1:3
				# switch between full and partial stored information cases
				if egoSze[dirItr] == brnSze[dirItr]
					if revSwt[dirItr] == 0
						# Green function range
						modRng[dirItr,divItr + 1] = 1:1:brnSze[dirItr]
						vecRng[dirItr,divItr + 1] = 1:1:brnSze[dirItr]
						symSgn[dirItr, divItr + 1] = 1.0
					else
						modRng[dirItr,divItr + 1] = 1:1:0
						vecRng[dirItr,divItr + 1] = 1:1:0
						symSgn[dirItr, divItr + 1] = 0.0
					end
				# partial information case
				else
					# switch between even and odd branch cases
					if brnSym[dirItr] == 0 
						# switch between source and target portions 
						if revSwt[dirItr] == 0
							begInd = 1
							endInd = Integer.(ceil.(mixInf.trgCel[dirItr] / 
								2)) + iseven(mixInf.trgCel[dirItr])
						# source coefficients
						else
							begInd = 2
							endInd = Integer.(floor.(mixInf.srcCel[dirItr] / 
								2)) + 1 - iseven(mixInf.trgCel[dirItr])
						end
					# odd coefficient branch
					else
						begInd = 1 
						# target coefficients
						if revSwt[dirItr] == 0
							endInd = Integer.(ceil.(mixInf.trgCel[dirItr] / 2)) 
						# source coefficients
						else
							endInd = Integer.(floor.(mixInf.srcCel[dirItr] / 2)) 
						end
					end
					# Green function range
					modRng[dirItr,divItr + 1] = begInd:1:endInd
					# vector range
					if revSwt[dirItr] == 0
						vecRng[dirItr,divItr + 1] = 1:1:(1 + endInd - begInd)
					else
						vecRng[dirItr,divItr + 1] = 
							brnSze[dirItr]:-1:(brnSze[dirItr] - endInd + begInd)
					end
					# sign symmetry for Green function multiplication
					symSgn[dirItr, divItr + 1] = 1.0 - 2.0 * revSwt[dirItr]
				end
			end
			# check that range is not empty
			if !isempty(CartesianIndices(tuple(modRng[:,divItr + 1]...)))
				copyto!(view(dirSymHst, :, divItr + 1), 
					eltype(dirSymHst).([symSgn[1,divItr + 1] * 
					symSgn[2,divItr + 1], symSgn[1,divItr + 1] * 
					symSgn[3,divItr + 1], symSgn[2,divItr + 1] * 
					symSgn[3,divItr + 1]]))
				Base.Threads.@spawn mulKerHst!(
					getproperty.(modRng[:,divItr + 1], :stop) .- 
					getproperty.(modRng[:,divItr + 1], :start) .+ 1, 
					view(dirSymHst, :, divItr + 
					1), view(prdVec, vecRng[:,divItr + 1]..., :),
					view(vecMod, modRng[:,divItr + 1]..., :),
					view(orgVec, vecRng[:,divItr + 1]..., :))
			end
		end
	end
	
	return nothing
end
#=
execute Hadamard product step on device
=#
function mulBrnDev!(mixInf::GlaExtInf, bId::Integer,
	prdVec::AbstractArray{T,4}, vecMod::AbstractArray{T,4}, 
	orgVec::AbstractArray{T,4}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# size of branch
	brnSze = (mixInf.trgCel .+ mixInf.srcCel) .÷ 2
	# unique information of Green function
	egoSze = size(vecMod)[1:3]
	# zero if branch is even in that direction, one if branch is odd
	brnSym = Bool.([mod(div(bId, ^(2, 3 - k)), 2) for k ∈ 1:3])
	# declare memory 
	dimInfDev = CuArray{Int32}(undef, 3, 8)
	dirSymDev = CuArray{Float32}(undef, 3, 8)
	# selection ranges for Green function and vector
	modRng = Array{StepRange}(undef, 3, 8)
	vecRng = Array{StepRange}(undef, 3, 8)
	# sign symmetry for Green function multiplication
	symSgn = Array{Float32}(undef, 3, 8)
	# perform Hadamard---forward and reverse operation in each direction
	@sync begin
		for divItr ∈ 0:7
			# one for reverse direction, zero for forward
			revSwt = [mod(div(divItr, ^(2, k - 1)), 2) for k ∈ 1:3]
			# set copy ranges
			for dirItr ∈ 1:3
				# switch between full and partial stored information cases
				if egoSze[dirItr] == brnSze[dirItr]
					if revSwt[dirItr] == 0
						# Green function range
						modRng[dirItr,divItr + 1] = 1:1:brnSze[dirItr]
						vecRng[dirItr,divItr + 1] = 1:1:brnSze[dirItr]
						symSgn[dirItr, divItr + 1] = 1.0
					else
						modRng[dirItr,divItr + 1] = 1:1:0
						vecRng[dirItr,divItr + 1] = 1:1:0
						symSgn[dirItr, divItr + 1] = 0.0
					end
				# partial information case
				else
					# switch between even and odd branch cases
					if brnSym[dirItr] == 0 
						# switch between source and target portions 
						if revSwt[dirItr] == 0
							begInd = 1
							endInd = Integer.(ceil.(mixInf.trgCel[dirItr] / 
								2)) + iseven(mixInf.trgCel[dirItr])
						# source coefficients
						else
							begInd = 2
							endInd = Integer.(floor.(mixInf.srcCel[dirItr] / 
								2)) + 1 - iseven(mixInf.trgCel[dirItr])
						end
					# odd coefficient branch
					else
						begInd = 1 
						# target coefficients
						if revSwt[dirItr] == 0
							endInd = Integer.(ceil.(mixInf.trgCel[dirItr] / 2)) 
						# source coefficients
						else
							endInd = Integer.(floor.(mixInf.srcCel[dirItr] / 2)) 
						end
					end
					# Green function range
					modRng[dirItr,divItr + 1] = begInd:1:endInd
					# vector range
					if revSwt[dirItr] == 0
						vecRng[dirItr,divItr + 1] = 1:1:(1 + endInd - begInd)
					else
						vecRng[dirItr,divItr + 1] = 
							brnSze[dirItr]:-1:(brnSze[dirItr] - endInd + begInd)
					end
					# sign symmetry for Green function multiplication
					symSgn[dirItr, divItr + 1] = 1.0 - 2.0 * revSwt[dirItr]
				end
			end
			# check that range is not empty
			if !isempty(CartesianIndices(tuple(modRng[:,divItr + 1]...)))
				# number of elements
				copyto!(view(dimInfDev, :, divItr + 1), 
					getproperty.(modRng[:, divItr + 1], :stop) .- 
					getproperty.(modRng[:, divItr + 1], :start) .+ 1)
				# symmetry information
				copyto!(view(dirSymDev, :, divItr + 1), 
					eltype(dirSymDev).([symSgn[1,divItr + 1] * 
						symSgn[2,divItr + 1], symSgn[1,divItr + 1] * 
						symSgn[3,divItr + 1], symSgn[2,divItr + 1] * 
						symSgn[3,divItr + 1]]))
				# launch kernel
				@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk mulKerDev!(
					view(dimInfDev, :, divItr + 1), 
					view(dirSymDev, :, divItr + 1),
					view(prdVec, vecRng[:,divItr + 1]..., :),
					view(vecMod, modRng[:,divItr + 1]..., :),
					view(orgVec, vecRng[:,divItr + 1]..., :)) 
			end
		end
	end
	# synchronize
	CUDA.synchronize(CUDA.stream())
	CUDA.unsafe_free!(dimInfDev)
	CUDA.unsafe_free!(dirSymDev)
	CUDA.synchronize(CUDA.stream())
	return nothing
end
#= 
host kernel for mulBrn!
=#
function mulKerHst!(dimInf::AbstractArray{<:Integer}, 
	dirSym::AbstractArray{<:Real}, prdVec::AbstractArray{T,4}, 
	vecMod::AbstractArray{T,4}, orgVec::AbstractArray{T,4},)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}

	@threads for itr ∈ CartesianIndices((1:dimInf[1], 1:dimInf[2], 
		1:dimInf[3])) 
		# vecMod follows ii, jj, kk, ij, ik, jk storage format
		@inbounds prdVec[itr,1] += vecMod[itr,1] * orgVec[itr,1] + 
			dirSym[1] * vecMod[itr,4] * orgVec[itr,2] + 
			dirSym[2] * vecMod[itr,5] * orgVec[itr,3]
		# j vector direction
		@inbounds prdVec[itr,2] += dirSym[1] * vecMod[itr,4] * orgVec[itr,1] + 
			vecMod[itr,2] * orgVec[itr,2] + 
			dirSym[3] * vecMod[itr,6] * orgVec[itr,3]
		# k vector direction
		@inbounds prdVec[itr,3] += dirSym[2] * vecMod[itr,5] * orgVec[itr,1] + 
			dirSym[3] * vecMod[itr,6] * orgVec[itr,2] + 
			vecMod[itr,3] * orgVec[itr,3]
	end
end
#=
device kernel for mulBrn!
=#
function mulKerDev!(dimInf::AbstractVector{<:Integer}, 
	dirSym::AbstractVector{<:Real}, prdVec::AbstractArray{T,4}, 
	vecMod::AbstractArray{T,4}, orgVec::AbstractArray{T,4})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 0x1) * blockDim().x 
	idY = threadIdx().y + (blockIdx().y - 0x1) * blockDim().y 
	idZ = threadIdx().z + (blockIdx().z - 0x1) * blockDim().z 
	# multiplication loops
	@inbounds for itrZ = idZ:strZ:dimInf[3], itrY = idY:strY:dimInf[2], 
		itrX = idX:strX:dimInf[1] 
		# vecMod follows ii, jj, kk, ij, ik, jk storage format
		prdVec[itrX,itrY,itrZ,1] += vecMod[itrX,itrY,itrZ,1] * 
			orgVec[itrX,itrY,itrZ,1] + 
			dirSym[1] * vecMod[itrX,itrY,itrZ,4] * orgVec[itrX,itrY,itrZ,2] + 
			dirSym[2] * vecMod[itrX,itrY,itrZ,5] * orgVec[itrX,itrY,itrZ,3]
		# j vector direction
		prdVec[itrX,itrY,itrZ,2] += dirSym[1] * vecMod[itrX,itrY,itrZ,4] * 
			orgVec[itrX,itrY,itrZ,1] + 
			vecMod[itrX,itrY,itrZ,2] * orgVec[itrX,itrY,itrZ,2] + 
			dirSym[3] * vecMod[itrX,itrY,itrZ,6] * orgVec[itrX,itrY,itrZ,3]
		# k vector direction
		prdVec[itrX,itrY,itrZ,3] += dirSym[2] * vecMod[itrX,itrY,itrZ,5] * 
			orgVec[itrX,itrY,itrZ,1] + 
			dirSym[3] * vecMod[itrX,itrY,itrZ,6] * orgVec[itrX,itrY,itrZ,2] + 
			vecMod[itrX,itrY,itrZ,3] * orgVec[itrX,itrY,itrZ,3]
	end
	return nothing
end
#=
merge partitions and return output vector 
=#
function mrgPrtHst(mixInf::GlaExtInf, cmpInf::GlaKerOpt, parNum::Integer, 
	prtVec::AbstractArray{T,5})::AbstractArray{T,4} where 
	T<:Union{ComplexF64,ComplexF32}
	# restructure partitioned vector
	mrgVec = Array{eltype(prtVec)}(undef, (mixInf.trgDiv .* 
		mixInf.trgCel)..., 3) 
	# break into partitions 
	@threads for itr ∈ CartesianIndices(prtVec) 
		# copy partition data
		@inbounds mrgVec[CartesianIndex((Tuple(itr)[1:3] .- (1,1,1)) .* 
				mixInf.trgDiv) + CartesianIndex(1,1,1) + 
				mixInf.trgPar[itr[5]], itr[4]] = prtVec[itr]
	end
	return mrgVec
end
#=
device code separated to clarify functionality and type inference
=#
function mrgPrtDev(mixInf::GlaExtInf, cmpInf::GlaKerOpt, parNum::Integer, 
	prtVec::AbstractArray{T,5})::AbstractArray{T,4} where 
	T<:Union{ComplexF64,ComplexF32}
	# restructure partitioned vector
	mrgVec = CuArray{eltype(prtVec)}(undef, 
		(mixInf.trgDiv .* mixInf.trgCel)..., 3) 
	CUDA.synchronize(CUDA.stream())
	# perform partitioning
	for parItr ∈ eachindex(1:parNum)
		for dirItr ∈ eachindex(1:3)
			@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk mrgPrtKer!(
				size(prtVec)[1:3]..., mixInf.trgDiv..., 
				Tuple(mixInf.trgPar[parItr])..., dirItr, parItr, mrgVec, 
				prtVec)
		end
	end
	# release active vector to reduce memory pressure
	CUDA.synchronize(CUDA.stream())
	CUDA.unsafe_free!(prtVec)
	CUDA.synchronize(CUDA.stream())
	return mrgVec
end
#=
device kernel to merge target vector partitions
=#
function mrgPrtKer!(maxX::Integer, maxY::Integer, maxZ::Integer, 
	stpX::Integer, stpY::Integer, stpZ::Integer,
	offX::Integer, offY::Integer, offZ::Integer, 
	dirItr::Integer, parItr::Integer,
	mrgVec::AbstractArray{T,4}, prtVec::AbstractArray{T,5})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# grid strides
	strX = gridDim().x * blockDim().x
	strY = gridDim().y * blockDim().y
	strZ = gridDim().z * blockDim().z
	# thread indices
	idX = threadIdx().x + (blockIdx().x - 0x1) * blockDim().x 
	idY = threadIdx().y + (blockIdx().y - 0x1) * blockDim().y 
	idZ = threadIdx().z + (blockIdx().z - 0x1) * blockDim().z
	# copy partition data
	@inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, 
		itrX = idX:strX:maxX
		mrgVec[(itrX - 1) * stpX + offX + 1, (itrY - 1) * stpY + offY + 1, 
			(itrZ - 1) * stpZ + offZ + 1, dirItr] = 
			prtVec[itrX,itrY,itrZ,dirItr,parItr]
	end
	return nothing
end
#=
binary branch indexing
=#
@inline function nxtBrnId(maxLvl::Integer, lvl::Integer, bId::Integer)::Integer
	return bId + ^(2, maxLvl - (lvl + 1))
end