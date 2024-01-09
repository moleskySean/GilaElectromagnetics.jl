# depth of level branching
const maxLvl = 3
#= 
temporary memory structure for testing new transformation ideas
=#
struct TrnVecMem
	# computation information
	cmpInf::GlaKerOpt
	# dimension information for vector
	dimInfC::AbstractVector{<:Integer} 
	dimInfD::AbstractVector{<:Integer} 
	# vector that will be acted on 
	actVec::AbstractVector{<:AbstractArray{T,3}} where 
	T <: Union{ComplexF64,ComplexF32}
	# Fourier transform plans
	fftPlnFwd::AbstractArray{<:AbstractFFTs.Plan}
	fftPlnRev::AbstractArray{<:AbstractFFTs.ScaledPlan}
	# internal phase information for implementation of operator action
	phzInf::AbstractArray{<:AbstractArray{T}} where 
	T<:Union{ComplexF64,ComplexF32}
end
#= 
branching function for performing Fourier transform embedding on sub-body
=#
function egoExp!(divMem::TrnVecMem, lvl::Integer, bId::Integer)::Nothing 

	if lvl > 0
		# forward FFT
		divMem.fftPlnFwd[lvl] * divMem.actVec[bId + 1]
		# GPU mode
		if divMem.cmpInf.dev == true
			CUDA.synchronize(CUDA.stream())
		end
	end 
	# branch until depth of block structure
	if lvl < length(divMem.dimInfC)	
		# split branch, includes phase operation and stream sync
		divBrn!(divMem.dimInfD, lvl, bId, divMem.phzInf[lvl + 1], 
			divMem.actVec, divMem.cmpInf)
		# execute split branches 
		# !asynchronous GPU causes mysterious errors + minimal speed up!
		if divMem.cmpInf.dev == true
			# origin branch
			egoExp!(divMem, lvl + 1, bId)	
			# phase modified branch
			egoExp!(divMem, lvl + 1, 
				nxtBrnId(length(divMem.dimInfC), lvl, bId))
		# !asynchronous CPU is fine + some speed up!
		else
			@sync begin
				# origin branch
				Base.Threads.@spawn egoExp!(divMem, lvl + 1, bId)
				# phase modified branch
				Base.Threads.@spawn egoExp!(divMem, lvl + 1, 
					nxtBrnId(length(divMem.dimInfC), lvl, bId))
			end
		end
	# terminate task and return control to previous level 
	else	
		if divMem.cmpInf.dev == true
			CUDA.synchronize(CUDA.stream())
		end
		return nothing
	end
	# terminate task and return control to previous level 
	if divMem.cmpInf.dev == true
		CUDA.synchronize(CUDA.stream())
	end
	return nothing
end
#= 
perform branch division---copy and phase modification
=#
function divBrn!(dimInf::AbstractVector{<:Integer}, 
	lvl::Integer, bId::Integer, phzVec::AbstractVector{T}, 
	actVec::AbstractVector{<:AbstractArray{T,3}}, 
	cmpInf::GlaKerOpt=GlaKerOpt(false))::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	phzDir = lvl + 1
	# number of next branch
	nId = nxtBrnId(maxLvl, lvl, bId)
	# creation of branch copy
	if cmpInf.dev == true
		@cuda threads=cmpInf.numTrd blocks=cmpInf.numBlk sptKer!(dimInf, nId, 
		bId, phzDir, phzVec, actVec)
		CUDA.synchronize(CUDA.stream())
	else 
		# @inbounds @threads 
		for itr ∈ CartesianIndices(actVec[bId + 1]) 
			actVec[nId + 1][itr] = phzVec[itr[phzDir]] * actVec[bId + 1][itr]
		end
	end
	return nothing
end
#=
device kernel for divBrn!
=#
function divKer!(dimInf::AbstractVector{<:Integer}, nId::Integer, bId::Integer, 
	phzDir::Integer, phzVec::AbstractVector{T}, 
	actVec::AbstractVector{<:AbstractArray{T,3}})::Nothing where 
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
			actVec[nId + 1][itrX,itrY,itrZ] = phzVec[itrX] * 
				actVec[bId + 1][itrX,itrY,itrZ]
		end
	elseif phzDir == 2
		# y phase---phzVec[itrY]
		@inbounds for itrZ = idZ:strZ:dimInf[3], 
			itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
			actVec[nId + 1][itrX,itrY,itrZ] = phzVec[itrY] * 
				actVec[bId + 1][itrX,itrY,itrZ]
		end
	else 
		# z phase---phzVec[itrZ]
		@inbounds for itrZ = idZ:strZ:dimInf[3], 
			itrY = idY:strY:dimInf[2], itrX = idX:strX:dimInf[1] 
			# x direction
			actVec[nId + 1][itrX,itrY,itrZ] = phzVec[itrZ] * 
				actVec[bId + 1][itrX,itrY,itrZ]
		end
	end
	return nothing
end
#=
binary branch indexing
=#
@inline function nxtBrnId(maxLvl::Integer, lvl::Integer, bId::Integer)::Integer
	return bId + ^(2, maxLvl - (lvl + 1))
end
#=
bundle separated Fourier transforms related to a single body. 
=#
# egoBun!(bunNum::NTuple{3,<:Integer})::Nothing
# end