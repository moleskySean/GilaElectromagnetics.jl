#=
Basic linear transformation properties 
=#
Base.eltype(opr::GlaOpr) = eltype(eltype(opr.mem.egoFur))
Base.size(opr::GlaOpr) = (3 * prod(opr.mem.trgVol.cel), 
	3 * prod(opr.mem.srcVol.cel))
Base.size(opr::GlaOpr, dim::Int) = (3 * prod(opr.mem.trgVol.cel), 
	3 * prod(opr.mem.srcVol.cel))[dim]
LinearAlgebra.issymmetric(::GlaOpr) = false
LinearAlgebra.isposdef(::GlaOpr) = false
LinearAlgebra.ishermitian(::GlaOpr) = false
LinearAlgebra.isdiag(::GlaOpr) = false
#=
Create operator adjoint.
=#
function Base.adjoint(opr::GlaOpr)::GlaOpr
	cmpInfCpy = deepcopy(opr.mem.cmpInf)
	frqPhz, intOrd, adjMod, devMod, numTrd, numBlk = cmpInfCpy.frqPhz, 
		cmpInfCpy.intOrd, cmpInfCpy.adjMod, cmpInfCpy.devMod, cmpInfCpy.numTrd, 
		cmpInfCpy.numBlk
	adjOpt = GlaKerOpt(frqPhz, intOrd, !adjMod, devMod, numTrd, numBlk)
	memCpy = deepcopy(opr.mem)
	trgVol, srcVol, mixInf, dimInf, egoFur, fftPlnFwd, fftPlnRev, phzInf = 
		memCpy.trgVol, memCpy.srcVol, memCpy.mixInf, memCpy.dimInf, 
		memCpy.egoFur, memCpy.fftPlnFwd, memCpy.fftPlnRev, memCpy.phzInf
	adjMem = GlaOprMem(adjOpt, srcVol, trgVol, mixInf, dimInf, egoFur, 
		fftPlnFwd, fftPlnRev, phzInf)
	return GlaOpr(adjMem)
end
#=
Call egoOpr! via * symbol, tensor definition of input vector.
=#
function Base.:*(opr::GlaOpr, 
	innVec::AbstractArray{T, 4})::AbstractArray{T,4} where T <: Complex
	@assert T <: eltype(opr) "Input array must have the same element type as the operator. 
		eltype(opr) = $(eltype(opr))"
	if opr.mem.cmpInf.devMod && !(innVec isa CuArray)
		@warn "Input array is not a CuArray. Copying data to GPU."
		innVec = CuArray(innVec)
	end
	# egoOpr! is mutating, so we need to copy the input
	return egoOpr!(opr.mem, deepcopy(innVec)) 
end 
#=
Call egoOpr! via * symbol, flattened definition of input vector.
=#
function Base.:*(opr::GlaOpr, 
	innVec::AbstractArray{T})::AbstractArray{T} where T <: Complex
	innVecArr = reshape(innVec, glaSze(opr, 2))
	outVec = opr * innVecArr
	if prod(size(innVec)) == prod(glaSze(opr, 1))
		return reshape(outVec, size(innVec))
	elseif ndims(innVec) == 1
		return vec(outVec)
	end
	return reshape(outVec, glaSze(opr, 1))
end
#=
Call egoOpr! via mul.
=#
function LinearAlgebra.mul!(outVec::AbstractArray{T}, opr::GlaOpr, 
	innVec::AbstractArray{T}, α::Number, β::Number)::AbstractArray{T} where 
	T <: Complex
	resVec = opr * innVec
	rmul!(resVec, α)
	rmul!(outVec, β)
	outVec .+= resVec
	return outVec
end

"""
    isadjoint(opr::GlaOpr)

Returns true if the operator is the adjoint of the Greens operator.

# Arguments
- `opr::GlaOpr`: The operator to check.

# Returns
- `true` if the operator is the adjoint, `false` otherwise.
"""
isadjoint(opr::GlaOpr) = opr.mem.cmpInf.adjMod

"""
    isselfoperator(opr::GlaOpr)

Returns true if the operator is a self Greens operator.

# Arguments
- `opr::GlaOpr`: The operator to check.

# Returns
- `true` if the operator is a self Greens operator, `false` otherwise.
"""
isselfoperator(opr::GlaOpr) = opr.mem.srcVol == opr.mem.trgVol

"""
	isexternaloperator(opr::GlaOpr)

Returns true if the operator is an external Greens operator.

# Arguments
- `opr::GlaOpr`: The operator to check.

# Returns
- `true` if the operator is an external Greens operator, `false` otherwise.
"""
isexternaloperator(opr::GlaOpr) = !isselfoperator(opr)

function Base.show(io::IO, opr::GlaOpr)
	if isadjoint(opr)
		print(io, "Adjoint ")
	end
	if isselfoperator(opr)
		print(io, "Self ")
	else
		print(io, "External ")
	end
	print(io, "GlaOpr for ")
	if isselfoperator(opr)
		print(io, "a $(eltype(opr)) (" * join(opr.mem.srcVol.cel, "×") *
			  ") volume ")
		print(io, "of size (" * join(opr.mem.srcVol.scl, "×") * ")λ")
	else
		print(io, "$(eltype(opr)) (" * join(opr.mem.srcVol.cel, "×") *
			  ") -> (" * join(opr.mem.trgVol.cel, "×") * ") volumes ")
		print(io, "of sizes (" * join(opr.mem.srcVol.scl, "×") *
			  ")λ -> (" * join(opr.mem.trgVol.scl, "×") * ")λ ")
		print(io, "with separation (" * join(opr.mem.trgVol.org .- 
			opr.mem.srcVol.org, ", ") * ")λ")
	end
end
