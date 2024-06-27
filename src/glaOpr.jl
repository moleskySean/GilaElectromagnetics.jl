"""
	GlaOpr

An abstraction for the electromagnetic Green function.

# Fields
- `mem::GlaOprMem`: The data needed to compute the Green function.
"""
struct GlaOpr
	mem::GlaOprMem
end

"""
	GlaOpr(num_cells::NTuple{3, Int}, scale::NTuple{3, Rational}, origin::NTuple{3, Rational}; use_gpu::Bool=false, set_type::DataType=ComplexF64)

Construct a self Green's operator.

# Arguments
- `cel::NTuple{3, Int}`: The number of cells in each dimension.
- `scl::NTuple{3, Rational}`: The size of each cell in each dimension (in units
of wavelength).
- `org::NTuple{3, Rational}=(0//1, 0//1, 0//1)`: The origin of the volume in each dimension (in
units of wavelength).
- `useGpu::Bool=false`: Whether to use the GPU (true) or CPU (false).
- `setType::DataType=ComplexF64`: The element type of the operator. Must be a
subtype of `Complex`.
"""
function GlaOpr(cel::NTuple{3, Int}, scl::NTuple{3, Rational}, org::NTuple{3, Rational}=(0//1, 0//1, 0//1); useGpu::Bool=false, setType::DataType=ComplexF64)
	if !(setType <: Complex)
		throw(ArgumentError("set_type must be a subtype of Complex"))
	end
	options = GlaKerOpt(useGpu)
	self_volume = GlaVol(cel, scl, org)
	self_mem = GlaOprMem(options, self_volume, setType=setType)
	return GlaOpr(self_mem)
end

"""
    GlaOpr(num_cells_source::NTuple{3, Int}, scale_source::NTuple{3, Rational}, origin_source::NTuple{3, Rational}, num_cells_target::NTuple{3, Int}, scale_target::NTuple{3, Rational}, origin_target::NTuple{3, Rational}; use_gpu::Bool=false, set_type::DataType=ComplexF64)

Construct an external Green's operator.

# Arguments
- `celSrc::NTuple{3, Int}`: The number of cells in each dimension of the source
volume.
- `sclSrc::NTuple{3, Rational}`: The size of each cell in each dimension of the
source volume (in units of wavelength).
- `orgSrc::NTuple{3, Rational}`: The origin of the source volume in each
dimension (in units of wavelength).
- `celTrg::NTuple{3, Int}`: The number of cells in each dimension of the target
volume.
- `sclTrg::NTuple{3, Rational}`: The size of each cell in each dimension of the
target volume (in units of wavelength).
- `orgTrg::NTuple{3, Rational}`: The origin of the target volume in each
dimension (in units of wavelength).
- `useGpu::Bool=false`: Whether to use the GPU (true) or CPU (false).
- `setType::DataType=ComplexF64`: The element type of the operator. Must be a
subtype of `Complex`.
"""
function GlaOpr(celSrc::NTuple{3, Int}, sclSrc::NTuple{3, Rational}, orgSrc::NTuple{3, Rational}, celTrg::NTuple{3, Int}, sclTrg::NTuple{3, Rational}, orgTrg::NTuple{3, Rational}; useGpu::Bool=false, setType::DataType=ComplexF64)
	if !(setType <: Complex)
		throw(ArgumentError("set_type must be a subtype of Complex"))
	end
	opt = GlaKerOpt(useGpu)
	volSrc = GlaVol(celSrc, sclSrc, orgSrc)
	volTrg = GlaVol(celTrg, sclTrg, orgTrg)
	extMem = GlaOprMem(opt, volTrg, volSrc, setType=setType)
	return GlaOpr(extMem)
end

Base.eltype(op::GlaOpr) = eltype(eltype(op.mem.egoFur))
Base.size(op::GlaOpr) = (3*prod(op.mem.trgVol.cel), 3*prod(op.mem.srcVol.cel))
Base.size(op::GlaOpr, dim::Int) = (3*prod(op.mem.trgVol.cel), 3*prod(op.mem.srcVol.cel))[dim]
LinearAlgebra.issymmetric(::GlaOpr) = true
LinearAlgebra.isposdef(::GlaOpr) = true
LinearAlgebra.ishermitian(::GlaOpr) = false
LinearAlgebra.isdiag(::GlaOpr) = false

"""
    glaSize(op::GreensOperator)

Returns the size of the input/output arrays for a GreensOperator in tensor form.

# Arguments
- `op::GreensOperator`: The operator to check.

# Returns
- A tuple of the sizes of the input and output arrays in tensor form.
"""
glaSize(op::GlaOpr) = ((op.mem.trgVol.cel..., 3), (op.mem.srcVol.cel..., 3))

"""
	glaSize(op::GreensOperator, dim::Int)

Returns the size of the input/output arrays for a GreensOperator in tensor form.

# Arguments
- `op::GreensOperator`: The operator to check.
- `dim::Int`: The length of the dimension to check.

# Returns
- The size of the input/output arrays for a GreensOperator in tensor form.
"""
glaSize(op::GlaOpr, dim::Int) = ((op.mem.trgVol.cel..., 3), (op.mem.srcVol.cel..., 3))[dim]

function Base.adjoint(op::GlaOpr)
	cmpInfCpy = deepcopy(op.mem.cmpInf)
	frqPhz, intOrd, adjMod, devMod, numTrd, numBlk = cmpInfCpy.frqPhz, cmpInfCpy.intOrd, cmpInfCpy.adjMod, cmpInfCpy.devMod, cmpInfCpy.numTrd, cmpInfCpy.numBlk
	adjOpt = GlaKerOpt(frqPhz, intOrd, !adjMod, devMod, numTrd, numBlk)
	memCpy = deepcopy(op.mem)
	trgVol, srcVol, mixInf, dimInf, egoFur, fftPlnFwd, fftPlnRev, phzInf = memCpy.trgVol, memCpy.srcVol, memCpy.mixInf, memCpy.dimInf, memCpy.egoFur, memCpy.fftPlnFwd, memCpy.fftPlnRev, memCpy.phzInf
	adjMem = GlaOprMem(adjOpt, srcVol, trgVol, mixInf, dimInf, egoFur, fftPlnFwd, fftPlnRev, phzInf)
	return GlaOpr(adjMem)
end

function Base.:*(op::GlaOpr, x::AbstractArray{T, 4}) where T <: Complex
	@assert T <: eltype(op) "Input array must have the same element type as the operator. eltype(op) = $(eltype(op))"
	if op.mem.cmpInf.devMod && !(x isa CuArray)
		@warn "Input array is not a CuArray. Copying data to GPU."
		x = CuArray(x)
	end
	return egoOpr!(op.mem, deepcopy(x)) # egoOpr! is mutating, so we need to copy the input
end

function Base.:*(op::GlaOpr, x::AbstractArray{T}) where T <: Complex
	xArr4 = reshape(x, glaSize(op, 2))
	out = op * xArr4
	if prod(size(x)) == prod(glaSize(op, 1))
		return reshape(out, size(x))
	elseif ndims(x) == 1
		return vec(out)
	end
	return reshape(out, glaSize(op, 1))
end

function LinearAlgebra.mul!(out::AbstractArray{T}, op::GlaOpr, inp::AbstractArray{T}, α::Number, β::Number) where T <: Complex
	res = op * inp
	rmul!(res, α)
	rmul!(out, β)
	out .+= res
	return out
end

"""
    isadjoint(op::GreensOperator)

Returns true if the operator is the adjoint of the Greens operator.

# Arguments
- `op::GreensOperator`: The operator to check.

# Returns
- `true` if the operator is the adjoint of the Greens operator, `false` otherwise.
"""
isadjoint(op::GlaOpr) = op.mem.cmpInf.adjMod

"""
    isselfoperator(op::GreensOperator)

Returns true if the operator is a self Greens operator.

# Arguments
- `op::GreensOperator`: The operator to check.

# Returns
- `true` if the operator is a self Greens operator, `false` otherwise.
"""
isselfoperator(op::GlaOpr) = op.mem.srcVol == op.mem.trgVol

"""
	isexternaloperator(op::GreensOperator)

Returns true if the operator is an external Greens operator.

# Arguments
- `op::GreensOperator`: The operator to check.

# Returns
- `true` if the operator is an external Greens operator, `false` otherwise.
"""
isexternaloperator(op::GlaOpr) = !isselfoperator(op)

function Base.show(io::IO, op::GlaOpr)
	if isadjoint(op)
		print(io, "Adjoint ")
	end
	if isselfoperator(op)
		print(io, "Self ")
	else
		print(io, "External ")
	end
	print(io, "GreensOperator for ")
	if isselfoperator(op)
		print(io, "a $(eltype(op)) (" * join(op.mem.srcVol.cel, "×") * ") volume ")
		print(io, "of size (" * join(op.mem.srcVol.scl, "×") * ")λ")
	else
		print(io, "$(eltype(op)) (" * join(op.mem.srcVol.cel, "×") * ") -> (" * join(op.mem.trgVol.cel, "×") * ") volumes ")
		print(io, "of sizes (" * join(op.mem.srcVol.scl, "×") * ")λ -> (" * join(op.mem.trgVol.scl, "×") * ")λ ")
		print(io, "with separation (" * join(op.mem.trgVol.org .- op.mem.srcVol.org, ", ") * ")λ")
	end
end
