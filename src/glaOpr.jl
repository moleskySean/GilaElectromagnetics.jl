"""
	GreensOperator

An abstraction for the electromagnetic Green function.

# Fields
- `mem::GlaOprMem`: The data needed to compute the Green function.
"""
struct GreensOperator
	mem::GlaOprMem
end

"""
	GreensOperator(num_cells::NTuple{3, Int}, scale::NTuple{3, Rational}, origin::NTuple{3, Rational}; use_gpu::Bool=false, set_type::DataType=ComplexF64)

Construct a self Green's operator.

# Arguments
- `num_cells::NTuple{3, Int}`: The number of cells in each dimension.
- `scale::NTuple{3, Rational}`: The size of each cell in each dimension (in
units of wavelength).
- `origin::NTuple{3, Rational}`: The origin of the volume in each dimension (in
units of wavelength).
- `use_gpu::Bool=false`: Whether to use the GPU (true) or CPU (false).
- `set_type::DataType=ComplexF64`: The element type of the operator. Must be a
subtype of `Complex`.
"""
function GreensOperator(num_cells::NTuple{3, Int}, scale::NTuple{3, Rational}, origin::NTuple{3, Rational}; use_gpu::Bool=false, set_type::DataType=ComplexF64)
	if !(set_type <: Complex)
		throw(ArgumentError("set_type must be a subtype of Complex"))
	end
	options = GlaKerOpt(use_gpu)
	self_volume = GlaVol(num_cells, scale, origin)
	self_mem = GlaOprMem(options, self_volume, setType=set_type)
	return GreensOperator(self_mem)
end

"""
    GreensOperator(num_cells_source::NTuple{3, Int}, scale_source::NTuple{3, Rational}, origin_source::NTuple{3, Rational}, num_cells_target::NTuple{3, Int}, scale_target::NTuple{3, Rational}, origin_target::NTuple{3, Rational}; use_gpu::Bool=false, set_type::DataType=ComplexF64)

Construct an external Green's operator.

# Arguments
- `num_cells_source::NTuple{3, Int}`: The number of cells in each dimension of
the source volume.
- `scale_source::NTuple{3, Rational}`: The size of each cell in each dimension
of the source volume (in units of wavelength).
- `origin_source::NTuple{3, Rational}`: The origin of the source volume in each
dimension (in units of wavelength).
- `num_cells_target::NTuple{3, Int}`: The number of cells in each dimension of
the target volume.
- `scale_target::NTuple{3, Rational}`: The size of each cell in each dimension
of the target volume (in units of wavelength).
- `origin_target::NTuple{3, Rational}`: The origin of the target volume in each
dimension (in units of wavelength).
- `use_gpu::Bool=false`: Whether to use the GPU (true) or CPU (false).
- `set_type::DataType=ComplexF64`: The element type of the operator. Must be a
subtype of `Complex`.
"""
function GreensOperator(num_cells_source::NTuple{3, Int}, scale_source::NTuple{3, Rational}, origin_source::NTuple{3, Rational}, num_cells_target::NTuple{3, Int}, scale_target::NTuple{3, Rational}, origin_target::NTuple{3, Rational}; use_gpu::Bool=false, set_type::DataType=ComplexF64)
	if !(set_type <: Complex)
		throw(ArgumentError("set_type must be a subtype of Complex"))
	end
	options = GlaKerOpt(use_gpu)
	volume_source = GlaVol(num_cells_source, scale_source, origin_source)
	volume_target = GlaVol(num_cells_target, scale_target, origin_target)
	ext_mem = GlaOprMem(options, volume_target, volume_source, setType=set_type)
	return GreensOperator(ext_mem)
end

Base.eltype(op::GreensOperator) = eltype(eltype(op.mem.egoFur))
Base.size(op::GreensOperator) = (3*prod(op.mem.trgVol.cel), 3*prod(op.mem.srcVol.cel))
Base.size(op::GreensOperator, dim::Int) = (3*prod(op.mem.trgVol.cel), 3*prod(op.mem.srcVol.cel))[dim]
LinearAlgebra.issymmetric(::GreensOperator) = true
LinearAlgebra.isposdef(::GreensOperator) = true
LinearAlgebra.ishermitian(::GreensOperator) = false
LinearAlgebra.isdiag(::GreensOperator) = false

gilasize(op::GreensOperator) = ((op.mem.trgVol.cel..., 3), (op.mem.srcVol.cel..., 3))
gilasize(op::GreensOperator, dim::Int) = ((op.mem.trgVol.cel..., 3), (op.mem.srcVol.cel..., 3))[dim]

function Base.adjoint(op::GreensOperator)
	cmpInf_copy = deepcopy(op.mem.cmpInf)
	frqPhz, intOrd, adjMod, devMod, numTrd, numBlk = cmpInf_copy.frqPhz, cmpInf_copy.intOrd, cmpInf_copy.adjMod, cmpInf_copy.devMod, cmpInf_copy.numTrd, cmpInf_copy.numBlk
	adjoint_options = GlaKerOpt(frqPhz, intOrd, !adjMod, devMod, numTrd, numBlk)
	mem_copy = deepcopy(op.mem)
	trgVol, srcVol, mixInf, dimInf, egoFur, fftPlnFwd, fftPlnRev, phzInf = mem_copy.trgVol, mem_copy.srcVol, mem_copy.mixInf, mem_copy.dimInf, mem_copy.egoFur, mem_copy.fftPlnFwd, mem_copy.fftPlnRev, mem_copy.phzInf
	adj_mem = GlaOprMem(adjoint_options, srcVol, trgVol, mixInf, dimInf, egoFur, fftPlnFwd, fftPlnRev, phzInf)
	return GreensOperator(adj_mem)
end

function Base.:*(op::GreensOperator, x::AbstractArray{T, 4}) where T <: Complex
	@assert T <: eltype(op) "Input array must have the same element type as the operator. eltype(op) = $(eltype(op))"
	if op.mem.cmpInf.devMod && !(x isa CuArray)
		@warn "Input array is not a CuArray. Copying data to GPU."
		x = CuArray(x)
	end
	return egoOpr!(op.mem, deepcopy(x)) # egoOpr! is mutating, so we need to copy the input
end

function Base.:*(op::GreensOperator, x::AbstractArray{T}) where T <: Complex
	x_arr4 = reshape(x, gilasize(op, 2))
	out = op * x_arr4
	if prod(size(x)) == prod(gilasize(op, 1))
		return reshape(out, size(x))
	elseif ndims(x) == 1
		return vec(out)
	end
	return reshape(out, gilasize(op, 1))
end

function LinearAlgebra.mul!(out::AbstractArray{T}, op::GreensOperator, inp::AbstractArray{T}, α::Number, β::Number) where T <: Complex
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
isadjoint(op::GreensOperator) = op.mem.cmpInf.adjMod

"""
    isselfoperator(op::GreensOperator)

Returns true if the operator is a self Greens operator.

# Arguments
- `op::GreensOperator`: The operator to check.

# Returns
- `true` if the operator is a self Greens operator, `false` otherwise.
"""
isselfoperator(op::GreensOperator) = op.mem.srcVol == op.mem.trgVol

"""
	isexternaloperator(op::GreensOperator)

Returns true if the operator is an external Greens operator.

# Arguments
- `op::GreensOperator`: The operator to check.

# Returns
- `true` if the operator is an external Greens operator, `false` otherwise.
"""
isexternaloperator(op::GreensOperator) = !isselfoperator(op)

function Base.show(io::IO, op::GreensOperator)
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
