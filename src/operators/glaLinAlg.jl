glaSze(opr::AbstractGlaOpr, dim::Int) = glaSze(opr)[dim]

Base.eltype(::AbstractGlaOpr{T}) where T<:AbstractFloat = Complex{T}
Base.eltype(::Type{<:AbstractGlaOpr{T}}) where T<:AbstractFloat = Complex{T}
Base.size(opr::AbstractGlaOpr) = prod.(glaSze(opr))
Base.size(opr::AbstractGlaOpr, i::Int) = prod(glaSze(opr, i))

# A work array of the requested shape on the device the operator computes with
Base.similar(opr::AbstractGlaOpr, ::Type{T}, dims::Tuple{Vararg{Int}}) where T = allocate(get_backend(opr), T, dims)
Base.similar(opr::AbstractGlaOpr, dims::Tuple{Vararg{Int}}) = similar(opr, eltype(opr), dims)

#= The shape methods an AbstractArray would have supplied. Densification is asked
for by name through Matrix, never as a side effect of a fallback. =#
Base.axes(opr::AbstractGlaOpr) = map(Base.OneTo, size(opr))
Base.axes(opr::AbstractGlaOpr, dim::Int) = axes(opr)[dim]
Base.CartesianIndices(opr::AbstractGlaOpr) = CartesianIndices(axes(opr))
Base.Matrix(opr::AbstractGlaOpr) = opr[:, :]

# Indexing functions
Base.getindex(opr::AbstractGlaOpr, i::Integer) = getindex(opr, CartesianIndices(opr)[i])
Base.getindex(opr::AbstractGlaOpr, i::CartesianIndex) = getindex(opr, i.I...)
const GlaIdx = Union{Integer, AbstractUnitRange{<:Integer}, AbstractVector{<:Integer}, Colon}
function Base.getindex(opr::AbstractGlaOpr, row::GlaIdx, col::GlaIdx)
    T = eltype(opr)
    numRow, numCol = size(opr)
    row = row === Colon() ? (1:numRow) : row
    col = col === Colon() ? (1:numCol) : col

    if row isa Integer && col isa Integer
        @warn """Scalar indexing is not recommended. Invocation of getindex resulted in scalar
        indexing of an AbstractGlaOpr. This is typically caused by calling an iterating
        implementation of a method. Single element access ($row, $col) does a costly full
        matrix-vector product."""
        e = zeros(T, size(opr, 2))
        e[col] = one(T)
        return Array(opr * adapt(get_backend(opr), e))[row]
    end

    rowInd = row isa Integer ? [row] : collect(row)
    colInd = col isa Integer ? [col] : collect(col)

    # Choose the cheapest way to compute the result (adjoint or forward)
    if length(rowInd) <= length(colInd)
        # Forward batched mat–vec
        # build a mini‐identity with 1‑hots in the input slots
        idt = zeros(T, numCol, length(colInd))
        for (j, idx) in enumerate(colInd)
            idt[idx, j] = one(T)
        end
        out = opr * adapt(get_backend(opr), idt)
        res = out[rowInd, :]
    else
        # Adjoint batched vec–mat
        # mini‑identity in the output slots
        idt = zeros(T, numRow, length(rowInd))
        for (i, idx) in enumerate(rowInd)
            idt[idx, i] = one(T)
        end
        outDag = _witAdj(adjOpr -> adjOpr * adapt(get_backend(adjOpr), idt), opr)
        res = copy(outDag[colInd, :]')
    end
    # An integer index drops its dimension, as for any AbstractMatrix
    (row isa Integer || col isa Integer) && return vec(res)
    return res
end
Base.setindex!(::AbstractGlaOpr, _, __...) = throw(ArgumentError("setindex! is not supported for AbstractGlaOpr"))

"""
    mulAct!(opr::AbstractGlaOpr{T}, act::AbstractVector{Complex{T}})

Apply `opr` to the flat vector `act`, on the operator's device, consuming `act`.
Returns a new flat vector. Internal; callers use `*` or `mul!`.
"""
function mulAct! end

# The buffer the primitive consumes, on the device the operator computes with
function _devCpy(opr::AbstractGlaOpr, inp::AbstractVector{<:Complex})
    dev = get_backend(opr)
    if !(get_backend(inp) isa typeof(dev))
        @warn "Input array is not on the operator's $(nameof(typeof(dev))) backend. Copying data there."
        return adapt(dev, Array(inp)) # The conversion is itself the defensive copy
    end
    return copy(inp)
end

"""
    mul!(out, opr::AbstractGlaOpr, inp, α::Number, β::Number)

Write `α * (opr * inp) + β * out` into `out`, leaving `inp` untouched.

Both arrays are flat vectors or, over a single volume, `(cel..., 3)` tensors, of
eltype `Complex{T}`: mixed precision throws rather than converting. `out` is not
read when `β` is zero, and the operator is not applied when `α` is zero.
"""
function LinearAlgebra.mul!(out::AbstractVector, opr::AbstractGlaOpr{T}, inp::AbstractVector, α::Number, β::Number) where T<:AbstractFloat
    eltype(out) == eltype(inp) == eltype(opr) || _prcErr(opr, out, inp)
    if length(inp) != size(opr, 2) || length(out) != size(opr, 1)
        throw(ArgumentError("An input of length $(length(inp)) and an output of length $(length(out)) do not fit this operator, which maps $(size(opr, 2)) degrees of freedom to $(size(opr, 1))."))
    end
    α, β = Complex{T}(α), Complex{T}(β)
    if iszero(α)
        iszero(β) ? fill!(out, zero(eltype(out))) : rmul!(out, β)
        return out
    end
    tmp = mulAct!(opr, _devCpy(opr, inp))
    iszero(β) ? (out .= α .* tmp) : (out .= α .* tmp .+ β .* out)
    return out
end
LinearAlgebra.mul!(out::AbstractArray{<:Any, 4}, opr::AbstractGlaOpr, inp::AbstractArray{<:Any, 4}, α::Number, β::Number) =
    (mul!(vec(out), opr, vec(inp), α, β); out)

# A matrix goes column by column
function LinearAlgebra.mul!(out::AbstractMatrix, opr::AbstractGlaOpr, inp::AbstractMatrix, α::Number, β::Number)
    for (outCol, inpCol) in zip(eachcol(out), eachcol(inp))
        mul!(outCol, opr, inpCol, α, β)
    end
    return out
end

# Mixed precision is an error rather than a conversion
function _prcErr(opr::AbstractGlaOpr, args...)
    argTyp = join(unique(string.(eltype.(args))), " and ")
    nam = Base.typename(typeof(opr)).name
    throw(ArgumentError("The operator eltype is $(eltype(opr)) and the argument eltype is $argTyp. Convert the data with $(eltype(opr)).(v), or the operator with $nam{$(real(eltype(first(args))))}(opr)."))
end
mulAct!(opr::AbstractGlaOpr, act::AbstractVector) = _prcErr(opr, act)

"""
    *(opr::AbstractGlaOpr, inp)

Apply an operator to a vector, a `(cel..., 3)` tensor, or a matrix of columns.

The result takes the form of the input, which is left untouched at the cost of
one copy. The eltype must be the `Complex{T}` of `opr`.
"""
function Base.:*(opr::AbstractGlaOpr, inp::AbstractVector)
    eltype(inp) == eltype(opr) || _prcErr(opr, inp)
    if length(inp) != size(opr, 2)
        throw(ArgumentError("An input of length $(length(inp)) does not fit this operator, which takes $(size(opr, 2)) degrees of freedom."))
    end
    return mulAct!(opr, _devCpy(opr, inp))
end
Base.:*(opr::AbstractGlaOpr, inp::AbstractArray{<:Any, 4}) = reshape(opr * vec(inp), glaSze(opr, 1))
Base.:*(opr::AbstractGlaOpr, inp::AbstractMatrix) = mul!(similar(inp, size(opr, 1), size(inp, 2)), opr, inp)

#= A field carries its tiling, so it goes through the methods that check it and
apply the normalization, and only the combine happens here. As in the vector
method, out is never read when β is zero and the operator, which may hide an
iterative solve, is never applied when α is zero. =#
function LinearAlgebra.mul!(out::GlaFld{T}, opr::AbstractGlaOpr{T}, inp::GlaFld{T}, α::Number, β::Number) where T<:AbstractFloat
    α, β = Complex{T}(α), Complex{T}(β)
    if iszero(α)
        iszero(β) ? fill!(out, zero(eltype(out))) : rmul!(out, β)
        return out
    end
    fld = opr * inp
    iszero(β) ? (out .= α .* fld) : axpby!(α, fld, β, out)
    return out
end

Base.adjoint(opr::AbstractGlaOpr) = adjoint!(deepcopy(opr))

# f applied to the adjoint of opr, restoring the memory opr points at afterwards
function _witAdj(f, opr::AbstractGlaOpr)
    adjOpr = adjoint!(opr)
    out = f(adjOpr)
    adjoint!(adjOpr)
    return out
end

# A zero β never reads the output, which may hold anything
function invMul!(w, opr::AbstractGlaOpr, v, α, β)
    iszero(β) && return w .= α .* solve(opr, v, slv(opr))
    return axpby!(α, solve(opr, v, slv(opr)), β, w)
end

invMulAdj!(w, opr::AbstractGlaOpr, v, α, β) = _witAdj(adjOpr -> invMul!(w, adjOpr, v, α, β), opr)

"""
    \\(opr::AbstractGlaOpr, inp::AbstractVector)
    ldiv!(out, opr::AbstractGlaOpr, inp)

Solve `opr * out = inp` iteratively with `slv(opr)`. On a vacuum operator this
solves G₀ itself, which is indefinite and slow to converge. A `GlaOpr` solves only
its G₀ half, with the default solver rather than its own; a `SctOpr` and a
`SusOpr` need no solve.
"""
LinearAlgebra.ldiv!(out, opr::AbstractGlaOpr, inp) = invMul!(out, opr, inp, one(eltype(opr)), zero(eltype(opr)))
Base.:\(opr::AbstractGlaOpr, inp::AbstractVector) = ldiv!(similar(inp, size(opr, 2)), opr, inp)

# An operator only applies to fields on the tiling it was built for
_chkFld(opr::AbstractGlaOpr, fld::GlaFld) = fld.cvol == _srcCvl(opr) ||
    throw(ArgumentError("The field does not live on the source tiling of the operator. An operator only applies to fields on the tiling it was built for."))

"""
    *(opr::AbstractGlaOpr, fld::GlaFld)

Apply `opr` to a field on its source tiling, giving a field on its target tiling.
"""
function Base.:*(opr::AbstractGlaOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat
    _chkFld(opr, fld)
    return GlaFld(opr * fld.dat, _trgCvl(opr))
end

"""
    \\(opr::AbstractGlaOpr, fld::GlaFld)
    ldiv!(out::GlaFld, opr::AbstractGlaOpr, inp::GlaFld)

Solve `opr * out = fld` on the tiling the field lives on.

An iterative solve with `slv(opr)`, as for a vector. The solve runs on the flat
data, since a solver's matrix-shaped `similar` has no `GlaFld` answer.
"""
function Base.:\(opr::AbstractGlaOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat
    _chkFld(opr, fld)
    return GlaFld(opr \ fld.dat, fld.cvol)
end
function LinearAlgebra.ldiv!(out::GlaFld{T}, opr::AbstractGlaOpr{T}, inp::GlaFld{T}) where T<:AbstractFloat
    _chkFld(opr, inp)
    ldiv!(out.dat, opr, inp.dat)
    return out
end
