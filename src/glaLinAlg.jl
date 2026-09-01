using LinearAlgebra
using CUDA

"""
    glaSze(opr::AbstractGlaOpr)

Returns the size of the input/output arrays for an `AbstractGlaOpr` in tensor form.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.
"""
function glaSze(opr::GlaOprVac)
    if isoverlappingoperator(opr)
        return ((length.(opr.trgMsk)..., 3), (length.(opr.srcMsk)..., 3))
    end
    return ((opr.mem.trgVol.cel..., 3), (opr.mem.srcVol.cel..., 3))
end
glaSze(opr::Union{AsyGlaOprVac, SymGlaOprVac}) = ((opr.mem.trgVol.cel..., 3), (opr.mem.srcVol.cel..., 3))
glaSze(opr::MulRegGlaOprVac) = glaSze.(opr.oprMat)
glaSze(opr::InvSctOpr) = glaSze(opr.oprVac)
glaSze(opr::SctOpr) = glaSze(opr.invSctOpr)
glaSze(opr::GlaOpr) = glaSze(opr.sctOpr)

"""
    glaSze(opr::AbstractGlaOpr, dim::Int)

Returns the size of the input/output arrays for an `AbstractGlaOpr` in tensor form in a specified dimension.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.
- `dim::Int`: The index of the dimension to check.
"""
glaSze(opr::AbstractGlaOpr, dim::Int) = glaSze(opr)[dim]
glaSze(opr::MulRegGlaOprVac, dim::Int) = map(x -> x[dim], glaSze(opr))
glaSze(opr::InvSctOpr, dim::Int) = glaSze(opr.oprVac, dim)
glaSze(opr::SctOpr, dim::Int) = glaSze(opr.invSctOpr, dim)
glaSze(opr::GlaOpr, dim::Int) = glaSze(opr.sctOpr, dim)

# Type and size definitions
Base.eltype(::AbstractGlaOpr) = ComplexF64
Base.size(opr::AbstractGlaOpr) = prod.(glaSze(opr))
function Base.size(opr::MulRegGlaOprVac)
    rowSzs = [size(opr.oprMat[i, 1], 1) for i in axes(opr.oprMat, 1)]
    colSzs = [size(opr.oprMat[1, j], 2) for j in axes(opr.oprMat, 2)]
    return (sum(rowSzs), sum(colSzs))
end
Base.size(opr::AbstractGlaOpr, i::Int) = prod(glaSze(opr, i))
Base.size(opr::MulRegGlaOprVac, i::Int) = size(opr)[i]
Base.size(opr::InvSctOpr) = size(opr.oprVac)
Base.size(opr::SctOpr) = size(opr.invSctOpr)
Base.size(opr::GlaOpr) = size(opr.sctOpr)
Base.size(opr::Union{InvSctOpr, SctOpr, GlaOpr}, i::Int) = size(opr)[i]

# Array type definition
Base.similar(opr::AbstractGlaOpr) = arrTyp(opr)(undef, size(opr, 1), size(opr, 2))
function Base.similar(opr::AbstractGlaOpr, ::Type{T}) where T
    AT = arrTyp(opr)
    if AT <: CuArray
        return CuArray{T}(undef, size(opr, 1), size(opr, 2))
    else
        return Array{T}(undef, size(opr, 1), size(opr, 2))
    end
end
function Base.similar(opr::AbstractGlaOpr, dims::Tuple{Vararg{Int}})
    AT = arrTyp(opr)
    if AT <: CuArray
        return CuArray{eltype(opr)}(undef, dims...)
    else
        return Array{eltype(opr)}(undef, dims...)
    end
end
function Base.similar(opr::AbstractGlaOpr, ::Type{T}, dims::Tuple{Vararg{Int}}) where T
    AT = arrTyp(opr)
    if AT <: CuArray
        return CuArray{T}(undef, dims...)
    else
        return Array{T}(undef, dims...)
    end
end

# Indexing functions
Base.IndexStyle(::Type{<:AbstractGlaOpr}) = IndexCartesian()
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
        e = fill!(arrTyp(opr)(undef, size(opr, 2)), zero(eltype(opr)))
        CUDA.@allowscalar e[col] = one(eltype(opr))
        return CUDA.@allowscalar (opr * e)[row]
    end

    rowInd = row isa Integer ? [row] : collect(row)
    colInd = col isa Integer ? [col] : collect(col)

    # Choose the cheapest way to compute the result (adjoint or forward)
    if length(rowInd) <= length(colInd)
        # Forward batched mat–vec
        # build a mini‐identity with 1‑hots in the input slots
        idt = fill!(arrTyp(opr)(undef, numCol, length(colInd)), zero(T))
        for (j, idx) in enumerate(colInd)
            CUDA.@allowscalar idt[idx, j] = one(T)
        end
        out = opr * idt
        res = out[rowInd, :]
    else
        # Adjoint batched vec–mat
        # mini‑identity in the output slots
        idt = fill!(arrTyp(opr)(undef, numRow, length(rowInd)), zero(T))
        for (i, idx) in enumerate(rowInd)
            CUDA.@allowscalar idt[idx, i] = one(T)
        end
        adjOpr = adjoint!(opr) # Compute with the adjoint operator
        outDag = adjOpr * idt
        adjoint!(adjOpr) # Restore the memory the caller's operator still points at
        res = copy(outDag[colInd, :]')
    end
    # An integer index drops its dimension, as for any AbstractMatrix
    (row isa Integer || col isa Integer) && return vec(res)
    return res
end
Base.setindex!(::AbstractGlaOpr, _, __...) = throw(ArgumentError("setindex! is not supported for AbstractGlaOpr"))

"""
    mulAct!(opr::AbstractGlaOpr, act::AbstractVector{ComplexF64})

Apply an operator to a vector. May (will) mutate `act`.

`act` holds source as a flat vector on the same device (CPU/GPU) the operator
computes with. 

# Returns
- `AbstractVector{ComplexF64}`: The result in the flat layout, freshly allocated
"""
function mulAct! end

function mulAct!(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}, act::AbstractVector{ComplexF64})
    if isoverlappingoperator(opr)
        # The masked input is read into the union volume
        actEmb = fill!(similar(act, opr.mem.srcVol.cel..., 3), zero(ComplexF64))
        actEmb[opr.srcMsk..., :] .= reshape(act, glaSze(opr, 2))
        return vec(egoOpr!(opr.mem, actEmb)[opr.trgMsk..., :])
    end
    return vec(egoOpr!(opr.mem, reshape(act, glaSze(opr, 2))))
end

# Block row sums over the flat layout, one region block at a time
function mulAct!(opr::MulRegGlaOprVac, act::AbstractVector{ComplexF64})
    rowSzs = [size(opr.oprMat[i, 1], 1) for i in axes(opr.oprMat, 1)]
    colSzs = [size(opr.oprMat[1, j], 2) for j in axes(opr.oprMat, 2)]
    rowOff = cumsum([0; rowSzs]); colOff = cumsum([0; colSzs])
    outVec = fill!(similar(act, sum(rowSzs)), zero(ComplexF64))
    for i in axes(opr.oprMat, 1), j in axes(opr.oprMat, 2)
        # A block eats its buffer, so every block gets its own copy of the slice
        innBlk = copy(view(act, (colOff[j] + 1):colOff[j + 1]))
        view(outVec, (rowOff[i] + 1):rowOff[i + 1]) .+= mulAct!(opr.oprMat[i, j], innBlk)
    end
    return outVec
end

#= (I - XG₀), in place on the input. In adjoint mode the vacuum operator is
already the adjoint and the susceptibility already conjugated, which leaves
I - G₀' X̄. A composite susceptibility is stored in the flat layout, a single
volume one as a cell tensor that broadcasts over the three components. =#
function mulAct!(opr::InvSctOpr, act::AbstractVector{ComplexF64})
    if opr.oprVac isa GlaCmpOprVac
        if length(act) != size(opr.oprVac, 2)
            throw(ArgumentError("An input of length $(length(act)) does not fit this operator, which has $(size(opr.oprVac, 2)) degrees of freedom."))
        end
        isadjoint(opr) && return act .-= mulAct!(opr.oprVac, opr.sus .* act)
        return act .-= opr.sus .* mulAct!(opr.oprVac, copy(act))
    end
    actTen = reshape(act, glaSze(opr, 2))
    if isadjoint(opr)
        actTen .-= reshape(mulAct!(opr.oprVac, vec(opr.sus .* actTen)), glaSze(opr, 1))
        return act
    end
    actTen .-= opr.sus .* reshape(mulAct!(opr.oprVac, copy(act)), glaSze(opr, 1))
    return act
end

mulAct!(opr::SctOpr, act::AbstractVector{ComplexF64}) = solve(opr.invSctOpr, act, opr.slv)

# G₀ after the solve, the two the other way around in adjoint mode
function mulAct!(opr::GlaOpr, act::AbstractVector{ComplexF64})
    isadjoint(opr) && return mulAct!(opr.sctOpr, mulAct!(opr.sctOpr.invSctOpr.oprVac, act))
    return mulAct!(opr.sctOpr.invSctOpr.oprVac, mulAct!(opr.sctOpr, act))
end

# The buffer the primitive consumes, on the device the operator computes with
function _devCpy(opr::AbstractGlaOpr, inp::AbstractVector{ComplexF64})
    if isgpu(opr) && !(inp isa CuArray)
        @warn "Input array is not a CuArray. Copying data to GPU."
        return CuArray(inp) # The conversion is itself the defensive copy
    end
    return copy(inp)
end

"""
    mul!(out, opr::AbstractGlaOpr, inp, α::Number, β::Number)

Write `α * (opr * inp) + β * out` into `out`, leaving `inp` untouched.

Both arrays are either flat vectors of degrees of freedom or, over a single
volume, `(cel..., 3)` tensors. `out` is never read when `β` is zero, and the
operator is never applied when `α` is zero.

# Returns
- `out`, holding the result

# Throws
- `ArgumentError`: If either array does not fit the operator
"""
function LinearAlgebra.mul!(out::AbstractVector{ComplexF64}, opr::AbstractGlaOpr, inp::AbstractVector{ComplexF64}, α::Number, β::Number)
    if length(inp) != size(opr, 2) || length(out) != size(opr, 1)
        throw(ArgumentError("An input of length $(length(inp)) and an output of length $(length(out)) do not fit this operator, which maps $(size(opr, 2)) degrees of freedom to $(size(opr, 1))."))
    end
    if iszero(α)
        iszero(β) ? fill!(out, zero(ComplexF64)) : rmul!(out, β)
        return out
    end
    tmp = mulAct!(opr, _devCpy(opr, inp))
    iszero(β) ? (out .= α .* tmp) : (out .= α .* tmp .+ β .* out)
    return out
end
LinearAlgebra.mul!(out::AbstractArray{ComplexF64, 4}, opr::AbstractGlaOpr, inp::AbstractArray{ComplexF64, 4}, α::Number, β::Number) =
    (mul!(vec(out), opr, vec(inp), α, β); out)

"""
    *(opr::AbstractGlaOpr, inp)

Apply an operator to a vector, a `(cel..., 3)` tensor, or a matrix of columns.

The result takes the form of the input, and the input is left untouched. The
product costs one copy of the input, which is the floor for an operator whose
kernel consumes what it is handed.

# Returns
- The result, on the target volume of `opr`

# Throws
- `ArgumentError`: If the input does not fit the operator
"""
function Base.:*(opr::AbstractGlaOpr, inp::AbstractVector{ComplexF64})
    if length(inp) != size(opr, 2)
        throw(ArgumentError("An input of length $(length(inp)) does not fit this operator, which takes $(size(opr, 2)) degrees of freedom."))
    end
    return mulAct!(opr, _devCpy(opr, inp))
end
Base.:*(opr::AbstractGlaOpr, inp::AbstractArray{ComplexF64, 4}) = reshape(opr * vec(inp), glaSze(opr, 1))
function Base.:*(opr::AbstractGlaOpr, inp::AbstractMatrix{ComplexF64})
    out = similar(inp, size(opr, 1), size(inp, 2))
    for (outCol, inpCol) in zip(eachcol(out), eachcol(inp))
        mul!(outCol, opr, inpCol)
    end
    return out
end

#= A field carries its tiling, so it goes through the methods that check it and
apply the normalization, and only the combine happens here. As in the vector
method, out is never read when β is zero and the operator, which may hide an
iterative solve, is never applied when α is zero. =#
function LinearAlgebra.mul!(out::GlaFld, opr::AbstractGlaOpr, inp::GlaFld, α::Number, β::Number)
    if iszero(α)
        iszero(β) ? fill!(out, zero(ComplexF64)) : rmul!(out, β)
        return out
    end
    fld = opr * inp
    iszero(β) ? (out .= α .* fld) : axpby!(α, fld, β, out)
    return out
end

function Base.:*(opr::MulRegGlaOprVac, innVec::Vector{<:AbstractArray{ComplexF64, 4}})
    m, n = size(opr.oprMat)
    @assert length(innVec) == n "expected $n source blocks, got $(length(innVec))"
    outVec = [opr.oprMat[i, 1] * innVec[1] for i in 1:m] # j = 1
    for i in 1:m, j in 2:n
        outVec[i] .+= opr.oprMat[i, j] * innVec[j]
    end
    return outVec
end

"""
    *(opr::GlaOprVac, fld::GlaFld)

Apply a vacuum operator to a field over a single region.

The field has to be a field over the source volume of the operator, which for a
`GlaFld` means a tiling of exactly one region equal to that volume. The result is
a field over the target volume, again as a tiling of one region.

A single region has a single cell volume, so the √ΔV normalization of `GlaFld` is
a scalar on each side rather than a diagonal, and it comes out of the operator as
the ratio of the two. That ratio is one whenever the two volumes share a cell
size, which covers every self operator.

# Arguments
- `opr::GlaOprVac`: The operator
- `fld::GlaFld`: The field, which must live on the source volume of `opr`

# Returns
- `GlaFld`: The result, on the target volume of `opr`

# Throws
- `ArgumentError`: If the field is not a field over the source volume of the
  operator, or if the operator takes the masked route
"""
function Base.:*(opr::GlaOprVac, fld::GlaFld)
    srcVol = opr.mem.srcVol
    if isoverlappingoperator(opr)
        throw(ArgumentError("This operator is built on the union of its two volumes and reads its input through a mask, so it does not take a field. Apply it to a plain array of the masked size instead."))
    end
    if nregions(fld.cvol) != 1 || regions(fld.cvol)[1] != srcVol
        throw(ArgumentError("The field does not live on the source volume of the operator, which is a ($(join(srcVol.cel, "×"))) cell volume of ($(join(srcVol.scl, "×")))λ³ cells."))
    end
    outDat = opr * fld.dat
    nrm = sqrt(Float64(prod(opr.mem.trgVol.scl) // prod(srcVol.scl)))
    nrm != 1 && rmul!(outDat, nrm)
    return GlaFld(outDat, GlaCmpVol(opr.mem.trgVol))
end

"""
    adjoint!(opr::AbstractGlaOpr)

Return the adjoint of an operator, reusing its memory.

The call may mutate whatever the argument holds, so the argument must not be used
again: the returned operator is the only valid handle on that memory. A second
call restores the memory, which is what makes `adjoint!(adjoint!(opr))` a valid
operator equal to the original. Code that needs the argument to survive should
call `adjoint` instead, which works on a copy.

Some types rearrange in place and hand the same object back, others return a new
wrapper around the same memory, so the return value always has to be used.

# Arguments
- `opr::AbstractGlaOpr`: The operator to adjoint

# Returns
- The adjoint operator
"""
function adjoint!(opr::GlaOprVac)
    # Mark the adjoint
    opr.mem.cmpInf.adjMod = !opr.mem.cmpInf.adjMod

    # Swap source and target volumes (transpose)
    opr.mem.trgVol, opr.mem.srcVol = opr.mem.srcVol, opr.mem.trgVol
    opr.mem.mixInf = GlaExtInf(opr.mem.trgVol, opr.mem.srcVol)

    # Take the conjugate of the Fourier coefficients (conjugate transpose)
    # Also swap the last two axes because they hold the source and target
    # partition of a cross-scale pair (which must be transposed for the adjoint)
    opr.mem.egoFur = collect(map(arr -> conj.(permutedims(arr, (1, 2, 3, 4, 6, 5))), opr.mem.egoFur))

    # The masks are immutable fields, so the transpose needs a new wrapper
    return GlaOprVac(opr.mem, opr.trgMsk, opr.srcMsk)
end
adjoint!(opr::Union{AsyGlaOprVac, SymGlaOprVac}) = opr # These operators are Hermitian (self-adjoint)
function adjoint!(opr::MulRegGlaOprVac)
    adjMat = similar(opr.oprMat, reverse(size(opr.oprMat))) # New operator matrix with adjoint size
    for i in axes(opr.oprMat, 1)
        for j in axes(opr.oprMat, 2)
            adjMat[j, i] = adjoint!(opr.oprMat[i, j])
        end
    end
    # note that now, opr's entries are all adjoint's or the original entries
    return MulRegGlaOprVac(adjMat)
end
function adjoint!(opr::InvSctOpr)
    opr.oprVac = adjoint!(opr.oprVac)
    opr.sus = conj(opr.sus)  # Conjugate the susceptibility
    return opr
end
function adjoint!(opr::SctOpr)
    opr.invSctOpr = adjoint!(opr.invSctOpr)
    return opr
end
function adjoint!(opr::GlaOpr)
    opr.sctOpr = adjoint!(opr.sctOpr)
    return opr
end
Base.adjoint(opr::AbstractGlaOpr) = adjoint!(deepcopy(opr))

# A zero β never reads the output, which may hold anything
function invMul!(w, opr::AbstractGlaOpr, v, α, β)
    iszero(β) && return w .= α .* solve(opr, v, slv(opr))
    return axpby!(α, solve(opr, v, slv(opr)), β, w)
end
function invMul!(w, opr::SctOpr, v, α, β)
    iszero(β) && return w .= α .* (opr.invSctOpr * v)
    return axpby!(α, opr.invSctOpr * v, β, w)
end
function invMul!(w, opr::GlaOpr, v, α, β)
    actG0Inv = solve(opr.sctOpr.invSctOpr.oprVac, v, slv(opr.sctOpr.invSctOpr))
    out = opr.sctOpr.invSctOpr.oprVac * actG0Inv
    iszero(β) && return w .= α .* out
    return axpby!(α, out, β, w)
end

function invMulAdj!(w, opr::AbstractGlaOpr, v, α, β)
    adjOpr = adjoint!(opr) # Compute with the adjoint operator
    out = invMul!(w, adjOpr, v, α, β) # Inverse adjoint matrix-vector product
    adjoint!(opr) # Restore the original operator
    return out
end
function invMulAdj!(w, opr::SctOpr, v, α, β)
    adjOpr = adjoint!(opr.invSctOpr) # Compute with the adjoint operator
    out = invMul!(w, adjOpr, v, α, β)
    adjoint!(opr.invSctOpr) # Restore the original operator
    return out
end
function invMulAdj!(w, opr::GlaOpr, v, α, β)
    adjOpr = adjoint!(opr.sctOpr.invSctOpr) # Compute with the adjoint operator
    actWInvDag = adjOpr * v
    adjoint!(opr.sctOpr.invSctOpr) # Restore the original operator

    adjOpr = adjoint!(opr.sctOpr.invSctOpr.oprVac) # Compute with the adjoint operator
    out = axpby!(α, adjOpr * actWInvDag, β, w)
    adjoint!(opr.sctOpr.invSctOpr.oprVac) # Restore the original operator
    return out
end
