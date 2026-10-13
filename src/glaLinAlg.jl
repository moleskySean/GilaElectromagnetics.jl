using LinearAlgebra

"""
    glaSze(opr::AbstractGlaOpr)
    glaSze(opr::AbstractGlaOpr, dim::Int)

The `(cel..., 3)` tensor shapes of the output and input of `opr`, like `size` for
the tensor form. Over several regions, one shape per region.
"""
function glaSze(opr::GlaOprVac)
    if isoverlappingoperator(opr)
        return ((length.(opr.trgMsk)..., 3), (length.(opr.srcMsk)..., 3))
    end
    return ((opr.mem.trgVol.cel..., 3), (opr.mem.srcVol.cel..., 3))
end
glaSze(opr::Union{AsyGlaOprVac, SymGlaOprVac}) = ((opr.mem.trgVol.cel..., 3), (opr.mem.srcVol.cel..., 3))
#= As for GlaCmpOprVac: a tiling of one region has the tensor shape of that
region, a tiling of several has one shape per region and no tensor form. =#
function glaSze(opr::SusOpr)
    sze = map(reg -> (reg.cel..., 3), regions(opr.cvol))
    return length(sze) == 1 ? (sze[1], sze[1]) : (sze, sze)
end
glaSze(opr::InvSctOpr) = glaSze(opr.oprVac)
glaSze(opr::SctOpr) = glaSze(opr.invSctOpr)
glaSze(opr::GlaOpr) = glaSze(opr.sctOpr)

glaSze(opr::AbstractGlaOpr, dim::Int) = glaSze(opr)[dim]
glaSze(opr::InvSctOpr, dim::Int) = glaSze(opr.oprVac, dim)
glaSze(opr::SctOpr, dim::Int) = glaSze(opr.invSctOpr, dim)
glaSze(opr::GlaOpr, dim::Int) = glaSze(opr.sctOpr, dim)

# Type and size definitions
Base.eltype(::AbstractGlaOpr{T}) where T<:AbstractFloat = Complex{T}
Base.eltype(::Type{<:AbstractGlaOpr{T}}) where T<:AbstractFloat = Complex{T}
Base.size(opr::AbstractGlaOpr) = prod.(glaSze(opr))
Base.size(opr::AbstractGlaOpr, i::Int) = prod(glaSze(opr, i))
function Base.size(opr::SusOpr)
    dofNum = sum(3 * prod(reg.cel) for reg in regions(opr.cvol))
    return (dofNum, dofNum)
end
Base.size(opr::SusOpr, i::Int) = size(opr)[i]
Base.size(opr::InvSctOpr) = size(opr.oprVac)
Base.size(opr::SctOpr) = size(opr.invSctOpr)
Base.size(opr::GlaOpr) = size(opr.sctOpr)
Base.size(opr::Union{InvSctOpr, SctOpr, GlaOpr}, i::Int) = size(opr)[i]

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
        adjOpr = adjoint!(opr) # Compute with the adjoint operator
        outDag = adjOpr * adapt(get_backend(adjOpr), idt)
        adjoint!(adjOpr) # Restore the memory the caller's operator still points at
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

function mulAct!(opr::Union{GlaOprVac{T}, AsyGlaOprVac{T}, SymGlaOprVac{T}}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
    if isoverlappingoperator(opr)
        # The masked input is read into the union volume
        actEmb = fill!(similar(act, opr.mem.srcVol.cel..., 3), zero(eltype(act)))
        actEmb[opr.srcMsk..., :] .= reshape(act, glaSze(opr, 2))
        return vec(egoOpr!(opr.mem, actEmb)[opr.trgMsk..., :])
    end
    return vec(egoOpr!(opr.mem, reshape(act, glaSze(opr, 2))))
end

# The diagonal shape, entry by entry on the flat layout
mulAct!(opr::SusOpr{T, <:AbstractVector{Complex{T}}}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat =
    act .*= opr.sus

#= The tensor shape, one region block at a time. A component occupies a
contiguous run of celNum entries inside a block, so the block reshapes to
(celNum, 3) and each row of the cell tensor is one broadcast of three terms. =#
function mulAct!(opr::SusOpr{T, <:AbstractArray{Complex{T}, 3}}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
    celOff, dofOff = 0, 0
    for reg in regions(opr.cvol)
        celNum = prod(reg.cel)
        actBlk = reshape(view(act, (dofOff + 1):(dofOff + 3 * celNum)), celNum, 3)
        innBlk = copy(actBlk) # later rows still read what earlier rows overwrite
        susBlk = view(opr.sus, (celOff + 1):(celOff + celNum), :, :)
        for dir in 1:3
            @views actBlk[:, dir] .= susBlk[:, dir, 1] .* innBlk[:, 1] .+
                susBlk[:, dir, 2] .* innBlk[:, 2] .+ susBlk[:, dir, 3] .* innBlk[:, 3]
        end
        celOff += celNum
        dofOff += 3 * celNum
    end
    return act
end

#= (I - XG₀), in place on the input. In adjoint mode the vacuum operator is
already the adjoint and the susceptibility already the conjugate transpose,
which leaves I - G₀' X'. =#
function mulAct!(opr::InvSctOpr{T}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
    isadjoint(opr) && return act .-= mulAct!(opr.oprVac, mulAct!(opr.sus, copy(act)))
    return act .-= mulAct!(opr.sus, mulAct!(opr.oprVac, copy(act)))
end

mulAct!(opr::SctOpr{T}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat = solve(opr.invSctOpr, act, opr.slv)

# G₀ after the solve, the two the other way around in adjoint mode
function mulAct!(opr::GlaOpr{T}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
    isadjoint(opr) && return mulAct!(opr.sctOpr, mulAct!(opr.sctOpr.invSctOpr.oprVac, act))
    return mulAct!(opr.sctOpr.invSctOpr.oprVac, mulAct!(opr.sctOpr, act))
end

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
function LinearAlgebra.mul!(out::AbstractVector{Complex{T}}, opr::AbstractGlaOpr{T}, inp::AbstractVector{Complex{T}}, α::Number, β::Number) where T<:AbstractFloat
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
LinearAlgebra.mul!(out::AbstractArray{Complex{T}, 4}, opr::AbstractGlaOpr{T}, inp::AbstractArray{Complex{T}, 4}, α::Number, β::Number) where T<:AbstractFloat =
    (mul!(vec(out), opr, vec(inp), α, β); out)

# A matrix goes column by column, as in the * method
function LinearAlgebra.mul!(out::AbstractMatrix{Complex{T}}, opr::AbstractGlaOpr{T}, inp::AbstractMatrix{Complex{T}}, α::Number, β::Number) where T<:AbstractFloat
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
LinearAlgebra.mul!(out::AbstractVector, opr::AbstractGlaOpr, inp::AbstractVector, α::Number, β::Number) = _prcErr(opr, out, inp)
LinearAlgebra.mul!(out::AbstractArray{<:Number, 4}, opr::AbstractGlaOpr, inp::AbstractArray{<:Number, 4}, α::Number, β::Number) = _prcErr(opr, out, inp)
LinearAlgebra.mul!(out::AbstractMatrix, opr::AbstractGlaOpr, inp::AbstractMatrix, α::Number, β::Number) = _prcErr(opr, out, inp)
Base.:*(opr::AbstractGlaOpr, inp::AbstractVector) = _prcErr(opr, inp)
Base.:*(opr::AbstractGlaOpr, inp::AbstractArray{<:Number, 4}) = _prcErr(opr, inp)
Base.:*(opr::AbstractGlaOpr, inp::AbstractMatrix) = _prcErr(opr, inp)

"""
    *(opr::AbstractGlaOpr, inp)

Apply an operator to a vector, a `(cel..., 3)` tensor, or a matrix of columns.

The result takes the form of the input, which is left untouched at the cost of
one copy. The eltype must be the `Complex{T}` of `opr`.
"""
function Base.:*(opr::AbstractGlaOpr{T}, inp::AbstractVector{Complex{T}}) where T<:AbstractFloat
    if length(inp) != size(opr, 2)
        throw(ArgumentError("An input of length $(length(inp)) does not fit this operator, which takes $(size(opr, 2)) degrees of freedom."))
    end
    return mulAct!(opr, _devCpy(opr, inp))
end
Base.:*(opr::AbstractGlaOpr{T}, inp::AbstractArray{Complex{T}, 4}) where T<:AbstractFloat = reshape(opr * vec(inp), glaSze(opr, 1))
function Base.:*(opr::AbstractGlaOpr{T}, inp::AbstractMatrix{Complex{T}}) where T<:AbstractFloat
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

"""
    *(opr::GlaOprVac, fld::GlaFld)

Apply a vacuum operator to a field whose tiling is the single region `srcVol`,
giving a field on `trgVol`. The √ΔV normalization contributes the scalar
`sqrt(ΔV_trg / ΔV_src)`. Overlapping (masked) operators take arrays, not fields.
"""
function Base.:*(opr::GlaOprVac{T}, fld::GlaFld{T}) where T<:AbstractFloat
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

The adjoint of `opr`, reusing its memory. Use only the return value afterwards:
the argument may have been mutated or replaced. `adjoint!(adjoint!(opr))` restores
the original; `adjoint` works on a copy instead.
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
function adjoint!(opr::SusOpr{T, <:AbstractVector{Complex{T}}}) where T<:AbstractFloat
    opr.adjMod = !opr.adjMod
    opr.sus .= conj.(opr.sus)
    return opr
end
# The adjoint of a tensor susceptibility transposes each cell as well
function adjoint!(opr::SusOpr{T, <:AbstractArray{Complex{T}, 3}}) where T<:AbstractFloat
    opr.adjMod = !opr.adjMod
    opr.sus .= conj.(permutedims(opr.sus, (1, 3, 2)))
    return opr
end
function adjoint!(opr::InvSctOpr)
    opr.oprVac = adjoint!(opr.oprVac)
    adjoint!(opr.sus)
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
# Both inverse actions below are matvecs, so both are admissible preconditioners
isvarying(::SctOpr) = false
#= The inverse of G₀(I - XG₀)⁻¹ is (I - XG₀)G₀⁻¹, so only the vacuum half is
solved for. The two halves swap in adjoint mode, as they do in mulAct!. =#
function invMul!(w, opr::GlaOpr, v, α, β)
    invSct = opr.sctOpr.invSctOpr
    out = isadjoint(opr) ? solve(invSct.oprVac, invSct * v, slv(invSct)) :
        invSct * solve(invSct.oprVac, v, slv(invSct))
    iszero(β) && return w .= α .* out
    return axpby!(α, out, β, w)
end

#= The pointwise inverse of a diagonal susceptibility, which exists only where
the susceptibility does. =#
function _invSus(opr::SusOpr{T, <:AbstractVector{Complex{T}}}) where T<:AbstractFloat
    if any(iszero, opr.sus)
        throw(ArgumentError("The susceptibility vanishes in $(count(iszero, opr.sus)) of the $(length(opr.sus)) degrees of freedom of this volume, and a vacuum cell has no inverse. Divide by a susceptibility with no zero entries, or solve with the scattering operator instead."))
    end
    return SusOpr(inv.(opr.sus), opr.cvol, opr.adjMod)
end

# The cofactor inverse of every cell tensor, all cells of a component at once
function _invSus(opr::SusOpr{T, <:AbstractArray{Complex{T}, 3}}) where T<:AbstractFloat
    susAdj = similar(opr.sus)
    for row in 1:3, col in 1:3
        rowNxt, rowPrv = mod1(row + 1, 3), mod1(row + 2, 3)
        colNxt, colPrv = mod1(col + 1, 3), mod1(col + 2, 3)
        @views susAdj[:, row, col] .=
            opr.sus[:, colNxt, rowNxt] .* opr.sus[:, colPrv, rowPrv] .-
            opr.sus[:, colNxt, rowPrv] .* opr.sus[:, colPrv, rowNxt]
    end
    susDet = @views opr.sus[:, 1, 1] .* susAdj[:, 1, 1] .+
        opr.sus[:, 1, 2] .* susAdj[:, 2, 1] .+ opr.sus[:, 1, 3] .* susAdj[:, 3, 1]
    if any(iszero, susDet)
        throw(ArgumentError("The susceptibility tensor is singular in $(count(iszero, susDet)) of the $(length(susDet)) cells of this volume, and a singular cell has no inverse. Divide by a susceptibility that is invertible in every cell, or solve with the scattering operator instead."))
    end
    return SusOpr(susAdj ./ susDet, opr.cvol, opr.adjMod)
end

#= The one operator whose inverse is not a solve: a susceptibility is diagonal
in position, so the inverse action is the pointwise one. =#
function invMul!(w, opr::SusOpr, v, α, β)
    out = mulAct!(_invSus(opr), _devCpy(opr, v))
    iszero(β) && return w .= α .* out
    return axpby!(α, out, β, w)
end
isvarying(::SusOpr) = false

function invMulAdj!(w, opr::AbstractGlaOpr, v, α, β)
    adjOpr = adjoint!(opr) # Compute with the adjoint operator
    out = invMul!(w, adjOpr, v, α, β) # Inverse adjoint matrix-vector product
    adjoint!(opr) # Restore the original operator
    return out
end

"""
    \\(opr::AbstractGlaOpr, inp::AbstractVector)
    ldiv!(out, opr::AbstractGlaOpr, inp)

Solve `opr * out = inp` iteratively with `slv(opr)`. On a vacuum operator this
solves G₀ itself, which is indefinite and slow to converge. A `SusOpr` divides
pointwise instead.
"""
LinearAlgebra.ldiv!(out, opr::AbstractGlaOpr, inp) = invMul!(out, opr, inp, one(eltype(opr)), zero(eltype(opr)))
Base.:\(opr::AbstractGlaOpr, inp::AbstractVector) = ldiv!(similar(inp, size(opr, 2)), opr, inp)
