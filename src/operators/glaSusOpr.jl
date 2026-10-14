"""
    SusOpr{T, A}

The susceptibility operator X, taking a field to the polarization current it
drives, over the tiling `cvol`.

`sus` is either a vector in the flat degree of freedom layout, applied entrywise,
or a `(celTot, 3, 3)` array of cell tensors in `coordinates(cvol)` order, applied
as a matrix in each cell (gyrotropy, rotated crystals). χ is dimensionless and
diagonal in position, so it commutes with the √ΔV normalization of `GlaFld`.
`X \\ f` divides pointwise and throws where χ vanishes.
"""
mutable struct SusOpr{T<:AbstractFloat, A<:AbstractArray{Complex{T}}} <: AbstractGlaOpr{T}
    sus::A
    cvol::GlaCmpVol
    adjMod::Bool
end

const SusceptibilityOperator = SusOpr

#= The susceptibility of one cell, repeated over the three vector components of
that cell. A region block of the flat layout is the vec of a (cel..., 3) array,
so the three copies of a block sit one after the other. =#
function _expSus(cvol::GlaCmpVol, susCel::Vector{Complex{T}}) where T<:AbstractFloat
    off = _dofOff(cvol) .÷ 3
    return reduce(vcat, [repeat(view(susCel, (off[idx] + 1):off[idx + 1]), 3) for idx in 1:nregions(cvol)])
end

# One scalar per cell read as the isotropic tensor of that cell
function _diaSus(susCel::Vector{Complex{T}}) where T<:AbstractFloat
    susTen = zeros(Complex{T}, length(susCel), 3, 3)
    for dir in 1:3
        susTen[:, dir, dir] .= susCel
    end
    return susTen
end

#= The susceptibility of one region, as one value per cell or, when the region
asks for anisotropy, as one 3×3 tensor per cell in the same cell order. =#
function _regSus(::Type{T}, reg::GlaVol, idx::Integer, sus) where T<:AbstractFloat
    celNum = prod(reg.cel)
    # Anything that is neither a number nor an array is a function of position
    if !(sus isa Number || sus isa AbstractArray)
        susCel = [sus(pos) for (pos, _, _) in coordinates(GlaCmpVol(reg))]
        eltype(susCel) <: Number && return Vector{Complex{T}}(susCel)
        for (cel, mat) in enumerate(susCel)
            size(mat) == (3, 3) || throw(ArgumentError("The susceptibility function returned a value of size $(size(mat)) at cell $cel of region $idx, and a susceptibility is either a number or a 3×3 tensor."))
        end
        return Array{Complex{T}}(permutedims(stack(susCel), (3, 1, 2)))
    end
    sus isa Number && return fill(Complex{T}(sus), celNum)
    if sus isa AbstractMatrix{<:Number}
        size(sus) == (3, 3) || throw(ArgumentError("The susceptibility matrix of region $idx has size $(size(sus)), and a matrix is read as the 3×3 tensor of a uniform medium. Pass a $(join(reg.cel, "×"))×3×3 array for a tensor that varies from cell to cell."))
        return repeat(reshape(Array{Complex{T}}(sus), 1, 3, 3), celNum, 1, 1)
    end
    if sus isa AbstractArray{<:Number, 3}
        size(sus) == reg.cel || throw(ArgumentError("The susceptibility tensor of region $idx has size $(size(sus)), but region $idx has $(join(reg.cel, "×")) cells."))
        return vec(Array{Complex{T}}(sus))
    end
    if sus isa AbstractArray{<:Number, 5}
        size(sus) == (reg.cel..., 3, 3) || throw(ArgumentError("The susceptibility tensor array of region $idx has size $(size(sus)), but region $idx needs $(join(reg.cel, "×"))×3×3 entries."))
        return reshape(Array{Complex{T}}(sus), celNum, 3, 3)
    end
    throw(ArgumentError("A susceptibility of type $(typeof(sus)) fits region $idx neither as one value per cell nor as one 3×3 tensor per cell."))
end

# Numbers, functions, per-region vectors and static 3×3 tensors are read on the host
susDev(sus) = sus isa AbstractArray{<:Number} && !(sus isa StaticArray) ? get_backend(sus) : CPU()

#= Any accepted susceptibility, read into one of the two storage shapes. The
regions are read one at a time and stacked, and a tiling that asks for a tensor
anywhere is stored as a tensor everywhere. =#
function _susDat(::Type{T}, cvol::GlaCmpVol, sus, dev::Backend) where T<:AbstractFloat
    regs = regions(cvol)
    celNum = sum(prod(reg.cel) for reg in regs)
    if sus isa AbstractVector{<:Number}
        length(sus) in (celNum, 3 * celNum) || throw(ArgumentError("A susceptibility vector of length $(length(sus)) fits neither the $(3 * celNum) degrees of freedom nor the $celNum cells of this composite volume."))
        susDof = length(sus) == 3 * celNum ? Vector{Complex{T}}(Array(sus)) :
            _expSus(cvol, Vector{Complex{T}}(Array(sus)))
        return adapt(dev, susDof)
    end
    if sus isa AbstractArray && ndims(sus) in (3, 5) && nregions(cvol) != 1
        throw(ArgumentError("A single susceptibility tensor only fits a composite volume of one region, and this one has $(nregions(cvol)). Pass one tensor per region as a vector."))
    end
    regSus = sus isa AbstractVector ? sus : fill(sus, length(regs))
    length(regSus) == length(regs) || throw(ArgumentError("Got $(length(regSus)) susceptibilities for a composite volume of $(length(regs)) regions."))
    blkLst = [_regSus(T, reg, idx, regSus[idx]) for (idx, reg) in enumerate(regs)]
    susDat = if any(blk -> ndims(blk) == 3, blkLst)
        reduce(vcat, (ndims(blk) == 3 ? blk : _diaSus(blk) for blk in blkLst))
    else
        _expSus(cvol, reduce(vcat, blkLst))
    end
    return adapt(dev, susDat)
end

"""
    SusOpr{T}(cvol::GlaCmpVol, sus; dev::Backend)
    SusOpr{T}(vol::GlaVol, sus; dev::Backend)
    SusOpr(vol, sus; dev::Backend)

The susceptibility operator of a volume, plain or composite. `dev` defaults to
the backend of `sus` when it is an array, else `CPU()`.

`sus` is a number, a 3×3 matrix, an array of one value per cell, an array of one
3×3 tensor per cell (`(cel..., 3, 3)`), a function of the cell center returning
either, a vector holding any of those per region, a vector of one value per cell
in layout order, or a vector already in the degree of freedom layout. The shapes
that name a cell grid only fit a tiling of one region.
"""
SusOpr{T}(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus)) where T<:AbstractFloat =
    SusOpr(_susDat(T, cvol, sus, dev), cvol, false)
SusOpr{T}(vol::GlaVol, sus; kw...) where T<:AbstractFloat = SusOpr{T}(GlaCmpVol(vol), sus; kw...)
SusOpr(args...; kw...) = SusOpr{dflPrc}(args...; kw...)

"""
    SusOpr{T}(opr::SusOpr)

Convert the storage precision of a susceptibility operator to `T`.
"""
SusOpr{T}(opr::SusOpr{T}) where T<:AbstractFloat = opr
SusOpr{T}(opr::SusOpr) where T<:AbstractFloat =
    SusOpr(Complex{T}.(opr.sus), opr.cvol, opr.adjMod)

Adapt.@adapt_structure SusOpr

isadjoint(opr::SusOpr) = opr.adjMod

isquasistatic(::SusOpr) = false

_relKnd(::SusOpr) = :self # A susceptibility never leaves its own cell

KernelAbstractions.get_backend(opr::SusOpr) = get_backend(opr.sus)

#= As for GlaCmpOprVac: a tiling of one region has the tensor shape of that
region, a tiling of several has one shape per region and no tensor form. =#
function glaSze(opr::SusOpr)
    sze = map(reg -> (reg.cel..., 3), regions(opr.cvol))
    return length(sze) == 1 ? (sze[1], sze[1]) : (sze, sze)
end

function Base.size(opr::SusOpr)
    dofNum = _dofOff(opr.cvol)[end]
    return (dofNum, dofNum)
end
Base.size(opr::SusOpr, i::Int) = size(opr)[i]

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

#= The pointwise inverse of a diagonal susceptibility, which exists only where
the susceptibility does. =#
function _invSus(opr::SusOpr{T, <:AbstractVector{Complex{T}}}) where T<:AbstractFloat
    if any(iszero, opr.sus)
        throw(ArgumentError("The susceptibility vanishes in $(count(iszero, opr.sus)) of the $(length(opr.sus)) degrees of freedom of this volume, and a vacuum cell has no inverse. Divide by a susceptibility with no zero entries, or solve with the scattering operator instead."))
    end
    return SusOpr(inv.(opr.sus), opr.cvol, opr.adjMod)
end

# The inverse of every cell tensor, with the cells moved to the last axis to read them as 3×3 matrices
function _invSus(opr::SusOpr{T, <:AbstractArray{Complex{T}, 3}}) where T<:AbstractFloat
    susCel = reinterpret(SMatrix{3, 3, Complex{T}, 9}, vec(permutedims(opr.sus, (2, 3, 1))))
    susDet = det.(susCel)
    if any(iszero, susDet)
        throw(ArgumentError("The susceptibility tensor is singular in $(count(iszero, susDet)) of the $(length(susDet)) cells of this volume, and a singular cell has no inverse. Divide by a susceptibility that is invertible in every cell, or solve with the scattering operator instead."))
    end
    susInv = reshape(reinterpret(Complex{T}, inv.(susCel)), 3, 3, :)
    return SusOpr(permutedims(susInv, (3, 1, 2)), opr.cvol, opr.adjMod)
end

#= The one operator whose inverse is not a solve: a susceptibility is diagonal
in position, so the inverse action is the pointwise one. =#
function invMul!(w, opr::SusOpr, v, α, β)
    out = mulAct!(_invSus(opr), _devCpy(opr, v))
    iszero(β) && return w .= α .* out
    return axpby!(α, out, β, w)
end
isvarying(::SusOpr) = false

#= A susceptibility carries no normalization factor of its own: it is
dimensionless and diagonal in position, so it commutes with the √ΔV of GlaFld. =#
_srcCvl(opr::SusOpr) = opr.cvol
_trgCvl(opr::SusOpr) = opr.cvol

#= As with GlaVacOprMem, this hook is the generic serializer's extension point
(struct fields, array elements, and the top level via serialize(io::IO, x)'s own
fallback to it), not io::IO overloads. Only the move to the host is special;
cvol and adjMod are plain data. =#
function Serialization.serialize(s::AbstractSerializer, opr::SusOpr)
    hst = adapt(CPU(), opr)
    Serialization.serialize_type(s, typeof(hst))
    serialize(s, hst.sus)
    serialize(s, hst.cvol)
    serialize(s, hst.adjMod)
end
function Serialization.deserialize(s::AbstractSerializer, ::Type{<:SusOpr})
    susDat = deserialize(s)
    cvol = deserialize(s)
    adjMod = deserialize(s)
    return SusOpr(susDat, cvol, adjMod)
end
