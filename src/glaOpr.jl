"""
    GilaOperators

This module provides the core operator types for the Gila package, including the vacuum Green function operator,
scattering operators, and their compositions.

# Types
- `AbstractGlaOpr`: Abstract base type for all operators
- `GlaOprVac`: Vacuum Green function operator G₀
- `InvSctOpr`: Inverse scattering operator (I - XG₀)⁻¹
- `SctOpr`: Scattering operator with solver
- `GlaOpr`: Full Green function operator G₀(I - XG₀)⁻¹

# Type Aliases
- `VacuumGreenOperator`: Alias for `GlaOprVac`
- `InverseScatteringOperator`: Alias for `InvSctOpr`
- `ScatteringOperator`: Alias for `SctOpr`
- `GreenOperator`: Alias for `GlaOpr`
"""
module GilaOperators

using ..GilaVolumes
using ..GilaFields
using ..GilaVacuum
using ..GilaTypes
using ..GilaSolvers
using Adapt
using KernelAbstractions
using Serialization
using StaticArrays: StaticArray

import LinearAlgebra: adjoint!

import ..GilaTypes: isadjoint, _shwRow
import ..GilaVacuum: egoCmpPos, setBlc!
import ..GilaVolumes: _lwrEdg, _uprEdg, _ovrLap, _volDesc, _sclStr, _ratStr
import ..GilaFields: _eqvCvl
import ..GilaSolvers: isvarying

export GlaOprVac, AsyGlaOprVac, SymGlaOprVac, GlaCmpOprVac, AsyGlaCmpOprVac, SymGlaCmpOprVac, InvSctOpr, SctOpr, GlaOpr, SusOpr
export VacuumGreenOperator, AsymVacuumGreenOperator, SymVacuumGreenOperator, CompositeVacuumGreenOperator, AsymCompositeVacuumGreenOperator, SymCompositeVacuumGreenOperator, InverseScatteringOperator, ScatteringOperator, GreenOperator, SusceptibilityOperator
export isadjoint, isselfoperator, isexternaloperator, isoverlappingoperator, isquasistatic, adjoint!, glaSze, slv, sus, asym, sym, setSus!, setBlc!

"""
    GlaOprVac{T}

Represents the vacuum Green function operator G₀, which describes electromagnetic
interactions in free space. `T` is the real storage precision, so the operator
acts on `Complex{T}` data.

# Fields
- `mem::GlaVacOprMem{T}`: Memory structure containing the operator's data, including
  volume information and Fourier coefficients
- `srcMsk::NTuple{3, StepRange{Int64, Int64}}`: Tuple of ranges defining the mask for
  the input volume (for overlapping operators only)
- `trgMsk::NTuple{3, StepRange{Int64, Int64}}`: Tuple of ranges defining the mask for
  the output volume (for overlapping operators only)
"""
struct GlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    mem::GlaVacOprMem{T}
    srcMsk::NTuple{3, StepRange{Int64, Int64}}
    trgMsk::NTuple{3, StepRange{Int64, Int64}}
end

GlaOprVac(mem::GlaVacOprMem{T}, srcMsk, trgMsk) where T<:AbstractFloat = GlaOprVac{T}(mem, srcMsk, trgMsk)

"""
    AsyGlaOprVac{T}

Represents the anti-Hermitian part of the vacuum Green function operator.

# Fields
- `mem::GlaVacOprMem{T}`: Memory structure containing the operator's data, including
  volume information and Fourier coefficients
"""
struct AsyGlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    mem::GlaVacOprMem{T}
end

"""
    SymGlaOprVac{T}

Represents the Hermitian part of the vacuum Green function operator.

# Fields
- `mem::GlaVacOprMem{T}`: Memory structure containing the operator's data, including
  volume information and Fourier coefficients
"""
struct SymGlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    mem::GlaVacOprMem{T}
end

"""
    SusOpr{T, A} ඞ

Represents the susceptibility operator X, the map taking a field to the
polarization current it drives. `T` is the real storage precision, so the
operator acts on `Complex{T}` data.

The two storage shapes are told apart by the type of `sus`. A vector with one
entry per degree of freedom covers a uniform, a per cell isotropic and a per
component diagonal susceptibility, and applies entrywise. A `(celTot, 3, 3)`
array holds the full tensor of every cell and applies it as a matrix in each
cell, which is the shape gyrotropy and rotated crystals need.

χ is dimensionless and diagonal in position, so it commutes with the √ΔV
normalization of `GlaFld` and carries no normalization factor of its own. It is
also the one operator whose inverse is not an iterative solve: `X \\ f` divides
pointwise, and throws where χ vanishes rather than returning infinities.

# Fields
- `sus::A`: The susceptibility, either a vector of `size(opr, 2)` entries in the
  flat degree of freedom layout or a `(celTot, 3, 3)` array of cell tensors, with
  the cells in `coordinates(cvol)` order, on the device the operator computes with
- `cvol::GlaCmpVol`: The tiling the susceptibility lives on, a plain volume being
  a tiling of one region
- `adjMod::Bool`: Whether the operator is the adjoint of the susceptibility it
  was built from
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
    susDof = Vector{Complex{T}}(undef, 3 * length(susCel))
    celOff, dofOff = 0, 0
    for reg in regions(cvol)
        celNum = prod(reg.cel)
        blk = view(susCel, (celOff + 1):(celOff + celNum))
        for dir in 1:3
            copyto!(view(susDof, (dofOff + (dir - 1) * celNum + 1):(dofOff + dir * celNum)), blk)
        end
        celOff += celNum
        dofOff += 3 * celNum
    end
    return susDof
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
    # Anything that is neither a number nor an array is a function of position
    if !(sus isa Number || sus isa AbstractArray)
        susCel = [sus(pos) for (pos, _, _) in coordinates(cvol)]
        if eltype(susCel) <: Number
            susDat = _expSus(cvol, Vector{Complex{T}}(susCel))
        else
            susDat = Array{Complex{T}}(undef, celNum, 3, 3)
            for (cel, mat) in enumerate(susCel)
                size(mat) == (3, 3) || throw(ArgumentError("The susceptibility function returned a value of size $(size(mat)) at cell $cel, and a susceptibility is either a number or a 3×3 tensor."))
                susDat[cel, :, :] .= mat
            end
        end
        return adapt(dev, susDat)
    end
    if ndims(sus) in (3, 5) && nregions(cvol) != 1
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

Construct the susceptibility operator of a volume, plain or composite.

The susceptibility can be given as a number, a 3×3 matrix, an array of one value
per cell, an array of one 3×3 tensor per cell (`(cel..., 3, 3)`), a function of
the cell center returning either a number or a 3×3 matrix, a vector holding any
of those per region, a vector of one value per cell in layout order, or a vector
already in the degree of freedom layout. The shapes that name a cell grid only
fit a tiling of one region; pass one per region otherwise.

# Arguments
- `cvol::GlaCmpVol`: The tiling, a plain `GlaVol` being read as a tiling of one region
- `sus`: The susceptibility, in any of the forms above
- `dev::Backend`: The KernelAbstractions backend (default: that of `sus` when it is an array, else `CPU()`) holding the susceptibility

# Returns
- `SusOpr{T}`: The susceptibility operator, of storage precision `T` (`dflPrc` when unrequested)

# Throws
- `ArgumentError`: If the shape of `sus` does not fit the tiling
"""
SusOpr{T}(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus)) where T<:AbstractFloat =
    SusOpr(_susDat(T, cvol, sus, dev), cvol, false)
SusOpr{T}(vol::GlaVol, sus; dev::Backend=susDev(sus)) where T<:AbstractFloat =
    SusOpr{T}(GlaCmpVol(vol), sus; dev=dev)
SusOpr(vol::Union{GlaVol, GlaCmpVol}, sus; dev::Backend=susDev(sus)) =
    SusOpr{dflPrc}(vol, sus; dev=dev)

"""
    SusOpr{T}(opr::SusOpr)

Convert the storage precision of a susceptibility operator to `T`.
"""
SusOpr{T}(opr::SusOpr{T}) where T<:AbstractFloat = opr
SusOpr{T}(opr::SusOpr) where T<:AbstractFloat =
    SusOpr(Complex{T}.(opr.sus), opr.cvol, opr.adjMod)

"""
    InvSctOpr{T, O}

Represents the inverse scattering operator (I - XG₀), where X is the susceptibility
tensor. This operator describes how electromagnetic fields interact with a material
medium.

# Fields
- `oprVac::O`: The vacuum Green function operator
- `sus::SusOpr{T}`: The susceptibility operator X, over the source tiling of the
  vacuum operator
"""
mutable struct InvSctOpr{T<:AbstractFloat, O<:AbstractGlaVacOpr{T}} <: AbstractGlaOpr{T}
    oprVac::O
    # Left abstract: setSus! rewrites this field with another shape
    sus::SusOpr{T}

    #= The susceptibility comes in any of the forms SusOpr accepts, and is the
    one place a precision conversion is allowed. =#
    function InvSctOpr{T}(oprVac::AbstractGlaVacOpr{T}, sus) where T<:AbstractFloat
        if oprVac isa GlaCmpOprVac && !isselfoperator(oprVac)
            throw(ArgumentError("An inverse scattering operator needs a self operator, and this composite operator maps between two different tilings."))
        end
        return new{T, typeof(oprVac)}(oprVac, sus isa SusOpr ? sus :
            SusOpr{T}(_srcCvl(oprVac), sus; dev=get_backend(oprVac)))
    end
end

# The tiling the currents of a vacuum operator live on
_srcCvl(oprVac::AbstractGlaVacOpr) = GlaCmpVol(oprVac.mem.srcVol)

InvSctOpr(oprVac::AbstractGlaVacOpr{T}, sus) where T<:AbstractFloat = InvSctOpr{T}(oprVac, sus)

"""
    SctOpr{T, I, S}

Represents the scattering operator (I - XG₀)⁻¹, which includes a solver for
computing the action of the inverse scattering operator.

# Fields
- `invSctOpr::I`: The inverse scattering operator
- `slv::S`: The solver to use for solving the linear system
"""
mutable struct SctOpr{T<:AbstractFloat, I<:InvSctOpr{T}, S<:GlaSlv} <: AbstractGlaOpr{T}
    invSctOpr::I
    slv::S
end
SctOpr{T}(invSctOpr::InvSctOpr{T}, slv::GlaSlv) where T<:AbstractFloat = SctOpr(invSctOpr, slv)

"""
    GlaOpr{T, S}

Represents the full Green function operator G₀(I - XG₀)⁻¹, which combines the
vacuum Green function with the scattering operator to describe electromagnetic
interactions in a material medium.

# Fields
- `sctOpr::S`: The scattering operator
"""
mutable struct GlaOpr{T<:AbstractFloat, S<:SctOpr{T}} <: AbstractGlaOpr{T}
    sctOpr::S
end
GlaOpr{T}(sctOpr::SctOpr{T}) where T<:AbstractFloat = GlaOpr(sctOpr)

# Type aliases for convenience
const VacuumGreenOperator = GlaOprVac
const AsymVacuumGreenOperator = AsyGlaOprVac
const SymVacuumGreenOperator = SymGlaOprVac
const InverseScatteringOperator = InvSctOpr
const ScatteringOperator = SctOpr
const GreenOperator = GlaOpr

# Returns true if the volumes share interior. Face, edge, and corner contact is
# not overlap: the external construction has contact corrections for it, and the
# union path cannot represent contact between volumes at different cell scales.
function ovrChk(vol1::GlaVol, vol2::GlaVol)
    # Upper/lower edges of the volumes
    lwrEdg1 = first.(vol1.grd) .- (vol1.scl .// 2)
    uprEdg1 = last.(vol1.grd) .+ (vol1.scl .//2)
    lwrEdg2 = first.(vol2.grd) .- (vol2.scl .// 2)
    uprEdg2 = last.(vol2.grd) .+ (vol2.scl .//2)

    # Volume overlap check
    return all(max.(lwrEdg1, lwrEdg2) .< min.(uprEdg1, uprEdg2)) # < not <= to exclude contact
end

# Returns the region where subVol is contained within vol
function mskRng(subVol::GlaVol, vol::GlaVol)
    stpVol = Rational.(step.(vol.grd))
    stpSub = Rational.(step.(subVol.grd))
    stpRat = stpSub .// stpVol
    @assert all(isinteger.(stpRat)) "Volumes must share a common scale grid for masking"
    stpRat = numerator.(stpRat)

    off = (first.(subVol.grd) .- first.(vol.grd)) .// stpVol # Offset between volumes
    @assert all(isinteger.(off)) "Sub-volume must align with volume grid for masking"
    off = numerator.(off)
    idxBeg = 1 .+ off # 1-based indexing
    idxEnd = idxBeg .+ ((subVol.cel .- 1) .* stpRat)
    return (idxBeg[1]:stpRat[1]:idxEnd[1],
            idxBeg[2]:stpRat[2]:idxEnd[2],
            idxBeg[3]:stpRat[3]:idxEnd[3])
end

"""
    GlaOprVac{T}(trgVol::GlaVol, srcVol::GlaVol; dev::Backend=CPU(), prxWrn::Bool=true, frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)
    GlaOprVac(trgVol::GlaVol, srcVol::GlaVol; dev::Backend=CPU(), prxWrn::Bool=true, frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)

Construct a vacuum Green function operator for external interactions between different volumes.

The storage precision `T` defaults to `dflPrc`; generation is done at `genPrc` and rounded once.

This constructor creates an external Green function operator that describes electromagnetic interactions between distinct regions in free space. The operator maps sources in the source volume to fields in the target volume, enabling the modeling of coupling effects between different parts of an electromagnetic system. For the computation to work correctly, the source and target volumes must share a common scale grid.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `dev::Backend=CPU()`: The KernelAbstractions backend to compute on
- `prxWrn::Bool=true`: Whether to warn when the volumes are closer than a third of a wavelength
- `frqPhz=1.0+0.0im`: Complex frequency phase factor, see `frqPhz(opt::GlaKerOpt)`
- `genPrc=Float64`: Generation precision, see `genPrc(opt::GlaKerOpt)`
- `qssApx::Bool=false`: Quasistatic approximation flag, see `qssApx(opt::GlaKerOpt)`
- `shpCch::Bool=false`: Cache the far-field geometry table of each cell shape, see `GlaVacOprMem`

# Returns
- `GlaOprVac`: The vacuum Green function operator

"""
function GlaOprVac{T}(trgVol::GlaVol, srcVol::GlaVol;
    dev::Backend=CPU(), prxWrn::Bool=true, frqPhz=1.0+0.0im, genPrc=Float64,
    qssApx::Bool=false, shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) where T<:AbstractFloat
    blc === nothing || trgVol == srcVol || haskey(blc, :blk) || throw(ArgumentError("A Bloch lattice is self volume only."))
    innMsk = ntuple(_ -> 0:0, 3)
    outMsk = ntuple(_ -> 0:0, 3)
    if trgVol != srcVol && ovrChk(trgVol, srcVol)
        # Overlap: create the union volume and mask out the input/output regions
        vol = uniVol(trgVol, srcVol)
        innMsk = mskRng(srcVol, vol)
        outMsk = mskRng(trgVol, vol)
        trgVol, srcVol = vol, vol
    end

    # Create the memory structure with appropriate GPU/CPU options
    opt = GlaKerOpt{T}(frqPhz, genPrc, qssApx, false, dev, blc)
    mem = GlaVacOprMem(opt, trgVol, srcVol; shpCch, prxWrn, hrmPrt)
    return GlaOprVac{T}(mem, innMsk, outMsk)
end
GlaOprVac(trgVol::GlaVol, srcVol::GlaVol; dev::Backend=CPU(), prxWrn::Bool=true, frqPhz=1.0+0.0im,
    genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) =
    GlaOprVac{dflPrc}(trgVol, srcVol; dev, prxWrn, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc)

"""
    GlaOprVac(mem::GlaVacOprMem)

Construct a vacuum Green function operator from a memory structure.

# Arguments
- `mem::GlaVacOprMem`: The memory structure containing the operator's data

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
function GlaOprVac(mem::GlaVacOprMem{T}) where T<:AbstractFloat
    innMsk = ntuple(_ -> 0:0, 3)
    outMsk = ntuple(_ -> 0:0, 3)
    trgVol, srcVol = mem.trgVol, mem.srcVol
    if trgVol != srcVol && ovrChk(trgVol, srcVol)
        # Overlap: create the union volume and mask out the input/output regions
        vol = uniVol(trgVol, srcVol)
        innMsk = mskRng(srcVol, vol)
        outMsk = mskRng(trgVol, vol)
        trgVol, srcVol = vol, vol
    end
    return GlaOprVac{T}(mem, innMsk, outMsk)
end

"""
    GlaOprVac{T}(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)
    GlaOprVac(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)

Construct a vacuum Green function operator for self-interactions on a single volume.

This constructor creates a self-interaction Green function operator where the source and target volumes are identical. The operator describes electromagnetic interactions within a single volume in free space, making it suitable for modeling self-coupling effects in electromagnetic systems.

# Arguments
- `vol::GlaVol`: The volume to compute the self Green function for
- `dev::Backend=CPU()`: The KernelAbstractions backend to compute on
- `frqPhz=1.0+0.0im`: Complex frequency phase factor, see `frqPhz(opt::GlaKerOpt)`
- `genPrc=Float64`: Generation precision, see `genPrc(opt::GlaKerOpt)`
- `qssApx::Bool=false`: Quasistatic approximation flag, see `qssApx(opt::GlaKerOpt)`
- `shpCch::Bool=false`: Cache the far-field geometry table of each cell shape, see `GlaVacOprMem`
- `blc=nothing`: Bloch lattice `(ax = (1, 2), P = (32, 32), θ = (0.7, 1//2))`: `vol` is the unit cell, periodic along the axes `ax` with periods `P` cells and Bloch phases `θ` (radians, or exact twists θ/2π as `Rational`s: `0//1`, `1//2`), G(R + Pᵢ eᵢ) = e^{-iθᵢ} G(R); see `setBlc!`

# Returns
- `GlaOprVac`: The vacuum Green function operator

"""
GlaOprVac{T}(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false,
    shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) where T<:AbstractFloat =
    GlaOprVac{T}(vol, vol; dev, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc)
GlaOprVac(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false,
    shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) =
    GlaOprVac{dflPrc}(vol, vol; dev, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc)

"""
    GlaOprVac{T}(opr::GlaOprVac)

Convert the storage precision of a vacuum Green function operator to `T`, without regenerating its Fourier coefficients.
"""
GlaOprVac{T}(opr::GlaOprVac{T}) where T<:AbstractFloat = opr
GlaOprVac{T}(opr::GlaOprVac) where T<:AbstractFloat =
    GlaOprVac{T}(GlaVacOprMem{T}(opr.mem), opr.srcMsk, opr.trgMsk)

"""
    GlaOprVac(opr::InvSctOpr)

Construct a vacuum Green function operator from an inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to convert into a vacuum Green function operator

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
GlaOprVac(opr::InvSctOpr) = opr.oprVac

"""
    GlaOprVac(opr::SctOpr)

Construct a vacuum Green function operator from a scattering operator.

# Arguments
- `opr::SctOpr`: The scattering operator to convert into a vacuum Green function operator

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
GlaOprVac(opr::SctOpr) = GlaOprVac(opr.invSctOpr)

"""
    GlaOprVac(opr::GlaOpr)

Construct a vacuum Green function operator from a full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to convert into a vacuum Green function operator

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
GlaOprVac(opr::GlaOpr) = GlaOprVac(opr.sctOpr)

# A vacuum operator is already the vacuum operator the unwrapping looks for
GlaOprVac(opr::AbstractGlaVacOpr) = opr

"""
    AsyGlaOprVac{T}(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)
    AsyGlaOprVac(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)

Construct the anti-Hermitian part of the vacuum Green function operator for self-interactions on a single volume.

This constructor creates the anti-Hermitian part of the vacuum Green function operator, which describes the radiated components of electromagnetic interactions within a single volume in free space. This operator shows up in the fluctuation-dissipation theorem and is thus related to certain losses in the system.

# Arguments
- `vol::GlaVol`: The volume to compute the anti-Hermitian part of the vacuum Green function for
- `dev::Backend=CPU()`: The KernelAbstractions backend to compute on
- `frqPhz=1.0+0.0im`: Complex frequency phase factor, see `frqPhz(opt::GlaKerOpt)`
- `genPrc=Float64`: Generation precision, see `genPrc(opt::GlaKerOpt)`
- `qssApx::Bool=false`: Quasistatic approximation flag, see `qssApx(opt::GlaKerOpt)`
- `shpCch::Bool=false`: Cache the far-field geometry table of each cell shape, see `GlaVacOprMem`
- `blc=nothing`: Bloch lattice of a periodic unit cell, see `GlaOprVac`

# Returns
- `AsyGlaOprVac`: The anti-Hermitian part of the vacuum Green function operator
"""
function AsyGlaOprVac{T}(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64,
    qssApx::Bool=false, shpCch::Bool=false, blc=nothing) where T<:AbstractFloat
    opt = GlaKerOpt{T}(frqPhz, genPrc, qssApx, false, dev, blc)
    return AsyGlaOprVac{T}(GlaVacOprMem(opt, vol, vol; shpCch, hrmPrt=:asy))
end
AsyGlaOprVac(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false,
    shpCch::Bool=false, blc=nothing) = AsyGlaOprVac{dflPrc}(vol; dev, frqPhz, genPrc, qssApx, shpCch, blc)

"""
    AsyGlaOprVac{T}(opr::AsyGlaOprVac)

Convert the storage precision of the operator to `T`, without regenerating its Fourier coefficients.
"""
AsyGlaOprVac{T}(opr::AsyGlaOprVac{T}) where T<:AbstractFloat = opr
AsyGlaOprVac{T}(opr::AsyGlaOprVac) where T<:AbstractFloat = AsyGlaOprVac{T}(GlaVacOprMem{T}(opr.mem))

"""
    AsyGlaOprVac(opr::GlaOprVac)

Construct the anti-Hermitian part of the vacuum Green function operator from a vacuum Green function operator.

This reuses the coefficients of `opr` rather than integrating again, so it
recovers the imaginary part from a symbol the size of the real one and costs
`eps * ‖real g‖ / ‖imag g‖`. Build from the volume when that matters.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its anti-Hermitian part

# Returns
- `AsyGlaOprVac`: The anti-Hermitian part of the vacuum Green function operator
"""
function AsyGlaOprVac(opr::GlaOprVac{T}) where T<:AbstractFloat
    srcVol, trgVol = opr.mem.srcVol, opr.mem.trgVol
    if srcVol != trgVol
        throw(ArgumentError("AsyGlaOprVac can only be constructed from a GlaOprVac with identical source and target volumes"))
    end
    opr.mem.cmpInf.blc === nothing || throw(ArgumentError("The Hermitian parts of a Bloch operator come from the volume: AsyGlaOprVac(vol; blc)."))
    mem = deepcopy(opr.mem)
    map!(fur -> complex.(imag.(fur)), mem.egoFur)
    return AsyGlaOprVac{T}(mem)
end

"""
    SymGlaOprVac{T}(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)
    SymGlaOprVac(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)

Construct the Hermitian part of the vacuum Green function operator for self-interactions on a single volume.

This constructor creates the Hermitian part of the vacuum Green function operator

# Arguments
- `vol::GlaVol`: The volume to compute the anti-Hermitian part of the vacuum Green function for
- `dev::Backend=CPU()`: The KernelAbstractions backend to compute on
- `frqPhz=1.0+0.0im`: Complex frequency phase factor, see `frqPhz(opt::GlaKerOpt)`
- `genPrc=Float64`: Generation precision, see `genPrc(opt::GlaKerOpt)`
- `qssApx::Bool=false`: Quasistatic approximation flag, see `qssApx(opt::GlaKerOpt)`
- `shpCch::Bool=false`: Cache the far-field geometry table of each cell shape, see `GlaVacOprMem`
- `blc=nothing`: Bloch lattice of a periodic unit cell, see `GlaOprVac`

# Returns
- `SymGlaOprVac`: The Hermitian part of the vacuum Green function operator
"""
function SymGlaOprVac{T}(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64,
    qssApx::Bool=false, shpCch::Bool=false, blc=nothing) where T<:AbstractFloat
    opt = GlaKerOpt{T}(frqPhz, genPrc, qssApx, false, dev, blc)
    return SymGlaOprVac{T}(GlaVacOprMem(opt, vol, vol; shpCch, hrmPrt=:sym))
end
SymGlaOprVac(vol::GlaVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false,
    shpCch::Bool=false, blc=nothing) = SymGlaOprVac{dflPrc}(vol; dev, frqPhz, genPrc, qssApx, shpCch, blc)

"""
    SymGlaOprVac{T}(opr::SymGlaOprVac)

Convert the storage precision of the operator to `T`, without regenerating its Fourier coefficients.
"""
SymGlaOprVac{T}(opr::SymGlaOprVac{T}) where T<:AbstractFloat = opr
SymGlaOprVac{T}(opr::SymGlaOprVac) where T<:AbstractFloat = SymGlaOprVac{T}(GlaVacOprMem{T}(opr.mem))

"""
    SymGlaOprVac(opr::GlaOprVac)

Construct the Hermitian part of the vacuum Green function operator from a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its Hermitian part

# Returns
- `SymGlaOprVac`: The Hermitian part of the vacuum Green function operator
"""
function SymGlaOprVac(opr::GlaOprVac{T}) where T<:AbstractFloat
    srcVol, trgVol = opr.mem.srcVol, opr.mem.trgVol
    if srcVol != trgVol
        throw(ArgumentError("SymGlaOprVac can only be constructed from a GlaOprVac with identical source and target volumes"))
    end
    opr.mem.cmpInf.blc === nothing || throw(ArgumentError("The Hermitian parts of a Bloch operator come from the volume: SymGlaOprVac(vol; blc)."))
    mem = deepcopy(opr.mem)
    # The Hermitian part is the large one, so reusing the symbol costs it nothing
    map!(fur -> complex.(real.(fur)), mem.egoFur)
    return SymGlaOprVac{T}(mem)
end

"""
    InvSctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus))
    InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus))

Construct an inverse scattering operator for external interactions between different volumes.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{<:Number}`: The susceptibility, in any of the forms `SusOpr` takes, converted to `Complex{T}` on construction
- `dev::Backend=get_backend(sus)`: The KernelAbstractions backend to compute on

# Returns
- `InvSctOpr{T}`: The inverse scattering operator, of storage precision `T` (`dflPrc` when unrequested)

This constructor creates an external inverse scattering operator that describes how electromagnetic fields interact with a material medium between distinct regions. The susceptibility lives on the source volume.
This constructor also takes the `frqPhz`, `genPrc`, `qssApx` and `shpCch` keywords of `GlaOprVac`, forwarded blindly.
"""
InvSctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), kwargs...) where T<:AbstractFloat =
    InvSctOpr{T}(GlaOprVac{T}(trgVol, srcVol; dev, kwargs...), sus)
InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), kwargs...) =
    InvSctOpr{dflPrc}(trgVol, srcVol, sus; dev, kwargs...)

"""
    InvSctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus))
    InvSctOpr(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus))

Construct an inverse scattering operator for self-interactions on a single volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `dev::Backend=get_backend(sus)`: The KernelAbstractions backend to compute on

# Returns
- `InvSctOpr{T}`: The inverse scattering operator, of storage precision `T` (`dflPrc` when unrequested)

This constructor creates a self-interaction inverse scattering operator that describes how electromagnetic fields interact with a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

This constructor also takes the `frqPhz`, `genPrc`, `qssApx` and `shpCch` keywords of `GlaOprVac`, forwarded blindly.
"""
InvSctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), kwargs...) where T<:AbstractFloat =
    InvSctOpr{T}(vol, vol, sus; dev, kwargs...)
InvSctOpr(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), kwargs...) =
    InvSctOpr{dflPrc}(vol, vol, sus; dev, kwargs...)

"""
    InvSctOpr{T}(opr::InvSctOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`.
"""
InvSctOpr{T}(opr::InvSctOpr{T}) where T<:AbstractFloat = opr
InvSctOpr{T}(opr::InvSctOpr) where T<:AbstractFloat =
    InvSctOpr{T}(Base.typename(typeof(opr.oprVac)).wrapper{T}(opr.oprVac), SusOpr{T}(opr.sus))

"""

    InvSctOpr(sctOpr::SctOpr)

Construct an inverse scattering operator from a scattering operator.

# Arguments
- `sctOpr::SctOpr`: The scattering operator to convert into an inverse scattering operator

# Returns
- `InvSctOpr`: The inverse scattering operator
"""
InvSctOpr(opr::SctOpr) = opr.invSctOpr

"""
    InvSctOpr(opr::GlaOpr)

Construct an inverse scattering operator from a full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to convert into an inverse scattering operator

# Returns
- `InvSctOpr`: The inverse scattering operator
"""
InvSctOpr(opr::GlaOpr) = InvSctOpr(opr.sctOpr)

"""
    SctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver())
    SctOpr(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver())

Construct a scattering operator for self-interactions on a single volume.

This constructor creates a self-interaction scattering operator that describes how electromagnetic fields interact with a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

Also takes the `frqPhz`, `genPrc`, `qssApx` and `shpCch` keywords of `GlaOprVac`, forwarded blindly.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `dev::Backend=get_backend(sus)`: The KernelAbstractions backend to compute on
- `slv::GlaSlv=GCRODRSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat =
    SctOpr{T}(vol, vol, sus; dev, slv, kwargs...)
SctOpr(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    SctOpr{dflPrc}(vol, vol, sus; dev, slv, kwargs...)

"""
    SctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver())
    SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver())

Construct a scattering operator for external interactions between different volumes.

This constructor creates an external scattering operator that describes how electromagnetic fields interact with a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

Also takes the `frqPhz`, `genPrc`, `qssApx` and `shpCch` keywords of `GlaOprVac`, forwarded blindly.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `dev::Backend=get_backend(sus)`: The KernelAbstractions backend to compute on
- `slv::GlaSlv=GCRODRSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
function SctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat
    invSctOpr = InvSctOpr{T}(trgVol, srcVol, sus; dev, kwargs...)
    return SctOpr{T}(invSctOpr, slv)
end
SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    SctOpr{dflPrc}(trgVol, srcVol, sus; dev, slv, kwargs...)

"""
    SctOpr(opr::GlaOprVac, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver())

Construct a scattering operator from a vacuum Green function operator.

This constructor creates a scattering operator that describes how electromagnetic fields interact with a material medium based on a given vacuum Green function operator. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into a scattering operator
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `slv::GlaSlv=GCRODRSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
function SctOpr(opr::GlaOprVac{T}, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver()) where T<:AbstractFloat
    invSctOpr = InvSctOpr{T}(opr, sus)
    return SctOpr{T}(invSctOpr, slv)
end

"""
    SctOpr{T}(opr::SctOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`. The solver is kept.
"""
SctOpr{T}(opr::SctOpr{T}) where T<:AbstractFloat = opr
SctOpr{T}(opr::SctOpr) where T<:AbstractFloat = SctOpr{T}(InvSctOpr{T}(opr.invSctOpr), opr.slv)

"""
    SctOpr(opr::GlaOpr)

Construct a scattering operator from a full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to convert into a scattering operator

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr(opr::GlaOpr) = opr.sctOpr

"""
    GlaOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver())
    GlaOpr(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver())

Construct a full Green function operator for self-interactions on a single volume.

This constructor creates a self-interaction full Green function operator that combines the vacuum Green function with the scattering operator to describe electromagnetic interactions in a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

Also takes the `frqPhz`, `genPrc`, `qssApx` and `shpCch` keywords of `GlaOprVac`, forwarded blindly.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `dev::Backend=get_backend(sus)`: The KernelAbstractions backend to compute on
- `slv::GlaSlv=GCRODRSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat =
    GlaOpr{T}(vol, vol, sus; dev, slv, kwargs...)
GlaOpr(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    GlaOpr{dflPrc}(vol, vol, sus; dev, slv, kwargs...)

"""
    GlaOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver())
    GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver())

Construct a full Green function operator for external interactions between different volumes.

This constructor creates an external full Green function operator that combines the vacuum Green function with the scattering operator to describe electromagnetic interactions in a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

Also takes the `frqPhz`, `genPrc`, `qssApx` and `shpCch` keywords of `GlaOprVac`, forwarded blindly.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `dev::Backend=get_backend(sus)`: The KernelAbstractions backend to compute on
- `slv::GlaSlv=GCRODRSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green function operator
"""
function GlaOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat
    sctOpr = SctOpr{T}(trgVol, srcVol, sus; dev, slv, kwargs...)
    return GlaOpr{T}(sctOpr)
end
GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    GlaOpr{dflPrc}(trgVol, srcVol, sus; dev, slv, kwargs...)

"""
    GlaOpr(opr::GlaOprVac, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver())

Construct a full Green function operator from a vacuum Green function operator.

This constructor creates a full Green function operator that combines the vacuum Green function with the scattering operator to describe electromagnetic interactions in a material medium based on a given vacuum Green function operator. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into a full Green function operator
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `slv::GlaSlv=GCRODRSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr(opr::GlaOprVac, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver()) = GlaOpr(SctOpr(opr, sus; slv=slv))

"""
    GlaOpr{T}(opr::GlaOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`. The solver is kept.
"""
GlaOpr{T}(opr::GlaOpr{T}) where T<:AbstractFloat = opr
GlaOpr{T}(opr::GlaOpr) where T<:AbstractFloat = GlaOpr{T}(SctOpr{T}(opr.sctOpr))

"""
    GlaOpr(opr::InvSctOpr)

Construct a full Green function operator from an inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to convert into a full Green function operator

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr(opr::InvSctOpr) = GlaOpr(SctOpr(opr))

Adapt.@adapt_structure GlaOprVac
Adapt.@adapt_structure AsyGlaOprVac
Adapt.@adapt_structure SymGlaOprVac
Adapt.@adapt_structure SusOpr
Adapt.@adapt_structure InvSctOpr
Adapt.@adapt_structure SctOpr
Adapt.@adapt_structure GlaOpr

"""
    asym(opr::GlaOprVac)

Construct the anti-Hermitian part of a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its anti-Hermitian part

# Returns
- `AsyGlaOprVac`: The anti-Hermitian part of the vacuum Green function operator
"""
asym(opr::GlaOprVac) = AsyGlaOprVac(opr)

"""
    sym(opr::GlaOprVac)

Construct the -ermitian part of a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its Hermitian part

# Returns
- `SymGlaOprVac`: The Hermitian part of the vacuum Green function operator
"""
sym(opr::GlaOprVac) = SymGlaOprVac(opr)

"""
    isadjoint(opr::AbstractGlaOpr)

Checks if the operator is the adjoint of the Green operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is the adjoint, `false` otherwise.
"""
isadjoint(opr::GlaOprVac) = opr.mem.cmpInf.adjMod
isadjoint(opr::SusOpr) = opr.adjMod
isadjoint(::Union{AsyGlaOprVac, SymGlaOprVac}) = false
isadjoint(opr::InvSctOpr) = isadjoint(opr.oprVac)
isadjoint(opr::SctOpr) = isadjoint(opr.invSctOpr)
isadjoint(opr::GlaOpr) = isadjoint(opr.sctOpr)

"""
    isquasistatic(opr::AbstractGlaOpr)

Checks if the operator was built from the quasistatic Green function rather than the full one.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is quasistatic, `false` otherwise.
"""
isquasistatic(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.cmpInf.qssApx
isquasistatic(::SusOpr) = false
isquasistatic(opr::InvSctOpr) = isquasistatic(opr.oprVac)
isquasistatic(opr::SctOpr) = isquasistatic(opr.invSctOpr)
isquasistatic(opr::GlaOpr) = isquasistatic(opr.sctOpr)

"""
    isselfoperator(opr::AbstractGlaOpr)

Checks if the operator is a self Green operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is a self Green operator, `false` otherwise.
"""
isselfoperator(opr::GlaOprVac) = (opr.mem.srcVol == opr.mem.trgVol) && all(==(0:0), opr.srcMsk) && all(==(0:0), opr.trgMsk)
isselfoperator(::Union{AsyGlaOprVac, SymGlaOprVac}) = true # Both are always self operators
isselfoperator(::SusOpr) = true # A susceptibility never leaves its own cell
isselfoperator(opr::InvSctOpr) = isselfoperator(opr.oprVac)
isselfoperator(opr::SctOpr) = isselfoperator(opr.invSctOpr)
isselfoperator(opr::GlaOpr) = isselfoperator(opr.sctOpr)

"""
    isexternaloperator(opr::GlaOprVac)

Checks if the operator is an external Green operator.

# Arguments
- `opr::GlaOprVac`: The operator to check.

# Returns
- `true` if the operator is an external Green operator, `false` otherwise.
"""
isexternaloperator(opr::GlaOprVac) = opr.mem.srcVol != opr.mem.trgVol && all(==(0:0), opr.srcMsk) && all(==(0:0), opr.trgMsk)
isexternaloperator(::Union{AsyGlaOprVac, SymGlaOprVac}) = false # Both are always self operators
isexternaloperator(::SusOpr) = false
isexternaloperator(opr::InvSctOpr) = isexternaloperator(opr.oprVac)
isexternaloperator(opr::SctOpr) = isexternaloperator(opr.invSctOpr)
isexternaloperator(opr::GlaOpr) = isexternaloperator(opr.sctOpr)

"""
    isoverlappingoperator(opr::GlaOprVac)

Checks if the operator is an overlapping Green operator.

# Arguments
- `opr::GlaOprVac`: The operator to check.

# Returns
- `true` if the operator is an overlapping Green operator, `false` otherwise.
"""
isoverlappingoperator(::AbstractGlaVacOpr) = false # Only the masked routes overlap
isoverlappingoperator(opr::GlaOprVac) = !(isselfoperator(opr) || isexternaloperator(opr))
isoverlappingoperator(::SusOpr) = false
isoverlappingoperator(opr::InvSctOpr) = isoverlappingoperator(opr.oprVac)
isoverlappingoperator(opr::SctOpr) = isoverlappingoperator(opr.invSctOpr)
isoverlappingoperator(opr::GlaOpr) = isoverlappingoperator(opr.sctOpr)

"""
    get_backend(opr::AbstractGlaOpr)

The KernelAbstractions backend the operator computes on.
"""
KernelAbstractions.get_backend(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = get_backend(opr.mem)
KernelAbstractions.get_backend(opr::SusOpr) = get_backend(opr.sus)
KernelAbstractions.get_backend(opr::InvSctOpr) = get_backend(opr.oprVac)
KernelAbstractions.get_backend(opr::SctOpr) = get_backend(opr.invSctOpr)
KernelAbstractions.get_backend(opr::GlaOpr) = get_backend(opr.sctOpr)

"""
    setSus!(opr::InvSctOpr, sus::AbstractArray{<:Number})

Sets the susceptibility tensor for the inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to modify.
- `sus`: The new susceptibility, in any of the forms `SusOpr` takes, converted to the precision of the operator.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::InvSctOpr{T}, sus) where T<:AbstractFloat
    opr.sus = sus isa SusOpr ? sus :
        SusOpr{T}(opr.sus.cvol, sus; dev=get_backend(opr.oprVac))
    return opr
end

"""
    setSus!(opr::SctOpr, sus::AbstractArray{<:Number})

Sets the susceptibility tensor for the scattering operator.

# Arguments
- `opr::SctOpr`: The scattering operator to modify.
- `sus::AbstractArray{<:Number}`: The new susceptibility tensor, either as a flat vector or a 3-tensor, converted to the precision of the operator.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::SctOpr, sus)
    setSus!(opr.invSctOpr, sus)
    return opr
end

"""
    setSus!(opr::GlaOpr, sus::AbstractArray{<:Number})

Sets the susceptibility tensor for the full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to modify.
- `sus::AbstractArray{<:Number}`: The new susceptibility tensor, either as a flat vector or a 3-tensor, converted to the precision of the operator.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::GlaOpr, sus)
    setSus!(opr.sctOpr, sus)
    return opr
end

"""
    setBlc!(opr::AbstractGlaOpr, θ::Tuple)

Set the Bloch phases `θ` of a Bloch self operator, or of the one inside a scattering operator, in place: a phase costs one fill and one transform, no plan or per-frequency cache is rebuilt.
"""
setBlc!(opr::AbstractGlaOpr, θ::Tuple) = (setBlc!(GlaOprVac(opr), θ); opr)
setBlc!(opr::AbstractGlaVacOpr, θ::Tuple) = (setBlc!(opr.mem, θ); opr)

"""
    slv(opr::AbstractGlaOpr)

Returns the solver associated with the operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator for which to get the solver.

# Returns
- The solver used by the operator, which is always a `GlaSlv` instance.
"""
slv(::AbstractGlaVacOpr) = GilaSolvers.GCRODRSolver() # Default solver
slv(opr::InvSctOpr) = slv(opr.oprVac)
slv(opr::SctOpr) = opr.slv
slv(opr::GlaOpr) = opr.sctOpr.slv

"""
    sus(opr::AbstractGlaOpr)

Returns the susceptibility operator of a scattering operator.

The operator returned is the one the scattering operator applies rather than a
copy, so an `adjoint!` on either is seen by both. `setSus!` replaces it, which
leaves an earlier handle on the susceptibility it replaced.

# Arguments
- `opr::AbstractGlaOpr`: The operator for which to get the susceptibility.

# Returns
- `SusOpr`: The susceptibility operator X of `opr`.
"""
sus(opr::InvSctOpr) = opr.sus
sus(opr::SctOpr) = sus(opr.invSctOpr)
sus(opr::GlaOpr) = sus(opr.sctOpr)

_strKnd(opr::GlaOprVac) = "G₀"
_strKnd(opr::AsyGlaOprVac) = "Asym(G₀)"
_strKnd(opr::SymGlaOprVac) = "Sym(G₀)"
_strKnd(opr::SusOpr) = ndims(opr.sus) == 3 ? "anisotropic X" : "X"
_strKnd(opr::InvSctOpr) = "(I - XG₀)"
_strKnd(opr::SctOpr) = "(I - XG₀)⁻¹"
_strKnd(opr::GlaOpr) = "G₀(I - XG₀)⁻¹"

_srcVol(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.srcVol
_srcVol(opr::InvSctOpr) = _srcVol(opr.oprVac)
_srcVol(opr::SctOpr) = _srcVol(opr.invSctOpr)
_srcVol(opr::GlaOpr) = _srcVol(opr.sctOpr)
_trgVol(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.trgVol
_trgVol(opr::InvSctOpr) = _trgVol(opr.oprVac)
_trgVol(opr::SctOpr) = _trgVol(opr.invSctOpr)
_trgVol(opr::GlaOpr) = _trgVol(opr.sctOpr)

# self, external or overlapping, lowercase to match the compact and block forms
_relKnd(opr::AbstractGlaOpr) = isselfoperator(opr) ? "self" :
    isexternaloperator(opr) ? "external" : "overlapping"

# A byte count as MiB or GiB, one decimal place
_szStr(bytes::Integer) = bytes >= 2^30 ? "$(round(bytes / 2^30, digits=1)) GiB" :
    bytes >= 2^20 ? "$(round(bytes / 2^20, digits=1)) MiB" :
    "$(round(bytes / 2^10, digits=1)) KiB"

# Bytes actually stored in every egoFur array of an operator's Fourier data;
# GlaSndOprVac and the composite types add their own methods in glaCmpOpr.jl
_szBytes(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = sum(sizeof, opr.mem.egoFur)

# The volume row: one volume for a self operator, source and target otherwise
_volRow(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = isselfoperator(opr) ?
    _volDesc(_srcVol(opr), "  →  ") : string(join(_srcVol(opr).cel, "×"), " cells of ",
    _sclStr(_srcVol(opr).scl), "  →  ", join(_trgVol(opr).cel, "×"), " cells of ", _sclStr(_trgVol(opr).scl))

# The center separation of an external operator's two volumes, as already computed
_sepRow(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) =
    "(" * join(_ratStr.(_trgVol(opr).org .- _srcVol(opr).org), ", ") * ")λ"

_kerRow(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) =
    "$(isquasistatic(opr) ? "quasistatic" : "full"), frequency $(frqPhz(opr.mem.cmpInf))"

function _dofStr(opr::AbstractGlaOpr)
    m, n = size(opr)
    return m == n ? "$n dof" : "$m × $n dof"
end
_stoRow(opr::AbstractGlaOpr) = "$(_dofStr(opr)), egoFur $(_szStr(_szBytes(opr)))"

#= The rows of the vacuum family's block show. InvSctOpr, SctOpr and GlaOpr reach
this through GlaOprVac(opr), which resolves to the plain or composite method
depending on what they were built over, so no isa check is needed here. =#
function _oprRow(opr::AbstractGlaOpr)
    rows = [("volume", _volRow(opr))]
    isexternaloperator(opr) && push!(rows, ("separation", _sepRow(opr)))
    push!(rows, ("kernel", _kerRow(opr)))
    push!(rows, ("storage", _stoRow(opr)))
    return rows
end

#= The direction-1 component of a per-cell susceptibility vector, one entry per
cell in coordinates(cvol) order. Exact for the uniform and per-cell isotropic
cases; for a per-component diagonal vector it is a representative, not a mean. =#
function _susCel(sus::AbstractVector, cvol::GlaCmpVol)
    off = _dofOff(cvol)
    cel = similar(sus, off[end] ÷ 3)
    celOff = 0
    for (i, reg) in enumerate(regions(cvol))
        n = prod(reg.cel)
        cel[celOff+1:celOff+n] .= view(sus, off[i]+1:off[i]+n)
        celOff += n
    end
    return cel
end

# A one-line summary of a susceptibility: uniform, ranged, or anisotropic
function _susSummary(x::SusOpr)
    ndims(x.sus) == 3 && return "anisotropic, $(size(x.sus, 1)) cells"
    cel = _susCel(Array(x.sus), x.cvol)
    allequal(cel) && return "uniform χ = $(first(cel))"
    lo, hi = extrema(abs, cel)
    return "χ ∈ [$lo, $hi], $(count(!iszero, cel)) of $(length(cel)) cells nonzero"
end

_oprRow(opr::SusOpr) = [("volume", sprint(show, opr.cvol)),
    ("susceptibility", _susSummary(opr)),
    ("storage", "$(_dofStr(opr)), sus $(_szStr(sizeof(opr.sus)))")]

_oprRow(opr::InvSctOpr) = push!(_oprRow(GlaOprVac(opr)), ("susceptibility", _susSummary(opr.sus)))
_oprRow(opr::Union{SctOpr, GlaOpr}) =
    push!(_oprRow(GlaOprVac(opr)), ("solver", string(nameof(typeof(slv(opr))))))

# The compact line's trailing size descriptor
function _dimStr(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac})
    isselfoperator(opr) && return join(_srcVol(opr).cel, "×")
    return join(_srcVol(opr).cel, "×") * " → " * join(_trgVol(opr).cel, "×")
end
_dimStr(opr::Union{InvSctOpr, SctOpr, GlaOpr}) = _dimStr(GlaOprVac(opr))
_dimStr(opr::SusOpr) = "$(size(opr, 2)) dof"

function Base.show(io::IO, opr::AbstractGlaOpr)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, nameof(typeof(opr)), "{", real(eltype(opr)), "} ", _strKnd(opr), " ",
        _relKnd(opr), " ", nameof(typeof(get_backend(opr))), " ", _dimStr(opr))
end

function Base.show(io::IO, ::MIME"text/plain", opr::AbstractGlaOpr)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, nameof(typeof(opr)), "{", real(eltype(opr)), "} — ", _strKnd(opr), ", ",
        _relKnd(opr), ", ", nameof(typeof(get_backend(opr))))
    for (lbl, val) in _oprRow(opr)
        _shwRow(io, lbl, val)
    end
end
include("glaLinAlg.jl")
include("glaCmpOpr.jl")

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

end # module
