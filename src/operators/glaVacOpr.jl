"""
    GlaOprVac{T}

The vacuum Green operator G₀, acting on `Complex{T}` data. `srcMsk` and `trgMsk`
pick the two volumes out of their union when they overlap, and are `0:0` otherwise.
"""
struct GlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    mem::GlaVacOprMem{T}
    srcMsk::NTuple{3, StepRange{Int64, Int64}}
    trgMsk::NTuple{3, StepRange{Int64, Int64}}
end

GlaOprVac(mem::GlaVacOprMem{T}, srcMsk, trgMsk) where T<:AbstractFloat = GlaOprVac{T}(mem, srcMsk, trgMsk)

"""
    AsyGlaOprVac{T}

The anti-Hermitian part `(G₀ - G₀†) / 2i` of a self vacuum operator.
"""
struct AsyGlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    mem::GlaVacOprMem{T}
end

"""
    SymGlaOprVac{T}

The Hermitian part `(G₀ + G₀†) / 2` of a self vacuum operator.
"""
struct SymGlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    mem::GlaVacOprMem{T}
end

const VacuumGreenOperator = GlaOprVac
const AsymVacuumGreenOperator = AsyGlaOprVac
const SymVacuumGreenOperator = SymGlaOprVac

"""
    GlaSndOprVac{T}

A cross-scale block of a composite operator, computed as a same-scale `GlaOprVac`
on the finer of the two meshes. Each coarse source value is repeated over its
`srcRat` block of fine cells (a unit-density pulse), and the fine field is summed
over each `trgRat` block of target cells.

`wgt` is `sqrt(ΔV_trg / ΔV_src) / prod(trgRat)`, the basis normalization times
the mean over a target block. The block is `wgt` times two zero-one matrices
around the inner operator, so its adjoint swaps the ratios and conjugates `wgt`.
"""
struct GlaSndOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    opr::GlaOprVac{T}
    trgRat::NTuple{3,Int}
    srcRat::NTuple{3,Int}
    wgt::Complex{T}
end

GlaSndOprVac(opr::GlaOprVac{T}, trgRat::NTuple{3,Int}, srcRat::NTuple{3,Int}, wgt::Number) where T<:AbstractFloat =
    GlaSndOprVac{T}(opr, trgRat, srcRat, wgt)

"""
    GlaSndOprVac{T}(opr::GlaSndOprVac)

Convert the storage precision of the block to `T`, without regenerating its Fourier coefficients.
"""
GlaSndOprVac{T}(opr::GlaSndOprVac{T}) where T<:AbstractFloat = opr
GlaSndOprVac{T}(opr::GlaSndOprVac) where T<:AbstractFloat =
    GlaSndOprVac{T}(GlaOprVac{T}(opr.opr), opr.trgRat, opr.srcRat, opr.wgt)

"""
    GlaCmpOprVac{T}

The vacuum Green operator between two composite volumes, as the block matrix of
pairwise vacuum operators over their regions: the Galerkin discretization on the
non-uniform mesh, with no stitching at region boundaries.

Blocks work in the √ΔV normalized basis of `GlaFld`, so block `(i, j)` carries the
scalar `sqrt(ΔV_i / ΔV_j)`. In this basis the Euclidean inner product is the
physical L² pairing and the self operator is complex-symmetric. A raw vector
holds √ΔV times the density `c`: `sqrt.(cellvolumes(cvol)) .* c`, or
fill a `GlaFld` with `discretize!`.

Cross-scale pairs in contact, and every cross-scale pair under a Bloch lattice,
are a `GlaSndOprVac`; every other block is a `GlaOprVac`.
"""
struct GlaCmpOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    trgCvl::GlaCmpVol
    srcCvl::GlaCmpVol
    # GlaOprVac and GlaSndOprVac blocks mixed
    blkMat::Matrix{AbstractGlaOpr{T}}
end

GlaCmpOprVac(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol, blkMat::AbstractMatrix{<:AbstractGlaOpr{T}}) where T<:AbstractFloat =
    GlaCmpOprVac{T}(trgCvl, srcCvl, blkMat)

"""
    GlaCmpOprVac{T}(opr::GlaCmpOprVac)

Convert the storage precision of every block to `T`, without regenerating any Fourier coefficients.
"""
GlaCmpOprVac{T}(opr::GlaCmpOprVac{T}) where T<:AbstractFloat = opr
function GlaCmpOprVac{T}(opr::GlaCmpOprVac) where T<:AbstractFloat
    blkMat = Matrix{AbstractGlaOpr{T}}(undef, size(opr.blkMat))
    for idx in eachindex(opr.blkMat)
        blk = opr.blkMat[idx]
        blkMat[idx] = Base.typename(typeof(blk)).wrapper{T}(blk)
    end
    return GlaCmpOprVac{T}(opr.trgCvl, opr.srcCvl, blkMat)
end

const CompositeVacuumGreenOperator = GlaCmpOprVac

"""
    AsyGlaCmpOprVac(opr::GlaCmpOprVac)

The anti-Hermitian part `Asym(G₀) = (G₀ - G₀†) / 2i` of the vacuum Green function
operator over a composite volume, built from the self composite operator `opr`.

The self operator is complex-symmetric in the √ΔV basis, so this is its entry
by entry imaginary part: a block matrix of the same shape, applied at the cost of
one `G₀`. `opr` must be a self operator not in adjoint mode, and is left untouched.
"""
struct AsyGlaCmpOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    opr::GlaCmpOprVac{T}

    AsyGlaCmpOprVac{T}(opr::GlaCmpOprVac{T}) where T<:AbstractFloat = new{T}(_hrmOpr(opr, true))
    #= Blocks that already carry the transformed coefficients, as read back by
    the deserializer. Taking the part again would apply it twice. =#
    AsyGlaCmpOprVac{T}(opr::GlaCmpOprVac{T}, ::Val{:raw}) where T<:AbstractFloat = new{T}(opr)
end

AsyGlaCmpOprVac(opr::GlaCmpOprVac{T}) where T<:AbstractFloat = AsyGlaCmpOprVac{T}(opr)
AsyGlaCmpOprVac(opr::GlaCmpOprVac{T}, raw::Val{:raw}) where T<:AbstractFloat = AsyGlaCmpOprVac{T}(opr, raw)

"""
    AsyGlaCmpOprVac{T}(opr::AsyGlaCmpOprVac)

Convert the storage precision of every block to `T`, without regenerating any Fourier coefficients.
"""
AsyGlaCmpOprVac{T}(opr::AsyGlaCmpOprVac{T}) where T<:AbstractFloat = opr
AsyGlaCmpOprVac{T}(opr::AsyGlaCmpOprVac) where T<:AbstractFloat =
    AsyGlaCmpOprVac{T}(GlaCmpOprVac{T}(opr.opr), Val(:raw))

"""
    SymGlaCmpOprVac(opr::GlaCmpOprVac)

The Hermitian part `Sym(G₀) = (G₀ + G₀†) / 2` of the vacuum Green function
operator over a composite volume, built the same way as `AsyGlaCmpOprVac` from
the entry by entry real part of the blocks.
"""
struct SymGlaCmpOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    opr::GlaCmpOprVac{T}

    SymGlaCmpOprVac{T}(opr::GlaCmpOprVac{T}) where T<:AbstractFloat = new{T}(_hrmOpr(opr, false))
    # Raw path, see AsyGlaCmpOprVac
    SymGlaCmpOprVac{T}(opr::GlaCmpOprVac{T}, ::Val{:raw}) where T<:AbstractFloat = new{T}(opr)
end

SymGlaCmpOprVac(opr::GlaCmpOprVac{T}) where T<:AbstractFloat = SymGlaCmpOprVac{T}(opr)
SymGlaCmpOprVac(opr::GlaCmpOprVac{T}, raw::Val{:raw}) where T<:AbstractFloat = SymGlaCmpOprVac{T}(opr, raw)

"""
    SymGlaCmpOprVac{T}(opr::SymGlaCmpOprVac)

Convert the storage precision of every block to `T`, without regenerating any Fourier coefficients.
"""
SymGlaCmpOprVac{T}(opr::SymGlaCmpOprVac{T}) where T<:AbstractFloat = opr
SymGlaCmpOprVac{T}(opr::SymGlaCmpOprVac) where T<:AbstractFloat =
    SymGlaCmpOprVac{T}(GlaCmpOprVac{T}(opr.opr), Val(:raw))

const AsymCompositeVacuumGreenOperator = AsyGlaCmpOprVac
const SymCompositeVacuumGreenOperator = SymGlaCmpOprVac

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
    GlaOprVac{T}(trgVol::GlaVol, srcVol::GlaVol; kwargs...)
    GlaOprVac{T}(vol::GlaVol; kwargs...)

The vacuum Green operator from `srcVol` to `trgVol`, or of `vol` on itself, stored
in `Complex{T}` (`T` defaults to `dflPrc`). Volumes that share interior are built
on their union and masked, so they must sit on a common grid.

# Keywords
- `dev::Backend=CPU()`: backend to compute on
- `frqPhz=1.0+0.0im`: complex frequency phase factor
- `genPrc=Float64`: precision the coefficients are generated in, then rounded once to `T`
- `qssApx::Bool=false`: use the quasistatic kernel
- `shpCch::Bool=false`: cache the far-field geometry table of each cell shape, see `GlaVacOprMem`
- `prxWrn::Bool=true`: warn when two separated volumes are closer than a third of a wavelength
- `blc=nothing`: Bloch lattice `(ax = (1, 2), P = (32, 32), θ = (0.7, 1//2))`, self
  operator only: `vol` is the unit cell, periodic along the axes `ax` with periods
  `P` cells and Bloch phases `θ` (radians, or exact twists θ/2π as `Rational`s),
  G(R + Pᵢ eᵢ) = e^{-iθᵢ} G(R); see `setBlc!`
"""
function GlaOprVac{T}(trgVol::GlaVol, srcVol::GlaVol;
    dev::Backend=CPU(), prxWrn::Bool=true, frqPhz=1.0+0.0im, genPrc=Float64,
    qssApx::Bool=false, shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) where T<:AbstractFloat
    blc === nothing || trgVol == srcVol || haskey(blc, :blk) || throw(ArgumentError("A Bloch lattice is self volume only."))
    innMsk = ntuple(_ -> 0:0, 3)
    outMsk = ntuple(_ -> 0:0, 3)
    if trgVol != srcVol && _ovrLap(trgVol, srcVol)
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

"""
    GlaOprVac(mem::GlaVacOprMem)

Wrap prepared operator memory, unmasked.
"""
GlaOprVac(mem::GlaVacOprMem) = GlaOprVac(mem, ntuple(_ -> 0:0, 3), ntuple(_ -> 0:0, 3))

GlaOprVac{T}(vol::GlaVol; kw...) where T<:AbstractFloat = GlaOprVac{T}(vol, vol; kw...)

# A constructor called without a precision stores in dflPrc
GlaOprVac(args...; kw...) = GlaOprVac{dflPrc}(args...; kw...)

"""
    GlaOprVac{T}(opr::GlaOprVac)

Convert the storage precision of a vacuum Green function operator to `T`, without regenerating its Fourier coefficients.
"""
GlaOprVac{T}(opr::GlaOprVac{T}) where T<:AbstractFloat = opr
GlaOprVac{T}(opr::GlaOprVac) where T<:AbstractFloat =
    GlaOprVac{T}(GlaVacOprMem{T}(opr.mem), opr.srcMsk, opr.trgMsk)

"""
    AsyGlaOprVac{T}(vol::GlaVol; kwargs...)

The anti-Hermitian part of the self vacuum operator of `vol`, the radiative part
that enters the fluctuation-dissipation theorem. Keywords as for `GlaOprVac`.
"""
AsyGlaOprVac{T}(vol::GlaVol; kw...) where T<:AbstractFloat =
    AsyGlaOprVac{T}(GlaOprVac{T}(vol; kw..., hrmPrt=:asy).mem)
AsyGlaOprVac(args...; kw...) = AsyGlaOprVac{dflPrc}(args...; kw...)

"""
    AsyGlaOprVac{T}(opr::AsyGlaOprVac)

Convert the storage precision of the operator to `T`, without regenerating its Fourier coefficients.
"""
AsyGlaOprVac{T}(opr::AsyGlaOprVac{T}) where T<:AbstractFloat = opr
AsyGlaOprVac{T}(opr::AsyGlaOprVac) where T<:AbstractFloat = AsyGlaOprVac{T}(GlaVacOprMem{T}(opr.mem))

"""
    AsyGlaOprVac(opr::GlaOprVac)

The anti-Hermitian part of a self operator, from its coefficients. Recovering the
imaginary part from a symbol the size of the real one costs
`eps * ‖real g‖ / ‖imag g‖` in accuracy; build from the volume when that matters.
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
    SymGlaOprVac{T}(vol::GlaVol; kwargs...)

The Hermitian part of the self vacuum operator of `vol`. Keywords as for `GlaOprVac`.
"""
SymGlaOprVac{T}(vol::GlaVol; kw...) where T<:AbstractFloat =
    SymGlaOprVac{T}(GlaOprVac{T}(vol; kw..., hrmPrt=:sym).mem)
SymGlaOprVac(args...; kw...) = SymGlaOprVac{dflPrc}(args...; kw...)

"""
    SymGlaOprVac{T}(opr::SymGlaOprVac)

Convert the storage precision of the operator to `T`, without regenerating its Fourier coefficients.
"""
SymGlaOprVac{T}(opr::SymGlaOprVac{T}) where T<:AbstractFloat = opr
SymGlaOprVac{T}(opr::SymGlaOprVac) where T<:AbstractFloat = SymGlaOprVac{T}(GlaVacOprMem{T}(opr.mem))

"""
    SymGlaOprVac(opr::GlaOprVac)

The Hermitian part of a self operator, from its coefficients.
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
    asym(opr::GlaOprVac)

The anti-Hermitian part of a self vacuum operator, see `AsyGlaOprVac`.
"""
asym(opr::GlaOprVac) = AsyGlaOprVac(opr)

"""
    sym(opr::GlaOprVac)

The Hermitian part of a self vacuum operator, see `SymGlaOprVac`.
"""
sym(opr::GlaOprVac) = SymGlaOprVac(opr)

# Normalization of block (i, j) in the √ΔV basis
_nrmWgt(trgReg::GlaVol, srcReg::GlaVol) =
    sqrt(Float64(prod(trgReg.scl) // prod(srcReg.scl)))

#= Check for contact between two regions to then run the contact quadrature
subroutine (sandwich). =#
_cntChk(trgReg::GlaVol, srcReg::GlaVol) =
    all(max.(_lwrEdg(trgReg), _lwrEdg(srcReg)) .<= min.(_uprEdg(trgReg), _uprEdg(srcReg)))

# The cross-scale contact block, computed on the finer of the two meshes
function _sndBlk(::Type{T}, trgReg::GlaVol, srcReg::GlaVol; frqPhz, genPrc, blc, kw...) where T<:AbstractFloat
    sclFin = min.(trgReg.scl, srcReg.scl)
    trgRat = ntuple(dir -> Int(trgReg.scl[dir] // sclFin[dir]), 3)
    srcRat = ntuple(dir -> Int(srcReg.scl[dir] // sclFin[dir]), 3)
    trgFin = GlaVol(Tuple(trgReg.cel .* trgRat), sclFin, trgReg.org)
    srcFin = GlaVol(Tuple(srcReg.cel .* srcRat), sclFin, srcReg.org)
    # a periodic pair's block is the fine pair's, equal cells
    blc === nothing || (blc = merge(blc, (; blk = _blcCmp([trgFin, srcFin], Complex{genPrc}(frqPhz), blc)[(1, 2)])))
    innOpr = GlaOprVac{T}(trgFin, srcFin; frqPhz, genPrc, blc, kw..., prxWrn=false)
    return GlaSndOprVac{T}(innOpr, trgRat, srcRat,
        _nrmWgt(trgReg, srcReg) / prod(trgRat))
end

function _cmpBlk(::Type{T}, trgReg::GlaVol, srcReg::GlaVol, isSlf::Bool,
    slfCmp::Bool, trgIdx::Integer, srcIdx::Integer; blc, kw...) where T<:AbstractFloat
    isSlf && return GlaOprVac{T}(trgReg; blc, kw...)
    if !slfCmp && _ovrLap(trgReg, srcReg)
        throw(ArgumentError("Target region $trgIdx spans $(_lwrEdg(trgReg)) to $(_uprEdg(trgReg)) and source region $srcIdx spans $(_lwrEdg(srcReg)) to $(_uprEdg(srcReg)), so the two overlap. A composite operator between two bodies needs the bodies to be disjoint."))
    end
    # periodic images put every cross-scale pair in contact
    trgReg.scl != srcReg.scl && (blc !== nothing || _cntChk(trgReg, srcReg)) &&
        return _sndBlk(T, trgReg, srcReg; blc, kw...)
    opr = GlaOprVac{T}(trgReg, srcReg; blc, kw..., prxWrn=false)
    nrm = _nrmWgt(trgReg, srcReg)
    # A real scalar on the Fourier coefficients survives adjoint! untouched
    nrm != 1 && map!(fur -> T(nrm) .* fur, opr.mem.egoFur)
    return opr
end

# One warning for the closest pair of regions, none when the two bodies meet
function _prxCmpChk(trgRegs, srcRegs)
    prs = [(prxGap(a, b), a, b) for a in trgRegs, b in srcRegs]
    any(pr -> isnothing(pr[1]), prs) && return nothing
    _, trgReg, srcReg = argmin(pr -> pr[1], prs)
    return prxChk(1.0, trgReg, srcReg)
end

# blcCmp's blocks of every ordered region pair of a periodic composite of periods L
_blcCmp(regs, frq, blc) = first(GilaVacuum.Bloch.blcCmp(regs, frq, blc.ax, blc.L, blc.θ))

"""
    GlaCmpOprVac{T}(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; kwargs...)
    GlaCmpOprVac{T}(cvol::GlaCmpVol; kwargs...)

The vacuum Green operator between two composite volumes, or of one on itself.
Two different volumes are two bodies and must not overlap.

Same-scale pairs and separated cross-scale pairs use the external path; a
cross-scale pair in contact is built on the finer mesh, see `GlaSndOprVac`.
Keywords as for `GlaOprVac`, without `prxWrn`; the Bloch periods `P` of `blc`
count cells of the finest region.
"""
function GlaCmpOprVac{T}(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol;
    dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false,
    shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) where T<:AbstractFloat
    slfCmp = trgCvl == srcCvl
    trgRegs, srcRegs = regions(trgCvl), regions(srcCvl)
    slfCmp || _prxCmpChk(trgRegs, srcRegs)
    if blc !== nothing
        slfCmp || throw(ArgumentError("A Bloch lattice is self volume only."))
        blc = (; blc.ax, L = ntuple(j -> blc.P[j] * minimum(reg -> reg.scl[blc.ax[j]], trgRegs), length(blc.ax)), blc.θ)
        blcBlk = _blcCmp(trgRegs, Complex{genPrc}(frqPhz), blc)
    end
    blkMat = Matrix{AbstractGlaOpr{T}}(undef, length(trgRegs), length(srcRegs))
    for trgIdx in eachindex(trgRegs), srcIdx in eachindex(srcRegs)
        blkBlc = blc === nothing ? nothing : merge(blc, (; blk = blcBlk[(trgIdx, srcIdx)]))
        blkMat[trgIdx, srcIdx] = _cmpBlk(T, trgRegs[trgIdx], srcRegs[srcIdx],
            slfCmp && trgIdx == srcIdx, slfCmp, trgIdx, srcIdx; dev, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc = blkBlc)
    end
    return GlaCmpOprVac{T}(trgCvl, srcCvl, blkMat)
end
GlaCmpOprVac{T}(cvol::GlaCmpVol; kw...) where T<:AbstractFloat = GlaCmpOprVac{T}(cvol, cvol; kw...)
GlaCmpOprVac(args...; kw...) = GlaCmpOprVac{dflPrc}(args...; kw...)

"""
    GlaOprVac{T}(cvol::GlaCmpVol; kwargs...)
    GlaOprVac{T}(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; kwargs...)

Same as `GlaCmpOprVac`: a composite volume needs a block matrix rather than one
circulant, so the result is a `GlaCmpOprVac`.
"""
GlaOprVac{T}(cvol::GlaCmpVol; kw...) where T<:AbstractFloat = GlaCmpOprVac{T}(cvol; kw...)
GlaOprVac{T}(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; kw...) where T<:AbstractFloat =
    GlaCmpOprVac{T}(trgCvl, srcCvl; kw...)

#= Sign picked up by each stored tensor component when the real space kernel is
reflected in the directions flagged in dirRfl. Storage order is xx, yy, zz, xy,
xz, yz, and a reflection flips a component once for each of its two tensor
indices that lies along it. =#
_rflSgn(dirRfl::NTuple{3,Bool}, cmp::Integer) =
    prod(dir -> dirRfl[dir] && count(==(dir), egoCmpPos[cmp]) == 1 ? -1 : 1, 1:3)

#= Fourier coefficients of the entry by entry real or imaginary part of a block.
A block is the leading corner of a circulant, so taking the real or imaginary
part entry by entry is the circulant of the real or imaginary part of its kernel,
whose coefficients are (fur[m] ± conj(fur[-m])) / 2 (with an extra 1/i for the
imaginary part). Negating the circulant index reverses a direction held in full,
and is the reflection sign of the component in a direction stored only up to that
reflection. A branch is even or odd in each direction, which shifts the even
reversal by one, and never mixes with the other branches. The subtraction is
the one a `hrmPrt` build avoids, and loses the same digits of the imaginary
part as `AsyGlaOprVac(opr::GlaOprVac)`. =#
function _hrmFur!(mem::GlaVacOprMem{T}, isAsy::Bool) where T<:AbstractFloat
    brnSze = div.(mem.mixInf.trgCel .+ mem.mixInf.srcCel, 2)
    for bId in 0:7
        fur = mem.egoFur[bId + 1]
        dirRfl = ntuple(dir -> size(fur, dir) != brnSze[dir], 3)
        negFur = conj.(fur)
        for dir in 1:3
            dirRfl[dir] && continue
            len = size(negFur, dir)
            # the reversal, shifted by one for an even branch
            idx = isodd(bId >> (3 - dir)) ? (len:-1:1) : [1; len:-1:2]
            negFur = negFur[ntuple(d -> d == dir ? idx : Colon(), ndims(negFur))...]
        end
        for cmp in 1:6
            sgn = _rflSgn(dirRfl, cmp)
            sgn == 1 || (view(negFur, :, :, :, cmp, :, :) .*= sgn)
        end
        # the scalar stays in T so a Complex{Float32} branch does not promote
        scl = isAsy ? Complex{T}(0, -0.5) : Complex{T}(0.5)
        fur .= isAsy ? (fur .- negFur) .* scl : (fur .+ negFur) .* scl
    end
    return mem
end

function _hrmBlk(opr::GlaOprVac, isAsy::Bool)
    mem = _hrmFur!(deepcopy(opr.mem), isAsy)
    return GlaOprVac(mem, opr.srcMsk, opr.trgMsk)
end

#= wgt is real and the aggregation and injection matrices hold zeros and ones, so
they all commute with taking the real or imaginary part of the block. =#
_hrmBlk(opr::GlaSndOprVac, isAsy::Bool) =
    GlaSndOprVac(_hrmBlk(opr.opr, isAsy), opr.trgRat, opr.srcRat, opr.wgt)

function _hrmOpr(opr::GlaCmpOprVac{T}, isAsy::Bool) where T<:AbstractFloat
    nam = isAsy ? "AsyGlaCmpOprVac" : "SymGlaCmpOprVac"
    if !isselfoperator(opr)
        throw(ArgumentError("$nam can only be constructed from a GlaCmpOprVac with identical target and source tilings, and this operator maps between two different ones."))
    end
    if isadjoint(opr)
        throw(ArgumentError("$nam can only be constructed from an operator that is not in adjoint mode. Call adjoint! on it to restore it first."))
    end
    any(blk -> blk isa GlaOprVac && blk.mem.cmpInf.blc !== nothing, opr.blkMat) &&
        throw(ArgumentError("The Hermitian parts of a Bloch operator come from the volume: $nam(cvol; blc)."))
    blkMat = Matrix{AbstractGlaOpr{T}}(undef, size(opr.blkMat))
    for idx in eachindex(opr.blkMat)
        blkMat[idx] = _hrmBlk(opr.blkMat[idx], isAsy)
    end
    return GlaCmpOprVac{T}(opr.trgCvl, opr.srcCvl, blkMat)
end

"""
    AsyGlaCmpOprVac{T}(cvol::GlaCmpVol; kwargs...)

The anti-Hermitian part of the self vacuum operator of `cvol`, built from the
volume rather than from a full operator. Keywords as for `GlaCmpOprVac`.
"""
# Val(:raw) because every block already holds the part, taken before its transform
AsyGlaCmpOprVac{T}(cvol::GlaCmpVol; kw...) where T<:AbstractFloat =
    AsyGlaCmpOprVac{T}(GlaCmpOprVac{T}(cvol; kw..., hrmPrt=:asy), Val(:raw))
AsyGlaCmpOprVac(args...; kw...) = AsyGlaCmpOprVac{dflPrc}(args...; kw...)

"""
    SymGlaCmpOprVac{T}(cvol::GlaCmpVol; kwargs...)

The Hermitian part of the self vacuum operator of `cvol`, built from the volume.
Keywords as for `GlaCmpOprVac`.
"""
SymGlaCmpOprVac{T}(cvol::GlaCmpVol; kw...) where T<:AbstractFloat =
    SymGlaCmpOprVac{T}(GlaCmpOprVac{T}(cvol; kw..., hrmPrt=:sym), Val(:raw))
SymGlaCmpOprVac(args...; kw...) = SymGlaCmpOprVac{dflPrc}(args...; kw...)

"""
    asym(opr::GlaCmpOprVac)

The anti-Hermitian part of a self composite operator, see `AsyGlaCmpOprVac`.
"""
asym(opr::GlaCmpOprVac) = AsyGlaCmpOprVac(opr)

"""
    sym(opr::GlaCmpOprVac)

The Hermitian part of a self composite operator, see `SymGlaCmpOprVac`.
"""
sym(opr::GlaCmpOprVac) = SymGlaCmpOprVac(opr)

Adapt.@adapt_structure GlaOprVac
Adapt.@adapt_structure AsyGlaOprVac
Adapt.@adapt_structure SymGlaOprVac

Adapt.@adapt_structure GlaSndOprVac
# A matrix of operators is not an array to move, and the raw path keeps the part from being taken twice
Adapt.adapt_structure(to, opr::GlaCmpOprVac) = GlaCmpOprVac(opr.trgCvl, opr.srcCvl, map(blk -> adapt(to, blk), opr.blkMat))
Adapt.adapt_structure(to, opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = Base.typename(typeof(opr)).wrapper(adapt(to, opr.opr), Val(:raw))

"""
    isadjoint(opr::AbstractGlaOpr)

Whether `opr` is in adjoint mode, see `adjoint!`.
"""
isadjoint(opr::GlaOprVac) = opr.mem.cmpInf.adjMod

isadjoint(::Union{AsyGlaOprVac, SymGlaOprVac}) = false

"""
    isquasistatic(opr::AbstractGlaOpr)

Whether `opr` was built from the quasistatic Green function rather than the full one.
"""
isquasistatic(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.cmpInf.qssApx

"""
    isselfoperator(opr::AbstractGlaOpr)

Whether `opr` maps a volume to itself.
"""
isselfoperator(opr::AbstractGlaOpr) = _relKnd(opr) === :self

"""
    isexternaloperator(opr::AbstractGlaOpr)

Whether `opr` maps between two volumes that share no interior.
"""
isexternaloperator(opr::AbstractGlaOpr) = _relKnd(opr) === :external

"""
    isoverlappingoperator(opr::AbstractGlaOpr)

Whether `opr` maps between two distinct volumes that share interior, built on their union and masked.
"""
isoverlappingoperator(opr::AbstractGlaOpr) = _relKnd(opr) === :overlapping

# How the target and source of an operator relate: :self, :external or :overlapping
_relKnd(opr::GlaOprVac) = !(all(==(0:0), opr.srcMsk) && all(==(0:0), opr.trgMsk)) ? :overlapping :
    opr.mem.srcVol == opr.mem.trgVol ? :self : :external
_relKnd(::Union{AsyGlaOprVac, SymGlaOprVac, AsyGlaCmpOprVac, SymGlaCmpOprVac}) = :self
_relKnd(::GlaSndOprVac) = :external
_relKnd(opr::GlaCmpOprVac) = opr.trgCvl == opr.srcCvl ? :self : :external

"""
    get_backend(opr::AbstractGlaOpr)

The KernelAbstractions backend the operator computes on.
"""
KernelAbstractions.get_backend(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = get_backend(opr.mem)

isadjoint(opr::GlaSndOprVac) = isadjoint(opr.opr)
isadjoint(opr::GlaCmpOprVac) = all(isadjoint, opr.blkMat)
isadjoint(::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = false
KernelAbstractions.get_backend(opr::Union{GlaSndOprVac, AsyGlaCmpOprVac, SymGlaCmpOprVac}) = get_backend(opr.opr)
KernelAbstractions.get_backend(opr::GlaCmpOprVac) = get_backend(first(opr.blkMat))
isquasistatic(opr::GlaSndOprVac) = isquasistatic(opr.opr)
isquasistatic(opr::GlaCmpOprVac) = all(isquasistatic, opr.blkMat)
isquasistatic(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = isquasistatic(opr.opr)

"""
    setBlc!(opr::AbstractGlaOpr, θ::Tuple)

Set the Bloch phases `θ` of a Bloch self operator, or of the one inside a scattering operator, in place: a phase costs one fill and one transform, no plan or per-frequency cache is rebuilt.
"""
setBlc!(opr::AbstractGlaOpr, θ::Tuple) = (setBlc!(GlaOprVac(opr), θ); opr)
setBlc!(opr::AbstractGlaVacOpr, θ::Tuple) = (setBlc!(opr.mem, θ); opr)

# One blcCmp call for every block, one per sandwich for its fine pair; a block in adjoint mode holds the adjoint of its transposed pair
function setBlc!(opr::GlaCmpOprVac, θ::Tuple)
    mem = first(opr.blkMat).mem
    frq, blc = Complex{genPrc(mem.cmpInf)}(frqPhz(mem.cmpInf)), merge(mem.cmpInf.blc, (; θ))
    blcBlk = _blcCmp(regions(opr.trgCvl), frq, blc)
    for idx in CartesianIndices(opr.blkMat)
        blk = opr.blkMat[idx]
        # adjoint! swaps the volumes of a sandwich's fine pair
        blk isa GlaSndOprVac ? setBlc!(blk.opr.mem, θ, _blcCmp([blk.opr.mem.trgVol, blk.opr.mem.srcVol], frq, blc)[isadjoint(blk) ? (2, 1) : (1, 2)]) :
            setBlc!(blk.mem, θ, blcBlk[isadjoint(blk) ? (idx[2], idx[1]) : Tuple(idx)])
    end
    return opr
end
setBlc!(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}, θ::Tuple) = (setBlc!(opr.opr, θ); opr)

"""
    slv(opr::AbstractGlaOpr)

The solver `opr` uses for `\\` and for applying (I - XG₀)⁻¹; a default
`GCRODRSolver()` for vacuum operators.
"""
slv(::AbstractGlaVacOpr) = GilaSolvers.GCRODRSolver() # Default solver

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

glaSze(opr::GlaSndOprVac) =
    ((glaSze(opr.opr, 1)[1:3] .÷ opr.trgRat..., 3),
     (glaSze(opr.opr, 2)[1:3] .÷ opr.srcRat..., 3))
glaSze(opr::GlaCmpOprVac) = glaSze.(opr.blkMat)
glaSze(opr::GlaCmpOprVac, dim::Int) = map(sze -> sze[dim], glaSze(opr))
glaSze(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = glaSze(opr.opr)
glaSze(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}, dim::Int) =
    glaSze(opr.opr, dim)

Base.size(opr::GlaCmpOprVac) = (_dofOff(opr.trgCvl)[end], _dofOff(opr.srcCvl)[end])
Base.size(opr::GlaCmpOprVac, dim::Int) = size(opr)[dim]
Base.size(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = size(opr.opr)
Base.size(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}, dim::Int) =
    size(opr.opr, dim)

function mulAct!(opr::Union{GlaOprVac{T}, AsyGlaOprVac{T}, SymGlaOprVac{T}}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
    if isoverlappingoperator(opr)
        # The masked input is read into the union volume
        actEmb = fill!(similar(act, opr.mem.srcVol.cel..., 3), zero(eltype(act)))
        actEmb[opr.srcMsk..., :] .= reshape(act, glaSze(opr, 2))
        return vec(egoOpr!(opr.mem, actEmb)[opr.trgMsk..., :])
    end
    return vec(egoOpr!(opr.mem, reshape(act, glaSze(opr, 2))))
end

function mulAct!(opr::GlaSndOprVac{T}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
    finAct = act
    # Repeat every coarse source value over the fine cells it covers
    if !all(opr.srcRat .== 1)
        rat, cel = opr.srcRat, glaSze(opr, 2)[1:3]
        finInn = similar(act, rat[1], cel[1], rat[2], cel[2], rat[3], cel[3], 3)
        finInn .= reshape(act, 1, cel[1], 1, cel[2], 1, cel[3], 3)
        finAct = vec(finInn)
    end
    out = reshape(mulAct!(opr.opr, finAct), glaSze(opr.opr, 1))
    # Sum every block of fine cells into the coarse target cell holding it
    if !all(opr.trgRat .== 1)
        rat = opr.trgRat
        cel = size(out)[1:3] .÷ rat
        blk = reshape(out, rat[1], cel[1], rat[2], cel[2], rat[3], cel[3], 3)
        out = reshape(sum(blk; dims=(1, 3, 5)), cel..., 3)
    end
    return rmul!(vec(out), opr.wgt)
end

#= Block row sums over the flat layout. A region block of the buffer is already
the input a block operator wants, so a slice needs no permutation. =#
function mulAct!(opr::GlaCmpOprVac{T}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
    trgOff, srcOff = _dofOff(opr.trgCvl), _dofOff(opr.srcCvl)
    if length(act) != srcOff[end]
        throw(ArgumentError("An input of length $(length(act)) does not fit the source volume of this operator, which has $(srcOff[end]) degrees of freedom."))
    end
    outDat = fill!(similar(act, trgOff[end]), zero(eltype(act)))
    for srcIdx in axes(opr.blkMat, 2), trgIdx in axes(opr.blkMat, 1)
        # A block eats its buffer, so every block gets its own copy of the slice
        innBlk = copy(view(act, (srcOff[srcIdx] + 1):srcOff[srcIdx + 1]))
        view(outDat, (trgOff[trgIdx] + 1):trgOff[trgIdx + 1]) .+=
            mulAct!(opr.blkMat[trgIdx, srcIdx], innBlk)
    end
    return outDat
end

mulAct!(opr::Union{AsyGlaCmpOprVac{T}, SymGlaCmpOprVac{T}},
    act::AbstractVector{Complex{T}}) where T<:AbstractFloat = mulAct!(opr.opr, act)

"""
    *(opr::GlaOprVac, fld::GlaFld)

Apply a vacuum operator to a field whose tiling is the single region `srcVol`,
giving a field on `trgVol`. The √ΔV normalization contributes the scalar
`sqrt(ΔV_trg / ΔV_src)`. Overlapping (masked) operators take arrays, not fields.
"""
function Base.:*(opr::GlaOprVac{T}, fld::GlaFld{T}) where T<:AbstractFloat
    if isoverlappingoperator(opr)
        throw(ArgumentError("This operator is built on the union of its two volumes and reads its input through a mask, so it does not take a field. Apply it to a plain array of the masked size instead."))
    end
    _chkFld(opr, fld)
    outDat = opr * fld.dat
    nrm = _nrmWgt(opr.mem.trgVol, opr.mem.srcVol)
    nrm != 1 && rmul!(outDat, nrm)
    return GlaFld(outDat, _trgCvl(opr))
end

# The tilings a vacuum operator reads from and writes to; a plain volume is a tiling of one region
_srcCvl(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = GlaCmpVol(opr.mem.srcVol)
_trgCvl(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = GlaCmpVol(opr.mem.trgVol)
_srcCvl(opr::GlaCmpOprVac) = opr.srcCvl
_trgCvl(opr::GlaCmpOprVac) = opr.trgCvl
_srcCvl(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = _srcCvl(opr.opr)
_trgCvl(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = _trgCvl(opr.opr)

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

adjoint!(opr::GlaSndOprVac) =
    GlaSndOprVac(adjoint!(opr.opr), opr.srcRat, opr.trgRat, conj(opr.wgt))

function adjoint!(opr::GlaCmpOprVac{T}) where T<:AbstractFloat
    adjMat = Matrix{AbstractGlaOpr{T}}(undef, reverse(size(opr.blkMat)))
    for trgIdx in axes(opr.blkMat, 1), srcIdx in axes(opr.blkMat, 2)
        adjMat[srcIdx, trgIdx] = adjoint!(opr.blkMat[trgIdx, srcIdx])
    end
    return GlaCmpOprVac{T}(opr.srcCvl, opr.trgCvl, adjMat)
end
# Both parts are Hermitian (self-adjoint)
adjoint!(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = opr
