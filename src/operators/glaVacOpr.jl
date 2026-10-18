"""
    GlaOprVac{T}

The vacuum Green operator G₀, acting on `Complex{T}` data. `srcMsk` and `trgMsk`
pick the two volumes out of their union when they overlap, and are `nothing` otherwise.
"""
struct GlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    mem::GlaVacOprMem{T}
    srcMsk::Union{Nothing, NTuple{3, StepRange{Int64, Int64}}}
    trgMsk::Union{Nothing, NTuple{3, StepRange{Int64, Int64}}}
end

GlaOprVac(mem::GlaVacOprMem{T}, srcMsk, trgMsk) where T<:AbstractFloat = GlaOprVac{T}(mem, srcMsk, trgMsk)

const VacuumGreenOperator = GlaOprVac

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
GlaCmpOprVac{T}(opr::GlaCmpOprVac) where T<:AbstractFloat =
    GlaCmpOprVac{T}(opr.trgCvl, opr.srcCvl, map(blk -> Base.typename(typeof(blk)).wrapper{T}(blk), opr.blkMat))

const CompositeVacuumGreenOperator = GlaCmpOprVac

# Returns the region where subVol is contained within vol
function mskRng(subVol::GlaVol, vol::GlaVol)
    stpVol = Rational.(step.(vol.grd))
    stpSub = Rational.(step.(subVol.grd))
    stpRat = stpSub .// stpVol
    all(isinteger, stpRat) || throw(ArgumentError("Volumes must share a common scale grid for masking"))
    stpRat = numerator.(stpRat)

    off = (first.(subVol.grd) .- first.(vol.grd)) .// stpVol # Offset between volumes
    all(isinteger, off) || throw(ArgumentError("Sub-volume must align with volume grid for masking"))
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
- `qssApx::Bool=false`: use the quasistatic kernel
- `shpCch::Bool=false`: cache the far-field geometry table of each cell shape, see `GlaVacOprMem`.
  The in-memory caches of the far-field and contact fills are never evicted: a process keeps
  20-26 MB per cell shape and one far-field set per shape and frequency of a sweep
- `prxWrn::Bool=true`: warn when two separated volumes are closer than a third of a wavelength
- `hrmPrt=nothing`: `:asy` or `:sym` keeps only the anti-Hermitian or Hermitian part
  of a self operator, taken from the kernel before its transform; see `asym` and `sym`
- `blc=nothing`: Bloch lattice `(ax = (1, 2), P = (32, 32), θ = (0.7, 1//2))`, self
  operator only: `vol` is the unit cell, periodic along the axes `ax` with periods
  `P` cells and Bloch phases `θ` (radians, or exact twists θ/2π as `Rational`s),
  G(R + Pᵢ eᵢ) = e^{-iθᵢ} G(R); see `setBlc!`
"""
function GlaOprVac{T}(trgVol::GlaVol, srcVol::GlaVol;
    dev::Backend=CPU(), prxWrn::Bool=true, frqPhz=1.0+0.0im,
    qssApx::Bool=false, shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing, blk=nothing) where T<:AbstractFloat
    blc === nothing || trgVol == srcVol || blk !== nothing || throw(ArgumentError("A Bloch lattice is self volume only."))
    innMsk = outMsk = nothing
    if trgVol != srcVol && _ovrLap(trgVol, srcVol)
        # Overlap: create the union volume and mask out the input/output regions
        vol = uniVol(trgVol, srcVol)
        innMsk = mskRng(srcVol, vol)
        outMsk = mskRng(trgVol, vol)
        trgVol, srcVol = vol, vol
    end

    # Create the memory structure with appropriate GPU/CPU options
    opt = GlaKerOpt{T}(dev; frqPhz, qssApx, hrmPrt, blc)
    mem = GlaVacOprMem(opt, trgVol, srcVol; shpCch, prxWrn, blk)
    return GlaOprVac{T}(mem, innMsk, outMsk)
end

"""
    GlaOprVac(mem::GlaVacOprMem)

Wrap prepared operator memory, unmasked.
"""
GlaOprVac(mem::GlaVacOprMem) = GlaOprVac(mem, nothing, nothing)

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
    asym(opr::Union{GlaOprVac, GlaCmpOprVac})
    sym(opr::Union{GlaOprVac, GlaCmpOprVac})

The anti-Hermitian part `(G₀ - G₀†) / 2i` of a self vacuum operator, the radiative
part that enters the fluctuation-dissipation theorem, or its Hermitian part
`(G₀ + G₀†) / 2`. The part is rebuilt from the volume and options of `opr` with
`hrmPrt`, which takes it from the kernel before the transform, so it costs a
build rather than reusing the coefficients of `opr`, and leaves `opr` untouched.
"""
asym(opr::Union{GlaOprVac, GlaCmpOprVac}) = _hrmBld(opr, :asy)
sym(opr::Union{GlaOprVac, GlaCmpOprVac}) = _hrmBld(opr, :sym)

function _hrmBld(opr::GlaOprVac{T}, hrmPrt::Symbol) where T<:AbstractFloat
    _hrmChk(opr)
    opt = opr.mem.cmpInf
    return GlaOprVac{T}(opr.mem.srcVol; dev=opt.dev, opt.frqPhz, opt.qssApx, opt.blc, hrmPrt)
end

# The blocks hold the Bloch periods L in λ, and the constructor takes P in cells of the finest region
function _hrmBld(opr::GlaCmpOprVac{T}, hrmPrt::Symbol) where T<:AbstractFloat
    _hrmChk(opr)
    opt = _blkOpt(first(opr.blkMat))
    blc = opt.blc === nothing ? nothing : (; opt.blc.ax, opt.blc.θ,
        P = map((L, ax) -> Int(L // minimum(reg -> reg.scl[ax], regions(opr.srcCvl))), opt.blc.L, opt.blc.ax))
    return GlaCmpOprVac{T}(opr.srcCvl; dev=opt.dev, opt.frqPhz, opt.qssApx, blc, hrmPrt)
end

function _hrmChk(opr::AbstractGlaVacOpr)
    isselfoperator(opr) || throw(ArgumentError("A Hermitian part needs a self operator, and this one is $(_relKnd(opr))."))
    isadjoint(opr) && throw(ArgumentError("A Hermitian part needs an operator that is not in adjoint mode. Call adjoint! on it to restore it first."))
end

# The options a block was generated with; a fine mesh block's are its inner operator's
_blkOpt(blk::GlaOprVac) = blk.mem.cmpInf
_blkOpt(blk::GlaSndOprVac) = _blkOpt(blk.opr)

# Normalization of block (i, j) in the √ΔV basis
_nrmWgt(trgReg::GlaVol, srcReg::GlaVol) =
    sqrt(Float64(prod(trgReg.scl) // prod(srcReg.scl)))

#= Check for contact between two regions to then run the contact quadrature
subroutine (sandwich). =#
_cntChk(trgReg::GlaVol, srcReg::GlaVol) =
    all(max.(_lwrEdg(trgReg), _lwrEdg(srcReg)) .<= min.(_uprEdg(trgReg), _uprEdg(srcReg)))

# The cross-scale contact block, computed on the finer of the two meshes
function _sndBlk(::Type{T}, trgReg::GlaVol, srcReg::GlaVol; frqPhz, blc, blk = nothing, kw...) where T<:AbstractFloat
    sclFin = min.(trgReg.scl, srcReg.scl)
    trgRat = ntuple(dir -> Int(trgReg.scl[dir] // sclFin[dir]), 3)
    srcRat = ntuple(dir -> Int(srcReg.scl[dir] // sclFin[dir]), 3)
    trgFin = GlaVol(Tuple(trgReg.cel .* trgRat), sclFin, trgReg.org)
    srcFin = GlaVol(Tuple(srcReg.cel .* srcRat), sclFin, srcReg.org)
    # a periodic pair's block is the fine pair's, equal cells
    blc === nothing || (blk = _blcCmp([trgFin, srcFin], ComplexF64(frqPhz), blc)[(1, 2)])
    innOpr = GlaOprVac{T}(trgFin, srcFin; frqPhz, blc, blk, kw..., prxWrn=false)
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
_blcCmp(regs, frq, blc) = Bloch.blcCmp(regs, frq, blc.ax, blc.L, blc.θ)

"""
    GlaCmpOprVac{T}(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; kwargs...)
    GlaCmpOprVac{T}(cvol::GlaCmpVol; kwargs...)

The vacuum Green operator between two composite volumes, or of one on itself.
Two different volumes are two bodies and must not overlap.

Same-scale pairs and separated cross-scale pairs use the external path; a
cross-scale pair in contact is built on the finer mesh, see `GlaSndOprVac`.
Keywords as for `GlaOprVac`; `prxWrn` warns once, for the closest pair of regions.
The Bloch periods `P` of `blc` count cells of the finest region.
"""
function GlaCmpOprVac{T}(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol;
    dev::Backend=CPU(), prxWrn::Bool=true, frqPhz=1.0+0.0im, qssApx::Bool=false,
    shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) where T<:AbstractFloat
    slfCmp = trgCvl == srcCvl
    trgRegs, srcRegs = regions(trgCvl), regions(srcCvl)
    prxWrn && !slfCmp && _prxCmpChk(trgRegs, srcRegs)
    if blc !== nothing
        slfCmp || throw(ArgumentError("A Bloch lattice is self volume only."))
        blc = (; blc.ax, L = ntuple(j -> blc.P[j] * minimum(reg -> reg.scl[blc.ax[j]], trgRegs), length(blc.ax)), blc.θ)
        blcBlk = _blcCmp(trgRegs, ComplexF64(frqPhz), blc)
    end
    blkMat = Matrix{AbstractGlaOpr{T}}(undef, length(trgRegs), length(srcRegs))
    for trgIdx in eachindex(trgRegs), srcIdx in eachindex(srcRegs)
        blk = blc === nothing ? nothing : blcBlk[(trgIdx, srcIdx)]
        blkMat[trgIdx, srcIdx] = _cmpBlk(T, trgRegs[trgIdx], srcRegs[srcIdx],
            slfCmp && trgIdx == srcIdx, slfCmp, trgIdx, srcIdx; dev, frqPhz, qssApx, shpCch, hrmPrt, blc, blk)
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

Adapt.@adapt_structure GlaOprVac

Adapt.@adapt_structure GlaSndOprVac
# A matrix of operators is not an array to move
Adapt.adapt_structure(to, opr::GlaCmpOprVac) = GlaCmpOprVac(opr.trgCvl, opr.srcCvl, map(blk -> adapt(to, blk), opr.blkMat))

"""
    isadjoint(opr::AbstractGlaOpr)

Whether `opr` is in adjoint mode, see `adjoint!`.
"""
isadjoint(opr::GlaOprVac) = opr.mem.cmpInf.adjMod

"""
    isquasistatic(opr::AbstractGlaOpr)

Whether `opr` was built from the quasistatic Green function rather than the full one.
"""
isquasistatic(opr::GlaOprVac) = opr.mem.cmpInf.qssApx

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
_relKnd(opr::GlaOprVac) = !isnothing(opr.srcMsk) ? :overlapping :
    opr.mem.srcVol == opr.mem.trgVol ? :self : :external
_relKnd(::GlaSndOprVac) = :external
_relKnd(opr::GlaCmpOprVac) = opr.trgCvl == opr.srcCvl ? :self : :external

"""
    get_backend(opr::AbstractGlaOpr)

The KernelAbstractions backend the operator computes on.
"""
KernelAbstractions.get_backend(opr::GlaOprVac) = get_backend(opr.mem)

isadjoint(opr::GlaSndOprVac) = isadjoint(opr.opr)
isadjoint(opr::GlaCmpOprVac) = all(isadjoint, opr.blkMat)
KernelAbstractions.get_backend(opr::GlaSndOprVac) = get_backend(opr.opr)
KernelAbstractions.get_backend(opr::GlaCmpOprVac) = get_backend(first(opr.blkMat))
isquasistatic(opr::GlaSndOprVac) = isquasistatic(opr.opr)
isquasistatic(opr::GlaCmpOprVac) = all(isquasistatic, opr.blkMat)

"""
    setBlc!(opr::AbstractGlaOpr, θ::Tuple)

Set the Bloch phases `θ` of a Bloch self operator, or of the one inside a scattering operator, in place. On a self operator a phase costs one fill and one transform and reuses the plans and the per-frequency `BlcSet`; a composite operator also rebuilds its per-frequency κ cache on every phase, and each fine-mesh block pays one more fill.
"""
setBlc!(opr::AbstractGlaOpr, θ::Tuple) = (setBlc!(vac(opr), θ); opr)
setBlc!(opr::AbstractGlaVacOpr, θ::Tuple) = (setBlc!(opr.mem, θ); opr)

# One blcCmp call for every block, one per sandwich for its fine pair; a block in adjoint mode holds the adjoint of its transposed pair
function setBlc!(opr::GlaCmpOprVac, θ::Tuple)
    mem = first(opr.blkMat).mem
    frq, blc = mem.cmpInf.frqPhz, merge(mem.cmpInf.blc, (; θ))
    blcBlk = _blcCmp(regions(opr.trgCvl), frq, blc)
    for idx in CartesianIndices(opr.blkMat)
        blk = opr.blkMat[idx]
        if blk isa GlaSndOprVac
            # adjoint! swaps the volumes of a sandwich's fine pair
            finMem = blk.opr.mem
            finBlc = _blcCmp([finMem.trgVol, finMem.srcVol], frq, blc)
            setBlc!(finMem, θ, finBlc[isadjoint(blk) ? (2, 1) : (1, 2)])
        else
            setBlc!(blk.mem, θ, blcBlk[isadjoint(blk) ? (idx[2], idx[1]) : Tuple(idx)])
        end
    end
    return opr
end

"""
    slv(opr::AbstractGlaOpr)

The solver `opr` uses for `\\` and for applying (I - XG₀)⁻¹; a default
`GCRODRSolver()` for vacuum operators.
"""
slv(::AbstractGlaVacOpr) = GCRODRSolver() # Default solver

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

glaSze(opr::GlaSndOprVac) =
    ((glaSze(opr.opr, 1)[1:3] .÷ opr.trgRat..., 3),
     (glaSze(opr.opr, 2)[1:3] .÷ opr.srcRat..., 3))
glaSze(opr::GlaCmpOprVac) = glaSze.(opr.blkMat)
glaSze(opr::GlaCmpOprVac, dim::Int) = map(sze -> sze[dim], glaSze(opr))

Base.size(opr::GlaCmpOprVac) = (_dofOff(opr.trgCvl)[end], _dofOff(opr.srcCvl)[end])
Base.size(opr::GlaCmpOprVac, dim::Int) = size(opr)[dim]

function mulAct!(opr::GlaOprVac{T}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
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
_srcCvl(opr::GlaOprVac) = GlaCmpVol(opr.mem.srcVol)
_trgCvl(opr::GlaOprVac) = GlaCmpVol(opr.mem.trgVol)
_srcCvl(opr::GlaCmpOprVac) = opr.srcCvl
_trgCvl(opr::GlaCmpOprVac) = opr.trgCvl

"""
    adjoint!(opr::AbstractGlaOpr)

The adjoint of `opr`, reusing its memory. Use only the return value afterwards:
the argument may have been mutated or replaced. `adjoint!(adjoint!(opr))` restores
the original; `adjoint` works on a copy instead.
"""
function adjoint!(opr::GlaOprVac)
    # A Hermitian part of a self operator is its own adjoint
    isselfoperator(opr) && opr.mem.cmpInf.hrmPrt !== nothing && return opr

    # Mark the adjoint
    opr.mem.cmpInf.adjMod = !opr.mem.cmpInf.adjMod

    # Swap source and target volumes (transpose)
    opr.mem.trgVol, opr.mem.srcVol = opr.mem.srcVol, opr.mem.trgVol
    opr.mem.mixInf = GlaExtInf(opr.mem.trgVol, opr.mem.srcVol)
    opr.mem.fftPlnFwd, opr.mem.adjFftPlnFwd = opr.mem.adjFftPlnFwd, opr.mem.fftPlnFwd
    opr.mem.fftPlnRev, opr.mem.adjFftPlnRev = opr.mem.adjFftPlnRev, opr.mem.fftPlnRev

    # Take the conjugate of the Fourier coefficients (conjugate transpose)
    # Also swap the last two axes because they hold the source and target
    # partition of a cross-scale pair (which must be transposed for the adjoint)
    opr.mem.egoFur = collect(map(arr -> conj.(permutedims(arr, (1, 2, 3, 4, 6, 5))), opr.mem.egoFur))

    # The masks are immutable fields, so the transpose needs a new wrapper
    return GlaOprVac(opr.mem, opr.trgMsk, opr.srcMsk)
end

adjoint!(opr::GlaSndOprVac) =
    GlaSndOprVac(adjoint!(opr.opr), opr.srcRat, opr.trgRat, conj(opr.wgt))

adjoint!(opr::GlaCmpOprVac{T}) where T<:AbstractFloat =
    GlaCmpOprVac{T}(opr.srcCvl, opr.trgCvl, permutedims(map(adjoint!, opr.blkMat)))
