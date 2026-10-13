"""
    GlaSndOprVac{T}

One block of a composite operator, computed on the finer of the two meshes it
connects.

The block wraps a same-scale `GlaOprVac` between the target and source regions
represented at a common fine cell size. A coarse side carries an integer cell
ratio: on the target side the fine fields are summed over each `trgRat` block of
cells, and on the source side each coarse coefficient is repeated over its
`srcRat` block of fine cells at weight one, which is what a pulse basis function
of unit density looks like on the fine mesh. A ratio of `(1, 1, 1)` leaves that
side alone.

`wgt` holds every scalar of the block at once: the `1 / prod(trgRat)` that turns
the sum over the target block into a mean, and the `sqrt(ΔV_trg / ΔV_src)` of the
normalized basis. Written this way the block matrix is `wgt` times a product of
two zero-one matrices with the inner operator between them, so the adjoint is the
same shape with the two ratios exchanged and `wgt` conjugated.

# Fields
- `opr::GlaOprVac`: The same-scale operator between the two regions on the fine
  mesh
- `trgRat::NTuple{3,Int}`: Fine cells per target cell in each dimension
- `srcRat::NTuple{3,Int}`: Fine cells per source cell in each dimension
- `wgt::Complex{T}`: The scalar multiplying the block
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

The vacuum Green function operator between two composite volumes.

The operator is the block matrix of pairwise vacuum operators over the regions of
the two tilings. Pulse basis functions of disjoint regions do not overlap, so the
block matrix is the Galerkin discretization of the Green operator on the
non-uniform mesh, with no stitching at region boundaries.

Blocks work in the √ΔV normalized basis of `GlaFld`, so block `(i, j)` carries
a factor `sqrt(ΔV_i / ΔV_j)`. A region has one cell size, so that factor is one
scalar per block. In this basis the Euclidean inner product is the physical L²
pairing and the self operator is complex-symmetric.

# Fields
- `trgCvl::GlaCmpVol`: The composite volume the fields land on
- `srcCvl::GlaCmpVol`: The composite volume the currents live on
- `blkMat::Matrix{AbstractGlaOpr{T}}`: The block matrix. Same-scale pairs and
  separated cross-scale pairs are a `GlaOprVac`, cross-scale pairs in contact
  (all of them under a Bloch lattice) are a `GlaSndOprVac`, and same-scale pairs whose kind of contact the external
  construction does not cover are a `GlaOprVac` on the union of the two regions
"""
struct GlaCmpOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    trgCvl::GlaCmpVol
    srcCvl::GlaCmpVol
    # Left abstract on purpose: a mix of GlaOprVac and GlaSndOprVac, one dispatch per FFT
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

A self composite operator is complex-symmetric in the √ΔV basis, so its
anti-Hermitian part is its entry by entry imaginary part, which is again a block
matrix over the same tiling. Every block carries the Fourier coefficients of the
imaginary part of its own kernel, so a matvec costs one application of `G₀`
rather than the two of the difference. `opr` itself is left untouched.

# Fields
- `opr::GlaCmpOprVac`: The block matrix, every block replaced by its entry by
  entry imaginary part

# Throws
- `ArgumentError`: If the operator maps between two different tilings, or is in
  adjoint mode
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

# Fields
- `opr::GlaCmpOprVac`: The block matrix, every block replaced by its entry by
  entry real part

# Throws
- `ArgumentError`: If the operator maps between two different tilings, or is in
  adjoint mode
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

# Start of each region block in the flat layout, plus the total length at the end
function _dofOff(cvol::GlaCmpVol)
    off = zeros(Int, nregions(cvol) + 1)
    for (idx, reg) in enumerate(regions(cvol))
        off[idx + 1] = off[idx] + 3 * prod(reg.cel)
    end
    return off
end

# Normalization of block (i, j) in the √ΔV basis
_nrmWgt(trgReg::GlaVol, srcReg::GlaVol) =
    sqrt(Float64(prod(trgReg.scl) // prod(srcReg.scl)))

#= Check for contact between two regions to then run the contact quadrature
subroutine (sandwich). =#
_cntChk(trgReg::GlaVol, srcReg::GlaVol) =
    all(max.(_lwrEdg(trgReg), _lwrEdg(srcReg)) .<= min.(_uprEdg(trgReg), _uprEdg(srcReg)))

# The cross-scale contact block, computed on the finer of the two meshes
function _sndBlk(::Type{T}, trgReg::GlaVol, srcReg::GlaVol; dev::Backend, frqPhz, genPrc,
    qssApx::Bool, shpCch::Bool, hrmPrt, blc) where T<:AbstractFloat
    sclFin = min.(trgReg.scl, srcReg.scl)
    trgRat = ntuple(dir -> Int(trgReg.scl[dir] // sclFin[dir]), 3)
    srcRat = ntuple(dir -> Int(srcReg.scl[dir] // sclFin[dir]), 3)
    trgFin = GlaVol(Tuple(trgReg.cel .* trgRat), sclFin, trgReg.org)
    srcFin = GlaVol(Tuple(srcReg.cel .* srcRat), sclFin, srcReg.org)
    # a periodic pair's block is the fine pair's, equal cells
    blc === nothing || (blc = merge(blc, (; blk = _blcCmp([trgFin, srcFin], Complex{genPrc}(frqPhz), blc)[(1, 2)])))
    innOpr = GlaOprVac{T}(trgFin, srcFin; dev, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc, prxWrn=false)
    return GlaSndOprVac{T}(innOpr, trgRat, srcRat,
        _nrmWgt(trgReg, srcReg) / prod(trgRat))
end

function _cmpBlk(::Type{T}, trgReg::GlaVol, srcReg::GlaVol, isSlf::Bool,
    slfCmp::Bool, trgIdx::Integer, srcIdx::Integer; dev::Backend, frqPhz, genPrc,
    qssApx::Bool, shpCch::Bool, hrmPrt, blc) where T<:AbstractFloat
    isSlf && return GlaOprVac{T}(trgReg; dev, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc)
    if !slfCmp && _ovrLap(trgReg, srcReg)
        throw(ArgumentError("Target region $trgIdx spans $(_lwrEdg(trgReg)) to $(_uprEdg(trgReg)) and source region $srcIdx spans $(_lwrEdg(srcReg)) to $(_uprEdg(srcReg)), so the two overlap. A composite operator between two bodies needs the bodies to be disjoint."))
    end
    # periodic images put every cross-scale pair in contact
    trgReg.scl != srcReg.scl && (blc !== nothing || _cntChk(trgReg, srcReg)) &&
        return _sndBlk(T, trgReg, srcReg; dev, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc)
    opr = GlaOprVac{T}(trgReg, srcReg; dev, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc, prxWrn=false)
    nrm = _nrmWgt(trgReg, srcReg)
    # A real scalar on the Fourier coefficients survives adjoint! untouched
    nrm != 1 && map!(fur -> T(nrm) .* fur, opr.mem.egoFur)
    return opr
end

# One warning for the closest pair of regions, none when the two bodies meet
function _prxCmpChk(trgRegs, srcRegs)
    prs = [(GilaVacuum.prxGap(a, b), a, b) for a in trgRegs, b in srcRegs]
    any(pr -> isnothing(pr[1]), prs) && return nothing
    _, trgReg, srcReg = argmin(pr -> pr[1], prs)
    return GilaVacuum.prxChk(1.0, trgReg, srcReg)
end

# blcCmp's blocks of every ordered region pair of a periodic composite of periods L
_blcCmp(regs, frq, blc) = first(GilaVacuum.Bloch.blcCmp(regs, frq, blc.ax, blc.L, blc.θ))

"""
    GlaCmpOprVac{T}(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)
    GlaCmpOprVac(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)

Construct the vacuum Green function operator between two composite volumes.

Each pair of regions gets the block that its geometry allows. Two regions of the
same cell size use the ordinary external path, in contact or not, unless they
touch in a way the contact correction does not cover, in which case the block is
built on the union of the two. Two regions of different cell size use the
partitioned cross-scale path when they are far enough apart, and the fine mesh
construction of `GlaSndOprVac` when they are close enough that the cross-scale
contact quadrature would run. A pair of identical regions, which only happens on
the diagonal of a self operator, uses the self path.

Passing the same composite volume twice gives the self operator of one body.
Passing two different ones gives the operator between two bodies, which have to
be disjoint.

# Arguments
- `trgCvl::GlaCmpVol`: The composite volume the fields land on
- `srcCvl::GlaCmpVol`: The composite volume the currents live on
- `dev::Backend=CPU()`: The KernelAbstractions backend to build the blocks on
- `frqPhz=1.0+0.0im`: Complex frequency phase factor, see `frqPhz(opt::GlaKerOpt)`
- `genPrc=Float64`: Generation precision, see `genPrc(opt::GlaKerOpt)`
- `qssApx::Bool=false`: Quasistatic approximation flag, see `qssApx(opt::GlaKerOpt)`
- `shpCch::Bool=false`: Cache the far-field geometry table of each cell shape, see `GlaVacOprMem`
- `blc=nothing`: Bloch lattice `(ax, P, θ)` of a periodic composite unit cell, as for `GlaOprVac(vol; blc)`, the periods `P` in cells of the finest region; self operators only, every cross-scale pair a `GlaSndOprVac`

# Returns
- `GlaCmpOprVac`: The composite operator

# Throws
- `ArgumentError`: If a region of one volume overlaps a region of the other
"""
function GlaCmpOprVac{T}(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol;
    dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false,
    shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) where T<:AbstractFloat
    slfCmp = trgCvl === srcCvl || trgCvl == srcCvl
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
GlaCmpOprVac(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im,
    genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) =
    GlaCmpOprVac{dflPrc}(trgCvl, srcCvl; dev, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc)

"""
    GlaCmpOprVac{T}(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)
    GlaCmpOprVac(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)

Construct the self vacuum Green function operator of a composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `dev::Backend=CPU()`: The KernelAbstractions backend to build the blocks on
- `frqPhz=1.0+0.0im`: Complex frequency phase factor, see `frqPhz(opt::GlaKerOpt)`
- `genPrc=Float64`: Generation precision, see `genPrc(opt::GlaKerOpt)`
- `qssApx::Bool=false`: Quasistatic approximation flag, see `qssApx(opt::GlaKerOpt)`
- `shpCch::Bool=false`: Cache the far-field geometry table of each cell shape, see `GlaVacOprMem`
- `blc=nothing`: Bloch lattice of a periodic composite unit cell, see `GlaCmpOprVac(trgCvl, srcCvl)`

# Returns
- `GlaCmpOprVac`: The composite operator
"""
GlaCmpOprVac{T}(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64,
    qssApx::Bool=false, shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) where T<:AbstractFloat =
    GlaCmpOprVac{T}(cvol, cvol; dev, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc)
GlaCmpOprVac(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false,
    shpCch::Bool=false, hrmPrt::Union{Nothing,Symbol}=nothing, blc=nothing) =
    GlaCmpOprVac{dflPrc}(cvol, cvol; dev, frqPhz, genPrc, qssApx, shpCch, hrmPrt, blc)

"""
    GlaOprVac{T}(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)
    GlaOprVac(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)

Construct the self vacuum Green function operator of a composite volume.

A composite volume needs a block matrix rather than one circulant, so the result
is a `GlaCmpOprVac` and not a `GlaOprVac`. Both are `AbstractGlaVacOpr`.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `dev::Backend=CPU()`: The KernelAbstractions backend to build the blocks on
- `frqPhz`, `genPrc`, `qssApx`, `shpCch`, `blc`: See `GlaCmpOprVac`

# Returns
- `GlaCmpOprVac`: The composite operator
"""
GlaOprVac{T}(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false,
    shpCch::Bool=false, blc=nothing) where T<:AbstractFloat =
    GlaCmpOprVac{T}(cvol; dev, frqPhz, genPrc, qssApx, shpCch, blc)
GlaOprVac(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false,
    shpCch::Bool=false, blc=nothing) = GlaCmpOprVac{dflPrc}(cvol; dev, frqPhz, genPrc, qssApx, shpCch, blc)

"""
    GlaOprVac{T}(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)
    GlaOprVac(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false, shpCch::Bool=false)

Construct the vacuum Green function operator between two composite volumes.

The result is a `GlaCmpOprVac`, for the reason given in the single volume method.

# Arguments
- `trgCvl::GlaCmpVol`: The composite volume the fields land on
- `srcCvl::GlaCmpVol`: The composite volume the currents live on
- `dev::Backend=CPU()`: The KernelAbstractions backend to build the blocks on
- `frqPhz`, `genPrc`, `qssApx`, `shpCch`: See `GlaCmpOprVac`

# Returns
- `GlaCmpOprVac`: The composite operator
"""
GlaOprVac{T}(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol;
    dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64, qssApx::Bool=false,
    shpCch::Bool=false) where T<:AbstractFloat =
    GlaCmpOprVac{T}(trgCvl, srcCvl; dev, frqPhz, genPrc, qssApx, shpCch)
GlaOprVac(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, genPrc=Float64,
    qssApx::Bool=false, shpCch::Bool=false) =
    GlaCmpOprVac{dflPrc}(trgCvl, srcCvl; dev, frqPhz, genPrc, qssApx, shpCch)

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
    AsyGlaCmpOprVac{T}(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, blc=nothing)
    AsyGlaCmpOprVac(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, blc=nothing)

Construct the anti-Hermitian part of the self vacuum Green function operator of a
composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `dev::Backend=CPU()`: The KernelAbstractions backend to build the blocks on
- `frqPhz`, `blc`: See `GlaCmpOprVac`

# Returns
- `AsyGlaCmpOprVac`: The anti-Hermitian part of the composite operator
"""
# Val(:raw) because every block already holds the part, taken before its transform
AsyGlaCmpOprVac{T}(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, blc=nothing) where T<:AbstractFloat =
    AsyGlaCmpOprVac{T}(GlaCmpOprVac{T}(cvol; dev, frqPhz, blc, hrmPrt=:asy), Val(:raw))
AsyGlaCmpOprVac(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, blc=nothing) =
    AsyGlaCmpOprVac{dflPrc}(cvol; dev, frqPhz, blc)

"""
    SymGlaCmpOprVac{T}(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, blc=nothing)
    SymGlaCmpOprVac(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, blc=nothing)

Construct the Hermitian part of the self vacuum Green function operator of a
composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `dev::Backend=CPU()`: The KernelAbstractions backend to build the blocks on
- `frqPhz`, `blc`: See `GlaCmpOprVac`

# Returns
- `SymGlaCmpOprVac`: The Hermitian part of the composite operator
"""
SymGlaCmpOprVac{T}(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, blc=nothing) where T<:AbstractFloat =
    SymGlaCmpOprVac{T}(GlaCmpOprVac{T}(cvol; dev, frqPhz, blc, hrmPrt=:sym), Val(:raw))
SymGlaCmpOprVac(cvol::GlaCmpVol; dev::Backend=CPU(), frqPhz=1.0+0.0im, blc=nothing) =
    SymGlaCmpOprVac{dflPrc}(cvol; dev, frqPhz, blc)

"""
    asym(opr::GlaCmpOprVac)

Construct the anti-Hermitian part of a self composite vacuum Green function
operator.

# Arguments
- `opr::GlaCmpOprVac`: The composite operator, which has to be a self operator

# Returns
- `AsyGlaCmpOprVac`: The anti-Hermitian part of the composite operator
"""
asym(opr::GlaCmpOprVac) = AsyGlaCmpOprVac(opr)

"""
    sym(opr::GlaCmpOprVac)

Construct the Hermitian part of a self composite vacuum Green function operator.

# Arguments
- `opr::GlaCmpOprVac`: The composite operator, which has to be a self operator

# Returns
- `SymGlaCmpOprVac`: The Hermitian part of the composite operator
"""
sym(opr::GlaCmpOprVac) = SymGlaCmpOprVac(opr)

glaSze(opr::GlaSndOprVac) =
    ((glaSze(opr.opr, 1)[1:3] .÷ opr.trgRat..., 3),
     (glaSze(opr.opr, 2)[1:3] .÷ opr.srcRat..., 3))
glaSze(opr::GlaCmpOprVac) = glaSze.(opr.blkMat)
glaSze(opr::GlaCmpOprVac, dim::Int) = map(sze -> sze[dim], glaSze(opr))
glaSze(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = glaSze(opr.opr)
glaSze(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}, dim::Int) =
    glaSze(opr.opr, dim)

Base.size(opr::GlaCmpOprVac) =
    (sum(3 * prod(reg.cel) for reg in regions(opr.trgCvl)),
     sum(3 * prod(reg.cel) for reg in regions(opr.srcCvl)))
Base.size(opr::GlaCmpOprVac, dim::Int) = size(opr)[dim]
Base.size(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = size(opr.opr)
Base.size(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}, dim::Int) =
    size(opr.opr, dim)

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
    *(opr::GlaCmpOprVac, fld::GlaFld)

Apply a composite operator to a composite field.

# Arguments
- `opr::GlaCmpOprVac`: The operator
- `fld::GlaFld`: The field, which must live on the source volume of `opr`

# Returns
- `GlaFld`: The result, on the target volume of `opr`

# Throws
- `ArgumentError`: If the field lives on a different tiling than the source
  volume of the operator
"""
function Base.:*(opr::GlaCmpOprVac{T}, fld::GlaFld{T}) where T<:AbstractFloat
    if !(fld.cvol === opr.srcCvl || fld.cvol == opr.srcCvl)
        throw(ArgumentError("The field lives on a different composite volume than the source volume of the operator. An operator only applies to fields on the tiling it was built for."))
    end
    return GlaFld(opr * fld.dat, opr.trgCvl)
end

"""
    *(opr::AsyGlaCmpOprVac, fld::GlaFld)

Apply a Hermitian or anti-Hermitian part of a composite operator to a composite
field.

# Arguments
- `opr::AsyGlaCmpOprVac`: The operator, either part
- `fld::GlaFld`: The field, which must live on the volume of `opr`

# Returns
- `GlaFld`: The result, on the same tiling

# Throws
- `ArgumentError`: If the field lives on a different tiling than the operator
"""
Base.:*(opr::Union{AsyGlaCmpOprVac{T}, SymGlaCmpOprVac{T}}, fld::GlaFld{T}) where T<:AbstractFloat = opr.opr * fld

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

Adapt.@adapt_structure GlaSndOprVac
# A matrix of operators is not an array to move, and the raw path keeps the part from being taken twice
Adapt.adapt_structure(to, opr::GlaCmpOprVac) = GlaCmpOprVac(opr.trgCvl, opr.srcCvl, map(blk -> adapt(to, blk), opr.blkMat))
Adapt.adapt_structure(to, opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = Base.typename(typeof(opr)).wrapper(adapt(to, opr.opr), Val(:raw))

isadjoint(opr::GlaSndOprVac) = isadjoint(opr.opr)
isadjoint(opr::GlaCmpOprVac) = all(isadjoint, opr.blkMat)
isadjoint(::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = false
isselfoperator(opr::GlaSndOprVac) = false
isselfoperator(opr::GlaCmpOprVac) =
    opr.trgCvl === opr.srcCvl || opr.trgCvl == opr.srcCvl
isselfoperator(::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = true
isexternaloperator(opr::GlaSndOprVac) = true
isexternaloperator(opr::GlaCmpOprVac) = !isselfoperator(opr)
isexternaloperator(::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = false
KernelAbstractions.get_backend(opr::Union{GlaSndOprVac, AsyGlaCmpOprVac, SymGlaCmpOprVac}) = get_backend(opr.opr)
KernelAbstractions.get_backend(opr::GlaCmpOprVac) = get_backend(first(opr.blkMat))
isquasistatic(opr::GlaSndOprVac) = isquasistatic(opr.opr)
isquasistatic(opr::GlaCmpOprVac) = all(isquasistatic, opr.blkMat)
isquasistatic(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = isquasistatic(opr.opr)

_strKnd(opr::GlaSndOprVac) = "fine mesh G₀"
_strKnd(opr::GlaCmpOprVac) = "composite G₀"
_strKnd(opr::AsyGlaCmpOprVac) = "composite Asym(G₀)"
_strKnd(opr::SymGlaCmpOprVac) = "composite Sym(G₀)"

#= A scattering operator built over a composite volume gets the same "composite"
marker its vacuum operator does, dispatched rather than branched (Phase 3). =#
_strKnd(::InvSctOpr{T, <:GlaCmpOprVac}) where T<:AbstractFloat = "composite (I - XG₀)"
_strKnd(::SctOpr{T, <:InvSctOpr{T, <:GlaCmpOprVac}}) where T<:AbstractFloat = "composite (I - XG₀)⁻¹"
_strKnd(::GlaOpr{T, <:SctOpr{T, <:InvSctOpr{T, <:GlaCmpOprVac}}}) where T<:AbstractFloat = "composite G₀(I - XG₀)⁻¹"

_szBytes(opr::GlaSndOprVac) = _szBytes(opr.opr)
_szBytes(opr::GlaCmpOprVac) = sum(_szBytes, opr.blkMat)
_szBytes(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = _szBytes(opr.opr)

function _dimStr(opr::GlaCmpOprVac)
    numTrg, numSrc = size(opr.blkMat)
    return isselfoperator(opr) ? "$numSrc region$(numSrc == 1 ? "" : "s")" : "$numTrg × $numSrc regions"
end
_dimStr(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) =
    "$(nregions(opr.opr.srcCvl)) region$(nregions(opr.opr.srcCvl) == 1 ? "" : "s")"

# The regions row replaces the bespoke composite show with a short table
function _oprRow(opr::GlaCmpOprVac)
    numTrg, numSrc = size(opr.blkMat)
    numSnd = count(blk -> blk isa GlaSndOprVac, opr.blkMat)
    rows = [("regions", "$numTrg target × $numSrc source, $numSnd fine mesh block$(numSnd == 1 ? "" : "s")"),
        ("targets", sprint(show, opr.trgCvl))]
    isselfoperator(opr) || push!(rows, ("sources", sprint(show, opr.srcCvl)))
    push!(rows, ("storage", _stoRow(opr)))
    return rows
end

function _oprRow(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac})
    numReg = nregions(opr.opr.srcCvl)
    numSnd = count(blk -> blk isa GlaSndOprVac, opr.opr.blkMat)
    return [("regions", "$numReg region$(numReg == 1 ? "" : "s"), $numSnd fine mesh block$(numSnd == 1 ? "" : "s")"),
        ("volume", sprint(show, opr.opr.srcCvl)),
        ("storage", _stoRow(opr))]
end

function Base.show(io::IO, opr::GlaSndOprVac)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, nameof(typeof(get_backend(opr))), " ")
    print(io, "fine mesh G₀ for $(eltype(opr)) (" *
        join(glaSze(opr, 2)[1:3], "×") * ") -> (" *
        join(glaSze(opr, 1)[1:3], "×") * ") volumes ")
    print(io, "on a (" * join(_srcVol(opr.opr).scl, "×") * ")λ³ mesh")
end
Base.show(io::IO, ::MIME"text/plain", opr::GlaSndOprVac) = show(io, opr)

_srcCvl(opr::GlaCmpOprVac) = opr.srcCvl

"""
    InvSctOpr{T}(cvol::GlaCmpVol, sus; dev::Backend)
    InvSctOpr(cvol::GlaCmpVol, sus; dev::Backend)

Construct the inverse scattering operator `(I - XG₀)` over a composite volume.

The vacuum operator is built for the tiling, and the susceptibility is stored in
the flat degree of freedom layout of `GlaFld`. Since the susceptibility and the
√ΔV normalization are both diagonal, they commute, and the operator in the
normalized basis is the same expression as on a uniform mesh.

The susceptibility can be given in any of the forms `SusOpr` takes, from a number
for a uniform medium to one 3×3 tensor per cell for an anisotropic one. Every
other composite scattering constructor takes the same forms.

Also takes the `frqPhz`, `genPrc`, `qssApx` and `shpCch` keywords of `GlaCmpOprVac`, forwarded blindly.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `sus`: The susceptibility, in any of the forms above
- `dev::Backend`: The KernelAbstractions backend (default: that of `sus` when it is an array, else `CPU()`) to build the operator on

# Returns
- `InvSctOpr`: The inverse scattering operator

# Throws
- `ArgumentError`: If the shape of `sus` does not fit the tiling
"""
InvSctOpr{T}(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), kwargs...) where T<:AbstractFloat =
    InvSctOpr{T}(GlaCmpOprVac{T}(cvol; dev, kwargs...), sus)
InvSctOpr(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), kwargs...) =
    InvSctOpr{dflPrc}(cvol, sus; dev, kwargs...)

"""
    SctOpr{T}(cvol::GlaCmpVol, sus; dev::Backend, slv::GlaSlv=GCRODRSolver())
    SctOpr(cvol::GlaCmpVol, sus; dev::Backend, slv::GlaSlv=GCRODRSolver())

Construct the scattering operator `(I - XG₀)⁻¹` over a composite volume.

Also takes the `frqPhz`, `genPrc`, `qssApx` and `shpCch` keywords of `GlaCmpOprVac`, forwarded blindly.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `sus`: The susceptibility, in any of the forms `InvSctOpr(::GlaCmpVol, sus)` takes
- `dev::Backend`: The KernelAbstractions backend (default: that of `sus` when it is an array, else `CPU()`) to build the operator on
- `slv::GlaSlv=GCRODRSolver()`: The solver used for the inverse

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr{T}(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat =
    SctOpr{T}(InvSctOpr{T}(cvol, sus; dev, kwargs...), slv)
SctOpr(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    SctOpr{dflPrc}(cvol, sus; dev, slv, kwargs...)

"""
    SctOpr(opr::GlaCmpOprVac, sus; slv::GlaSlv=GCRODRSolver())

Construct the scattering operator from a composite vacuum operator.

# Arguments
- `opr::GlaCmpOprVac`: The composite vacuum operator, which has to be a self operator
- `sus`: The susceptibility, in any of the forms `InvSctOpr(::GlaCmpVol, sus)` takes
- `slv::GlaSlv=GCRODRSolver()`: The solver used for the inverse

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr(opr::GlaCmpOprVac{T}, sus; slv::GlaSlv=GCRODRSolver()) where T<:AbstractFloat =
    SctOpr{T}(InvSctOpr{T}(opr, sus), slv)

"""
    GlaOpr{T}(cvol::GlaCmpVol, sus; dev::Backend, slv::GlaSlv=GCRODRSolver())
    GlaOpr(cvol::GlaCmpVol, sus; dev::Backend, slv::GlaSlv=GCRODRSolver())

Construct the full Green function operator `G₀(I - XG₀)⁻¹` over a composite volume.

Also takes the `frqPhz`, `genPrc`, `qssApx` and `shpCch` keywords of `GlaCmpOprVac`, forwarded blindly.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `sus`: The susceptibility, in any of the forms `InvSctOpr(::GlaCmpVol, sus)` takes
- `dev::Backend`: The KernelAbstractions backend (default: that of `sus` when it is an array, else `CPU()`) to build the operator on
- `slv::GlaSlv=GCRODRSolver()`: The solver used for the inverse

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr{T}(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat =
    GlaOpr{T}(SctOpr{T}(cvol, sus; dev, slv, kwargs...))
GlaOpr(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    GlaOpr{dflPrc}(cvol, sus; dev, slv, kwargs...)

"""
    GlaOpr(opr::GlaCmpOprVac, sus; slv::GlaSlv=GCRODRSolver())

Construct the full Green function operator from a composite vacuum operator.

# Arguments
- `opr::GlaCmpOprVac`: The composite vacuum operator, which has to be a self operator
- `sus`: The susceptibility, in any of the forms `InvSctOpr(::GlaCmpVol, sus)` takes
- `slv::GlaSlv=GCRODRSolver()`: The solver used for the inverse

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr(opr::GlaCmpOprVac{T}, sus; slv::GlaSlv=GCRODRSolver()) where T<:AbstractFloat =
    GlaOpr{T}(SctOpr(opr, sus; slv=slv))

#= The tiling a scattering operator reads its input on, checked against the
field. A plain vacuum operator carries a single volume, which is a tiling of one
region. =#
function _chkSctFld(oprVac::GlaCmpOprVac, fld::GlaFld)
    fld.cvol === oprVac.srcCvl || fld.cvol == oprVac.srcCvl || throw(ArgumentError("The field lives on a different composite volume than the operator. An operator only applies to fields on the tiling it was built for."))
    return oprVac.srcCvl
end

function _chkSctFld(oprVac::AbstractGlaVacOpr, fld::GlaFld)
    srcVol = oprVac.mem.srcVol
    if nregions(fld.cvol) != 1 || regions(fld.cvol)[1] != srcVol
        throw(ArgumentError("The field does not live on the source volume of the operator, which is a ($(join(srcVol.cel, "×"))) cell volume of ($(join(srcVol.scl, "×")))λ³ cells."))
    end
    return fld.cvol
end

#= A susceptibility carries no normalization factor of its own: it is
dimensionless and diagonal in position, so it commutes with the √ΔV of GlaFld. =#
function _chkSusFld(opr::SusOpr, fld::GlaFld)
    if !_eqvCvl(fld.cvol, opr.cvol)
        throw(ArgumentError("The field lives on a different composite volume than the susceptibility. A susceptibility only applies to fields on the tiling it was built for."))
    end
    return opr.cvol
end
function Base.:*(opr::SusOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat
    cvol = _chkSusFld(opr, fld)
    return GlaFld(opr * fld.dat, cvol)
end
function Base.:\(opr::SusOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat
    cvol = _chkSusFld(opr, fld)
    return GlaFld(opr \ fld.dat, cvol)
end

"""
    *(opr::InvSctOpr, fld::GlaFld)

Apply an inverse scattering operator to a field.

# Arguments
- `opr::InvSctOpr`: The operator, which has to be built on a composite volume
- `fld::GlaFld`: The field, which has to live on the tiling of `opr`

# Returns
- `GlaFld`: The result, on the same tiling

# Throws
- `ArgumentError`: If the field lives on a different tiling
"""

function Base.:*(opr::InvSctOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat
    cvol = _chkSctFld(opr.oprVac, fld)
    return GlaFld(opr * fld.dat, cvol)
end

"""
    *(opr::SctOpr, fld::GlaFld)

Apply a scattering operator to a field, which runs the iterative solve.

# Arguments
- `opr::SctOpr`: The operator, which has to be built on a composite volume
- `fld::GlaFld`: The field, which has to live on the tiling of `opr`

# Returns
- `GlaFld`: The result, on the same tiling

# Throws
- `ArgumentError`: If the field lives on a different tiling
"""
function Base.:*(opr::SctOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat
    cvol = _chkSctFld(opr.invSctOpr.oprVac, fld)
    return GlaFld(solve(opr.invSctOpr, fld.dat, opr.slv), cvol)
end

#= The full operator on a field is the vacuum operator after the solve, the two
the other way around in adjoint mode. Both halves take a field, so the tiling
rides along and the checks happen inside them. =#
Base.:*(opr::GlaOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat = isadjoint(opr) ?
    opr.sctOpr * (opr.sctOpr.invSctOpr.oprVac * fld) :
    opr.sctOpr.invSctOpr.oprVac * (opr.sctOpr * fld)

"""
    \\(opr::AbstractGlaOpr, fld::GlaFld)
    ldiv!(out::GlaFld, opr::AbstractGlaOpr, inp::GlaFld)

Solve `opr * out = fld` on the tiling the field lives on.

As for a vector, the solve is iterative and uses the operator's own solver
(`slv(opr)`). It runs on the flat buffer and rewraps: a solver allocates its
Krylov basis with a matrix shaped `similar`, which a `GlaFld` cannot answer with
a field, so the tiling has to ride along outside the solve.

# Returns
- `GlaFld`: The solution, on the tiling of `fld`

# Throws
- `ArgumentError`: If the field lives on a different tiling than the operator
"""
function Base.:\(opr::AbstractGlaOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat
    cvol = _chkSctFld(GlaOprVac(opr), fld)
    return GlaFld(opr \ fld.dat, cvol)
end
function LinearAlgebra.ldiv!(out::GlaFld{T}, opr::AbstractGlaOpr{T}, inp::GlaFld{T}) where T<:AbstractFloat
    _chkSctFld(GlaOprVac(opr), inp)
    ldiv!(out.dat, opr, inp.dat)
    return out
end

